"""Research Agent Graph Module.

This module defines the LangGraph-based research agent that performs
multi-step web research with reflection and citation generation.
"""

import logging
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent.context import Context
from agent.prompts import (
    answer_instructions,
    get_current_date,
    query_writer_instructions,
    reflection_instructions,
    web_searcher_instructions,
)
from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.tools_and_schemas import Reflection, SearchQueryList
from agent.utils import (
    get_research_topic,
)

load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

# Enable LangSmith tracing (optional)
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Context for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Context.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries
    llm = init_chat_model(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        model_provider=os.getenv("MODEL_PROVIDER", "openai"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=1.0,
        max_retries=2,
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


from langchain_community.tools.tavily_search import TavilySearchResults


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using Tavily Search API + LLM synthesis.

    Executes a web search using Tavily, then uses an LLM to synthesize the results
    into a well-structured, cited summary following the web_searcher_instructions.

    Args:
        state: Current graph state containing the search query
        config: Context for the runnable

    Returns:
        Dictionary with state update, including sources and research results
    """
    # Initialize Tavily search
    tavily_search = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
    )

    # Execute search
    search_query = state["search_query"]

    try:
        logger.info(f"🔍 Starting Tavily search for query: {search_query}")
        search_results = tavily_search.invoke({"query": search_query})
        logger.info("✅ Tavily search completed")

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Tavily search exception: {error_msg}")

        # Detect specific error types
        if "432" in error_msg or "Client Error" in error_msg:
            logger.error("🚫 Tavily API Error 432: API quota exhausted or API Key invalid")
            logger.error("💡 Solutions:")
            logger.error("   1. Check TAVILY_API_KEY in .env file")
            logger.error("   2. Visit https://tavily.com to check quota")
            logger.error("   3. Apply for a new API Key if needed")
        elif "401" in error_msg or "Unauthorized" in error_msg:
            logger.error("🚫 Tavily API Error 401: API Key invalid or unauthorized")
        elif "429" in error_msg or "Too Many Requests" in error_msg:
            logger.error("🚫 Tavily API Error 429: Too many requests, rate limit triggered")
        elif "timeout" in error_msg.lower():
            logger.error("🚫 Tavily API Timeout: Request timeout")
        else:
            logger.error(f"🚫 Tavily API Unknown error: {error_msg}")

        search_results = []

    # Debug: print search results type and content
    logger.info(f"📊 Search results type: {type(search_results)}")

    # Check if error object is returned (error in string form)
    if isinstance(search_results, str):
        logger.error(f"❌ Tavily returned error string: {search_results}")
        if "HTTPError" in search_results or "432" in search_results:
            logger.error("🚫 Tavily API quota error (HTTP 432)")
            logger.error("💡 Please check your Tavily API quota and key validity")
        search_results = []

    # Ensure search_results is a list
    if not isinstance(search_results, list):
        logger.warning(f"⚠️ Unexpected search_results type: {type(search_results)}")
        logger.warning("⚠️ Converting to empty list")
        search_results = []

    logger.info(f"📊 Search results count: {len(search_results)}")

    # Process results
    sources_gathered = []
    raw_results_for_llm = []  # Raw results to pass to LLM

    if len(search_results) == 0:
        logger.error(f"❌ No search results returned for query: '{search_query}'")
        logger.warning("⚠️ Returning placeholder to avoid workflow interruption")
        # Return a placeholder to avoid complete failure
        return {
            "sources_gathered": [],
            "search_query": [state["search_query"]],
            "web_research_result": [
                f"⚠️ Unable to retrieve search results for '{search_query}' (possibly due to API quota limit)."
            ],
        }

    # Step 1: Collect source information and raw content
    for idx, result in enumerate(search_results):
        citation_id = f"[{state['id']}-{idx}]"

        # Check if result is a dictionary
        if not isinstance(result, dict):
            logger.warning(f"⚠️ Skipping non-dict result at index {idx}: {type(result)}")
            continue

        logger.info(
            f"📄 Processing result {idx}: url={result.get('url', 'N/A')}, title={result.get('title', 'N/A')}"
        )

        # Collect source information
        url = result.get("url", "")
        title = result.get("title", "Unknown Title")
        content = result.get("content", "")

        if not url:
            logger.warning(f"⚠️ Result {idx} has no URL, skipping")
            continue

        sources_gathered.append({"url": url, "title": title, "citation_id": citation_id})

        # Prepare structured data for LLM
        if content:
            raw_results_for_llm.append(
                {
                    "citation_id": citation_id,
                    "title": title,
                    "url": url,
                    "content": content[:2000],  # Limit length per result to avoid token overflow
                }
            )
            logger.info(f"✅ Prepared content for LLM synthesis with citation_id: {citation_id}")
        else:
            logger.warning(f"⚠️ Result {idx} has no content")

    logger.info(f"📊 Collected {len(sources_gathered)} sources for query: {search_query}")

    if len(sources_gathered) == 0:
        logger.error(f"❌ No sources collected! search_results was: {search_results}")
        return {
            "sources_gathered": [],
            "search_query": [state["search_query"]],
            "web_research_result": ["⚠️ Unable to collect valid search sources."],
        }

    # Step 2: Use LLM to synthesize high-quality summary (with citations)
    logger.info("🤖 Using LLM to synthesize search results into structured summary...")

    # Build search results text for LLM
    search_results_text = ""
    for item in raw_results_for_llm:
        search_results_text += f"\n\n--- Source {item['citation_id']} ---\n"
        search_results_text += f"Title: {item['title']}\n"
        search_results_text += f"URL: {item['url']}\n"
        search_results_text += f"Content: {item['content']}\n"

    # Format prompt
    current_date = get_current_date()
    formatted_prompt = web_searcher_instructions.format(
        current_date=current_date,
        research_topic=search_query,
    )

    # Add search results to prompt
    formatted_prompt += f"\n\nSearch Results:\n{search_results_text}"
    formatted_prompt += "\n\nIMPORTANT INSTRUCTIONS:"
    formatted_prompt += (
        "\n- You MUST include citation markers (e.g., [0-0], [0-1]) at the END of each sentence"
    )
    formatted_prompt += "\n- Use the exact citation_id format provided in the sources above"
    formatted_prompt += "\n- Example: 'QEMU is an open-source virtualization tool [0-0]. It supports multiple architectures [0-1].'"
    formatted_prompt += (
        "\n- Write a well-structured summary (200-400 words) that synthesizes the key findings"
    )
    formatted_prompt += "\n- Focus on factual information from the search results only"

    # Initialize LLM
    llm = init_chat_model(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        model_provider=os.getenv("MODEL_PROVIDER", "openai"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.3,  # Lower temperature ensures more accurate citations
        max_retries=2,
    )

    try:
        # Call LLM to generate summary
        synthesized_result = llm.invoke(formatted_prompt)
        synthesized_text = synthesized_result.content

        logger.info(f"✅ LLM synthesis completed, length: {len(synthesized_text)} chars")
        logger.info(f"📝 Synthesized text preview: {synthesized_text[:200]}...")

    except Exception as e:
        logger.error(f"❌ LLM synthesis failed: {e!s}")
        logger.warning("⚠️ Falling back to simple concatenation")

        # Fallback: simple concatenation
        research_text_parts = []
        for item in raw_results_for_llm:
            research_text_parts.append(f"{item['content']} {item['citation_id']}")
        synthesized_text = "\n\n".join(research_text_parts)

    logger.info(f"✅ Successfully collected {len(sources_gathered)} sources")
    logger.info(f"🔗 First source: {sources_gathered[0] if sources_gathered else 'N/A'}")

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [synthesized_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Context for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Context.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    current_loop_count = state.get("research_loop_count", 0) + 1

    logger.info(f"🔄 Reflection Loop {current_loop_count} starting...")
    logger.info(f"📊 Current search query count: {len(state.get('search_query', []))}")
    logger.info(f"📚 Web research results count: {len(state.get('web_research_result', []))}")

    # Format the prompt
    current_date = get_current_date()

    # Truncate summaries to prevent token overflow
    # Keep max 1000 characters per summary
    web_results = state.get("web_research_result", [])
    truncated_summaries = []
    for idx, summary in enumerate(web_results):
        truncated = summary[:1000] if len(summary) > 1000 else summary
        if len(summary) > 1000:
            truncated += f"\n... [Summary {idx + 1} truncated, original length: {len(summary)} chars]"
        truncated_summaries.append(truncated)

    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(truncated_summaries),
    )

    # init Reasoning Model with increased max_tokens
    llm = init_chat_model(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        model_provider=os.getenv("MODEL_PROVIDER", "openai"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=1.0,
        max_retries=2,
        max_tokens=2000,
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    logger.info(
        f"✅ Reflection result: is_sufficient={result.is_sufficient}, follow_up_queries={len(result.follow_up_queries)}"
    )

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": current_loop_count,
        "number_of_ran_queries": len(state.get("search_query", [])),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Context for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_answer")
    """
    configurable = Context.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )

    current_loop = state.get("research_loop_count", 0)
    is_sufficient = state.get("is_sufficient", False)
    follow_up_count = len(state.get("follow_up_queries", []))

    logger.info(f"🔍 Evaluate Research - Loop: {current_loop}/{max_research_loops}")
    logger.info(f"📊 Is Sufficient: {is_sufficient}")
    logger.info(f"❓ Follow-up queries: {follow_up_count}")

    # Check termination conditions
    if is_sufficient:
        logger.info("✅ Research is sufficient, finalizing answer...")
        return "finalize_answer"
    elif current_loop >= max_research_loops:
        logger.info(f"⚠️ Max research loops ({max_research_loops}) reached, finalizing answer...")
        return "finalize_answer"
    else:
        logger.info(f"🔄 Continuing research with {follow_up_count} follow-up queries...")
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model
    llm = init_chat_model(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        model_provider=os.getenv("MODEL_PROVIDER", "openai"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=1.0,
        max_retries=2,
        max_tokens=2000,
    )
    result = llm.invoke(formatted_prompt)

    # ============ Academic-style citation system ============
    content = result.content

    # Step 0: Remove incorrect code block markers (if LLM mistakenly generated bash/code blocks)
    import re

    content = re.sub(r"```[\w]*\n", "", content)  # Remove opening markers
    content = re.sub(r"\n```", "", content)  # Remove closing markers
    content = content.replace("```", "")  # Remove any remaining ```

    logger.info(f"📝 Original content preview: {content[:500]}...")

    # Step 1: Build citation_id -> source mapping
    citation_map = {}
    all_sources = state.get("sources_gathered", [])
    logger.info(f"📊 Total sources_gathered from state: {len(all_sources)}")

    # Check if sources exist
    if len(all_sources) == 0:
        logger.error("❌ CRITICAL: No sources_gathered in state!")
        logger.error("💡 Possible causes:")
        logger.error("   1. Tavily API quota exhausted (HTTP 432)")
        logger.error("   2. All search queries failed")
        logger.error("   3. Network connection issues")
        logger.warning("⚠️ Will generate answer without citations")

    for source in all_sources:
        citation_id = source.get("citation_id", "")
        if citation_id:
            citation_map[citation_id] = source

    logger.info(f"📚 Total available sources in citation_map: {len(citation_map)}")
    if citation_map:
        logger.info(
            f"🔍 Citation map keys sample: {list(citation_map.keys())[:10]}"
        )  # Only show first 10

    # Step 2: Extract actually used citations (in order of appearance)
    used_sources = []  # Ordered list
    citation_to_number = {}  # citation_id -> citation number (e.g. 1, 2, 3)
    seen_urls = set()

    # Iterate through all possible citation_ids, check if they appear in the text
    for citation_id, source in citation_map.items():
        if citation_id in content:
            logger.info(f"✅ Found citation_id in content: {citation_id}")
            url = source.get("url", "")

            # Deduplication: same URL keeps only one number
            if url and url in seen_urls:
                # Find existing citation number
                for idx, existing_source in enumerate(used_sources, 1):
                    if existing_source.get("url") == url:
                        citation_to_number[citation_id] = idx
                        break
            else:
                # New source, assign new number
                used_sources.append(source)
                citation_to_number[citation_id] = len(used_sources)
                if url:
                    seen_urls.add(url)
        else:
            logger.debug(f"❌ Citation_id NOT found in content: {citation_id}")

    logger.info(f"✅ Used sources in final answer: {len(used_sources)}")
    logger.info(f"📝 Citation mapping: {citation_to_number}")

    # Step 3: If no citations found, use all sources
    if not used_sources and all_sources:
        logger.warning("⚠️ No citations found in LLM output, using all sources")
        used_sources = all_sources

    # Step 4: Replace citation_id with standard academic citation format [number]
    content_with_citations = content
    for citation_id, ref_number in sorted(
        citation_to_number.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if citation_id in content_with_citations:
            inline_citation = f"[{ref_number}]"
            logger.info(f"🔗 Replacing '{citation_id}' with '{inline_citation}'")
            content_with_citations = content_with_citations.replace(citation_id, inline_citation)

    # Step 5: Optimize citation positions (remove excess spaces and newlines)
    content_with_citations = re.sub(
        r"\n+\s*(\[\d+\])",  # Multiple newlines + possible spaces + [1]
        r" \1",  # Single space + [1]
        content_with_citations,
    )

    content_with_citations = re.sub(
        r"(\[\d+\])\s*\n(?!\n)",  # [1] + space + single newline (not followed by newline)
        r"\1 ",  # [1] + space
        content_with_citations,
    )

    content_with_citations = re.sub(
        r"(\[\d+\])\s+(\[\d+\])",  # [1]  [2]
        r"\1 \2",  # [1] [2]
        content_with_citations,
    )

    # Step 6: Clean up LLM-generated references section to avoid duplication
    content_with_citations = re.sub(
        r"\n*#+\s*References.*", "", content_with_citations, flags=re.IGNORECASE | re.DOTALL
    )
    content_with_citations = re.sub(
        r"\n*References:.*", "", content_with_citations, flags=re.IGNORECASE | re.DOTALL
    )

    # Step 7: Add unified references list at the bottom of the article
    if used_sources:
        logger.info(f"📚 Preparing to add {len(used_sources)} references to the final answer")
        references = "\n\n---\n\n## 📚 References\n\n"
        for idx, source in enumerate(used_sources, 1):
            url = source.get("url", "")
            title = source.get("title", "Untitled")

            # Academic citation format: [number] Title - URL
            references += f"[{idx}] {title}\n"
            if url:
                references += f"    {url}\n\n"
            else:
                references += "    (URL not provided)\n\n"

        content_with_citations += references
        logger.info(f"✅ Successfully added {len(used_sources)} references to the final answer")
    else:
        logger.error("❌ ERROR: No sources available! Check sources_gathered in state!")
        logger.error("🔍 Diagnostic Information:")
        logger.error(f"   - Total sources_gathered: {len(all_sources)}")
        logger.error(f"   - Citation map size: {len(citation_map)}")
        logger.error(f"   - Used sources: {len(used_sources)}")
        logger.error("💡 This usually happens when:")
        logger.error("   1. Tavily API quota is exhausted (HTTP 432)")
        logger.error("   2. All web searches failed")
        logger.error("   3. LLM didn't preserve citation markers from summaries")

        # Add warning note instead of empty list
        content_with_citations += "\n\n---\n\n## 📚 References\n\n"
        content_with_citations += (
            "*⚠️ Unable to provide reference sources due to API limitations. Please check Tavily API quota.*\n"
        )

    return {
        "messages": [AIMessage(content=content_with_citations)],
        "sources_gathered": used_sources,
    }


# Create the research agent graph
builder = StateGraph(OverallState, config_schema=Context)

# Define the nodes
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entry point
builder.add_edge(START, "generate_query")

# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges("generate_query", continue_to_web_research, ["web_research"])

# Reflect on the web research
builder.add_edge("web_research", "reflection")

# Evaluate the research
builder.add_conditional_edges("reflection", evaluate_research, ["web_research", "finalize_answer"])

# Finalize the answer
builder.add_edge("finalize_answer", END)

# Compile the graph
graph = builder.compile(
    name="pro-search-agent",
    # Explicitly declare this is a chat-compatible graph
    interrupt_before=None,
    interrupt_after=None,
)
