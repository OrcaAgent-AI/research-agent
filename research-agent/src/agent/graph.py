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

# 配置日志记录器
logger = logging.getLogger(__name__)

# 确保 LangSmith 追踪被启用（可选）
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
    # 初始化 Tavily 搜索
    tavily_search = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
    )

    # 执行搜索
    search_query = state["search_query"]

    try:
        logger.info(f"🔍 Starting Tavily search for query: {search_query}")
        search_results = tavily_search.invoke({"query": search_query})
        logger.info("✅ Tavily search completed")

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Tavily search exception: {error_msg}")

        # 检测特定错误类型
        if "432" in error_msg or "Client Error" in error_msg:
            logger.error("🚫 Tavily API Error 432: API 配额已用完或 API Key 无效")
            logger.error("💡 解决方案：")
            logger.error("   1. 检查 .env 文件中的 TAVILY_API_KEY")
            logger.error("   2. 访问 https://tavily.com 检查配额")
            logger.error("   3. 如需要，申请新的 API Key")
        elif "401" in error_msg or "Unauthorized" in error_msg:
            logger.error("🚫 Tavily API Error 401: API Key 无效或未授权")
        elif "429" in error_msg or "Too Many Requests" in error_msg:
            logger.error("🚫 Tavily API Error 429: 请求过于频繁，触发限流")
        elif "timeout" in error_msg.lower():
            logger.error("🚫 Tavily API Timeout: 请求超时")
        else:
            logger.error(f"🚫 Tavily API 未知错误: {error_msg}")

        search_results = []

    # 调试：打印搜索结果类型和内容
    logger.info(f"📊 Search results type: {type(search_results)}")

    # 检查是否返回了错误对象（字符串形式的错误）
    if isinstance(search_results, str):
        logger.error(f"❌ Tavily returned error string: {search_results}")
        if "HTTPError" in search_results or "432" in search_results:
            logger.error("🚫 Tavily API 配额错误 (HTTP 432)")
            logger.error("💡 请检查您的 Tavily API 配额和 Key 有效性")
        search_results = []

    # 确保 search_results 是列表
    if not isinstance(search_results, list):
        logger.warning(f"⚠️ Unexpected search_results type: {type(search_results)}")
        logger.warning("⚠️ Converting to empty list")
        search_results = []

    logger.info(f"📊 Search results count: {len(search_results)}")

    # 处理结果
    sources_gathered = []
    raw_results_for_llm = []  # 用于传递给 LLM 的原始结果

    if len(search_results) == 0:
        logger.error(f"❌ No search results returned for query: '{search_query}'")
        logger.warning("⚠️ 返回占位符以避免流程中断")
        # 返回一个占位符，避免完全失败
        return {
            "sources_gathered": [],
            "search_query": [state["search_query"]],
            "web_research_result": [
                f"⚠️ 未能获取关于 '{search_query}' 的搜索结果（可能是 API 配额限制）。"
            ],
        }

    # 步骤1: 收集来源信息和原始内容
    for idx, result in enumerate(search_results):
        citation_id = f"[{state['id']}-{idx}]"

        # 检查 result 是否为字典
        if not isinstance(result, dict):
            logger.warning(f"⚠️ Skipping non-dict result at index {idx}: {type(result)}")
            continue

        logger.info(
            f"📄 Processing result {idx}: url={result.get('url', 'N/A')}, title={result.get('title', 'N/A')}"
        )

        # 收集来源信息
        url = result.get("url", "")
        title = result.get("title", "未知标题")
        content = result.get("content", "")

        if not url:
            logger.warning(f"⚠️ Result {idx} has no URL, skipping")
            continue

        sources_gathered.append({"url": url, "title": title, "citation_id": citation_id})

        # 准备给 LLM 的结构化数据
        if content:
            raw_results_for_llm.append(
                {
                    "citation_id": citation_id,
                    "title": title,
                    "url": url,
                    "content": content[:2000],  # 限制每个结果的长度，避免 token 超限
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
            "web_research_result": ["⚠️ 未能收集到有效的搜索来源。"],
        }

    # 步骤2: 使用 LLM 合成高质量摘要（带引用）
    logger.info("🤖 Using LLM to synthesize search results into structured summary...")

    # 构建给 LLM 的搜索结果文本
    search_results_text = ""
    for item in raw_results_for_llm:
        search_results_text += f"\n\n--- Source {item['citation_id']} ---\n"
        search_results_text += f"Title: {item['title']}\n"
        search_results_text += f"URL: {item['url']}\n"
        search_results_text += f"Content: {item['content']}\n"

    # 格式化 prompt
    current_date = get_current_date()
    formatted_prompt = web_searcher_instructions.format(
        current_date=current_date,
        research_topic=search_query,
    )

    # 添加搜索结果到 prompt
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

    # 初始化 LLM
    llm = init_chat_model(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        model_provider=os.getenv("MODEL_PROVIDER", "openai"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.3,  # 较低温度确保更准确的引用
        max_retries=2,
    )

    try:
        # 调用 LLM 生成摘要
        synthesized_result = llm.invoke(formatted_prompt)
        synthesized_text = synthesized_result.content

        logger.info(f"✅ LLM synthesis completed, length: {len(synthesized_text)} chars")
        logger.info(f"📝 Synthesized text preview: {synthesized_text[:200]}...")

    except Exception as e:
        logger.error(f"❌ LLM synthesis failed: {e!s}")
        logger.warning("⚠️ Falling back to simple concatenation")

        # 回退方案：简单拼接
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

    # 截断摘要以防止 token 超限
    # 每个摘要最多保留前 1000 个字符
    web_results = state.get("web_research_result", [])
    truncated_summaries = []
    for idx, summary in enumerate(web_results):
        truncated = summary[:1000] if len(summary) > 1000 else summary
        if len(summary) > 1000:
            truncated += f"\n... [摘要 {idx + 1} 已截断，原长度: {len(summary)} 字符]"
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

    # 检查终止条件
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

    # ============ 学术风格引用系统 ============
    content = result.content

    # 步骤0: 移除错误的代码块标记（如果LLM误生成了bash/code块）
    import re

    content = re.sub(r"```[\w]*\n", "", content)  # 移除开始标记
    content = re.sub(r"\n```", "", content)  # 移除结束标记
    content = content.replace("```", "")  # 移除任何残留的```

    logger.info(f"📝 Original content preview: {content[:500]}...")

    # 步骤1: 构建 citation_id -> source 的映射
    citation_map = {}
    all_sources = state.get("sources_gathered", [])
    logger.info(f"📊 Total sources_gathered from state: {len(all_sources)}")

    # 检查是否有来源
    if len(all_sources) == 0:
        logger.error("❌ CRITICAL: No sources_gathered in state!")
        logger.error("💡 可能的原因:")
        logger.error("   1. Tavily API 配额已用完 (HTTP 432)")
        logger.error("   2. 所有搜索查询都失败了")
        logger.error("   3. 网络连接问题")
        logger.warning("⚠️ 将生成不带引用的答案")

    for source in all_sources:
        citation_id = source.get("citation_id", "")
        if citation_id:
            citation_map[citation_id] = source

    logger.info(f"📚 Total available sources in citation_map: {len(citation_map)}")
    if citation_map:
        logger.info(
            f"🔍 Citation map keys sample: {list(citation_map.keys())[:10]}"
        )  # 只显示前10个

    # 步骤2: 提取实际使用的引用（按出现顺序）
    used_sources = []  # 有序列表
    citation_to_number = {}  # citation_id -> 引用编号（如 1, 2, 3）
    seen_urls = set()

    # 遍历所有可能的 citation_id，检查是否在文本中出现
    for citation_id, source in citation_map.items():
        if citation_id in content:
            logger.info(f"✅ Found citation_id in content: {citation_id}")
            url = source.get("url", "")

            # 去重：相同 URL 只保留一个编号
            if url and url in seen_urls:
                # 查找已有的引用编号
                for idx, existing_source in enumerate(used_sources, 1):
                    if existing_source.get("url") == url:
                        citation_to_number[citation_id] = idx
                        break
            else:
                # 新来源，分配新编号
                used_sources.append(source)
                citation_to_number[citation_id] = len(used_sources)
                if url:
                    seen_urls.add(url)
        else:
            logger.debug(f"❌ Citation_id NOT found in content: {citation_id}")

    logger.info(f"✅ Used sources in final answer: {len(used_sources)}")
    logger.info(f"📝 Citation mapping: {citation_to_number}")

    # 步骤3: 如果没有找到任何引用，使用所有来源
    if not used_sources and all_sources:
        logger.warning("⚠️ No citations found in LLM output, using all sources")
        used_sources = all_sources

    # 步骤4: 替换 citation_id 为标准学术引用格式 [数字]
    content_with_citations = content
    for citation_id, ref_number in sorted(
        citation_to_number.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if citation_id in content_with_citations:
            inline_citation = f"[{ref_number}]"
            logger.info(f"🔗 Replacing '{citation_id}' with '{inline_citation}'")
            content_with_citations = content_with_citations.replace(citation_id, inline_citation)

    # 步骤5: 优化引用位置（移除多余空格和换行）
    content_with_citations = re.sub(
        r"\n+\s*(\[\d+\])",  # 多个换行 + 可能的空格 + [1]
        r" \1",  # 单个空格 + [1]
        content_with_citations,
    )

    content_with_citations = re.sub(
        r"(\[\d+\])\s*\n(?!\n)",  # [1] + 空格 + 单换行（后面不是换行）
        r"\1 ",  # [1] + 空格
        content_with_citations,
    )

    content_with_citations = re.sub(
        r"(\[\d+\])\s+(\[\d+\])",  # [1]  [2]
        r"\1 \2",  # [1] [2]
        content_with_citations,
    )

    # 步骤6: 清理LLM可能生成的参考文献部分，避免重复
    content_with_citations = re.sub(
        r"\n*#+\s*参考文献.*", "", content_with_citations, flags=re.IGNORECASE | re.DOTALL
    )
    content_with_citations = re.sub(
        r"\n*参考文献[:：].*", "", content_with_citations, flags=re.IGNORECASE | re.DOTALL
    )
    content_with_citations = re.sub(
        r"\n*References[:：].*", "", content_with_citations, flags=re.IGNORECASE | re.DOTALL
    )

    # 步骤7: 在文章底部添加统一的参考文献列表
    if used_sources:
        logger.info(f"📚 Preparing to add {len(used_sources)} references to the final answer")
        references = "\n\n---\n\n## 📚 参考文献\n\n"
        for idx, source in enumerate(used_sources, 1):
            url = source.get("url", "")
            title = source.get("title", "Untitled")

            # 学术引用格式：[编号] 标题 - URL
            references += f"[{idx}] {title}\n"
            if url:
                references += f"    {url}\n\n"
            else:
                references += "    (URL未提供)\n\n"

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

        # 添加警告说明而不是空列表
        content_with_citations += "\n\n---\n\n## 📚 参考文献\n\n"
        content_with_citations += (
            "*⚠️ 由于 API 限制，无法提供参考文献来源。建议检查 Tavily API 配额。*\n"
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
