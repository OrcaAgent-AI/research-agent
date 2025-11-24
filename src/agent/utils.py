"""Utility functions for the research agent."""

from typing import List

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


# ==================== DEPRECATED: Gemini-specific functions ====================
# The following functions were designed for Google Gemini's grounding_metadata API
# and are no longer used in the Tavily-based implementation.
# Kept for reference only - can be removed in future cleanup.
# ==============================================================================

# def resolve_urls(urls_to_resolve: List[Any], id: int) -> Dict[str, str]:
#     """
#     [DEPRECATED] Create a map of the vertex ai search urls to short urls.
#     This was used with Gemini's grounding_metadata.grounding_chunks.
#     Now replaced by Tavily's direct URL handling.
#     """
#     pass

# def insert_citation_markers(text, citations_list):
#     """
#     [DEPRECATED] Inserts citation markers based on Gemini's grounding indices.
#     Now replaced by custom citation_id system in web_research node.
#     """
#     pass

# def get_citations(response, resolved_urls_map):
#     """
#     [DEPRECATED] Extracts citations from Gemini's grounding_metadata.
#     Now replaced by Tavily search results processing.
#     """
#     pass

