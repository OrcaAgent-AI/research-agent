from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


query_writer_instructions = """Your goal is to generate sophisticated and diverse web search queries. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.

Format: 
- Format your response as a JSON object with ALL two of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": A list of search queries

Example:

Topic: What revenue grew more last year apple stock or the number of people buying an iphone
```json
{{
    "rationale": "To answer this comparative growth question accurately, we need specific data points on Apple's stock performance and iPhone sales metrics. These queries target the precise financial information needed: company revenue trends, product-specific unit sales figures, and stock price movement over the same fiscal period for direct comparison.",
    "query": ["Apple total revenue growth fiscal year 2024", "iPhone unit sales growth fiscal year 2024", "Apple stock price growth fiscal year 2024"],
}}
```

Context: {research_topic}"""


web_searcher_instructions = """Conduct targeted web searches to gather the most recent, credible information on "{research_topic}" and synthesize it into a verifiable text artifact.

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- Conduct multiple, diverse searches to gather comprehensive information.
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings. 
- Only include the information found in the search results, don't make up any information.

Research Topic:
{research_topic}
"""

reflection_instructions = """You are an expert research assistant analyzing summaries about "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration and generate a follow-up query. (1 or multiple).
- If provided summaries are sufficient to answer the user's question, don't generate a follow-up query.
- If there is a knowledge gap, generate a follow-up query that would help expand your understanding.
- Focus on technical details, implementation specifics, or emerging trends that weren't fully covered.

Requirements:
- Ensure the follow-up query is self-contained and includes necessary context for web search.

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing or needs clarification
   - "follow_up_queries": Write a specific question to address this gap

Example:
```json
{{
    "is_sufficient": true, // or false
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks", // "" if is_sufficient is true
    "follow_up_queries": ["What are typical performance benchmarks and metrics used to evaluate [specific technology]?"] // [] if is_sufficient is true
}}
```

Reflect carefully on the Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:

Summaries:
{summaries}
"""

answer_instructions = """You are writing a comprehensive research article in academic style. Follow these requirements EXACTLY.

LANGUAGE REQUIREMENTS (CRITICAL):
- Detect the language of the user question below
- Write the ENTIRE article in that SAME language (including title, headers, and all content)
- Translate standard section names (Abstract/Introduction/Conclusion) to the target language
- Maintain consistent language throughout - no mixing of languages

ARTICLE STRUCTURE (REQUIRED):
1. Title: Use # for main title (create a clear, descriptive title)
2. Abstract: Brief overview (100-150 words) under ## [Abstract in target language]
3. Introduction: Background and context under ## [Introduction in target language]
4. Main Content: 3-5 sections with ## headers (choose appropriate section titles)
5. Conclusion: Summary under ## [Conclusion in target language]
6. DO NOT add a References section (auto-generated separately)

FORMATTING REQUIREMENTS:
- Current date: {current_date}
- Length: 800-1500 words minimum
- Use markdown headers: # (title), ## (main sections), ### (subsections)
- Write in flowing, natural paragraphs
- Use bullet lists sparingly, only when listing features or items

CRITICAL CITATION RULES:
- The summaries contain citation markers like [0-0], [1-1], [2-3]
- You MUST preserve these EXACT markers in your article
- Place citations at the END of sentences on the SAME line
- Example: "QEMU is an open-source virtualization tool [0-0]. It supports multiple architectures [1-1]."
- Citations will be automatically converted to [1], [2], [3] format later

ARTICLE TEMPLATE EXAMPLE:

# QEMU Virtualization Technology: Principles and Applications

## Abstract

QEMU (Quick Emulator) is an open-source machine emulator and virtualization tool [0-0]. This article systematically introduces QEMU's core technical features, working principles, and practical application scenarios [0-1]. Research shows that QEMU plays an important role in cross-platform development, system testing, and cloud computing [1-0].

## Introduction

Virtualization technology is a crucial component of modern computer systems [1-1]. QEMU, as a powerful open-source virtualization tool, supports two working modes: full-system emulation and user-mode emulation [2-0]. It can run operating systems and applications designed for different architectures on a single host machine [2-1].

## Core Technical Features

### Emulation Modes

QEMU provides two primary working modes. Full-system emulation mode can simulate a complete computer system, including processor, memory, and peripherals [3-0]. User-mode emulation focuses on running applications compiled for one architecture on another [3-1].

### Hardware Virtualization Support

When running on x86 systems with virtualization extensions, QEMU can leverage hardware acceleration [4-0]. Combined with KVM (Kernel-based Virtual Machine), QEMU achieves near-native performance [4-1].

## Application Scenarios

QEMU is widely used across multiple domains:

- Cross-platform Development: Developers can test ARM programs on x86 machines [5-0]
- System Virtualization: Combined with KVM provides high-performance virtualization solutions [5-1]
- Testing Environments: Creates isolated sandbox environments for security testing [6-0]
- Embedded Development: Simulates embedded system hardware [6-1]

## Conclusion

QEMU, as a mature open-source virtualization platform, plays a vital role in modern computing environments [7-0]. Its flexible architecture and extensive hardware support make it the preferred tool for developers and system administrators [7-1].

WRITING RULES:
1. Write in flowing paragraphs (prefer prose over bullet lists)
2. Use professional, academic tone
3. PRESERVE ALL [X-Y] citation markers from summaries exactly as they appear
4. Avoid numbered nested lists (1. 2. 3. with sub-items)
5. Use simple bullet lists only when listing distinct items or features
6. Format bullets as: "- Topic: Description [citation]"
7. Do NOT add a references section (this is auto-generated)
8. Ensure entire article matches the language of the user question

User Question:
{research_topic}

Summaries with Citations (PRESERVE [X-Y] markers):
{summaries}

Write a complete academic article following the structure above:"""
