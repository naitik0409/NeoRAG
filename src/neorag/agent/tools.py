from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from neorag.agent.prompts import NO_RESULTS_OBSERVATION
from neorag.retrieval.passage import Passage, format_context
from neorag.retrieval.retriever import Retriever


@dataclass
class ToolResult:
    observation: str
    passages: list[Passage]


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[[dict], ToolResult]


class SearchDocumentsTool:
    def __init__(self, retriever: Retriever, top_k: int | None = None):
        self.retriever = retriever
        self.top_k = top_k

    def __call__(self, action_input: dict) -> ToolResult:
        query = str(action_input.get("query", "")).strip()
        if not query:
            return ToolResult(observation=NO_RESULTS_OBSERVATION, passages=[])
        retriever = self.retriever
        if self.top_k:
            original_final = retriever.final_k
            retriever.final_k = self.top_k
            try:
                passages = retriever.retrieve(query)
            finally:
                retriever.final_k = original_final
        else:
            passages = retriever.retrieve(query)
        if not passages:
            return ToolResult(observation=NO_RESULTS_OBSERVATION, passages=[])
        text, _ = format_context(passages)
        return ToolResult(observation=text, passages=passages)


def build_web_tool(web_searcher, top_k: int | None = None) -> Tool:
    from neorag.web.websearch import WebSearchTool

    searcher = WebSearchTool(web_searcher, top_k=top_k)
    return Tool(
        name="web_search",
        description=(
            "Search the public web for fresh information. "
            'Input: {"query": "<search words>"}. Returns numbered passages with URLs as sources.'
        ),
        fn=searcher,
    )


def default_tools(
    retriever: Retriever,
    web_searcher=None,
    search_top_k: int | None = None,
    web_top_k: int | None = None,
) -> list[Tool]:
    search = Tool(
        name="search",
        description=(
            "Search the private document collection. "
            'Input: {"query": "<search words>"}. Returns numbered passages.'
        ),
        fn=SearchDocumentsTool(retriever, top_k=search_top_k),
    )
    tools = [search]
    if web_searcher is not None:
        tools.append(build_web_tool(web_searcher, top_k=web_top_k))
    return tools
