from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neorag.agent.tools import Tool

_INTRO = """You are NeoRAG, a retrieval-grounded answering agent. You answer only from information returned by tools, never from imagination. The tools may search a private document collection, the public web, or both."""

_ACTION_FORMAT = """Each turn you must output exactly ONE JSON object and nothing else. No markdown, no extra text. Format:

{"thought": "<your reasoning>", "action": "<tool name or answer/give_up>", "action_input": {<tool-specific input>}}"""

_RULES = """Rules:
1. After each tool call you will receive an OBSERVATION containing numbered passages like [1], [2].
2. In the final "answer", cite the numbers of the passages you used, e.g. "citations": [1, 3].
3. Never invent facts. If observations do not contain the answer, try another tool or rephrase (up to the step limit) or choose "give_up".
4. Keep queries short and keyword-like; rewrite failed queries using synonyms."""

_TOOL_DOC_TEMPLATE = 'Tool "{name}": {description}\nInput example: {input_example}'


def _default_input_example(tool: Any) -> str:
    if "query" in tool.description.lower():
        return '{"query": "<search words>"}'
    return '{"value": "<input>"}'


def build_system_prompt(tools: list[Tool]) -> str:
    tool_docs = [
        _TOOL_DOC_TEMPLATE.format(
            name=t.name, description=t.description, input_example=_default_input_example(t)
        )
        for t in tools
    ]
    tool_section = "\n\n".join(tool_docs) if tool_docs else "No tools are available; answer or give_up directly."
    return f"{_INTRO}\n\n{_ACTION_FORMAT}\n\n{_RULES}\n\nAvailable tools:\n{tool_section}\n"


SYSTEM_PROMPT = (
    f"{_INTRO}\n\n{_ACTION_FORMAT}\n\n{_RULES}\n\n"
    'Tools:\nTool "search": Search the private document collection. Input example: {"query": "<search words>"}\n'
)

OBSERVATION_HEADER = "OBSERVATION (numbered passages retrieved for your query)"

LAST_CHANCE_PROMPT = """Based only on the OBSERVATIONS above, now respond with exactly ONE JSON object using the "answer" action:

{"thought": "<reasoning>", "action": "answer", "action_input": {"answer": "<final answer grounded in the passages>", "citations": [<numbers>]}}

If the passages truly do not contain the answer, use the "give_up" action instead."""

NO_RESULTS_OBSERVATION = "No passages matched this query. Try different or simpler keywords."
