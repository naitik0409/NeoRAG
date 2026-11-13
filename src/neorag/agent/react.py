from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from neorag.agent.prompts import (
    LAST_CHANCE_PROMPT,
    NO_RESULTS_OBSERVATION,
    OBSERVATION_HEADER,
    build_system_prompt,
)
from neorag.agent.tools import Tool
from neorag.logging_utils import logger
from neorag.retrieval.passage import Passage


@dataclass
class AgentResult:
    answer: str
    citations: list[int] = field(default_factory=list)
    passages: dict[int, Passage] = field(default_factory=dict)
    steps_used: int = 0
    gave_up: bool = False
    trace: list[dict] = field(default_factory=list)


def extract_json_object(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def parse_action(text: str) -> dict:
    obj = extract_json_object(text)
    if not obj:
        return {"thought": "", "action": "invalid", "action_input": {}}
    action = str(obj.get("action", "invalid")).strip().lower()
    action_input = obj.get("action_input", obj.get("action_input:", {}))
    if not isinstance(action_input, dict):
        action_input = {"value": action_input}
    thought = str(obj.get("thought", ""))
    return {"thought": thought, "action": action, "action_input": action_input}


def _format_observation(fresh: list[tuple[int, Passage]]) -> str:
    if not fresh:
        return NO_RESULTS_OBSERVATION
    lines = [
        f"[{n}] (source: {p.metadata.get('source', 'unknown')}, score: {p.score:.3f})\n{p.text}"
        for n, p in fresh
    ]
    return "\n\n".join(lines)


class RAGAgent:
    def __init__(self, llm, tools: list[Tool], max_steps: int = 5, system_prompt: str | None = None):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.system_prompt = system_prompt

    def run(self, question: str, on_step=None) -> AgentResult:
        system_content = self.system_prompt or build_system_prompt(list(self.tools.values()))
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ]
        seen_ids: set[str] = set()
        passage_registry: dict[int, Passage] = {}
        next_number = 1
        trace: list[dict] = []
        gave_up = False
        answer: str | None = None
        citations: list[int] = []
        tool_names = sorted(self.tools.keys())
        consecutive_failures = 0

        for step in range(1, self.max_steps + 1):
            raw = self.llm.generate(messages)
            parsed = parse_action(raw)
            action = parsed["action"]
            action_input = parsed["action_input"]
            trace.append({"step": step, "thought": parsed["thought"], "action": action, "raw": raw})
            if on_step is not None:
                try:
                    on_step({"step": step, "action": action, "thought": parsed["thought"]})
                except Exception:
                    pass
            logger.info(f"[agent step {step}] action={action} thought={parsed['thought'][:120]}")

            if action == "answer":
                candidate = str(action_input.get("answer", "")).strip()
                if candidate:
                    answer = candidate
                    citations = self._validate_citations(action_input.get("citations"), passage_registry)
                    break
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    break

            if action == "give_up":
                gave_up = True
                answer = str(action_input.get("answer", "")).strip() or (
                    "I could not find the answer in the provided documents."
                )
                break

            tool = self.tools.get(action)
            if tool is not None:
                result = tool.fn(action_input)
                consecutive_failures = 0
                fresh = [p for p in result.passages if p.id not in seen_ids]
                for p in fresh:
                    seen_ids.add(p.id)
                numbered = []
                for p in fresh:
                    passage_registry[next_number] = p
                    numbered.append((next_number, p))
                    next_number += 1
                observation = _format_observation(numbered)
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"{OBSERVATION_HEADER}:\n{observation}"})
                continue

            messages.append({"role": "assistant", "content": raw})
            consecutive_failures += 1
            if consecutive_failures >= 2:
                break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That was not a valid action. Respond with exactly one JSON object using "
                        f'one of these actions: {", ".join(tool_names + ["answer", "give_up"])}. '
                        'Example: {"thought": "...", "action": "...", "action_input": {...}}'
                    ),
                }
            )

        if answer is None:
            logger.info("Step limit reached; forcing a final grounded answer.")
            messages.append({"role": "user", "content": LAST_CHANCE_PROMPT})
            raw = self.llm.generate(messages)
            parsed = parse_action(raw)
            answer = str(parsed["action_input"].get("answer", "")).strip()
            citations = self._validate_citations(
                parsed["action_input"].get("citations"), passage_registry
            )
            if parsed["action"] == "give_up":
                gave_up = True
            if not answer:
                gave_up = True
                answer = "I could not find the answer in the provided documents."

        return AgentResult(
            answer=answer,
            citations=citations,
            passages=passage_registry,
            steps_used=len(trace),
            gave_up=gave_up,
            trace=trace,
        )

    def _validate_citations(
        self, raw_citations: object, registry: dict[int, Passage]
    ) -> list[int]:
        if isinstance(raw_citations, (int, float)):
            raw_citations = [raw_citations]
        if not isinstance(raw_citations, list):
            return []
        valid = []
        for c in raw_citations:
            try:
                n = int(c)
            except (TypeError, ValueError):
                continue
            if n in registry:
                valid.append(n)
        return sorted(set(valid))
