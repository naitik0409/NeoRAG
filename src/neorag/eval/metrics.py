from __future__ import annotations

import re
from collections import Counter

from neorag.agent.react import AgentResult


def _normalize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction)
    gold_tokens = _normalize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def citation_precision(result: AgentResult) -> float:
    if not result.citations:
        return 0.0
    return sum(1 for c in result.citations if c in result.passages) / len(result.citations)


def evaluate_agent(
    results: list[tuple[dict, AgentResult]],
) -> dict:
    if not results:
        return {"n": 0, "token_f1": 0.0, "answer_rate": 0.0, "citation_precision": 0.0, "gave_up_rate": 0.0}
    f1s, citps, gave_up = [], [], 0
    for gold, result in results:
        f1s.append(token_f1(result.answer, gold.get("answer", "")))
        citps.append(citation_precision(result))
        gave_up += 1 if result.gave_up else 0
    n = len(results)
    return {
        "n": n,
        "token_f1": sum(f1s) / n,
        "answer_rate": (n - gave_up) / n,
        "citation_precision": sum(citps) / n,
        "gave_up_rate": gave_up / n,
        "avg_steps": sum(r.steps_used for _, r in results) / n,
    }
