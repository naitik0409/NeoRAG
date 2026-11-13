from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Passage:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


def format_context(passages: list[Passage], max_chars: int = 12000) -> tuple[str, list[Passage]]:
    lines = []
    kept: list[Passage] = []
    used = 0
    for i, p in enumerate(passages, start=1):
        header = f"[{i}] (source: {p.metadata.get('source', 'unknown')}, score: {p.score:.3f})"
        block = f"{header}\n{p.text}"
        if used + len(block) > max_chars:
            break
        lines.append(block)
        kept.append(p)
        used += len(block)
    return "\n\n".join(lines), kept
