from __future__ import annotations

import json
from pathlib import Path

from neorag.agent.react import extract_json_object
from neorag.agent.prompts import SYSTEM_PROMPT, OBSERVATION_HEADER
from neorag.data.chunker import Chunk
from neorag.logging_utils import logger

RAG_QA_SYSTEM_PROMPT = (
    "You are a retrieval-grounded assistant. Answer strictly using the numbered passages "
    "provided in the user message and cite the passage numbers you used, like [1] or [1][3]. "
    "If the passages do not contain the answer, say you cannot find it."
)

QA_GENERATION_PROMPT = """Read the passage below and produce {n} distinct question/answer pairs.

Rules:
- Answers must be grounded ONLY in the passage, concise (1-3 sentences).
- Vary question style: factual, explanatory, comparative.
- If the passage is trivial or has no extractable facts, return an empty list.

Return exactly ONE JSON object: {{"questions": [{{"question": "...", "answer": "..."}}, ...]}}

PASSAGE:
{passage}"""


def _chunk_to_passage(chunk: Chunk, number: int = 1) -> str:
    source = chunk.metadata.get("source", "unknown")
    return f"[{number}] (source: {source})\n{chunk.text}"


def generate_qa_for_chunk(llm, chunk: Chunk, n: int = 1) -> list[dict]:
    prompt = QA_GENERATION_PROMPT.format(n=n, passage=chunk.text)
    raw = llm.generate(
        [
            {"role": "system", "content": "You output exactly one JSON object and nothing else."},
            {"role": "user", "content": prompt},
        ],
        max_new_tokens=1024,
        temperature=0.8,
    )
    obj = extract_json_object(raw)
    if not obj:
        return []
    items = obj.get("questions")
    if not isinstance(items, list):
        return []
    cleaned = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            cleaned.append({"question": question, "answer": answer})
    return cleaned[:n]


def build_context_only_example(chunk: Chunk) -> dict:
    messages = [
        {"role": "system", "content": RAG_QA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"CONTEXT:\n{_chunk_to_passage(chunk)}\n\nQUESTION: Summarize the key points of the passage.",
        },
        {
            "role": "assistant",
            "content": f"[1] {chunk.text.split(chr(10))[0][:400].strip()}",
        },
    ]
    return {"messages": messages, "meta": {"kind": "template", "chunk_id": chunk.chunk_id}}


def build_direct_example(chunk: Chunk, question: str, answer: str) -> dict:
    messages = [
        {"role": "system", "content": RAG_QA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"CONTEXT:\n{_chunk_to_passage(chunk)}\n\nQUESTION: {question}",
        },
        {"role": "assistant", "content": f"[1] {answer}"},
    ]
    return {
        "messages": messages,
        "meta": {"kind": "direct_qa", "chunk_id": chunk.chunk_id, "question": question, "answer": answer},
    }


def build_trajectory_example(chunk: Chunk, question: str, answer: str) -> dict:
    search_action = json.dumps(
        {
            "thought": f"I need to find information about: {question[:80]}",
            "action": "search",
            "action_input": {"query": " ".join(question.split()[:8])},
        },
        ensure_ascii=False,
    )
    answer_action = json.dumps(
        {"thought": "The retrieved passage contains the answer.", "action": "answer",
         "action_input": {"answer": answer, "citations": [1]}},
        ensure_ascii=False,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": search_action},
        {"role": "user", "content": f"{OBSERVATION_HEADER}:\n{_chunk_to_passage(chunk)}"},
        {"role": "assistant", "content": answer_action},
    ]
    return {
        "messages": messages,
        "meta": {"kind": "trajectory", "chunk_id": chunk.chunk_id, "question": question, "answer": answer},
    }


def build_sft_examples(
    llm,
    chunks: list[Chunk],
    examples_per_chunk: int = 1,
    include_trajectories: bool = True,
    include_template: bool = False,
    limit: int | None = None,
) -> list[dict]:
    examples: list[dict] = []
    for i, chunk in enumerate(chunks):
        if limit and len(examples) >= limit:
            break
        generated = generate_qa_for_chunk(llm, chunk, n=examples_per_chunk)
        if generated:
            for qa in generated:
                examples.append(build_direct_example(chunk, qa["question"], qa["answer"]))
                if include_trajectories:
                    examples.append(
                        build_trajectory_example(chunk, qa["question"], qa["answer"])
                    )
        elif include_template:
            examples.append(build_context_only_example(chunk))
        if (i + 1) % 25 == 0:
            logger.info(f"QA generation progress: {i + 1}/{len(chunks)} chunks, {len(examples)} examples")
    return examples


def write_jsonl(examples: list[dict], out_path: str | Path) -> int:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(examples)} examples to {path}")
    return len(examples)
