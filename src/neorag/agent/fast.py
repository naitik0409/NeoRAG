from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from neorag.data.qa_dataset import RAG_QA_SYSTEM_PROMPT
from neorag.retrieval.passage import Passage


def _build_context(passages: list[Passage], max_chars: int) -> tuple[str, int]:
    lines = []
    used = 0
    count = 0
    for i, p in enumerate(passages, start=1):
        header = f"[{i}] (source: {p.metadata.get('source', 'unknown')})"
        block = f"{header}\n{p.text}"
        if used + len(block) > max_chars:
            break
        lines.append(block)
        used += len(block)
        count = i
    return "\n\n".join(lines), count


def fast_answer(
    llm,
    retriever,
    web_searcher,
    question: str,
    max_context_chars: int = 12000,
    on_phase=None,
) -> tuple[str, list[Passage]]:
    phases = on_phase or (lambda s: None)

    def fetch_web() -> list[dict]:
        if web_searcher is None:
            return []
        try:
            return web_searcher.fetch_pages(question)
        except Exception:
            return []

    phases("searching local docs and the web")
    with ThreadPoolExecutor(max_workers=2) as pool:
        local_future = pool.submit(retriever.retrieve, question)
        web_future = pool.submit(fetch_web)
        local_passages = list(local_future.result() or [])
        web_pages = web_future.result() or []

    phases(f"reading {len(local_passages) + len(web_pages)} sources")
    web_passages = (
        web_searcher.rank_pages_to_passages(question, web_pages) if web_searcher else []
    )
    passages = local_passages + web_passages
    if not passages:
        return "I could not find the answer in the provided documents or on the web.", []

    context, count = _build_context(passages, max_context_chars)
    passages = passages[:count]

    phases("generating answer")
    messages = [
        {"role": "system", "content": RAG_QA_SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
    ]
    answer = llm.generate(messages)
    return answer, passages
