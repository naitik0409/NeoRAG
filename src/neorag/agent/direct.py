from __future__ import annotations

from neorag.data.qa_dataset import RAG_QA_SYSTEM_PROMPT
from neorag.retrieval.passage import Passage, format_context


def direct_answer(llm, retriever, question: str) -> tuple[str, list[Passage]]:
    passages = retriever.retrieve(question)
    if not passages:
        return "I could not find the answer in the provided documents.", []
    context, _ = format_context(passages)
    messages = [
        {"role": "system", "content": RAG_QA_SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
    ]
    answer = llm.generate(messages)
    return answer, passages
