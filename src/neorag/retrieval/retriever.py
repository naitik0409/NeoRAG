from __future__ import annotations

from neorag.embeddings.embedder import Embedder
from neorag.logging_utils import logger
from neorag.retrieval.passage import Passage
from neorag.retrieval.reranker import CrossEncoderReranker
from neorag.store.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int = 8,
        final_k: int = 4,
        reranker: CrossEncoderReranker | None = None,
        min_score: float = 0.0,
    ):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.final_k = final_k
        self.reranker = reranker
        self.min_score = min_score

    def retrieve(self, query: str) -> list[Passage]:
        if self.store.count() == 0:
            logger.warning("Vector store is empty; run `neorag ingest` first.")
            return []
        qvec = self.embedder.embed_query(query)
        hits = self.store.query(qvec, k=self.top_k)
        passages = [
            Passage(id=h["id"], text=h["text"], score=h["score"], metadata=h["metadata"])
            for h in hits
            if h["score"] >= self.min_score
        ]
        if self.reranker is not None and passages:
            passages = self.reranker.rerank(query, passages)
        return passages[: self.final_k]
