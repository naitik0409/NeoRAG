from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from neorag.data.chunker import Chunk
from neorag.logging_utils import logger


class VectorStore:
    def __init__(self, persist_dir: str | Path, collection_name: str = "neorag"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> int:
        if not chunks:
            return 0
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks length mismatch")
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))}
            for c in chunks
        ]
        self.collection.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings.tolist()
        )
        return len(chunks)

    def query(
        self, query_embedding: np.ndarray, k: int = 8, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        count = self.collection.count()
        if count == 0:
            return []
        k = min(k, count)
        result = self.collection.query(
            query_embeddings=query_embedding.astype(float).tolist(), n_results=k, where=where
        )
        hits = []
        for i in range(len(result["ids"][0])):
            score = 1.0 - float(result["distances"][0][i])
            hits.append(
                {
                    "id": result["ids"][0][i],
                    "text": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i] or {},
                    "score": score,
                }
            )
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Cleared collection {name}")

    def delete_doc(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})
