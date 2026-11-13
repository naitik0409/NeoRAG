from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from neorag.data.loaders import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = chunk_size - chunk_overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(
    documents: list[Document], chunk_size: int = 512, chunk_overlap: int = 64
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        pieces = chunk_text(doc.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        total = len(pieces)
        for idx, piece in enumerate(pieces):
            metadata = dict(doc.metadata)
            metadata.update({"chunk_index": idx, "total_chunks": total, "doc_id": doc.doc_id})
            digest = hashlib.sha1(f"{doc.doc_id}:{idx}".encode("utf-8")).hexdigest()[:16]
            chunks.append(Chunk(chunk_id=digest, doc_id=doc.doc_id, text=piece, metadata=metadata))
    return chunks
