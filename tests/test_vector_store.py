import numpy as np
import pytest

from neorag.data.chunker import Chunk
from neorag.data.loaders import Document
from neorag.data.chunker import chunk_documents


def make_chunks(n: int = 2) -> list[Chunk]:
    doc = Document(doc_id="doc1", text=" ".join(f"word{i}" for i in range(40)), metadata={"source": "a.txt", "title": "a"})
    return chunk_documents([doc], chunk_size=20, chunk_overlap=4)[:n]


def test_store_roundtrip(tmp_path):
    from neorag.store.vector_store import VectorStore

    store = VectorStore(tmp_path / "chroma", "test_collection")
    assert store.count() == 0

    chunks = make_chunks(2)
    vectors = np.random.rand(2, 8).astype(np.float32)
    n = store.add_chunks(chunks, vectors)
    assert n == 2
    assert store.count() == 2

    hits = store.query(vectors[0], k=2)
    assert len(hits) == 2
    assert hits[0]["id"] == chunks[0].chunk_id
    assert hits[0]["score"] >= hits[1]["score"]
    assert hits[0]["metadata"]["source"] == "a.txt"


def test_store_clear(tmp_path):
    from neorag.store.vector_store import VectorStore

    store = VectorStore(tmp_path / "chroma2", "neorag_c2")
    chunks = make_chunks(1)
    store.add_chunks(chunks, np.random.rand(1, 8).astype(np.float32))
    assert store.count() == 1
    store.clear()
    assert store.count() == 0


def test_store_mismatch_raises(tmp_path):
    from neorag.store.vector_store import VectorStore

    store = VectorStore(tmp_path / "chroma3", "neorag_c3")
    with pytest.raises(ValueError):
        store.add_chunks(make_chunks(2), np.random.rand(1, 8).astype(np.float32))
