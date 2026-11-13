from neorag.data.chunker import chunk_documents, chunk_text
from neorag.data.loaders import Document


def make_doc(text: str, doc_id: str = "d1") -> Document:
    return Document(doc_id=doc_id, text=text, metadata={"source": "x.txt", "title": "x"})


def test_chunk_text_short_returns_single():
    text = "alpha bravo charlie delta"
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
    assert chunks == [text]


def test_chunk_text_sizes_and_overlap():
    words = [f"w{i}" for i in range(100)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
    assert all(len(c.split()) <= 20 for c in chunks)
    assert len(chunks) > 1
    first, second = chunks[0], chunks[1]
    overlap_words = set(first.split()) & set(second.split())
    assert len(overlap_words) == 5


def test_chunk_text_empty():
    assert chunk_text("   ", chunk_size=10, chunk_overlap=2) == []


def test_chunk_text_invalid_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=2, chunk_overlap=2)
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=0, chunk_overlap=0)


def test_chunk_documents_metadata():
    doc = make_doc(" ".join(f"word{i}" for i in range(60)))
    chunks = chunk_documents([doc], chunk_size=20, chunk_overlap=4)
    assert len(chunks) >= 2
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["total_chunks"] == len(chunks) for c in chunks)
    assert all(len(c.text.split()) <= 20 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.doc_id == "d1" for c in chunks)
    assert all(c.metadata["source"] == "x.txt" for c in chunks)


def test_chunk_documents_multiple_docs():
    docs = [make_doc(" ".join(f"a{i}" for i in range(30)), "d1"), make_doc("b", "d2")]
    chunks = chunk_documents(docs, chunk_size=10, chunk_overlap=2)
    assert all(c.doc_id in {"d1", "d2"} for c in chunks)
    assert any(c.doc_id == "d2" and c.text == "b" for c in chunks)
