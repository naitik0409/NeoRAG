from neorag.agent.fast import fast_answer
from neorag.retrieval.passage import Passage


class FakeLLM:
    captured = None

    def generate(self, messages, **kwargs):
        FakeLLM.captured = messages
        return "RAG combines retrieval with generation [1][2]."


class FakeRetriever:
    def retrieve(self, question):
        return [Passage(id="d1", text="local text", score=0.9, metadata={"source": "doc.txt"})]


class FakeWebSearcher:
    def fetch_pages(self, question):
        return [{"url": "http://w.com", "title": "W", "text": "web text here", "snippet": ""}]

    def rank_pages_to_passages(self, question, pages, top_k=None):
        return [Passage(id="w0", text="web text here", score=0.8, metadata={"source": "http://w.com", "origin": "web"})]


def test_fast_answer_merges_local_and_web():
    answer, passages = fast_answer(
        FakeLLM(), FakeRetriever(), FakeWebSearcher(), "q", on_phase=lambda s: None
    )
    assert len(passages) == 2
    assert "[1] (source: doc.txt)" in FakeLLM.captured[-1]["content"]
    assert "[2] (source: http://w.com)" in FakeLLM.captured[-1]["content"]
    assert "RAG combines" in answer


def test_fast_answer_docs_only():
    answer, passages = fast_answer(FakeLLM(), FakeRetriever(), None, "q")
    assert "[2]" not in FakeLLM.captured[-1]["content"]
    assert len(passages) == 1


def test_fast_answer_no_passages_fallback():
    empty_retriever = type("R", (), {"retrieve": lambda self, q: []})()
    answer, passages = fast_answer(FakeLLM(), empty_retriever, None, "q")
    assert "could not find" in answer.lower()
    assert passages == []


def test_fast_answer_context_cap():
    long_passages = [Passage(id=f"p{i}", text="x" * 500, score=0.5, metadata={"source": f"s{i}"}) for i in range(50)]
    retriever = type("R", (), {"retrieve": lambda self, q: long_passages})()
    _, passages = fast_answer(FakeLLM(), retriever, None, "q", max_context_chars=2000)
    assert len(passages) < 50
