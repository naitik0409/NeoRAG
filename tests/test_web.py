import numpy as np

from neorag.web.scraper import PageScraper, slugify
from neorag.web.search import unwrap_ddg_href
from neorag.web.websearch import WebSearcher, WebSearchTool


def test_unwrap_ddg_redirect():
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc123"
    assert unwrap_ddg_href(href) == "https://example.com/page"


def test_unwrap_direct_url():
    assert unwrap_ddg_href("https://example.com/page") == "https://example.com/page"
    assert unwrap_ddg_href("//example.com/page") == "https://example.com/page"


def test_slugify():
    assert slugify("What is RAG? (2026)") == "what-is-rag-2026"
    assert slugify("!!!") == "untitled"


def test_scraper_extract_offline(tmp_path):
    scraper = PageScraper(cache_dir=tmp_path)
    html = "<html><head><title>My Title</title></head><body><script>x()</script><p>hello world</p><p>second line</p></body></html>"
    page = scraper._extract(html, "http://example.com")
    assert page.title == "My Title"
    assert "hello world" in page.text
    assert "second line" in page.text
    assert "x()" not in page.text


class FakeEmbedder:
    def __init__(self):
        self.table = {
            "alpha": np.array([1.0, 0.0]),
            "beta": np.array([0.0, 1.0]),
        }

    def embed_query(self, text: str) -> np.ndarray:
        return self.table["alpha"]

    def embed(self, texts):
        return np.array([self.table["alpha" if "alpha" in t else "beta"] for t in texts])


def make_searcher(fake_search, fake_fetch, embedder=FakeEmbedder()):
    return WebSearcher(
        scraper=None,
        embedder=embedder,
        chunk_size=50,
        chunk_overlap=0,
        top_k=2,
        scrape_pages=2,
        max_results=5,
    )


def test_websearcher_ranks_and_returns_passages():
    def fake_fetch(url):
        text = (
            "alpha alpha alpha alpha alpha alpha"
            if url == "http://a.com"
            else "beta beta beta beta beta beta"
        )
        return type("P", (), {"url": url, "title": "T", "text": text})()

    searcher = WebSearcher(
        scraper=type("S", (), {"fetch": staticmethod(fake_fetch)})(),
        embedder=FakeEmbedder(),
        chunk_size=50,
        chunk_overlap=0,
        top_k=2,
        scrape_pages=2,
        max_results=5,
    )
    searcher.search = lambda q: [
        {"url": "http://a.com", "title": "A", "snippet": ""},
        {"url": "http://b.com", "title": "B", "snippet": ""},
    ]

    passages = searcher.retrieve("alpha")
    assert len(passages) == 2
    assert "alpha" in passages[0].text
    assert passages[0].metadata["origin"] == "web"
    assert passages[0].metadata["source"] == "http://a.com"
    assert passages[0].score >= passages[1].score


def test_websearcher_falls_back_to_snippets(monkeypatch):
    searcher = WebSearcher(
        scraper=None, embedder=None, chunk_size=50, chunk_overlap=0, top_k=3, scrape_pages=2
    )
    monkeypatch.setattr(
        searcher,
        "search",
        lambda q: [{"url": "http://x.com", "title": "X", "snippet": "useful snippet"}],
    )
    searcher.scraper = type("S", (), {"fetch": staticmethod(lambda url: None)})()
    passages = searcher.retrieve("anything")
    assert len(passages) == 1
    assert passages[0].text == "useful snippet"
    assert passages[0].metadata["origin"] == "web_snippet"


def test_time_sensitive_cache_bypass(tmp_path):
    from neorag.web.websearch import WebSearcher

    calls: list[bool] = []

    class RecordingScraper:
        def fetch(self, url, use_cache=True):
            calls.append(use_cache)
            return type("P", (), {"url": url, "title": "T", "text": "alpha alpha price today"})()

    searcher = WebSearcher(
        scraper=RecordingScraper(),
        embedder=FakeEmbedder(),
        chunk_size=50,
        chunk_overlap=0,
        top_k=2,
        scrape_pages=1,
        freshness="auto",
    )
    searcher.search = lambda q: [{"url": "http://a.com", "title": "A", "snippet": ""}]
    searcher.retrieve("gold price today in INR")
    assert calls == [False]

    calls.clear()
    searcher.retrieve("what is retrieval augmented generation")
    assert calls == [True]

    searcher.freshness = "always"
    calls.clear()
    searcher.retrieve("anything at all")
    assert calls == [False]

    searcher.freshness = "cached"
    calls.clear()
    searcher.retrieve("gold price today")
    assert calls == [True]


def test_websearch_tool_result_shape(monkeypatch):
    searcher = WebSearcher(
        scraper=None, embedder=None, chunk_size=50, chunk_overlap=0, top_k=3, scrape_pages=2
    )
    monkeypatch.setattr(
        searcher,
        "search",
        lambda q: [{"url": "http://x.com", "title": "X", "snippet": "a snippet with content"}],
    )
    searcher.scraper = type("S", (), {"fetch": staticmethod(lambda url: None)})()
    tool = WebSearchTool(searcher)
    result = tool({"query": "rag systems"})
    assert result.observation.startswith("[1]")
    assert len(result.passages) == 1
    empty = tool({})
    assert "empty" in empty.observation.lower()
