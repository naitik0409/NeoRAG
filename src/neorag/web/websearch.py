from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import re

import numpy as np

from neorag.agent.tools import ToolResult
from neorag.data.chunker import chunk_text
from neorag.logging_utils import logger
from neorag.retrieval.passage import Passage, format_context
from neorag.web.scraper import PageScraper
from neorag.web.search import duckduckgo_search, searxng_search


class WebSearcher:
    def __init__(
        self,
        scraper: PageScraper,
        embedder=None,
        chunk_size: int = 220,
        chunk_overlap: int = 32,
        top_k: int = 6,
        scrape_pages: int = 3,
        max_results: int = 8,
        timeout: int = 20,
        searx_instances: list[str] | None = None,
        freshness: str = "auto",
    ):
        self.scraper = scraper
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.scrape_pages = scrape_pages
        self.max_results = max_results
        self.timeout = timeout
        self.searx_instances = searx_instances or []
        self.freshness = freshness

    _TIME_SENSITIVE_RE = re.compile(
        r"\b(today|current(ly)?|now|latest|live|right now|price|prices|rate|rates|stock|stocks|"
        r"share|shares|weather|news|score|scores|tonight|yesterday|this week|exchange|"
        r"update(d)?|spot)\b",
        re.IGNORECASE,
    )

    def _is_time_sensitive(self, query: str) -> bool:
        return bool(self._TIME_SENSITIVE_RE.search(query))

    def _should_use_cache(self, query: str) -> bool:
        if self.freshness == "always":
            return False
        if self.freshness == "cached":
            return True
        return not self._is_time_sensitive(query)

    def _fetch(self, url: str, use_cache: bool):
        try:
            return self.scraper.fetch(url, use_cache=use_cache)
        except TypeError:
            return self.scraper.fetch(url)

    def search(self, query: str) -> list[dict]:
        results = duckduckgo_search(query, max_results=self.max_results, timeout=self.timeout)
        for instance in self.searx_instances:
            if len(results) >= self.max_results:
                break
            results.extend(
                searxng_search(
                    query,
                    instance,
                    max_results=self.max_results - len(results),
                    timeout=self.timeout,
                )
            )
        deduped, seen = [], set()
        for r in results:
            if r["url"] in seen:
                continue
            seen.add(r["url"])
            deduped.append(r)
        return deduped

    def _rank_passages(self, query: str, passages: list[Passage]) -> list[Passage]:
        if self.embedder is None or not passages:
            return passages
        qvec = self.embedder.embed_query(query)
        vectors = self.embedder.embed([p.text for p in passages])
        qnorm = np.linalg.norm(qvec)
        vnorms = np.linalg.norm(vectors, axis=1)
        denom = vnorms * (qnorm if qnorm > 0 else 1)
        denom[denom == 0] = 1e-9
        scores = (vectors @ qvec) / denom
        for p, s in zip(passages, scores):
            p.score = float(s)
        passages.sort(key=lambda p: p.score, reverse=True)
        return passages

    def fetch_pages(self, query: str, max_pages: int | None = None) -> list[dict]:
        results = self.search(query)
        if not results:
            return []
        limit = max_pages or self.scrape_pages
        selected = results[:limit]
        use_cache = self._should_use_cache(query)
        fetched: list[tuple[dict, object | None]] = []
        with ThreadPoolExecutor(max_workers=min(4, len(selected))) as pool:
            fetched = list(pool.map(lambda r: (r, self._fetch(r["url"], use_cache)), selected))
        pages: list[dict] = []
        for r, page in fetched:
            if page and page.text:
                pages.append(
                    {"url": page.url, "title": page.title or r["title"], "text": page.text, "snippet": r.get("snippet", "")}
                )
            elif r.get("snippet"):
                pages.append({"url": r["url"], "title": r["title"], "text": "", "snippet": r["snippet"]})
        return pages

    def rank_pages_to_passages(self, query: str, pages: list[dict], top_k: int | None = None) -> list[Passage]:
        if not pages:
            return []
        passages: list[Passage] = []
        for page in pages:
            if page.get("text"):
                for i, piece in enumerate(chunk_text(page["text"], self.chunk_size, self.chunk_overlap)):
                    passages.append(
                        Passage(
                            id=f"web:{page['url']}:{i}",
                            text=piece,
                            score=0.0,
                            metadata={"source": page["url"], "title": page["title"], "origin": "web"},
                        )
                    )
            elif page.get("snippet"):
                passages.append(
                    Passage(
                        id=f"web-snippet:{page['url']}",
                        text=page["snippet"],
                        score=0.0,
                        metadata={"source": page["url"], "title": page["title"], "origin": "web_snippet"},
                    )
                )
        if not passages:
            return []
        return self._rank_passages(query, passages)[: (top_k or self.top_k)]

    def retrieve(self, query: str) -> list[Passage]:
        pages = self.fetch_pages(query)
        return self.rank_pages_to_passages(query, pages)

    def collect_corpus_text(self, query: str, max_pages: int | None = None) -> list[dict]:
        pages = self.fetch_pages(query, max_pages=max_pages)
        return [p for p in pages if p.get("text")]


class WebSearchTool:
    def __init__(self, searcher: WebSearcher, top_k: int | None = None):
        self.searcher = searcher
        self.top_k = top_k

    def __call__(self, action_input: dict) -> ToolResult:
        query = str(action_input.get("query", "")).strip()
        if not query:
            return ToolResult(observation="No web results (empty query).", passages=[])
        passages = self.searcher.retrieve(query)
        if not passages:
            return ToolResult(
                observation="No web results found for this query. Try different keywords.",
                passages=[],
            )
        if self.top_k:
            passages = passages[: self.top_k]
        text, _ = format_context(passages)
        return ToolResult(observation=text, passages=passages)


def build_web_searcher(cfg, embedder=None) -> WebSearcher:
    scraper = PageScraper(
        cache_dir=cfg.resolve(cfg.web.cache_dir),
        timeout=cfg.web.timeout,
        max_chars=cfg.web.max_chars_per_page,
        delay=cfg.web.delay,
    )
    return WebSearcher(
        scraper=scraper,
        embedder=embedder,
        chunk_size=cfg.web.chunk_size,
        chunk_overlap=32,
        top_k=cfg.web.top_k,
        scrape_pages=cfg.web.scrape_pages,
        max_results=cfg.web.max_results,
        timeout=cfg.web.timeout,
        searx_instances=cfg.web.searx_instances,
        freshness=cfg.web.freshness,
    )
