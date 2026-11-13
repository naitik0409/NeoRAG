from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

from neorag.web.search import HEADERS, _clean, _http_get

_STRIP_TAGS = ("script", "style", "noscript", "nav", "footer", "header", "aside", "form", "iframe")


@dataclass
class WebPage:
    url: str
    title: str
    text: str
    fetched_at: str


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "untitled"


def _cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]


class PageScraper:
    def __init__(
        self,
        cache_dir: str | Path,
        timeout: int = 20,
        max_chars: int = 8000,
        max_age_hours: float = 168.0,
        delay: float = 1.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_chars = max_chars
        self.max_age_hours = max_age_hours
        self.delay = delay
        self._last_fetch = 0.0

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / f"{_cache_key(url)}.json"

    def _load_cache(self, url: str) -> WebPage | None:
        path = self._cache_path(url)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            fetched = datetime.fromisoformat(data["fetched_at"])
            age_hours = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
            if age_hours <= self.max_age_hours:
                return WebPage(
                    url=data["url"], title=data["title"], text=data["text"], fetched_at=data["fetched_at"]
                )
        except Exception:
            return None
        return None

    def _save_cache(self, page: WebPage) -> None:
        path = self._cache_path(page.url)
        path.write_text(
            json.dumps(
                {
                    "url": page.url,
                    "title": page.title,
                    "text": page.text,
                    "fetched_at": page.fetched_at,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _extract(self, html: str, url: str) -> WebPage:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(_STRIP_TAGS):
            tag.decompose()
        title = _clean(soup.title.get_text()) if soup.title else url
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)
        text = text[: self.max_chars]
        return WebPage(
            url=url, title=title, text=text, fetched_at=datetime.now(timezone.utc).isoformat()
        )

    def fetch(self, url: str, use_cache: bool = True) -> WebPage | None:
        if use_cache:
            cached = self._load_cache(url)
            if cached:
                return cached
        elapsed = time.time() - self._last_fetch
        if self.delay and self._last_fetch and elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        try:
            html = _http_get(url, timeout=self.timeout)
        except Exception:
            self._last_fetch = time.time()
            return None
        self._last_fetch = time.time()
        if not html:
            return None
        page = self._extract(html, url)
        if page.text:
            self._save_cache(page)
            return page
        return None
