from __future__ import annotations

import urllib.parse
import urllib.request

from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}


def unwrap_ddg_href(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    if "uddg=" in href:
        parsed = urllib.parse.urlparse(href)
        params = urllib.parse.parse_qs(parsed.query)
        uddg = params.get("uddg", [None])[0]
        if uddg:
            return urllib.parse.unquote(uddg)
    return href


def _clean(text: str) -> str:
    return " ".join(text.split())


def _parse_ddg_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for anchor in soup.select("a.result__a"):
        href = anchor.get("href", "")
        url = unwrap_ddg_href(href)
        title = _clean(anchor.get_text())
        if not url.startswith("http") or not title:
            continue
        snippet = ""
        snippet_node = anchor.find_parent("div", class_="result")
        if snippet_node:
            snippet_tag = snippet_node.select_one("a.result__snippet")
            if snippet_tag:
                snippet = _clean(snippet_tag.get_text())
        results.append({"url": url, "title": title, "snippet": snippet})
    return results


def _parse_ddg_lite(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for anchor in soup.select("a.result-link"):
        url = unwrap_ddg_href(anchor.get("href", ""))
        title = _clean(anchor.get_text())
        if not url.startswith("http") or not title:
            continue
        row = anchor.find_parent("tr")
        snippet = ""
        if row:
            next_row = row.find_next_sibling("tr")
            if next_row:
                snippet_tag = next_row.select_one("td.result-snippet")
                if snippet_tag:
                    snippet = _clean(snippet_tag.get_text())
        results.append({"url": url, "title": title, "snippet": snippet})
    return results


def _http_post(url: str, data: bytes, timeout: int) -> str:
    req = urllib.request.Request(
        url,
        data=data,
        headers={**HEADERS, "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _http_get(url: str, timeout: int) -> str:
    req = urllib.request.Request(url, headers=HEADERS, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def duckduckgo_search(query: str, max_results: int = 8, timeout: int = 20) -> list[dict]:
    body = urllib.parse.urlencode({"q": query, "kl": "wt-wt"}).encode("utf-8")
    try:
        html = _http_post("https://html.duckduckgo.com/html/", body, timeout)
        results = _parse_ddg_html(html)
    except Exception:
        results = []
    if not results:
        try:
            html = _http_get(
                "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(query), timeout
            )
            results = _parse_ddg_lite(html)
        except Exception:
            results = []
    deduped, seen = [], set()
    for r in results:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        deduped.append(r)
        if len(deduped) >= max_results:
            break
    return deduped


def searxng_search(query: str, instance: str, max_results: int = 8, timeout: int = 20) -> list[dict]:
    url = f"{instance.rstrip('/')}/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json"}
    )
    try:
        raw = _http_get(url, timeout)
        import json

        data = json.loads(raw)
        out = []
        for item in data.get("results", [])[:max_results]:
            u = item.get("url", "")
            t = _clean(item.get("title", ""))
            if u.startswith("http") and t:
                out.append({"url": u, "title": t, "snippet": _clean(item.get("content", ""))})
        return out
    except Exception:
        return []
