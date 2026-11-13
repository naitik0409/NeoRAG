# Security

## API keys

- Keys live **only** in `.env` at the project root. That file is git-ignored and must never be committed.
- `.env.example` ships with placeholder values only - safe for the repository.
- The key is read at process start by the zero-dependency loader (`neorag/env.py`) and passed to the LLM client as a bearer token. It is never logged, never rendered in the UI, and never written to disk by NeoRAG.
- If a key was ever pasted into a chat, ticket, or screenshot - **rotate it immediately** (Groq: console.groq.com -> API Keys). A rotated key makes old copies worthless.

## What leaves your machine

| Data | Destination | When |
|---|---|---|
| Your question (+ retrieved context) | the LLM endpoint you configured | every answer |
| Search query | DuckDuckGo / SearxNG | every web search |
| HTTP GET | public websites being scraped | every web fetch |
| Nothing | anywhere | local-docs-only mode (`--direct`, web off) |

- Documents you ingest stay on disk and in the local ChromaDB. They are only sent out in the form of retrieved passages attached to a question.
- Use `llm.backend: local` for fully-offline operation: then the only outbound traffic is web search/scraping (disable with `web.enabled: false` or `--direct`).

## Scraping etiquette

- Politeness delay between sequential fetches (`web.delay`), small page caps (`max_chars_per_page`), a bounded result count, and a local cache that avoids re-hitting sites. Respect site terms; keep `scrape_pages` modest.

## Data boundaries

- `storage/web_cache/` and `storage/chroma/` are plain local folders - deleting them removes all traces.
- The fine-tuning dataset (`data/finetune/sft.jsonl`) contains passages from your corpus; treat it as sensitive as the corpus itself.

## Reporting

Security issues: open a private issue at https://github.com/naitik0409/NeoRAG/issues or contact the developer via the GitHub profile.
