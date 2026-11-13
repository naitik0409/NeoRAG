# Changelog

All notable changes to NeoRAG are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [SemVer](https://semver.org/).

## [0.2.0] - 2026-11-13

### Added
- **PyPI package** (`pip install neorag`) with `neorag` console command.
- Zero-config experience: with only `GROQ_API_KEY` set, NeoRAG runs the Groq free tier with no config file; data dirs land in the working directory.
- `NEORAG_LLM_BACKEND` env override for the LLM backend.
- Fast hybrid mode (default): parallel local-docs + web retrieval, single LLM call per question.
- Agent mode (`--agent`, `/mode`): ReAct JSON loop with `search` + `web_search` tools, step trace via `/steps`.
- Free web research module ("own Tavily"): DuckDuckGo HTML search (SearxNG fallback), concurrent page scraping, 7-day cache, `research` CLI command that saves pages to the corpus and ingests them.
- Freshness-aware caching: time-sensitive queries bypass the page cache automatically (`web.freshness: auto | always | cached`).
- Rich terminal UI: banner, phase-aware spinner, answer panel, source list, zero log spam; slash commands `/help /mode /web /steps /stats /clear /quit`.
- Model-agnostic LLM layer: local HuggingFace backend + any OpenAI-compatible API backend (verified live on Groq: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`).
- Fine-tuning pipeline: synthetic QA + agent-trajectory dataset generation, chat-template SFT with prompt masking, LoRA/QLoRA trainer, adapter merge, evaluation (token-F1, answer rate, citation precision), free Colab training notebook.
- Production documentation set, MIT license.

### Fixed
- Groq access blocked by Cloudflare (error 1010) - switched to the official OpenAI SDK with browser-UA fallback.
- `gpt-oss` reasoning models returning empty content - `reasoning_effort=low`, reasoning-text fallback, empty-answer fast-fail.
- Thread race on transformers lazy imports during startup warmup - load locks in embedder and reranker.
- Citation regex tuple crash in fast-mode rendering.
- Windows console `UnicodeEncodeError` on UTF-8 answers.
- ChromaDB collection-name validation in tests.

## [0.1.0] - 2026-11-13

### Added
- Initial release: ingestion (8 file types), chunking, MiniLM embeddings, ChromaDB store, retriever with optional cross-encoder reranker, ReAct agent, project scaffold, test suite.
