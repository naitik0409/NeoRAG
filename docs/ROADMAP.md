# Roadmap

Ordered by expected value. All items stay within the project's principles: free information layer, model-agnostic LLM slot, self-hostable.

## Next

1. **Server mode** - FastAPI wrapper (`/ask`, `/ingest`, `/research`) + OpenAI-compatible `/v1/chat/completions` endpoint, so any UI or tool can use NeoRAG as its backend.
2. **Multi-hop agent trajectories** - dataset generator that chains 2-3 searches before answering, teaching the fine-tuned model deeper tool use.
3. **Better default embedder** - switch default to `BAAI/bge-small-en-v1.5` after benchmarking, still free and CPU-friendly.

## Later

4. **Hybrid retrieval** - BM25 (rank-bm25, free) fused with dense scores; improves exact-match queries (product codes, names).
5. **Streaming answers** - token streaming into the terminal UI for lower perceived latency.
6. **Local fine-tune server** - one command to train on Modal/Colab free tiers and pull the merged model back automatically.
7. **Document intelligence** - tables from PDFs (pdfplumber), OCR fallback for scanned files (pytesseract), section-aware chunking.
8. **Feedback loop** - thumbs-up/down in the chat UI captured into a preferences dataset; optional DPO stage after SFT.
9. **Scheduler UI** - `research` + `ingest` cron presets with a `neorag schedule` command.
10. **Packaging** - Docker image, `pipx`-installable entry point.

## Non-goals

- Paid-API dependence for any core path (retrieval, scraping, embeddings, storage).
- Telemetry or phoning home.
- LangChain-style framework abstractions - NeoRAG stays dependency-light and explicit.
