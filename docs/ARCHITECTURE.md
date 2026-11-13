# Architecture

## Module map

```
src/neorag/
  config.py               dataclass config + YAML loading + `auto` resolution (device/dtype/quantization)
  env.py                  zero-dependency .env loader
  logging_utils.py        rich logging (silenced to ERROR inside the chat UI)
  pipeline.py             component wiring (embedder, store, retriever, web searcher, LLM, agent)
  cli.py                  argparse command surface: ingest, research, chat, build-dataset, train, merge, evaluate, wipe
  ui.py                   rich terminal UI: banner, spinner with live phases, panels, slash commands

  data/
    loaders.py            Document loading: txt, md, pdf, docx, html, csv, json -> Document(doc_id, text, metadata)
    chunker.py            word-window chunking with overlap -> Chunk(chunk_id, doc_id, text, metadata)
    qa_dataset.py         SFT data generation: direct QA examples + agent trajectory examples + template fallback

  embeddings/
    embedder.py           MiniLM-style mean-pooling embedder (transformers), thread-safe lazy load, batched

  store/
    vector_store.py       ChromaDB PersistentClient wrapper: upsert, cosine query, clear, per-doc delete

  retrieval/
    passage.py            Passage dataclass + context formatting ([n] numbered)
    retriever.py          dense retrieval (embed -> query -> optional rerank -> top-k)
    reranker.py           optional cross-encoder reranker (thread-safe lazy load)

  web/                    the "free Tavily" - no keys, no credits
    search.py             DuckDuckGo HTML/lite scraping + SearxNG JSON, redirect unwrapping
    scraper.py            concurrent-safe page fetcher with 7-day cache and politeness delay
    websearch.py          WebSearcher: search -> fetch pages -> chunk -> embed -> rank; freshness-aware cache

  agent/
    prompts.py            system prompt builder (tool list is injected dynamically)
    tools.py              Tool/ToolResult; SearchDocumentsTool + WebSearchTool
    react.py              ReAct JSON loop: extract_json_object, parse_action, citation validation, fast-fail
    direct.py             single-shot docs-only RAG
    fast.py               hybrid one-call mode: parallel local+web fetch, merged numbered context, 1 generation

  llm/
    base.py               LLM protocol + factory (backend: local | api)
    local_hf.py           any HuggingFace causal LM; dtype auto; optional 4-bit (bitsandbytes, CUDA only)
    openai_compat.py      any OpenAI-compatible endpoint; official SDK first, browser-UA requests fallback;
                          reasoning_effort for gpt-oss; reasoning-text fallback for empty content

  training/
    data_format.py        JSONL -> Dataset, chat-template formatting with prompt-token masking, padding collator
    sft.py                LoRA/QLoRA SFT (transformers Trainer + peft), auto target-module discovery, adapter merge

  eval/
    metrics.py            token-F1, citation precision, answer rate, gave-up rate
```

## Request flow

### Fast mode (default)
```
question
  -> Thread A: Retriever.retrieve (embed query, query Chroma)      \
  -> Thread B: WebSearcher.fetch_pages (DDG search + scrape)        > parallel
  -> merge passages (local + web), embed + rank web chunks, cap context
  -> ONE LLM call (system: cite [n] only from context)
  -> answer panel + unique source list
```

### Agent mode
```
question
  -> LLM outputs {"thought", "action", "action_input"}
  -> action in {search, web_search} -> ToolResult(passages) -> numbered OBSERVATION
  -> repeat (max_steps) or {"action": "answer", citations}
  -> invalid/empty answers fast-fail after 2 consecutive failures -> forced grounded final answer
```

## Design decisions

- **Zero-cost information layer.** Search and scraping are pure HTTP + parsing. No search-API dependency, so no credit exhaustion.
- **Freshness beats cache for time-sensitive data.** Queries matching a time-sensitive lexicon bypass the page cache; everything else reuses the 7-day cache.
- **Model-agnostic LLM slot.** Two backends cover every deployment: `local` (transformers, incl. fine-tuned checkpoints) and `api` (OpenAI-compatible). Switching models is a config change only.
- **Agent protocol is learnable data, not a framework.** The JSON action grammar lives in the prompts and in the SFT dataset, so fine-tuning teaches the base model the same protocol it must follow at inference.
- **Thread discipline.** Network (scraping, API) runs in threads; all embedding/inference happens on the main thread or behind a load lock - no concurrent transformer calls.
- **Citations everywhere.** Every passage is numbered globally per session; answers must cite; renderers validate and display unique sources.
