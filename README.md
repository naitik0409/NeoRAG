# NeoRAG

A self-hosted, **100% free** RAG agent you fine-tune from a weak open-source model into a powerful retrieval-grounded specialist.

## Design principles

1. **Information gathering is always free.** Web search (DuckDuckGo scraping) + page scraping + local embeddings + local vector store — no API keys, no credits, no rate-limit death like Tavily's free tier.
2. **The LLM is only for thinking/answering** and is **model-agnostic**: any model in the world fits the same slot — local HuggingFace checkpoints or ANY OpenAI-compatible endpoint (free tiers like Groq/OpenRouter, local Ollama/LM Studio/vLLM, or paid clouds if you ever want).
3. **Fine-tune the weak agent into a strong one.** The SFT dataset teaches the model grounded answering with citations AND the agent JSON protocol (search → observe → answer), so after training the small model behaves like a disciplined RAG agent.

## Architecture

```
data/corpus/* ──ingest──► chunker ──embed──► ChromaDB ──┐
                                                        ├─► Retriever ─┐
web: DuckDuckGo+scraper ───────────embed──────► (live) ─┘              ├─► Agent (ReAct JSON loop) ──► cited answer
                                                                       │        or direct one-shot RAG
LLM: local HF / any OpenAI-compatible API ────────────────────────────┘
                ▲
                └── fine-tuned by you: build-dataset → train (LoRA) → merge
```

## Install

```powershell
python -m pip install -e .
# optional extras (trl, faiss, sentence-transformers, bitsandbytes CUDA):
python -m pip install -e ".[train]"
```

Everything else in `requirements.txt` is already free/open-source.

## Quickstart

```powershell
# 1. Drop your files (txt, md, pdf, docx, html, csv, json) into data/corpus/
python -m neorag.cli ingest

# 2. Your own free Tavily: search the web, scrape pages, save + ingest automatically
python -m neorag.cli research "transformer architecture explained"

# 3. Chat (agent loop with search + web_search tools, citations included)
python -m neorag.cli chat
python -m neorag.cli chat --direct   # single-shot RAG, faster on CPU
```

## Fine-tuning loop (weak → powerful)

```powershell
python -m neorag.cli build-dataset          # synthetic QA + agent trajectories from YOUR corpus (free, local LLM)
python -m neorag.cli train                  # LoRA/QLoRA SFT (needs GPU: local CUDA or Colab free tier - see notebooks/)
python -m neorag.cli merge                  # merge adapter into a standalone model
# then point config/neorag.yaml at it: llm.model_id: checkpoints/neorag-sft-merged
```

On a CPU-only machine, run `train` in the free Google Colab notebook (`notebooks/colab_training.ipynb`) — no cost.

## Any LLM, one config

Edit `config/neorag.yaml` → `llm:` section:

| You want | backend | settings |
|---|---|---|
| Local HuggingFace model (free) | `local` | `model_id: Qwen/Qwen2.5-0.5B-Instruct` |
| Your fine-tuned model | `local` | `model_id: checkpoints/neorag-sft-merged` |
| Groq free tier | `api` | `NEORAG_LLM_API_BASE=https://api.groq.com/openai/v1`, `NEORAG_LLM_MODEL=llama-3.1-8b-instant`, `NEORAG_LLM_API_KEY=...` (free key) |
| OpenRouter free models | `api` | `NEORAG_LLM_API_BASE=https://openrouter.ai/api/v1`, `NEORAG_LLM_MODEL=meta-llama/llama-3.1-8b-instruct:free` |
| Ollama (local, free) | `api` | `NEORAG_LLM_API_BASE=http://localhost:11434/v1`, `NEORAG_LLM_MODEL=llama3.1` |
| LM Studio / vLLM / any OpenAI-compatible | `api` | same pattern |
| Paid clouds (OpenAI, Gemini, ...) | `api` | same pattern — works but not required |

## CLI reference

| command | what it does |
|---|---|
| `ingest` | chunk + embed your corpus into the local vector store |
| `research "<query>"` | free web search + scrape → saves to `data/corpus/web/` and ingests |
| `chat` | agent chat with `search` (docs) + `web_search` (web) tools |
| `chat --direct` | one-shot RAG (no agent loop) |
| `build-dataset` | generate the SFT fine-tuning set (direct QA + agent trajectories) |
| `train` | LoRA fine-tune |
| `merge` | merge LoRA adapter into a standalone model |
| `evaluate` | token-F1 / answer-rate / citation-precision on generated QA |
| `wipe` | clear the vector store |

## Web cache

Scraped pages are cached in `storage/web_cache/` (default 7 days) so repeated research queries cost zero time/bandwidth.

## Tests

```powershell
$env:PYTHONPATH="src"; python -m pytest tests -q
```
