<div align="center">

# NeoRAG

**A self-hosted RAG agent you fine-tune from a weak open-source model into a retrieval-grounded specialist.**

*100% free information layer · model-agnostic LLM slot · fully local retrieval*

[![PyPI](https://img.shields.io/pypi/v/neorag?color=blue&logo=pypi)](https://pypi.org/project/neorag/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-36%20passing-brightgreen)](tests)
[![Author](https://img.shields.io/badge/author-Naitik-8A2BE2)](https://github.com/naitik0409)

</div>

---

## What is NeoRAG?

NeoRAG is an **agentic Retrieval-Augmented Generation system** built on three uncompromising principles:

| # | Principle | Meaning |
|---|---|---|
| 1 | **Information gathering is always free** | Web search, page scraping, embeddings, and the vector store need zero API keys and zero credits — no rate-limit death, no billing surprises. |
| 2 | **The LLM slot is model-agnostic** | Any model in the world fits: local HuggingFace checkpoints, Groq free tier, OpenRouter free models, Ollama, LM Studio, vLLM, or paid clouds. Switching models is one config change. |
| 3 | **The weak model becomes a strong one** | A fine-tuning loop teaches your small local model grounded, cited answering *and* the agent's JSON tool protocol — so it behaves like a disciplined RAG specialist, not a chatbot. |

NeoRAG answers questions from your local documents **and** the live web, always with numbered citations, always through one command-line interface.

---

## How it works

```
                        ┌─────────────────────────────────────────────────┐
 data/corpus/*          │                NEO RAG PIPELINE                 │
 (pdf/md/docx/...)      │                                                 │
      │ ingest          │  ┌─────────┐   ┌──────────┐   ┌───────────┐    │
      ▼                 │  │ chunker │──▶│ embedder │──▶│ ChromaDB  │    │
 ┌──────────────┐       │  └─────────┘   │ (MiniLM) │   │ (local)   │    │
 │  load corpus │       │                └──────────┘   └─────┬─────┘    │
 └──────────────┘       │                                     │ retrieve │
                        │  ┌──────────────────────────┐       ▼          │
 web (free):            │  │ DuckDuckGo + scraper     │  ┌──────────┐    │
 DuckDuckGo HTML search │  │ 7-day cache, freshness-  │─▶│ Retriever│    │
 SearxNG fallback       │  │ aware (time-sensitive    │  └────┬─────┘    │
 ──────────────────────▶│  │ queries bypass cache)    │       │          │
                        │  └──────────────────────────┘       ▼          │
                        │                            ┌───────────┐       │
                        │                            │    LLM    │       │
                        │                            │ any model │       │
                        │                            └─────┬─────┘       │
                        └──────────────────────────────────│─────────────┘
                                                           ▼
                                            ┌──────────────────────────┐
                                            │  FAST MODE   1 LLM call  │
                                            │  AGENT MODE  ReAct loop  │
                                            │  → cited answer + sources│
                                            └──────────────────────────┘
```

---

## Installation

**From PyPI (recommended):**

```bash
pip install neorag
```

**From source:**

```bash
git clone https://github.com/naitik0409/NeoRAG.git
cd NeoRAG
pip install -e .
```

**Optional extras:**

```bash
pip install "neorag[train]"   # LoRA fine-tuning: trl, bitsandbytes, sentence-transformers, faiss
pip install "neorag[dev]"     # pytest + ruff
```

Requirements: Python 3.10+. **No GPU needed to chat** — CPU-only machines run the full fast mode. A GPU (local CUDA or free Colab T4) is only needed for the `train` step.

---

## Quickstart (60 seconds)

The fastest path is the free Groq tier — one key, no config file needed:

```bash
# 1. Get a free key at https://console.groq.com  (no card, no credits)
# 2. Make it visible:
setx GROQ_API_KEY "gsk_your_key"        # Windows (new terminal after)
export GROQ_API_KEY="gsk_your_key"      # macOS/Linux

# 3. Chat — that's it. Zero config.
neorag chat
```

Or put the key in a `.env` file in your working directory:

```ini
GROQ_API_KEY=gsk_your_key
```

Without any key, NeoRAG still works fully offline with a local model (`Qwen/Qwen2.5-0.5B-Instruct` by default — weak but free, and exactly what the fine-tuning loop upgrades):

```bash
neorag chat --backend local
```

---

## Answer modes

| Mode | How it works | Best for | Start with |
|---|---|---|---|
| **Fast** *(default)* | Parallel local-docs + web retrieval, merged numbered context, **one** LLM call | everyday questions, speed | `neorag chat` |
| **Agent** | ReAct loop: the model reasons, calls `search` / `web_search` tools step by step, cites, then answers | hard multi-hop questions | `neorag chat --agent` |
| **Direct** | Your local corpus only — web disabled | private knowledge bases | `neorag chat --direct` |

Toggle anytime inside chat with `/mode`, `/web`, `/steps`.

---

## Your own free "Tavily"

NeoRAG's web module needs no API key — it is search + scraping, entirely yours:

- **Search:** DuckDuckGo HTML (with `lite` endpoint + SearxNG fallbacks), redirect unwrapping
- **Scrape:** concurrent page fetching, 7-day local cache, politeness delay, per-page text caps
- **Freshness-aware:** time-sensitive queries (prices, news, "today", "latest"...) **bypass the cache automatically**; general questions stay cached for speed — controlled by `web.freshness: auto | always | cached`
- **Corpus building:** research a topic once, own it forever

```bash
neorag research "transformer architecture explained"   # search + scrape + ingest
neorag research "daily gold price india" --pages 5
neorag chat --direct                                    # now answers from your corpus
```

Everything lands in `data/corpus/web/<topic>/` and is embedded on the spot.

---

## Fine-tuning loop: weak → powerful

The core promise: upgrade a small open model into a retrieval-grounded specialist with your own data — for free.

| Step | Command | What it does |
|---|---|---|
| 1. Generate data | `neorag build-dataset` | Synthetic QA pairs + **agent-trajectory examples** (the model learns the exact JSON tool protocol it must follow at inference) from your corpus |
| 2. Train | `neorag train` | LoRA/QLoRA SFT with prompt-token masking (chat-template aware). Local CUDA — or the bundled free Colab notebook on a T4 |
| 3. Merge | `neorag merge` | Bake the adapter into a standalone model |
| 4. Deploy | point config at it | `llm.model_id: checkpoints/neorag-sft-merged` |

```bash
neorag build-dataset --per-chunk 2
neorag train                    # or: notebooks/colab_training.ipynb (free T4)
neorag merge
neorag evaluate --dataset data/finetune/sft.jsonl --limit 25 --direct
```

Evaluation metrics: token-F1, answer rate, citation precision, gave-up rate.

---

## Any LLM, one config

| You want | `llm.backend` | Settings |
|---|---|---|
| Local HuggingFace model *(default, free)* | `local` | `model_id: Qwen/Qwen2.5-0.5B-Instruct` |
| **Your fine-tuned model** | `local` | `model_id: checkpoints/neorag-sft-merged` |
| Groq free tier *(default when `GROQ_API_KEY` set)* | `api` | base `https://api.groq.com/openai/v1` — live models: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` |
| OpenRouter free models | `api` | base `https://openrouter.ai/api/v1`, model `...:free` |
| Ollama (local, free) | `api` | base `http://localhost:11434/v1` |
| LM Studio / vLLM / any OpenAI-compatible server | `api` | same pattern |
| Paid clouds (OpenAI, Gemini, ...) | `api` | same pattern — optional, never required |

No config file at all? Set three env vars and go:

```bash
export NEORAG_LLM_BACKEND=api
export NEORAG_LLM_API_BASE=https://openrouter.ai/api/v1
export NEORAG_LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free
```

Full key reference: [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

---

## Commands

| Command | Purpose |
|---|---|
| `neorag chat [--agent] [--direct] [--steps]` | interactive terminal UI (fast mode by default) |
| `neorag ingest [--clear]` | chunk + embed `data/corpus/` into the local vector store |
| `neorag research "<query>" [--pages N] [--no-ingest]` | free web search + scrape → saved + embedded |
| `neorag build-dataset [--per-chunk N] [--limit N]` | generate the SFT fine-tuning set |
| `neorag train` / `neorag merge` | LoRA fine-tune / bake the adapter |
| `neorag evaluate [--dataset ...] [--limit N]` | F1 / answer rate / citation precision |
| `neorag wipe` | clear the vector store |

**Chat commands:** `/mode` fast↔agent · `/web` toggle web tool · `/steps` show agent trace · `/stats` corpus + model info · `/clear` · `/help` · `/quit`

Supported document types: `txt`, `md`, `pdf`, `docx`, `html`, `csv`, `json` — drop files into `data/corpus/` and run `neorag ingest`.

---

## What's inside

```
neorag/
├── data/          loaders (8 formats), chunker, QA-dataset generator
├── embeddings/    MiniLM mean-pooling embedder (thread-safe, batched)
├── store/         ChromaDB persistent wrapper
├── retrieval/     dense retriever + optional cross-encoder reranker
├── web/           DDG/SearxNG search, scraper with cache + freshness, WebSearcher
├── agent/         ReAct loop, tools, prompts, fast/direct one-call paths
├── llm/           local HF backend + OpenAI-compatible backend (SDK + fallback)
├── training/      data formatting (prompt masking), LoRA/QLoRA trainer, merge
├── eval/          token-F1, answer rate, citation precision
├── ui/            rich terminal UI: banner, live spinner phases, panels
└── cli.py         argparse command surface
```

---

## Documentation

| Doc | Contents |
|---|---|
| [`docs/USAGE.md`](docs/USAGE.md) | every command, chat commands, env vars |
| [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) | every config key explained |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | module map, data flow, design decisions |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | production setup, service install, backups |
| [`docs/SECURITY.md`](docs/SECURITY.md) | API key hygiene, data boundaries |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | known issues and fixes |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | planned work |

---

## Development

```bash
git clone https://github.com/naitik0409/NeoRAG.git
cd NeoRAG
pip install -e ".[dev]"
pytest tests -q        # 36 tests
```

Issues and pull requests welcome: https://github.com/naitik0409/NeoRAG/issues

---

<div align="center">

**Author:** [Naitik](https://github.com/naitik0409) · [GitHub profile](https://github.com/naitik0409) · [Repository](https://github.com/naitik0409/NeoRAG)

Licensed under the [MIT License](LICENSE) — free to use, modify, and distribute.

</div>
