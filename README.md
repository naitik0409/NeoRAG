# NeoRAG

A self-hosted, **100% free** RAG agent you fine-tune from a weak open-source model into a powerful retrieval-grounded specialist.

**Developer:** [Naitik](https://github.com/naitik0409) · License: MIT

## Why NeoRAG

1. **Information gathering is always free.** Web search (DuckDuckGo scraping, SearxNG fallback) + page scraping + local embeddings + local vector store — no API keys, no credits, no rate-limit death.
2. **Realtime-aware.** Time-sensitive questions (prices, news, "today") automatically bypass the page cache; general questions stay cached for speed.
3. **The LLM slot is model-agnostic.** Any model in the world fits — local HuggingFace checkpoints or ANY OpenAI-compatible endpoint (Groq free tier, OpenRouter free models, Ollama, LM Studio, vLLM, or paid clouds).
4. **Fine-tune the weak agent into a strong one.** The SFT dataset teaches grounded cited answering AND the agent JSON protocol, so your small local model behaves like a disciplined RAG specialist.

## Architecture

```
data/corpus/* ──ingest──► chunker ──embed──► ChromaDB ──┐
                                                        ├─► Retriever ─┐
web: DuckDuckGo + scraper (freshness-aware cache) ──────┘              ├─► FAST MODE: 1 LLM call ──► cited answer
                                                                       ├─► AGENT MODE: ReAct JSON loop
LLM: local HF / any OpenAI-compatible API ─────────────────────────────┘
                ▲
                └── fine-tune loop: build-dataset → train (LoRA) → merge
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Quickstart

```powershell
python -m pip install -e .

# Put your free Groq key in .env:  GROQ_API_KEY=gsk_...

python -m neorag.cli chat              # fast hybrid mode (default)
python -m neorag.cli chat --agent      # step-by-step agent mode
python -m neorag.cli chat --direct     # docs-only, no web

# your own free Tavily - research the web into your corpus
python -m neorag.cli research "how do transformers work"

# add your own documents (pdf/txt/md/docx/html/csv/json)
#   drop files into data/corpus/ then:
python -m neorag.cli ingest
```

## Fine-tuning loop (weak → powerful)

```powershell
python -m neorag.cli build-dataset     # synthetic QA + agent trajectories (free, local)
python -m neorag.cli train             # LoRA/QLoRA SFT - local CUDA or free Colab (notebooks/colab_training.ipynb)
python -m neorag.cli merge             # merge adapter into a standalone model
# then in config/neorag.yaml:  llm.model_id: checkpoints/neorag-sft-merged
```

## Any LLM, one config

| You want | backend | settings |
|---|---|---|
| Local HuggingFace model (free) | `local` | `model_id: Qwen/Qwen2.5-0.5B-Instruct` |
| Your fine-tuned model | `local` | `model_id: checkpoints/neorag-sft-merged` |
| Groq free tier | `api` | base `https://api.groq.com/openai/v1`, models verified live: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` |
| OpenRouter free models | `api` | base `https://openrouter.ai/api/v1`, model `...:free` |
| Ollama (local, free) | `api` | base `http://localhost:11434/v1` |
| LM Studio / vLLM / any OpenAI-compatible | `api` | same pattern |
| Paid clouds (OpenAI, Gemini, ...) | `api` | same pattern — optional, never required |

## Documentation

| Doc | Contents |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | every command, chat commands, env vars |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | every config key explained |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | module map, data flow, design decisions |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | production setup, service install, backups |
| [docs/SECURITY.md](docs/SECURITY.md) | API key hygiene, data boundaries |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | known issues and fixes |
| [docs/ROADMAP.md](docs/ROADMAP.md) | planned work |
| [CHANGELOG.md](CHANGELOG.md) | release history |

## Tests

```powershell
$env:PYTHONPATH="src"; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"; python -m pytest tests -q
```

## License

MIT — see [LICENSE](LICENSE).
