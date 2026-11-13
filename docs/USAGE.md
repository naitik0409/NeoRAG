# Usage

## Two ways to run

**Installed from PyPI** (`pip install neorag`):

```bash
neorag <command> [options]
```

**From a source checkout:**

```bash
python -m neorag.cli <command> [options]
```

Both are identical. Global option: `--config <path>` to use a different YAML config.

## Zero-config quickstart

No config file is needed. NeoRAG looks for `config/neorag.yaml` in the current directory (or `~/.neorag/config.yaml`); if neither exists it runs on built-in defaults, and if `GROQ_API_KEY` is set it automatically uses the Groq free tier.

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | set this alone → Groq + `openai/gpt-oss-120b`, zero config |
| `NEORAG_LLM_BACKEND` | `api` or `local` (force local even with a Groq key present) |
| `NEORAG_LLM_API_BASE` | override the OpenAI-compatible endpoint |
| `NEORAG_LLM_MODEL` | override the model name sent to the endpoint |
| `NEORAG_LLM_API_KEY` | generic key name for non-Groq providers |
| `HF_TOKEN` | only for gated HuggingFace model downloads |

Data (`data/corpus/`, `storage/chroma/`, `storage/web_cache/`) is created relative to the directory you run from — run from your project folder to keep things tidy.

## Commands

| Command | Purpose |
|---|---|
| `chat` | interactive terminal UI (default: fast hybrid mode) |
| `ingest` | chunk + embed `data/corpus/` into the local vector store |
| `research "<query>"` | free web search + scrape -> saves `data/corpus/web/<slug>/` and ingests |
| `build-dataset` | generate the SFT fine-tuning set from your corpus |
| `train` | LoRA/QLoRA fine-tune (GPU required - local CUDA or free Colab) |
| `merge` | merge the LoRA adapter into a standalone model |
| `evaluate` | token-F1 / answer rate / citation precision on generated QA |
| `wipe` | clear the vector store |

### chat options

| Flag | Effect |
|---|---|
| `--agent` | start in agent mode (step-by-step ReAct loop) |
| `--direct` | docs-only fast mode (web disabled) |
| `--steps` | show the agent step trace under answers |
| `--backend {local,api}` | override `llm.backend` |
| `--model <name>` | override model (local `model_id` or api `api_model`) |
| `--api-base <url>` | override `llm.api_base` |

### build-dataset options

| Flag | Effect |
|---|---|
| `--per-chunk N` | QA pairs per chunk (default 1) |
| `--limit N` | cap total chunks processed |
| `--no-trajectories` | skip agent-trajectory examples |
| `--template-fallback` | add summarization examples when the LLM produces nothing |
| `--out <path>` | output JSONL path |

### evaluate options

| Flag | Effect |
|---|---|
| `--dataset <path>` | SFT JSONL with `meta.question` / `meta.answer` (kind `direct_qa`) |
| `--limit N` | evaluate first N questions |
| `--direct` | evaluate the one-shot path instead of the agent |
| `--out <path>` | write summary JSON |

## Chat commands

| Command | Effect |
|---|---|
| `/mode` | toggle fast (one call) / agent (ReAct loop) |
| `/web` | toggle the `web_search` tool |
| `/steps` | toggle agent step trace |
| `/stats` | corpus size, model, tools, mode |
| `/clear` | clear screen |
| `/help` | command list |
| `/quit` | exit |

## Environment variables

Loaded from `.env` at the project root (never committed).

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | key for the current `api` backend (Groq free tier) |
| `NEORAG_LLM_API_BASE` | override the OpenAI-compatible endpoint |
| `NEORAG_LLM_MODEL` | override the model name sent to the endpoint |
| `NEORAG_LLM_API_KEY` | generic key name for non-Groq providers |
| `HF_TOKEN` | only for gated HuggingFace model downloads |

Only the LLM needs a key. Search, scraping, embeddings, and the vector store are key-free.

## Typical flows

**Ask anything (web + docs):** `python -m neorag.cli chat`

**Build a private knowledge base:** drop files in `data/corpus/` -> `ingest` -> `chat --direct`.

**Build a research corpus:** `research "topic a"` (repeat for topics) -> `chat --direct` searches only your corpus.

**Fine-tune:** `build-dataset` -> `train` (Colab if no GPU) -> `merge` -> point `llm.model_id` at the merged dir.
