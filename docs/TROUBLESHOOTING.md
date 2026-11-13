# Troubleshooting

Known issues and their resolutions - all verified on this project.

## LLM / API

**`LLM API error 403: error code: 1010`**
Cloudflare blocked raw `urllib` requests to Groq. Fixed in 0.1.0 (official OpenAI SDK with browser-UA fallback). If you still see it, update `src/neorag/llm/openai_compat.py` is present - you are on an old checkout.

**`The model <name> does not exist or you do not have access` (404)**
Groq rotates free models. List live models:
```powershell
python -c "from openai import OpenAI; import os; from neorag.env import load_env_file; load_env_file('.env'); c=OpenAI(base_url='https://api.groq.com/openai/v1', api_key=os.environ['GROQ_API_KEY']); print([m.id for m in c.models.list().data])"
```
Set a valid one in `config/neorag.yaml -> llm.api_model`.

**Empty answers / `(empty answer)` / wasted steps**
`gpt-oss` reasoning models can spend the whole budget on hidden reasoning. Fixes in place: `reasoning_effort: low`, reasoning-text fallback, empty-answer fast-fail, forced final answer at the step limit. If you change models, keep `max_new_tokens >= 1024`.

**`Missing API key` / `401 Invalid API Key`**
Paste the key into `.env`: `GROQ_API_KEY=gsk_...`. Get a free key at console.groq.com.

**`429` / rate limited**
Free-tier limits. Wait, or switch `api_model` to `openai/gpt-oss-20b`.

## Chat UI

**Unicode garbling (`You ⯛`, `�`) or `UnicodeEncodeError: 'charmap'`**
Fixed in 0.1.0: stdout/stderr are reconfigured to UTF-8 in the CLI. Use Windows Terminal for best rendering.

**No logs anymore?**
Intentional - the chat UI silences `neorag` logging to ERROR so answers stay clean. Run any other command (e.g. `ingest`) for full logs.

## Web research

**Stale prices / old news**
Cache is freshness-aware: time-sensitive queries bypass it. Force never-cache with `web.freshness: always`, or clear `storage/web_cache/`.

**No web results found**
DDG HTML sometimes rate-limits aggressive scraping; the code falls back to the `lite` endpoint, then SearxNG instances if configured. Increase `web.delay`, lower `scrape_pages`, or add public SearxNG instances in `web.searx_instances`.

**A page failed to scrape**
Some sites block bots or render via JavaScript. NeoRAG falls back to the search snippet for that page. Add the content as a file in `data/corpus/` instead if it matters.

## Local ML

**`ImportError: cannot import name 'AutoModel'`**
A thread race in transformers lazy imports when two threads load models simultaneously. Fixed in 0.1.0 with load locks in the embedder and reranker.

**HF symlink warning on Windows**
Harmless (cache uses a slower path). Silence with `$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"` or enable Windows Developer Mode.

**First question is slow**
One-time model download/load (embedder ~90 MB) and warmup. Later questions in the same session are fast; keep the session open.

## Ingestion / store

**`Corpus chunks 0` in the banner**
Run `python -m neorag.cli ingest` after adding files to `data/corpus/`.

**Old documents still appear after deleting files**
The store persists. Run `python -m neorag.cli ingest --clear` to rebuild from the current corpus, or `wipe` for a full reset.

## Training

**CPU-only machine and `train` is unusable**
Expected - training needs a GPU. Use the free Colab notebook: `notebooks/colab_training.ipynb` (T4, free tier).

**`bitsandbytes` / 4-bit errors on CPU**
4-bit quantization is CUDA-only; the code falls back to unquantized weights automatically.
