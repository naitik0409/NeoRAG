# Deployment

## Requirements

- Python 3.10+ (tested on 3.12), 4 GB RAM minimum for chat; 8 GB+ recommended
- No GPU required for chat (CPU-only machines run the full fast mode)
- GPU (local CUDA or free Colab T4) only for the `train` step
- Outbound internet for: first model downloads (HuggingFace), web search, scraping, LLM API

## Standard install

```powershell
git clone https://github.com/naitik0409/NeoRAG.git
cd NeoRAG
python -m pip install -e .
copy .env.example .env   # then paste your Groq key into .env
python -m neorag.cli chat
```

Optional extras: `python -m pip install -e ".[train]"` (trl, faiss, bitsandbytes) or `".[dev]"` (pytest, ruff).

## Running as a background service (Windows)

Create `service.ps1`:

```powershell
Set-Location C:\path\to\NeoRAG
$env:PYTHONPATH = "src"
python -m neorag.cli chat
```

Register with Task Scheduler or `nssm install NeoRAGChat <python> -m neorag.cli chat` with the app directory as working directory. For unattended use, prefer the one-call paths:

```powershell
python -m neorag.cli research "daily gold price india"   # cron-able corpus refresh
python -m neorag.cli ingest                              # re-index after corpus changes
```

## Running as a service (Linux)

```bash
[Unit]
Description=NeoRAG research refresh
[Service]
WorkingDirectory=/opt/NeoRAG
Environment=PYTHONPATH=src
ExecStart=/usr/bin/python3 -m neorag.cli research "daily briefing"
```

Schedule with a systemd timer for periodic corpus refreshes.

## Data layout in production

| Path | Contents | Back up? |
|---|---|---|
| `data/corpus/` | your documents + `web/` research pages | yes |
| `storage/chroma/` | vector index | yes (or rebuild via `ingest`) |
| `storage/web_cache/` | scraped page cache (7 days) | no |
| `data/finetune/sft.jsonl` | fine-tuning dataset | yes |
| `checkpoints/` | LoRA adapters and merged models | yes |
| `.env` | keys | never commit; store in your secret manager |

## Capacity notes

- ChromaDB handles hundreds of thousands of chunks on a laptop; re-run `ingest --clear` after large corpus changes.
- Web cache grows with usage; it is safe to delete `storage/web_cache/` anytime.
- Free-tier LLM rate limits are the main throughput ceiling; `openai/gpt-oss-20b` is the faster fallback model.

## Health checks

```powershell
python -m neorag.cli chat  # banner shows model, corpus chunks, tools
```

- `Corpus chunks 0` -> run `ingest`.
- Ask a time-sensitive question and verify the cited source dates.
- `python -m pytest tests -q` for the 36-test suite.
