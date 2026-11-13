# Configuration

All settings live in `config/neorag.yaml`. Relative paths resolve against the project root.

## llm

| Key | Default | Meaning |
|---|---|---|
| `backend` | `api` | `local` (HuggingFace transformers) or `api` (any OpenAI-compatible endpoint) |
| `model_id` | `Qwen/Qwen2.5-0.5B-Instruct` | model for `backend: local`; also the fallback name if `api_model` unset. Point it at `checkpoints/neorag-sft-merged` after fine-tuning |
| `api_base` | `https://api.groq.com/openai/v1` | endpoint base URL; overridable via `NEORAG_LLM_API_BASE` |
| `api_model` | `openai/gpt-oss-120b` | model name sent to the endpoint; overridable via `NEORAG_LLM_MODEL` |
| `api_key_env` | `GROQ_API_KEY` | env var that holds the key |
| `reasoning_effort` | `low` | sent for `gpt-oss` models; keeps agent JSON reliable and fast |
| `max_new_tokens` | `2048` | generation budget per call (covers reasoning tokens) |
| `temperature` / `top_p` / `repetition_penalty` | `0.7` / `0.9` / `1.05` | sampling |
| `dtype` / `device` / `quantize_4bit` | `auto` | resolved at load: bf16 if CUDA supports, float32 on CPU; 4-bit only on CUDA |

## embeddings

| Key | Default | Meaning |
|---|---|---|
| `model_id` | `sentence-transformers/all-MiniLM-L6-v2` | free, local, CPU-friendly; upgrade to `BAAI/bge-small-en-v1.5` for quality |
| `batch_size` | `32` | embedding batch |
| `normalize` | `true` | L2-normalize vectors (cosine via dot product) |
| `max_length` | `512` | tokenizer truncation |
| `device` | `auto` | `cuda` when available |

## store

| Key | Default |
|---|---|
| `persist_dir` | `storage/chroma` |
| `collection` | `neorag` |

## ingest

| Key | Default | Meaning |
|---|---|---|
| `source_dir` | `data/corpus` | corpus root (includes `web/` subfolder from `research`) |
| `chunk_size` | `512` | words per chunk |
| `chunk_overlap` | `64` | word overlap between chunks |
| `file_types` | txt, md, pdf, docx, html, htm, csv, json | loaders available |

## retrieval

| Key | Default | Meaning |
|---|---|---|
| `top_k` | `8` | candidates fetched from the store |
| `final_k` | `4` | passages kept after (optional) reranking |
| `reranker` | `none` | `cross-encoder` enables `cross-encoder/ms-marco-MiniLM-L-6-v2` (free, local) |
| `min_score` | `0.0` | drop hits below this cosine score |

## agent

| Key | Default | Meaning |
|---|---|---|
| `max_steps` | `5` | agent-mode tool-call round trips before forced final answer |
| `max_context_chars` | `12000` | context cap for the merged prompt (fast mode) |

## web

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | builds the web searcher and `web_search` tool |
| `max_results` | `8` | search results fetched per query |
| `scrape_pages` | `3` | pages scraped per query (concurrent) |
| `top_k` | `6` | web passages kept |
| `chunk_size` | `220` | words per web chunk |
| `max_chars_per_page` | `8000` | text cap per page |
| `timeout` | `20` | per-request seconds |
| `delay` | `1.0` | politeness delay between sequential fetches (parallel fetches bypass) |
| `cache_dir` | `storage/web_cache` | 7-day page cache |
| `freshness` | `auto` | `auto`: bypass cache for time-sensitive queries; `always`: never cache; `cached`: always cache |
| `searx_instances` | `[]` | extra SearxNG JSON endpoints |

## training

| Key | Default | Meaning |
|---|---|---|
| `base_model_id` | `Qwen/Qwen2.5-0.5B-Instruct` | the weak model to upgrade |
| `dataset_path` | `data/finetune/sft.jsonl` | SFT JSONL from `build-dataset` |
| `output_dir` | `checkpoints/neorag-sft` | adapter output (`-adapter`, `-merged` suffixes) |
| `lora_r` / `lora_alpha` / `lora_dropout` | `16` / `32` / `0.05` | LoRA capacity |
| `lora_target_modules` | `auto` | auto-discovers all Linear suffixes (q/k/v/o/gate/up/down proj...) |
| `quantize_4bit` | `auto` | QLoRA on CUDA only |
| `epochs` / `learning_rate` | `3` / `2e-4` | schedule: cosine + warmup ratio 0.03 |
| `per_device_batch_size` / `gradient_accumulation_steps` | `1` / `8` | effective batch 8 |
| `max_seq_len` | `2048` | truncation window |
| `eval_split` | `0.02` | validation fraction |
