from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _auto_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _auto_dtype(device: str) -> str:
    if device == "cpu":
        return "float32"
    try:
        import torch

        if torch.cuda.is_bf16_supported():
            return "bfloat16"
        return "float16"
    except Exception:
        return "float16"


@dataclass
class LLMConfig:
    backend: str = "local"
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "auto"
    quantize_4bit: str = "auto"
    device: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    api_base: str | None = None
    api_key_env: str = "NEORAG_LLM_API_KEY"
    api_model: str | None = None
    reasoning_effort: str = "low"


@dataclass
class EmbeddingConfig:
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    batch_size: int = 32
    normalize: bool = True
    max_length: int = 512


@dataclass
class StoreConfig:
    provider: str = "chroma"
    persist_dir: str = "storage/chroma"
    collection: str = "neorag"


@dataclass
class IngestConfig:
    source_dir: str = "data/corpus"
    chunk_size: int = 512
    chunk_overlap: int = 64
    file_types: list[str] = field(
        default_factory=lambda: ["txt", "md", "pdf", "docx", "html", "htm", "csv", "json"]
    )


@dataclass
class RetrievalConfig:
    top_k: int = 8
    final_k: int = 4
    reranker: str = "none"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    min_score: float = 0.0


@dataclass
class AgentConfig:
    max_steps: int = 5
    max_context_chars: int = 12000


@dataclass
class WebConfig:
    enabled: bool = True
    max_results: int = 8
    scrape_pages: int = 3
    max_chars_per_page: int = 8000
    top_k: int = 6
    chunk_size: int = 220
    timeout: int = 20
    delay: float = 1.0
    cache_dir: str = "storage/web_cache"
    freshness: str = "auto"
    searx_instances: list[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    base_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "checkpoints/neorag-sft"
    dataset_path: str = "data/finetune/sft.jsonl"
    eval_split: float = 0.02
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "auto"
    quantize_4bit: str = "auto"
    epochs: float = 3.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_seq_len: int = 2048
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    bf16: str = "auto"
    seed: int = 42


@dataclass
class NeoRAGConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    web: WebConfig = field(default_factory=WebConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    root: Path = PROJECT_ROOT

    def resolve(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.root / p


def _resolve_auto(section: dict[str, Any], device_key: str, dtype_key: str, quant_key: str) -> None:
    if section.get(device_key, "auto") == "auto":
        section[device_key] = _auto_device()
    device = section[device_key]
    if section.get(dtype_key, "auto") == "auto":
        section[dtype_key] = _auto_dtype(device)
    if section.get(quant_key, "auto") == "auto":
        section[quant_key] = device == "cuda"


def _merge(dataclass_obj: Any, overrides: dict[str, Any]) -> Any:
    fields = {f: getattr(dataclass_obj, f) for f in dataclass_obj.__dataclass_fields__}
    for key, value in overrides.items():
        if key not in fields:
            raise ValueError(f"Unknown config key: {key}")
        fields[key] = value
    return type(dataclass_obj)(**fields)


def load_config(path: str | Path | None = None) -> NeoRAGConfig:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config" / "neorag.yaml"
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    cfg = NeoRAGConfig(root=cfg_path.resolve().parents[1].parent if cfg_path.exists() else PROJECT_ROOT)

    sections = {
        "llm": LLMConfig,
        "embeddings": EmbeddingConfig,
        "store": StoreConfig,
        "ingest": IngestConfig,
        "retrieval": RetrievalConfig,
        "agent": AgentConfig,
        "web": WebConfig,
        "training": TrainingConfig,
    }
    for name, cls in sections.items():
        section_raw = dict(raw.get(name) or {})
        section = _merge(getattr(cfg, name), section_raw)
        if name == "llm":
            d = asdict(section)
            _resolve_auto(d, "device", "dtype", "quantize_4bit")
            d["api_base"] = os.environ.get("NEORAG_LLM_API_BASE", d["api_base"])
            if os.environ.get("NEORAG_LLM_MODEL"):
                d["api_model"] = os.environ["NEORAG_LLM_MODEL"]
            section = LLMConfig(**d)
        if name == "training":
            d = asdict(section)
            if d["bf16"] == "auto":
                d["bf16"] = "true" if _auto_device() == "cuda" else "false"
            section = TrainingConfig(**d)
        setattr(cfg, name, section)

    cfg.embeddings.device = _auto_device() if cfg.embeddings.device == "auto" else cfg.embeddings.device
    return cfg
