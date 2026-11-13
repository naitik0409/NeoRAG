from __future__ import annotations

import torch
from torch import nn

from neorag.logging_utils import logger
from neorag.training.data_format import SFTCollator, build_sft_features, load_jsonl, messages_to_dataset


def find_target_modules(model: nn.Module) -> list[str]:
    suffixes: dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last = name.split(".")[-1]
            if any(bad in last.lower() for bad in ("head", "embed", "output")):
                continue
            suffixes[last] = suffixes.get(last, 0) + 1
    targets = [s for s, c in sorted(suffixes.items(), key=lambda kv: -kv[1]) if c >= 4]
    if not targets:
        targets = list(suffixes.keys())
    logger.info(f"LoRA target modules: {targets}")
    return targets


def _bnb_config(dtype: torch.dtype):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_base_model(cfg, for_training: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    quantize = bool(cfg.quantize_4bit) and cuda
    dtype = torch.bfloat16 if (cuda and torch.cuda.is_bf16_supported()) else (
        torch.float16 if cuda else torch.float32
    )

    kwargs = {"dtype": dtype} if quantize or not cuda else {"dtype": dtype}
    try:
        if quantize:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model_id, quantization_config=_bnb_config(dtype), device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, dtype=dtype)
            model.to("cuda" if cuda else "cpu")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, torch_dtype=dtype)
        model.to("cuda" if cuda else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if for_training:
        model.config.use_cache = False
        if quantize:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    return model, tokenizer


def _training_args(cfg) -> "object":
    from transformers import TrainingArguments

    cuda = torch.cuda.is_available()
    optim = "paged_adamw_8bit" if (bool(cfg.quantize_4bit) and cuda) else "adamw_torch"
    return TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps",
        eval_steps=cfg.save_steps,
        bf16=cuda and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=optim,
        seed=cfg.seed,
        report_to=[],
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )


def train_sft(cfg) -> str:
    from peft import LoraConfig, TaskType, get_peft_model

    from neorag.config import TrainingConfig

    assert isinstance(cfg, TrainingConfig)
    model, tokenizer = load_base_model(cfg, for_training=True)

    if cfg.lora_target_modules == "auto":
        target_modules = find_target_modules(model)
    else:
        target_modules = [m.strip() for m in str(cfg.lora_target_modules).split(",") if m.strip()]

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    records = load_jsonl(cfg.dataset_path)
    if not records:
        raise FileNotFoundError(f"No training records found at {cfg.dataset_path}")
    ds = messages_to_dataset(records)
    features = build_sft_features(ds, tokenizer, max_seq_len=cfg.max_seq_len)
    split = features.train_test_split(test_size=min(cfg.eval_split, 0.2), seed=cfg.seed) if cfg.eval_split > 0 else {"train": features, "test": None}
    train_ds, eval_ds = split["train"], split.get("test")

    args = _training_args(cfg)
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=SFTCollator(tokenizer),
    )
    trainer.train()

    adapter_dir = cfg.output_dir if str(cfg.output_dir).endswith("adapter") else str(cfg.output_dir) + "-adapter"
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info(f"LoRA adapter saved to {adapter_dir}")
    return adapter_dir


def merge_adapter(cfg, adapter_dir: str | None = None, merged_dir: str | None = None) -> str:
    from peft import PeftModel

    adapter_dir = adapter_dir or str(cfg.output_dir) + "-adapter"
    merged_dir = merged_dir or str(cfg.output_dir) + "-merged"

    load_cfg = type(cfg)(**{**{f: getattr(cfg, f) for f in cfg.__dataclass_fields__}, "quantize_4bit": False})
    model, tokenizer = load_base_model(load_cfg, for_training=False)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    logger.info(f"Merged model saved to {merged_dir}")
    return merged_dir
