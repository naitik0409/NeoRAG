from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

IGNORE_INDEX = -100


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def messages_to_dataset(records: list[dict]) -> Dataset:
    return Dataset.from_list([{"messages": r["messages"]} for r in records])


def build_sft_features(ds: Dataset, tokenizer, max_seq_len: int = 2048) -> Dataset:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        messages = example["messages"]
        prompt_messages = messages[:-1]
        target_message = messages[-1]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        target_text = target_message["content"]

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
        labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids + [tokenizer.eos_token_id]
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[-max_seq_len:]
            labels = labels[-max_seq_len:]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    return ds.map(tokenize, remove_columns=ds.column_names, desc="Tokenizing SFT features")


class SFTCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_ids, batch_mask, batch_labels = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            labels = list(f["labels"])
            mask = list(f["attention_mask"])
            pad_len = max_len - len(ids)
            batch_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_mask.append(mask + [0] * pad_len)
            batch_labels.append(labels + [IGNORE_INDEX] * pad_len)
        return {
            "input_ids": batch_ids,
            "attention_mask": batch_mask,
            "labels": batch_labels,
        }
