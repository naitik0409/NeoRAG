import json

from neorag.training.data_format import SFTCollator, IGNORE_INDEX, load_jsonl, messages_to_dataset


class FakeTokenizer:
    pad_token = "[PAD]"
    pad_token_id = 0
    eos_token_id = 2


def test_load_jsonl_roundtrip(tmp_path):
    path = tmp_path / "sft.jsonl"
    records = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}]
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    loaded = load_jsonl(path)
    assert loaded == records


def test_collator_pads_and_masks():
    collator = SFTCollator(FakeTokenizer())
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [4, 5]},
    ]
    batch = collator(features)
    assert len(batch["input_ids"][0]) == 3
    assert batch["input_ids"][1] == [4, 5, 0]
    assert batch["attention_mask"][1] == [1, 1, 0]
    assert batch["labels"][1] == [4, 5, IGNORE_INDEX]


def test_messages_to_dataset():
    records = [{"messages": [{"role": "user", "content": "q"}]}]
    ds = messages_to_dataset(records)
    assert ds[0]["messages"] == records[0]["messages"]
