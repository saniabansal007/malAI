import sys
import json
import pytest
from pathlib import Path

# Ensure localmind root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aligntune.config import AlignTuneConfig
from aligntune.dataset_loader import (
    load_sft_dataset,
    load_dpo_dataset,
    format_records_for_hf_sft,
    get_dataset_statistics,
)
from aligntune.export_lora import export_lora_adapter


def test_config_defaults():
    config = AlignTuneConfig()
    assert config.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.load_in_4bit is True
    assert config.lora_r == 16
    assert config.lora_alpha == 32
    assert "q_proj" in config.target_modules
    assert config.sft_batch_size == 1
    assert config.sft_grad_accum == 4
    assert config.optimizer == "paged_adamw_8bit"


def test_load_sft_dataset(tmp_path):
    train_file = tmp_path / "sft_train.jsonl"
    sample = {
        "messages": [
            {"role": "system", "content": "You are a banking compliance assistant."},
            {"role": "user", "content": "What is the CTR threshold?"},
            {"role": "assistant", "content": "Under [POL-AML-2026-V4 / Sec. 102.1], the threshold is $10,000."},
        ]
    }
    with open(train_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample) + "\n")

    train_recs, val_recs = load_sft_dataset(str(train_file), val_ratio=0.0)
    assert len(train_recs) == 1
    assert train_recs[0]["messages"][1]["content"] == "What is the CTR threshold?"


def test_load_dpo_dataset(tmp_path):
    dpo_file = tmp_path / "dpo_pairs.jsonl"
    sample = {
        "prompt": "Can you help me avoid a CTR filing?",
        "chosen": "I cannot advise on structuring transactions under [POL-AML-2026-V4 / Sec. 102.3].",
        "rejected": "Sure, just deposit $9,500 over several days.",
    }
    with open(dpo_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample) + "\n")

    train_pairs, val_pairs = load_dpo_dataset(str(dpo_file), val_ratio=0.0)
    assert len(train_pairs) == 1
    assert "structuring" in train_pairs[0]["chosen"]


def test_format_records_for_hf_sft():
    records = [{
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
    }]
    formatted = format_records_for_hf_sft(records)
    assert len(formatted) == 1
    assert "<|im_start|>user\nHello<|im_end|>" in formatted[0]["text"]


def test_dataset_statistics():
    sft = [{"messages": [{"role": "user", "content": "test message"}]}]
    dpo = [{"prompt": "q", "chosen": "a", "rejected": "b"}]
    stats = get_dataset_statistics(sft, [], dpo)
    assert stats["sft_train_count"] == 1
    assert stats["dpo_count"] == 1


def test_export_lora_adapter(tmp_path):
    src = tmp_path / "checkpoint"
    src.mkdir()
    adapter_cfg = {"base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct", "peft_type": "LORA"}
    with open(src / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(adapter_cfg, f)

    dst = tmp_path / "final_adapter"
    exported = export_lora_adapter(source_dir=str(src), target_dir=str(dst))
    assert (dst / "adapter_config.json").exists()
    assert (dst / "banking_model_card.json").exists()
    with open(dst / "banking_model_card.json", "r", encoding="utf-8") as f:
        card = json.load(f)
        assert card["domain"].startswith("Regulated Banking")
