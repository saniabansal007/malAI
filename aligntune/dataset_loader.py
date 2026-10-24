"""Dataset Loader and Formatter for AlignTune (SFT & DPO).

Loads curated JSONL files produced by CuratorKIT and prepares them for
Hugging Face TRL (SFTTrainer & DPOTrainer) and Unsloth.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {err}")
    return records


def load_sft_dataset(
    train_path: str,
    val_path: Optional[str] = None,
    val_ratio: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load SFT dataset in ChatML format.

    Each record contains a 'messages' list of role/content dicts.
    Returns:
        (train_records, val_records)
    """
    train_records = _read_jsonl(train_path)

    # Validate schema
    for idx, rec in enumerate(train_records):
        if "messages" not in rec or not isinstance(rec["messages"], list):
            raise ValueError(f"SFT record #{idx} missing 'messages' list")
        roles = [m.get("role") for m in rec["messages"]]
        if "user" not in roles or "assistant" not in roles:
            raise ValueError(f"SFT record #{idx} missing user or assistant role")

    if val_path and Path(val_path).exists():
        val_records = _read_jsonl(val_path)
    else:
        # Split train_records if no separate val file
        split_idx = max(1, int(len(train_records) * (1.0 - val_ratio)))
        val_records = train_records[split_idx:]
        train_records = train_records[:split_idx]

    return train_records, val_records


def load_dpo_dataset(
    dpo_path: str,
    val_ratio: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load DPO preference pairs (prompt, chosen, rejected).

    Returns:
        (train_pairs, val_pairs)
    """
    records = _read_jsonl(dpo_path)

    # Validate DPO schema
    for idx, rec in enumerate(records):
        for req_field in ["prompt", "chosen", "rejected"]:
            if req_field not in rec or not rec[req_field]:
                raise ValueError(f"DPO record #{idx} missing or empty field: {req_field}")

    split_idx = max(1, int(len(records) * (1.0 - val_ratio)))
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    return train_records, val_records


def format_records_for_hf_sft(records: List[Dict[str, Any]], tokenizer: Any = None) -> List[Dict[str, str]]:
    """Format records into text or messages for TRL SFTTrainer."""
    formatted = []
    for rec in records:
        msgs = rec["messages"]
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            formatted.append({"text": text})
        else:
            # Fallback ChatML plaintext rendering
            parts = []
            for m in msgs:
                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
            formatted.append({"text": "\n".join(parts)})
    return formatted


def get_dataset_statistics(
    sft_train: List[Dict[str, Any]],
    sft_val: List[Dict[str, Any]],
    dpo_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate token count estimations and dataset breakdown."""
    def est_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    sft_lens = []
    for rec in sft_train + sft_val:
        total = sum(est_tokens(m.get("content", "")) for m in rec.get("messages", []))
        sft_lens.append(total)

    dpo_lens = []
    for rec in dpo_records:
        total = est_tokens(rec.get("prompt", "")) + est_tokens(rec.get("chosen", ""))
        dpo_lens.append(total)

    avg_sft = sum(sft_lens) / len(sft_lens) if sft_lens else 0
    avg_dpo = sum(dpo_lens) / len(dpo_lens) if dpo_lens else 0

    return {
        "sft_train_count": len(sft_train),
        "sft_val_count": len(sft_val),
        "dpo_count": len(dpo_records),
        "estimated_avg_sft_tokens": round(avg_sft, 1),
        "estimated_avg_dpo_tokens": round(avg_dpo, 1),
    }
