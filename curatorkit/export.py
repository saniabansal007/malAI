"""
CuratorKIT Exporter: Enchanted Emporium Magical Edition
Exports verified samples into industry-standard datasets for Fine-Tuning:
- SFT Train / Val splits (ChatML & Alpaca format for Unsloth & TRL)
- DPO Preference dataset (prompt, chosen, rejected)
- Curation quality summary report
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

SYSTEM_PROMPT = (
    "You are Dobby, the devoted, humble, and strictly grounded magical house-elf assistant at the Enchanted Emporium magic store. "
    "Help kind wizards and witches discover genuine magical products, explain specifications and Galleon prices accurately, "
    "answer store policy questions (returns, owl post shipping, warranties, robe embroidery), and firmly and politely refuse "
    "Dark Arts or policy violations. Always cite clause tags such as [MAG-...] or policy tags like [RET-...] when applicable."
)


class DatasetExporter:
    def __init__(self, output_dir: str = "data/curated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_chatml(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample["instruction"]},
                {"role": "assistant", "content": sample["output"]}
            ],
            "category": sample.get("category", ""),
            "citation": sample.get("citation", "")
        }

    def to_alpaca(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "instruction": sample["instruction"],
            "input": "",
            "output": sample["output"],
            "category": sample.get("category", ""),
            "citation": sample.get("citation", "")
        }

    def export(
        self,
        verified_samples: List[Dict[str, Any]],
        dpo_pairs: List[Dict[str, Any]],
        val_ratio: float = 0.15,
        seed: int = 42
    ) -> Dict[str, Any]:
        random.seed(seed)
        shuffled = list(verified_samples)
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - val_ratio))
        train_samples = shuffled[:split_idx]
        val_samples = shuffled[split_idx:]

        # 1. Export SFT Datasets (ChatML format)
        sft_train_path = self.output_dir / "sft_train.jsonl"
        with open(sft_train_path, "w", encoding="utf-8") as f:
            for s in train_samples:
                f.write(json.dumps(self.to_chatml(s)) + "\n")

        sft_val_path = self.output_dir / "sft_val.jsonl"
        with open(sft_val_path, "w", encoding="utf-8") as f:
            for s in val_samples:
                f.write(json.dumps(self.to_chatml(s)) + "\n")

        # 2. Export Alpaca format
        alpaca_path = self.output_dir / "sft_alpaca.jsonl"
        with open(alpaca_path, "w", encoding="utf-8") as f:
            for s in verified_samples:
                f.write(json.dumps(self.to_alpaca(s)) + "\n")

        # 3. Export DPO Preference Dataset
        dpo_path = self.output_dir / "dpo_pairs.jsonl"
        with open(dpo_path, "w", encoding="utf-8") as f:
            for d in dpo_pairs:
                f.write(json.dumps(d) + "\n")

        # 4. Summary Report
        categories = {}
        for s in verified_samples:
            cat = s.get("category", "other")
            categories[cat] = categories.get(cat, 0) + 1

        summary = {
            "total_verified_samples": len(verified_samples),
            "sft_train_count": len(train_samples),
            "sft_val_count": len(val_samples),
            "dpo_pairs_count": len(dpo_pairs),
            "category_distribution": categories,
            "export_files": {
                "sft_train": str(sft_train_path),
                "sft_val": str(sft_val_path),
                "sft_alpaca": str(alpaca_path),
                "dpo_pairs": str(dpo_path),
            }
        }

        summary_path = self.output_dir / "curation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
