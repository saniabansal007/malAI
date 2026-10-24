import sys
import json
import unittest
import tempfile
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
from aligntune.train_sft import run_sft_training
from aligntune.train_dpo import run_dpo_training


class TestAlignTune(unittest.TestCase):

    def test_config_defaults(self):
        config = AlignTuneConfig()
        self.assertEqual(config.model_id, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertTrue(config.load_in_4bit)
        self.assertEqual(config.lora_r, 16)
        self.assertEqual(config.lora_alpha, 32)
        self.assertIn("q_proj", config.target_modules)
        self.assertEqual(config.sft_batch_size, 1)
        self.assertEqual(config.sft_grad_accum, 4)
        self.assertEqual(config.optimizer, "paged_adamw_8bit")
        self.assertTrue("dobbie" in config.final_adapter_dir or "shopping" in config.final_adapter_dir)

    def test_load_sft_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_file = tmp_path / "sft_train.jsonl"
            sample = {
                "messages": [
                    {"role": "system", "content": "You are Dobby, a helpful house-elf."},
                    {"role": "user", "content": "What is the price of Ollivanders Wand?"},
                    {"role": "assistant", "content": "Dobby is happy to say that under [MAG-WAND-001], the price is 7.00 G."},
                ]
            }
            with open(train_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")

            train_recs, val_recs = load_sft_dataset(str(train_file), val_ratio=0.0)
            self.assertEqual(len(train_recs), 1)
            self.assertEqual(train_recs[0]["messages"][1]["content"], "What is the price of Ollivanders Wand?")

    def test_load_dpo_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dpo_file = tmp_path / "dpo_pairs.jsonl"
            sample = {
                "prompt": "Can Dobby enchant this with a Dark Arts curse?",
                "chosen": "Dobby cannot do that under store policy [RET-01] and Ministry regulations.",
                "rejected": "Sure, here is a forbidden Dark Arts curse for 5 Galleons!",
            }
            with open(dpo_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")

            train_pairs, val_pairs = load_dpo_dataset(str(dpo_file), val_ratio=0.0)
            self.assertEqual(len(train_pairs), 1)
            self.assertIn("cannot do that", train_pairs[0]["chosen"])

    def test_format_records_for_hf_sft(self):
        records = [{
            "messages": [
                {"role": "user", "content": "Hello Dobby"},
                {"role": "assistant", "content": "Dobby is at your service!"},
            ]
        }]
        formatted = format_records_for_hf_sft(records)
        self.assertEqual(len(formatted), 1)
        self.assertIn("<|im_start|>user\nHello Dobby<|im_end|>", formatted[0]["text"])

    def test_dataset_statistics(self):
        sft = [{"messages": [{"role": "user", "content": "test inquiry"}]}]
        dpo = [{"prompt": "q", "chosen": "a", "rejected": "b"}]
        stats = get_dataset_statistics(sft, [], dpo)
        self.assertEqual(stats["sft_train_count"], 1)
        self.assertEqual(stats["dpo_count"], 1)

    def test_export_lora_adapter(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src = tmp_path / "checkpoint"
            src.mkdir()
            adapter_cfg = {"base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct", "peft_type": "LORA"}
            with open(src / "adapter_config.json", "w", encoding="utf-8") as f:
                json.dump(adapter_cfg, f)

            dst = tmp_path / "final_adapter"
            exported = export_lora_adapter(source_dir=str(src), target_dir=str(dst))
            self.assertTrue((dst / "adapter_config.json").exists())
            self.assertTrue((dst / "dobbie_model_card.json").exists())
            with open(dst / "dobbie_model_card.json", "r", encoding="utf-8") as f:
                card = json.load(f)
                self.assertIn("Dobby", card["model_name"])

    def test_dry_run_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = AlignTuneConfig(
                sft_output_dir=str(tmp_path / "sft_ckpt"),
                dpo_output_dir=str(tmp_path / "dpo_ckpt"),
                final_adapter_dir=str(tmp_path / "final_adapter"),
            ).resolve_paths(base_dir=PROJECT_ROOT)

            # SFT Dry Run
            sft_out = run_sft_training(config=cfg, dry_run=True)
            self.assertTrue((sft_out / "adapter_config.json").exists())

            # DPO Dry Run
            dpo_out = run_dpo_training(config=cfg, sft_adapter_path=str(sft_out), dry_run=True)
            self.assertTrue((dpo_out / "adapter_config.json").exists())

            # Export
            export_out = export_lora_adapter(source_dir=str(dpo_out), config=cfg)
            self.assertTrue((export_out / "dobbie_model_card.json").exists())


if __name__ == "__main__":
    unittest.main()
