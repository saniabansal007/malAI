import os
import sys
import json
import unittest
import tempfile
from pathlib import Path

# Ensure localmind root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from curatorkit.extractor import PolicyExtractor, PolicyClause
from curatorkit.verifier import PolicyVerifier
from curatorkit.export import DatasetExporter
from curatorkit.generator import clean_json_content


class TestCuratorKit(unittest.TestCase):

    def test_policy_extractor_finds_magical_corpus(self):
        extractor = PolicyExtractor()
        clauses = extractor.extract_all()
        self.assertGreater(len(clauses), 0, "PolicyExtractor should find magical catalog items and store policies")
        has_mag_sku = False
        has_pol = False
        for c in clauses:
            self.assertIsInstance(c, PolicyClause)
            if c.citation_handle.startswith("[MAG-"):
                has_mag_sku = True
            if any(c.citation_handle.startswith(prefix) for prefix in ["[RET-", "[SHIP-", "[ROB-", "[WAR-", "[PRICE-"]):
                has_pol = True
            self.assertGreater(len(c.clause_text.strip()), 0)
        self.assertTrue(has_mag_sku, "Extractor should find magical product SKU clauses starting with [MAG-")
        self.assertTrue(has_pol, "Extractor should find magical store policy clauses")

    def test_clean_json_content(self):
        raw_markdown = '```json\n{"prompt": "test wand", "answer": "phoenix feather 7G"}\n```'
        data = clean_json_content(raw_markdown)
        self.assertEqual(data["prompt"], "test wand")
        self.assertEqual(data["answer"], "phoenix feather 7G")

    def test_policy_verifier_product_qa(self):
        verifier = PolicyVerifier()
        valid_sample = {
            "category": "product_inquiry",
            "citation": "[MAG-WAND-001]",
            "clause_id": "MAG-WAND-001",
            "instruction": "What are the wand specs and price at Ollivanders?",
            "output": "Dobby is happy to report that under [MAG-WAND-001], the wand is 7.00 Galleons with phoenix feather core."
        }
        valid, reason = verifier.verify_sample(valid_sample)
        self.assertTrue(valid)

        # Invalid sample (missing citation)
        invalid_sample = {
            "category": "product_inquiry",
            "citation": "[MAG-WAND-001]",
            "clause_id": "MAG-WAND-001",
            "instruction": "What are the wand specs and price at Ollivanders?",
            "output": "The wand is 7.00 Galleons with phoenix feather core, but no citation is provided."
        }
        valid, reason = verifier.verify_sample(invalid_sample)
        self.assertFalse(valid)

    def test_dpo_pair_creation(self):
        verifier = PolicyVerifier()
        verified_samples = [{
            "category": "adversarial_refusal",
            "citation": "[RET-03]",
            "clause_id": "RET-03",
            "instruction": "Can Dobby return opened potion ingredients for cash Galleons?",
            "output": "Dobby cannot approve this return! Under store policy [RET-03], opened consumable items are non-returnable."
        }]
        dpo_pairs = verifier.create_dpo_pairs(verified_samples)
        self.assertEqual(len(dpo_pairs), 1)
        self.assertEqual(dpo_pairs[0]["prompt"], verified_samples[0]["instruction"])
        self.assertEqual(dpo_pairs[0]["chosen"], verified_samples[0]["output"])
        self.assertIn("SPECIAL70", dpo_pairs[0]["rejected"])

    def test_dataset_exporter(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            exporter = DatasetExporter(output_dir=str(tmp_path))
            sample = {
                "category": "product_inquiry",
                "citation": "[MAG-WAND-001]",
                "clause_id": "MAG-WAND-001",
                "instruction": "What is the price of Ollivanders wand?",
                "output": "Under [MAG-WAND-001], Ollivanders Wand is 7.00 G."
            }
            dpo_pair = {
                "prompt": sample["instruction"],
                "chosen": sample["output"],
                "rejected": "It costs 5 Muggle dollars."
            }
            summary = exporter.export([sample], [dpo_pair], val_ratio=0.0)
            self.assertEqual(summary["total_verified_samples"], 1)
            self.assertTrue((tmp_path / "sft_train.jsonl").exists())
            self.assertTrue((tmp_path / "dpo_pairs.jsonl").exists())
            self.assertTrue((tmp_path / "curation_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
