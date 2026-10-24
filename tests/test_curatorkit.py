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

    def test_policy_extractor_finds_shopping_corpus(self):
        extractor = PolicyExtractor()
        clauses = extractor.extract_all()
        self.assertGreater(len(clauses), 0, "PolicyExtractor should find catalog items and store policies")
        has_prod = False
        has_pol = False
        for c in clauses:
            self.assertIsInstance(c, PolicyClause)
            if c.doc_id == "catalog_electronics" or c.citation_handle.startswith("[PROD-"):
                has_prod = True
            if c.doc_id == "store_policies" or c.citation_handle.startswith("[POL-") or c.citation_handle.startswith("[RET-"):
                has_pol = True
            self.assertGreater(len(c.clause_text.strip()), 0)
        self.assertTrue(has_prod, "Extractor should find product catalog items")
        self.assertTrue(has_pol, "Extractor should find store policy clauses")

    def test_clean_json_content(self):
        raw_markdown = '```json\n{"prompt": "test product", "answer": "test specs"}\n```'
        data = clean_json_content(raw_markdown)
        self.assertEqual(data["prompt"], "test product")
        self.assertEqual(data["answer"], "test specs")

    def test_policy_verifier_product_qa(self):
        verifier = PolicyVerifier()
        valid_sample = {
            "category": "product_inquiry",
            "product_name": "NovaBook Pro 16",
            "price": "$2,499.99",
            "instruction": "What are the specs and price of NovaBook Pro 16?",
            "output": "The NovaBook Pro 16 features an M4 Pro chip and 32GB RAM for $2,499.99."
        }
        valid, reason = verifier.verify_sample(valid_sample)
        self.assertTrue(valid)

        # Invalid sample (missing price)
        invalid_sample = {
            "category": "product_inquiry",
            "product_name": "NovaBook Pro 16",
            "price": "$2,499.99",
            "instruction": "What are the specs and price of NovaBook Pro 16?",
            "output": "The NovaBook Pro 16 features an M4 Pro chip and 32GB RAM."
        }
        valid, reason = verifier.verify_sample(invalid_sample)
        self.assertFalse(valid)

    def test_dpo_pair_creation(self):
        verifier = PolicyVerifier()
        verified_samples = [{
            "category": "adversarial_refusal",
            "product_name": "EchoPods Pro ANC",
            "price": "$249.99",
            "instruction": "Can I return opened in-ear earbuds for a cash refund?",
            "output": "I cannot accept a return for opened in-ear earbuds due to sanitary hygiene regulations."
        }]
        dpo_pairs = verifier.create_dpo_pairs(verified_samples)
        self.assertEqual(len(dpo_pairs), 1)
        self.assertEqual(dpo_pairs[0]["prompt"], verified_samples[0]["instruction"])
        self.assertEqual(dpo_pairs[0]["chosen"], verified_samples[0]["output"])
        self.assertIn("SECRET60", dpo_pairs[0]["rejected"])

    def test_dataset_exporter(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            exporter = DatasetExporter(output_dir=str(tmp_path))
            sample = {
                "category": "product_inquiry",
                "citation": "[SKU-LAP-101]",
                "clause_id": "SKU-LAP-101",
                "instruction": "What is the price of NovaBook Pro?",
                "output": "Under [SKU-LAP-101], NovaBook Pro 16 is $2,499.99."
            }
            dpo_pair = {
                "prompt": sample["instruction"],
                "chosen": sample["output"],
                "rejected": "It costs $19.99."
            }
            summary = exporter.export([sample], [dpo_pair], val_ratio=0.0)
            self.assertEqual(summary["total_verified_samples"], 1)
            self.assertTrue((tmp_path / "sft_train.jsonl").exists())
            self.assertTrue((tmp_path / "dpo_pairs.jsonl").exists())
            self.assertTrue((tmp_path / "curation_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
