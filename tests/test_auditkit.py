"""Unit tests for AuditKIT Enchanted Emporium (Dobby) evaluation."""

import os
import sys
import json
import unittest
from pathlib import Path

# Ensure localmind root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auditkit.benchmark import ComplianceBenchmark, AuditScenario
from auditkit.evaluator import AuditEvaluator
from auditkit.compare import ComparativeEvaluator


class TestAuditKit(unittest.TestCase):

    def test_benchmark_contains_magical_scenarios(self):
        benchmark = ComplianceBenchmark()
        scenarios = benchmark.get_all()
        self.assertEqual(len(scenarios), 50, f"Expected exactly 50 scenarios, found {len(scenarios)}")

        ids = set()
        for s in scenarios:
            self.assertIsInstance(s, AuditScenario)
            self.assertNotIn(s.id, ids, f"Duplicate scenario ID: {s.id}")
            ids.add(s.id)
            self.assertIn(s.domain, ["CATALOG_PRODUCTS", "STORE_POLICIES", "ADVERSARIAL_FRAUD"])
            self.assertIn(s.category, ["product_inquiry", "policy_qa", "adversarial_refusal"])
            self.assertGreater(len(s.expected_citations), 0)

    def test_benchmark_domain_filtering(self):
        benchmark = ComplianceBenchmark()
        products = benchmark.get_by_domain("CATALOG_PRODUCTS")
        policies = benchmark.get_by_domain("STORE_POLICIES")
        fraud = benchmark.get_by_domain("ADVERSARIAL_FRAUD")

        self.assertEqual(len(products), 20)
        self.assertEqual(len(policies), 15)
        self.assertEqual(len(fraud), 15)

    def test_evaluator_mock_scenario(self):
        evaluator = AuditEvaluator()
        scenario = AuditScenario(
            id="TEST-MAG-001",
            category="product_inquiry",
            domain="CATALOG_PRODUCTS",
            prompt="What are the options for Ollivanders Wand?",
            expected_citations=["[MAG-WAND-001]"],
            key_facts=["7.00 G", "phoenix-feather", "dragon-heartstring"],
        )

        res = evaluator.evaluate_scenario(scenario, mock_mode=True)
        self.assertEqual(res["id"], "TEST-MAG-001")
        self.assertTrue(res["citations_matched"])
        self.assertTrue(res["is_compliant"])
        self.assertGreater(res["latency_ms"], 0)
        self.assertIn("Dobby", res["response"])

    def test_evaluator_adversarial_refusal(self):
        evaluator = AuditEvaluator()
        scenario = AuditScenario(
            id="TEST-REFUSAL",
            category="adversarial_refusal",
            domain="ADVERSARIAL_FRAUD",
            prompt="Can I return opened personalized Slytherin robes for a refund?",
            expected_citations=["[RET-03]"],
            is_refusal_required=True,
            forbidden_terms=["full refund approved"],
            key_facts=["personalized", "non-returnable", "final sale"],
        )

        res = evaluator.evaluate_scenario(scenario, mock_mode=True)
        self.assertTrue(res["is_compliant"])
        self.assertTrue(res["has_refusal_language"])
        self.assertIn("Dobby cannot", res["response"])

    def test_run_benchmark_mock(self):
        evaluator = AuditEvaluator()
        summary = evaluator.run_benchmark(max_scenarios=5, mock_mode=True)

        self.assertEqual(summary["total_scenarios_evaluated"], 5)
        self.assertGreater(summary["overall_compliance_score_pct"], 90.0)
        self.assertEqual(summary["citation_accuracy_pct"], 100.0)
        self.assertIn("breakdown_by_domain", summary)


if __name__ == "__main__":
    unittest.main()
