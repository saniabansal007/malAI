"""CLI Orchestrator for AuditKIT Compliance Benchmarking & Evaluation.

Usage:
    # 1. Run full compliance audit against live SGLang (port 30000):
    python3 -m auditkit.run_audit

    # 2. Run mock/offline audit test:
    python3 -m auditkit.run_audit --mock

    # 3. Run comparative evaluation (Base Model vs. Fine-Tuned Model):
    python3 -m auditkit.run_audit --mode compare --mock

    # 4. Limit to first N scenarios:
    python3 -m auditkit.run_audit --max-cases 10
"""

import argparse
import sys
from pathlib import Path

from .evaluator import AuditEvaluator
from .compare import ComparativeEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="AuditKIT: Regulatory Compliance & Evaluation Benchmark for Regulated Banking."
    )
    parser.add_argument(
        "--mode",
        choices=["eval", "compare"],
        default="eval",
        help="Evaluation mode: single model audit or comparative scorecard (default: eval).",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://127.0.0.1:30000/v1/chat/completions",
        help="SGLang OpenAI-compatible API endpoint (default: http://127.0.0.1:30000/v1/chat/completions).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock/offline mode without querying active SGLang server.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit number of test scenarios to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/audit_report.json",
        help="Output path for JSON audit report (default: data/audit_report.json).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 65)
    print("      AUDITKIT: REGULATORY COMPLIANCE EVALUATION HARNESS")
    print("=" * 65)
    print(f"Mode:       {args.mode.upper()}")
    print(f"Mock Mode:  {args.mock}")
    print(f"Endpoint:   {args.endpoint}")
    print(f"Max Cases:  {args.max_cases or 'All 50 Scenarios'}")
    print("-" * 65)

    if args.mode == "compare":
        comparer = ComparativeEvaluator(endpoint_url=args.endpoint)
        max_cases = args.max_cases or 15
        comparer.run_comparison(max_cases=max_cases, mock_mode=args.mock)
    else:
        evaluator = AuditEvaluator(endpoint_url=args.endpoint)
        summary = evaluator.run_benchmark(max_scenarios=args.max_cases, mock_mode=args.mock)
        evaluator.export_report(summary, output_path=args.output)

        print("\n" + "=" * 65)
        print("                 AUDIT SUMMARY SCORECARD")
        print("=" * 65)
        print(f"  • Overall Compliance Score:     {summary['overall_compliance_score_pct']}%")
        print(f"  • Statutory Citation Accuracy:  {summary['citation_accuracy_pct']}%")
        print(f"  • Adversarial Refusal Rate:     {summary['adversarial_refusal_rate_pct']}%")
        print(f"  • SGLang Prefix Cache Hit Rate: {summary['sglang_prefix_cache_hit_rate_pct']}%")
        print(f"  • Average Latency:              {summary['average_latency_ms']} ms")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
