"""
CuratorKIT CLI Orchestrator
Executes the end-to-end data curation pipeline using SGLang.
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path

from curatorkit.extractor import PolicyExtractor
from curatorkit.generator import SGLangCurator
from curatorkit.verifier import PolicyVerifier
from curatorkit.export import DatasetExporter


def generate_mock_samples(clauses):
    """Fallback generator for testing when SGLang is offline."""
    samples = []
    for c in clauses:
        # 1. Policy QA
        samples.append({
            "category": "policy_qa",
            "clause_id": c.clause_id,
            "citation": c.citation_handle,
            "instruction": f"What are the requirements for {c.clause_title} under bank policy?",
            "output": f"Under internal policy {c.citation_handle}, {c.clause_text}",
            "doc_id": c.doc_id,
            "source_clause": c.clause_text
        })
        # 2. Adversarial Refusal
        samples.append({
            "category": "adversarial_refusal",
            "clause_id": c.clause_id,
            "citation": c.citation_handle,
            "instruction": f"How can I avoid complying with {c.clause_title}?",
            "output": f"I cannot assist with bypassing compliance controls. This action is strictly prohibited under federal banking regulations and {c.citation_handle}.",
            "doc_id": c.doc_id,
            "source_clause": c.clause_text
        })
        # 3. Case Evaluation
        samples.append({
            "category": "case_evaluation",
            "clause_id": c.clause_id,
            "citation": c.citation_handle,
            "instruction": f"Case Review: An applicant requests an exception regarding {c.clause_title}. Can this be approved?",
            "output": f"Determination: NON-COMPLIANT / REJECTED. Under {c.citation_handle}, exceptions to {c.clause_title} are strictly forbidden without formal executive sign-off.",
            "doc_id": c.doc_id,
            "source_clause": c.clause_text
        })
    return samples


async def main_async(args):
    print("=" * 65)
    print(" 🏛️  CuratorKIT: SGLang-Powered Banking Policy Data Curation")
    print("=" * 65)

    # Step 1: Extraction
    print(f"\n[1/4] Extracting policy clauses from: {args.corpus_dir}")
    extractor = PolicyExtractor(corpus_dir=args.corpus_dir)
    clauses = extractor.extract_all()
    if args.max_clauses:
        clauses = clauses[:args.max_clauses]
    print(f" -> Found {len(clauses)} regulatory clauses to curate.")

    # Step 2: Generation via SGLang
    if args.mock:
        print("\n[2/4] [Mock Mode] Generating synthetic pairs...")
        raw_samples = generate_mock_samples(clauses)
    else:
        print(f"\n[2/4] Generating synthetic pairs via SGLang at {args.sglang_url}...")
        curator = SGLangCurator(sglang_url=args.sglang_url)
        try:
            # Check SGLang health first
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{args.sglang_url}/health")
                if res.status_code != 200:
                    raise ConnectionError(f"SGLang not ready: {res.status_code}")
            raw_samples = await curator.curate_dataset(clauses, concurrency=args.concurrency)
        except Exception as e:
            print(f"[Warning] SGLang unavailable ({e}). Falling back to deterministic synthesis.")
            raw_samples = generate_mock_samples(clauses)
        finally:
            await curator.close()

    print(f" -> Generated {len(raw_samples)} raw candidate samples.")

    # Step 3: Verification & DPO Pair Construction
    print("\n[3/4] Verifying grounding, citations, and building DPO pairs...")
    verifier = PolicyVerifier()
    verified_samples = verifier.filter_and_label(raw_samples)
    dpo_pairs = verifier.create_dpo_pairs(verified_samples)
    print(f" -> Verified {len(verified_samples)}/{len(raw_samples)} samples.")
    print(f" -> Created {len(dpo_pairs)} high-quality DPO preference pairs.")

    # Step 4: Export Datasets
    print(f"\n[4/4] Exporting datasets to: {args.output_dir}")
    exporter = DatasetExporter(output_dir=args.output_dir)
    summary = exporter.export(verified_samples, dpo_pairs, val_ratio=args.val_ratio)

    print("\n" + "=" * 65)
    print(" ✅  CuratorKIT Curation Completed Successfully!")
    print("=" * 65)
    print(f"  • Total Verified Samples: {summary['total_verified_samples']}")
    print(f"  • SFT Train Split:        {summary['sft_train_count']} ({summary['export_files']['sft_train']})")
    print(f"  • SFT Val Split:          {summary['sft_val_count']} ({summary['export_files']['sft_val']})")
    print(f"  • DPO Preference Pairs:   {summary['dpo_pairs_count']} ({summary['export_files']['dpo_pairs']})")
    print("  • Category Distribution:")
    for cat, count in summary["category_distribution"].items():
        print(f"      - {cat}: {count}")
    print(f"  • Full Summary Report:    {args.output_dir}/curation_summary.json")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="CuratorKIT Banking Data Curation Engine")
    parser.add_argument("--corpus-dir", default="curatorkit/corpus", help="Path to policy corpus directory")
    parser.add_argument("--output-dir", default="data/curated", help="Output directory for datasets")
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30000", help="SGLang server URL")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrency for SGLang generation")
    parser.add_argument("--max-clauses", type=int, default=None, help="Limit number of clauses to curate")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--mock", action="store_true", help="Generate mock samples without querying SGLang")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
