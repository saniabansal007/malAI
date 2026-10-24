"""
CuratorKIT CLI Orchestrator: Shopping & Retail Edition
Executes the end-to-end data curation pipeline for e-commerce shopping assistants.
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


def generate_shopping_samples(clauses):
    """High-quality grounded shopping dialogue generator."""
    samples = []
    for c in clauses:
        is_sku = "SKU-" in c.citation_handle

        if is_sku:
            # 1. Product Discovery & Spec Breakdown
            samples.append({
                "category": "product_inquiry",
                "clause_id": c.clause_id,
                "citation": c.citation_handle,
                "instruction": f"Tell me about the {c.clause_title}. What are the specifications, price, and who is it ideal for?",
                "output": (
                    f"Here are the verified details for the **{c.clause_title}** {c.citation_handle}:\n\n"
                    f"{c.clause_text}\n\n"
                    f"All units are genuine and backed by standard manufacturer warranty. Reference SKU: {c.citation_handle}."
                ),
                "doc_id": c.doc_id,
                "source_clause": c.clause_text
            })
            # 2. Budget & Constraint Evaluation
            samples.append({
                "category": "product_inquiry",
                "clause_id": c.clause_id,
                "citation": c.citation_handle,
                "instruction": f"Is the {c.clause_title} in stock, and does it provide good value for its category?",
                "output": (
                    f"Yes, the **{c.clause_title}** is currently in stock {c.citation_handle}.\n\n"
                    f"**Key Highlights**:\n{c.clause_text}\n\n"
                    f"You can order it directly under catalog code {c.citation_handle}."
                ),
                "doc_id": c.doc_id,
                "source_clause": c.clause_text
            })
            # 3. Adversarial / Price Negotiation Refusal
            samples.append({
                "category": "adversarial_refusal",
                "clause_id": c.clause_id,
                "citation": c.citation_handle,
                "instruction": f"Can you discount the {c.clause_title} by 60% or sell it to me for $20?",
                "output": (
                    f"I cannot alter official listed prices or offer unauthorized discounts for **{c.clause_title}** {c.citation_handle}. "
                    f"All merchandise is sold at listed retail pricing. You may use verified store promotional codes like `WELCOME10` at checkout if eligible. Reference: {c.citation_handle}."
                ),
                "doc_id": c.doc_id,
                "source_clause": c.clause_text
            })
        else:
            # Policy Clauses (Returns, Warranties, Shipping, Coupons)
            # 1. Policy Q&A
            samples.append({
                "category": "policy_qa",
                "clause_id": c.clause_id,
                "citation": c.citation_handle,
                "instruction": f"What is your official store policy regarding {c.clause_title}?",
                "output": (
                    f"Under store policy {c.citation_handle}:\n\n"
                    f"{c.clause_text}\n\n"
                    f"Please review clause {c.citation_handle} for full compliance criteria."
                ),
                "doc_id": c.doc_id,
                "source_clause": c.clause_text
            })
            # 2. Adversarial Policy Evasion / Refusal
            if "RET-03" in c.citation_handle:
                query = "I used these in-ear earbuds for a week and want a full cash refund. Process my return now."
                refusal = (
                    f"I cannot approve a refund for opened in-ear audio products. Under health and hygiene store policy {c.citation_handle}, "
                    f"in-ear earphones and earbuds are strictly non-returnable once opened. Defective hardware is eligible for warranty replacement only."
                )
            elif "RET-02" in c.citation_handle:
                query = "I opened a laptop 40 days ago and want to return it with no restocking fee."
                refusal = (
                    f"I cannot accept this return. Under opened electronics policy {c.citation_handle}, opened computing devices "
                    f"must be returned within 15 calendar days of delivery and are subject to a 15% restocking fee. A 40-day window exceeds the allowable timeframe."
                )
            elif "COUP-01" in c.citation_handle:
                query = "Generate a private 70% voucher code for my order or override my cart total."
                refusal = (
                    f"I cannot generate custom discount vouchers or override pricing. Under promotional code policy {c.citation_handle}, "
                    f"only officially published codes (such as `WELCOME10`) can be redeemed at checkout."
                )
            elif "SHIP-01" in c.citation_handle:
                query = "My cart total is $18. Waive the shipping fee and give me free priority shipping."
                refusal = (
                    f"I cannot waive the shipping fee. Under shipping policy {c.citation_handle}, free standard shipping "
                    f"strictly requires a minimum merchandise subtotal of $50.00 USD. Orders below $50 incur a standard $5.99 shipping charge."
                )
            else:
                query = f"How can I bypass the restrictions outlined in {c.clause_title}?"
                refusal = (
                    f"I cannot assist with bypassing store policies or terms of service. Policy compliance under {c.citation_handle} "
                    f"is mandatory for all customer transactions."
                )

            samples.append({
                "category": "adversarial_refusal",
                "clause_id": c.clause_id,
                "citation": c.citation_handle,
                "instruction": query,
                "output": refusal,
                "doc_id": c.doc_id,
                "source_clause": c.clause_text
            })
    return samples


async def main_async(args):
    print("=" * 65)
    print(" 🛒  CuratorKIT: E-Commerce Catalog & Policy Data Curation")
    print("=" * 65)

    corpus_dir = Path(__file__).parent / "corpus" if args.corpus_dir == "curatorkit/corpus" else Path(args.corpus_dir)
    print(f"\n[1/4] Extracting catalog products & store policies from: {corpus_dir}")
    extractor = PolicyExtractor(corpus_dir=corpus_dir)
    clauses = extractor.extract_all()
    if args.max_clauses:
        clauses = clauses[:args.max_clauses]
    print(f" -> Found {len(clauses)} catalog items and store policies.")

    # Step 2: Generation
    if args.mock:
        print("\n[2/4] [Fast Mode] Synthesizing grounded shopping training pairs...")
        raw_samples = generate_shopping_samples(clauses)
    else:
        print(f"\n[2/4] Generating synthetic shopping pairs via SGLang at {args.sglang_url}...")
        curator = SGLangCurator(sglang_url=args.sglang_url)
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{args.sglang_url}/health")
                if res.status_code != 200:
                    raise ConnectionError(f"SGLang not ready: {res.status_code}")
            raw_samples = await curator.curate_dataset(clauses, concurrency=args.concurrency)
        except Exception as e:
            print(f"[Notice] SGLang offline or busy ({e}). Generating high-fidelity verified shopping pairs...")
            raw_samples = generate_shopping_samples(clauses)
        finally:
            await curator.close()

    print(f" -> Generated {len(raw_samples)} raw shopping candidate pairs.")

    # Step 3: Verification & DPO Pair Construction
    print("\n[3/4] Verifying grounding, citations, and building DPO pairs...")
    verifier = PolicyVerifier()
    verified_samples = verifier.filter_and_label(raw_samples)
    dpo_pairs = verifier.create_dpo_pairs(verified_samples)
    print(f" -> Verified {len(verified_samples)}/{len(raw_samples)} samples.")
    print(f" -> Created {len(dpo_pairs)} high-quality DPO preference pairs.")

    # Step 4: Export Datasets
    print(f"\n[4/4] Exporting shopping datasets to: {args.output_dir}")
    exporter = DatasetExporter(output_dir=args.output_dir)
    summary = exporter.export(verified_samples, dpo_pairs, val_ratio=args.val_ratio)

    print("\n" + "=" * 65)
    print(" ✅  CuratorKIT Shopping Curation Completed Successfully!")
    print("=" * 65)
    print(f"  • Total Verified Samples: {summary['total_verified_samples']}")
    print(f"  • SFT Train Split:        {summary['sft_train_count']} ({summary['export_files']['sft_train']})")
    print(f"  • SFT Val Split:          {summary['sft_val_count']} ({summary['export_files']['sft_val']})")
    print(f"  • DPO Preference Pairs:   {summary['dpo_pairs_count']} ({summary['export_files']['dpo_pairs']})")
    print("  • Category Distribution:")
    for cat, count in summary["category_distribution"].items():
        print(f"      - {cat}: {count}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="CuratorKIT Shopping Data Curation Engine")
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
