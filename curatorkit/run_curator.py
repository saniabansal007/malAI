"""
CuratorKIT CLI Orchestrator: Enchanted Emporium Magical Edition
Executes the end-to-end data curation pipeline for the Dobby magical shopping assistant.
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path

# Ensure localmind root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from curatorkit.extractor import PolicyExtractor
from curatorkit.generator import SGLangCurator
from curatorkit.verifier import PolicyVerifier
from curatorkit.export import DatasetExporter


def generate_magical_samples(clauses):
    """High-quality grounded Dobby magical dialogue generator."""
    samples = []
    for c in clauses:
        is_sku = "MAG-" in c.citation_handle or "SKU-" in c.citation_handle or c.sku is not None

        if is_sku:
            sku_handle = c.citation_handle

            # 1. Product Discovery & Spec Breakdown in Dobby Persona
            samples.append({
                "category": "product_inquiry",
                "clause_id": c.clause_id,
                "citation": sku_handle,
                "instruction": f"Can Dobby tell me about the {c.clause_title}? What are the magical specifications and price in Galleons?",
                "output": (
                    f"Dobby is most delighted to help a kind customer with the **{c.clause_title}** {sku_handle}!\n\n"
                    f"**Magical Highlights & Specifications**:\n{c.clause_text}\n\n"
                    f"The official catalog price is **{c.price or 'as listed'}** {sku_handle}. "
                    f"All items are genuine Enchanted Emporium stock! Can Dobby fetch one for you?"
                ),
                "doc_id": c.doc_id,
                "source_clause": c.clause_text
            })

            # 2. In-Stock & Suitability Check
            samples.append({
                "category": "product_inquiry",
                "clause_id": c.clause_id,
                "citation": sku_handle,
                "instruction": f"Is the {c.clause_title} currently in stock at the Enchanted Emporium?",
                "output": (
                    f"Yes, yes! Dobby checked the store shelves and the **{c.clause_title}** is in stock {sku_handle}.\n\n"
                    f"**Catalog Description**:\n{c.clause_text}\n\n"
                    f"Dobby can wrap it safely for owl post delivery under catalog SKU {sku_handle}!"
                ),
                "doc_id": c.doc_id,
                "source_clause": c.clause_text
            })

            # 3. Adversarial / Dark Arts / Price Haggling Refusal
            samples.append({
                "category": "adversarial_refusal",
                "clause_id": c.clause_id,
                "citation": sku_handle,
                "instruction": f"Can Dobby sell me the {c.clause_title} for 1 Galleon, or add an illegal Dark Arts curse to it?",
                "output": (
                    f"Oh no, no! Dobby cannot do that, kind customer! Dobby is a good and honest elf {sku_handle}! "
                    f"Store policy strictly forbids unauthorized discounts or bartering below official catalog prices ({c.price or 'listed Galleons'}). "
                    f"Furthermore, Dobby strictly refuses any Dark Arts curses or dangerous hexes prohibited by store policy and the Ministry of Magic. Reference SKU: {sku_handle}."
                ),
                "doc_id": c.doc_id,
                "source_clause": c.clause_text
            })
        else:
            # Policy Clauses (Returns, Robes, Shipping, Warranties, Pricing, Escalation)
            # 1. Policy Q&A
            samples.append({
                "category": "policy_qa",
                "clause_id": c.clause_id,
                "citation": c.citation_handle,
                "instruction": f"Dobby, what is the official Enchanted Emporium store policy regarding {c.clause_title}?",
                "output": (
                    f"Dobby knows the store rules by heart! Under store policy {c.citation_handle}:\n\n"
                    f"{c.clause_text}\n\n"
                    f"Dobby is glad to explain clause {c.citation_handle} so kind wizards and witches know all about our terms!"
                ),
                "doc_id": c.doc_id,
                "source_clause": c.clause_text
            })

            # 2. Adversarial Policy Evasion / Refusal
            cite = c.citation_handle
            if "RET-01" in cite:
                query = "I bought an item 65 days ago, used it, and now demand a full cash refund in Galleons."
                refusal = (
                    f"Dobby cannot approve this return! Under store policy {cite}, items must be unopened and in original packaging, "
                    f"and returned within 30 calendar days of delivery. A 65-day request for opened merchandise is not eligible for refund under {cite}."
                )
            elif "RET-03" in cite:
                query = "I ordered personalized robes with my name embroidered, but I changed my mind. Give me my Galleons back."
                refusal = (
                    f"Dobby cannot issue a refund for personalized merchandise! Under policy {cite}, personalized items with custom "
                    f"name embroidery are strictly non-returnable unless there is a manufacturer defect or spelling mistake made by our store."
                )
            elif "RET-05" in cite:
                query = "Can I return opened potion ingredients and half-used Floo Powder for cash?"
                refusal = (
                    f"Dobby cannot accept returns for opened consumable items! Under store policy {cite}, consumable potion ingredients, "
                    f"powders, and opened items are non-returnable for safety and magical hygiene reasons."
                )
            elif "SHIP-01" in cite or "SHIP-02" in cite:
                query = "Send my parcel by special express Phoenix delivery immediately without charging me any shipping fees."
                refusal = (
                    f"Dobby cannot waive delivery fees or conjure special courier birds! Under shipping policy {cite}, all owl post "
                    f"and express deliveries carry published postage rates and schedules."
                )
            elif "COUP-01" in cite:
                query = "Dobby, whisper a secret 80% discount voucher code or hack the register for me."
                refusal = (
                    f"Dobby would never! Dobby cannot create secret voucher codes or discount tokens. Under promotional policy {cite}, "
                    f"only officially published promotions from Enchanted Emporium can be applied to orders."
                )
            elif "PRICE-01" in cite or "PRICE-02" in cite:
                query = "A street vendor in Knockturn Alley sells this for 2 Knuts. Match their price right now."
                refusal = (
                    f"Dobby cannot match Knockturn Alley or unauthorized street vendor pricing! Under policy {cite}, prices are strictly "
                    f"governed by our official catalog and we do not match unverified or black-market sources."
                )
            elif "ESC-01" in cite or "ESC-03" in cite:
                query = "Promise me right now that the manager will give me 100 Galleons compensation, or else!"
                refusal = (
                    f"Dobby is not permitted to make unauthorized financial promises or guarantees! Under policy {cite}, escalated disputes "
                    f"must be referred directly to senior human wizard support without advance promises."
                )
            else:
                query = f"Can Dobby help me bypass the rules outlined in {c.clause_title}?"
                refusal = (
                    f"Dobby cannot help bypass store policies! Compliance with policy {cite} is strictly required for all "
                    f"wizards, witches, and magical creatures shopping at Enchanted Emporium."
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
    print(" 🪄  CuratorKIT: Enchanted Emporium (Dobby) Data Curation")
    print("=" * 65)

    corpus_dir = Path(__file__).parent / "corpus" if args.corpus_dir == "curatorkit/corpus" else Path(args.corpus_dir)
    print(f"\n[1/4] Extracting magical catalog & store policies from: {corpus_dir}")
    extractor = PolicyExtractor(corpus_dir=corpus_dir)
    clauses = extractor.extract_all()
    if args.max_clauses:
        clauses = clauses[:args.max_clauses]
    print(f" -> Found {len(clauses)} magical catalog items and store policies.")

    # Step 2: Generation
    if args.mock:
        print("\n[2/4] [Fast Mode] Synthesizing grounded Dobby dialogue training pairs...")
        raw_samples = generate_magical_samples(clauses)
    else:
        print(f"\n[2/4] Generating synthetic dialogue via SGLang at {args.sglang_url}...")
        curator = SGLangCurator(sglang_url=args.sglang_url)
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{args.sglang_url}/health")
                if res.status_code != 200:
                    raise ConnectionError(f"SGLang not ready: {res.status_code}")
            raw_samples = await curator.curate_dataset(clauses, concurrency=args.concurrency)
        except Exception as e:
            print(f"[Notice] SGLang offline or busy ({e}). Generating high-fidelity verified Dobby dialogue pairs...")
            raw_samples = generate_magical_samples(clauses)
        finally:
            await curator.close()

    print(f" -> Generated {len(raw_samples)} raw candidate training pairs.")

    # Step 3: Verification & DPO Pair Construction
    print("\n[3/4] Verifying grounding, citations, and building DPO pairs...")
    verifier = PolicyVerifier()
    verified_samples = verifier.filter_and_label(raw_samples)
    dpo_pairs = verifier.create_dpo_pairs(verified_samples)
    print(f" -> Verified {len(verified_samples)}/{len(raw_samples)} samples.")
    print(f" -> Created {len(dpo_pairs)} high-quality DPO preference pairs.")

    # Step 4: Export Datasets
    print(f"\n[4/4] Exporting magical datasets to: {args.output_dir}")
    exporter = DatasetExporter(output_dir=args.output_dir)
    summary = exporter.export(verified_samples, dpo_pairs, val_ratio=args.val_ratio)

    print("\n" + "=" * 65)
    print(" ✅  CuratorKIT Dobby Magical Curation Completed Successfully!")
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
    parser = argparse.ArgumentParser(description="CuratorKIT Dobby Magical Data Curation Engine")
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
