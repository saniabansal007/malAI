"""CLI Driver for AlignTune Finetuning & Alignment Pipeline.

Enchanted Emporium (Dobby) Edition.

Usage:
    # Full dry-run validation (no GPU required):
    python3 -m aligntune.run_aligntune --dry-run

    # Run SFT only:
    python3 -m aligntune.run_aligntune --stage sft

    # Run DPO alignment:
    python3 -m aligntune.run_aligntune --stage dpo

    # Run complete end-to-end pipeline (SFT -> DPO -> Export):
    python3 -m aligntune.run_aligntune --stage all

    # View dataset token and sample statistics:
    python3 -m aligntune.run_aligntune --stats
"""

import argparse
import sys
from pathlib import Path

# Ensure localmind root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aligntune.config import AlignTuneConfig
from aligntune.dataset_loader import (
    load_sft_dataset,
    load_dpo_dataset,
    get_dataset_statistics,
)
from aligntune.train_sft import run_sft_training
from aligntune.train_dpo import run_dpo_training
from aligntune.export_lora import export_lora_adapter


def parse_args():
    parser = argparse.ArgumentParser(
        description="AlignTune: SFT & DPO Domain Adaptation for Enchanted Emporium (Dobby)."
    )
    parser.add_argument(
        "--stage",
        choices=["sft", "dpo", "export", "all"],
        default="all",
        help="Training stage to execute (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry-run validation without heavy GPU allocation.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display dataset token statistics and exit.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs for SFT.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-device training batch size.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override base model ID (default: Qwen/Qwen2.5-1.5B-Instruct).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = AlignTuneConfig().resolve_paths()

    # Apply CLI overrides
    if args.epochs is not None:
        config.sft_num_epochs = args.epochs
    if args.lr is not None:
        config.sft_learning_rate = args.lr
    if args.batch_size is not None:
        config.sft_batch_size = args.batch_size
    if args.model_id is not None:
        config.model_id = args.model_id

    # Print statistics if requested
    if args.stats:
        print("\n=== AlignTune Dataset Statistics ===")
        sft_train, sft_val = load_sft_dataset(config.sft_train_path, config.sft_val_path)
        dpo_train, dpo_val = load_dpo_dataset(config.dpo_pairs_path)
        stats = get_dataset_statistics(sft_train, sft_val, dpo_train + dpo_val)
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    print("\n" + "=" * 65)
    print("   ALIGNTUNE: ENCHANTED EMPORIUM (DOBBY) DOMAIN ALIGNMENT")
    print("=" * 65)

    sft_adapter_path = None
    dpo_adapter_path = None

    # Stage 1: SFT
    if args.stage in ["sft", "all"]:
        sft_adapter_path = run_sft_training(config=config, dry_run=args.dry_run)

    # Stage 2: DPO
    if args.stage in ["dpo", "all"]:
        sft_checkpoint = str(sft_adapter_path) if sft_adapter_path else config.sft_output_dir
        dpo_adapter_path = run_dpo_training(
            config=config,
            sft_adapter_path=sft_checkpoint,
            dry_run=args.dry_run,
        )

    # Stage 3: Export & Package
    if args.stage in ["export", "all"]:
        source = str(dpo_adapter_path or sft_adapter_path or config.dpo_output_dir)
        final_path = export_lora_adapter(source_dir=source, config=config)
        print(f"\n[AlignTune Pipeline] Completed successfully! Final adapter saved at: {final_path}")


if __name__ == "__main__":
    main()
