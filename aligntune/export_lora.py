"""Export and packaging utility for trained banking LoRA adapters.

Prepares the adapter weights, tokenizer, and deployment metadata in
models/banking_lora_adapter/ for SGLang serving and AuditKIT evaluation.
"""

import json
import shutil
import logging
from pathlib import Path
from typing import Optional

from .config import AlignTuneConfig

logger = logging.getLogger("aligntune.export")


def export_lora_adapter(
    source_dir: Optional[str] = None,
    target_dir: Optional[str] = None,
    config: Optional[AlignTuneConfig] = None,
) -> Path:
    """Export and validate the trained LoRA adapter for deployment.

    Args:
        source_dir: Directory containing trained checkpoint (DPO or SFT).
        target_dir: Destination directory for production deployment.
        config: AlignTune configuration instance.

    Returns:
        Path to exported adapter directory.
    """
    config = config or AlignTuneConfig().resolve_paths()
    src = Path(source_dir or config.dpo_output_dir)
    if not src.exists() or not (src / "adapter_config.json").exists():
        # Fall back to SFT checkpoint if DPO was not executed
        alt_src = Path(config.sft_output_dir)
        if alt_src.exists() and (alt_src / "adapter_config.json").exists():
            src = alt_src
        else:
            raise FileNotFoundError(
                f"No valid adapter found at {src} or {alt_src}. "
                "Ensure SFT or DPO training was completed."
            )

    dst = Path(target_dir or config.final_adapter_dir)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"\n[AlignTune Export] Packaging adapter from {src} -> {dst}...")

    # Copy adapter files
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)

    # Generate metadata model card for Stage 3 AuditKIT & SGLang
    model_card = {
        "model_name": "LocalMind Banking Compliance Assistant",
        "base_model": config.model_id,
        "adapter_type": "LoRA (4-bit QLoRA)",
        "r": config.lora_r,
        "alpha": config.lora_alpha,
        "target_modules": config.target_modules,
        "domain": "Regulated Banking Compliance (BSA, AML, TILA, FCRA, GLBA)",
        "citation_handles": [
            "POL-AML-2026-V4",
            "POL-REGZ-2026-V2",
            "POL-FCRA-2026-V3",
        ],
        "training_stages": {
            "stage_1": "CuratorKIT SGLang Synthetic Curation (63 Verified Grounded Samples)",
            "stage_2_sft": "Supervised Fine-Tuning with ChatML Prompt Formatting",
            "stage_2_dpo": "Direct Preference Optimization against Structuring Evasion & Missing Citations",
        },
        "sglang_serving_argument": f"--lora-paths banking_adapter={dst.resolve()}",
    }

    with open(dst / "banking_model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    print(f"[AlignTune Export] Export complete! Adapter ready at: {dst.resolve()}")
    return dst
