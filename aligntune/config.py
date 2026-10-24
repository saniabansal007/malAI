"""Configuration and Hyperparameters for AlignTune.

Optimized for consumer GPUs with limited VRAM (such as RTX 4050 6GB)
using 4-bit QLoRA, paged AdamW optimizer, and gradient checkpointing.
Adapted for the Enchanted Emporium (Dobby) fine-tuning pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class AlignTuneConfig:
    """Master configuration for SFT, DPO, and LoRA export."""

    # Model and Base Architecture
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_seq_length: int = 1024
    load_in_4bit: bool = True
    device_map: str = "auto"

    # LoRA Hyperparameters (PEFT)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Paths (relative to repository root)
    sft_train_path: str = "data/curated/sft_train.jsonl"
    sft_val_path: str = "data/curated/sft_val.jsonl"
    dpo_pairs_path: str = "data/curated/dpo_pairs.jsonl"

    sft_output_dir: str = "models/checkpoints/dobbie_sft"
    dpo_output_dir: str = "models/checkpoints/dobbie_dpo"
    final_adapter_dir: str = "models/dobbie_magic_store_lora"

    # SFT Training Hyperparameters (6GB VRAM Profile)
    sft_num_epochs: int = 3
    sft_batch_size: int = 1
    sft_grad_accum: int = 4
    sft_learning_rate: float = 2e-4
    sft_warmup_ratio: float = 0.1
    sft_weight_decay: float = 0.01

    # DPO Training Hyperparameters
    dpo_num_epochs: int = 2
    dpo_batch_size: int = 1
    dpo_grad_accum: int = 4
    dpo_learning_rate: float = 5e-5
    dpo_beta: float = 0.1
    dpo_warmup_ratio: float = 0.1

    # Hardware & Optimization Flags (RTX 4050 supports native BF16)
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    optimizer: str = "paged_adamw_8bit"
    logging_steps: int = 5
    save_strategy: str = "no"  # Save final model explicitly to conserve disk space
    seed: int = 42

    def resolve_paths(self, base_dir: Optional[Path] = None) -> "AlignTuneConfig":
        """Resolve paths relative to project root."""
        root = base_dir or Path.cwd()
        self.sft_train_path = str(root / self.sft_train_path)
        self.sft_val_path = str(root / self.sft_val_path)
        self.dpo_pairs_path = str(root / self.dpo_pairs_path)
        self.sft_output_dir = str(root / self.sft_output_dir)
        self.dpo_output_dir = str(root / self.dpo_output_dir)
        self.final_adapter_dir = str(root / self.final_adapter_dir)
        return self
