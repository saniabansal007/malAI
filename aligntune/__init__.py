"""AlignTune: Domain Adaptation & Alignment Pipeline for Regulated Banking Models.

Provides SFT and DPO training pipelines using Unsloth and Hugging Face TRL/PEFT,
optimized for consumer GPUs (e.g., RTX 4050 6GB VRAM) with 4-bit QLoRA.
"""

__version__ = "0.1.0"
__all__ = [
    "AlignTuneConfig",
    "load_sft_dataset",
    "load_dpo_dataset",
    "run_sft_training",
    "run_dpo_training",
    "export_lora_adapter",
]

from .config import AlignTuneConfig
from .dataset_loader import load_sft_dataset, load_dpo_dataset
from .train_sft import run_sft_training
from .train_dpo import run_dpo_training
from .export_lora import export_lora_adapter
