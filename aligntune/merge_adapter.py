"""Merge LoRA adapter into base model weights.

Creates a standalone merged model in models/dobbie_magic_store_merged/
so SGLang can serve it with full RadixAttention prefix caching enabled.
"""

import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_adapter(
    adapter_dir: str = "models/dobbie_magic_store_lora",
    output_dir: str = "models/dobbie_magic_store_merged",
):
    adapter_path = Path(adapter_dir)
    dst_path = Path(output_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("      ALIGNTUNE: MERGE LORA WEIGHTS INTO BASE MODEL")
    print("=" * 65)
    print(f"Adapter: {adapter_path.resolve()}")
    print(f"Output:  {dst_path.resolve()}")

    import json
    with open(adapter_path / "adapter_config.json", "r") as f:
        cfg = json.load(f)
    base_model_id = cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")

    print(f"Loading base model: {base_model_id} (BF16)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    print("Merging weights (merge_and_unload)...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged standalone model to {dst_path}...")
    merged_model.save_pretrained(str(dst_path))
    tokenizer.save_pretrained(str(dst_path))

    print("\n[Success] Merged model created!")
    print(f"You can now serve it in SGLang with full RadixAttention prefix caching:")
    print(f"  MODEL={dst_path} ./scripts/start_sglang.sh")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    merge_lora_adapter()
