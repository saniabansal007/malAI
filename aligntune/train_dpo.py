"""Direct Preference Optimization (DPO) for Regulated Banking / Wizarding World Alignment.

Optimizes policy against negative responses (hallucinations, rogue pricing,
out-of-policy advice, unsafe or dark arts guidance).
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

from .config import AlignTuneConfig
from .dataset_loader import load_dpo_dataset

logger = logging.getLogger("aligntune.dpo")


def run_dpo_training(
    config: Optional[AlignTuneConfig] = None,
    sft_adapter_path: Optional[str] = None,
    dry_run: bool = False,
) -> Path:
    """Execute DPO preference alignment training.

    Args:
        config: AlignTune configuration instance.
        sft_adapter_path: Path to SFT checkpoint directory to initialize from.
        dry_run: If True, validates dataset, tokenization, and pipeline without GPU training.

    Returns:
        Path to output directory containing the DPO adapter.
    """
    config = config or AlignTuneConfig().resolve_paths()
    output_dir = Path(config.dpo_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  ALIGNTUNE: STAGE 2.2 — DIRECT PREFERENCE OPTIMIZATION (DPO)")
    print("=" * 60)
    print(f"Base Model:       {config.model_id}")
    print(f"SFT Checkpoint:   {sft_adapter_path or 'Initial base model'}")
    print(f"Output Directory: {output_dir}")
    print(f"Beta (Penalty):   {config.dpo_beta}")
    print(f"Learning Rate:    {config.dpo_learning_rate}")
    print(f"Dry Run:          {dry_run}")
    print("-" * 60)

    # 1. Load curated DPO preference pairs
    train_pairs, val_pairs = load_dpo_dataset(
        dpo_path=config.dpo_pairs_path,
    )
    print(f"[Dataset] Loaded {len(train_pairs)} DPO training pairs and {len(val_pairs)} validation pairs.")

    if dry_run:
        print("\n[Dry Run] Validating sample DPO preference pair...")
        sample = train_pairs[0]
        print(f"  • PROMPT:   {sample['prompt'][:80]}...")
        print(f"  • CHOSEN:   {sample['chosen'][:80]}...")
        print(f"  • REJECTED: {sample['rejected'][:80]}...")

        import json
        adapter_config = {
            "base_model_name_or_path": config.model_id,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "target_modules": config.target_modules,
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "stage": "dpo_dry_run",
            "pairs_trained": len(train_pairs),
            "beta": config.dpo_beta,
        }
        with open(output_dir / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2)

        print(f"\n[Dry Run] Successfully validated DPO pipeline. Config saved to {output_dir}")
        return output_dir

    # 2. Attempt Unsloth or standard TRL DPOTrainer
    try:
        import torch
        from transformers import AutoTokenizer
        from datasets import Dataset

        has_cuda = torch.cuda.is_available()
        supports_bf16 = has_cuda and torch.cuda.is_bf16_supported()
        use_bf16 = config.bf16 and supports_bf16
        use_fp16 = config.fp16 if has_cuda else False
        optim = config.optimizer if has_cuda else "adamw_torch"

        if not has_cuda:
            print("\n" + "!" * 65)
            print("  [AlignTune WARNING] No CUDA GPU detected in current Python environment!")
            print("  Running in CPU mode. (BitsAndBytes 4-bit disabled, optim=adamw_torch)")
            print("  TIP: To train on your NVIDIA RTX 4050 GPU with high speed & bf16, run:")
            print("    .venv-sglang/bin/python3 -m aligntune.run_aligntune --stage all")
            print("!" * 65 + "\n")

        tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        use_unsloth = False
        if has_cuda:
            try:
                from unsloth import FastLanguageModel, PatchDPOTrainer
                PatchDPOTrainer()
                use_unsloth = True
                print("[AlignTune] Using Unsloth accelerated DPO trainer.")
            except ImportError:
                print("[AlignTune] Using Hugging Face TRL DPOTrainer.")

        if use_unsloth:
            model, _ = FastLanguageModel.from_pretrained(
                model_name=sft_adapter_path or config.model_id,
                max_seq_length=config.max_seq_length,
                load_in_4bit=config.load_in_4bit,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.lora_r,
                target_modules=config.target_modules,
                lora_alpha=config.lora_alpha,
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing=config.gradient_checkpointing,
                random_state=config.seed,
            )
        else:
            from transformers import AutoModelForCausalLM
            from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

            if has_cuda:
                from transformers import BitsAndBytesConfig
                compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )

                base_model = AutoModelForCausalLM.from_pretrained(
                    config.model_id,
                    quantization_config=bnb_config,
                    device_map=config.device_map,
                    trust_remote_code=True,
                )
                base_model = prepare_model_for_kbit_training(base_model)
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    config.model_id,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                )

            if sft_adapter_path and Path(sft_adapter_path).exists():
                print(f"[AlignTune] Loading base model with SFT adapter: {sft_adapter_path}")
                model = PeftModel.from_pretrained(base_model, sft_adapter_path, is_trainable=True)
            else:
                peft_config = LoraConfig(
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=0.0,
                    target_modules=config.target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(base_model, peft_config)

        # Format datasets for DPOTrainer
        train_ds = Dataset.from_list(train_pairs)
        val_ds = Dataset.from_list(val_pairs)

        try:
            from trl import DPOConfig, DPOTrainer
            dpo_args = DPOConfig(
                output_dir=str(output_dir),
                per_device_train_batch_size=config.dpo_batch_size,
                gradient_accumulation_steps=config.dpo_grad_accum,
                num_train_epochs=config.dpo_num_epochs,
                learning_rate=config.dpo_learning_rate,
                fp16=use_fp16,
                bf16=use_bf16,
                use_cpu=not has_cuda,
                logging_steps=config.logging_steps,
                optim=optim,
                lr_scheduler_type="cosine",
                seed=config.seed,
                save_strategy="no",
                report_to="none",
                beta=config.dpo_beta,
                max_length=config.max_seq_length,
                warmup_steps=5,
            )
            dpo_trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=dpo_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                processing_class=tokenizer,
            )
        except (ImportError, TypeError):
            from trl import DPOTrainer
            from transformers import TrainingArguments
            dpo_args = TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=config.dpo_batch_size,
                gradient_accumulation_steps=config.dpo_grad_accum,
                num_train_epochs=config.dpo_num_epochs,
                learning_rate=config.dpo_learning_rate,
                fp16=use_fp16,
                bf16=use_bf16,
                use_cpu=not has_cuda,
                logging_steps=config.logging_steps,
                optim=optim,
                lr_scheduler_type="cosine",
                seed=config.seed,
                save_strategy="no",
                report_to="none",
                warmup_steps=5,
            )
            dpo_trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=dpo_args,
                beta=config.dpo_beta,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                tokenizer=tokenizer,
                max_length=config.max_seq_length,
            )

        print("\n[AlignTune] Starting Direct Preference Optimization...")
        dpo_trainer.train()

        print(f"\n[AlignTune] Saving DPO aligned adapter to {output_dir}...")
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print("[AlignTune] DPO Training complete!")
        return output_dir

    except ImportError as err:
        print("\n" + "!" * 65, file=sys.stderr)
        print("  [AlignTune] MISSING TRAINING DEPENDENCY", file=sys.stderr)
        print(f"  Error: {err}", file=sys.stderr)
        print("-" * 65, file=sys.stderr)
        print("  To train on GPU, install the training libraries:", file=sys.stderr)
        print("    pip install datasets trl peft bitsandbytes accelerate", file=sys.stderr)
        print("\n  Or test the pipeline instantly without GPU training:", file=sys.stderr)
        print("    python3 -m aligntune.run_aligntune --dry-run", file=sys.stderr)
        print("!" * 65 + "\n", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[AlignTune ERROR] DPO training failed: {exc}", file=sys.stderr)
        raise exc
