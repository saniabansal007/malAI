"""Supervised Fine-Tuning (SFT) for Regulated Banking / Wizarding World Adaptation.

Supports Unsloth (fast kernels) with seamless fallback to Hugging Face
TRL (SFTTrainer) + PEFT + BitsAndBytes 4-bit QLoRA on GPU, or CPU fallback.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

from .config import AlignTuneConfig
from .dataset_loader import load_sft_dataset, format_records_for_hf_sft

logger = logging.getLogger("aligntune.sft")


def run_sft_training(
    config: Optional[AlignTuneConfig] = None,
    dry_run: bool = False,
) -> Path:
    """Execute SFT training.

    Args:
        config: AlignTune configuration instance.
        dry_run: If True, validates dataset, tokenization, and pipeline without GPU training.

    Returns:
        Path to output directory containing the SFT adapter.
    """
    config = config or AlignTuneConfig().resolve_paths()
    output_dir = Path(config.sft_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  ALIGNTUNE: STAGE 2.1 — SUPERVISED FINE-TUNING (SFT)")
    print("=" * 60)
    print(f"Base Model:       {config.model_id}")
    print(f"Output Directory: {output_dir}")
    print(f"Quantization:     4-bit QLoRA (Rank={config.lora_r}, Alpha={config.lora_alpha})")
    print(f"Epochs:           {config.sft_num_epochs}")
    print(f"Batch Size:       {config.sft_batch_size} (Grad Accum: {config.sft_grad_accum})")
    print(f"Learning Rate:    {config.sft_learning_rate}")
    print(f"Dry Run:          {dry_run}")
    print("-" * 60)

    # 1. Load curated SFT datasets
    train_records, val_records = load_sft_dataset(
        train_path=config.sft_train_path,
        val_path=config.sft_val_path,
    )
    print(f"[Dataset] Loaded {len(train_records)} training and {len(val_records)} validation samples.")

    if dry_run:
        print("\n[Dry Run] Validating sample ChatML structure...")
        sample_msg = train_records[0]["messages"]
        for m in sample_msg:
            print(f"  • {m['role'].upper()}: {m['content'][:80]}...")

        # Create simulated adapter metadata for dry-run
        import json
        adapter_config = {
            "base_model_name_or_path": config.model_id,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "target_modules": config.target_modules,
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "stage": "sft_dry_run",
            "samples_trained": len(train_records),
        }
        with open(output_dir / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2)

        print(f"\n[Dry Run] Successfully validated SFT pipeline. Config saved to {output_dir}")
        return output_dir

    # 2. Attempt Unsloth or standard Hugging Face PEFT + TRL
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

        # Try Unsloth FastLanguageModel first (only on CUDA)
        use_unsloth = False
        if has_cuda:
            try:
                from unsloth import FastLanguageModel
                use_unsloth = True
                print("[AlignTune] Using Unsloth accelerated kernels.")
            except ImportError:
                print("[AlignTune] Unsloth not installed; falling back to Hugging Face PEFT + TRL.")

        if use_unsloth:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.model_id,
                max_seq_length=config.max_seq_length,
                load_in_4bit=config.load_in_4bit,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.lora_r,
                target_modules=config.target_modules,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                use_gradient_checkpointing=config.gradient_checkpointing,
                random_state=config.seed,
            )
        else:
            from transformers import AutoModelForCausalLM
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if has_cuda:
                from transformers import BitsAndBytesConfig
                compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_id,
                    quantization_config=bnb_config,
                    device_map=config.device_map,
                    trust_remote_code=True,
                )
                model = prepare_model_for_kbit_training(model)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_id,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                )

            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)

        # Prepare Hugging Face datasets
        train_formatted = format_records_for_hf_sft(train_records, tokenizer)
        val_formatted = format_records_for_hf_sft(val_records, tokenizer)

        train_ds = Dataset.from_list(train_formatted)
        val_ds = Dataset.from_list(val_formatted)

        try:
            from trl import SFTConfig, SFTTrainer
            sft_args = SFTConfig(
                output_dir=str(output_dir),
                per_device_train_batch_size=config.sft_batch_size,
                gradient_accumulation_steps=config.sft_grad_accum,
                num_train_epochs=config.sft_num_epochs,
                learning_rate=config.sft_learning_rate,
                fp16=use_fp16,
                bf16=use_bf16,
                use_cpu=not has_cuda,
                logging_steps=config.logging_steps,
                optim=optim,
                weight_decay=config.sft_weight_decay,
                lr_scheduler_type="cosine",
                seed=config.seed,
                save_strategy="no",
                report_to="none",
                dataset_text_field="text",
                max_length=config.max_seq_length,
                warmup_steps=5,
            )
            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                args=sft_args,
            )
        except (ImportError, TypeError):
            from trl import SFTTrainer
            from transformers import TrainingArguments
            sft_args = TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=config.sft_batch_size,
                gradient_accumulation_steps=config.sft_grad_accum,
                num_train_epochs=config.sft_num_epochs,
                learning_rate=config.sft_learning_rate,
                fp16=use_fp16,
                bf16=use_bf16,
                use_cpu=not has_cuda,
                logging_steps=config.logging_steps,
                optim=optim,
                weight_decay=config.sft_weight_decay,
                lr_scheduler_type="cosine",
                seed=config.seed,
                save_strategy="no",
                report_to="none",
                warmup_steps=5,
            )
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                dataset_text_field="text",
                max_seq_length=config.max_seq_length,
                args=sft_args,
            )

        print("\n[AlignTune] Starting Supervised Fine-Tuning...")
        trainer.train()

        print(f"\n[AlignTune] Saving SFT adapter to {output_dir}...")
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print("[AlignTune] SFT Training complete!")
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
        print(f"\n[AlignTune ERROR] SFT training failed: {exc}", file=sys.stderr)
        raise exc
