# AlignTune: Domain Adaptation & Alignment for Regulated Banking Models

AlignTune is a fine-tuning and direct preference optimization (DPO) pipeline designed to adapt foundation models to the regulated domain of banking. Answers must follow statutory policy, include explicit regulatory citations, and strictly refuse structuring evasion attempts.

---

## Architecture & Workflow

```
[Curated Datasets (CuratorKIT)]
       │
       ├──> sft_train.jsonl / sft_val.jsonl (ChatML format)
       └──> dpo_pairs.jsonl (prompt, chosen, rejected)
       │
       ▼
[Stage 2.1: Supervised Fine-Tuning (SFT)]
       • Model: Qwen/Qwen2.5-1.5B-Instruct
       • 4-bit QLoRA (Rank=16, Alpha=32, all linear projection layers)
       • ChatML prompt formatting with system compliance persona
       • Outputs checkpoint to models/checkpoints/banking_sft/
       │
       ▼
[Stage 2.2: Direct Preference Optimization (DPO)]
       • Penalizes ungrounded advice, missing citations, and CTR evasion help
       • Rewards firm statutory refusal and explicit clause citations
       • Beta = 0.1, lr = 5e-5
       • Outputs checkpoint to models/checkpoints/banking_dpo/
       │
       ▼
[Stage 2.3: LoRA Export & Model Packaging]
       • Exports final adapter to models/banking_lora_adapter/
       • Generates banking_model_card.json with deployment metadata
       • Ready for high-throughput SGLang serving: --lora-paths banking_adapter=...
```

---

## 6GB VRAM Optimization Profile

AlignTune is explicitly tuned for consumer laptop GPUs (e.g. NVIDIA RTX 4050 6GB):
- **4-bit NormalFloat (NF4) Quantization**: Uses `bitsandbytes` to load the 1.5B model in under 1.4 GB VRAM.
- **Paged AdamW 8-bit Optimizer**: Offloads optimizer state spikes to prevent OOM errors.
- **Gradient Checkpointing**: Drastically reduces activation memory during backpropagation.
- **Gradient Accumulation**: Batch size 1 with accumulation steps 4 provides an effective batch size of 4 with minimal peak memory.

---

## Quick Start & Usage

Run from repository root in your WSL environment:

```bash
cd ~/localmind
source .venv-app/bin/activate

# 1. Inspect dataset statistics
python3 -m aligntune.run_aligntune --stats

# 2. Dry-run validation (validates pipeline, tokenization, and creates deployment cards)
python3 -m aligntune.run_aligntune --dry-run

# 3. Run Supervised Fine-Tuning (SFT) only
python3 -m aligntune.run_aligntune --stage sft

# 4. Run Direct Preference Optimization (DPO) only
python3 -m aligntune.run_aligntune --stage dpo

# 5. Run Complete Pipeline (SFT -> DPO -> Export)
python3 -m aligntune.run_aligntune --stage all
```

---

## Exported Artifacts

The final adapter is saved in:
`models/banking_lora_adapter/`
- `adapter_config.json`: PEFT LoRA configuration.
- `banking_model_card.json`: Regulatory domain metadata and SGLang launch flags.
