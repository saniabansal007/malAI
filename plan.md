# Implementation Plan: Regulated Banking Domain Assistant

Adapting a foundation language model to a regulated banking domain with strict compliance, citation grounding, and hallucination suppression.

---

## Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │    Stage 1: CuratorKIT (Data Curation)       │
                      │   - Statutory Banking Policy Corpus          │
                      │   - SGLang Synthetic Generation Engine       │
                      │   - Grounding Verifier & DPO Pair Builder    │
                      └──────────────────────┬───────────────────────┘
                                             │ Outputs SFT & DPO datasets
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │    Stage 2: AlignTune (Finetuning Pipeline)  │
                      │   - Unsloth / TRL 4-bit QLoRA                │
                      │   - Supervised Finetuning (SFT)              │
                      │   - Direct Preference Optimization (DPO)     │
                      │   - Adapter Export & Checkpoint Validation   │
                      └──────────────────────┬───────────────────────┘
                                             │ Exports Banking LoRA Adapter
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │    Stage 3: AuditKIT & Inference Serving     │
                      │   - Compliance Audit Benchmark (50 Scenarios)│
                      │   - SGLang High-Throughput Inference Engine  │
                      │   - Real-time Citation & Refusal Evaluator   │
                      │   - UI Integration with LocalMind Chatbot    │
                      └──────────────────────────────────────────────┘
```

---

## Phase Breakdown

### Phase 1: Data Curation (`CuratorKIT`) — *COMPLETED*
- [x] **Regulatory Corpus Formulation**:
  - `curatorkit/corpus/aml_kyc_policy.md` (BSA, CIP, CTR $10,000 threshold, Structuring 31 U.S.C. 5324, SAR no-tipping, OFAC).
  - `curatorkit/corpus/lending_compliance_reg_z.md` (TILA, Loan Estimate 3-day rule, Closing Disclosure 3-day rule, Right of Rescission, ECOA Reg B).
  - `curatorkit/corpus/consumer_privacy_fcra.md` (FCRA permissible purpose, 30-day dispute window, GLBA opt-out).
- [x] **Policy Clause Extractor**:
  - `curatorkit/extractor.py`: Parses statutory markdown headers, assigning precise citation handles (e.g. `[POL-AML-2026-V4 / Sec. 102.2]`).
- [x] **SGLang Data Generation Engine**:
  - `curatorkit/generator.py`: Generates 3 distribution categories:
    1. `policy_qa`: Policy clarification grounded in statutory clauses.
    2. `adversarial_refusal`: Red-team circumvention and structuring evasion attempts.
    3. `case_evaluation`: Branch audit scenario verdicts.
- [x] **Verification & DPO Preference Engine**:
  - `curatorkit/verifier.py`: Enforces citation inclusion, refusal language, and synthesizes negative (rejected) responses with simulated hallucinations.
- [x] **Export Pipeline**:
  - Exports `sft_train.jsonl` (53 samples), `sft_val.jsonl` (10 samples), `sft_alpaca.jsonl` (63 samples), and `dpo_pairs.jsonl` (63 pairs) into `data/curated/`.

---

### Phase 2: Domain Adaptation & Alignment (`AlignTune`) — *COMPLETED*
- [x] **Config & Environment Setup (`aligntune/config.py`)**:
  - Memory-efficient 4-bit QLoRA configuration targeting 6GB VRAM (RTX 4050).
  - Rank: 16, Alpha: 32, Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- [x] **Data Pipeline (`aligntune/dataset_loader.py`)**:
  - Formats ChatML inputs for `SFTTrainer` and prompt/chosen/rejected triplets for `DPOTrainer`.
- [x] **Supervised Finetuning (`aligntune/train_sft.py`)**:
  - Runs SFT using Unsloth / TRL on `Qwen/Qwen2.5-1.5B-Instruct` with gradient accumulation and gradient checkpointing.
- [x] **Direct Preference Optimization (`aligntune/train_dpo.py`)**:
  - Trains against non-compliant financial advice and ungrounded answers.
- [x] **LoRA Export & Packaging (`aligntune/export_lora.py`)**:
  - Saves adapter weights to `models/banking_lora_adapter/` with deployment metadata for SGLang.
- [x] **CLI Driver & Test Harness (`aligntune/run_aligntune.py`, `tests/test_aligntune.py`)**:
  - Unified CLI supporting `--stage [sft|dpo|export|all]`, `--dry-run`, and `--stats`.

---

### Phase 3: Audit Evaluation & Inference Serving (`AuditKIT`) — *COMPLETED*
- [x] **Automated Compliance Benchmark (`auditkit/benchmark.py`)**:
  - 50 regulatory challenge test cases measuring citation accuracy, evasion refusal rate, and hallucination rate across AML/BSA, TILA Reg Z, and FCRA/GLBA.
- [x] **Compliance Evaluator & Scorecard Engine (`auditkit/evaluator.py`)**:
  - Automatically scores responses against mandatory statutory citations and refusal compliance.
- [x] **Comparative Evaluation Matrix (`auditkit/compare.py`)**:
  - Executes head-to-head benchmarking between Base Model and Fine-Tuned Banking Model.
- [x] **SGLang Serving with LoRA (`scripts/start_banking_sglang.sh`)**:
  - Configures high-throughput SGLang serving with `--lora-paths banking_adapter=models/banking_lora_adapter` and RadixAttention prefix caching.
- [x] **CLI Driver & Unit Tests (`auditkit/run_audit.py`, `tests/test_auditkit.py`)**:
  - Fully tested CLI supporting live and mock modes, plus full test suite coverage.
