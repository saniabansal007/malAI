# malAI: Enchanted Emporium (Dobby AI Assistant)

A high-performance, private **Wizarding World E-Commerce Storefront & AI Assistant** running entirely offline on consumer GPUs (tuned for **RTX 4050 6GB VRAM** using **SGLang** with RadixAttention prefix caching).

malAI transforms foundation language models into **Dobby**, the devoted, humble, and strictly grounded magical house-elf assistant at the **Enchanted Emporium** in Diagon Alley. Dobby knows the entire 28-item store catalog, accurately answers store policy questions, calculates pricing in fictional Galleons (**G**), strictly refuses Dark Arts artifacts, and assists customers in an interactive shopping storefront.

---
## Technical Architecture

malAI operates a 4-phase domain adaptation, serving, and evaluation lifecycle:

```
[CuratorKIT] (Stage 1) ───> [AlignTune] (Stage 2) ───> [AuditKIT] (Stage 3) ───> [Storefront & RAG] (Stage 4)
 Catalog & Policy Curation   4-bit QLoRA SFT + DPO    50-Scenario Compliance Benchmark   FastAPI + SGLang Server
 (154 Verified Samples)     (RTX 4050 6GB Profile)   (88% Compliance, 87% Cache Hit)     Gringotts Vault & Snitch Game
```

1. **CuratorKIT (Stage 1: Grounded Wizarding Data Curation)**
   - Extracts structured clauses from [`curatorkit/corpus/catalog_magical.md`](file:///home/sania/localmind/curatorkit/corpus/catalog_magical.md) and [`curatorkit/corpus/store_policies_magical.md`](file:///home/sania/localmind/curatorkit/corpus/store_policies_magical.md).
   - Generates verified training samples: SFT train (130), SFT val (24), DPO preference pairs (130), and Alpaca format.
2. **AlignTune (Stage 2: 4-bit QLoRA & DPO Alignment)**
   - Fine-tunes `Qwen/Qwen2.5-1.5B-Instruct` using 4-bit QLoRA (Rank=16, Alpha=32) and Direct Preference Optimization (DPO).
   - Hardware-aware runner with automatic CUDA detection, bfloat16 activation, and CPU fallback.
   - Packages LoRA adapters into [`models/dobbie_magic_store_lora/`](file:///home/sania/localmind/models/dobbie_magic_store_lora/).
3. **AuditKIT (Stage 3: Automated Compliance & Safety Benchmark)**
   - 50 standardized test scenarios evaluating product knowledge, store policies (unopened 30-day returns, owl post delivery rates), and adversarial Dark Arts refusals.
   - Live hardware benchmark results on RTX 4050: **88.0% Compliance Score**, **90.0% Citation Accuracy**, **86.7% Dark Arts Refusal Rate**, and **87.2% SGLang Prefix Cache Hit Rate**.
4. **Backend & Storefront (Stage 4: Serving & RAG)**
   - FastAPI server ([`backend/main.py`](file:///home/sania/localmind/backend/main.py)) with BM25 indexing over 63 magical items and policies.
   - High-throughput SGLang inference on port `:30000` with RadixAttention prefix caching.

---

## Key Highlights & Features

* **Enchanted Emporium Storefront (`frontend/index.html`)**:
  - **Portrait Card Catalog**: Clean display showing portrait product artwork slots, item names, and an **Inspect** quick-view action.
  - **Product Specifications Modal**: Detailed look at crafting variants, wand woods & cores, broom handling specs, star ratings, and Ministry safety advisories.
  - **Buy Now & Gringotts Vault Economy**: Replaced "Add to Cart" with a live purchasing mechanism. Deducts Galleons directly from your Gringotts Vault balance and generates an instant **Gringotts Payment Receipt** popup confirming item details and Owl Post dispatch.
  - **Customizable Product Images**: Reserved slots at `frontend/assets/products/*.jpg` with intelligent magical icon fallback placeholders.

* **Golden Snitch Reflex Catch Mini-Game**:
  - Quidditch pitch stadium mini-game accessible from the top-right navbar or when short on Galleons during checkout.
  - Tap or click the darting Golden Snitch within 15 seconds to earn Galleons deposited straight into your Gringotts Vault.
  - Dynamic flight & evasion physics, streak multipliers (up to 2.5x), and synthesized Web Audio bell chimes (zero external audio files required).

* **Interactive Dobby Helper Desk**:
  - Floating launcher featuring a custom vector portrait of **Dobby the House-Elf** with ambient golden pulse animations.
  - Real-time Server-Sent Events (SSE) token streaming, live SGLang KV-cache hit rate indicator, and tokens-per-second (`tps`) telemetry.
  - Completely natural, warm dialogue style without reciting internal SKU numbers or raw technical codes.

* **Dark Arts Refusal & Compliance Grounding**:
  - Firmly and politely refuses requests for cursed artifacts (Hand of Glory, cursed opals, Dark Marks), forbidden hexes, or attempts to pay in counterfeit leprechaun gold.
  - Grounded answers backed by live BM25 retrieval over 63 magical catalog clauses and store policies.

---



## Quick Start

### 1. Fast Development / Mock Mode (0 GPU Required)
To explore the storefront UI, Golden Snitch mini-game, Buy Now flow, and Dobby chat without loading the heavy model into GPU memory:

```bash
# Inside WSL2 Ubuntu or Linux terminal
./run_mock.sh
```
Open your browser to: **[http://localhost:8000](http://localhost:8000)**

---

### 2. Full GPU Serving (RTX 4050 Laptop GPU)
To launch the SGLang inference server with full GPU acceleration alongside the FastAPI application:

```bash
# Starts SGLang on :30000 and the web storefront on :8000
./run.sh
```
Open your browser to: **[http://localhost:8000](http://localhost:8000)**

---

## Pipeline CLI Commands

### 1. Knowledge Base & Dataset Curation
```bash
# Extract catalog/policies and generate verified SFT + DPO datasets
.venv-app/bin/python3 -m curatorkit.run_curator --mock
```

### 2. Fine-Tuning Pipeline (AlignTune)
```bash
# Check dataset token statistics
.venv-app/bin/python3 -m aligntune.run_aligntune --stats

# Validate SFT + DPO pipeline execution in dry-run mode
.venv-app/bin/python3 -m aligntune.run_aligntune --dry-run

# Run complete SFT + DPO training on GPU (use .venv-sglang)
.venv-sglang/bin/python3 -m aligntune.run_aligntune --stage all
```

### 3. Compliance & Safety Benchmark (AuditKIT)
```bash
# Run offline compliance evaluation across all 50 scenarios
.venv-app/bin/python3 -m auditkit.run_audit --mock

# Run live audit against running SGLang server (:30000)
.venv-app/bin/python3 -m auditkit.run_audit
```

### 4. Run Unit Test Suite
```bash
.venv-app/bin/python3 -m unittest discover -s tests
```

---

## Project Structure

```
malAI/
├── aligntune/                 # Stage 2: SFT & DPO QLoRA alignment pipeline
│   ├── config.py             # Hyperparameters optimized for 6GB VRAM
│   ├── dataset_loader.py     # ChatML and DPO preference pair loaders
│   ├── export_lora.py        # Adapter packaging and model card generator
│   ├── merge_adapter.py      # Merge LoRA weights into base model
│   ├── run_aligntune.py      # Master CLI runner
│   ├── train_dpo.py          # Direct Preference Optimization trainer
│   └── train_sft.py          # Supervised Fine-Tuning trainer
├── auditkit/                  # Stage 3: Automated compliance & safety benchmark
│   ├── benchmark.py          # 50 wizarding world evaluation test cases
│   ├── compare.py            # Comparative evaluation & scoring metrics
│   ├── evaluator.py          # Grounding, pricing, and Dark Arts refusal evaluator
│   └── run_audit.py          # Benchmark evaluation CLI driver
├── backend/                   # FastAPI Web Server & RAG Engine
│   ├── main.py               # Streaming chat endpoint (/api/chat), metrics, & static mount
│   ├── rag.py                # BM25 magical corpus indexing & search
│   └── chat.py               # Dialogue state management & tool routing
├── curatorkit/                # Stage 1: Synthetic data curation from magical corpus
│   ├── corpus/               # Markdown knowledge base
│   │   ├── catalog_magical.md       # 28 products with specs, variants, & Galleon prices
│   │   └── store_policies_magical.md # Returns, owl post shipping, warranties, & advisories
│   ├── export.py             # Export to ChatML, Alpaca, and DPO JSONL
│   ├── extractor.py          # Structured clause & product parser
│   ├── generator.py          # Synthetic dialogue generator
│   ├── run_curator.py        # Master curation CLI runner
│   └── verifier.py           # Grounding, item name, and price verifier
├── data/                      # Curated datasets & index
│   ├── curated/              # sft_train.jsonl, sft_val.jsonl, dpo_pairs.jsonl
│   └── index.json            # 63 indexed magical clauses and policies
├── frontend/                  # Single-Page Web Storefront
│   ├── assets/               # Visual assets & icons
│   │   ├── dobby.svg         # Vector character portrait of Dobby the House-Elf
│   │   └── products/         # Product portrait image slots (MAG-*.jpg)
│   ├── index.html            # Storefront with Snitch mini-game & Dobby helper desk
│   └── products.json         # Structured catalog records for all 28 items
├── models/                    # Model checkpoints & adapters
│   ├── checkpoints/          # Active SFT and DPO checkpoints (dobbie_sft, dobbie_dpo)
│   ├── dobbie_magic_store_lora/ # Packaged LoRA adapter weights & metadata card
│   └── shopping_model_merged/ # Standalone merged model weights fallback (2.9 GB)
├── scripts/                   # Launch & setup scripts
│   ├── setup.sh              # Environment initialization script
│   ├── start_dobbie_sglang.sh # Active SGLang server launcher for Dobby
│   └── start_sglang.sh       # SGLang launcher tuned for 6GB GPU
├── tests/                     # Unit test suite
│   ├── mock_sglang.py        # Offline simulated SGLang server
│   ├── test_aligntune.py     # AlignTune unit tests
│   ├── test_auditkit.py      # AuditKIT unit tests
│   ├── test_backend_rag.py   # RAG index and prompt tests
│   └── test_curatorkit.py    # CuratorKIT unit tests
├── run.sh                     # Full stack launcher (SGLang + Backend)
└── run_mock.sh                # Fast offline demo launcher
```

---

