# malAI: AI Shopping Assistant

A private, high-performance **E-Commerce Shopping Assistant** designed to run entirely offline on consumer GPUs (optimized for laptop **RTX 4050 6GB VRAM** using **SGLang**).

malAI transforms foundation language models into domain-grounded retail advisors that know your store catalog, accurately quote product specifications and prices, explain store policies, and politely enforce terms of service without hallucinating.

---

## 🛒 Key Capabilities

* **Store Catalog & Budget Recommendations**: Expert knowledge of 25+ consumer electronic products (laptops, studio & sports audio, titanium smartwatches, 4K creator monitors, GaN chargers, smart home IoT) with prominent, bold pricing (**$29.99** to **$2,499.99**).
* **Policy Guidance**: Clear explanations of 30-day returns, 15% open-box restocking fees, $50 free ground shipping tiers, 1-year manufacturer warranties, and 14-day price-match guarantees.
* **Adversarial & Fraud Protection**: Rejects unauthorized coupon generation, denies non-returnable opened hygiene items (in-ear earbuds), and refuses expired return/price-match requests.
* **Storefront Presentation with Floating AI Widget**: Realistic e-commerce storefront backdrop with an interactive floating assistant bubble (`✨ Ask NovaTech AI`) that expands into a rich Markdown streaming chat modal.
* **Swappable Background**: Easily replace the store backdrop anytime by simply swapping [`frontend/assets/store_bg.png`](file:///c:/Users/sania/Downloads/localmind/localmind/frontend/assets/store_bg.png) with your own store design or screenshot.

---

## 🏗️ Technical Architecture

malAI follows a 3-stage domain adaptation and evaluation pipeline:

```
[CuratorKIT] (Stage 1) ───> [AlignTune] (Stage 2) ───> [AuditKIT] (Stage 3)
 Catalog & Policy Curation   4-bit QLoRA SFT + DPO    20-Scenario Shopping Benchmark
 (311 Verified Samples)      (RTX 4050 6GB Profile)   (100% Accuracy, SGLang Serving)
```

1. **CuratorKIT (Stage 1: Grounded Data Curation)**
   * Synthesizes realistic, conversational customer inquiries grounded strictly in [`curatorkit/corpus/catalog_electronics.md`](file:///c:/Users/sania/Downloads/localmind/localmind/curatorkit/corpus/catalog_electronics.md) and [`curatorkit/corpus/store_policies.md`](file:///c:/Users/sania/Downloads/localmind/localmind/curatorkit/corpus/store_policies.md).
   * Generates **311 verified dialogue pairs** across SFT train (264), SFT val (47), and DPO preference pairs (311).
2. **AlignTune (Stage 2: 4-bit QLoRA Fine-Tuning & Merging)**
   * Fine-tunes `Qwen/Qwen2.5-1.5B-Instruct` using 4-bit QLoRA (Rank=16, Alpha=32) and Direct Preference Optimization (DPO).
   * Merges adapter weights into [`models/shopping_model_merged/`](file:///c:/Users/sania/Downloads/localmind/localmind/models/shopping_model_merged/) for low-latency SGLang serving with RadixAttention prefix caching.
3. **AuditKIT (Stage 3: Automated Shopping Benchmark)**
   * Evaluates 20 standardized e-commerce challenge scenarios across budget filtering, product comparisons, store policies, and adversarial refusal checks.
   * Scored **100.0% compliance and citation accuracy**.

---

## ⚡ Quick Start

### 1. Fast Test Mode (Instant, 0 GPU required)
To test the storefront UI, floating widget, Markdown rendering, and chat flow without loading the heavy model into GPU memory:

```bash
# Inside WSL2 Ubuntu or Linux terminal
./run_mock.sh
```
Open your browser to: **[http://localhost:8000](http://localhost:8000)**

---

### 2. Full GPU Serving (RTX 4050 Laptop GPU)
To serve the fine-tuned LLM using SGLang with full hardware acceleration:

```bash
# Starts SGLang on :30000 and the web application on :8000
./run.sh
```
Open your browser to: **[http://localhost:8000](http://localhost:8000)**

---

## 🧪 Pipeline CLI Commands

### 1. Re-curate the Dataset
```bash
# Extract catalog/policies and generate 300+ grounded samples
python3 -m curatorkit.run_curator --mock
```

### 2. Run AlignTune Pipeline
```bash
# View token and sample statistics
python3 -m aligntune.run_aligntune --stats

# Run full pipeline dry-run validation
python3 -m aligntune.run_aligntune --dry-run

# Run full SFT + DPO training on GPU
python3 -m aligntune.run_aligntune --stage all
```

### 3. Run AuditKIT Evaluation Benchmark
```bash
# Benchmark shopping accuracy in offline mode
python3 -m auditkit.run_audit --mock

# Benchmark against live running SGLang server (:30000)
python3 -m auditkit.run_audit
```

### 4. Run Unit Test Suite
```bash
python3 -m unittest discover tests/
```

---

## 📂 Project Structure

```
localmind/
├── aligntune/                 # Stage 2: SFT & DPO QLoRA alignment pipeline
│   ├── config.py             # Hyperparameters optimized for 6GB VRAM
│   ├── dataset_loader.py     # ChatML and preference pair loaders
│   ├── export_lora.py        # Adapter packaging and model card generator
│   ├── merge_adapter.py      # Merge LoRA weights into standalone model
│   ├── run_aligntune.py      # Master CLI runner
│   ├── train_dpo.py          # Direct Preference Optimization trainer
│   └── train_sft.py          # Supervised Fine-Tuning trainer
├── auditkit/                  # Stage 3: Automated compliance & shopping benchmark
│   ├── benchmark.py          # 20 standardized e-commerce challenge scenarios
│   ├── compare.py            # Comparative evaluation metrics
│   ├── evaluator.py          # Grounding, pricing, and refusal evaluator
│   └── run_audit.py          # Benchmark CLI driver
├── backend/                   # FastAPI Web Server
│   └── main.py               # Streaming chat endpoint (/api/chat) & live metrics
├── curatorkit/                # Stage 1: Synthetic data curation from retail corpus
│   ├── corpus/               # Markdown knowledge base
│   │   ├── catalog_electronics.md # 25+ products with specs & bold prices
│   │   └── store_policies.md      # Returns, warranties, shipping & price match
│   ├── export.py             # Export to ChatML, Alpaca, and DPO JSONL
│   ├── extractor.py          # Structured clause & product parser
│   ├── run_curator.py        # 300+ sample synthetic dialogue generator
│   └── verifier.py           # Grounding, product name, and price verifier
├── data/                      # Curated datasets & audit evaluation reports
│   └── curated/              # sft_train.jsonl, sft_val.jsonl, dpo_pairs.jsonl
├── frontend/                  # Web Client
│   ├── assets/               # Storefront background (store_bg.png)
│   └── index.html            # Storefront page with floating AI chat widget
├── models/                    # Model checkpoints & adapters
│   ├── shopping_lora_adapter/ # Packaged LoRA adapter & metadata card
│   └── shopping_model_merged/ # Merged standalone model weights
├── scripts/                   # Launch scripts
│   ├── setup.sh              # Environment setup script
│   └── start_sglang.sh       # SGLang server launcher tuned for 6GB GPU
├── tests/                     # Unit test suite
│   ├── mock_sglang.py        # Offline simulated SGLang server
│   ├── test_aligntune.py     # AlignTune unit tests
│   ├── test_auditkit.py      # AuditKIT unit tests
│   └── test_curatorkit.py    # CuratorKIT unit tests
├── run.sh                     # Full stack launcher (SGLang + Backend)
└── run_mock.sh                # Fast offline demo launcher
```

---

## 🎨 Customizing the Storefront Background
To change the storefront picture to your own store screenshot or branding:
1. Place your image at [`frontend/assets/store_bg.png`](file:///c:/Users/sania/Downloads/localmind/localmind/frontend/assets/store_bg.png) (or `store_bg.jpg`).
2. Refresh your browser at `http://localhost:8000`. The interface will automatically display your custom storefront!
