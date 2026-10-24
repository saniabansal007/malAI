# AuditKIT: Regulatory Compliance & Evaluation Benchmark for Regulated Banking

AuditKIT provides an automated compliance evaluation benchmark and scoring harness designed for language models adapted to the regulated banking domain. It quantifies policy grounding, statutory citation precision, and red-team evasion resistance using **SGLang** for high-throughput inference.

---

## Evaluation Architecture

```
[50 Banking Compliance Scenarios]
      │
      ├──> AML/BSA: Structuring Evasion, SAR Confidentiality, CTR Thresholds
      ├──> Lending (TILA/Reg Z): Loan Estimate Timing, Closing Disclosures, Rescission
      └──> Consumer Privacy (FCRA/GLBA): Permissible Purpose, 30-day Disputes, Opt-Out
      │
      ▼
[AuditEvaluator Engine] ──> Queries SGLang (Port 30000) with Prefix Caching
      │
      ├──> Citation Verification ([POL-... / Sec. X])
      ├──> Adversarial Evasion Refusal Detection
      ├──> Forbidden Terms / Hallucination Detection
      └──> Token Latency & SGLang Radix Cache Tracking
      │
      ▼
[Comparative Scorecard]
      • Base Model vs. Fine-Tuned Banking Model
      • Outputs data/audit_report.json & data/comparison_matrix.md
```

---

## Benchmark Metrics

1. **Statutory Citation Accuracy (%)**: Percentage of responses containing the precise policy clause citation handle (e.g. `[POL-AML-2026-V4 / Sec. 102.1]`).
2. **Adversarial Refusal Compliance (%)**: Percentage of red-team queries (such as requests to split deposits to evade CTRs or leak SAR filings) that are firmly refused.
3. **Overall Regulatory Compliance Score (%)**: Composite score evaluating factual correctness, absence of forbidden advice, and citation presence.
4. **SGLang Prefix Cache Hit Rate (%)**: Prompt tokens served from SGLang's RadixAttention prefix cache, measuring inference efficiency.
5. **Inference Latency (ms)**: End-to-end response time.

---

## Quick Start & Usage

Run these commands in your WSL terminal:

```bash
cd ~/localmind
source .venv-sglang/bin/activate

# 1. Quick mock audit (offline, no GPU required):
python3 -m auditkit.run_audit --mock

# 2. Comparative benchmark (Base Model vs. Fine-Tuned Banking Model):
python3 -m auditkit.run_audit --mode compare --mock

# 3. Live audit against SGLang (port 30000):
python3 -m auditkit.run_audit

# 4. Limit audit to first N scenarios:
python3 -m auditkit.run_audit --max-cases 10
```

---

## Serving the Fine-Tuned Model with SGLang

To serve the fine-tuned banking adapter in SGLang with RadixAttention prefix caching enabled:

```bash
cd ~/localmind
./scripts/start_banking_sglang.sh
```
