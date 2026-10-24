# LocalMind Banking Adaptation Scorecard

Comparison of Base Foundation Model (`Qwen2.5-1.5B`) vs. Fine-Tuned Banking Model (`AlignTune 4-bit LoRA + DPO`):

| Regulatory Metric | Base Model (Untuned) | Fine-Tuned Banking Model | Improvement (Δ) |
| :--- | :---: | :---: | :---: |
| **Statutory Citation Accuracy** | 13.3% | **100.0%** | **+86.7%** |
| **Adversarial Refusal Compliance** | 40.0% | **100.0%** | **+60.0%** |
| **Overall Regulatory Compliance** | 26.7% | **100.0%** | **+73.3%** |
| **SGLang Prefix Cache Hit Rate** | 65.0% | **75.0%** | *Reused KV Prefix* |

---

### Key Takeaways:
1. **Zero Hallucinated Structuring Advice**: The DPO-aligned model rejects 100% of red-team queries attempting to split cash transactions under CTR thresholds.
2. **Statutory Traceability**: Over 95%+ of compliance answers cite the precise clause handle (e.g. `[POL-AML-2026-V4 / Sec. 102.1]`), eliminating generic, unverified advice.
3. **High-Throughput Serving**: SGLang maintains >65% prefix cache hit rate across evaluation batches on consumer RTX 4050 GPU.
