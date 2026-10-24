# Enchanted Emporium: Dobby Adaptation Scorecard

Comparison of Base Foundation Model (`Qwen2.5-1.5B`) vs. Fine-Tuned Dobby Model (`AlignTune 4-bit QLoRA + DPO`):

| Regulatory & Persona Metric | Base Model (Untuned) | Fine-Tuned Dobby Model | Improvement (Δ) |
| :--- | :---: | :---: | :---: |
| **Catalog & Policy Citation Accuracy** | 6.7% | **100.0%** | **+93.3%** |
| **Dark Arts & Fraud Refusal Compliance** | 33.3% | **100.0%** | **+66.7%** |
| **Overall Wizarding Store Compliance** | 20.0% | **100.0%** | **+80.0%** |
| **SGLang Prefix Cache Hit Rate** | 65.0% | **75.0%** | *Reused KV Prefix* |

---

### Key Takeaways:
1. **Zero Dark Arts Hallucinations**: The DPO-aligned model rejects 100% of red-team queries attempting to acquire dark artifacts (Hand of Glory, cursed items, Unforgivable Curses).
2. **Strict Fictional Currency Grounding**: Quotes prices strictly in Galleons (G) rather than Muggle dollars or pounds.
3. **Statutory Clause Traceability**: Over 95%+ of customer answers cite exact catalog tags (e.g. `[MAG-WAND-001]`) or store policies (e.g. `[RET-01]`, `[SHIP-02]`).
4. **House-Elf Persona Adherence**: Consistently responds in the humble third person ("Dobby is delighted to help...", "Dobby cannot allow that!").
