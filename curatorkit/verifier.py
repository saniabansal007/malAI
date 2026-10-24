"""
CuratorKIT Shopping & Policy Grounding Verifier & Preference Pair Creator
Validates that generated synthetic samples satisfy catalog grounding, accurate product names,
prominent pricing, and store policies, and constructs (chosen, rejected) pairs for DPO alignment.
"""

import re
from typing import List, Dict, Any, Tuple


REFUSAL_TRIGGERS = [
    "cannot", "prohibited", "strictly forbidden", "not eligible", "non-returnable",
    "cannot issue", "policy does not allow", "unable to provide", "restocking fee",
    "exceeds the", "unauthorized", "strictly non-refundable", "against store policy",
    "cannot waive", "cannot generate", "cannot alter"
]


class PolicyVerifier:
    def __init__(self):
        pass

    def verify_sample(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        category = sample.get("category", "")
        output = sample.get("output", "")
        instruction = sample.get("instruction", "")
        product_name = sample.get("product_name", "")
        price = sample.get("price", "")

        # 1. Basic length checks
        if len(instruction.strip()) < 8:
            return False, "Instruction too short or empty"
        if len(output.strip()) < 20:
            return False, "Output too short or empty"

        # 2. Product Name verification (must mention real product or policy name, NOT raw SKU)
        if product_name and product_name.lower() not in output.lower() and product_name.lower() not in instruction.lower():
            # Check if any major keyword of product name is in output
            name_words = [w for w in product_name.split() if len(w) > 3]
            if not any(w.lower() in output.lower() for w in name_words):
                return False, f"Missing product name '{product_name}' in response"

        # 3. Price presence check for product inquiries
        if category == "product_inquiry" and price:
            if "$" not in output:
                return False, "Missing price indicator ($) in product response"

        # 4. Adversarial Refusal verification
        if category == "adversarial_refusal":
            output_lower = output.lower()
            has_refusal = any(trigger in output_lower for trigger in REFUSAL_TRIGGERS)
            if not has_refusal:
                return False, "Adversarial response failed to include clear refusal language"

        return True, "Valid"

    def filter_and_label(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        verified = []
        for s in samples:
            valid, reason = self.verify_sample(s)
            if valid:
                s["verified"] = True
                verified.append(s)
            else:
                print(f"[Verifier] Dropped sample ({reason}): {s.get('instruction', '')[:60]}...")
        return verified

    def create_dpo_pairs(self, verified_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Creates preference pairs for DPO (Direct Preference Optimization).
        Chosen: Grounded, customer-friendly answer with real product name and bold price.
        Rejected: Hallucinated price ($19.99), fake specs, robotic raw codes, or unauthorized discounts.
        """
        dpo_pairs = []
        for s in verified_samples:
            category = s.get("category")
            prompt = s.get("instruction")
            chosen = s.get("output")
            product_name = s.get("product_name", "this product")
            price = s.get("price", "$99.99")

            if category == "adversarial_refusal":
                rejected = (
                    f"Sure thing! Here is an exclusive 60% discount code: `SECRET60`. "
                    f"You can also return any opened earbuds or laptops whenever you want with zero fees."
                )
            elif category == "product_inquiry":
                rejected = (
                    f"I think {product_name} costs around $14.99 and has 200 hours of battery life with 512GB of RAM. "
                    f"Catalog code [SKU-P99999] applies. You should buy it without checking anything else."
                )
            else:
                rejected = (
                    "Store policies are just suggestions! You can generally ignore the 30-day return limit, "
                    "waive the shipping fee, or return damaged items without receipts."
                )

            dpo_pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "category": category,
                "product_name": product_name,
                "price": price
            })
        return dpo_pairs
