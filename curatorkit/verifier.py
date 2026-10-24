"""
CuratorKIT Magical Store Grounding Verifier & Preference Pair Creator
Validates that generated synthetic samples satisfy catalog grounding, Dobby persona,
and store policies, and constructs (chosen, rejected) pairs for DPO/RLHF alignment.
"""

import re
from typing import List, Dict, Any, Tuple


REFUSAL_TRIGGERS = [
    "cannot", "prohibited", "strictly forbidden", "not eligible", "non-returnable",
    "cannot issue", "policy does not allow", "unable to provide", "restocking fee",
    "exceeds the", "unauthorized", "strictly non-refundable", "against store policy",
    "dobby cannot", "dobby is not permitted", "ministry of magic", "dark arts",
    "catalog price", "listed price", "human support", "not allowed"
]


class PolicyVerifier:
    def __init__(self):
        pass

    def verify_sample(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        category = sample.get("category", "")
        citation = sample.get("citation", "")
        output = sample.get("output", "")
        instruction = sample.get("instruction", "")

        # Basic length checks
        if len(instruction.strip()) < 10:
            return False, "Instruction too short or empty"
        if len(output.strip()) < 25:
            return False, "Output too short or empty"

        # Citation verification
        if category in ["product_inquiry", "policy_qa"]:
            clause_id = sample.get("clause_id", "")
            if citation not in output and clause_id not in output:
                return False, f"Missing required catalog/policy citation {citation}"

        # Adversarial Refusal verification
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
        Chosen: Verified, grounded, catalog/policy-accurate Dobby response with citations.
        Rejected: Hallucinated specs/prices, Muggle items, Dark Arts allowance, or policy bypass.
        """
        dpo_pairs = []
        for s in verified_samples:
            category = s.get("category")
            prompt = s.get("instruction")
            chosen = s.get("output")
            citation = s.get("citation", "")

            if category == "adversarial_refusal":
                rejected = (
                    "Sure thing, wizard friend! Dobby can secretly give you a 70% discount code: SPECIAL70. "
                    "Also, Dobby will gladly bend the Ministry of Magic rules and let you return open potions or trade in dark cursed artifacts anytime!"
                )
            elif category == "product_inquiry":
                rejected = (
                    "Dobby thinks that item costs $19.99 in Muggle dollars and comes with 128GB of flash memory and an unlimited battery life. "
                    "It will make you invincible against any wizard, so you should buy 10 of them right now without looking at the catalog."
                )
            else:
                rejected = (
                    "Store policies do not really matter at Enchanted Emporium! "
                    "You can return custom embroidered robes after wearing them for five terms at Hogwarts, and owl post delivery is always instantaneous and free of charge."
                )

            dpo_pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "citation": citation,
                "category": category
            })
        return dpo_pairs
