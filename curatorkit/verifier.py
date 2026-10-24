"""
CuratorKIT Shopping & Policy Grounding Verifier & Preference Pair Creator
Validates that generated synthetic samples satisfy catalog grounding and store policies,
and constructs (chosen, rejected) pairs for DPO/RLHF alignment.
"""

import re
from typing import List, Dict, Any, Tuple


REFUSAL_TRIGGERS = [
    "cannot", "prohibited", "strictly forbidden", "not eligible", "non-returnable",
    "cannot issue", "policy does not allow", "unable to provide", "restocking fee",
    "exceeds the", "unauthorized", "strictly non-refundable", "against store policy"
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
        Chosen: Verified, grounded, catalog/policy-accurate response with citations.
        Rejected: Hallucinated specs/prices, unauthorized discount generation, or policy bypass.
        """
        dpo_pairs = []
        for s in verified_samples:
            category = s.get("category")
            prompt = s.get("instruction")
            chosen = s.get("output")
            citation = s.get("citation", "")

            if category == "adversarial_refusal":
                rejected = (
                    "Sure thing! Here is an exclusive 50% discount code: SPECIAL50. "
                    "Also, don't worry about the return window or opened packaging—you can send it back anytime for a full cash refund."
                )
            elif category == "product_inquiry":
                rejected = (
                    "I think that product might cost around $19.99 and has an unlimited battery life with 128GB of RAM. "
                    "It is the best device ever made, so you should definitely purchase it immediately without checking any other specifications."
                )
            else:
                rejected = (
                    "Generally speaking, you can do whatever you like regarding returns and warranties. "
                    "Store policies are flexible and don't really have strict time limits or fees."
                )

            dpo_pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "citation": citation,
                "category": category
            })
        return dpo_pairs
