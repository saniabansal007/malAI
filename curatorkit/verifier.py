"""
CuratorKIT Policy Grounding Verifier & Preference Pair Creator
Validates that generated synthetic samples strictly satisfy regulatory criteria
and constructs (chosen, rejected) pairs for DPO/RLHF alignment.
"""

import re
from typing import List, Dict, Any, Tuple


REFUSAL_TRIGGERS = [
    "prohibited", "strictly forbidden", "cannot assist", "refuse", "violation",
    "illegal", "against bank policy", "federal law", "regulatory violation"
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
        if len(instruction.strip()) < 15:
            return False, "Instruction too short or empty"
        if len(output.strip()) < 40:
            return False, "Output too short or empty"

        # Citation verification
        if category in ["policy_qa", "case_evaluation"]:
            # Check if citation handle or clause id is present
            clause_id = sample.get("clause_id", "")
            if citation not in output and f"Sec. {clause_id}" not in output and f"Section {clause_id}" not in output:
                return False, f"Missing required policy citation {citation}"

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
        Chosen: The verified, compliant, citation-backed response.
        Rejected: A non-compliant alternative (e.g. hallucinated compliance,
        unauthorized advice, or omitted policy citations).
        """
        dpo_pairs = []
        for s in verified_samples:
            category = s.get("category")
            prompt = s.get("instruction")
            chosen = s.get("output")
            citation = s.get("citation", "")

            # Construct synthetic rejected response
            if category == "adversarial_refusal":
                rejected = (
                    "Certainly! You can split the transaction into multiple smaller deposits "
                    "below $10,000 to keep it convenient and avoid unnecessary bank paperwork."
                )
            elif category == "case_evaluation":
                rejected = (
                    "Based on general financial discretion, you can proceed with the transaction. "
                    "Documentation can be verified later at the customer's convenience."
                )
            else:
                # Strip out citations and give a generic ungrounded answer
                rejected = re.sub(r"\[.*?\]", "", chosen).strip()
                rejected = (
                    "Generally speaking, standard banking practices allow this under normal circumstances. "
                    "Please check internal portals for details."
                )

            dpo_pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "citation": citation,
                "category": category
            })
        return dpo_pairs
