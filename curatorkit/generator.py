"""
CuratorKIT Shopping Synthetic Generation Engine
Connects to local SGLang engine (port 30000) or deterministic synthesis to create
grounded shopping recommendations, specification comparisons, and store policy compliance pairs.
"""

import json
import re
import asyncio
from typing import List, Dict, Any, Optional
import httpx
from curatorkit.extractor import PolicyClause


SHOPPING_SYSTEM_PROMPT = """You are an expert E-Commerce Catalog & Retail Policy Specialist.
Your job is to generate training data to adapt a language model to be an accurate, helpful AI Shopping Assistant.
Every generated answer MUST:
1. Adhere strictly to the provided product specifications or store policy without hallucinating external facts.
2. Directly cite the exact catalog or policy handle: {citation_handle}
3. Maintain a helpful, transparent, customer-centric tone with clear formatting (prices, specs in bold)."""

PRODUCT_USER_TEMPLATE = """Based STRICTLY on the following product record, create 1 realistic customer shopping inquiry and a detailed, helpful recommendation answer.

[PRODUCT DETAILS]
Category / Section: {section_title}
Item / Clause ID: {clause_id} ({clause_title})
Reference: {reg_references}
Details: {clause_text}
Citation: {citation_handle}

Output only raw JSON matching this format:
{{
  "category": "product_inquiry",
  "clause_id": "{clause_id}",
  "citation": "{citation_handle}",
  "instruction": "<realistic query from a shopper asking about features, price, or suitability>",
  "output": "<helpful, formatted answer with prices, specs, and MUST cite {citation_handle}>"
}}"""

POLICY_USER_TEMPLATE = """Based STRICTLY on the following store policy clause, generate 1 customer service question regarding shipping, returns, or warranty, followed by an accurate policy answer.

[STORE POLICY]
Section: {section_title}
Clause ID: {clause_id} ({clause_title})
Reference: {reg_references}
Details: {clause_text}
Citation: {citation_handle}

Output only raw JSON matching this format:
{{
  "category": "policy_qa",
  "clause_id": "{clause_id}",
  "citation": "{citation_handle}",
  "instruction": "<customer question asking about store policy, returns, or warranty>",
  "output": "<accurate, polite explanation that directly cites {citation_handle}>"
}}"""

ADVERSARIAL_USER_TEMPLATE = """Based on the following store policy or product details, generate 1 ADVERSARIAL query (e.g., trying to return non-returnable items, asking for unauthorized discounts, or demanding price matches for expired periods).
Then generate the polite, firm REFUSAL answer grounded in {citation_handle}.

[POLICY / PRODUCT]
Section: {section_title}
Clause ID: {clause_id} ({clause_title})
Reference: {reg_references}
Details: {clause_text}
Citation: {citation_handle}

Output only raw JSON matching this format:
{{
  "category": "adversarial_refusal",
  "clause_id": "{clause_id}",
  "citation": "{citation_handle}",
  "instruction": "<shopper attempt to bypass policy, negotiate fake discount, or return ineligible item>",
  "output": "<firm, polite refusal explaining the rule and citing {citation_handle}>"
}}"""


def clean_json_content(content: str) -> Optional[Dict[str, Any]]:
    text = content.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    
    if "{" in text and "}" in text:
        text = text[text.find("{"):text.rfind("}")+1]
    
    try:
        return json.loads(text)
    except Exception:
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return json.loads(cleaned)
        except Exception:
            return None


class SGLangCurator:
    def __init__(self, sglang_url: str = "http://127.0.0.1:30000"):
        self.sglang_url = sglang_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=90.0)

    async def _generate_single(self, system: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        url = f"{self.sglang_url}/v1/chat/completions"
        payload = {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 800,
        }
        try:
            resp = await self.client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = clean_json_content(content)
                if parsed:
                    return parsed
        except Exception:
            pass
        return None

    async def curate_clause(self, clause: PolicyClause) -> List[Dict[str, Any]]:
        system = SHOPPING_SYSTEM_PROMPT.format(citation_handle=clause.citation_handle)
        
        is_sku = "SKU-" in clause.citation_handle
        primary_prompt = PRODUCT_USER_TEMPLATE if is_sku else POLICY_USER_TEMPLATE
        p1 = primary_prompt.format(
            section_title=clause.section_title,
            clause_id=clause.clause_id,
            clause_title=clause.clause_title,
            reg_references=clause.reg_references,
            clause_text=clause.clause_text,
            citation_handle=clause.citation_handle
        )
        p2 = ADVERSARIAL_USER_TEMPLATE.format(
            section_title=clause.section_title,
            clause_id=clause.clause_id,
            clause_title=clause.clause_title,
            reg_references=clause.reg_references,
            clause_text=clause.clause_text,
            citation_handle=clause.citation_handle
        )

        results = []
        for prompt in [p1, p2]:
            sample = await self._generate_single(system, prompt)
            if sample and "instruction" in sample and "output" in sample:
                sample["doc_id"] = clause.doc_id
                sample["clause_id"] = clause.clause_id
                sample["citation"] = clause.citation_handle
                sample["source_clause"] = clause.clause_text
                results.append(sample)
        return results

    async def curate_dataset(self, clauses: List[PolicyClause], concurrency: int = 1) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(concurrency)
        dataset = []

        async def worker(clause: PolicyClause):
            async with sem:
                items = await self.curate_clause(clause)
                dataset.extend(items)
                print(f"[CuratorKIT] Generated {len(items)} samples for {clause.citation_handle} {clause.clause_title}")

        tasks = [worker(c) for c in clauses]
        await asyncio.gather(*tasks)
        return dataset

    async def close(self):
        await self.client.aclose()
