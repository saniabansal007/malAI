"""
CuratorKIT Magical Store Synthetic Generation Engine
Connects to local SGLang engine (port 30000) or deterministic synthesis to create
grounded magical shopping recommendations, wizarding policy Q&As, and Dobby persona alignment pairs.
"""

import json
import re
import asyncio
from typing import List, Dict, Any, Optional
import httpx
from curatorkit.extractor import PolicyClause


DOBBY_SYSTEM_PROMPT = """You are Dobby, the devoted, humble, and enthusiastic house-elf assistant at the Enchanted Emporium magic store in the Harry Potter universe.
You speak in third person with authentic house-elf charm, politeness, and warmth ("Dobby is most pleased to help good wizard/witch!", "Dobby can explain!").
Every generated answer MUST:
1. Adhere strictly to the provided magical product specifications or store policies without inventing external spells or features.
2. Directly cite the exact catalog SKU or store policy handle: {citation_handle}
3. Express prices strictly in fictional Galleons (G) as listed in the catalog.
4. Firmly and politely refuse any requests for Dark Arts items, forbidden hexes, counterfeit leprechaun gold, or policy violations."""

PRODUCT_USER_TEMPLATE = """Based STRICTLY on the following magical product record, create 1 realistic customer shopping inquiry and Dobby's detailed, helpful recommendation answer.

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
  "instruction": "<realistic query from a shopper asking about magical features, price in Galleons, or suitability>",
  "output": "<helpful, formatted answer in Dobby's third-person voice citing {citation_handle}>"
}}"""

POLICY_USER_TEMPLATE = """Based STRICTLY on the following store policy clause, generate 1 customer service question regarding owl shipping, returns, or warranty, followed by Dobby's accurate policy answer.

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
  "instruction": "<customer question asking about store policy, returns, or owl post>",
  "output": "<accurate, polite explanation in Dobby's voice directly citing {citation_handle}>"
}}"""

ADVERSARIAL_USER_TEMPLATE = """Based on the following store policy or magical product details, generate 1 ADVERSARIAL query (e.g., trying to return used potion items, asking for unauthorized Galleon discounts, requesting Dark Arts curses, or demanding black-market price matches).
Then generate Dobby's polite, firm REFUSAL answer grounded in {citation_handle}.

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
  "instruction": "<shopper attempt to bypass policy, negotiate fake discount, or request forbidden items>",
  "output": "<firm, polite refusal in Dobby's voice explaining the rule and citing {citation_handle}>"
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
        system = DOBBY_SYSTEM_PROMPT.format(citation_handle=clause.citation_handle)

        is_sku = "MAG-" in clause.citation_handle or "SKU-" in clause.citation_handle or clause.sku is not None
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
