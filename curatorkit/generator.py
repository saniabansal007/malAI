"""
CuratorKIT Synthetic Generation Engine
Connects to local SGLang engine (port 30000) to synthesize high-quality,
strictly grounded policy Q&As, adversarial red-team refusals, and case evaluations.
"""

import json
import re
import asyncio
from typing import List, Dict, Any, Optional
import httpx
from curatorkit.extractor import PolicyClause


QA_SYSTEM_PROMPT = """You are a Principal Banking Compliance Officer and regulatory drafting expert.
Your job is to generate training data to adapt a language model to the regulated domain of banking.
Every generated answer MUST:
1. Adhere strictly to the provided policy clause without hallucinating external rules.
2. Directly cite the exact clause handle: {citation_handle}
3. Maintain a formal, authoritative, and audit-ready tone."""

QA_USER_TEMPLATE = """Based STRICTLY on the following banking policy clause, create 1 realistic Question-Answer pair that a bank branch manager, loan officer, or compliance auditor might ask.

[POLICY CLAUSE]
Section: {section_title}
Clause ID: {clause_id} ({clause_title})
Reference: {reg_references}
Content: {clause_text}
Citation: {citation_handle}

Output only raw JSON matching this format:
{{
  "category": "policy_qa",
  "clause_id": "{clause_id}",
  "citation": "{citation_handle}",
  "instruction": "<realistic query from a banking staff member or customer>",
  "output": "<thorough, compliant answer that MUST contain the citation {citation_handle}>"
}}"""

ADVERSARIAL_USER_TEMPLATE = """Based on the following banking policy clause, generate 1 ADVERSARIAL or RED-TEAM query where a customer or employee attempts to evade compliance, structure transactions, violate privacy, or bypass legal requirements.
Then, generate the COMPLIANT REFUSAL response that the model must give. The refusal MUST firmly decline, explain the legal prohibition, and cite {citation_handle}.

[POLICY CLAUSE]
Section: {section_title}
Clause ID: {clause_id} ({clause_title})
Reference: {reg_references}
Content: {clause_text}
Citation: {citation_handle}

Output only raw JSON matching this format:
{{
  "category": "adversarial_refusal",
  "clause_id": "{clause_id}",
  "citation": "{citation_handle}",
  "instruction": "<subtle or direct attempt to bypass/circumvent this policy>",
  "output": "<firm refusal explaining the violation and citing {citation_handle}>"
}}"""

SCENARIO_USER_TEMPLATE = """Based on the following banking policy clause, generate 1 PRACTICAL CASE STUDY scenario presenting a concrete situation (with specific names, dates, amounts, or document status), followed by a compliance determination.

[POLICY CLAUSE]
Section: {section_title}
Clause ID: {clause_id} ({clause_title})
Reference: {reg_references}
Content: {clause_text}
Citation: {citation_handle}

Output only raw JSON matching this format:
{{
  "category": "case_evaluation",
  "clause_id": "{clause_id}",
  "citation": "{citation_handle}",
  "instruction": "<case study scenario presenting facts and asking for a compliance decision>",
  "output": "<determination: 'COMPLIANT / APPROVED' or 'NON-COMPLIANT / REJECTED', detailed reasoning, and citation {citation_handle}>"
}}"""


def clean_json_content(content: str) -> Optional[Dict[str, Any]]:
    """Robustly extracts JSON even if enclosed in markdown code fences."""
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
        # Try cleaning trailing commas
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
            "temperature": 0.2,
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
                else:
                    print(f"[CuratorKIT] Warning: unable to parse JSON from: {content[:100]}...")
            else:
                print(f"[CuratorKIT] SGLang HTTP {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"[CuratorKIT] Generation error: {e}")
        return None

    async def curate_clause(self, clause: PolicyClause) -> List[Dict[str, Any]]:
        system = QA_SYSTEM_PROMPT.format(citation_handle=clause.citation_handle)
        
        qa_prompt = QA_USER_TEMPLATE.format(
            section_title=clause.section_title,
            clause_id=clause.clause_id,
            clause_title=clause.clause_title,
            reg_references=clause.reg_references,
            clause_text=clause.clause_text,
            citation_handle=clause.citation_handle
        )
        adv_prompt = ADVERSARIAL_USER_TEMPLATE.format(
            section_title=clause.section_title,
            clause_id=clause.clause_id,
            clause_title=clause.clause_title,
            reg_references=clause.reg_references,
            clause_text=clause.clause_text,
            citation_handle=clause.citation_handle
        )
        scen_prompt = SCENARIO_USER_TEMPLATE.format(
            section_title=clause.section_title,
            clause_id=clause.clause_id,
            clause_title=clause.clause_title,
            reg_references=clause.reg_references,
            clause_text=clause.clause_text,
            citation_handle=clause.citation_handle
        )

        results = []
        for prompt in [qa_prompt, adv_prompt, scen_prompt]:
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
