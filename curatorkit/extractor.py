"""
CuratorKIT Shopping & Policy Extractor
Extracts structured catalog items and policy clauses with unique SKUs and policy citation handles.
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class PolicyClause:
    doc_id: str
    reg_references: str
    section_id: str
    section_title: str
    clause_id: str
    clause_title: str
    clause_text: str
    citation_handle: str
    product_name: str = ""
    price: str = ""
    category: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PolicyExtractor:
    def __init__(self, corpus_dir: str = None):
        if corpus_dir is None:
            self.corpus_dir = Path(__file__).parent / "corpus"
        else:
            self.corpus_dir = Path(corpus_dir)

    def extract_from_file(self, file_path: Path) -> List[PolicyClause]:
        content = file_path.read_text(encoding="utf-8")
        doc_id = file_path.stem

        clauses: List[PolicyClause] = []

        # Find all sections (## Section ...)
        section_blocks = re.split(r"(?=##\s+Section\s+)", content)
        for s_block in section_blocks:
            if not s_block.strip().startswith("##"):
                continue
            s_match = re.search(r"##\s+Section\s*([^:\n]+):\s*([^\n]+)", s_block)
            sec_num = s_match.group(1).strip() if s_match else "1"
            sec_title = s_match.group(2).strip() if s_match else doc_id

            # Find all clauses/products (### Clause ...)
            clause_blocks = re.split(r"(?=###\s+Clause\s+)", s_block)
            for c_block in clause_blocks:
                if not c_block.strip().startswith("###"):
                    continue
                c_header = re.search(r"###\s+Clause\s+([^:\n]+):\s*([^\n]+)", c_block)
                if not c_header:
                    continue
                clause_id = c_header.group(1).strip()
                clause_title = c_header.group(2).strip()

                # Extract text body following the header
                body = c_block[c_header.end():].strip()

                # Find citation tag like [POL-RET-01] or [SKU-LAP-101]
                cite_match = re.search(r"\[(POL-[A-Z0-9_-]+|SKU-[A-Z0-9_-]+)\]", c_block)
                if cite_match:
                    citation_handle = f"[{cite_match.group(1)}]"
                else:
                    citation_handle = f"[{clause_id}]"

                price_match = re.search(r"Price:\s*(\$[\d,]+(?:\.\d{2})?)", body)
                price = price_match.group(1) if price_match else ""

                cat_match = re.search(r"Category:\s*([^|\n]+)", body)
                category = cat_match.group(1).strip() if cat_match else sec_title

                prod_match = re.search(r"Product Name:\s*([^\n|]+)", body)
                product_name = prod_match.group(1).strip() if prod_match else clause_title

                clauses.append(PolicyClause(
                    doc_id=doc_id,
                    reg_references="E-Commerce Store Policies & Official Catalog",
                    section_id=sec_num,
                    section_title=sec_title,
                    clause_id=clause_id,
                    clause_title=clause_title,
                    clause_text=body,
                    citation_handle=citation_handle,
                    product_name=product_name,
                    price=price,
                    category=category,
                ))
        return clauses

    def extract_all(self) -> List[PolicyClause]:
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {self.corpus_dir}")

        all_clauses = []
        for file in sorted(self.corpus_dir.glob("*.md")):
            all_clauses.extend(self.extract_from_file(file))
        return all_clauses


if __name__ == "__main__":
    extractor = PolicyExtractor()
    extracted = extractor.extract_all()
    print(f"Extracted {len(extracted)} distinct shopping & policy clauses:")
    for c in extracted:
        print(f" - {c.citation_handle} {c.clause_title} ({len(c.clause_text)} chars)")
