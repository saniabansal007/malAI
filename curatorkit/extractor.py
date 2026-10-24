"""
CuratorKIT Magical Store Catalog & Policy Extractor
Extracts structured magical catalog items and store policy clauses with unique SKUs and policy citation handles.
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


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
    sku: Optional[str] = None
    price: Optional[str] = None
    category: Optional[str] = None
    in_stock: Optional[str] = None

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

        if "catalog" in doc_id:
            reg_ref = "Enchanted Emporium Magical Product Catalog"
        else:
            reg_ref = "Enchanted Emporium Store Operations & Customer Service Policies"

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

                # Extract SKU if present (e.g. - SKU: `MAG-WAND-001`)
                sku_match = re.search(r"SKU:\s*`?([A-Za-z0-9_-]+)`?", c_block)
                sku = sku_match.group(1).strip() if sku_match else None

                # Extract Price if present (e.g. - Price: 7.00 G)
                price_match = re.search(r"Price:\s*([^\n]+)", c_block)
                price = price_match.group(1).strip() if price_match else None

                # Extract Category if present
                cat_match = re.search(r"Category:\s*([^\n]+)", c_block)
                category = cat_match.group(1).strip() if cat_match else None

                # Extract In Stock if present
                stock_match = re.search(r"In Stock:\s*([^\n]+)", c_block)
                in_stock = stock_match.group(1).strip() if stock_match else None

                # Find citation tag
                cite_match = re.search(r"\[(POL-[A-Z0-9_-]+|SKU-[A-Z0-9_-]+|MAG-[A-Z0-9_-]+)\]", c_block)
                if sku:
                    citation_handle = f"[{sku}]"
                elif cite_match:
                    citation_handle = f"[{cite_match.group(1)}]"
                else:
                    citation_handle = f"[{clause_id}]"

                clauses.append(PolicyClause(
                    doc_id=doc_id,
                    reg_references=reg_ref,
                    section_id=sec_num,
                    section_title=sec_title,
                    clause_id=clause_id,
                    clause_title=clause_title,
                    clause_text=body,
                    citation_handle=citation_handle,
                    sku=sku,
                    price=price,
                    category=category,
                    in_stock=in_stock
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
    print(f"Extracted {len(extracted)} distinct magical catalog items & store policies:")
    for c in extracted:
        sku_info = f" (SKU: {c.sku}, Price: {c.price})" if c.sku else ""
        print(f" - {c.citation_handle} {c.clause_title}{sku_info}")
