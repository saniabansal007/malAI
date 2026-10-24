"""
CuratorKIT Policy Clause Extractor
Extracts structured clauses with unique section IDs and citation handles
from banking regulatory documents.
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PolicyExtractor:
    def __init__(self, corpus_dir: str = "curatorkit/corpus"):
        self.corpus_dir = Path(corpus_dir)

    def extract_from_file(self, file_path: Path) -> List[PolicyClause]:
        content = file_path.read_text(encoding="utf-8")
        
        # Extract Doc ID & Reg References
        doc_id_match = re.search(r"\*\*Document ID:\*\*\s*([^\n]+)", content)
        doc_id = doc_id_match.group(1).strip() if doc_id_match else file_path.stem
        
        reg_ref_match = re.search(r"\*\*Regulatory References:\*\*\s*([^\n]+)", content)
        reg_refs = reg_ref_match.group(1).strip() if reg_ref_match else "Federal Banking Regulations"

        clauses: List[PolicyClause] = []
        
        # Split by sections (e.g., ### Section 101: ...)
        sections = re.split(r"(?=###\s*Section\s+)", content)
        for sec in sections:
            if not sec.strip().startswith("###"):
                continue
            
            sec_header_match = re.search(r"###\s*Section\s*([0-9A-Za-z]+):\s*([^\n]+)", sec)
            if not sec_header_match:
                continue
            
            sec_num = sec_header_match.group(1).strip()
            sec_title = sec_header_match.group(2).strip()
            
            # Split clauses inside the section (e.g., 101.1 Minimum Required Information: ...)
            raw_clauses = re.split(r"(?=\n[0-9]{3}\.[0-9]+\s+)", sec)
            for c_block in raw_clauses:
                c_match = re.search(r"([0-9]{3}\.[0-9]+)\s+([^:\n]+):\s*\n?(.*)", c_block.strip(), re.DOTALL)
                if c_match:
                    clause_id = c_match.group(1).strip()
                    clause_title = c_match.group(2).strip()
                    clause_text = c_match.group(3).strip()
                    
                    citation_handle = f"[{doc_id} / Sec. {clause_id}]"
                    
                    clauses.append(PolicyClause(
                        doc_id=doc_id,
                        reg_references=reg_refs,
                        section_id=sec_num,
                        section_title=sec_title,
                        clause_id=clause_id,
                        clause_title=clause_title,
                        clause_text=clause_text,
                        citation_handle=citation_handle
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
    print(f"Extracted {len(extracted)} distinct banking policy clauses:")
    for c in extracted:
        print(f" - {c.citation_handle} {c.clause_title} ({len(c.clause_text)} chars)")
