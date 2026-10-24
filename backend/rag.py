"""
Enchanted Emporium BM25 Knowledge Base & Retrieval Engine
Indexes magical catalog products and store policies with structured citation handles.
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PATH = ROOT / "data/index.json"
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())


class Index:
    def __init__(self, index_path: Path = PATH):
        self.index_path = Path(index_path)
        self.chunks: List[Dict[str, Any]] = (
            json.loads(self.index_path.read_text(encoding="utf-8"))
            if self.index_path.exists() and self.index_path.stat().st_size > 2
            else []
        )
        if not self.chunks:
            self.index_magical_corpus()

    def add(self, src: str, text: str, size: int = 140, step: int = 110):
        w = text.split()
        for i in range(0, max(len(w), 1), step):
            piece = " ".join(w[i:i + size])
            if piece.strip():
                self.chunks.append({"src": src, "text": piece})
        self._save()

    def add_clause(self, src: str, text: str):
        """Adds a single discrete catalog product or store policy clause without breaking it up."""
        if text.strip():
            self.chunks.append({"src": src, "text": text.strip()})
        self._save()

    def clear(self):
        self.chunks = []
        self._save()

    def _save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(self.chunks, indent=2), encoding="utf-8")

    def sources(self) -> List[str]:
        return sorted({c["src"] for c in self.chunks})

    def search(self, q: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []
        docs = [Counter(tok(c["text"])) for c in self.chunks]
        n = len(docs)
        avg = sum(sum(d.values()) for d in docs) / n or 1
        df = Counter(t for d in docs for t in d)
        scores = []
        q_tokens = set(tok(q))
        for c, d in zip(self.chunks, docs):
            L = sum(d.values())
            s = 0.0
            for t in q_tokens:
                if t in d:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * d[t] * 2.2 / (d[t] + 1.2 * (0.25 + 0.75 * L / avg))
            src_lower = c["src"].lower()
            if any(t in src_lower for t in q_tokens if len(t) > 3):
                s += 1.5
            scores.append((s, c))
        return [c for s, c in sorted(scores, key=lambda x: -x[0])[:k] if s > 0]

    def index_magical_corpus(self, corpus_dir: str = None):
        """Indexes all clauses and products from the Enchanted Emporium magical corpus."""
        if corpus_dir is None:
            corpus_path = ROOT / "curatorkit/corpus"
        else:
            corpus_path = Path(corpus_dir)

        from curatorkit.extractor import PolicyExtractor
        extractor = PolicyExtractor(corpus_dir=str(corpus_path))
        clauses = extractor.extract_all()

        self.chunks = []
        for c in clauses:
            is_sku = "MAG-" in c.citation_handle or c.sku is not None
            if is_sku:
                src = f"{c.citation_handle} {c.clause_title}"
                text = (
                    f"Product: {c.clause_title} | SKU: {c.sku} | Price: {c.price} | "
                    f"Category: {c.category} | In Stock: {c.in_stock}\n"
                    f"Specifications & Details: {c.clause_text}\n"
                    f"Reference SKU: {c.citation_handle}"
                )
            else:
                src = f"{c.citation_handle} {c.clause_title}"
                text = (
                    f"Store Policy {c.citation_handle}: {c.clause_title}\n"
                    f"Section: {c.section_title}\n"
                    f"Rules: {c.clause_text}\n"
                    f"Official Clause Handle: {c.citation_handle}"
                )
            self.chunks.append({"src": src, "text": text})

        self._save()
        print(f"[RAG] Indexed {len(self.chunks)} magical catalog items and store policies into {self.index_path}")


library = Index()

if __name__ == "__main__":
    library.index_magical_corpus()
    print(f"Total indexed items: {len(library.chunks)}")
    test_queries = ["Ollivanders wand", "Nimbus 2000 price", "return opened merchandise", "owl post delivery"]
    print("\nTest Retrieval Queries:")
    for tq in test_queries:
        hits = library.search(tq, k=2)
        print(f"\nQuery: '{tq}' -> {len(hits)} hits:")
        for h in hits:
            print(f" - {h['src']}: {h['text'][:100]}...")
