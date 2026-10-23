"""Tiny dependency-free BM25 index (swap for embeddings + Chroma if you have time)."""
import json, math, re
from collections import Counter
from pathlib import Path

PATH = Path("data/index.json")
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())

class Index:
    def __init__(self):
        self.chunks = json.loads(PATH.read_text()) if PATH.exists() else []
    def add(self, src, text, size=140, step=110):
        w = text.split()
        for i in range(0, max(len(w), 1), step):
            piece = " ".join(w[i:i + size])
            if piece.strip():
                self.chunks.append({"src": src, "text": piece})
        PATH.write_text(json.dumps(self.chunks))
    def clear(self):
        self.chunks = []; PATH.write_text("[]")
    def sources(self):
        return sorted({c["src"] for c in self.chunks})
    def search(self, q, k=6):
        if not self.chunks: return []
        docs = [Counter(tok(c["text"])) for c in self.chunks]
        n = len(docs); avg = sum(sum(d.values()) for d in docs) / n or 1
        df = Counter(t for d in docs for t in d)
        scores = []
        for c, d in zip(self.chunks, docs):
            L = sum(d.values()); s = 0
            for t in set(tok(q)):
                if t in d:
                    idf = math.log(1 + (n - df[t] + .5) / (df[t] + .5))
                    s += idf * d[t] * 2.2 / (d[t] + 1.2 * (.25 + .75 * L / avg))
            scores.append((s, c))
        return [c for s, c in sorted(scores, key=lambda x: -x[0])[:k] if s > 0]

library = Index()

