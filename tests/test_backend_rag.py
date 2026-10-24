import unittest
import json
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag import Index
from backend.chat import BASE_SYSTEM
from backend.main import system_prompt


class TestBackendRAG(unittest.TestCase):

    def test_rag_indexes_magical_corpus(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            index = Index(index_path=tmp_path)
            index.index_magical_corpus(corpus_dir=str(PROJECT_ROOT / "curatorkit/corpus"))
            self.assertGreaterEqual(len(index.chunks), 60, "Should index at least 60 magical clauses/products")

            # Check that products and policies exist in chunks
            srcs = [c["src"] for c in index.chunks]
            self.assertTrue(any("[MAG-WAND-001]" in s for s in srcs), "Ollivanders wand should be indexed")
            self.assertTrue(any("[MAG-BROOM-001]" in s for s in srcs), "Nimbus 2000 should be indexed")
            self.assertTrue(any("[RET-01]" in s for s in srcs), "Return policy RET-01 should be indexed")
            self.assertTrue(any("[SHIP-01]" in s for s in srcs), "Shipping policy SHIP-01 should be indexed")

            # Test BM25 search
            wand_hits = index.search("Ollivanders wand starter", k=3)
            self.assertGreater(len(wand_hits), 0)
            self.assertIn("[MAG-WAND-001]", wand_hits[0]["src"])

            broom_hits = index.search("Nimbus 2000 broomstick", k=3)
            self.assertGreater(len(broom_hits), 0)
            self.assertIn("[MAG-BROOM-001]", broom_hits[0]["src"])

            ret_hits = index.search("unopened 30 days return window", k=3)
            self.assertGreater(len(ret_hits), 0)
            self.assertIn("[RET-01]", ret_hits[0]["src"])

            ship_hits = index.search("owl post delivery standard", k=3)
            self.assertGreater(len(ship_hits), 0)
            self.assertTrue(any("SHIP-" in h["src"] for h in ship_hits))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_dobby_system_prompts(self):
        self.assertIn("Dobby", BASE_SYSTEM)
        self.assertIn("Enchanted Emporium", BASE_SYSTEM)
        self.assertIn("Galleons", BASE_SYSTEM)

        sp = system_prompt()
        self.assertIn("Dobby", sp)
        self.assertIn("Enchanted Emporium", sp)
        self.assertIn("Galleon", sp)


if __name__ == "__main__":
    unittest.main()
