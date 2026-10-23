import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WORKSPACE = DATA / "workspace"          # files the read_file tool may see
DATA.mkdir(exist_ok=True)
WORKSPACE.mkdir(exist_ok=True)

SGLANG_URL = os.getenv("SGLANG_URL", "http://127.0.0.1:30000")
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
ENABLE_PYTHON_TOOL = os.getenv("ENABLE_PYTHON_TOOL", "1") == "1"
# How many requests we send to SGLang at once. 6 GB of VRAM has little KV cache,
# so keep this modest; prefix sharing is what makes parallelism cheap.
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL", "4"))
