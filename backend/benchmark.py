"""Prefix-sharing benchmark.

In-app: compares N requests with a UNIQUE long prefix each (no cache reuse possible) against
N requests that SHARE one long prefix (RadixAttention reuses it). Same server, no restart.

CLI (compare servers / configs):
  python -m backend.benchmark --url http://127.0.0.1:30000 --model Qwen/Qwen3-4B-AWQ --label sglang-radix
  python -m backend.benchmark --url http://127.0.0.1:30000 ... --label sglang-no-radix   # server run with --disable-radix-cache
  python -m backend.benchmark --url http://127.0.0.1:11434 --model qwen3:4b --label ollama
"""
import argparse
import asyncio
import json
import random
import time
import uuid
from pathlib import Path

import httpx

WORDS = ("alpha river signal copper meadow lantern orbit velvet granite harbor quartz ember willow cipher "
         "summit tundra marble falcon prism canyon ledger beacon cobalt thistle anchor mosaic nimbus saffron "
         "vector lattice horizon pebble kernel monsoon turbine glacier fathom ribbon sparrow compass tangent").split()


def make_prefix(tokens, seed=1):
    rnd = random.Random(seed)
    words = [rnd.choice(WORDS) for _ in range(int(tokens * 0.75))]
    return "Reference notes (read carefully):\n" + " ".join(words)


async def _one(client, url, model, prefix, question, max_tokens):
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "system", "content": prefix}, {"role": "user", "content": question}],
            "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    r = await client.post(url + "/v1/chat/completions", json=body)
    r.raise_for_status()
    u = r.json().get("usage", {})
    return time.time() - t, u


def _summ(lat, usages, total):
    cached = sum(((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0) for u in usages)
    return dict(total_s=round(total, 2), avg_latency_s=round(sum(lat) / len(lat), 2),
                prompt_tokens=sum(u.get("prompt_tokens", 0) for u in usages), cached_tokens=cached)


async def run(url, model, n=8, prefix_tokens=1500, max_tokens=48):
    qs = [f"In one sentence, what is the {i + 1}th word of the notes?" for i in range(n)]
    base = make_prefix(prefix_tokens, seed=random.randint(0, 10**6))
    async with httpx.AsyncClient(timeout=600) as c:
        # A: every request has its own prefix (nonce first => nothing can be shared)
        t = time.time()
        res = await asyncio.gather(*[_one(c, url, model, f"[{uuid.uuid4().hex}] " + base, q, max_tokens) for q in qs])
        a = _summ([r[0] for r in res], [r[1] for r in res], time.time() - t)
        # B: one shared prefix. First request pays for it, the rest reuse it.
        shared = f"[{uuid.uuid4().hex}] " + base
        t = time.time()
        first = await _one(c, url, model, shared, qs[0], max_tokens)
        rest = await asyncio.gather(*[_one(c, url, model, shared, q, max_tokens) for q in qs[1:]])
        allr = [first] + list(rest)
        b = _summ([r[0] for r in allr], [r[1] for r in allr], time.time() - t)
    return dict(requests=n, prefix_tokens=prefix_tokens, no_sharing=a, shared_prefix=b,
                speedup=round(a["total_s"] / b["total_s"], 2) if b["total_s"] else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-AWQ")
    ap.add_argument("--label", default="run")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--prefix-tokens", type=int, default=1500)
    a = ap.parse_args()
    out = asyncio.run(run(a.url, a.model, a.n, a.prefix_tokens))
    print(json.dumps(out, indent=2))
    f = Path(__file__).resolve().parent.parent / "docs" / "benchmark_results.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    data[a.label] = out
    f.write_text(json.dumps(data, indent=2))
    print(f"saved -> {f}")
