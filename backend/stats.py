"""Rolling record of requests this app sent to SGLang (for the live panel)."""
import time
from collections import deque


class Stats:
    def __init__(self):
        self.recent = deque(maxlen=60)
        self.total = dict(requests=0, prompt_tokens=0, cached_tokens=0, completion_tokens=0)
        self.cache_reported = False

    def add(self, latency, usage, ttft=None, kind="llm"):
        u = usage or {}
        pt = u.get("prompt_tokens") or 0
        ct = u.get("completion_tokens") or 0
        details = u.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        if cached is not None:
            self.cache_reported = True
        cached = cached or 0
        self.total["requests"] += 1
        self.total["prompt_tokens"] += pt
        self.total["cached_tokens"] += cached
        self.total["completion_tokens"] += ct
        self.recent.append(dict(t=time.time(), kind=kind, latency=latency, ttft=ttft,
                                prompt=pt, cached=cached, completion=ct))

    def summary(self):
        r = list(self.recent)
        avg = sum(x["latency"] for x in r) / len(r) if r else 0
        t = self.total
        return dict(**t, avg_latency=avg, cache_reported=self.cache_reported,
                    saved_pct=(100 * t["cached_tokens"] / t["prompt_tokens"]) if t["prompt_tokens"] else 0,
                    recent=r[-30:])


STATS = Stats()
