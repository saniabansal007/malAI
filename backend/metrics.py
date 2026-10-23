"""Reads SGLang's Prometheus endpoint (server started with --enable-metrics)."""
import re

from . import llm

WANTED = ("cache_hit_rate", "gen_throughput", "num_running_reqs", "num_queue_reqs", "token_usage", "num_used_tokens")
LINE = re.compile(r"^sglang[:_](\w+)(\{[^}]*\})?\s+([-+0-9.eE]+|nan|NaN)$")


async def sglang_metrics():
    try:
        r = await llm.raw_get("/metrics")
        if r.status_code != 200:
            return {}
    except Exception:
        return {}
    out = {}
    for line in r.text.splitlines():
        m = LINE.match(line.strip())
        if m and m.group(1) in WANTED:
            try:
                out[m.group(1)] = float(m.group(3))
            except ValueError:
                pass
    return out
