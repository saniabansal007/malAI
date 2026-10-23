"""Thin async client for SGLang's OpenAI-compatible server.

Everything the app does goes through here, so prompts that share a prefix
(same system message + same history) automatically hit RadixAttention.
"""
import asyncio
import json
import re
import time

import httpx

from . import config
from .stats import STATS

_client = httpx.AsyncClient(base_url=config.SGLANG_URL, timeout=httpx.Timeout(600.0, connect=5.0))
_sem = asyncio.Semaphore(config.MAX_PARALLEL)


class SGLangDown(RuntimeError):
    pass


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return re.sub(r"^.*?</think>", "", text, flags=re.S).strip()


def _body(messages, *, stream=False, thinking=False, max_tokens=600, temperature=0.4, schema=None, n=1):
    b = {
        "model": config.MODEL,
        "messages": messages,
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Qwen3 chat-template switch: reasoning on/off per request.
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    if n > 1:
        b["n"] = n
    if schema:
        # Constrained decoding: SGLang compiles the schema to a grammar, so the
        # model physically cannot emit invalid JSON.
        b["response_format"] = {"type": "json_schema",
                                "json_schema": {"name": "out", "schema": schema, "strict": True}}
    if stream:
        b["stream_options"] = {"include_usage": True}
    return b


async def _post(body):
    try:
        r = await _client.post("/v1/chat/completions", json=body)
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        raise SGLangDown(f"Can't reach SGLang at {config.SGLANG_URL}. Start it with scripts/start_sglang.sh") from e
    if r.status_code != 200:
        raise RuntimeError(f"SGLang error {r.status_code}: {r.text[:300]}")
    return r.json()


async def complete(messages, *, kind="llm", **kw):
    """Non-streaming call. Returns (list_of_texts, usage)."""
    async with _sem:
        t = time.time()
        j = await _post(_body(messages, **kw))
        dt = time.time() - t
    STATS.add(dt, j.get("usage"), kind=kind)
    texts = [strip_think(c["message"].get("content") or "") for c in j["choices"]]
    return texts, j.get("usage") or {}


async def json_complete(messages, schema, *, max_tokens=300, kind="json", **kw):
    texts, usage = await complete(messages, schema=schema, max_tokens=max_tokens,
                                  temperature=kw.pop("temperature", 0.1), kind=kind, **kw)
    raw = texts[0]
    try:
        return json.loads(raw), usage
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            raise RuntimeError(f"Model returned invalid JSON: {raw[:200]}")
        return json.loads(m.group(0)), usage


def _split(buf, in_think):
    """Split a text buffer into ('token'|'thinking', text) events on <think> tags."""
    events = []
    while True:
        tag = "</think>" if in_think else "<think>"
        i = buf.find(tag)
        if i >= 0:
            if i:
                events.append(("thinking" if in_think else "token", buf[:i]))
            buf = buf[i + len(tag):]
            if in_think:
                buf = buf.lstrip("\n")
            in_think = not in_think
            continue
        keep = 0
        for k in range(min(len(tag) - 1, len(buf)), 0, -1):  # hold back a partial tag
            if tag.startswith(buf[-k:]):
                keep = k
                break
        out = buf[:len(buf) - keep]
        if out:
            events.append(("thinking" if in_think else "token", out))
        return events, buf[len(buf) - keep:], in_think


async def stream(messages, *, thinking=False, max_tokens=1100, temperature=0.6):
    """Yields ('token'|'thinking', text) and finally ('done', {usage, ttft, elapsed})."""
    body = _body(messages, stream=True, thinking=thinking, max_tokens=max_tokens, temperature=temperature)
    t0, ttft, usage, buf, in_think = time.time(), None, {}, "", False
    try:
        async with _client.stream("POST", "/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"SGLang error {r.status_code}: {(await r.aread())[:300]!r}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                j = json.loads(data)
                if j.get("usage"):
                    usage = j["usage"]
                for ch in j.get("choices", []):
                    d = ch.get("delta", {})
                    if d.get("reasoning_content"):
                        yield "thinking", d["reasoning_content"]
                    piece = d.get("content") or ""
                    if not piece:
                        continue
                    if ttft is None:
                        ttft = time.time() - t0
                    evs, buf, in_think = _split(buf + piece, in_think)
                    for ev in evs:
                        yield ev
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        raise SGLangDown(f"Can't reach SGLang at {config.SGLANG_URL}. Start it with scripts/start_sglang.sh") from e
    if buf:
        yield ("thinking" if in_think else "token"), buf
    elapsed = time.time() - t0
    STATS.add(elapsed, usage, ttft=ttft, kind="stream")
    yield "done", {"usage": usage, "ttft": ttft or elapsed, "elapsed": elapsed}


async def health():
    try:
        r = await _client.get("/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


async def raw_get(path, timeout=3.0):
    return await _client.get(path, timeout=timeout)
