"""A tiny fake of SGLang's OpenAI-compatible API, so you can develop the UI and test the
backend without a GPU:   uvicorn tests.mock_sglang:app --port 30000
It honours json_schema (returns schema-valid JSON), streaming, n>1, usage and a fake
prefix cache keyed on the first message."""
import asyncio
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

app = FastAPI()
SEEN = set()
HITS = {"n": 0, "q": 0}


def fake(schema):
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if t == "object":
        return {k: fake(v) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [fake(schema.get("items", {})) if i else "What are the main findings?" if schema["items"].get("type") == "string" else fake(schema["items"])
                for i in range(max(schema.get("minItems", 1), 3 if schema.get("minItems") else 1))]
    if t == "integer":
        return schema.get("minimum", 5) if schema.get("minimum", 0) == 0 else 8
    if t == "string":
        return "demo"
    return None


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(f'sglang:cache_hit_rate{{model_name="m"}} 0.62\nsglang:gen_throughput{{model_name="m"}} 41.5\n'
                             f'sglang:num_running_reqs{{model_name="m"}} 1\nsglang:num_queue_reqs{{model_name="m"}} 0\n'
                             f'sglang:token_usage{{model_name="m"}} 0.31\n')


@app.post("/v1/chat/completions")
async def chat(req: Request):
    b = await req.json()
    msgs = b["messages"]
    plen = sum(len(m["content"]) for m in msgs) // 4 + 5
    key = msgs[0]["content"]
    cached = int(plen * 0.9) if key in SEEN else 0
    SEEN.add(key)
    await asyncio.sleep(0.02 if cached else 0.15)
    usage = {"prompt_tokens": plen, "completion_tokens": 24, "prompt_tokens_details": {"cached_tokens": cached}}
    schema = (b.get("response_format") or {}).get("json_schema", {}).get("schema")
    if schema:
        texts = [json.dumps(fake(schema))]
    else:
        think = (b.get("chat_template_kwargs") or {}).get("enable_thinking")
        texts = [("<think>Let me work through this.</think>\n\n" if think else "") +
                 "This is a **mock** answer from the fake SGLang server, citing [1]."] * b.get("n", 1)
    if not b.get("stream"):
        return JSONResponse({"choices": [{"index": i, "message": {"role": "assistant", "content": t}} for i, t in enumerate(texts)],
                             "usage": usage})

    async def gen():
        for w in texts[0].replace(" ", " \u0000").split("\u0000"):
            yield "data: " + json.dumps({"choices": [{"delta": {"content": w}}]}) + "\n\n"
            await asyncio.sleep(0.01)
        yield "data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/flush_cache")
async def flush():
    SEEN.clear()
    return {"ok": True}
