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
        user_msg = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
        u_lower = user_msg.lower()
        if "laptop" in u_lower:
            reply = (
                "For a lightweight and portable laptop under $1,000, I recommend the **NovaBook Slim 14**, currently on sale for **$899.99**.\n\n"
                "- **Processor:** AMD Ryzen 7 8840HS (8 cores / 16 threads)\n"
                "- **Memory & Storage:** 16GB LPDDR5X RAM, 512GB PCIe 4.0 NVMe SSD\n"
                "- **Display:** 14-inch 2.8K OLED (120Hz, 100% DCI-P3 color gamut)\n"
                "- **Battery Life:** Up to 18 hours\n"
                "- **Weight:** 1.25 kg ultra-light aluminum unibody\n\n"
                "It is in stock and qualifies for **Free Standard Ground Shipping**!"
            )
        elif "return" in u_lower:
            reply = (
                "Here is an overview of our customer-friendly **Store Return Policy**:\n\n"
                "1. **30-Day Window:** Unopened items in original factory packaging qualify for a **100% full refund**.\n"
                "2. **Open-Box Restocking Fee:** Opened items in like-new condition are subject to a **15% restocking fee**.\n"
                "3. **Hygiene Exclusion:** Opened in-ear headphones (such as EchoPods) cannot be returned due to sanitary hygiene regulations.\n"
                "4. **Defective on Arrival (DOA):** Claims filed within 48 hours receive immediate free replacement with prepaid return shipping."
            )
        elif "compare" in u_lower or "echopods" in u_lower:
            reply = (
                "Here is a side-by-side comparison between **EchoPods Pro ANC** and **EchoBuds Sport**:\n\n"
                "| Specification | EchoPods Pro ANC | EchoBuds Sport |\n"
                "| :--- | :--- | :--- |\n"
                "| **Price** | **$249.99** | **$89.99** |\n"
                "| **Noise Cancellation** | Hybrid ANC (up to 45dB) | Passive Noise Isolation |\n"
                "| **Battery Life** | 8h earbuds / 30h with case | 9h earbuds / 32h with case |\n"
                "| **Water Resistance** | IPX5 (Sweat/Splash) | IP68 (Waterproof & Dustproof) |\n"
                "| **Drivers** | 11mm Beryllium drivers | 10mm Graphene drivers |\n\n"
                "Both products come with our standard **1-Year Manufacturer Warranty** and free shipping!"
            )
        else:
            reply = (
                "👋 Hello! I am your **NovaTech AI Shopping Assistant**.\n\n"
                "I can help you explore our electronics catalog, check product specifications and stock, "
                "or explain store policies including **Free Ground Shipping on orders over $50**, our **30-Day Return Policy**, "
                "and **14-Day Price Match Guarantees**.\n\n"
                "What kind of product or budget are you looking for today?"
            )
        texts = [("<think>Thinking...</think>\n\n" if think else "") + reply] * b.get("n", 1)
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
