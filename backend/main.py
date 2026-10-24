import asyncio, json, os, re, sys, time
from pathlib import Path
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

SG = os.getenv("SGLANG_URL", "http://127.0.0.1:30000")
MODEL = os.getenv(
    "MODEL",
    "models/shopping_model_merged"
    if Path("models/shopping_model_merged").exists()
    else (
        "models/banking_model_merged"
        if Path("models/banking_model_merged").exists()
        else "Qwen/Qwen2.5-1.5B-Instruct"
    )
)

Path("data").mkdir(exist_ok=True)
app = FastAPI(title="malAI")
http = httpx.AsyncClient(timeout=600)
live = {"tokens": 0, "tps": 0.0}

def sse(o):
    return f"data: {json.dumps(o)}\n\n"

def system_prompt():
    return (
        "You are malAI, an intelligent, helpful, and friendly AI Shopping Assistant for NovaTech Electronics. "
        "Help customers discover products, compare specs and prices accurately, highlight prices in bold, "
        "and answer store policy questions (returns, shipping, warranties, price matches) with clear customer service guidance."
    )

def body(messages, **kw):
    return {
        "model": MODEL,
        "messages": messages,
        "temperature": kw.pop("temperature", 0.4),
        "chat_template_kwargs": {"enable_thinking": False},
        **kw
    }

async def complete(messages, schema=None, max_tokens=500):
    b = body(messages, max_tokens=max_tokens)
    if schema:
        b["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    t = time.time()
    j = (await http.post(f"{SG}/v1/chat/completions", json=b)).json()
    n = j["usage"]["completion_tokens"]
    live["tokens"] += n
    live["tps"] = n / max(time.time() - t, 1e-3)
    return j["choices"][0]["message"]["content"]

async def stream(messages, max_tokens=1200):
    t = time.time()
    n = 0
    async with http.stream(
        "POST",
        f"{SG}/v1/chat/completions",
        json=body(messages, max_tokens=max_tokens, stream=True)
    ) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                if choices and "delta" in choices[0]:
                    d = choices[0]["delta"].get("content")
                    if d:
                        n += 1
                        live["tokens"] += 1
                        live["tps"] = n / max(time.time() - t, 1e-3)
                        yield d

class ChatReq(BaseModel):
    messages: list
    docs: bool = False

@app.post("/api/chat")
async def chat(req: ChatReq):
    async def gen():
        msgs = [{"role": "system", "content": system_prompt()}] + req.messages
        async for tkn in stream(msgs):
            yield sse({"t": "tok", "v": tkn})
        yield sse({"t": "end"})
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/api/metrics")
async def metrics():
    out = {"tokens": live["tokens"], "tps": round(live["tps"], 1), "up": False}
    try:
        txt = (await http.get(f"{SG}/metrics", timeout=2)).text
        out["up"] = True
        for key, name in [
            ("cache_hit_rate", "cache"),
            ("gen_throughput", "gen_tps"),
            ("num_running_reqs", "running"),
            ("num_queue_reqs", "queued")
        ]:
            m = re.search(rf"sglang:{key}\S*\s+([\d.eE+-]+)", txt)
            if m:
                out[name] = round(float(m.group(1)), 3)
    except Exception:
        pass
    return out

@app.get("/")
async def root():
    p = Path("frontend/index.html")
    content = p.read_text(encoding="utf-8") if p.exists() else "Not found"
    return HTMLResponse(content, headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    })

app.mount("/", StaticFiles(directory="frontend", html=True), name="ui")
