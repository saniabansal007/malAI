import asyncio, io, json, os, re, sys, time
from pathlib import Path
import httpx
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from rag import Index
except ImportError:
    from backend.rag import Index


SG = os.getenv("SGLANG_URL", "http://127.0.0.1:30000")
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
Path("data").mkdir(exist_ok=True)
MEM = Path("data/memory.json")
app = FastAPI(title="LocalMind")
http = httpx.AsyncClient(timeout=600)
idx = Index()
live = {"tokens": 0, "tps": 0.0}

def sse(o): return f"data: {json.dumps(o)}\n\n"
def ctx_block(chunks): return "\n\n".join(f"[{i+1}] ({c['src']}) {c['text']}" for i, c in enumerate(chunks))
def memory(): return json.loads(MEM.read_text()) if MEM.exists() else []

# Stable text goes FIRST in every prompt so SGLang's RadixAttention can reuse it.
def system_prompt(chunks=None):
    s = "You are LocalMind, a helpful private assistant running fully offline."
    if memory(): s += "\nKnown facts about the user:\n- " + "\n- ".join(memory())
    if chunks: s += "\n\nSources (cite as [n]):\n" + ctx_block(chunks)
    return s

def body(messages, **kw):
    return {"model": MODEL, "messages": messages, "temperature": kw.pop("temperature", 0.4),
            "chat_template_kwargs": {"enable_thinking": False}, **kw}

async def complete(messages, schema=None, max_tokens=500):
    b = body(messages, max_tokens=max_tokens)
    if schema:
        b["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    t = time.time()
    j = (await http.post(f"{SG}/v1/chat/completions", json=b)).json()
    n = j["usage"]["completion_tokens"]; live["tokens"] += n
    live["tps"] = n / max(time.time() - t, 1e-3)
    return j["choices"][0]["message"]["content"]

async def stream(messages, max_tokens=1200):
    t = time.time(); n = 0
    async with http.stream("POST", f"{SG}/v1/chat/completions",
                           json=body(messages, max_tokens=max_tokens, stream=True)) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                if choices and "delta" in choices[0]:
                    d = choices[0]["delta"].get("content")
                    if d:
                        n += 1; live["tokens"] += 1
                        live["tps"] = n / max(time.time() - t, 1e-3)
                        yield d


class ChatReq(BaseModel):
    messages: list
    docs: bool = False

@app.post("/api/chat")
async def chat(req: ChatReq):
    last = req.messages[-1]["content"]
    if last.lower().startswith("remember "):
        MEM.write_text(json.dumps(memory() + [last[9:].strip()]))
    chunks = idx.search(last, 6) if req.docs else None
    async def gen():
        if chunks: yield sse({"t": "sources", "sources": chunks})
        msgs = [{"role": "system", "content": system_prompt(chunks)}] + req.messages
        async for tkn in stream(msgs): yield sse({"t": "tok", "v": tkn})
        yield sse({"t": "end"})
    return StreamingResponse(gen(), media_type="text/event-stream")

PLAN = {"type": "object", "properties": {"questions": {"type": "array", "minItems": 3, "maxItems": 5,
        "items": {"type": "string"}}}, "required": ["questions"]}
CRIT = {"type": "object", "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 10},
        "issue": {"type": "string"}}, "required": ["score", "issue"]}

class ResReq(BaseModel):
    question: str

@app.post("/api/research")
async def research(req: ResReq):
    async def gen():
        chunks = idx.search(req.question, 10)
        sys = {"role": "system", "content": system_prompt(chunks)}  # identical prefix for every agent
        yield sse({"t": "stage", "v": "plan"})
        plan = json.loads(await complete([sys, {"role": "user", "content":
              f"Break this into 3-5 independent sub-questions: {req.question}"}], PLAN))["questions"]
        yield sse({"t": "plan", "questions": plan})
        q = asyncio.Queue()
        async def worker(i, sq):
            await q.put({"t": "agent", "i": i, "state": "running"})
            ans = await complete([sys, {"role": "user", "content": sq}], max_tokens=350)
            c = json.loads(await complete([sys, {"role": "user", "content":
                f"Question: {sq}\nAnswer: {ans}\nScore the answer 1-10 for accuracy and grounding in sources."}], CRIT, 120))
            if c["score"] < 6:  # critic loop: one revision pass
                ans = await complete([sys, {"role": "user", "content":
                    f"{sq}\nFix this issue in your answer: {c['issue']}\nPrevious: {ans}"}], max_tokens=350)
            await q.put({"t": "agent", "i": i, "state": "done", "score": c["score"], "answer": ans})
            return sq, ans
        task = asyncio.gather(*[worker(i, s) for i, s in enumerate(plan)])
        while not task.done() or not q.empty():
            try: yield sse(await asyncio.wait_for(q.get(), 0.2))
            except asyncio.TimeoutError: pass
        notes = "\n\n".join(f"Q: {a}\nA: {b}" for a, b in task.result())
        yield sse({"t": "stage", "v": "write"})
        async for tkn in stream([sys, {"role": "user", "content":
              f"Write a concise cited report answering: {req.question}\n\nResearch notes:\n{notes}"}]):
            yield sse({"t": "tok", "v": tkn})
        yield sse({"t": "end"})
    return StreamingResponse(gen(), media_type="text/event-stream")

class ExtReq(BaseModel):
    text: str
    fields: list[str]

@app.post("/api/extract")  # schema-guaranteed JSON via constrained decoding
async def extract(r: ExtReq):
    schema = {"type": "object", "properties": {f: {"type": "string"} for f in r.fields}, "required": r.fields}
    out = await complete([{"role": "user", "content": f"Extract the fields from the text.\n\n{r.text}"}], schema)
    return json.loads(out)

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    if file.filename.lower().endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)
    else: text = raw.decode("utf-8", "ignore")
    idx.add(file.filename, text)
    return {"sources": idx.sources(), "chunks": len(idx.chunks)}

@app.get("/api/docs")
async def docs(): return {"sources": idx.sources(), "chunks": len(idx.chunks)}
@app.delete("/api/docs")
async def clear(): idx.clear(); return {"sources": []}
@app.get("/api/memory")
async def get_mem(): return memory()
@app.delete("/api/memory")
async def del_mem(): MEM.write_text("[]"); return []

@app.get("/api/metrics")
async def metrics():
    out = {"tokens": live["tokens"], "tps": round(live["tps"], 1), "up": False}
    try:
        txt = (await http.get(f"{SG}/metrics", timeout=2)).text
        out["up"] = True
        for key, name in [("cache_hit_rate", "cache"), ("gen_throughput", "gen_tps"),
                          ("num_running_reqs", "running"), ("num_queue_reqs", "queued")]:
            m = re.search(rf"sglang:{key}\S*\s+([\d.eE+-]+)", txt)
            if m: out[name] = round(float(m.group(1)), 3)
    except Exception: pass
    return out

@app.post("/api/benchmark")
async def benchmark():
    """Same 8 requests sharing a long prefix: cache flushed before every request vs. cache kept."""
    base = " ".join(c["text"] for c in idx.chunks[:25]) or ("The quick brown fox jumps over the lazy dog. " * 300)
    prefix = {"role": "system", "content": "Document:\n" + base[:9000]}
    qs = [f"In one sentence, give takeaway number {i} from the document." for i in range(8)]
    async def run(flush_each):
        await http.post(f"{SG}/flush_cache"); t = time.time()
        for qn in qs:
            if flush_each: await http.post(f"{SG}/flush_cache")
            await complete([prefix, {"role": "user", "content": qn}], max_tokens=24)
        return round(time.time() - t, 2)
    nocache, cached = await run(True), await run(False)
    return {"no_cache_s": nocache, "cache_s": cached, "speedup": round(nocache / max(cached, 1e-3), 2)}

@app.get("/")
async def root():
    from fastapi.responses import HTMLResponse
    p = Path("frontend/index.html")
    content = p.read_text(encoding="utf-8") if p.exists() else "Not found"
    return HTMLResponse(content, headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    })

app.mount("/", StaticFiles(directory="frontend", html=True), name="ui")
