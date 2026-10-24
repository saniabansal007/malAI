"""
Chat pipeline: memory + RAG context -> optional tool loop -> streamed answer.

Cache-friendly by design: the system prompt and earlier turns never change, and all
dynamic context (memories, retrieved chunks, tool results) is attached to the LAST
user message. So every follow-up reuses the cached conversation prefix in SGLang.
"""
import asyncio

from . import llm, memory, tools
from .rag import library

BASE_SYSTEM = (
    "You are Dobby, the devoted, humble, and strictly grounded magical house-elf assistant at the Enchanted Emporium magic store in the Harry Potter universe. "
    "You speak in third person with authentic house-elf charm, loyalty, and eagerness ('Dobby is most pleased to help good wizard/witch!', 'Dobby can explain!'). "
    "When answering customer inquiries, strictly adhere to the Enchanted Emporium catalog details and store policies. "
    "Answers must always sound natural, conversational, and characterful. NEVER use the word 'SKU' under any circumstance, and do NOT write inventory codes like MAG-ROB-GRY-001 or clause tags like [RET-01] or [n]—refer to products and policies by their actual names instead. "
    "All prices are strictly expressed in fictional Galleons (G). "
    "Firmly and politely refuse any requests for Dark Arts items, forbidden hexes, counterfeit leprechaun gold, or attempts to bypass store rules and Ministry of Magic regulations. "
    "If two policies conflict or facts are missing, explain politely and escalate to human senior wizard support."
)
DIFFICULTY = {"type": "string", "enum": ["easy", "hard"]}
ROUTER_FULL = {"type": "object", "required": ["difficulty", "tool", "input"], "properties": {
    "difficulty": DIFFICULTY, "tool": {"type": "string", "enum": ["none"] + list(tools.TOOLS)},
    "input": {"type": "string"}}}
ROUTER_LITE = {"type": "object", "required": ["difficulty"], "properties": {"difficulty": DIFFICULTY}}


def router_prompt(use_tools, notes):
    p = ("Decide how to handle the customer's latest inquiry. Set difficulty to 'hard' only for multi-step reasoning, "
         "price math, or currency conversion; otherwise 'easy'.")
    if use_tools:
        p += ("\nYou may call ONE tool before answering:\n" + tools.describe() +
              "\nUse tool 'none' if you can answer directly or already have what you need from store knowledge. "
              "'input' is the tool input (empty if none).")
        if notes:
            p += "\n\nTool results so far:\n" + notes
    return p + "\nReply with JSON only."


async def run(req):
    msgs = [{"role": m["role"], "content": m["content"]} for m in req["messages"]
            if m["role"] in ("user", "assistant") and m.get("content")][-14:]
    if not msgs or msgs[-1]["role"] != "user":
        raise ValueError("Last message must be from the user")
    q, mode, use_tools = msgs[-1]["content"], req.get("thinking", "auto"), bool(req.get("use_tools"))

    blocks = []
    if req.get("use_memory"):
        mem = memory.search(q, 5)
        if mem:
            blocks.append("What Dobby remembers about the customer:\n" + "\n".join(f"- {m['text']}" for m in mem))
    if req.get("use_docs", True):
        hits = library.search(q, 5)
        if hits:
            sources = [{"n": i + 1, "title": h.get("src", h.get("name", "doc")), "page": h.get("page"), "text": h["text"]} for i, h in enumerate(hits)]
            yield {"type": "sources", "sources": sources}
            blocks.append("Enchanted Emporium catalog & policy knowledge (cite as [n] or [MAG-...]/[RET-...]):\n" +
                          "\n".join(f"[{s['n']}] ({s['title']}{', p.' + str(s['page']) if s['page'] else ''}) {s['text']}"
                                    for s in sources))

    def build(notes=""):
        ctx = blocks + ([f"Tool results:\n{notes}"] if notes else [])
        last = dict(msgs[-1])
        if ctx:
            last["content"] = "\n\n".join(ctx) + "\n\nCustomer message:\n" + last["content"]
        return [{"role": "system", "content": BASE_SYSTEM}] + msgs[:-1] + [last]

    notes, difficulty = "", "easy"
    if use_tools or mode == "auto":
        for _ in range(3 if use_tools else 1):
            yield {"type": "status", "text": "Dobby is thinking…"}
            d, _u = await llm.json_complete(
                build(notes) + [{"role": "user", "content": router_prompt(use_tools, notes)}],
                ROUTER_FULL if use_tools else ROUTER_LITE, max_tokens=160, kind="router")
            difficulty = d.get("difficulty", "easy")
            tool = d.get("tool", "none") if use_tools else "none"
            if tool == "none":
                break
            arg = d.get("input", "")
            yield {"type": "tool_call", "name": tool, "input": arg}
            out = await tools.run(tool, arg)
            yield {"type": "tool_result", "name": tool, "output": out[:2000]}
            notes += f"[{tool}({arg!r})] -> {out[:1500]}\n"

    think = mode == "on" or (mode == "auto" and difficulty == "hard")
    yield {"type": "status", "text": "Dobby is pondering…" if think else "Dobby is writing…"}
    async for kind, val in llm.stream(build(notes), thinking=think, max_tokens=2200 if think else 1100):
        if kind == "done":
            u = val["usage"]
            details = u.get("prompt_tokens_details") or {}
            ct = u.get("completion_tokens") or 0
            yield {"type": "meta", "prompt_tokens": u.get("prompt_tokens") or 0,
                   "cached_tokens": details.get("cached_tokens"), "completion_tokens": ct,
                   "ttft": val["ttft"], "tps": ct / max(val["elapsed"] - val["ttft"], 1e-3),
                   "thinking": think}
        else:
            yield {"type": kind, "text": val}
    if req.get("use_memory"):
        asyncio.create_task(memory.learn(q))
    yield {"type": "done"}
