"""Deep-research pipeline: planner -> parallel researchers (n=2 branches) -> critic -> writer.

Every call shares ONE identical system message (instructions + retrieved sources). SGLang's
RadixAttention computes that prefix once; the planner, all researchers, all critics and the
writer reuse it. That's why running them in parallel is cheap on a 6 GB GPU.
"""
import asyncio
import time

from . import llm, tools
from .rag import library

SYSTEM = ("You are a careful research analyst working fully offline. Ground every claim in the numbered SOURCES "
          "below and cite them like [1] or [2][3]. If the sources don't cover something, say so plainly "
          "instead of guessing.\n\nSOURCES:\n")
PLAN = {"type": "object", "required": ["sub_questions"], "properties": {
    "sub_questions": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 6}}}
CRITIC = {"type": "object", "required": ["best_index", "score", "issue"], "properties": {
    "best_index": {"type": "integer", "minimum": 0, "maximum": 1},
    "score": {"type": "integer", "minimum": 1, "maximum": 10},
    "issue": {"type": "string"}}}


def fmt_source(s):
    loc = f", p.{s['page']}" if s.get("page") else ""
    return f"[{s['n']}] ({s['title']}{loc}) {s['text']}"


async def run(question, use_web=False, n_sub=4):
    q = asyncio.Queue()

    async def emit(**ev):
        await q.put(ev)

    async def pipeline():
        try:
            await _pipeline(question, use_web, max(2, min(int(n_sub), 6)), emit)
        except Exception as e:
            await emit(type="error", message=str(e))
        finally:
            await q.put(None)

    task = asyncio.create_task(pipeline())
    try:
        while (ev := await q.get()) is not None:
            yield ev
    finally:
        task.cancel()


async def _pipeline(question, use_web, n_sub, emit):
    t0 = time.time()
    agg = dict(calls=0, prompt=0, cached=0, completion=0)

    def track(u):
        d = (u.get("prompt_tokens_details") or {})
        agg["calls"] += 1
        agg["prompt"] += u.get("prompt_tokens") or 0
        agg["cached"] += d.get("cached_tokens") or 0
        agg["completion"] += u.get("completion_tokens") or 0

    sources = [{"n": i + 1, "title": h.get("src", h.get("name", "doc")), "page": h.get("page"), "text": h["text"], "kind": "doc"}
               for i, h in enumerate(library.search(question, 10))]
    await emit(type="sources", sources=sources)
    ctx = "\n".join(fmt_source(s) for s in sources) or "(no documents indexed; rely on general knowledge and say so)"
    system = {"role": "system", "content": SYSTEM + ctx}   # identical for every call below

    # 1. Planner (constrained JSON)
    await emit(type="stage", id="planner", status="running")
    plan, u = await llm.json_complete(
        [system, {"role": "user", "content": f"Research question: {question}\nSplit it into {n_sub} distinct, "
                                              "answerable sub-questions. Return JSON."}],
        PLAN, max_tokens=300, kind="planner")
    track(u)
    subs = [s.strip() for s in plan["sub_questions"] if s.strip()][:n_sub]
    await emit(type="stage", id="planner", status="done", subs=subs)

    # Optional web evidence, numbered globally so citations stay consistent
    extra = {i: [] for i in range(len(subs))}
    if use_web:
        found = await asyncio.gather(*[tools.web_search(s, 3) for s in subs], return_exceptions=True)
        for i, res in enumerate(found):
            if isinstance(res, Exception):
                continue
            for r in res:
                s = {"n": len(sources) + 1, "title": r["title"] or r["url"], "page": None,
                     "text": r["text"], "kind": "web", "url": r["url"]}
                sources.append(s)
                extra[i].append(s)
        await emit(type="sources", sources=sources)

    # 2+3. Per sub-question: 2 candidate answers (one request, n=2) then a critic pass
    critic_left = len(subs)
    await emit(type="stage", id="critic", status="idle")

    async def chain(i, sub):
        nonlocal critic_left
        await emit(type="stage", id=f"r{i}", status="running", question=sub)
        web = ("\nExtra sources:\n" + "\n".join(fmt_source(s) for s in extra[i])) if extra[i] else ""
        ask = (f"Sub-question: {sub}{web}\nAnswer in 3-6 sentences with [n] citations. "
               "If the sources are insufficient, say what is missing.")
        cands, u = await llm.complete([system, {"role": "user", "content": ask}], n=2, temperature=0.8,
                                      max_tokens=350, kind="researcher")
        track(u)
        cands = (cands + cands)[:2]
        await emit(type="stage", id=f"r{i}", status="critic")
        await emit(type="stage", id="critic", status="running")
        judge = (f"Sub-question: {sub}\n\nCandidate 0:\n{cands[0]}\n\nCandidate 1:\n{cands[1]}\n\n"
                 "Pick the candidate that is better grounded in the sources, with correct citations and no "
                 "invented facts. Give best_index, a score 1-10 for it, and one short sentence naming its main "
                 "weakness (or 'none'). Return JSON.")
        v, u = await llm.json_complete([system, {"role": "user", "content": judge}], CRITIC, max_tokens=120, kind="critic")
        track(u)
        best = cands[min(max(int(v["best_index"]), 0), 1)]
        critic_left -= 1
        await emit(type="stage", id=f"r{i}", status="done", score=v["score"], issue=v["issue"], answer=best)
        if critic_left == 0:
            await emit(type="stage", id="critic", status="done")
        return {"q": sub, "answer": best, "score": v["score"], "issue": v["issue"]}

    findings = await asyncio.gather(*[chain(i, s) for i, s in enumerate(subs)])

    # 4. Writer (streamed)
    await emit(type="stage", id="writer", status="running")
    notes = "\n\n".join(f"### {f['q']}\n{f['answer']}\n(critic score {f['score']}/10; weakness: {f['issue']})"
                        for f in findings)
    prompt = (f"Research question: {question}\n\nVerified findings from your team:\n{notes}\n\n"
              "Write the final report in Markdown: a short summary first, then sections for each finding, "
              "keeping the [n] citations. End with a 'Limits' section listing gaps or low-scoring areas.")
    async for kind, val in llm.stream([system, {"role": "user", "content": prompt}], max_tokens=1400, temperature=0.4):
        if kind == "done":
            track(val["usage"])
        elif kind == "token":
            await emit(type="token", text=val)
    await emit(type="stage", id="writer", status="done")
    await emit(type="done", sources=sources, stats=dict(
        elapsed=time.time() - t0, calls=agg["calls"], prompt_tokens=agg["prompt"], cached_tokens=agg["cached"],
        completion_tokens=agg["completion"],
        saved_pct=100 * agg["cached"] / agg["prompt"] if agg["prompt"] else 0))
