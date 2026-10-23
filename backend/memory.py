"""Long-term memory: durable facts about the user, extracted with constrained JSON."""
import json
import time
import uuid

from . import config, llm, rag

FILE = config.DATA / "memory.json"
SCHEMA = {"type": "object",
          "properties": {"facts": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
          "required": ["facts"]}


def _load():
    return json.loads(FILE.read_text()) if FILE.exists() else []


def _save(items):
    FILE.write_text(json.dumps(items, indent=1))


def list_all():
    return _load()


def add(text, source="manual"):
    items = _load()
    if any(i["text"].lower() == text.lower() for i in items):
        return None
    item = {"id": uuid.uuid4().hex[:8], "text": text.strip(), "source": source, "t": time.time()}
    items.append(item)
    _save(items)
    return item


def delete(mid=None):
    _save([] if mid is None else [i for i in _load() if i["id"] != mid])


def search(query, k=5):
    items = _load()
    if not items:
        return []
    texts = [i["text"] for i in items]
    hits = rank_texts(query, texts, k)
    return [items[i] for i, _ in hits]


def rank_texts(query, texts, k):
    embs = rag.embed(texts)
    return rag.rank(query, texts, k, embs)


async def learn(user_message):
    """Background task: pull lasting facts/preferences out of what the user said."""
    try:
        obj, _ = await llm.json_complete(
            [{"role": "system", "content": "Extract durable facts or preferences the USER states about themselves "
                                           "(name, job, projects, tools, preferences). Ignore questions and "
                                           "temporary details. Return an empty list if there are none."},
             {"role": "user", "content": user_message[:1500]}],
            SCHEMA, max_tokens=120, kind="memory")
        for f in obj.get("facts", []):
            if 3 < len(f) < 200:
                add(f, source="auto")
    except Exception as e:
        print("[memory] learn failed:", e)
