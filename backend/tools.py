"""Tools the model can call. The model picks one via a constrained-JSON router step,
so tool calls are always well-formed."""
import ast
import asyncio
import datetime
import math
import operator
import sys
import tempfile

from . import config

TOOLS = {
    "calculator": "Evaluate a math expression, e.g. '(12*7)/3 + sqrt(16)'. Input: the expression.",
    "python": "Run a short Python 3 script and return its printed output. Input: the code.",
    "web_search": "Search the web for current information. Input: a search query.",
    "read_file": "Read a text file from the workspace folder (use '.' to list files). Input: file name.",
    "datetime": "Get the current local date and time. Input: empty.",
}
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}


def describe():
    return "\n".join(f"- {k}: {v}" for k, v in TOOLS.items())


def _calc(expr):
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 1000:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name) and n.id in _NAMES and not callable(_NAMES[n.id]):
            return _NAMES[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and callable(_NAMES.get(n.func.id)):
            return _NAMES[n.func.id](*[ev(a) for a in n.args])
        raise ValueError("unsupported expression")
    return str(ev(ast.parse(expr.strip(), mode="eval").body))


async def _python(code):
    if not config.ENABLE_PYTHON_TOOL:
        return "The Python tool is disabled (ENABLE_PYTHON_TOOL=0)."
    # NOTE: a subprocess with a timeout, not a hardened sandbox. Local use only.
    with tempfile.TemporaryDirectory() as d:
        p = await asyncio.create_subprocess_exec(sys.executable, "-I", "-c", code, cwd=d,
                                                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(p.communicate(), timeout=8)
        except asyncio.TimeoutError:
            p.kill()
            return "Timed out after 8 seconds."
    return out.decode(errors="replace")[:3000] or "(no output)"


def _read(name):
    name = name.strip()
    if name in ("", ".", "/"):
        files = sorted(f.name for f in config.WORKSPACE.iterdir() if f.is_file())
        return "Files: " + (", ".join(files) or "(workspace is empty)")
    path = (config.WORKSPACE / name).resolve()
    if config.WORKSPACE.resolve() not in path.parents or not path.is_file():
        return f"File not found in workspace: {name}"
    return path.read_text(errors="replace")[:6000]


def _ddg(query, n=4):
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    with DDGS() as d:
        return [{"title": r.get("title", ""), "url": r.get("href", ""), "text": r.get("body", "")}
                for r in d.text(query, max_results=n)]


async def web_search(query, n=4):
    return await asyncio.to_thread(_ddg, query, n)


async def run(name, arg):
    try:
        if name == "calculator":
            return _calc(arg)
        if name == "python":
            return await _python(arg)
        if name == "read_file":
            return _read(arg)
        if name == "datetime":
            return datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M")
        if name == "web_search":
            res = await web_search(arg)
            return "\n".join(f"{r['title']} ({r['url']}): {r['text']}" for r in res) or "No results."
        return f"Unknown tool {name}"
    except Exception as e:
        return f"{name} failed: {e}"
