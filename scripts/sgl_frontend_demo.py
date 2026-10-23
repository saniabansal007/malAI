"""SGLang's native frontend language: ONE long document, several questions answered in
parallel with fork(). The document prefix is computed once (RadixAttention) and every fork
reuses it. Run with the SGLang server up:  python scripts/sgl_frontend_demo.py
Needs the sglang venv:  source .venv-sglang/bin/activate
(Written against the sglang frontend API; if a call name differs in your version, see the docs.)
"""
import sys
import time

import sglang as sgl

DOC = open(sys.argv[1]).read() if len(sys.argv) > 1 else "SGLang is a fast serving framework. " * 200
QUESTIONS = ["Summarise the document in one sentence.", "List two key terms.", "What is the tone of the text?"]


@sgl.function
def ask_many(s, doc, questions):
    s += sgl.system("You answer questions about the document concisely.")
    s += sgl.user("Document:\n" + doc)
    forks = s.fork(len(questions))
    for f, q in zip(forks, questions):
        f += sgl.user(q)
        f += sgl.assistant(sgl.gen("answer", max_tokens=96))
    s["answers"] = [f["answer"] for f in forks]


if __name__ == "__main__":
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://127.0.0.1:30000"))
    t = time.time()
    state = ask_many.run(doc=DOC, questions=QUESTIONS)
    for q, a in zip(QUESTIONS, state["answers"]):
        print(f"Q: {q}\nA: {a.strip()}\n")
    print(f"{time.time() - t:.2f}s")
