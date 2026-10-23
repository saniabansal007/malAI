# LocalMind

A private, general-purpose AI workspace that runs entirely on your own GPU, built around **SGLang**.
One engine, several modes: chat, chat over your documents, tool use, deep research with parallel agents,
guaranteed-valid JSON extraction, long-term memory, and a live view of what SGLang is doing underneath.

Target hardware: laptop **RTX 4050 (6 GB)**, Windows + WSL2 (native Linux works too).
Default model: `Qwen/Qwen3-4B-AWQ` (about 2.7 GB, 4-bit).

## Quick start

```bash
# inside WSL2 Ubuntu (or Linux), in this folder
./scripts/setup.sh        # once. LIGHT=1 ./scripts/setup.sh skips the embedding model
./run.sh                  # starts SGLang, waits for it, then serves the app
# open http://localhost:8000
```

No GPU handy (or working on the UI)? `./run_mock.sh` starts a fake SGLang so you can click through everything.

**WSL2 prerequisites:** install the NVIDIA driver on Windows (not inside Linux), then `nvidia-smi` should work inside WSL.
Keep the project inside the WSL filesystem (`~/localmind`), not `/mnt/c/...`, or model loading is slow.

## How the SGLang features are used

| Feature | Where | Why it matters |
|---|---|---|
| **RadixAttention prefix cache** | Every mode. The system prompt and chat history are never rewritten; dynamic context goes into the *last* user message (`backend/chat.py`). In research, all agents share one identical system message holding the sources (`backend/agents.py`). | Follow-up turns and parallel agents reuse computed KV instead of recomputing it. This is what makes a 6 GB GPU viable. |
| **Constrained decoding (JSON schema)** | Tool router, research planner and critic, memory extraction, Extract mode (`response_format: json_schema`) | Output is valid by construction, so a 4B model can run an agent loop without parse failures. |
| **Parallel requests + `n=2` branching** | Research: sub-questions run concurrently, each produces two drafts in one request | Continuous batching plus a shared prefix: more agents, little extra cost. |
| **Per-request reasoning switch** | A constrained "difficulty" decision turns Qwen3 thinking on only for hard questions (`chat_template_kwargs.enable_thinking`) | Fast answers for easy things, deliberate ones when needed. |
| **Streaming OpenAI-compatible API** | Chat and report streaming | Standard clients just work. |
| **Metrics + cache report** | Live panel, cache tape under each answer (`--enable-metrics`, `--enable-cache-report`) | Makes the speedup visible in the demo. |
| **Native frontend language (`fork`)** | `scripts/sgl_frontend_demo.py` | Shows the same shared-prefix idea in SGLang's own DSL. |

## Modes

- **Chat**: streaming, history kept in your browser. Toggles: *Use my documents* (RAG with clickable `[n]` citations), *Tools* (calculator, Python, web search, workspace files, date/time), *Memory*, and *Reasoning* (auto/always/never).
- **Deep research**: planner, parallel researchers, critic, writer, drawn as a live graph. The footer reports how many prompt tokens came from the cache.
- **Extract**: paste text and a JSON Schema, get schema-valid JSON back.
- **Library**: drop PDF/Markdown/text files. Embeddings run on CPU (`bge-small`), so the GPU stays free for SGLang.
- **Memory**: facts the model learns about you; view and delete them.
- **Benchmark**: unique prefixes vs one shared prefix on the same server, with a speedup chart.

## Benchmark protocol for your demo

1. `./run.sh`, then Benchmark tab (or `python -m backend.benchmark --label sglang-radix`).
2. Restart SGLang with the radix cache off: `EXTRA="--disable-radix-cache" ./scripts/start_sglang.sh`, run `python -m backend.benchmark --label sglang-no-radix`.
3. Optional baseline: Ollama with the same model, `python -m backend.benchmark --url http://127.0.0.1:11434 --model qwen3:4b --label ollama`.
4. Results accumulate in `docs/benchmark_results.json`. Put the chart in your slides. Report your own measured numbers, not anyone else's.

## Configuration

Environment variables: `SGLANG_URL`, `MODEL`, `EMBED_MODEL`, `MAX_PARALLEL` (default 4), `ENABLE_PYTHON_TOOL` (default 1).
Launch variables for `start_sglang.sh`: `MODEL`, `CTX`, `MEM`, `PORT`, `EXTRA`.

## Troubleshooting

- **CUDA out of memory at launch**: `MEM=0.70 CTX=4096 ./scripts/start_sglang.sh`, or `MODEL=Qwen/Qwen3-1.7B`.
- **Unknown argument errors**: SGLang's flags change between releases; compare with `python -m sglang.launch_server --help` and edit `scripts/start_sglang.sh`.
- **FlashInfer / attention backend errors**: add `EXTRA="--attention-backend triton"`.
- **Cache numbers show "unavailable"**: the server needs `--enable-cache-report` (already in the script).
- **Research feels slow**: lower sub-questions to 3, or set `MAX_PARALLEL=2`.
- **Python tool**: runs in a subprocess with an 8 s timeout. It is *not* a hardened sandbox, so keep the app on localhost, or set `ENABLE_PYTHON_TOOL=0`.

## Layout

```
backend/    FastAPI app: chat.py (pipeline), agents.py (research), rag.py, memory.py, tools.py, llm.py (SGLang client), benchmark.py
frontend/   index.html, css/, js/  (no build step, no external requests, works offline)
scripts/    setup.sh, start_sglang.sh, sgl_frontend_demo.py
tests/      mock_sglang.py (fake server for GPU-free development)
data/       your library index, memory, and the tools workspace
```
