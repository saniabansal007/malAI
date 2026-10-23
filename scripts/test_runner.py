import time
import os
import torch

os.environ["SGLANG_IS_FLASHINFER_AVAILABLE"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

print("[1] Importing SGLang server args & model runner...", flush=True)
from sglang.srt.server_args import prepare_server_args, ServerArgs
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.configs.model_config import ModelConfig

cmd = [
    "--model-path", "Qwen/Qwen3-4B-AWQ",
    "--host", "127.0.0.1",
    "--port", "30000",
    "--context-length", "8192",
    "--mem-fraction-static", "0.75",
    "--attention-backend", "triton",
    "--disable-cuda-graph",
]

server_args = prepare_server_args(cmd)
model_config = ModelConfig(
    path=server_args.model_path,
    model_override_args=server_args.json_model_override_args,
)

print(f"[2] Initializing ModelRunner for {server_args.model_path}...", flush=True)
t0 = time.time()
runner = ModelRunner(
    model_config=model_config,
    mem_fraction_static=server_args.mem_fraction_static,
    gpu_id=0,
    tp_rank=0,
    tp_size=1,
    nccl_port=30368,
    server_args=server_args,
)
print(f"[3] ModelRunner initialized in {time.time() - t0:.2f}s!", flush=True)
print("TEST SUCCESS: Model loaded and ready!", flush=True)
