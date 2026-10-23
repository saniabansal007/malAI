from vllm.config import ModelConfig

try:
    c = ModelConfig(
        model="Qwen/Qwen3-4B-AWQ",
        quantization="awq",
        tokenizer="Qwen/Qwen3-4B-AWQ",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="auto",
        seed=0,
        skip_tokenizer_init=True,
    )
    print("SUCCESS: ModelConfig created:", c.model, c.tokenizer_mode)
except Exception as e:
    import traceback
    traceback.print_exc()
