import sys
import os
import time
from huggingface_hub import hf_hub_download, snapshot_download

def main():
    repo_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Checking / downloading all files for {repo_id} from Hugging Face...", flush=True)
    print("Files will be cached in ~/.cache/huggingface/hub/", flush=True)
    start_time = time.time()
    try:
        path = snapshot_download(
            repo_id=repo_id,
            resume_download=True,
            max_workers=4,
        )
        elapsed = time.time() - start_time
        print(f"SUCCESS: Model downloaded to {path} in {elapsed:.1f}s", flush=True)
    except Exception as e:
        print(f"Error during download: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
