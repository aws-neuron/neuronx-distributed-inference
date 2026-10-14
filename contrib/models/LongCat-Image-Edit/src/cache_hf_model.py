import subprocess
import sys
import torch

CACHE_DIR = "/opt/dlami/nvme/longcat_hf_cache"
MODEL_ID = "meituan-longcat/LongCat-Image-Edit"

if __name__ == "__main__":
    # Install diffusers from source (LongCat classes are in latest diffusers)
    print("Installing diffusers from source (required for LongCat classes)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/huggingface/diffusers",
        "--quiet",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "accelerate", "sentencepiece", "qwen-vl-utils", "Pillow",
        "--quiet",
    ])

    print(f"\nDownloading {MODEL_ID} to {CACHE_DIR}...")
    from diffusers import LongCatImageEditPipeline
    pipe = LongCatImageEditPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
    )
    print("Model downloaded successfully!")
    print(f"  Transformer type: {type(pipe.transformer).__name__}")
    print(f"  Text encoder type: {type(pipe.text_encoder).__name__}")
    print(f"  VAE type: {type(pipe.vae).__name__}")
