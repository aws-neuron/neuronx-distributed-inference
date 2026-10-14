import torch
from diffusers import QwenImageEditPlusPipeline

CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"

if __name__ == "__main__":
    print(f"Downloading {MODEL_ID} to {CACHE_DIR}...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR
    )
    print("Model downloaded successfully!")
