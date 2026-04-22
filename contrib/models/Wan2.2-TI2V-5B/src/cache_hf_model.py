import torch
from diffusers import AutoencoderKLWan, WanPipeline

CACHE_DIR = "/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"
MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DTYPE = torch.bfloat16
vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32, cache_dir=CACHE_DIR)
pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=DTYPE, cache_dir=CACHE_DIR)
