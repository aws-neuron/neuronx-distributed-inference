"""
Compile all HunyuanVideo-1.5 Neuron models.

Compiles:
  1. Transformer backbone (NxDI, TP=2) — ~10 min
  2. Qwen2.5-VL encoder (NxDI, TP=2) — ~2 min
  3. ByT5 encoder (torch_neuronx.trace) — ~30s
  4. Token refiner (torch_neuronx.trace) — ~30s
  5. Token reorder (torch_neuronx.trace) — ~10s
  6. cond_type_embed weights (extracted) — instant
  7. VAE decoder (24 traced shards) — ~4.5 hrs

Usage:
    python compile_all.py --output_dir .
    python compile_all.py --output_dir . --skip_transformer  # skip slow backbone
    python compile_all.py --output_dir . --only vae          # compile only VAE
"""
import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch

HF_MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def compile_transformer(output_dir: str):
    """Compile the 54-block transformer backbone (NxDI, TP=2)."""
    from modeling_hunyuan_video15_transformer import (
        HunyuanVideo15TransformerConfig, NeuronHunyuanVideo15Transformer,
    )
    save_path = str(Path(output_dir) / "compiled_transformer")
    config = HunyuanVideo15TransformerConfig.from_pretrained(HF_MODEL_ID)
    model = NeuronHunyuanVideo15Transformer(model_path=HF_MODEL_ID, config=config)
    model.compile(save_path)
    print(f"Transformer compiled: {save_path}")
    del model; gc.collect()


def compile_qwen2vl(output_dir: str):
    """Compile the Qwen2.5-VL encoder (NxDI, TP=2)."""
    from modeling_qwen2vl_encoder import Qwen2VLEncoderConfig, NeuronQwen2VLEncoder
    save_path = str(Path(output_dir) / "compiled_qwen2vl")
    config = Qwen2VLEncoderConfig.from_pretrained(QWEN_MODEL_ID)
    model = NeuronQwen2VLEncoder(model_path=QWEN_MODEL_ID, config=config)
    model.compile(save_path)
    print(f"Qwen2.5-VL encoder compiled: {save_path}")
    del model; gc.collect()


def compile_traced_models(output_dir: str):
    """Compile ByT5, token refiner, and token reorder (torch_neuronx.trace)."""
    import torch_neuronx
    from modeling_hunyuan_video15_text import (
        compile_byt5_encoder, compile_token_refiner,
    )
    out = Path(output_dir)
    compile_byt5_encoder(str(out / "byt5_traced.pt"))
    gc.collect()
    compile_token_refiner(str(out / "refiner_traced.pt"), hf_model_path=HF_MODEL_ID)
    gc.collect()
    # Token reorder: use topk instead of sort (sort unsupported on trn2)
    _compile_token_reorder_trn2(str(out / "reorder_traced.pt"))
    gc.collect()


def _compile_token_reorder_trn2(save_path: str):
    """Compile token reorder using topk (trn2-compatible, sort is unsupported)."""
    import torch.nn as nn
    import torch_neuronx

    class TokenReorderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("im_zeros", torch.zeros(729))
            self.register_buffer("positions", torch.arange(1985, dtype=torch.float32))
            self.register_buffer("large_offset", torch.arange(1985, dtype=torch.float32) * 0.0001)

        def forward(self, mllm_mask, byt5_mask):
            combined = torch.cat([mllm_mask, byt5_mask, self.im_zeros])
            n_valid = combined.sum()
            zero_mask = (self.positions < n_valid).float()
            # topk with position tiebreaker preserves original token order
            scored = combined + (1.0 - self.large_offset)
            _, idx = torch.topk(scored, k=1985, sorted=True)
            return idx, zero_mask

    model = TokenReorderModel().eval()
    mm = torch.ones(1000)
    bm = torch.ones(256)
    os.environ["NEURON_CC_FLAGS"] = "-O1 --auto-cast=none"
    traced = torch_neuronx.trace(model, (mm, bm))
    torch.jit.save(traced, save_path)
    print(f"Token reorder compiled (trn2): {save_path}")


def extract_cond_type_weights(output_dir: str):
    """Extract cond_type_embed weights from HF model."""
    from diffusers import HunyuanVideo15Transformer3DModel
    hf = HunyuanVideo15Transformer3DModel.from_pretrained(
        HF_MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
    )
    w = hf.cond_type_embed.weight.data.clone()
    save_path = str(Path(output_dir) / "cond_type_embed_weight.pt")
    torch.save(w, save_path)
    print(f"cond_type_embed weights saved: {save_path}")
    del hf; gc.collect()


def compile_vae(output_dir: str):
    """Compile VAE decoder as 24 shards."""
    from modeling_hunyuan_video15_vae import compile_vae_shards
    save_dir = str(Path(output_dir) / "vae_shards")
    compile_vae_shards(save_dir, hf_model_path=HF_MODEL_ID)


def main():
    parser = argparse.ArgumentParser(description="Compile HunyuanVideo-1.5 for Neuron")
    parser.add_argument("--output_dir", default="./compiled", help="Output directory")
    parser.add_argument("--skip_transformer", action="store_true")
    parser.add_argument("--skip_qwen", action="store_true")
    parser.add_argument("--skip_vae", action="store_true")
    parser.add_argument("--only", choices=["transformer", "qwen", "traced", "weights", "vae"])
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    steps = []
    if args.only:
        steps = [args.only]
    else:
        if not args.skip_transformer:
            steps.append("transformer")
        if not args.skip_qwen:
            steps.append("qwen")
        steps.extend(["traced", "weights"])
        if not args.skip_vae:
            steps.append("vae")

    for step in steps:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Compiling: {step}")
        print(f"{'='*60}")
        if step == "transformer":
            compile_transformer(args.output_dir)
        elif step == "qwen":
            compile_qwen2vl(args.output_dir)
        elif step == "traced":
            compile_traced_models(args.output_dir)
        elif step == "weights":
            extract_cond_type_weights(args.output_dir)
        elif step == "vae":
            compile_vae(args.output_dir)
        print(f"  Completed in {time.time()-t0:.0f}s")

    print(f"\n{'='*60}")
    print(f"All compilations complete. Artifacts in: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
