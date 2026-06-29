#!/usr/bin/env python3
"""
Synthetic smoke test for MiniMax-M3 on Neuron.

Builds a tiny M3-shaped checkpoint with random weights, then exercises the
full compile -> load -> generate -> TTFT/ITL flow without needing the 854GB
official checkpoint. The goal is to validate the modeling code on real
Trn hardware while the full download is in flight.

The synthetic config differs from the released one only in size (fewer
layers, fewer experts, smaller hidden), so any code path that would fail on
the real config (Gemma RMSNorm, partial RoPE, dense + MoE mix, SwiGLU-OAI)
also fails here.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

_TEST_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _TEST_DIR.parent.parent
sys.path.insert(0, str(_MODEL_DIR / "src"))

from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from transformers import AutoTokenizer
from modeling_minimax_m3 import (  # noqa: E402
    NeuronMiniMaxM3ForCausalLM,
    MiniMaxM3InferenceConfig,
    MiniMaxM3NeuronConfig,
)


SYNTH_DIR = Path(os.environ.get("M3_SYNTH_DIR", "/mnt/data/models/MiniMax-M3-synthetic"))
COMPILED_DIR = Path(os.environ.get("M3_SYNTH_COMPILED", "/mnt/data/neuron_models/MiniMax-M3-synthetic"))
REAL_MODEL_DIR = Path(os.environ.get("M3_MODEL_PATH", "/mnt/data/models/MiniMax-M3"))

# Small but architecturally faithful config.
SYNTH = {
    "model_type": "minimax_m3",
    "architectures": ["MiniMaxM3SparseForCausalLM"],
    "vocab_size": 200064,                # match real tokenizer
    "hidden_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "intermediate_size": 256,            # MoE per-expert FFN
    "dense_intermediate_size": 1024,
    "shared_intermediate_size": 256,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,          # 1 dense layer, 3 MoE layers
    "moe_layer_freq": [0, 1, 1, 1],
    "max_position_embeddings": 4096,
    "rope_theta": 5000000.0,
    "rotary_dim": 32,                    # partial RoPE: 32 / 64 = 0.5
    "partial_rotary_factor": 0.5,
    "rms_norm_eps": 1e-6,
    "use_gemma_norm": True,
    "use_qk_norm": True,
    "tie_word_embeddings": False,
    "hidden_act": "swigluoai",
    "swiglu_alpha": 1.702,
    "swiglu_limit": 7.0,
    "routed_scaling_factor": 2.0,
    "scoring_func": "sigmoid",
    "pad_token_id": 0,
    "bos_token_id": 200000,
    "eos_token_id": 200000,
}


def _create_synthetic_checkpoint(out_dir: Path):
    """Materialize a tiny but architecturally faithful M3 checkpoint."""
    if (out_dir / "config.json").exists() and any(out_dir.glob("*.safetensors")):
        print(f"[synthetic] reusing existing checkpoint at {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SYNTH
    h = cfg["hidden_size"]
    hd = cfg["head_dim"]
    n_q = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    moe_inter = cfg["intermediate_size"]
    dense_inter = cfg["dense_intermediate_size"]
    shared_inter = cfg["shared_intermediate_size"]
    num_experts = cfg["num_local_experts"]
    V = cfg["vocab_size"]

    g = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.bfloat16) * 0.02

    sd = {}
    # Embeddings + final norm + LM head — matches the released layout
    # ("language_model.model.*" prefix). The converter strips it.
    sd["language_model.model.embed_tokens.weight"] = rand(V, h)
    sd["language_model.model.norm.weight"] = torch.zeros(h, dtype=torch.bfloat16)
    sd["language_model.lm_head.weight"] = rand(V, h)

    for li in range(cfg["num_hidden_layers"]):
        layer = f"language_model.model.layers.{li}"
        # norms (Gemma-style — weight=0 → scale=1)
        sd[f"{layer}.input_layernorm.weight"] = torch.zeros(h, dtype=torch.bfloat16)
        sd[f"{layer}.post_attention_layernorm.weight"] = torch.zeros(h, dtype=torch.bfloat16)
        # attention projections
        sd[f"{layer}.self_attn.q_proj.weight"] = rand(n_q * hd, h)
        sd[f"{layer}.self_attn.k_proj.weight"] = rand(n_kv * hd, h)
        sd[f"{layer}.self_attn.v_proj.weight"] = rand(n_kv * hd, h)
        sd[f"{layer}.self_attn.o_proj.weight"] = rand(h, n_q * hd)
        # per-head Gemma QK norm (head_dim,)
        sd[f"{layer}.self_attn.q_norm.weight"] = torch.zeros(hd, dtype=torch.bfloat16)
        sd[f"{layer}.self_attn.k_norm.weight"] = torch.zeros(hd, dtype=torch.bfloat16)

        if cfg["moe_layer_freq"][li] == 0:
            # Dense (SwiGLU-OAI). HF release stores gate/up/down separately.
            sd[f"{layer}.mlp.gate_proj.weight"] = rand(dense_inter, h)
            sd[f"{layer}.mlp.up_proj.weight"] = rand(dense_inter, h)
            sd[f"{layer}.mlp.down_proj.weight"] = rand(h, dense_inter)
        else:
            # MoE: router + experts + shared experts
            sd[f"{layer}.block_sparse_moe.gate.weight"] = rand(num_experts, h)
            sd[f"{layer}.block_sparse_moe.e_score_correction_bias"] = torch.zeros(
                num_experts, dtype=torch.bfloat16
            )
            for e in range(num_experts):
                p = f"{layer}.block_sparse_moe.experts.{e}"
                sd[f"{p}.w1.weight"] = rand(moe_inter, h)    # gate
                sd[f"{p}.w2.weight"] = rand(h, moe_inter)    # down
                sd[f"{p}.w3.weight"] = rand(moe_inter, h)    # up
            shared = f"{layer}.block_sparse_moe.shared_experts"
            sd[f"{shared}.gate_proj.weight"] = rand(shared_inter, h)
            sd[f"{shared}.up_proj.weight"] = rand(shared_inter, h)
            sd[f"{shared}.down_proj.weight"] = rand(h, shared_inter)

    save_file(sd, str(out_dir / "model.safetensors"))

    # Build the released VL-style config with text_config nesting so the
    # MiniMaxM3InferenceConfig promotion logic exercises the real code path.
    released_config = {
        "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
        "model_type": "minimax_m3_vl",
        "text_config": cfg,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(released_config, f, indent=2)
    print(f"[synthetic] wrote {len(sd)} tensors and config.json to {out_dir}")


def _copy_tokenizer(src: Path, dst: Path):
    """Copy tokenizer artifacts from the real checkpoint (or skip)."""
    if not src.exists():
        print(f"[tokenizer] real model dir {src} not present; will skip tokenizer.")
        return False
    import shutil
    for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "vocab.json", "merges.txt", "added_tokens.json"):
        srcp = src / fn
        if srcp.exists() and not (dst / fn).exists():
            shutil.copy(srcp, dst / fn)
    return (dst / "tokenizer.json").exists()


def main():
    _create_synthetic_checkpoint(SYNTH_DIR)
    has_tok = _copy_tokenizer(REAL_MODEL_DIR, SYNTH_DIR)

    print(f"\n[compile] tp_degree=8, seq_len=128, bsz=1")
    neuron_config = MiniMaxM3NeuronConfig(
        tp_degree=8,
        batch_size=1,
        seq_len=128,
        max_context_length=128,
        torch_dtype=torch.bfloat16,
    )

    # Build a PretrainedConfig manually to bypass AutoConfig (model_type is
    # not registered in transformers — that's intentional for the synthetic test).
    from transformers.configuration_utils import PretrainedConfig
    cfg = SYNTH.copy()
    # Promote text_config fields to top-level so InferenceConfig sees them.
    hf_cfg = PretrainedConfig(**cfg)
    config = MiniMaxM3InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(hf_config=hf_cfg),
    )

    print(f"  num_hidden_layers={config.num_hidden_layers}")
    print(f"  num_local_experts={config.num_local_experts}")
    print(f"  moe_layer_freq={config.moe_layer_freq}")
    print(f"  head_dim={config.head_dim} rotary_dim={getattr(config, 'rotary_dim', '?')}")

    # Compile if missing
    if not (COMPILED_DIR / "model.pt").exists():
        COMPILED_DIR.mkdir(parents=True, exist_ok=True)
        model = NeuronMiniMaxM3ForCausalLM(str(SYNTH_DIR), config)
        t0 = time.perf_counter()
        model.compile(str(COMPILED_DIR))
        print(f"[compile] done in {time.perf_counter() - t0:.1f}s")
    else:
        print(f"[compile] reusing existing NEFF at {COMPILED_DIR}")

    # Load
    print("\n[load] ...")
    model = NeuronMiniMaxM3ForCausalLM(str(SYNTH_DIR), config)
    t0 = time.perf_counter()
    model.load(str(COMPILED_DIR))
    print(f"[load] done in {time.perf_counter() - t0:.1f}s")

    # Inputs: synthetic random tokens (10 of them).
    input_ids = torch.randint(0, 30000, (1, 10), dtype=torch.long)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    # Warmup
    print("\n[warmup] 2 prefill calls")
    for _ in range(2):
        with torch.no_grad():
            _ = model(input_ids, position_ids=position_ids)

    # TTFT
    print("[ttft] measuring prefill latency over 5 runs")
    ttft_ms = []
    for _ in range(5):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids, position_ids=position_ids)
        ttft_ms.append((time.perf_counter() - t0) * 1000)
    avg_ttft = sum(ttft_ms) / len(ttft_ms)
    print(f"[ttft] mean={avg_ttft:.2f} ms  values={[f'{x:.2f}' for x in ttft_ms]}")

    # Get logits
    if hasattr(out, "logits"):
        logits = out.logits
    elif isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    print(f"[ttft] logits shape: {tuple(logits.shape)}")

    # ITL — measure 10 decode steps
    print("\n[itl] measuring decode latency for 10 steps")
    gen_ids = input_ids.clone()
    # prime decode
    with torch.no_grad():
        out = model(gen_ids, position_ids=torch.arange(gen_ids.shape[1]).unsqueeze(0))

    itl_ms = []
    for _ in range(10):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(gen_ids, position_ids=torch.arange(gen_ids.shape[1]).unsqueeze(0))
        itl_ms.append((time.perf_counter() - t0) * 1000)
        next_tok = torch.argmax(out.logits if hasattr(out, "logits") else out[0],
                                dim=-1)[:, -1:]
        gen_ids = torch.cat([gen_ids, next_tok], dim=-1)
        if gen_ids.shape[1] >= 100:
            break
    avg_itl = sum(itl_ms) / len(itl_ms)
    throughput = 1000.0 / avg_itl if avg_itl > 0 else 0.0
    print(f"[itl]  mean={avg_itl:.2f} ms/token  throughput={throughput:.2f} tok/s")
    print(f"[itl]  first_3={[f'{x:.2f}' for x in itl_ms[:3]]} "
          f"last_3={[f'{x:.2f}' for x in itl_ms[-3:]]}")

    print("\nAll synthetic checks passed.")


if __name__ == "__main__":
    main()
