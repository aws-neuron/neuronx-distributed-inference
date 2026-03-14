#!/usr/bin/env python3
"""
LTX-2.3 Load with Real Weights & Forward Test
================================================
Loads the compiled tp_0.pt, injects properly TP-sharded weights from
safetensors, and runs a single forward pass to validate correctness.

This script handles the full weight loading pipeline:
1. Load safetensors -> strip ComfyUI prefix
2. Map safetensors keys to JIT model parameter names
3. Shard each weight per TP rank according to the sharding pattern
4. Inject into each rank's JIT model
5. Load onto Neuron devices
6. Run forward pass

Usage:
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
  python3 load_with_weights.py
"""

import os
import sys
import time
import json
import gc
import torch
import numpy as np

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "/home/ubuntu/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors"
COMPILE_DIR = "/home/ubuntu/ltx23_neuron/compiler_workdir_tp4_lnc2_v2"
TP_DEGREE = 4
BATCH = 1
VIDEO_SEQ = 768
AUDIO_SEQ = 26
TEXT_SEQ = 256


def load_config():
    from safetensors import safe_open

    with safe_open(MODEL_PATH, framework="pt") as f:
        metadata = f.metadata()
    return json.loads(metadata["config"])


def precompute_inputs(config):
    """Build example inputs using native ltx-core preprocessors."""
    sys.path.insert(0, "/home/ubuntu/ltx23_nxdi")
    from modeling_ltx23 import replace_sdpa_with_bmm

    replace_sdpa_with_bmm()

    from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.tools import (
        VideoLatentTools,
        VideoLatentPatchifier,
        VideoLatentShape,
        AudioLatentTools,
        AudioPatchifier,
        AudioLatentShape,
        SpatioTemporalScaleFactors,
    )

    dtype = torch.bfloat16
    ltx_model = LTXModelConfigurator.from_config(config)
    ltx_model = ltx_model.to(dtype=dtype)
    ltx_model.eval()

    torch.manual_seed(42)

    video_shape = VideoLatentShape(
        batch=BATCH, channels=128, frames=4, height=12, width=16
    )
    v_patchifier = VideoLatentPatchifier(patch_size=1)
    v_scale = SpatioTemporalScaleFactors(time=1, width=8, height=8)
    video_tools = VideoLatentTools(
        target_shape=video_shape,
        patchifier=v_patchifier,
        scale_factors=v_scale,
        causal_fix=False,
        fps=24.0,
    )
    video_state = video_tools.create_initial_state(device="cpu", dtype=dtype)

    audio_shape = AudioLatentShape(
        batch=BATCH, channels=8, frames=AUDIO_SEQ, mel_bins=16
    )
    a_patchifier = AudioPatchifier(patch_size=16)
    audio_tools = AudioLatentTools(patchifier=a_patchifier, target_shape=audio_shape)
    audio_state = audio_tools.create_initial_state(device="cpu", dtype=dtype)

    # CRITICAL: Use random noise as latent, NOT zeros from create_initial_state().
    # The denoising loop starts with pure noise at sigma=1.0.
    video_latent = torch.randn_like(video_state.latent)
    audio_latent = torch.randn_like(audio_state.latent)

    sigma = torch.tensor([1.0], dtype=dtype)
    v_ts = sigma.unsqueeze(1).expand(BATCH, video_latent.shape[1])
    a_ts = sigma.unsqueeze(1).expand(BATCH, audio_latent.shape[1])

    ctx_v = torch.randn(BATCH, TEXT_SEQ, 4096, dtype=dtype)
    ctx_a = torch.randn(BATCH, TEXT_SEQ, 2048, dtype=dtype)
    ctx_mask = torch.ones(BATCH, TEXT_SEQ, dtype=dtype)
    ctx_mask[:, 50:] = 0

    video_mod = Modality(
        latent=video_latent,
        sigma=sigma,
        timesteps=v_ts,
        positions=video_state.positions,
        context=ctx_v,
        enabled=True,
        context_mask=ctx_mask,
        attention_mask=None,
    )
    audio_mod = Modality(
        latent=audio_latent,
        sigma=sigma,
        timesteps=a_ts,
        positions=audio_state.positions,
        context=ctx_a,
        enabled=True,
        context_mask=ctx_mask,
        attention_mask=None,
    )

    with torch.no_grad():
        va = ltx_model.video_args_preprocessor.prepare(video_mod, audio_mod)
        aa = ltx_model.audio_args_preprocessor.prepare(audio_mod, video_mod)

    video_pe_cos, video_pe_sin = va.positional_embeddings
    audio_pe_cos, audio_pe_sin = aa.positional_embeddings
    ca_video_pe_cos, ca_video_pe_sin = va.cross_positional_embeddings
    ca_audio_pe_cos, ca_audio_pe_sin = aa.cross_positional_embeddings

    inputs = (
        va.x.to(dtype),
        aa.x.to(dtype),
        va.context.to(dtype),
        aa.context.to(dtype),
        va.timesteps.to(dtype),
        aa.timesteps.to(dtype),
        va.embedded_timestep.to(dtype),
        aa.embedded_timestep.to(dtype),
        va.cross_scale_shift_timestep.to(dtype),
        aa.cross_scale_shift_timestep.to(dtype),
        va.cross_gate_timestep.to(dtype),
        aa.cross_gate_timestep.to(dtype),
        video_pe_cos.to(dtype),
        video_pe_sin.to(dtype),
        audio_pe_cos.to(dtype),
        audio_pe_sin.to(dtype),
        ca_video_pe_cos.to(dtype),
        ca_video_pe_sin.to(dtype),
        ca_audio_pe_cos.to(dtype),
        ca_audio_pe_sin.to(dtype),
        va.context_mask.to(dtype),
        aa.context_mask.to(dtype).clone(),
        va.prompt_timestep.to(dtype),
        aa.prompt_timestep.to(dtype),
    )

    del ltx_model
    gc.collect()
    return inputs


def shard_weight(full_weight, jit_param_name, tp_rank, tp_size):
    """Shard a full weight tensor for the given TP rank.

    Determines the sharding pattern from the JIT parameter name:
    - ColumnParallelLinear (to_q, to_k, to_v, to_gate_logits, GEGLU proj):
        weight sharded on dim 0, bias sharded on dim 0
    - RowParallelLinear (to_out->0, ff->net->2):
        weight sharded on dim 1, bias NOT sharded
    - DistributedRMSNorm (q_norm, k_norm):
        weight sharded on dim 0
    - SPMDRank: rank tensor, select element for this rank
    - Unsharded (scale_shift_table, norm_out, proj_out, etc.):
        return full weight unchanged
    """
    name = jit_param_name

    # SPMDRank: select this rank's value
    if "spmd_rank" in name:
        return torch.tensor([tp_rank], dtype=torch.int32)

    # Determine sharding from the parameter name
    shard_size = None
    shard_dim = None

    # Check if this is a sharded parameter
    is_column_weight = False
    is_column_bias = False
    is_row_weight = False
    is_row_bias = False
    is_norm_weight = False

    # Column-parallel: to_q, to_k, to_v, to_gate_logits
    # Use delimited patterns (->X->) to avoid false matches like
    # "to_v" matching "audio_to_video_attn"
    for col_name in ["->to_q->", "->to_k->", "->to_v->", "->to_gate_logits->"]:
        if col_name in name:
            if name.endswith("weight"):
                is_column_weight = True
            elif name.endswith("bias"):
                is_column_bias = True
            break

    # Column-parallel: GEGLU gate proj (ff->net->0->proj)
    if "ff->net->0->proj" in name:
        if name.endswith("weight"):
            is_column_weight = True
        elif name.endswith("bias"):
            is_column_bias = True

    # Row-parallel: output projection (to_out->0)
    if "to_out->0" in name:
        if name.endswith("weight"):
            is_row_weight = True
        elif name.endswith("bias"):
            is_row_bias = True  # Bias not sharded for RowParallel

    # Row-parallel: FFN down projection (ff->net->2)
    if "ff->net->2" in name:
        if name.endswith("weight"):
            is_row_weight = True
        elif name.endswith("bias"):
            is_row_bias = True  # Bias not sharded

    # DistributedRMSNorm: q_norm, k_norm
    if ("q_norm" in name or "k_norm" in name) and name.endswith("weight"):
        is_norm_weight = True

    # Apply sharding
    if is_column_weight or is_column_bias or is_norm_weight:
        # Shard on dim 0
        shard_size = full_weight.shape[0] // tp_size
        return full_weight[shard_size * tp_rank : shard_size * (tp_rank + 1)].clone()

    elif is_row_weight:
        # Shard on dim 1
        shard_size = full_weight.shape[1] // tp_size
        return full_weight[:, shard_size * tp_rank : shard_size * (tp_rank + 1)].clone()

    elif is_row_bias:
        # Not sharded — full copy
        return full_weight.clone()

    else:
        # Unsharded (scale_shift_table, norm_out, proj_out, audio variants)
        return full_weight.clone()


def load_and_shard_weights():
    """Load safetensors and create per-rank state dicts."""
    from safetensors.torch import load_file

    print("  Loading safetensors...", flush=True)
    t0 = time.time()
    full_sd = load_file(MODEL_PATH)
    print(f"  Loaded {len(full_sd)} tensors in {time.time() - t0:.1f}s", flush=True)

    # Strip ComfyUI prefix
    prefix = "model.diffusion_model."
    stripped_sd = {}
    for k, v in full_sd.items():
        if k.startswith(prefix):
            stripped_sd[k[len(prefix) :]] = v
        else:
            stripped_sd[k] = v

    # Filter to backbone keys (same as compile_transformer.py)
    backbone_prefixes = (
        "transformer_blocks.",
        "norm_out.",
        "proj_out.",
        "scale_shift_table",
        "audio_norm_out.",
        "audio_proj_out.",
        "audio_scale_shift_table",
    )
    backbone_sd = {}
    for k, v in stripped_sd.items():
        if k.startswith(backbone_prefixes):
            backbone_sd[k] = v.to(torch.bfloat16).contiguous()

    # Add SPMDRank (will be per-rank in shard_weight)
    backbone_sd["spmd_rank.rank"] = torch.arange(0, TP_DEGREE, dtype=torch.int32)

    del full_sd, stripped_sd
    gc.collect()

    print(f"  {len(backbone_sd)} backbone keys", flush=True)

    # Convert safetensors key format to JIT param format
    # safetensors: "transformer_blocks.0.attn1.to_q.weight"
    # JIT:        "weights.transformer_blocks->0->attn1->to_q->weight"
    def sf_key_to_jit_key(sf_key):
        return "weights." + sf_key.replace(".", "->")

    # Create per-rank state dicts
    rank_sds = [{} for _ in range(TP_DEGREE)]
    for sf_key, full_weight in backbone_sd.items():
        jit_key = sf_key_to_jit_key(sf_key)
        for rank in range(TP_DEGREE):
            sharded = shard_weight(full_weight, jit_key, rank, TP_DEGREE)
            rank_sds[rank][jit_key] = sharded

    del backbone_sd
    gc.collect()

    return rank_sds


def main():
    print("=" * 60, flush=True)
    print("LTX-2.3 Load with Real Weights & Forward Test", flush=True)
    print("=" * 60, flush=True)

    # 1. Load config
    print("\n[1/5] Loading config...", flush=True)
    config = load_config()
    tc = config["transformer"]
    print(f"  {tc['num_layers']} layers, {tc['num_attention_heads']} heads", flush=True)

    # 2. Precompute inputs
    print("\n[2/5] Precomputing inputs...", flush=True)
    inputs = precompute_inputs(config)
    print(f"  Got {len(inputs)} input tensors", flush=True)

    # 3. Load and shard weights
    print("\n[3/5] Loading and sharding weights...", flush=True)
    rank_sds = load_and_shard_weights()
    for rank, sd in enumerate(rank_sds):
        print(f"  Rank {rank}: {len(sd)} parameters", flush=True)

    # 4. Load compiled models and inject weights
    print("\n[4/5] Loading compiled models onto Neuron...", flush=True)
    import torch_neuronx
    from neuronx_distributed.trace.trace import TensorParallelNeuronModel

    tp_0_path = os.path.join(COMPILE_DIR, "tp_0.pt")

    models = []
    t0 = time.time()
    for rank in range(TP_DEGREE):
        print(f"  Loading rank {rank}...", flush=True)
        with torch_neuronx.contexts.disable_nrt_load():
            model = torch.jit.load(tp_0_path)

        # Inject per-rank weights
        model_sd = dict(model.named_parameters())
        injected = 0
        missing = 0
        mismatched = 0
        for jit_key, sharded_weight in rank_sds[rank].items():
            if jit_key in model_sd:
                if model_sd[jit_key].shape == sharded_weight.shape:
                    model_sd[jit_key].data.copy_(sharded_weight)
                    injected += 1
                else:
                    mismatched += 1
                    if rank == 0 and mismatched <= 5:
                        print(
                            f"    MISMATCH: {jit_key}: model={model_sd[jit_key].shape} vs shard={sharded_weight.shape}",
                            flush=True,
                        )
            else:
                missing += 1

        if rank == 0:
            print(
                f"    Injected {injected}, mismatched {mismatched}, missing {missing}",
                flush=True,
            )

        models.append(model)

    print(f"  All models loaded in {time.time() - t0:.1f}s", flush=True)

    del rank_sds
    gc.collect()

    # Create TensorParallelNeuronModel
    neuron_model = TensorParallelNeuronModel(models)
    print(f"  TP degree: {neuron_model.tp_degree}", flush=True)

    # 5. Run forward pass
    print("\n[5/5] Running forward pass...", flush=True)
    with torch.no_grad():
        t0 = time.time()
        video_out, audio_out = neuron_model(*inputs)
        elapsed = time.time() - t0

    print(f"\n  Forward pass completed in {elapsed:.2f}s", flush=True)
    print(f"  Video output: {video_out.shape}, dtype={video_out.dtype}", flush=True)
    print(f"  Audio output: {audio_out.shape}, dtype={audio_out.dtype}", flush=True)

    v_np = video_out.float().numpy()
    a_np = audio_out.float().numpy()

    print(
        f"\n  Video stats: min={v_np.min():.4f}, max={v_np.max():.4f}, "
        f"mean={v_np.mean():.4f}, std={v_np.std():.4f}",
        flush=True,
    )
    print(
        f"  Audio stats: min={a_np.min():.4f}, max={a_np.max():.4f}, "
        f"mean={a_np.mean():.4f}, std={a_np.std():.4f}",
        flush=True,
    )

    has_nan = np.isnan(v_np).any() or np.isnan(a_np).any()
    has_inf = np.isinf(v_np).any() or np.isinf(a_np).any()
    print(f"\n  NaN: {has_nan}, Inf: {has_inf}", flush=True)

    # Save outputs
    output_path = "/home/ubuntu/ltx23_neuron/test_outputs_real_weights.pt"
    torch.save(
        {
            "video_output": video_out.cpu(),
            "audio_output": audio_out.cpu(),
            "inputs": tuple(t.cpu() for t in inputs),
        },
        output_path,
    )
    print(f"  Saved to {output_path}", flush=True)

    all_ok = not has_nan and not has_inf
    print(f"\n{'=' * 60}", flush=True)
    print(f"  RESULT: {'PASS' if all_ok else 'FAIL'}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Second pass
    print("\n  Running second forward pass (warmed up)...", flush=True)
    with torch.no_grad():
        t0 = time.time()
        video_out2, audio_out2 = neuron_model(*inputs)
        elapsed2 = time.time() - t0
    print(f"  Second pass: {elapsed2:.2f}s", flush=True)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
