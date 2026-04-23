#!/usr/bin/env python3
"""
LTX-2.3 Full 48-Block TP=4 Compilation
=======================================
Compiles the LTX-2.3 22B DiT transformer backbone (48 blocks) with TP=4,
LNC=2 for trn2.3xlarge using native ltx-core model.

Uses the NeuronLTX23TransformerBackbone from modeling_ltx23.py which:
- Builds the model via LTXModelConfigurator.from_config()
- Applies TP sharding (Column/RowParallelLinear, DistributedRMSNorm)
- Constructs TransformerArgs from flat tensors for native block forward
- Applies SPMDRank for per-rank RoPE slicing

Output: COMPILE_DIR/tp_0.pt

Usage:
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
  NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
    torchrun --nproc_per_node=4 compile_transformer.py
"""

import os
import sys
import time
import gc
import json
import shutil
import torch
import torch.nn as nn

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TP_DEGREE = 4
BATCH = 1
VIDEO_SEQ = 768  # 4 frames * 12 * 16 patches (384x512 resolution, patchsize=1x2x2)
LATENT_H = 12  # 384 / 32
LATENT_W = 16  # 512 / 32
AUDIO_SEQ = 26  # audio tokens for ~2s
TEXT_SEQ = 256  # max text sequence length

MODEL_PATH = "/home/ubuntu/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors"
COMPILE_DIR = "/home/ubuntu/ltx23_neuron/compiler_workdir_tp4_lnc2_v2"

# Architecture constants (from safetensors metadata)
NUM_HEADS = 32
AUDIO_NUM_HEADS = 32
INNER_DIM = 4096  # NUM_HEADS * 128
AUDIO_INNER_DIM = 2048  # AUDIO_NUM_HEADS * 64
AUDIO_CA_DIM = 2048  # audio_cross_attention_dim


def load_config_from_safetensors():
    """Load the transformer config from safetensors metadata."""
    from safetensors import safe_open

    with safe_open(MODEL_PATH, framework="pt") as f:
        metadata = f.metadata()
    config = json.loads(metadata["config"])
    tc = config["transformer"]
    print(
        f"  Loaded config: {tc['num_layers']} layers, "
        f"{tc['num_attention_heads']} heads, "
        f"head_dim={tc['attention_head_dim']}",
        flush=True,
    )
    return config


def precompute_inputs(config):
    """Build example inputs using the native ltx-core preprocessing.

    Creates a temporary unsharded model to run patchify_proj, adaln_single,
    rope, etc. on CPU, producing the 24 flat tensors for the backbone.

    Uses VideoLatentTools/AudioLatentTools for correct position generation.
    """
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

    # Build full unsharded model on CPU
    ltx_model = LTXModelConfigurator.from_config(config)
    ltx_model = ltx_model.to(dtype=dtype)
    ltx_model.eval()

    torch.manual_seed(123)

    # Video: patch_size=1 (no spatial patchification in DiT), VAE channels=128
    # 768 tokens = 4 frames * 12h * 16w in the VAE latent grid
    video_shape = VideoLatentShape(
        batch=BATCH, channels=128, frames=4, height=LATENT_H, width=LATENT_W
    )
    v_patchifier = VideoLatentPatchifier(patch_size=1)
    v_scale = SpatioTemporalScaleFactors.default()  # time=8, height=32, width=32
    video_tools = VideoLatentTools(
        target_shape=video_shape,
        patchifier=v_patchifier,
        scale_factors=v_scale,
        causal_fix=False,
        fps=24.0,
    )
    video_state = video_tools.create_initial_state(device="cpu", dtype=dtype)

    # Audio: VAE z_channels=8, mel_bins_latent=16, patchified dim=128
    audio_shape = AudioLatentShape(
        batch=BATCH, channels=8, frames=AUDIO_SEQ, mel_bins=16
    )
    a_patchifier = AudioPatchifier(patch_size=16)
    audio_tools = AudioLatentTools(patchifier=a_patchifier, target_shape=audio_shape)
    audio_state = audio_tools.create_initial_state(device="cpu", dtype=dtype)

    # CRITICAL: Use random noise as compile-time latent, NOT zeros from
    # create_initial_state(). The denoising loop starts at sigma=1.0 with
    # pure noise, which produces hidden_states ~10x larger after patchify_proj.
    # Compiling with zeros causes numerical issues in the Neuron graph when
    # actual noise-level inputs are fed at runtime.
    # State is a frozen dataclass so we create noise tensors separately.
    video_latent = torch.randn_like(video_state.latent)
    audio_latent = torch.randn_like(audio_state.latent)

    print(
        f"  Video: latent={video_latent.shape}, "
        f"positions={video_state.positions.shape}",
        flush=True,
    )
    print(
        f"  Audio: latent={audio_latent.shape}, "
        f"positions={audio_state.positions.shape}",
        flush=True,
    )
    print(
        f"  Video latent norm: {video_latent.float().norm():.2f} (noise, not zeros)",
        flush=True,
    )

    # Sigma for timestep: use sigma=1.0 (start of denoising) to compile
    # with representative input magnitudes. The compiled graph must handle
    # all sigma values from 1.0 to 0.0 during denoising.
    sigma = torch.tensor([1.0], dtype=dtype)
    v_ts = sigma.unsqueeze(1).expand(BATCH, video_latent.shape[1])
    a_ts = sigma.unsqueeze(1).expand(BATCH, audio_latent.shape[1])

    # Context: already projected by connector (random for compilation)
    ctx_v = torch.randn(BATCH, TEXT_SEQ, INNER_DIM, dtype=dtype)
    ctx_a = torch.randn(BATCH, TEXT_SEQ, AUDIO_INNER_DIM, dtype=dtype)
    ctx_mask = torch.ones(BATCH, TEXT_SEQ, dtype=dtype)
    ctx_mask[:, 50:] = 0  # mask out padding

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

    # Run model preprocessors to get TransformerArgs
    with torch.no_grad():
        va = ltx_model.video_args_preprocessor.prepare(video_mod, audio_mod)
        aa = ltx_model.audio_args_preprocessor.prepare(audio_mod, video_mod)

    # Extract flat tensors from TransformerArgs
    video_pe_cos, video_pe_sin = va.positional_embeddings
    audio_pe_cos, audio_pe_sin = aa.positional_embeddings
    ca_video_pe_cos, ca_video_pe_sin = va.cross_positional_embeddings
    ca_audio_pe_cos, ca_audio_pe_sin = aa.cross_positional_embeddings

    inputs = (
        va.x.to(dtype),  # hidden_states
        aa.x.to(dtype),  # audio_hidden_states
        va.context.to(dtype),  # encoder_hidden_states
        aa.context.to(dtype),  # audio_encoder_hidden_states
        va.timesteps.to(dtype),  # temb (per-token, 9*inner_dim)
        aa.timesteps.to(dtype),  # temb_audio
        va.embedded_timestep.to(dtype),  # embedded_timestep (per-token)
        aa.embedded_timestep.to(dtype),  # audio_embedded_timestep
        va.cross_scale_shift_timestep.to(dtype),  # video_ca_ss
        aa.cross_scale_shift_timestep.to(dtype),  # audio_ca_ss
        va.cross_gate_timestep.to(dtype),  # video_ca_gate
        aa.cross_gate_timestep.to(dtype),  # audio_ca_gate
        video_pe_cos.to(dtype),  # video_rot_cos
        video_pe_sin.to(dtype),  # video_rot_sin
        audio_pe_cos.to(dtype),  # audio_rot_cos
        audio_pe_sin.to(dtype),  # audio_rot_sin
        ca_video_pe_cos.to(dtype),  # ca_video_rot_cos
        ca_video_pe_sin.to(dtype),  # ca_video_rot_sin
        ca_audio_pe_cos.to(dtype),  # ca_audio_rot_cos
        ca_audio_pe_sin.to(dtype),  # ca_audio_rot_sin
        va.context_mask.to(dtype),  # encoder_attention_mask
        aa.context_mask.to(
            dtype
        ).clone(),  # audio_encoder_attention_mask (must be distinct tensor object)
        va.prompt_timestep.to(dtype),  # prompt_timestep
        aa.prompt_timestep.to(dtype),  # audio_prompt_timestep
    )

    print(f"\n  Generated {len(inputs)} input tensors", flush=True)
    for i, t in enumerate(inputs):
        print(f"    [{i:2d}] {str(t.shape):40s} {t.dtype}", flush=True)

    del ltx_model
    gc.collect()
    return inputs


# Module-level reference for get_model_fn closure
_CONFIG = None


def get_model_fn():
    """Build the TP-sharded backbone model on this rank.

    Called by the NxD tracing machinery. Must return (model, input_output_aliases).
    """
    from modeling_ltx23 import (
        NeuronLTX23TransformerBackbone,
        replace_sdpa_with_bmm,
    )
    from neuronx_distributed.parallel_layers import parallel_state

    replace_sdpa_with_bmm()

    # Build a minimal InferenceConfig-like object with the attributes the backbone needs
    class SimpleConfig:
        pass

    config = SimpleConfig()

    # neuron_config
    class NeuronConfigLike:
        tp_degree = TP_DEGREE
        world_size = TP_DEGREE
        torch_dtype = torch.bfloat16

    config.neuron_config = NeuronConfigLike()
    config.ltx_config_dict = _CONFIG  # The full config dict from safetensors

    backbone = NeuronLTX23TransformerBackbone(config)
    backbone.eval()

    return backbone, None


def main():
    global _CONFIG

    import torch_neuronx
    import torch_xla.core.xla_model as xm
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.parallel_layers.checkpointing import NXD_SKIP_RENDEZVOUS
    from neuronx_distributed.parallel_layers.utils import requires_init_pg_override

    rank = int(os.environ.get("RANK", "0"))

    # Initialize process group (torchrun sets env vars)
    if requires_init_pg_override():
        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=TP_DEGREE)
    torch.multiprocessing.set_sharing_strategy("file_system")

    if rank == 0:
        print("=" * 60, flush=True)
        print("LTX-2.3 Full Compile: TP=%d, LNC=2" % TP_DEGREE, flush=True)
        print("=" * 60, flush=True)

        print("\n[1/4] Loading model config...", flush=True)
        full_config = load_config_from_safetensors()
        _CONFIG = full_config
        tc = full_config["transformer"]
        num_layers = tc.get("num_layers", "?")

        print("\n[2/4] Precomputing inputs (%s blocks)..." % num_layers, flush=True)
        example_inputs = precompute_inputs(full_config)
        print(f"  Got {len(example_inputs)} inputs", flush=True)

        print("\n[3/4] Building TP-sharded model (rank 0)...", flush=True)
        os.makedirs(COMPILE_DIR, exist_ok=True)

        os.environ[NXD_SKIP_RENDEZVOUS] = "1"
        try:
            model, input_output_alias = get_model_fn()
        finally:
            del os.environ[NXD_SKIP_RENDEZVOUS]

        print(f"\n[4/4] Compiling {num_layers} blocks...", flush=True)
        rank_workdir = os.path.join(COMPILE_DIR, "_tp0")
        if os.path.exists(rank_workdir):
            shutil.rmtree(rank_workdir)

        compiler_args = [
            "--model-type=transformer",
            "-O1",
            "--auto-cast",
            "matmult",
            "--lnc",
            "2",
            "--tensorizer-options=--enable-ccop-compute-overlap",
        ]

        t0 = time.time()
        neff_filename, metaneff, flattener, packer, weights = (
            torch_neuronx.xla_impl.trace._trace(
                model,
                example_inputs,
                None,
                input_output_alias,
                rank_workdir,
                compiler_args,
                False,
            )
        )

        # Debug: print flattener layout and test extraction
        from torch_neuronx.xla_impl.structure import extract as struct_extract

        print(f"  Flattener layout: {flattener.layout}", flush=True)
        test_layout, test_uniques, test_constants = struct_extract(example_inputs)
        print(f"  Input layout:     {test_layout}", flush=True)
        print(f"  Match: {flattener.layout == test_layout}", flush=True)
        print(f"  Flattener exclude: {flattener.exclude}", flush=True)

        traced_model = torch_neuronx.xla_impl.trace.create_neuron_model(
            neff_filename,
            metaneff,
            flattener,
            packer,
            example_inputs,
            input_output_alias,
            weights,
        )

        tp_0_path = os.path.join(COMPILE_DIR, "tp_0.pt")
        torch.jit.save(traced_model, tp_0_path)
        elapsed = time.time() - t0
        size_gb = os.path.getsize(tp_0_path) / 1e9
        print(f"  Compiled and saved in {elapsed:.1f}s", flush=True)
        print(f"  Output: {tp_0_path} ({size_gb:.1f} GB)", flush=True)

    # All ranks must reach rendezvous
    xm.rendezvous("done-compilation")
    if rank == 0:
        print("\nAll ranks rendezvous'd. Compilation complete!", flush=True)


if __name__ == "__main__":
    main()
