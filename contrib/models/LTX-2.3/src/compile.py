#!/usr/bin/env python3
"""
LTX-2.3 Unified Compilation Script
====================================
Compiles all components of the LTX-2.3 model for Neuron:
  - transformer: DiT backbone (full-res or half-res)
  - encoder: Gemma3 12B text encoder
  - vae: Video decoder (TP-sharded tiled decode)

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

  # Compile full-resolution DiT backbone (384x512, VIDEO_SEQ=768)
  NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \\
    torchrun --nproc_per_node=4 compile.py transformer

  # Compile half-resolution DiT backbone (192x256, VIDEO_SEQ=192) for two-stage mode
  NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \\
    torchrun --nproc_per_node=4 compile.py transformer --halfres

  # Compile Gemma3 encoder (uses parallel_model_trace, no torchrun needed)
  NEURON_FUSE_SOFTMAX=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \\
    python3 compile.py encoder

  # Compile VAE decoder (TP=4, tiled)
  NEURON_RT_VISIBLE_CORES=0-3 python3 compile.py vae
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# Transformer (DiT backbone) compilation
# ============================================================================

# Default environment for transformer compilation
os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
os.environ.setdefault("NEURON_CUSTOM_SILU", "1")
os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Architecture constants (from safetensors metadata)
NUM_HEADS = 32
AUDIO_NUM_HEADS = 32
INNER_DIM = 4096  # NUM_HEADS * 128
AUDIO_INNER_DIM = 2048  # AUDIO_NUM_HEADS * 64
AUDIO_CA_DIM = 2048  # audio_cross_attention_dim

# Module-level reference for get_model_fn closure
_CONFIG = None
_TP_DEGREE = 4


def load_config_from_safetensors(model_path):
    """Load the transformer config from safetensors metadata."""
    from safetensors import safe_open

    with safe_open(model_path, framework="pt") as f:
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


def precompute_inputs(
    config, video_seq, latent_h, latent_w, text_seq=256, audio_seq=26
):
    """Build example inputs using the native ltx-core preprocessing."""
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
    batch = 1

    # Build full unsharded model on CPU
    ltx_model = LTXModelConfigurator.from_config(config)
    ltx_model = ltx_model.to(dtype=dtype)
    ltx_model.eval()

    torch.manual_seed(123)

    video_shape = VideoLatentShape(
        batch=batch, channels=128, frames=4, height=latent_h, width=latent_w
    )
    v_patchifier = VideoLatentPatchifier(patch_size=1)
    v_scale = SpatioTemporalScaleFactors.default()
    video_tools = VideoLatentTools(
        target_shape=video_shape,
        patchifier=v_patchifier,
        scale_factors=v_scale,
        causal_fix=False,
        fps=24.0,
    )
    video_state = video_tools.create_initial_state(device="cpu", dtype=dtype)

    audio_shape = AudioLatentShape(
        batch=batch, channels=8, frames=audio_seq, mel_bins=16
    )
    a_patchifier = AudioPatchifier(patch_size=16)
    audio_tools = AudioLatentTools(patchifier=a_patchifier, target_shape=audio_shape)
    audio_state = audio_tools.create_initial_state(device="cpu", dtype=dtype)

    # Use random noise (not zeros) for representative input magnitudes
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

    sigma = torch.tensor([1.0], dtype=dtype)
    v_ts = sigma.unsqueeze(1).expand(batch, video_latent.shape[1])
    a_ts = sigma.unsqueeze(1).expand(batch, audio_latent.shape[1])

    ctx_v = torch.randn(batch, text_seq, INNER_DIM, dtype=dtype)
    ctx_a = torch.randn(batch, text_seq, AUDIO_INNER_DIM, dtype=dtype)
    ctx_mask = torch.ones(batch, text_seq, dtype=dtype)
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

    print(f"\n  Generated {len(inputs)} input tensors", flush=True)
    for i, t in enumerate(inputs):
        print(f"    [{i:2d}] {str(t.shape):40s} {t.dtype}", flush=True)

    del ltx_model
    gc.collect()
    return inputs


def get_transformer_model_fn():
    """Build the TP-sharded backbone model on this rank."""
    from modeling_ltx23 import (
        NeuronLTX23TransformerBackbone,
        replace_sdpa_with_bmm,
    )
    from neuronx_distributed.parallel_layers import parallel_state

    replace_sdpa_with_bmm()

    class SimpleConfig:
        pass

    config = SimpleConfig()

    class NeuronConfigLike:
        tp_degree = _TP_DEGREE
        world_size = _TP_DEGREE
        torch_dtype = torch.bfloat16

    config.neuron_config = NeuronConfigLike()
    config.ltx_config_dict = _CONFIG

    backbone = NeuronLTX23TransformerBackbone(config)
    backbone.eval()

    return backbone, None


def compile_transformer(args):
    """Compile the DiT transformer backbone."""
    global _CONFIG, _TP_DEGREE

    import torch_neuronx
    import torch_xla.core.xla_model as xm
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.parallel_layers.checkpointing import NXD_SKIP_RENDEZVOUS
    from neuronx_distributed.parallel_layers.utils import requires_init_pg_override

    _TP_DEGREE = args.tp_degree

    # Resolution configuration
    if args.halfres:
        latent_h = 6  # 192 / 32
        latent_w = 8  # 256 / 32
        video_seq = 192  # 4 * 6 * 8
        res_label = "half-res (192x256)"
        default_compile_dir = (
            "/home/ubuntu/ltx23_neuron/compiler_workdir_tp4_lnc2_halfres"
        )
    else:
        latent_h = args.latent_h
        latent_w = args.latent_w
        video_seq = 4 * latent_h * latent_w
        res_label = f"full-res ({latent_h * 32}x{latent_w * 32})"
        default_compile_dir = "/home/ubuntu/ltx23_neuron/compiler_workdir_tp4_lnc2_v2"

    compile_dir = args.compile_dir or default_compile_dir

    rank = int(os.environ.get("RANK", "0"))

    if requires_init_pg_override():
        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=_TP_DEGREE)
    torch.multiprocessing.set_sharing_strategy("file_system")

    if rank == 0:
        print("=" * 60, flush=True)
        print(f"LTX-2.3 Compile: {res_label}, TP={_TP_DEGREE}, LNC=2", flush=True)
        print("=" * 60, flush=True)

        print("\n[1/4] Loading model config...", flush=True)
        full_config = load_config_from_safetensors(args.model_path)
        _CONFIG = full_config
        tc = full_config["transformer"]
        num_layers = tc.get("num_layers", "?")

        print(
            f"\n[2/4] Precomputing inputs ({num_layers} blocks, video_seq={video_seq})...",
            flush=True,
        )
        example_inputs = precompute_inputs(full_config, video_seq, latent_h, latent_w)
        print(f"  Got {len(example_inputs)} inputs", flush=True)

        print("\n[3/4] Building TP-sharded model (rank 0)...", flush=True)
        os.makedirs(compile_dir, exist_ok=True)

        os.environ[NXD_SKIP_RENDEZVOUS] = "1"
        try:
            model, input_output_alias = get_transformer_model_fn()
        finally:
            del os.environ[NXD_SKIP_RENDEZVOUS]

        print(f"\n[4/4] Compiling {num_layers} blocks...", flush=True)
        rank_workdir = os.path.join(compile_dir, "_tp0")
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
        traced_model = torch_neuronx.trace(
            model,
            example_inputs,
            compiler_workdir=rank_workdir,
            compiler_args=compiler_args,
            inline_weights_to_neff=False,
        )

        tp_0_path = os.path.join(compile_dir, "tp_0.pt")
        torch.jit.save(traced_model, tp_0_path)
        elapsed = time.time() - t0
        size_gb = os.path.getsize(tp_0_path) / 1e9
        print(f"  Compiled and saved in {elapsed:.1f}s", flush=True)
        print(f"  Output: {tp_0_path} ({size_gb:.1f} GB)", flush=True)

    xm.rendezvous("done-compilation")
    if rank == 0:
        print("\nAll ranks rendezvous'd. Compilation complete!", flush=True)


# ============================================================================
# Gemma3 encoder compilation
# ============================================================================


def compile_encoder(args):
    """Compile the Gemma3 12B text encoder for Neuron TP=4."""
    import torch_neuronx
    from neuronx_distributed.trace import parallel_model_trace, parallel_model_save

    tp_degree = args.tp_degree
    seq_len = args.seq_len
    compile_dir = args.compile_dir or "/home/ubuntu/gemma3_encoder_compiled"

    print("=" * 60)
    print(f"Compiling Gemma3 encoder (TP={tp_degree}, seq={seq_len})")
    print("=" * 60)

    os.makedirs(compile_dir, exist_ok=True)

    def get_model_fn(tp_degree=tp_degree):
        from modeling_gemma3_encoder import Gemma3TextEncoderModel

        model = Gemma3TextEncoderModel(
            vocab_size=262208,
            hidden_size=3840,
            num_hidden_layers=48,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=256,
            intermediate_size=15360,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000.0,
            max_position_embeddings=131072,
            query_pre_attn_scalar=256,
            pad_token_id=0,
            dtype=torch.bfloat16,
        )
        model = model.to(dtype=torch.bfloat16)
        model.eval()
        return model, None

    input_ids = torch.zeros(1, seq_len, dtype=torch.int64)
    attention_mask = torch.ones(1, seq_len, dtype=torch.int64)

    # Tensorizer flags for 3.1x speedup (644ms vs 2000ms)
    compiler_args = (
        "--model-type=transformer -O1 --auto-cast=none --lnc=2 "
        "--tensorizer-options='--enable-ccop-compute-overlap "
        "--cc-pipeline-tiling-factor=1 "
        "--vectorize-strided-dma "
        "--enable-scalar-dge-vectorization'"
    )
    os.environ["NEURON_CC_FLAGS"] = compiler_args
    print(f"  Compiler flags: {compiler_args}")

    t0 = time.time()
    traced = parallel_model_trace(
        get_model_fn,
        (input_ids, attention_mask),
        tp_degree=tp_degree,
        compiler_workdir=os.path.join(compile_dir, "compiler_workdir"),
        compiler_args=compiler_args,
        inline_weights_to_neff=False,
    )
    elapsed = time.time() - t0
    print(f"  Compile: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    parallel_model_save(traced, compile_dir)
    tp0_size = os.path.getsize(os.path.join(compile_dir, "tp_0.pt")) / 1e9
    print(f"  Saved tp_0.pt: {tp0_size:.2f} GB")

    # Quick forward test with random weights
    print("\nLoading for forward test...")
    from neuronx_distributed.trace.trace import (
        _mock_parallel_state,
        init_on_device,
        get_sharded_checkpoint,
        replace_weights,
        TensorParallelNeuronModel,
    )

    _mock_parallel_state(1, 0)
    with init_on_device(torch.device("cpu")):
        ref_model, _ = get_model_fn()
    checkpoint = ref_model.state_dict()
    total_params = sum(v.numel() for v in checkpoint.values())
    print(f"  Checkpoint: {len(checkpoint)} keys, {total_params / 1e9:.2f} B params")
    del ref_model
    gc.collect()

    models = []
    for rank in range(tp_degree):
        t0r = time.time()
        ckpt = {k: v.clone() for k, v in checkpoint.items()}
        _mock_parallel_state(tp_degree, rank)
        with init_on_device(torch.device("meta")):
            model, _ = get_model_fn()
        get_sharded_checkpoint(ckpt, model, rank, tp_degree)
        with torch_neuronx.contexts.disable_nrt_load():
            traced_model = torch.jit.load(os.path.join(compile_dir, "tp_0.pt"))
        replace_weights(traced_model, ckpt)
        models.append(traced_model)
        print(f"  [rank {rank}] {time.time() - t0r:.1f}s")
        gc.collect()
    del checkpoint
    gc.collect()

    compiled = TensorParallelNeuronModel(models)
    print(f"  All {tp_degree} ranks loaded")

    print("\nForward pass...")
    _ = compiled(input_ids, attention_mask)  # warmup
    t0 = time.time()
    output = compiled(input_ids, attention_mask)
    elapsed = time.time() - t0

    expected = (1, seq_len, 3840, 49)
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Output shape: {tuple(output.shape)} (expected {expected})")
    print(f"  NaN: {'FAIL' if torch.isnan(output).any() else 'PASS'}")
    print(f"  Inf: {'FAIL' if torch.isinf(output).any() else 'PASS'}")
    if tuple(output.shape) == expected:
        print("\n  *** GEMMA3 ENCODER COMPILE + FORWARD: PASSED ***")
    else:
        print("\n  *** SHAPE MISMATCH -- FAILED ***")


# ============================================================================
# VAE decoder compilation
# ============================================================================


def compile_vae(args):
    """Compile the TP-sharded VAE decoder."""
    from modeling_vae_23 import compile_vae_decoder

    compile_dir = args.compile_dir or "/home/ubuntu/ltx23_vae_tp4"

    # Verify tile area constraint
    latent_h = args.height // 32
    latent_w = args.width // 32
    area = latent_h * latent_w
    if area > 64:
        print(
            f"ERROR: Tile area {latent_h}x{latent_w} = {area} exceeds 64-element SRAM limit"
        )
        print("Reduce tile dimensions. Maximum compilable tiles:")
        print("  8x8 (256x256 px), 4x16 (128x512 px)")
        sys.exit(1)

    compile_vae_decoder(
        tp_degree=args.tp_degree,
        tile_height=args.height,
        tile_width=args.width,
        num_frames=args.num_frames,
        output_dir=compile_dir,
        compiler_workdir=args.compiler_workdir or "/home/ubuntu/compiler_workdir_vae23",
        model_path=args.model_path,
    )


# ============================================================================
# Main entry point with subcommands
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="LTX-2.3 unified compilation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full-res transformer (requires torchrun)
  torchrun --nproc_per_node=4 compile.py transformer

  # Half-res transformer for two-stage mode
  torchrun --nproc_per_node=4 compile.py transformer --halfres

  # Gemma3 encoder (no torchrun needed)
  python3 compile.py encoder

  # VAE decoder
  python3 compile.py vae
""",
    )
    subparsers = parser.add_subparsers(dest="component", help="Component to compile")
    subparsers.required = True

    # --- transformer subcommand ---
    p_trans = subparsers.add_parser("transformer", help="Compile DiT backbone")
    p_trans.add_argument(
        "--halfres",
        action="store_true",
        help="Compile half-resolution (192x256) for two-stage mode",
    )
    p_trans.add_argument(
        "--latent-h", type=int, default=12, help="Latent height (default: 12 for 384px)"
    )
    p_trans.add_argument(
        "--latent-w", type=int, default=16, help="Latent width (default: 16 for 512px)"
    )
    p_trans.add_argument(
        "--tp-degree", type=int, default=4, help="TP degree (default: 4)"
    )
    p_trans.add_argument(
        "--model-path",
        default="/home/ubuntu/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors",
        help="Path to LTX-2.3 safetensors file",
    )
    p_trans.add_argument("--compile-dir", default=None, help="Output directory")
    p_trans.set_defaults(func=compile_transformer)

    # --- encoder subcommand ---
    p_enc = subparsers.add_parser("encoder", help="Compile Gemma3 12B text encoder")
    p_enc.add_argument(
        "--tp-degree", type=int, default=4, help="TP degree (default: 4)"
    )
    p_enc.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    p_enc.add_argument("--compile-dir", default=None, help="Output directory")
    p_enc.set_defaults(func=compile_encoder)

    # --- vae subcommand ---
    p_vae = subparsers.add_parser("vae", help="Compile TP-sharded VAE decoder")
    p_vae.add_argument(
        "--height", type=int, default=128, help="Tile pixel height (default: 128)"
    )
    p_vae.add_argument(
        "--width", type=int, default=512, help="Tile pixel width (default: 512)"
    )
    p_vae.add_argument("--num-frames", type=int, default=121, help="Number of frames")
    p_vae.add_argument(
        "--tp-degree", type=int, default=4, help="TP degree (default: 4)"
    )
    p_vae.add_argument("--compile-dir", default=None, help="Output directory")
    p_vae.add_argument("--compiler-workdir", default=None, help="Compiler working dir")
    p_vae.add_argument(
        "--model-path",
        default="/home/ubuntu/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors",
        help="Path to LTX-2.3 safetensors file",
    )
    p_vae.set_defaults(func=compile_vae)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
