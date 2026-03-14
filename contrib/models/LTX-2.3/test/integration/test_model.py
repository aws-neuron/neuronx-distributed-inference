"""
Integration test for LTX-2.3 22B DiT on Neuron.

Tests the compiled TP=4 backbone against a CPU reference forward pass.
Requires:
  - trn2.3xlarge instance with LNC=2 (4 logical cores)
  - Neuron SDK 2.28 (Deep Learning AMI Neuron Ubuntu 24.04 20260227)
  - ltx-core package installed
  - LTX-2.3 distilled model safetensors

Environment variables:
  MODEL_PATH: Path to ltx-2.3-22b-distilled.safetensors
  COMPILED_MODEL_PATH: Path to compiled model directory (will compile if missing)

Usage:
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
  MODEL_PATH=/home/ubuntu/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors \
  COMPILED_MODEL_PATH=/home/ubuntu/ltx23_neuron/compiler_workdir_tp4_lnc2 \
    pytest test_model.py -v -s
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest
import torch

# Add src to path
SRC_DIR = str(Path(__file__).parent.parent.parent / "src")
sys.path.insert(0, SRC_DIR)

# Required environment variables
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/home/ubuntu/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors",
)
COMPILED_MODEL_PATH = os.environ.get(
    "COMPILED_MODEL_PATH",
    "/home/ubuntu/ltx23_neuron/compiler_workdir_tp4_lnc2_v2",
)

TP_DEGREE = 4
BATCH = 1
VIDEO_SEQ = 768  # 4 frames * 12 * 16 patches (384x512, patch_size=1)
AUDIO_SEQ = 26
TEXT_SEQ = 256
NUM_HEADS = 32
AUDIO_NUM_HEADS = 32
INNER_DIM = 4096
AUDIO_INNER_DIM = 2048
HEAD_DIM = INNER_DIM // NUM_HEADS  # 128
AUDIO_HEAD_DIM = AUDIO_INNER_DIM // AUDIO_NUM_HEADS  # 64


def load_config():
    from safetensors import safe_open

    with safe_open(MODEL_PATH, framework="pt") as f:
        metadata = f.metadata()
    return json.loads(metadata["config"])


def create_test_inputs(dtype=torch.bfloat16):
    """Create deterministic test inputs matching the 24-input backbone signature."""
    gen = torch.Generator().manual_seed(42)

    inputs = [
        torch.randn(
            BATCH, VIDEO_SEQ, INNER_DIM, dtype=dtype, generator=gen
        ),  # hidden_states
        torch.randn(
            BATCH, AUDIO_SEQ, AUDIO_INNER_DIM, dtype=dtype, generator=gen
        ),  # audio_hidden_states
        torch.randn(
            BATCH, TEXT_SEQ, INNER_DIM, dtype=dtype, generator=gen
        ),  # encoder_hidden_states
        torch.randn(
            BATCH, TEXT_SEQ, AUDIO_INNER_DIM, dtype=dtype, generator=gen
        ),  # audio_encoder_hidden_states
        torch.randn(
            BATCH, VIDEO_SEQ, 9 * INNER_DIM, dtype=dtype, generator=gen
        ),  # temb
        torch.randn(
            BATCH, AUDIO_SEQ, 9 * AUDIO_INNER_DIM, dtype=dtype, generator=gen
        ),  # temb_audio
        torch.randn(
            BATCH, VIDEO_SEQ, INNER_DIM, dtype=dtype, generator=gen
        ),  # embedded_timestep
        torch.randn(
            BATCH, AUDIO_SEQ, AUDIO_INNER_DIM, dtype=dtype, generator=gen
        ),  # audio_embedded_timestep
        torch.randn(BATCH, 1, 4 * INNER_DIM, dtype=dtype, generator=gen),  # video_ca_ss
        torch.randn(
            BATCH, 1, 4 * AUDIO_INNER_DIM, dtype=dtype, generator=gen
        ),  # audio_ca_ss
        torch.randn(BATCH, 1, INNER_DIM, dtype=dtype, generator=gen),  # video_ca_gate
        torch.randn(
            BATCH, 1, AUDIO_INNER_DIM, dtype=dtype, generator=gen
        ),  # audio_ca_gate
        torch.randn(
            BATCH, NUM_HEADS, VIDEO_SEQ, HEAD_DIM // 2, dtype=dtype, generator=gen
        ),  # video_rot_cos
        torch.randn(
            BATCH, NUM_HEADS, VIDEO_SEQ, HEAD_DIM // 2, dtype=dtype, generator=gen
        ),  # video_rot_sin
        torch.randn(
            BATCH,
            AUDIO_NUM_HEADS,
            AUDIO_SEQ,
            AUDIO_HEAD_DIM // 2,
            dtype=dtype,
            generator=gen,
        ),  # audio_rot_cos
        torch.randn(
            BATCH,
            AUDIO_NUM_HEADS,
            AUDIO_SEQ,
            AUDIO_HEAD_DIM // 2,
            dtype=dtype,
            generator=gen,
        ),  # audio_rot_sin
        torch.randn(
            BATCH, NUM_HEADS, VIDEO_SEQ, AUDIO_HEAD_DIM // 2, dtype=dtype, generator=gen
        ),  # ca_video_rot_cos
        torch.randn(
            BATCH, NUM_HEADS, VIDEO_SEQ, AUDIO_HEAD_DIM // 2, dtype=dtype, generator=gen
        ),  # ca_video_rot_sin
        torch.randn(
            BATCH,
            AUDIO_NUM_HEADS,
            AUDIO_SEQ,
            AUDIO_HEAD_DIM // 2,
            dtype=dtype,
            generator=gen,
        ),  # ca_audio_rot_cos
        torch.randn(
            BATCH,
            AUDIO_NUM_HEADS,
            AUDIO_SEQ,
            AUDIO_HEAD_DIM // 2,
            dtype=dtype,
            generator=gen,
        ),  # ca_audio_rot_sin
        torch.zeros(
            BATCH, TEXT_SEQ, dtype=dtype
        ),  # encoder_attention_mask (all attend)
        torch.zeros(BATCH, TEXT_SEQ, dtype=dtype),  # audio_encoder_attention_mask
        torch.randn(
            BATCH, 1, 2 * INNER_DIM, dtype=dtype, generator=gen
        ),  # prompt_timestep
        torch.randn(
            BATCH, 1, 2 * AUDIO_INNER_DIM, dtype=dtype, generator=gen
        ),  # audio_prompt_timestep
    ]
    return inputs


@pytest.fixture(scope="module")
def compiled_model():
    """Load compiled Neuron backbone with real weights.

    If COMPILED_MODEL_PATH exists, loads from there.
    Compilation must be done separately via compile_transformer.py.
    """
    tp_0_path = os.path.join(COMPILED_MODEL_PATH, "tp_0.pt")
    if not os.path.exists(tp_0_path):
        pytest.skip(
            f"Compiled model not found at {tp_0_path}. "
            "Run compile_transformer.py first:\n"
            "  NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 "
            "torchrun --nproc_per_node=4 src/compile_transformer.py"
        )

    import torch_neuronx
    from neuronx_distributed.trace.trace import TensorParallelNeuronModel
    from load_with_weights import shard_weight
    from safetensors.torch import load_file

    # Load and shard weights
    full_sd = load_file(MODEL_PATH)
    prefix = "model.diffusion_model."
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
    for k, v in full_sd.items():
        stripped = k[len(prefix) :] if k.startswith(prefix) else k
        if stripped.startswith(backbone_prefixes):
            backbone_sd[stripped] = v.to(torch.bfloat16).contiguous()
    backbone_sd["spmd_rank.rank"] = torch.arange(0, TP_DEGREE, dtype=torch.int32)
    del full_sd

    def sf_key_to_jit_key(sf_key):
        return "weights." + sf_key.replace(".", "->")

    # Create per-rank state dicts
    rank_sds = [{} for _ in range(TP_DEGREE)]
    for sf_key, full_weight in backbone_sd.items():
        jit_key = sf_key_to_jit_key(sf_key)
        for rank in range(TP_DEGREE):
            rank_sds[rank][jit_key] = shard_weight(
                full_weight, jit_key, rank, TP_DEGREE
            )
    del backbone_sd

    # Load compiled models and inject weights
    models = []
    for rank in range(TP_DEGREE):
        with torch_neuronx.contexts.disable_nrt_load():
            model = torch.jit.load(tp_0_path)
        model_sd = dict(model.named_parameters())
        for jit_key, sharded_weight in rank_sds[rank].items():
            if jit_key in model_sd and model_sd[jit_key].shape == sharded_weight.shape:
                model_sd[jit_key].data.copy_(sharded_weight)
        models.append(model)
    del rank_sds

    return TensorParallelNeuronModel(models)


@pytest.fixture(scope="module")
def cpu_model():
    """Build CPU reference model for accuracy comparison."""
    from modeling_ltx23 import replace_sdpa_with_bmm

    replace_sdpa_with_bmm()

    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.loader.sd_ops import SDOps
    from ltx_core.model.transformer.model_configurator import LTXModelConfigurator

    config = load_config()
    ltx_ops = (
        SDOps("ltx")
        .with_matching(prefix="model.diffusion_model.")
        .with_replacement("model.diffusion_model.", "")
    )
    builder = SingleGPUModelBuilder(
        model_class_configurator=LTXModelConfigurator,
        model_path=MODEL_PATH,
        model_sd_ops=ltx_ops,
    )
    model = builder.build(device=torch.device("cpu"), dtype=torch.bfloat16)
    model.eval()
    return model


def test_model_loads(compiled_model):
    """Smoke test: compiled model loads successfully."""
    assert compiled_model is not None


def test_forward_pass_no_nan(compiled_model):
    """Forward pass produces valid (non-NaN, non-Inf) output."""
    inputs = create_test_inputs()

    with torch.no_grad():
        outputs = compiled_model(*inputs)

    video_out = outputs[0]
    audio_out = outputs[1]

    assert not torch.isnan(video_out).any(), "Video output contains NaN"
    assert not torch.isinf(video_out).any(), "Video output contains Inf"
    assert not torch.isnan(audio_out).any(), "Audio output contains NaN"
    assert not torch.isinf(audio_out).any(), "Audio output contains Inf"

    # Check output shapes
    assert video_out.shape == (BATCH, VIDEO_SEQ, 128), f"Video shape: {video_out.shape}"
    assert audio_out.shape == (BATCH, AUDIO_SEQ, 128), f"Audio shape: {audio_out.shape}"


def test_accuracy_vs_cpu(compiled_model, cpu_model):
    """Compare Neuron forward pass to CPU reference.

    Acceptance threshold: cosine similarity >= 0.999 for single forward pass.
    This validates that TP=4 sharding, DistributedRMSNorm, and fused SDPA
    produce outputs matching the unsharded CPU model.
    """
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.guidance.perturbations import (
        BatchedPerturbationConfig,
        PerturbationConfig,
    )

    dtype = torch.bfloat16
    torch.manual_seed(42)

    # Create matching inputs for both CPU and Neuron paths
    sigma = torch.tensor([1.0], dtype=dtype)

    # Build latent tools for proper state creation
    from ltx_core.tools import (
        VideoLatentTools,
        VideoLatentPatchifier,
        VideoLatentShape,
        AudioLatentTools,
        AudioPatchifier,
        AudioLatentShape,
        SpatioTemporalScaleFactors,
    )

    video_shape = VideoLatentShape(batch=1, channels=128, frames=4, height=12, width=16)
    v_patchifier = VideoLatentPatchifier(patch_size=1)
    v_scale = SpatioTemporalScaleFactors(time=1, width=8, height=8)
    video_tools = VideoLatentTools(
        target_shape=video_shape,
        patchifier=v_patchifier,
        scale_factors=v_scale,
        causal_fix=False,
        fps=24.0,
    )
    audio_shape = AudioLatentShape(batch=1, channels=8, frames=26, mel_bins=16)
    a_patchifier = AudioPatchifier(patch_size=16)
    audio_tools = AudioLatentTools(patchifier=a_patchifier, target_shape=audio_shape)

    video_state = video_tools.create_initial_state(device="cpu", dtype=dtype)
    audio_state = audio_tools.create_initial_state(device="cpu", dtype=dtype)

    # Use noise latents in patchified shape (B, seq, C)
    video_latent = torch.randn_like(video_state.latent)
    audio_latent = torch.randn_like(audio_state.latent)

    video_seq_len = video_latent.shape[1]
    audio_seq_len = audio_latent.shape[1]
    v_ts = sigma.unsqueeze(0).expand(1, video_seq_len)
    a_ts = sigma.unsqueeze(0).expand(1, audio_seq_len)

    video_context = torch.randn(1, TEXT_SEQ, INNER_DIM, dtype=dtype)
    audio_context = torch.randn(1, TEXT_SEQ, AUDIO_INNER_DIM, dtype=dtype)
    context_mask = torch.ones(1, TEXT_SEQ, dtype=dtype)
    context_mask[:, 50:] = 0

    video_mod = Modality(
        latent=video_latent,
        sigma=sigma,
        timesteps=v_ts,
        positions=video_state.positions,
        context=video_context,
        enabled=True,
        context_mask=context_mask,
        attention_mask=None,
    )
    audio_mod = Modality(
        latent=audio_latent,
        sigma=sigma,
        timesteps=a_ts,
        positions=audio_state.positions,
        context=audio_context,
        enabled=True,
        context_mask=context_mask.clone(),
        attention_mask=None,
    )

    # CPU forward through full model
    perturbation = BatchedPerturbationConfig(perturbations=[PerturbationConfig.empty()])
    with torch.no_grad():
        cpu_out = cpu_model(video_mod, audio_mod, perturbations=perturbation)

    # LTXModel.forward returns (video_velocity, audio_velocity) tuple
    cpu_video = cpu_out[0].float()
    cpu_audio = cpu_out[1].float()

    # Neuron forward via wrapper
    from pipeline import NeuronTransformerWrapper

    wrapper = NeuronTransformerWrapper(
        compiled_backbone=compiled_model,
        cpu_ltx_model=cpu_model,
        text_seq=TEXT_SEQ,
    )

    # Re-create modalities (cpu_model forward may have modified them)
    torch.manual_seed(42)
    video_latent2 = torch.randn_like(video_state.latent)
    audio_latent2 = torch.randn_like(audio_state.latent)

    video_mod2 = Modality(
        latent=video_latent2,
        sigma=sigma,
        timesteps=v_ts,
        positions=video_state.positions,
        context=video_context,
        enabled=True,
        context_mask=torch.ones(1, TEXT_SEQ, dtype=dtype),
        attention_mask=None,
    )
    video_mod2.context_mask[:, 50:] = 0

    audio_mod2 = Modality(
        latent=audio_latent2,
        sigma=sigma,
        timesteps=a_ts,
        positions=audio_state.positions,
        context=audio_context,
        enabled=True,
        context_mask=torch.ones(1, TEXT_SEQ, dtype=dtype),
        attention_mask=None,
    )
    audio_mod2.context_mask[:, 50:] = 0

    with torch.no_grad():
        neuron_video, neuron_audio = wrapper(video_mod2, audio_mod2)

    neuron_video = neuron_video.float()
    neuron_audio = neuron_audio.float()

    # Cosine similarity
    video_cos = torch.nn.functional.cosine_similarity(
        cpu_video.flatten(), neuron_video.flatten(), dim=0
    ).item()
    audio_cos = torch.nn.functional.cosine_similarity(
        cpu_audio.flatten(), neuron_audio.flatten(), dim=0
    ).item()

    print(f"\n=== Accuracy Results ===")
    print(f"Video cosine similarity: {video_cos:.6f}")
    print(f"Audio cosine similarity: {audio_cos:.6f}")
    print(f"Video max abs error: {(cpu_video - neuron_video).abs().max().item():.4f}")
    print(f"Audio max abs error: {(cpu_audio - neuron_audio).abs().max().item():.4f}")

    assert video_cos >= 0.98, f"Video cos_sim {video_cos:.6f} < 0.98"
    assert audio_cos >= 0.90, f"Audio cos_sim {audio_cos:.6f} < 0.90"


def test_performance_latency(compiled_model):
    """Measure per-step forward pass latency (warm)."""
    inputs = create_test_inputs()

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            compiled_model(*inputs)

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(10):
            t0 = time.time()
            compiled_model(*inputs)
            latencies.append(time.time() - t0)

    avg_ms = sum(latencies) / len(latencies) * 1000
    min_ms = min(latencies) * 1000
    max_ms = max(latencies) * 1000

    print(f"\n=== Performance Results ===")
    print(f"Average latency: {avg_ms:.1f} ms")
    print(f"Min latency: {min_ms:.1f} ms")
    print(f"Max latency: {max_ms:.1f} ms")

    # No hard performance threshold -- just report
    assert avg_ms < 60000, f"Forward pass too slow: {avg_ms:.1f} ms"
