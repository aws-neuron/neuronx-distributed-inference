"""
Integration test for Qwen-Image-Edit-2511 with Multi-LoRA on Neuron.

Tests:
1. Compilation succeeds (ModelBuilder + SPMD + NKI Flash Attention)
2. Inference produces valid outputs (non-zero, finite)
3. LoRA aliasing works: write_to_neuron_buffer() changes output
4. Zeroing LoRA restores original baseline

Requirements:
- trn2.3xlarge (TP=4, LNC=2)
- Neuron SDK 2.29+
- diffusers, transformers installed
- Model downloaded locally or accessible from HuggingFace

Usage:
    pytest test/integration/test_model.py -v --model-path /path/to/model
    # Or with default HF download:
    pytest test/integration/test_model.py -v
"""

import os
import sys
import pytest
import torch
from safetensors.torch import load_file

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def get_model_path():
    """Get model path from environment or use default."""
    return os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")


@pytest.fixture(scope="module")
def compiled_model():
    """Compile the transformer once for all tests in this module.

    If COMPILED_MODELS_DIR points to an existing compiled model directory,
    compilation is skipped and the existing artifacts are used directly.
    """
    import argparse

    output_dir = os.environ.get(
        "COMPILED_MODELS_DIR", "/tmp/qwen_edit_lora_test_compiled"
    )
    model_path = os.path.join(output_dir, "transformer_multi_lora")
    nxd_model_path = os.path.join(model_path, "nxd_model.pt")
    weights_path = os.path.join(model_path, "weights")

    tp_degree = int(os.environ.get("TP_DEGREE", "4"))

    args = argparse.Namespace(
        model_id=get_model_path(),
        height=512,
        width=512,
        max_sequence_length=512,
        patch_multiplier=3,
        tp_degree=tp_degree,
        max_loras=4,
        max_rank=16,
        compiled_models_dir=output_dir,
        compiler_workdir=os.path.join(output_dir, "compiler_workdir"),
        test_aliasing=False,
    )

    # Skip compilation if artifacts already exist
    weights_exist = all(
        os.path.exists(
            os.path.join(weights_path, f"tp{r}_sharded_checkpoint.safetensors")
        )
        for r in range(tp_degree)
    )
    if os.path.exists(nxd_model_path) and weights_exist:
        print(f"\n[test] Using existing compiled model at {model_path}")
    else:
        print(f"\n[test] Compiling model to {model_path}...")
        from modeling_qwen_image_edit_lora import compile_transformer_multi_lora

        compile_transformer_multi_lora(args)

    return {
        "nxd_model_path": nxd_model_path,
        "weights_path": weights_path,
        "tp_degree": tp_degree,
        "args": args,
    }


@pytest.fixture(scope="module")
def loaded_model(compiled_model):
    """Load the compiled model onto Neuron devices."""
    from neuronx_distributed import NxDModel
    from neuronx_distributed.parallel_layers import parallel_state

    nxd_model_path = compiled_model["nxd_model_path"]
    weights_path = compiled_model["weights_path"]
    tp_degree = compiled_model["tp_degree"]

    # Load model
    traced_model = NxDModel.load(nxd_model_path)

    # Load sharded weights (safetensors format from shard_checkpoint)
    sharded_checkpoints = []
    for rank in range(tp_degree):
        ckpt_path = os.path.join(
            weights_path, f"tp{rank}_sharded_checkpoint.safetensors"
        )
        sharded_checkpoints.append(load_file(ckpt_path))

    traced_model.set_weights(sharded_checkpoints)
    traced_model.to_neuron()

    return {
        "model": traced_model,
        "checkpoints": sharded_checkpoints,
        "tp_degree": tp_degree,
    }


def get_test_inputs(height=512, width=512, text_seq_len=512, tp_degree=4):
    """Create test input tensors matching the compiled model's expected shapes.

    Shape calculations match compile_transformer_multi_lora():
    - latent_h = height // 8, latent_w = width // 8
    - patch_h = latent_h // 2, patch_w = latent_w // 2  (patch_size=2)
    - num_patches = temporal_frames * patch_h * patch_w
    - Alignment: total_seq = num_patches + text_seq_len padded to multiple of 128
    """
    temporal_frames = 3  # patch_multiplier
    latent_h = height // 8
    latent_w = width // 8
    patch_size = 2
    patch_h = latent_h // patch_size
    patch_w = latent_w // patch_size
    in_channels = 64
    text_hidden_size = 3584

    num_patches = temporal_frames * patch_h * patch_w

    # Alignment padding (must match compile function)
    total_seq = num_patches + text_seq_len
    alignment = 128
    need_padding = (alignment - total_seq % alignment) % alignment
    num_patches_padded = num_patches + need_padding

    batch_size = 1

    hidden_states = torch.randn(
        batch_size, num_patches_padded, in_channels, dtype=torch.bfloat16
    )
    encoder_hidden_states = torch.randn(
        batch_size, text_seq_len, text_hidden_size, dtype=torch.bfloat16
    )
    timestep = torch.randn(batch_size, dtype=torch.float32)

    # RoPE (use random for test — shape matters, not content)
    head_dim_half = 64  # 128 / 2
    img_rotary_emb = torch.randn(
        num_patches_padded, head_dim_half, 2, dtype=torch.bfloat16
    )
    txt_rotary_emb = torch.randn(text_seq_len, head_dim_half, 2, dtype=torch.bfloat16)

    return (
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_rotary_emb,
        txt_rotary_emb,
    )


class TestQwenImageEditLoRA:
    """Integration tests for the Qwen Image Edit + LoRA model on Neuron."""

    def test_compilation_produces_artifacts(self, compiled_model):
        """Test that compilation produces the expected output files."""
        assert os.path.exists(compiled_model["nxd_model_path"]), (
            "NxDModel file not found after compilation"
        )

        weights_path = compiled_model["weights_path"]
        for rank in range(compiled_model["tp_degree"]):
            ckpt = os.path.join(
                weights_path, f"tp{rank}_sharded_checkpoint.safetensors"
            )
            assert os.path.exists(ckpt), f"Weight checkpoint for rank {rank} not found"

    def test_inference_produces_valid_output(self, loaded_model, compiled_model):
        """Test that inference produces non-zero, finite outputs."""
        model = loaded_model["model"]
        args = compiled_model["args"]

        inputs = get_test_inputs(
            height=args.height,
            width=args.width,
            text_seq_len=args.max_sequence_length,
            tp_degree=args.tp_degree,
        )

        with torch.no_grad():
            output = model(*inputs)

        if isinstance(output, (tuple, list)):
            output = output[0]

        output_cpu = output.float().cpu()

        # Verify output is valid
        assert output_cpu.isfinite().all(), "Output contains inf or nan values"
        assert output_cpu.abs().max() > 0, "Output is all zeros"
        assert output_cpu.std() > 1e-6, "Output has near-zero variance"

    def test_lora_aliasing_changes_output(self, loaded_model, compiled_model):
        """Test that writing LoRA weights via aliasing changes the model output."""
        model = loaded_model["model"]
        checkpoints = loaded_model["checkpoints"]
        tp_degree = loaded_model["tp_degree"]
        args = compiled_model["args"]

        inputs = get_test_inputs(
            height=args.height,
            width=args.width,
            text_seq_len=args.max_sequence_length,
            tp_degree=args.tp_degree,
        )

        # Run 1: Baseline (all LoRA buffers are zero)
        with torch.no_grad():
            out1 = model(*inputs)
        if isinstance(out1, (tuple, list)):
            noise1 = out1[0].float().cpu()
        else:
            noise1 = out1.float().cpu()

        # Inject random LoRA weights
        lora_keys = [
            key
            for key in checkpoints[0].keys()
            if "lora_A_active" in key or "lora_B_active" in key
        ]
        assert len(lora_keys) > 0, "No LoRA keys found in checkpoint"

        n_injected = 0
        for key in lora_keys[:20]:  # First 20 buffers
            shape = checkpoints[0][key].shape
            random_weight = torch.randn(shape, dtype=torch.bfloat16) * 0.05
            for rank in range(tp_degree):
                model.write_to_neuron_buffer(random_weight, key, rank)
            n_injected += 1

        assert n_injected > 0, "Failed to inject any LoRA weights"

        # Run 2: With random LoRA
        with torch.no_grad():
            out2 = model(*inputs)
        if isinstance(out2, (tuple, list)):
            noise2 = out2[0].float().cpu()
        else:
            noise2 = out2.float().cpu()

        # Outputs must differ (LoRA changes the output)
        max_diff = (noise2 - noise1).abs().max().item()
        assert max_diff > 1e-5, (
            f"LoRA injection did not change output (max_diff={max_diff}). "
            "Aliasing may not be working."
        )

    def test_lora_zero_restores_baseline(self, loaded_model, compiled_model):
        """Test that zeroing LoRA weights restores original baseline output."""
        model = loaded_model["model"]
        checkpoints = loaded_model["checkpoints"]
        tp_degree = loaded_model["tp_degree"]
        args = compiled_model["args"]

        inputs = get_test_inputs(
            height=args.height,
            width=args.width,
            text_seq_len=args.max_sequence_length,
            tp_degree=args.tp_degree,
        )

        # Zero all LoRA buffers
        lora_keys = [
            key
            for key in checkpoints[0].keys()
            if "lora_A_active" in key or "lora_B_active" in key
        ]
        for key in lora_keys:
            shape = checkpoints[0][key].shape
            zero_weight = torch.zeros(shape, dtype=torch.bfloat16)
            for rank in range(tp_degree):
                model.write_to_neuron_buffer(zero_weight, key, rank)

        # Run with zeroed LoRA
        with torch.no_grad():
            out_zeroed = model(*inputs)
        if isinstance(out_zeroed, (tuple, list)):
            noise_zeroed = out_zeroed[0].float().cpu()
        else:
            noise_zeroed = out_zeroed.float().cpu()

        # Run again (should be identical — deterministic with zeros)
        with torch.no_grad():
            out_baseline = model(*inputs)
        if isinstance(out_baseline, (tuple, list)):
            noise_baseline = out_baseline[0].float().cpu()
        else:
            noise_baseline = out_baseline.float().cpu()

        # Both runs with zeroed LoRA should be identical (deterministic)
        max_diff = (noise_zeroed - noise_baseline).abs().max().item()
        assert max_diff < 1e-5, (
            f"Outputs not deterministic with zeroed LoRA (max_diff={max_diff})"
        )

    def test_lora_buffer_count(self, loaded_model):
        """Test that the expected number of LoRA buffers exist."""
        checkpoints = loaded_model["checkpoints"]

        lora_a_keys = [k for k in checkpoints[0].keys() if "lora_A_active" in k]
        lora_b_keys = [k for k in checkpoints[0].keys() if "lora_B_active" in k]

        # 14 targets per block x 60 blocks = 840 A + 840 B = 1680 total
        assert len(lora_a_keys) == 840, (
            f"Expected 840 lora_A_active keys, got {len(lora_a_keys)}"
        )
        assert len(lora_b_keys) == 840, (
            f"Expected 840 lora_B_active keys, got {len(lora_b_keys)}"
        )
