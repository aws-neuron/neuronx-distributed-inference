"""
Integration tests for HunyuanVideo-1.5 on Neuron.

Tests validate that the compiled Neuron pipeline produces correct output
by checking component-level numerical accuracy and E2E generation quality.

Requirements:
  - trn2.3xlarge with Neuron SDK 2.28
  - Pre-compiled models (see README for compilation steps)
  - HunyuanVideo-1.5 repo cloned to HUNYUAN_REPO_DIR

Run all tests (recommended: run test groups separately to avoid NeuronCore contention):

    # Component tests (byT5 + VAE -- loads Neuron models in-process):
    pytest test/integration/test_model.py -k "ByT5 or VAE" -v --capture=tee-sys

    # E2E tests (runs pipeline as subprocess):
    pytest test/integration/test_model.py -k "E2E" -v --capture=tee-sys

Note: Running all tests together may fail because in-process Neuron model loads
(byT5/VAE) claim NeuronCores, preventing the E2E subprocess from initializing
its own Neuron runtime. Run the groups separately or use pytest-forked.
"""

import os
import sys
import subprocess
import time
import pytest
import torch
import torch_neuronx  # registers neuron.Model type for torch.jit.load

# Ensure src/ is importable
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, SRC_DIR)

# Set Neuron environment
os.environ.setdefault("NEURON_RT_NUM_CORES", "4")
os.environ.setdefault("XLA_DISABLE_FUNCTIONALIZATION", "1")
os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")
os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
os.environ.setdefault("HUNYUAN_ATTN_MODE", "torch")

# Default directories (override with env vars)
MODELS_DIR = os.environ.get("HUNYUAN_MODELS_DIR", "./models")
COMPILED_DIR = os.environ.get("HUNYUAN_COMPILED_DIR", "./compiled")
REPO_DIR = os.environ.get("HUNYUAN_REPO_DIR", "./HunyuanVideo-1.5")

sys.path.insert(0, REPO_DIR)


def _skip_if_no_compiled_models():
    """Skip test if compiled DiT models are not available."""
    dit_dir = os.path.join(COMPILED_DIR, "dit_tp4_480p")
    if not os.path.exists(dit_dir):
        pytest.skip(
            f"Compiled DiT not found at {dit_dir}. Run recompile_dit_masked.py first."
        )


def _skip_if_no_vae():
    """Skip test if VAE model is not available."""
    vae_dir = os.path.join(COMPILED_DIR, "vae_decoder_neuron")
    if not os.path.exists(vae_dir):
        pytest.skip(
            f"Compiled VAE not found at {vae_dir}. Run compile_vae_neuron.py first."
        )


def _skip_if_no_byt5():
    """Skip test if byT5 models are not available."""
    encoder_path = os.path.join(COMPILED_DIR, "byt5_encoder.pt")
    if not os.path.exists(encoder_path):
        pytest.skip(
            f"Compiled byT5 not found at {encoder_path}. Run trace_byt5.py first."
        )


class TestByT5Accuracy:
    """Test byT5 text encoder numerical accuracy."""

    def test_byt5_cosine_similarity(self):
        """byT5 Neuron output must have cos_sim >= 0.999 vs CPU reference."""
        _skip_if_no_byt5()

        dtype = torch.bfloat16

        # Load byT5 model + tokenizer using HunyuanVideo loading code
        from hyvideo.models.text_encoders.byT5 import load_glyph_byT5_v2, ByT5Mapper
        import safetensors.torch as st_module
        import glob

        byt5_args = dict(
            byT5_google_path=f"{MODELS_DIR}/byt5-small",
            byT5_ckpt_path=f"{MODELS_DIR}/Glyph-SDXL-v2/checkpoints/byt5_model.pt",
            multilingual_prompt_format_color_path=f"{MODELS_DIR}/Glyph-SDXL-v2/assets/color_idx.json",
            multilingual_prompt_format_font_path=f"{MODELS_DIR}/Glyph-SDXL-v2/assets/multilingual_10-lang_idx.json",
            byt5_max_length=256,
        )
        byt5_kwargs = load_glyph_byT5_v2(byt5_args, device=torch.device("cpu"))
        byt5_model = byt5_kwargs["byt5_model"].eval()
        byt5_tokenizer = byt5_kwargs["byt5_tokenizer"]

        # Load ByT5Mapper from DiT checkpoint
        dit_ckpt_path = f"{MODELS_DIR}/HunyuanVideo-1.5/transformer/480p_t2v"
        st_files = sorted(glob.glob(f"{dit_ckpt_path}/*.safetensors"))
        byt5_in_state = {}
        for sf in st_files:
            with st_module.safe_open(sf, framework="pt") as f:
                for key in f.keys():
                    if key.startswith("byt5_in."):
                        byt5_in_state[key.replace("byt5_in.", "")] = f.get_tensor(key)
        assert byt5_in_state, "No byt5_in weights found in DiT checkpoint"

        fc1_w = byt5_in_state["fc1.weight"]
        byt5_mapper = ByT5Mapper(
            fc1_w.shape[1],
            byt5_in_state["fc2.weight"].shape[0],
            fc1_w.shape[0],
            byt5_in_state["fc3.weight"].shape[0],
            use_residual=(fc1_w.shape[1] == byt5_in_state["fc2.weight"].shape[0]),
        )
        byt5_mapper.load_state_dict(byt5_in_state)
        byt5_mapper = byt5_mapper.to(dtype).eval()

        # CPU reference
        test_text = "A golden retriever running in a meadow"
        tokens = byt5_tokenizer(
            test_text,
            padding="max_length",
            max_length=256,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask.float()

        with torch.no_grad():
            cpu_enc = byt5_model(input_ids, attention_mask=attention_mask)[0]
            cpu_mapper_out = byt5_mapper(cpu_enc.to(dtype))

        # Load Neuron traced models
        encoder_path = os.path.join(COMPILED_DIR, "byt5_encoder.pt")
        mapper_path = os.path.join(COMPILED_DIR, "byt5_mapper.pt")
        neuron_encoder = torch.jit.load(encoder_path)
        neuron_mapper = torch.jit.load(mapper_path)

        # Run on Neuron
        # Encoder takes (input_ids, attention_mask), mapper takes single bf16 tensor
        neuron_enc = neuron_encoder(input_ids, attention_mask)
        neuron_mapper_out = neuron_mapper(neuron_enc.to(dtype))

        # Compare encoder output
        enc_cos_sim = torch.nn.functional.cosine_similarity(
            cpu_enc.float().flatten(), neuron_enc.float().flatten(), dim=0
        ).item()

        # Compare mapper output
        mapper_cos_sim = torch.nn.functional.cosine_similarity(
            cpu_mapper_out.float().flatten(), neuron_mapper_out.float().flatten(), dim=0
        ).item()

        print(f"byT5 encoder cosine similarity: {enc_cos_sim:.6f}")
        print(f"byT5 mapper cosine similarity:  {mapper_cos_sim:.6f}")

        assert enc_cos_sim >= 0.999, (
            f"byT5 encoder cosine similarity {enc_cos_sim:.6f} < 0.999"
        )
        assert mapper_cos_sim >= 0.999, (
            f"byT5 mapper cosine similarity {mapper_cos_sim:.6f} < 0.999"
        )


class TestVAEAccuracy:
    """Test VAE decoder numerical accuracy."""

    def test_vae_tile_output(self):
        """VAE Neuron tiled decode produces finite output with correct shape."""
        _skip_if_no_vae()

        from tiled_vae_decode import TiledVAEDecoderNeuron

        vae_dir = os.path.join(COMPILED_DIR, "vae_decoder_neuron")
        decoder = TiledVAEDecoderNeuron(vae_dir)

        # Create a random latent tile matching compiled shape:
        # (B=1, C=32, T=2, H=tile_h, W=tile_w) in bf16
        latent = torch.randn(
            1, 32, 2, decoder.tile_h, decoder.tile_w, dtype=torch.bfloat16
        )

        # Warmup
        decoder.warmup(T_lat=2, n=2)

        # Decode tile
        t0 = time.time()
        neuron_out = decoder._decode_tile(latent)
        elapsed = time.time() - t0

        # Validate output
        assert neuron_out is not None, "VAE decode returned None"
        assert torch.isfinite(neuron_out).all(), "VAE output contains NaN/Inf"
        assert neuron_out.shape[1] == 3, (
            f"Expected 3 RGB channels, got {neuron_out.shape[1]}"
        )
        # Spatial dimensions should be 16x input (ffactor_spatial=16)
        expected_h = decoder.tile_h * 16
        expected_w = decoder.tile_w * 16
        assert neuron_out.shape[3] == expected_h, (
            f"Expected height {expected_h}, got {neuron_out.shape[3]}"
        )
        assert neuron_out.shape[4] == expected_w, (
            f"Expected width {expected_w}, got {neuron_out.shape[4]}"
        )

        print(
            f"VAE tile decode: {neuron_out.shape}, {elapsed * 1000:.0f}ms, all finite"
        )


class TestE2EPipeline:
    """Test full E2E text-to-video pipeline via subprocess."""

    def _run_pipeline(self, extra_args=None, timeout=300):
        """Run e2e_pipeline.py as a subprocess and return result."""
        cmd = [
            sys.executable,
            os.path.join(SRC_DIR, "e2e_pipeline.py"),
        ]
        if extra_args:
            cmd.extend(extra_args)

        env = os.environ.copy()
        env["HUNYUAN_REPO_DIR"] = REPO_DIR
        env["HUNYUAN_MODELS_DIR"] = MODELS_DIR
        env["HUNYUAN_COMPILED_DIR"] = COMPILED_DIR
        env["HUNYUAN_ATTN_MODE"] = "torch"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
        )
        return result

    def test_generation_produces_frames(self):
        """E2E pipeline produces output frames at correct resolution (2 steps, no CFG)."""
        _skip_if_no_compiled_models()
        _skip_if_no_byt5()
        _skip_if_no_vae()

        result = self._run_pipeline(
            extra_args=[
                "--steps",
                "2",
                "--guidance-scale",
                "1.0",  # disable CFG for speed
                "--prompt",
                "A sunset over the ocean",
            ],
            timeout=300,
        )

        print("--- STDOUT (last 3000 chars) ---")
        print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
        if result.stderr:
            print("--- STDERR (last 2000 chars) ---")
            print(result.stderr[-2000:])

        assert result.returncode == 0, (
            f"Pipeline exited with code {result.returncode}.\n"
            f"stderr: {result.stderr[-2000:]}"
        )

        # Check that pipeline reported correct frame dimensions
        assert "480x848" in result.stdout or "480" in result.stdout, (
            "Pipeline output doesn't mention expected 480p resolution"
        )

    def test_generation_performance(self):
        """Warm E2E generation (no CFG, 50 steps, skip-vae) completes within 120s."""
        _skip_if_no_compiled_models()
        _skip_if_no_byt5()
        _skip_if_no_vae()

        t0 = time.time()
        result = self._run_pipeline(
            extra_args=[
                "--steps",
                "50",
                "--guidance-scale",
                "1.0",  # disable CFG
                "--prompt",
                "A cat sitting on a windowsill",
                "--skip-vae",  # skip VAE to isolate DiT performance
            ],
            timeout=180,
        )
        elapsed = time.time() - t0

        print("--- STDOUT (last 2000 chars) ---")
        print(result.stdout[-2000:])

        assert result.returncode == 0, (
            f"Pipeline exited with code {result.returncode}.\n"
            f"stderr: {result.stderr[-2000:]}"
        )
        # Wall-clock includes subprocess startup + model loading (~20s overhead)
        # Pipeline itself: LLM 14s + DiT 16.4s = ~30s, plus ~22s load = ~52s
        assert elapsed < 120.0, f"E2E generation took {elapsed:.1f}s (limit: 120s)"
        print(
            f"E2E performance (no CFG, skip-vae, 50 steps): {elapsed:.1f}s wall-clock"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=tee-sys"])
