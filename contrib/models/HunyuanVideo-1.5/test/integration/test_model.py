#!/usr/bin/env python3
"""
Integration tests for HunyuanVideo-1.5 NeuronX implementation.

This is a multi-component video diffusion pipeline, not a single text decoder.
Tests validate that each component can be loaded and that the end-to-end
pipeline produces valid video frames.
"""

import pytest
import sys
from pathlib import Path

src_dir = str(Path(__file__).parent.parent.parent / "src")
sys.path.insert(0, src_dir)


# Paths — update these for your environment
MODEL_PATH = "/home/ubuntu/models/HunyuanVideo-1.5/"
COMPILED_DIR = "/home/ubuntu/neuron_models/HunyuanVideo-1.5/"

HF_MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


@pytest.fixture(scope="module")
def compiled_dir():
    p = Path(COMPILED_DIR)
    if not p.exists():
        pytest.skip(f"Compiled artifacts not found: {COMPILED_DIR}")
    return p


class TestArtifactsExist:
    """Verify all required compiled artifacts are present."""

    def test_transformer_neff(self, compiled_dir):
        assert (compiled_dir / "compiled_transformer" / "model.pt").exists()

    def test_qwen_neff(self, compiled_dir):
        assert (compiled_dir / "compiled_qwen2vl" / "model.pt").exists()

    def test_byt5_neff(self, compiled_dir):
        assert (compiled_dir / "byt5_traced.pt").exists()

    def test_refiner_neff(self, compiled_dir):
        assert (compiled_dir / "refiner_traced.pt").exists()

    def test_reorder_neff(self, compiled_dir):
        assert (compiled_dir / "reorder_traced.pt").exists()

    def test_cond_type_weights(self, compiled_dir):
        assert (compiled_dir / "cond_type_embed_weight.pt").exists()


class TestModelImports:
    """Verify that all model classes can be imported."""

    def test_import_transformer(self):
        from modeling_hunyuan_video15_transformer import (
            HunyuanVideo15TransformerConfig,
            NeuronHunyuanVideo15Transformer,
        )
        assert HunyuanVideo15TransformerConfig is not None
        assert NeuronHunyuanVideo15Transformer is not None

    def test_import_qwen_encoder(self):
        from modeling_qwen2vl_encoder import (
            Qwen2VLEncoderConfig,
            NeuronQwen2VLEncoder,
        )
        assert Qwen2VLEncoderConfig is not None
        assert NeuronQwen2VLEncoder is not None

    def test_import_vae(self):
        from modeling_hunyuan_video15_vae import NeuronVAEDecoder
        assert NeuronVAEDecoder is not None

    def test_import_text_utils(self):
        from modeling_hunyuan_video15_text import (
            compile_byt5_encoder,
            compile_token_refiner,
        )
        assert compile_byt5_encoder is not None
        assert compile_token_refiner is not None


class TestTransformerConfig:
    """Verify transformer config loads from HuggingFace."""

    def test_config_loads(self):
        from modeling_hunyuan_video15_transformer import HunyuanVideo15TransformerConfig
        config = HunyuanVideo15TransformerConfig.from_pretrained(HF_MODEL_ID)
        assert config is not None

    def test_config_has_neuron_config(self):
        from modeling_hunyuan_video15_transformer import HunyuanVideo15TransformerConfig
        config = HunyuanVideo15TransformerConfig.from_pretrained(HF_MODEL_ID)
        assert hasattr(config, "neuron_config")


class TestEndToEnd:
    """End-to-end pipeline test — generates frames and validates output."""

    @pytest.mark.slow
    def test_generate_frames(self, compiled_dir, tmp_path):
        from run_inference import generate

        output_dir = str(tmp_path / "frames")
        generate(
            prompt="A cat walking on a beach",
            num_steps=2,  # minimal steps for smoke test
            guidance_scale=3.0,
            seed=42,
            output_dir=output_dir,
            compiled_dir=str(compiled_dir),
            vae_on_cpu=True,  # skip VAE NEFF requirement
        )

        frames = list(Path(output_dir).glob("frame_*.png"))
        assert len(frames) > 0, "No frames generated"

    @pytest.mark.slow
    def test_frame_dimensions(self, compiled_dir, tmp_path):
        from PIL import Image
        from run_inference import generate

        output_dir = str(tmp_path / "frames")
        generate(
            prompt="A simple test scene",
            num_steps=2,
            guidance_scale=3.0,
            seed=42,
            output_dir=output_dir,
            compiled_dir=str(compiled_dir),
            vae_on_cpu=True,
        )

        frame = Image.open(next(Path(output_dir).glob("frame_*.png")))
        assert frame.size == (640, 480), f"Expected 640x480, got {frame.size}"


if __name__ == "__main__":
    print("Run with: pytest <this_file> --capture=tee-sys")
    print("  Skip slow tests: pytest <this_file> -m 'not slow'")
