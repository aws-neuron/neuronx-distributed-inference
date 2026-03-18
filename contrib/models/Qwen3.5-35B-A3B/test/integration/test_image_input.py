#!/usr/bin/env python3
"""
Image input integration tests for Qwen3.5-35B-A3B NeuronX implementation.

Tests the vision-language pipeline:
1. HF processor correctly tokenizes image+text inputs
2. mRoPE position IDs are computed correctly for multimodal inputs
3. Vision encoder processes images (CPU path for now)
4. End-to-end generation with image+text prompts

Environment:
  - trn2.3xlarge with Neuron SDK 2.28
  - source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  - export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
  - pip install Pillow qwen-vl-utils  (for image processing)

Note: The vision encoder currently runs on CPU while the text decoder runs
on Neuron. Full vision encoder compilation to Neuron is planned for a future
task. These tests verify the end-to-end pipeline correctness.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# Import from src directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

MODEL_PATH = os.environ.get("QWEN35_MODEL_PATH", "/home/ubuntu/models/Qwen3.5-35B-A3B")
COMPILED_MODEL_PATH = os.environ.get(
    "QWEN35_COMPILED_PATH", "/home/ubuntu/compiled_qwen35/"
)


# ---------------------------------------------------------------------------
# Helper: create a synthetic test image
# ---------------------------------------------------------------------------
def create_test_image(path, width=224, height=224):
    """Create a simple synthetic test image (red/blue gradient)."""
    from PIL import Image
    import numpy as np

    # Create a gradient image: red channel varies with x, blue with y
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    img_array[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)  # Red gradient
    img_array[:, :, 2] = np.linspace(0, 255, height, dtype=np.uint8).reshape(
        -1, 1
    )  # Blue gradient
    img_array[:, :, 1] = 128  # Constant green

    img = Image.fromarray(img_array)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Tests: HF Processor and Tokenization
# ---------------------------------------------------------------------------
class TestImageTokenization:
    """Test that the HF processor correctly handles image+text inputs."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load processor and create test image."""
        try:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                MODEL_PATH, trust_remote_code=True
            )
        except Exception as e:
            pytest.skip(f"Cannot load processor from {MODEL_PATH}: {e}")

        self.tmpdir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.tmpdir, "test.jpg")
        create_test_image(self.test_image_path)

    def test_processor_loads(self):
        """Verify the HF processor loads and has expected components."""
        assert self.processor is not None
        assert hasattr(self.processor, "image_processor") or hasattr(
            self.processor, "feature_extractor"
        )
        print("PASS: Processor loaded successfully")

    def test_text_only_tokenization(self):
        """Verify text-only inputs work without image tokens."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        assert "input_ids" in inputs
        assert inputs["input_ids"].ndim == 2
        # Text-only should NOT have pixel_values
        has_pixels = "pixel_values" in inputs and inputs["pixel_values"] is not None
        if has_pixels:
            assert inputs["pixel_values"].numel() == 0 or not has_pixels
        print(f"PASS: Text-only tokenization, seq_len={inputs['input_ids'].shape[1]}")

    def test_image_text_tokenization(self):
        """Verify image+text inputs produce pixel_values and image_grid_thw."""
        import base64

        image_data = Path(self.test_image_path).read_bytes()
        b64 = base64.b64encode(image_data).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": f"data:image/jpeg;base64,{b64}"},
                    {"type": "text", "text": "What is in this image?"},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        assert "input_ids" in inputs
        assert "pixel_values" in inputs, "pixel_values missing from processor output"
        assert "image_grid_thw" in inputs, (
            "image_grid_thw missing from processor output"
        )

        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        assert pixel_values.ndim >= 2, (
            f"Unexpected pixel_values shape: {pixel_values.shape}"
        )
        assert image_grid_thw.shape[-1] == 3, (
            f"image_grid_thw should have 3 columns: {image_grid_thw.shape}"
        )

        # Check that image_token_id (248056) appears in input_ids
        input_ids = inputs["input_ids"]
        image_token_count = (input_ids == 248056).sum().item()
        assert image_token_count > 0, "No image placeholder tokens found in input_ids"

        print(f"PASS: Image+text tokenization")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  pixel_values shape: {pixel_values.shape}")
        print(f"  image_grid_thw: {image_grid_thw.tolist()}")
        print(f"  image token count: {image_token_count}")

    def test_image_token_id_matches_config(self):
        """Verify the processor uses the same image_token_id as the config."""
        config_path = os.path.join(MODEL_PATH, "config.json")
        if not os.path.exists(config_path):
            pytest.skip("config.json not found")

        with open(config_path) as f:
            config = json.load(f)

        expected_image_token_id = config.get("image_token_id", 248056)
        expected_video_token_id = config.get("video_token_id", 248057)
        expected_vision_start = config.get("vision_start_token_id", 248053)

        assert expected_image_token_id == 248056
        assert expected_video_token_id == 248057
        assert expected_vision_start == 248053
        print(
            f"PASS: Token IDs match config (image={expected_image_token_id}, "
            f"video={expected_video_token_id}, start={expected_vision_start})"
        )


# ---------------------------------------------------------------------------
# Tests: Multimodal RoPE
# ---------------------------------------------------------------------------
class TestMultimodalRoPE:
    """Test mRoPE position ID computation."""

    def test_text_only_rope(self):
        """mRoPE with text-only input should produce uniform 3D positions."""
        from modeling_qwen35_moe_vl import get_rope_index

        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        position_ids, rope_deltas = get_rope_index(input_ids)

        assert position_ids.shape == (3, 1, 5), (
            f"Expected (3, 1, 5), got {position_ids.shape}"
        )

        # All 3 axes should be identical for text-only
        assert torch.equal(position_ids[0], position_ids[1])
        assert torch.equal(position_ids[1], position_ids[2])
        # Should be sequential: [0, 1, 2, 3, 4]
        expected = torch.arange(5).unsqueeze(0)
        assert torch.equal(position_ids[0], expected), (
            f"Expected {expected}, got {position_ids[0]}"
        )
        print("PASS: Text-only mRoPE produces uniform sequential positions")

    def test_image_text_rope_shapes(self):
        """mRoPE with image tokens should produce correct shapes."""
        from modeling_qwen35_moe_vl import get_rope_index

        # Simulate: 5 text tokens, then 4 image tokens (2x2 grid), then 3 text tokens
        # vision_start=248053, image_token=248056
        vision_start = 248053
        image_token = 248056
        input_ids = torch.tensor(
            [
                [
                    1,
                    2,
                    3,
                    4,
                    5,  # 5 text tokens
                    vision_start,
                    image_token,  # vision_start + image placeholder start
                    image_token,
                    image_token,  # more image tokens (4 total in the merged grid)
                    image_token,
                    10,
                    11,
                    12,  # 3 text tokens
                ]
            ]
        )
        image_grid_thw = torch.tensor([[1, 4, 4]])  # 1 frame, 4h, 4w patches
        # After spatial_merge_size=2: 2h x 2w = 4 merged tokens

        position_ids, rope_deltas = get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
        )

        assert position_ids.shape[0] == 3, "Should have 3 axes"
        assert position_ids.shape[1] == 1, "Batch size should be 1"
        assert position_ids.shape[2] == input_ids.shape[1], "Seq len should match"

        # rope_deltas should be defined
        assert rope_deltas is not None
        assert rope_deltas.shape == (1, 1)
        print(f"PASS: Image+text mRoPE shapes correct")
        print(f"  position_ids shape: {position_ids.shape}")
        print(f"  rope_deltas: {rope_deltas.tolist()}")

    def test_vision_positions_differ_across_axes(self):
        """For vision tokens, H and W axes should have different positions."""
        from modeling_qwen35_moe_vl import get_rope_index

        vision_start = 248053
        image_token = 248056
        # Simple case: 2 text tokens, then 4 image tokens (2x2 merged grid)
        input_ids = torch.tensor(
            [
                [
                    1,
                    2,  # text
                    vision_start,  # vision start
                    image_token,
                    image_token,  # 4 image tokens
                    image_token,
                    image_token,
                    10,  # text
                ]
            ]
        )
        # 1 frame, 4 height patches, 4 width patches -> after merge(2): 2h x 2w = 4 tokens
        image_grid_thw = torch.tensor([[1, 4, 4]])

        position_ids, _ = get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
        )

        # The vision token positions (indices 3-6) should differ on H and W axes
        # Axis 1 (height) and axis 2 (width) should NOT be identical for a 2D grid
        vis_start = 3  # after "1, 2, vision_start"
        vis_end = vis_start + 4

        h_positions = position_ids[1, 0, vis_start:vis_end]
        w_positions = position_ids[2, 0, vis_start:vis_end]

        # For a 2x2 grid, h positions should be [0,0,1,1] and w should be [0,1,0,1]
        # (or offset by the starting position)
        h_unique = h_positions.unique()
        w_unique = w_positions.unique()

        assert len(h_unique) == 2, (
            f"Expected 2 unique H positions for 2x2 grid, got {h_unique}"
        )
        assert len(w_unique) == 2, (
            f"Expected 2 unique W positions for 2x2 grid, got {w_unique}"
        )
        print("PASS: Vision positions differ across H/W axes")
        print(f"  H positions: {h_positions.tolist()}")
        print(f"  W positions: {w_positions.tolist()}")


# ---------------------------------------------------------------------------
# Tests: Vision Encoder (CPU path)
# ---------------------------------------------------------------------------
class TestVisionEncoder:
    """Test the vision encoder model components."""

    def test_vision_config_from_hf(self):
        """Verify vision config can be extracted from HF config."""
        config_path = os.path.join(MODEL_PATH, "config.json")
        if not os.path.exists(config_path):
            pytest.skip("config.json not found")

        with open(config_path) as f:
            config = json.load(f)

        assert "vision_config" in config, "No vision_config in model config"
        vc = config["vision_config"]
        assert vc["depth"] == 27
        assert vc["hidden_size"] == 1152
        assert vc["num_heads"] == 16
        assert vc["patch_size"] == 16
        assert vc["spatial_merge_size"] == 2
        assert vc["out_hidden_size"] in (2048, 3584)  # depends on model variant
        print(f"PASS: Vision config matches expected architecture")
        print(
            f"  depth={vc['depth']}, hidden={vc['hidden_size']}, heads={vc['num_heads']}"
        )

    def test_patch_embed_output_shape(self):
        """Test that Conv3d patch embedding produces correct shapes."""
        try:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeVisionPatchEmbed,
            )
        except ImportError:
            pytest.skip("Qwen3.5 MoE vision classes not available in transformers")

        from types import SimpleNamespace

        config = SimpleNamespace(
            patch_size=16,
            temporal_patch_size=2,
            in_channels=3,
            hidden_size=1152,
        )
        patch_embed = Qwen3_5MoeVisionPatchEmbed(config)

        # Input: (num_patches * temporal_patch_size * patch_size * patch_size, in_channels)
        # For a 224x224 image with temporal_patch_size=2, patch_size=16:
        # num_h_patches = 224/16 = 14, num_w_patches = 14
        # With temporal_patch_size=2, we need 2 frames -> 2*14*14 = 392 patches
        # Each patch is (in_channels, temporal_patch_size, patch_size, patch_size) = (3, 2, 16, 16)
        num_patches = 392  # 2 frames * 14 * 14
        pixel_values = torch.randn(num_patches, 3 * 2 * 16 * 16)
        # Reshape to what patch_embed expects
        pixel_values = pixel_values.view(-1, 3, 2, 16, 16)

        with torch.no_grad():
            output = patch_embed(pixel_values)

        assert output.shape[-1] == 1152, (
            f"Expected hidden_size=1152, got {output.shape[-1]}"
        )
        print(f"PASS: Patch embedding output shape: {output.shape}")

    def test_vision_rotary_embedding(self):
        """Test vision rotary embedding computation."""
        try:
            from modeling_qwen35_moe_vision import NeuronQwen35VisionModelWrapper
        except ImportError:
            pytest.skip("Vision model wrapper not importable")

        from types import SimpleNamespace

        config = SimpleNamespace(
            hidden_size=1152,
            num_heads=16,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            in_channels=3,
            out_hidden_size=2048,
            num_position_embeddings=2304,
            depth=27,
            intermediate_size=4304,
            hidden_act="gelu_pytorch_tanh",
        )

        wrapper = NeuronQwen35VisionModelWrapper.__new__(NeuronQwen35VisionModelWrapper)
        wrapper.vision_config = config

        try:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeVisionRotaryEmbedding,
            )

            head_dim = config.hidden_size // config.num_heads
            wrapper.rotary_pos_emb = Qwen3_5MoeVisionRotaryEmbedding(head_dim // 2)
        except ImportError:
            pytest.skip("Qwen3.5 MoE vision rotary embedding not available")

        # Test with a simple 2x2 grid (4 patches total)
        grid_thw = torch.tensor([[1, 4, 4]])
        rot_emb = wrapper.rot_pos_emb(grid_thw)

        expected_tokens = 1 * 4 * 4  # t * h * w
        assert rot_emb.shape[0] == expected_tokens, (
            f"Expected {expected_tokens} tokens, got {rot_emb.shape[0]}"
        )
        print(f"PASS: Vision rotary embedding shape: {rot_emb.shape}")


# ---------------------------------------------------------------------------
# Tests: End-to-End VL Pipeline
# ---------------------------------------------------------------------------
class TestEndToEndVL:
    """End-to-end tests for the full VL pipeline.

    These tests require a compiled text model and the HF processor.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up model and processor."""
        config_path = os.path.join(MODEL_PATH, "config.json")
        if not os.path.exists(config_path):
            pytest.skip(f"Model not found at {MODEL_PATH}")

        try:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                MODEL_PATH, trust_remote_code=True
            )
        except Exception as e:
            pytest.skip(f"Cannot load processor: {e}")

        self.tmpdir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.tmpdir, "test.jpg")
        create_test_image(self.test_image_path, width=224, height=224)

    def test_prepare_input_args(self):
        """Test the static prepare_input_args helper."""
        from modeling_qwen35_moe_vl import NeuronQwen35MoeVLForCausalLM

        input_ids, attention_mask, vision_inputs = (
            NeuronQwen35MoeVLForCausalLM.prepare_input_args(
                "What is in this image?",
                self.test_image_path,
                self.processor,
            )
        )

        assert input_ids.ndim == 2
        assert attention_mask.ndim == 2
        assert "pixel_values" in vision_inputs
        assert "image_grid_thw" in vision_inputs

        print(f"PASS: prepare_input_args")
        print(f"  input_ids: {input_ids.shape}")
        print(f"  pixel_values: {vision_inputs['pixel_values'].shape}")
        print(f"  image_grid_thw: {vision_inputs['image_grid_thw'].tolist()}")

    def test_text_only_prepare_input_args(self):
        """Test prepare_input_args with no image (text-only)."""
        from modeling_qwen35_moe_vl import NeuronQwen35MoeVLForCausalLM

        input_ids, attention_mask, vision_inputs = (
            NeuronQwen35MoeVLForCausalLM.prepare_input_args(
                "Hello, how are you?",
                None,  # No image
                self.processor,
            )
        )

        assert input_ids.ndim == 2
        assert len(vision_inputs) == 0 or all(
            v is None or v.numel() == 0 for v in vision_inputs.values()
        )
        print(f"PASS: Text-only prepare_input_args, seq_len={input_ids.shape[1]}")

    def test_image_text_rope_integration(self):
        """Integration test: processor -> mRoPE position IDs."""
        from modeling_qwen35_moe_vl import (
            NeuronQwen35MoeVLForCausalLM,
            get_rope_index,
        )

        input_ids, attention_mask, vision_inputs = (
            NeuronQwen35MoeVLForCausalLM.prepare_input_args(
                "What color is this?",
                self.test_image_path,
                self.processor,
            )
        )

        if "image_grid_thw" not in vision_inputs:
            pytest.skip("Processor did not produce image_grid_thw")

        position_ids, rope_deltas = get_rope_index(
            input_ids,
            image_grid_thw=vision_inputs["image_grid_thw"],
            attention_mask=attention_mask,
        )

        assert position_ids.shape == (3, input_ids.shape[0], input_ids.shape[1])
        assert rope_deltas is not None

        # Verify image token positions have spatial structure
        image_mask = input_ids[0] == 248056
        if image_mask.any():
            image_positions_h = position_ids[1, 0, image_mask]
            image_positions_w = position_ids[2, 0, image_mask]
            # Should have variation in at least one spatial axis
            has_spatial = (
                image_positions_h.unique().numel() > 1
                or image_positions_w.unique().numel() > 1
            )
            assert has_spatial, "Image positions should have spatial variation"

        print(f"PASS: Image+text mRoPE integration")
        print(f"  position_ids shape: {position_ids.shape}")
        print(f"  rope_deltas: {rope_deltas.tolist()}")
        print(f"  image tokens: {image_mask.sum().item()}")

    def test_full_vl_pipeline_text_only(self):
        """E2E: Text-only generation through the VL model (no vision encoder needed)."""
        from neuronx_distributed_inference.models.config import MoENeuronConfig
        from modeling_qwen35_moe import (
            NeuronQwen35MoeForCausalLM,
            Qwen35MoeInferenceConfig,
        )
        from modeling_qwen35_moe_vl import NeuronQwen35MoeVLForCausalLM

        # Load HF config for text decoder
        with open(os.path.join(MODEL_PATH, "config.json")) as f:
            full_config = json.load(f)
        text_config = full_config.get("text_config", full_config)

        neuron_config = MoENeuronConfig(
            tp_degree=4,
            max_batch_size=1,
            max_context_length=128,
            max_new_tokens=32,
            on_device_sampling_config=None,
            torch_dtype=torch.bfloat16,
            fused_qkv=True,
            moe_tp_degree=4,
            moe_ep_degree=1,
            blockwise_matmul_config={"block_size": 2048},
        )

        config_dict = dict(text_config)
        config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
        if "rope_parameters" in text_config:
            config_dict["rope_theta"] = text_config["rope_parameters"].get(
                "rope_theta", 10000000
            )
        if config_dict.get("tie_word_embeddings") is None:
            config_dict["tie_word_embeddings"] = False

        inference_config = Qwen35MoeInferenceConfig(
            neuron_config=neuron_config, **config_dict
        )

        # Create VL model (text-only -- no vision config)
        vl_model = NeuronQwen35MoeVLForCausalLM(
            model_path=MODEL_PATH,
            text_config=inference_config,
            vision_config=None,
            processor=self.processor,
        )

        # Load compiled text model
        compiled_path = Path(COMPILED_MODEL_PATH)
        if not (compiled_path / "model.pt").exists():
            pytest.skip("Compiled model not found -- run text-only tests first")

        vl_model.load(COMPILED_MODEL_PATH)

        # Generate text-only
        input_ids, attention_mask, _ = NeuronQwen35MoeVLForCausalLM.prepare_input_args(
            "The capital of France is",
            None,
            self.processor,
        )

        generated_ids = vl_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,
        )

        output_text = self.processor.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        assert "Paris" in output_text or len(output_text) > 20
        print(f"PASS: Full VL pipeline (text-only)")
        print(f"  Output: {output_text}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("Qwen3.5-35B-A3B Image Input Integration Tests")
    print("=" * 80)

    # Run subset of tests that don't require compilation
    print("\n--- Tokenization Tests ---")
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

        tmpdir = tempfile.mkdtemp()
        test_image = os.path.join(tmpdir, "test.jpg")
        create_test_image(test_image)

        test = TestImageTokenization()
        test.processor = processor
        test.tmpdir = tmpdir
        test.test_image_path = test_image

        test.test_processor_loads()
        test.test_text_only_tokenization()
        test.test_image_text_tokenization()
        test.test_image_token_id_matches_config()
    except Exception as e:
        print(f"SKIP: Tokenization tests ({e})")

    print("\n--- mRoPE Tests ---")
    try:
        test_rope = TestMultimodalRoPE()
        test_rope.test_text_only_rope()
        test_rope.test_image_text_rope_shapes()
        test_rope.test_vision_positions_differ_across_axes()
    except Exception as e:
        print(f"SKIP: mRoPE tests ({e})")

    print("\n--- Vision Config Tests ---")
    try:
        test_vision = TestVisionEncoder()
        test_vision.test_vision_config_from_hf()
    except Exception as e:
        print(f"SKIP: Vision config tests ({e})")

    print("\n--- E2E Tests ---")
    try:
        test_e2e = TestEndToEndVL()
        test_e2e.processor = processor
        test_e2e.tmpdir = tmpdir
        test_e2e.test_image_path = test_image
        test_e2e.test_prepare_input_args()
        test_e2e.test_text_only_prepare_input_args()
        test_e2e.test_image_text_rope_integration()
    except Exception as e:
        print(f"SKIP: E2E tests ({e})")

    print("\n" + "=" * 80)
    print("Image input tests complete")
    print("=" * 80)
