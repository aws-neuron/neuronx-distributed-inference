"""
Minimal unit tests for NeuronQwen2VisionModel and related classes
Tests basic functionality using mocks to ensure they work on CPU without Neuron dependencies.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from types import SimpleNamespace


@pytest.fixture
def mock_vision_neuron_config():
    """Create a mock vision neuron config based on actual usage"""
    return SimpleNamespace(
        batch_size=1,
        seq_len=1012,  # VISION_SEQ_LENGTH for 1 image
        tp_degree=4,
        world_size=4,
        enable_bucketing=True,
        save_sharded_checkpoint=True,
        torch_dtype=torch.bfloat16,
        buckets=[1],
        cc_pipeline_tiling_factor=2,
        fused_qkv=True,
        qkv_kernel_enabled=False,
        attn_kernel_enabled=True,
        mlp_kernel_enabled=True,
        cast_type="as-declared",
        logical_neuron_cores=2,
    )


@pytest.fixture
def mock_text_neuron_config():
    """Create a mock text neuron config based on actual usage"""
    return SimpleNamespace(
        batch_size=1,
        seq_len=512,  # TEXT_SEQ_LENGTH
        ctx_batch_size=1,
        tp_degree=4,
        world_size=4,
        torch_dtype=torch.bfloat16,
        attention_dtype=torch.bfloat16,
        rpl_reduce_dtype=torch.bfloat16,
        cp_degree=1,
        save_sharded_checkpoint=True,
        sequence_parallel_enabled=True,
        fused_qkv=True,
        qkv_kernel_enabled=True,
        mlp_kernel_enabled=False,
        enable_bucketing=True,
        context_encoding_buckets=[512],
        token_generation_buckets=[512],
        buckets=[512],
        attn_block_tkg_nki_kernel_enabled=False,
        attn_block_tkg_nki_kernel_cache_update=False,
        cc_pipeline_tiling_factor=2,
        cast_type="as-declared",
        logical_neuron_cores=2,
    )


@pytest.fixture
def mock_vision_config(mock_vision_neuron_config):
    """Create a mock vision config based on Qwen2-VL-7B"""
    return SimpleNamespace(
        depth=32,
        embed_dim=1280,
        mlp_ratio=4,
        num_heads=16,
        in_channels=3,
        hidden_size=3584,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        hidden_act="quick_gelu",
        head_dim=80,  # embed_dim // num_heads
        num_cores_per_group=1,
        neuron_config=mock_vision_neuron_config,
    )


@pytest.fixture
def mock_text_config(mock_text_neuron_config):
    """Create a mock text config based on Qwen2-VL-7B"""
    return SimpleNamespace(
        neuron_config=mock_text_neuron_config,
    )


@pytest.fixture
def mock_full_config(mock_vision_config, mock_text_config):
    """Create a mock full inference config based on Qwen2-VL-7B"""
    return SimpleNamespace(
        vision_config=mock_vision_config,
        text_config=mock_text_config,
        # Top-level attributes (copied from text_config per QWEN2_VL_TEXT_CONFIG_KEYS)
        hidden_size=3584,
        num_attention_heads=28,
        num_hidden_layers=28,
        num_key_value_heads=4,
        pad_token_id=151643,
        vocab_size=152064,
        intermediate_size=18944,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_scaling=None,
        hidden_act="silu",
        bos_token_id=151643,
        eos_token_id=151645,
        qkv_bias=True,
        o_bias=False,
        vision_token_id=151654,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        num_cores_per_group=1,
    )


class TestQwen2VLVisionBackboneInferenceConfig:
    """Test suite for Qwen2VLInferenceConfig"""
    def test_config_required_attributes(self):
        """Test that config has required attributes"""
        from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLInferenceConfig

        with patch('neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl.ImageToTextInferenceConfig.__init__', return_value=None), \
             patch.object(Qwen2VLInferenceConfig, 'add_special_config'), \
             patch.object(Qwen2VLInferenceConfig, 'validate_model_supported_configs'):

            config = Qwen2VLInferenceConfig(
                text_neuron_config=None,
                vision_neuron_config=None
            )
            required_attrs = config.get_required_attributes()

            expected_attrs = [
                "text_config",
                "vision_config",
                "text_config.hidden_size",
                "text_config.num_attention_heads",
                "text_config.num_hidden_layers",
                "text_config.num_key_value_heads",
                "text_config.pad_token_id",
                "text_config.vocab_size",
                "text_config.max_position_embeddings",
                "text_config.rope_theta",
                "text_config.rms_norm_eps",
                "text_config.hidden_act",
                "vision_config.depth",
                "vision_config.mlp_ratio",
                "vision_config.num_heads",
                "vision_config.in_channels",
                "vision_config.patch_size",
                "vision_config.spatial_merge_size",
                "vision_config.temporal_patch_size",
            ]

            assert set(required_attrs) == set(expected_attrs)


class TestQwen2VLVisionRotaryEmbedding:
    """Test suite for Qwen2VLVisionRotaryEmbedding"""

    def test_forward(self):
        """Test rotary embedding forward pass"""
        from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_vision import Qwen2VLVisionRotaryEmbedding

        rotary_emb = Qwen2VLVisionRotaryEmbedding()

        x = torch.randn(2, 16, 256, 80, dtype=torch.bfloat16)
        cos = torch.randn(2, 256, 80, dtype=torch.float32)
        sin = torch.randn(2, 256, 80, dtype=torch.float32)
        position_embeddings = (cos, sin)

        out_cos, out_sin = rotary_emb(x, position_embeddings)

        assert out_cos.shape == cos.shape
        assert out_sin.shape == sin.shape

    def test_forward_with_float16(self):
        """Test rotary embedding forward pass with float16 input"""
        from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_vision import Qwen2VLVisionRotaryEmbedding

        rotary_emb = Qwen2VLVisionRotaryEmbedding()

        x = torch.randn(2, 16, 256, 80, dtype=torch.float16)
        cos = torch.randn(2, 256, 80, dtype=torch.float32)
        sin = torch.randn(2, 256, 80, dtype=torch.float32)
        position_embeddings = (cos, sin)

        out_cos, out_sin = rotary_emb(x, position_embeddings)

        assert out_cos.dtype == torch.float16
        assert out_sin.dtype == torch.float16


class TestNeuronQwen2VLForImageEncoding:
    """Test suite for NeuronQwen2VLForImageEncoding"""

    def test_get_compiler_args(self, mock_full_config):
        """Test get_compiler_args returns correct string"""
        from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_vision import NeuronQwen2VLForImageEncoding

        with patch('neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_vision.NeuronApplicationBase.__init__'):
            app = NeuronQwen2VLForImageEncoding.__new__(NeuronQwen2VLForImageEncoding)
            app._model_cls = MagicMock(__name__="NeuronQwen2VisionModel")

            compiler_args = app.get_compiler_args()

            assert "--auto-cast=none" in compiler_args
            assert "--model-type=transformer" in compiler_args
            assert "--enable-ccop-compute-overlap" in compiler_args
            assert "--cc-pipeline-tiling-factor=2" in compiler_args
            assert "-O1" in compiler_args
            assert "--verify-hlo=true" in compiler_args

    def test_update_state_dict_for_tied_weights(self):
        """Test update_state_dict_for_tied_weights is a no-op"""
        from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_vision import NeuronQwen2VLForImageEncoding

        state_dict = {"test_key": torch.randn(10, 10)}
        original_keys = set(state_dict.keys())

        NeuronQwen2VLForImageEncoding.update_state_dict_for_tied_weights(state_dict)

        assert set(state_dict.keys()) == original_keys

    def test_convert_hf_to_neuron_state_dict_qkv(self, mock_full_config):
        """Test state dict conversion for qkv layers"""
        from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_vision import NeuronQwen2VLForImageEncoding

        state_dict = {
            'visual.blocks.0.attn.qkv.weight': torch.randn(3840, 1280),
            'visual.blocks.0.attn.qkv.bias': torch.randn(3840),
        }

        converted = NeuronQwen2VLForImageEncoding.convert_hf_to_neuron_state_dict(
            state_dict, mock_full_config
        )

        # Check qkv key transformation
        assert 'blocks.0.attn.qkv_proj.Wqkv.weight' in converted
        assert 'blocks.0.attn.qkv_proj.Wqkv.bias' in converted

        # Check dtype conversion
        assert converted['blocks.0.attn.qkv_proj.Wqkv.weight'].dtype == torch.bfloat16

    def test_convert_hf_to_neuron_state_dict_proj(self, mock_full_config):
        """Test state dict conversion for projection layers"""
        from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_vision import NeuronQwen2VLForImageEncoding

        state_dict = {
            'visual.blocks.0.attn.proj.weight': torch.randn(1280, 1280),
            'visual.blocks.0.attn.proj.bias': torch.randn(1280),
        }

        converted = NeuronQwen2VLForImageEncoding.convert_hf_to_neuron_state_dict(
            state_dict, mock_full_config
        )

        # Check proj key transformation
        assert 'blocks.0.attn.o_proj.weight' in converted
        assert 'blocks.0.attn.o_proj.bias' in converted

    def test_convert_hf_to_neuron_state_dict_removes_visual_prefix(self, mock_full_config):
        """Test that visual. prefix is removed from all keys"""
        from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_vision import NeuronQwen2VLForImageEncoding

        state_dict = {
            'visual.merger.ln_q.weight': torch.randn(1280),
            'visual.merger.mlp.0.weight': torch.randn(5120, 5120),
            'visual.patch_embed.proj.weight': torch.randn(1280, 3, 14, 14),
            'visual.blocks.0.norm1.weight': torch.randn(1280),
        }

        converted = NeuronQwen2VLForImageEncoding.convert_hf_to_neuron_state_dict(
            state_dict, mock_full_config
        )

        # Check visual. prefix is removed
        assert 'merger.ln_q.weight' in converted
        assert 'merger.mlp.0.weight' in converted
        assert 'patch_embed.proj.weight' in converted
        assert 'blocks.0.norm1.weight' in converted

        # Ensure no keys start with visual.
        assert not any(key.startswith('visual.') for key in converted.keys())


class TestNeuronQwen2VLForCausalLM:
    """Test suite for NeuronQwen2VLForCausalLM"""

    def test_get_required_kwargs(self):
        """Test get_required_kwargs returns correct list"""
        from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl import NeuronQwen2VLForCausalLM

        with patch('neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl.NeuronBaseForImageToText.__init__'):
            model = NeuronQwen2VLForCausalLM.__new__(NeuronQwen2VLForCausalLM)

            required_kwargs = model.get_required_kwargs()

            expected_kwargs = ["pixel_values", "vision_mask", "image_grid_thw"]
            assert required_kwargs == expected_kwargs

    def test_get_vision_compiler_args(self, mock_full_config):
        """Test get_vision_compiler_args returns correct string"""
        from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl import NeuronQwen2VLForCausalLM

        with patch('neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl.NeuronBaseForImageToText.__init__'):
            model = NeuronQwen2VLForCausalLM.__new__(NeuronQwen2VLForCausalLM)
            model.vision_config = mock_full_config.vision_config

            compiler_args = model.get_vision_compiler_args()

            assert "--auto-cast=none" in compiler_args
            assert "--model-type=transformer" in compiler_args
            assert "-O1" in compiler_args
            assert f"--cc-pipeline-tiling-factor={mock_full_config.vision_config.neuron_config.cc_pipeline_tiling_factor}" in compiler_args

    def test_get_compiler_args(self, mock_full_config):
        """Test get_compiler_args returns correct string for text model"""
        from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl import NeuronQwen2VLForCausalLM

        with patch('neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl.NeuronBaseForImageToText.__init__'):
            model = NeuronQwen2VLForCausalLM.__new__(NeuronQwen2VLForCausalLM)
            model.text_config = mock_full_config.text_config

            compiler_args = model.get_compiler_args()

            assert "--auto-cast=none" in compiler_args
            assert "--model-type=transformer" in compiler_args
            assert "-O1" in compiler_args
            assert f"--cc-pipeline-tiling-factor={mock_full_config.text_config.neuron_config.cc_pipeline_tiling_factor}" in compiler_args
