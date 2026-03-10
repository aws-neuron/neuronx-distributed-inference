import unittest
import math
from unittest.mock import MagicMock

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeInferenceConfig,
)
from neuronx_distributed_inference.models.config import (
    MoENeuronConfig,
    SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config


class TestMaybePadIntermediate(unittest.TestCase):
    """Test suite for maybe_pad_intermediate method"""

    def setUp(self):
        """Set up base configuration for tests"""
        self.base_hf_config = Qwen3MoeConfig(
            _attn_implementation="eager",
            attention_bias=False,
            attention_dropout=0.0,
            decoder_sparse_step=1,
            bos_token_id=151643,
            eos_token_id=151645,
            head_dim=128,
            hidden_act="silu",
            hidden_size=4096,
            initializer_range=0.02,
            intermediate_size=12288,
            max_position_embeddings=40960,
            max_window_layers=94,
            mlp_only_layers=None,
            moe_intermediate_size=1536,  # Base intermediate size
            norm_topk_prob=True,
            num_attention_heads=64,
            num_experts=128,
            num_experts_per_tok=8,
            num_hidden_layers=94,
            num_key_value_heads=4,
            output_router_logits=False,
            rms_norm_eps=1e-6,
            rope_scaling=None,
            rope_theta=1000000.0,
            router_aux_loss_coef=0.001,
            sliding_window=None,
            tie_word_embeddings=False,
            torch_dtype="float32",
            use_cache=True,
            use_sliding_window=False,
            vocab_size=151936,
        )

    def test_no_padding_when_shard_on_intermediate_disabled(self):
        """Test that no padding occurs when use_shard_on_intermediate_dynamic_while is False"""
        neuron_config = MoENeuronConfig(
            tp_degree=1,
            moe_tp_degree=1,
            batch_size=1,
            max_context_length=512,
            seq_len=512 * 10,
            torch_dtype="float32",
        )
        # Ensure shard on intermediate is disabled (default)
        neuron_config.blockwise_matmul_config.use_shard_on_intermediate_dynamic_while = False

        # Use an intermediate size that would require padding if the feature was enabled
        self.base_hf_config.moe_intermediate_size = 1500  # Not divisible by 256

        inference_config = Qwen3MoeInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(hf_config=self.base_hf_config),
        )

        # Verify no padding was applied
        self.assertEqual(inference_config.moe_intermediate_size, 1500)
        self.assertEqual(getattr(inference_config, "moe_intermediate_pad_size", 0), 0)

    def test_no_padding_when_divisible_by_256(self):
        """Test that no padding occurs when I_TP is already divisible by 256"""
        neuron_config = MoENeuronConfig(
            tp_degree=1,
            moe_tp_degree=4,
            batch_size=1,
            max_context_length=512,
            seq_len=512 * 10,
            torch_dtype="float32",
        )
        # Enable shard on intermediate
        neuron_config.blockwise_matmul_config.use_shard_on_intermediate_dynamic_while = True

        # Use an intermediate size where I_TP is divisible by 256
        # With moe_tp_degree=4, I_TP = 1024 / 4 = 256 (divisible by 256)
        self.base_hf_config.moe_intermediate_size = 1024

        inference_config = Qwen3MoeInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(hf_config=self.base_hf_config),
        )

        # Verify no padding was applied
        self.assertEqual(inference_config.moe_intermediate_size, 1024)
        self.assertEqual(getattr(inference_config, "moe_intermediate_pad_size", 0), 0)

    def test_padding_when_not_divisible_by_256(self):
        """Test that padding is applied when I_TP is not divisible by 256"""
        moe_tp_degree = 4
        neuron_config = MoENeuronConfig(
            tp_degree=1,
            moe_tp_degree=moe_tp_degree,
            batch_size=1,
            max_context_length=512,
            seq_len=512 * 10,
            torch_dtype="float32",
        )
        # Enable shard on intermediate
        neuron_config.blockwise_matmul_config.use_shard_on_intermediate_dynamic_while = True

        # Use an intermediate size where I_TP is not divisible by 256
        # With moe_tp_degree=4, I_TP = 1536 / 4 = 384 (not divisible by 256)
        original_size = 1536
        self.base_hf_config.moe_intermediate_size = original_size

        inference_config = Qwen3MoeInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(hf_config=self.base_hf_config),
        )

        # Calculate expected values
        I_TP = original_size // moe_tp_degree  # 384
        expected_padded_I_TP = math.ceil(I_TP / SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP) * SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP  # ceil(384/256) * 256 = 2 * 256 = 512
        expected_padded_size = expected_padded_I_TP * moe_tp_degree  # 512 * 4 = 2048
        expected_pad_size = expected_padded_size - original_size  # 2048 - 1536 = 512

        # Verify padding was applied correctly
        self.assertEqual(inference_config.moe_intermediate_size, expected_padded_size)
        self.assertEqual(inference_config.moe_intermediate_pad_size, expected_pad_size)

    def test_padding_with_moe_tp_degree_1(self):
        """Test padding calculation with moe_tp_degree=1"""
        moe_tp_degree = 1
        neuron_config = MoENeuronConfig(
            tp_degree=1,
            moe_tp_degree=moe_tp_degree,
            batch_size=1,
            max_context_length=512,
            seq_len=512 * 10,
            torch_dtype="float32",
        )
        # Enable shard on intermediate
        neuron_config.blockwise_matmul_config.use_shard_on_intermediate_dynamic_while = True

        # Use an intermediate size not divisible by 256
        # With moe_tp_degree=1, I_TP = 300 / 1 = 300 (not divisible by 256)
        original_size = 300
        self.base_hf_config.moe_intermediate_size = original_size

        inference_config = Qwen3MoeInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(hf_config=self.base_hf_config),
        )

        # Calculate expected values
        I_TP = original_size // moe_tp_degree  # 300
        expected_padded_I_TP = math.ceil(I_TP / SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP) * SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP  # ceil(300/256) * 256 = 2 * 256 = 512
        expected_padded_size = expected_padded_I_TP * moe_tp_degree  # 512 * 1 = 512
        expected_pad_size = expected_padded_size - original_size  # 512 - 300 = 212

        # Verify padding was applied correctly
        self.assertEqual(inference_config.moe_intermediate_size, expected_padded_size)
        self.assertEqual(inference_config.moe_intermediate_pad_size, expected_pad_size)

    def test_padding_with_different_moe_tp_degrees(self):
        """Test padding calculation with various moe_tp_degree values"""
        test_cases = [
            # (moe_tp_degree, original_size, expected_padded_size, expected_pad_size)
            (2, 600, 1024, 424),     # I_TP=300 -> 512, padded=512*2=1024
            (4, 1500, 2048, 548),    # I_TP=375 -> 512, padded=512*4=2048
            (8, 2000, 2048, 48),     # I_TP=250 -> 256, padded=256*8=2048
        ]

        for moe_tp_degree, original_size, expected_padded_size, expected_pad_size in test_cases:
            with self.subTest(moe_tp_degree=moe_tp_degree, original_size=original_size):
                neuron_config = MoENeuronConfig(
                    tp_degree=1,
                    moe_tp_degree=moe_tp_degree,
                    batch_size=1,
                    max_context_length=512,
                    seq_len=512 * 10,
                    torch_dtype="float32",
                )
                # Enable shard on intermediate
                neuron_config.blockwise_matmul_config.use_shard_on_intermediate_dynamic_while = True

                self.base_hf_config.moe_intermediate_size = original_size

                inference_config = Qwen3MoeInferenceConfig(
                    neuron_config,
                    load_config=load_pretrained_config(hf_config=self.base_hf_config),
                )

                # Verify padding was applied correctly
                self.assertEqual(
                    inference_config.moe_intermediate_size,
                    expected_padded_size,
                    f"moe_tp_degree={moe_tp_degree}, original={original_size}"
                )
                self.assertEqual(
                    inference_config.moe_intermediate_pad_size,
                    expected_pad_size,
                    f"moe_tp_degree={moe_tp_degree}, original={original_size}"
                )

    def test_padding_edge_case_just_above_multiple(self):
        """Test padding when I_TP is just above a multiple of 256"""
        moe_tp_degree = 1
        neuron_config = MoENeuronConfig(
            tp_degree=1,
            moe_tp_degree=moe_tp_degree,
            batch_size=1,
            max_context_length=512,
            seq_len=512 * 10,
            torch_dtype="float32",
        )
        # Enable shard on intermediate
        neuron_config.blockwise_matmul_config.use_shard_on_intermediate_dynamic_while = True

        # Use I_TP = 257 (just above 256)
        original_size = 257
        self.base_hf_config.moe_intermediate_size = original_size

        inference_config = Qwen3MoeInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(hf_config=self.base_hf_config),
        )

        # Should pad to 512 (next multiple)
        expected_padded_size = 512
        expected_pad_size = 512 - 257

        self.assertEqual(inference_config.moe_intermediate_size, expected_padded_size)
        self.assertEqual(inference_config.moe_intermediate_pad_size, expected_pad_size)

    def test_padding_with_large_intermediate_size(self):
        """Test padding with a large intermediate size"""
        moe_tp_degree = 8
        neuron_config = MoENeuronConfig(
            tp_degree=1,
            moe_tp_degree=moe_tp_degree,
            batch_size=1,
            max_context_length=512,
            seq_len=512 * 10,
            torch_dtype="float32",
        )
        # Enable shard on intermediate
        neuron_config.blockwise_matmul_config.use_shard_on_intermediate_dynamic_while = True

        # Large intermediate size
        # I_TP = 10000 / 8 = 1250, which is not divisible by 256
        # ceil(1250 / 256) = 5, so padded I_TP = 5 * 256 = 1280
        # padded size = 1280 * 8 = 10240
        original_size = 10000
        self.base_hf_config.moe_intermediate_size = original_size

        inference_config = Qwen3MoeInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(hf_config=self.base_hf_config),
        )

        expected_padded_size = 10240
        expected_pad_size = 240

        self.assertEqual(inference_config.moe_intermediate_size, expected_padded_size)
        self.assertEqual(inference_config.moe_intermediate_pad_size, expected_pad_size)

    def test_pad_size_is_always_non_negative(self):
        """Test that moe_intermediate_pad_size is always >= 0"""
        moe_tp_degree = 4
        neuron_config = MoENeuronConfig(
            tp_degree=1,
            moe_tp_degree=moe_tp_degree,
            batch_size=1,
            max_context_length=512,
            seq_len=512 * 10,
            torch_dtype="float32",
        )
        # Enable shard on intermediate
        neuron_config.blockwise_matmul_config.use_shard_on_intermediate_dynamic_while = True

        # Test with a size that's already properly aligned
        self.base_hf_config.moe_intermediate_size = 1024  # I_TP = 256, already aligned

        inference_config = Qwen3MoeInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(hf_config=self.base_hf_config),
        )

        # Pad size should be 0 (attribute not set when no padding needed, so use getattr with default)
        pad_size = getattr(inference_config, "moe_intermediate_pad_size", 0)
        self.assertEqual(pad_size, 0)
        self.assertGreaterEqual(pad_size, 0)


if __name__ == "__main__":
    unittest.main()
