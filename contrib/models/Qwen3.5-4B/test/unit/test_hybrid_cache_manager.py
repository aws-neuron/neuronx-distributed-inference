# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
from math import prod
from unittest.mock import patch

import torch

_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)

from neuronx_distributed_inference.models.config import NeuronConfig
from src.modeling_qwen35 import HybridDeltaNetCacheManager, Qwen35InferenceConfig


def _make_config(**overrides):
    neuron_overrides = overrides.pop("neuron_overrides", {})
    neuron_kwargs = dict(
        tp_degree=overrides.pop("tp_degree", 4),
        batch_size=1,
        max_batch_size=2,
        kv_cache_batch_size=2,
        seq_len=16,
        torch_dtype=torch.bfloat16,
    )
    neuron_kwargs.update(neuron_overrides)
    neuron_config = NeuronConfig(**neuron_kwargs)
    defaults = dict(
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        intermediate_size=9216,
        vocab_size=248320,
        rms_norm_eps=1e-6,
        max_position_embeddings=262144,
        rope_theta=10000000,
        hidden_act="silu",
        tie_word_embeddings=True,
        linear_num_value_heads=32,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        use_hybrid_cache_manager=True,
    )
    defaults.update(overrides)
    return Qwen35InferenceConfig(neuron_config=neuron_config, **defaults)


def _numel(shape):
    return prod(int(dim) for dim in shape)


def _managed_cache_numel(mgr):
    return sum(param.numel() for param in mgr.past_key_values)


def _deltanet_state_numel(config, max_batch_size):
    recurrent = (
        max_batch_size
        * config.linear_num_value_heads
        * config.linear_key_head_dim
        * config.linear_value_head_dim
    )
    conv_dim = (
        2 * config.linear_num_key_heads * config.linear_key_head_dim
        + config.linear_num_value_heads * config.linear_value_head_dim
    )
    conv = max_batch_size * conv_dim * (config.linear_conv_kernel_dim - 1)
    return recurrent + conv


class TestHybridDeltaNetCacheManager(unittest.TestCase):
    def test_allocates_per_layer_cache_shapes(self):
        config = _make_config()
        mgr = HybridDeltaNetCacheManager(config, num_kv_head=config.num_key_value_heads)

        self.assertEqual(len(mgr.past_key_values), config.num_hidden_layers * 2)
        self.assertEqual(list(mgr.past_key_values[0].shape), [2, 32, 128, 128])
        self.assertEqual(list(mgr.past_key_values[1].shape), [2, 8192, 3])
        self.assertEqual(mgr.layer_types[3], "full_attention")
        self.assertEqual(mgr.past_key_values[6].dim(), 4)
        self.assertEqual(mgr.past_key_values[7].shape[2], 16)

    def test_get_cache_slices_only_full_attention_layers(self):
        config = _make_config()
        mgr = HybridDeltaNetCacheManager(config, num_kv_head=config.num_key_value_heads)

        cache = mgr.get_cache(seq_len=4, seq_ids=torch.tensor([1]))
        recurrent_state, conv_state = cache[0]
        full_k, full_v = cache[3]

        self.assertEqual(list(recurrent_state.shape), [1, 32, 128, 128])
        self.assertEqual(list(conv_state.shape), [1, 8192, 3])
        self.assertEqual(full_k.shape[0], 2)
        self.assertEqual(full_v.shape[0], 2)
        self.assertEqual(full_k.shape[2], 4)
        self.assertEqual(full_v.shape[2], 4)

    def test_get_seq_length_uses_first_full_attention_layer(self):
        config = _make_config()
        mgr = HybridDeltaNetCacheManager(config, num_kv_head=config.num_key_value_heads)

        nested_cache = mgr.get_cache(seq_len=5, seq_ids=torch.tensor([0]))
        flat_cache = [tensor for layer_cache in nested_cache for tensor in layer_cache]

        self.assertEqual(nested_cache[0][1].shape[2], 3)
        self.assertEqual(mgr.get_seq_length(nested_cache), 5)
        self.assertEqual(mgr.get_seq_length(flat_cache), 5)

    def test_get_cache_selects_deltanet_state_rows_by_seq_ids(self):
        config = _make_config()
        mgr = HybridDeltaNetCacheManager(config, num_kv_head=config.num_key_value_heads)
        with torch.no_grad():
            mgr.past_key_values[0][0, ...].fill_(7)
            mgr.past_key_values[0][1, ...].fill_(13)
            mgr.past_key_values[1][0, ...].fill_(17)
            mgr.past_key_values[1][1, ...].fill_(19)

        recurrent_state, conv_state = mgr.get_cache(
            seq_len=4,
            seq_ids=torch.tensor([1, 0]),
        )[0]

        self.assertTrue(torch.all(recurrent_state[0] == 13))
        self.assertTrue(torch.all(recurrent_state[1] == 7))
        self.assertTrue(torch.all(conv_state[0] == 19))
        self.assertTrue(torch.all(conv_state[1] == 17))

    def test_deltanet_update_scatters_by_seq_id(self):
        config = _make_config()
        mgr = HybridDeltaNetCacheManager(config, num_kv_head=config.num_key_value_heads)
        recurrent = torch.ones((1, 32, 128, 128), dtype=torch.bfloat16)
        conv = torch.ones((1, 8192, 3), dtype=torch.bfloat16)

        updated_recurrent, updated_conv = mgr.update_deltanet_state_by_layer_id(
            idx=0,
            seq_ids=torch.tensor([1]),
            state_per_layer=(recurrent, conv),
        )

        self.assertTrue(torch.all(updated_recurrent[0] == 0))
        self.assertTrue(torch.all(updated_conv[0] == 0))
        self.assertTrue(torch.all(updated_recurrent[1] == 1))
        self.assertTrue(torch.all(updated_conv[1] == 1))

    def test_deltanet_full_batch_update_replaces_state_cache(self):
        config = _make_config()
        mgr = HybridDeltaNetCacheManager(config, num_kv_head=config.num_key_value_heads)
        recurrent = torch.ones((2, 32, 128, 128), dtype=torch.bfloat16)
        conv = torch.ones((2, 8192, 3), dtype=torch.bfloat16)
        recurrent[0].fill_(3)
        recurrent[1].fill_(5)
        conv[0].fill_(11)
        conv[1].fill_(13)

        updated_recurrent, updated_conv = mgr.update_deltanet_state_by_layer_id(
            idx=0,
            seq_ids=None,
            state_per_layer=(recurrent, conv),
        )

        self.assertTrue(torch.all(updated_recurrent[0] == 3))
        self.assertTrue(torch.all(updated_recurrent[1] == 5))
        self.assertTrue(torch.all(updated_conv[0] == 11))
        self.assertTrue(torch.all(updated_conv[1] == 13))

    def test_deltanet_full_batch_update_scatters_non_identity_seq_ids(self):
        config = _make_config()
        mgr = HybridDeltaNetCacheManager(config, num_kv_head=config.num_key_value_heads)
        recurrent = torch.ones((2, 32, 128, 128), dtype=torch.bfloat16)
        conv = torch.ones((2, 8192, 3), dtype=torch.bfloat16)
        recurrent[0].fill_(3)
        recurrent[1].fill_(5)
        conv[0].fill_(11)
        conv[1].fill_(13)

        updated_recurrent, updated_conv = mgr.update_deltanet_state_by_layer_id(
            idx=0,
            seq_ids=torch.tensor([1, 0]),
            state_per_layer=(recurrent, conv),
        )

        self.assertTrue(torch.all(updated_recurrent[0] == 5))
        self.assertTrue(torch.all(updated_recurrent[1] == 3))
        self.assertTrue(torch.all(updated_conv[0] == 13))
        self.assertTrue(torch.all(updated_conv[1] == 11))

    def test_deltanet_update_maps_out_of_range_seq_id_to_padding_row(self):
        config = _make_config(neuron_overrides={"kv_cache_padding_size": 1})
        mgr = HybridDeltaNetCacheManager(config, num_kv_head=config.num_key_value_heads)
        recurrent = torch.ones((1, 32, 128, 128), dtype=torch.bfloat16)
        conv = torch.ones((1, 8192, 3), dtype=torch.bfloat16)

        updated_recurrent, updated_conv = mgr.update_deltanet_state_by_layer_id(
            idx=0,
            seq_ids=torch.tensor([99]),
            state_per_layer=(recurrent, conv),
        )

        self.assertTrue(torch.all(updated_recurrent[0] == 0))
        self.assertTrue(torch.all(updated_recurrent[1] == 0))
        self.assertTrue(torch.all(updated_recurrent[2] == 1))
        self.assertTrue(torch.all(updated_conv[2] == 1))

    def test_deltanet_state_shapes_do_not_scale_with_sequence_length(self):
        short_config = _make_config(neuron_overrides={"seq_len": 128})
        long_config = _make_config(neuron_overrides={"seq_len": 2048})
        short_mgr = HybridDeltaNetCacheManager(
            short_config, num_kv_head=short_config.num_key_value_heads
        )
        long_mgr = HybridDeltaNetCacheManager(
            long_config, num_kv_head=long_config.num_key_value_heads
        )

        self.assertEqual(short_mgr.past_key_values[0].shape, long_mgr.past_key_values[0].shape)
        self.assertEqual(short_mgr.past_key_values[1].shape, long_mgr.past_key_values[1].shape)
        self.assertLess(short_mgr.past_key_values[7].shape[2], long_mgr.past_key_values[7].shape[2])

    def test_get_cache_trims_padding_row_without_seq_ids(self):
        config = _make_config(neuron_overrides={"kv_cache_padding_size": 1})
        mgr = HybridDeltaNetCacheManager(config, num_kv_head=config.num_key_value_heads)

        recurrent_state, conv_state = mgr.get_cache(seq_len=4)[0]

        self.assertEqual(list(recurrent_state.shape), [2, 32, 128, 128])
        self.assertEqual(list(conv_state.shape), [2, 8192, 3])

    def test_update_cache_dispatches_deltanet_and_full_attention_layers(self):
        config = _make_config()
        mgr = HybridDeltaNetCacheManager(config, num_kv_head=config.num_key_value_heads)
        new_key_values = []
        for idx in range(4):
            first = mgr.past_key_values[2 * idx]
            second = mgr.past_key_values[2 * idx + 1]
            new_key_values.append(
                (
                    torch.full_like(first, fill_value=idx + 1),
                    torch.full_like(second, fill_value=idx + 11),
                )
            )

        position_ids = torch.arange(16, dtype=torch.long).unsqueeze(0).expand(2, -1)
        full_k_update = torch.full_like(mgr.past_key_values[6], fill_value=4)
        full_v_update = torch.full_like(mgr.past_key_values[7], fill_value=14)
        with patch.object(
            mgr, "update_kv_by_layer_id", return_value=(full_k_update, full_v_update)
        ) as update_kv:
            updated = mgr.update_cache(
                is_for_context_encoding=True,
                seq_ids=torch.tensor([0, 1], dtype=torch.int32),
                position_ids=position_ids,
                new_key_values=new_key_values,
                seq_len=16,
            )

        self.assertEqual(update_kv.call_count, 1)
        self.assertEqual(update_kv.call_args.kwargs["idx"], 3)
        self.assertTrue(torch.all(updated[0] == 1))
        self.assertTrue(torch.all(updated[1] == 11))
        self.assertTrue(torch.all(updated[6] == 4))
        self.assertTrue(torch.all(updated[7] == 14))

    def test_managed_cache_removes_dummy_kv_for_deltanet_layers(self):
        config = _make_config(neuron_overrides={"seq_len": 1024})
        mgr = HybridDeltaNetCacheManager(config, num_kv_head=config.num_key_value_heads)
        max_batch_size = (
            config.neuron_config.kv_cache_batch_size
            + config.neuron_config.kv_cache_padding_size
        )
        full_kv_per_layer = _numel(mgr.k_shape) + _numel(mgr.v_shape)
        deltanet_layers = config.layer_types.count("linear_attention")
        legacy_total_numel = (
            full_kv_per_layer * config.num_hidden_layers
            + _deltanet_state_numel(config, max_batch_size) * deltanet_layers
        )
        expected_savings = full_kv_per_layer * deltanet_layers

        self.assertEqual(
            legacy_total_numel - _managed_cache_numel(mgr),
            expected_savings,
        )
        self.assertLess(_managed_cache_numel(mgr), legacy_total_numel)

    def test_rejects_unsupported_hybrid_modes(self):
        unsupported_cases = [
            ({"padding_side": "left"}, "left padding"),
            ({"flash_decoding_enabled": True}, "flash decoding"),
        ]

        for neuron_overrides, expected_error in unsupported_cases:
            with self.subTest(expected_error=expected_error):
                config = _make_config(neuron_overrides=neuron_overrides)
                with self.assertRaisesRegex(ValueError, expected_error):
                    HybridDeltaNetCacheManager(
                        config, num_kv_head=config.num_key_value_heads
                    )

        config = _make_config()
        config.neuron_config.kv_cache_quant = True
        with self.assertRaisesRegex(ValueError, "KV cache quantization"):
            HybridDeltaNetCacheManager(config, num_kv_head=config.num_key_value_heads)

        config = _make_config(
            neuron_overrides={
                "attention_dp_degree": 2,
                "batch_size": 2,
                "ctx_batch_size": 2,
                "tkg_batch_size": 2,
                "max_batch_size": 2,
                "kv_cache_batch_size": 2,
                "is_continuous_batching": True,
            }
        )
        with self.assertRaisesRegex(ValueError, "attention data parallelism"):
            HybridDeltaNetCacheManager(config, num_kv_head=config.num_key_value_heads)

        config = _make_config()
        config.neuron_config.kv_cache_tiling = True
        with self.assertRaisesRegex(ValueError, "KV cache tiling"):
            HybridDeltaNetCacheManager(config, num_kv_head=config.num_key_value_heads)

    def test_legacy_config_default_is_disabled(self):
        config = _make_config(use_hybrid_cache_manager=False)
        self.assertFalse(config.use_hybrid_cache_manager)


if __name__ == "__main__":
    unittest.main()
