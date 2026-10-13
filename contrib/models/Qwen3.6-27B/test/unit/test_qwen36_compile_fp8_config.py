# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parents[5]
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

_COMPILE_PATH = (
    _REPO_ROOT
    / "contrib"
    / "models"
    / "Qwen3.6-27B"
    / "test"
    / "integration"
    / "qwen36_27b_compile_fp8.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "qwen36_compile_fp8_under_test",
    _COMPILE_PATH,
)
_COMPILE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _COMPILE
_SPEC.loader.exec_module(_COMPILE)


class _FakeQwen35InferenceConfig:
    def __init__(self, *, neuron_config, **config_dict):
        self.neuron_config = neuron_config
        self.config_dict = config_dict


class _FakeNeuronConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.output_logits = kwargs.get("output_logits", False)
        self.on_device_sampling_config = kwargs.get("on_device_sampling_config")
        self.disable_argmax_kernel = kwargs.get("disable_argmax_kernel", False)
        self.disable_context_encoding_argmax_kernel = kwargs.get(
            "disable_context_encoding_argmax_kernel", False
        )


class _FakeOnDeviceSamplingConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeChunkedPrefillConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _fake_config_module():
    module = types.ModuleType("neuronx_distributed_inference.models.config")
    module.NeuronConfig = _FakeNeuronConfig
    module.OnDeviceSamplingConfig = _FakeOnDeviceSamplingConfig
    module.ChunkedPrefillConfig = _FakeChunkedPrefillConfig
    return module


def _fake_qwen_module():
    module = types.ModuleType("src.modeling_qwen35")
    module.Qwen35InferenceConfig = _FakeQwen35InferenceConfig
    return module


def _args(**overrides):
    defaults = dict(
        model_path="/tmp/qwen36",
        quantized_checkpoints_path="/tmp/qwen36-fp8",
        weight_dtype="fp8_mlp_only",
        seq_len=2048,
        max_context_length=None,
        cte_bucket=512,
        cte_buckets=["256,512"],
        prefix_buckets=None,
        context_encoding_bucket_pairs=None,
        omit_zero_prefix_pair=False,
        token_generation_buckets=None,
        token_generation_batches=None,
        disable_token_generation_wlo=False,
        weights_to_skip_layout_optimization=None,
        block_size=256,
        pa_num_blocks=8,
        pa_headroom_blocks=0,
        tp_degree=4,
        logical_nc_config=2,
        max_num_seqs=1,
        ctx_batch_size=1,
        skip_warmup=False,
        async_mode=False,
        enable_prefix_caching=True,
        enable_hybrid_apc=True,
        enable_vllm_chunked_prefill=False,
        text_only_cte=True,
        compact_cte_attention_mask=True,
        cold_zero_conv_fast_path=False,
        enable_deltanet_decode_nki=False,
        deltanet_cte_backend="env",
        disable_on_device_sampling=True,
        disable_argmax_kernel=False,
        disable_context_encoding_argmax_kernel=False,
        output_logits_with_on_device_sampling=False,
        kernel_q_tile_size=128,
        kernel_kv_tile_size=1024,
        enable_fused_qkv=False,
        enable_qkv_nki_kernels=False,
        enable_qkv_cte_nki_kernel_fuse_rope=False,
        enable_qwen_qk_norm_rope_nki_kernel=False,
        enable_qwen_output_gate_nki_kernel=False,
        enable_qwen_qkv_gate_packed_kernel=False,
        enable_qwen_gated_o_proj_nki_kernel=False,
        enable_split_qkv_tkg_nki_kernel=False,
        enable_attn_block_tkg_nki_kernel=False,
        enable_attn_block_tkg_cascaded_attention=False,
        enable_attn_block_tkg_cache_update=False,
        enable_out_proj_nki_kernel=False,
        enable_mlp_cte_nki_kernel=False,
        enable_mlp_tkg_nki_kernel=False,
        enable_quantized_mlp_kernel=False,
        enable_k_cache_transposed=False,
        enable_kv_cache_quant=False,
        prefix_cte_attention_chunk_size=None,
        prefix_cte_attention_backend="attention_cte",
        prefix_cte_attention_segment_size=None,
        disable_static_hybrid_cache=False,
        gdn_checkpoint_interval=256,
        max_gdn_checkpoint_slots=8,
        gdn_recurrent_cache_dtype="float32",
        gdn_conv_cache_dtype="bfloat16",
        hybrid_cache_mode="all",
        hybrid_apc_require_vllm_metadata=False,
        hybrid_apc_enable_backed_prefix_reads=False,
        hybrid_apc_commit_during_token_generation=False,
        quantize_edge_mlp_layers=False,
        quantize_lm_head=False,
        fp8_quantize_linear_attn_gates=False,
        fp8_exclude_groups=[],
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestQwen36CompileFp8Config(unittest.TestCase):
    def test_fp8_environment_defaults_are_set_without_overriding_existing(self):
        with patch.dict(os.environ, {}, clear=True):
            _COMPILE._ensure_fp8_environment()
            self.assertEqual(os.environ["XLA_HANDLE_SPECIAL_SCALAR"], "1")
            self.assertEqual(os.environ["UNSAFE_FP8FNCAST"], "1")

        with patch.dict(
            os.environ,
            {
                "XLA_HANDLE_SPECIAL_SCALAR": "custom",
                "UNSAFE_FP8FNCAST": "custom",
            },
            clear=True,
        ):
            _COMPILE._ensure_fp8_environment()
            self.assertEqual(os.environ["XLA_HANDLE_SPECIAL_SCALAR"], "custom")
            self.assertEqual(os.environ["UNSAFE_FP8FNCAST"], "custom")

    def test_host_sampling_compile_keeps_output_logits_enabled(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(disable_on_device_sampling=True),
            )

        self.assertTrue(config.neuron_config.output_logits)
        self.assertIsNone(config.neuron_config.on_device_sampling_config)
        self.assertEqual(config.neuron_config.pa_num_blocks, 8)
        self.assertTrue(config.neuron_config.quantized)

    def test_full_fp8_keeps_hybrid_checkpoint_bank_out_of_conversion(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, modules = _COMPILE._build_config(
                _args(weight_dtype="fp8_full", quantize_lm_head=True),
            )

        self.assertEqual(config.config_dict["gdn_recurrent_cache_dtype"], "float32")
        self.assertIn("hybrid_gdn_checkpoint_cache.recurrent_slots", modules)
        self.assertIn("hybrid_gdn_checkpoint_cache.conv_slots", modules)
        self.assertIn(
            "hybrid_gdn_checkpoint_cache.recurrent_slots",
            config.neuron_config.modules_to_not_convert,
        )

    def test_compile_can_trace_batched_token_generation(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(
                    disable_on_device_sampling=True,
                    weight_dtype="bf16_control",
                    quantized_checkpoints_path=None,
                    max_num_seqs=2,
                    ctx_batch_size=1,
                    skip_warmup=True,
                    pa_num_blocks=16,
                ),
            )

        self.assertEqual(config.neuron_config.batch_size, 2)
        self.assertEqual(config.neuron_config.ctx_batch_size, 1)
        self.assertEqual(config.neuron_config.tkg_batch_size, 2)
        self.assertEqual(config.neuron_config.pa_num_blocks, 16)
        self.assertTrue(config.neuron_config.skip_warmup)

    def test_compile_can_enable_block_tkg_attention_kernel_flags(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(
                    enable_qkv_nki_kernels=True,
                    enable_attn_block_tkg_nki_kernel=True,
                    enable_attn_block_tkg_cascaded_attention=True,
                    enable_attn_block_tkg_cache_update=True,
                ),
            )

        self.assertTrue(config.neuron_config.qkv_kernel_enabled)
        self.assertTrue(config.neuron_config.qkv_nki_kernel_enabled)
        self.assertTrue(config.neuron_config.fused_qkv)
        self.assertTrue(config.neuron_config.attn_block_tkg_nki_kernel_enabled)
        self.assertTrue(
            config.neuron_config.attn_block_tkg_nki_kernel_cascaded_attention,
        )
        self.assertTrue(config.neuron_config.attn_block_tkg_nki_kernel_cache_update)

    def test_compile_can_enable_fused_qkv_without_qkv_kernel(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(enable_fused_qkv=True),
            )

        self.assertTrue(config.neuron_config.fused_qkv)
        self.assertFalse(
            getattr(config.neuron_config, "qkv_kernel_enabled", False),
        )
        self.assertFalse(
            getattr(config.neuron_config, "qkv_nki_kernel_enabled", False),
        )

    def test_compile_can_enable_qkv_cte_rope_fusion_flag(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={
                "num_hidden_layers": 2,
                "head_dim": 256,
                "rope_dim": 256,
            },
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(
                    enable_qkv_nki_kernels=True,
                    enable_qkv_cte_nki_kernel_fuse_rope=True,
                ),
            )

        self.assertTrue(config.neuron_config.qkv_kernel_enabled)
        self.assertTrue(config.neuron_config.qkv_nki_kernel_enabled)
        self.assertTrue(config.neuron_config.qkv_cte_nki_kernel_fuse_rope)

    def test_compile_rejects_qkv_cte_rope_fusion_for_partial_rope(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={
                "num_hidden_layers": 2,
                "head_dim": 256,
                "rope_dim": 64,
            },
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            with self.assertRaisesRegex(ValueError, "partial-RoPE Qwen3.6"):
                _COMPILE._build_config(
                    _args(
                        enable_qkv_nki_kernels=True,
                        enable_qkv_cte_nki_kernel_fuse_rope=True,
                    ),
                )

    def test_compile_can_enable_qwen_qk_norm_rope_nki_kernel(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(enable_qwen_qk_norm_rope_nki_kernel=True),
            )

        self.assertTrue(config.config_dict["use_qwen_qk_norm_rope_nki"])

    def test_compile_can_enable_qwen_output_gate_nki_kernel(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(enable_qwen_output_gate_nki_kernel=True),
            )

        self.assertTrue(config.config_dict["use_qwen_output_gate_nki"])

    def test_compile_can_enable_qwen_qkv_gate_packed_kernel(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(enable_qwen_qkv_gate_packed_kernel=True),
            )

        self.assertTrue(config.config_dict["use_qwen_qkv_gate_packed"])

    def test_compile_can_enable_qwen_gated_o_proj_nki_kernel(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(enable_qwen_gated_o_proj_nki_kernel=True),
            )

        self.assertTrue(config.config_dict["use_qwen_gated_o_proj_nki"])

    def test_compile_can_enable_split_qkv_tkg_kernel_without_stock_qkv(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(enable_split_qkv_tkg_nki_kernel=True),
            )

        self.assertTrue(config.neuron_config.qkv_tkg_nki_kernel_enabled)
        self.assertFalse(getattr(config.neuron_config, "fused_qkv", False))
        self.assertFalse(
            getattr(config.neuron_config, "qkv_kernel_enabled", False),
        )
        self.assertFalse(
            getattr(config.neuron_config, "qkv_nki_kernel_enabled", False),
        )

    def test_compile_can_enable_output_projection_kernel(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(enable_out_proj_nki_kernel=True),
            )

        self.assertTrue(config.neuron_config.out_proj_kernel_enabled)

    def test_compile_can_enable_quantized_mlp_tkg_kernel(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(
                    weight_dtype="fp8_full",
                    enable_mlp_tkg_nki_kernel=True,
                    enable_quantized_mlp_kernel=True,
                ),
            )

        self.assertTrue(config.neuron_config.mlp_kernel_enabled)
        self.assertTrue(config.neuron_config.mlp_tkg_nki_kernel_enabled)
        self.assertTrue(config.neuron_config.quantized_mlp_kernel_enabled)

    def test_compile_can_enable_quantized_mlp_cte_kernel_without_tkg_flag(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(
                    weight_dtype="fp8_full",
                    enable_mlp_cte_nki_kernel=True,
                    enable_quantized_mlp_kernel=True,
                ),
            )

        self.assertTrue(config.neuron_config.mlp_kernel_enabled)
        self.assertFalse(
            getattr(config.neuron_config, "mlp_tkg_nki_kernel_enabled", False)
        )
        self.assertTrue(config.neuron_config.quantized_mlp_kernel_enabled)

    def test_decode_memory_flags_are_forwarded(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(
                    enable_k_cache_transposed=True,
                    enable_kv_cache_quant=True,
                    hybrid_apc_commit_during_token_generation=True,
                ),
            )

        self.assertTrue(config.neuron_config.k_cache_transposed)
        self.assertTrue(config.neuron_config.kv_cache_quant)
        self.assertEqual(config.neuron_config.kv_quant_config, {"direct_cast": True})
        self.assertTrue(
            config.config_dict["hybrid_apc_commit_during_token_generation"],
        )

    def test_on_device_sampling_compile_uses_sampler_config(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(disable_on_device_sampling=False),
            )

        self.assertIsNotNone(config.neuron_config.on_device_sampling_config)
        self.assertFalse(config.neuron_config.output_logits)
        self.assertTrue(config.neuron_config.vocab_parallel)
        self.assertEqual(config.neuron_config.pa_num_blocks, 8)

    def test_on_device_sampling_can_disable_custom_argmax_kernel(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(
                    disable_on_device_sampling=False,
                    disable_argmax_kernel=True,
                ),
            )

        self.assertIsNotNone(config.neuron_config.on_device_sampling_config)
        self.assertTrue(config.neuron_config.vocab_parallel)
        self.assertTrue(config.neuron_config.disable_argmax_kernel)

    def test_on_device_sampling_can_disable_context_encoding_argmax_kernel(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(
                    disable_on_device_sampling=False,
                    disable_context_encoding_argmax_kernel=True,
                ),
            )

        self.assertIsNotNone(config.neuron_config.on_device_sampling_config)
        self.assertTrue(config.neuron_config.vocab_parallel)
        self.assertFalse(config.neuron_config.disable_argmax_kernel)
        self.assertTrue(config.neuron_config.disable_context_encoding_argmax_kernel)

    def test_on_device_sampling_can_also_return_logits_for_debug(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(
                    disable_on_device_sampling=False,
                    output_logits_with_on_device_sampling=True,
                ),
            )

        self.assertIsNotNone(config.neuron_config.on_device_sampling_config)
        self.assertTrue(config.neuron_config.output_logits)
        self.assertTrue(config.neuron_config.vocab_parallel)

    def test_bf16_control_compile_disables_quantization_and_keeps_host_logits(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, modules = _COMPILE._build_config(
                _args(
                    disable_on_device_sampling=True,
                    weight_dtype="bf16_control",
                    quantized_checkpoints_path=None,
                ),
            )

        self.assertTrue(config.neuron_config.output_logits)
        self.assertFalse(config.neuron_config.quantized)
        self.assertIsNone(config.neuron_config.on_device_sampling_config)
        self.assertGreater(len(modules), 0)

    def test_fp8_mlp_only_keeps_edge_mlp_layers_in_bf16_by_default(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 4},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            _config, modules = _COMPILE._build_config(
                _args(disable_on_device_sampling=True),
            )

        self.assertIn("layers.0.mlp", modules)
        self.assertIn("layers.3.mlp", modules)
        self.assertNotIn("layers.1.mlp", modules)

    def test_fp8_full_quantizes_attention_and_edge_mlp_by_default(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 4},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, modules = _COMPILE._build_config(
                _args(disable_on_device_sampling=True, weight_dtype="fp8_full"),
            )

        self.assertTrue(config.neuron_config.quantized)
        self.assertIn(
            r".*\.scale$",
            config.neuron_config.weights_to_skip_layout_optimization,
        )
        self.assertIn(
            r".*\.weight_scale$",
            config.neuron_config.weights_to_skip_layout_optimization,
        )
        self.assertIn(
            r".*linear_attn\.conv1d_weight\.weight$",
            config.neuron_config.weights_to_skip_layout_optimization,
        )
        self.assertNotIn("layers.0.mlp", modules)
        self.assertNotIn("layers.3.mlp", modules)
        self.assertNotIn("layers.0.self_attn", modules)
        self.assertNotIn("layers.0.linear_attn", modules)
        self.assertIn("layers.0.linear_attn.conv1d_weight", modules)
        self.assertIn("layers.0.linear_attn.A_log_weight", modules)
        self.assertIn("layers.0.linear_attn.dt_bias_weight", modules)
        self.assertIn("layers.0.linear_attn.in_proj_a", modules)
        self.assertIn("layers.0.linear_attn.in_proj_b", modules)
        self.assertIn("layers.0.linear_attn.in_proj_ba", modules)
        self.assertIn("lm_head", modules)

    def test_fp8_full_can_quantize_lm_head_when_requested(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            _config, modules = _COMPILE._build_config(
                _args(
                    disable_on_device_sampling=True,
                    weight_dtype="fp8_full",
                    quantize_lm_head=True,
                ),
            )

        self.assertNotIn("lm_head", modules)
        self.assertNotIn("model.lm_head", modules)

    def test_fp8_full_keeps_linear_attention_gate_projections_bf16(self):
        self.assertFalse(
            _COMPILE._is_full_fp8_weight(
                "layers.0.linear_attn.in_proj_a.weight",
                quantize_lm_head=True,
            ),
        )
        self.assertFalse(
            _COMPILE._is_full_fp8_weight(
                "layers.0.linear_attn.in_proj_b.weight",
                quantize_lm_head=True,
            ),
        )
        self.assertFalse(
            _COMPILE._is_full_fp8_weight(
                "layers.0.linear_attn.in_proj_ba.weight",
                quantize_lm_head=True,
            ),
        )
        self.assertTrue(
            _COMPILE._is_full_fp8_weight(
                "layers.0.linear_attn.in_proj_qkv.weight",
                quantize_lm_head=True,
            ),
        )

    def test_fp8_full_can_use_legacy_fp8_linear_attention_gate_policy(self):
        self.assertTrue(
            _COMPILE._is_full_fp8_weight(
                "layers.0.linear_attn.in_proj_a.weight",
                quantize_lm_head=True,
                quantize_linear_attn_gates=True,
            ),
        )
        self.assertTrue(
            _COMPILE._is_full_fp8_weight(
                "layers.0.linear_attn.in_proj_b.weight",
                quantize_lm_head=True,
                quantize_linear_attn_gates=True,
            ),
        )
        self.assertFalse(
            _COMPILE._is_full_fp8_weight(
                "layers.0.linear_attn.in_proj_ba.weight",
                quantize_lm_head=True,
                quantize_linear_attn_gates=True,
            ),
        )

    def test_legacy_fp8_linear_attention_gate_policy_matches_old_config(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            _config, modules = _COMPILE._build_config(
                _args(
                    weight_dtype="fp8_full",
                    quantize_lm_head=True,
                    fp8_quantize_linear_attn_gates=True,
                ),
            )

        self.assertNotIn("layers.0.linear_attn.in_proj_a", modules)
        self.assertNotIn("layers.0.linear_attn.in_proj_b", modules)
        self.assertNotIn("layers.0.linear_attn.in_proj_ba", modules)

    def test_fp8_full_can_exclude_remaining_linear_attention_matmuls(self):
        for weight_name in (
            "layers.0.linear_attn.in_proj_qkv.weight",
            "layers.0.linear_attn.in_proj_z.weight",
            "layers.0.linear_attn.out_proj.weight",
        ):
            self.assertFalse(
                _COMPILE._is_full_fp8_weight(
                    weight_name,
                    quantize_lm_head=False,
                    fp8_exclude_groups={"linear_attn"},
                ),
            )
        self.assertTrue(
            _COMPILE._is_full_fp8_weight(
                "layers.0.mlp.up_proj.weight",
                quantize_lm_head=False,
                fp8_exclude_groups={"linear_attn"},
            ),
        )

    def test_fp8_full_exclude_groups_are_reflected_in_config(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, modules = _COMPILE._build_config(
                _args(
                    weight_dtype="fp8_full",
                    fp8_exclude_groups=["linear_attn", "mlp"],
                ),
            )

        self.assertIn("layers.0.linear_attn", modules)
        self.assertIn("layers.0.mlp", modules)
        self.assertIn("model.layers.1.linear_attn", modules)
        self.assertIn("layers.0.linear_attn", config.neuron_config.modules_to_not_convert)

    def test_user_wlo_skip_patterns_are_appended_and_deduplicated(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(
                    disable_on_device_sampling=True,
                    weight_dtype="fp8_full",
                    weights_to_skip_layout_optimization=[
                        r".*\.scale$",
                        r".*custom_skip.*",
                    ],
                ),
            )

        self.assertEqual(
            config.neuron_config.weights_to_skip_layout_optimization,
            [
                r".*\.scale$",
                r".*\.weight_scale$",
                r".*linear_attn\.conv1d_weight\.weight$",
                r".*custom_skip.*",
            ],
        )

    def test_bf16_control_does_not_add_fp8_wlo_skips_by_default(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(
                    disable_on_device_sampling=True,
                    weight_dtype="bf16_control",
                    quantized_checkpoints_path=None,
                ),
            )

        self.assertFalse(
            hasattr(config.neuron_config, "weights_to_skip_layout_optimization"),
        )

    def test_compile_can_disable_token_generation_wlo(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(disable_token_generation_wlo=True),
            )

        self.assertTrue(config.config_dict["disable_token_generation_wlo"])

    def test_compile_can_disable_token_generation_wlo_from_env(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            os.environ,
            {"QWEN36_DISABLE_TOKEN_GENERATION_WLO": "1"},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(_args())

        self.assertTrue(config.config_dict["disable_token_generation_wlo"])

    def test_compile_forwards_cold_cte_fast_path_flags(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(
                    text_only_cte=False,
                    compact_cte_attention_mask=False,
                    cold_zero_conv_fast_path=True,
                ),
            )

        self.assertFalse(config.config_dict["use_text_only_cte_inputs"])
        self.assertFalse(config.config_dict["use_compact_cte_attention_mask"])
        self.assertTrue(config.config_dict["use_cold_zero_conv_fast_path"])

    def test_long_prefix_buckets_must_fit_max_context_length(self):
        with self.assertRaisesRegex(ValueError, "Largest prefix bucket"):
            _COMPILE._validate_prefix_buckets_fit_context(
                _args(enable_prefix_caching=True),
                max_context_length=512,
                prefix_buckets=[512, 131072],
            )

    def test_sparse_context_encoding_bucket_pairs_are_forwarded(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(
                    cte_buckets=["512,1536"],
                    prefix_buckets=["256,512,65536"],
                    max_context_length=65536,
                    seq_len=65536,
                    pa_num_blocks=256,
                    context_encoding_bucket_pairs=[
                        "512:256,512:512",
                        "1536:256",
                        "1536:65536",
                    ],
                ),
            )

        self.assertEqual(
            config.neuron_config.context_encoding_bucket_pairs,
            [
                [512, 0],
                [512, 256],
                [512, 512],
                [1536, 0],
                [1536, 256],
                [1536, 65536],
            ],
        )

    def test_prefix_cte_attention_chunk_size_is_forwarded(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(prefix_cte_attention_chunk_size=32768),
            )

        self.assertEqual(config.neuron_config.prefix_cte_attention_chunk_size, 32768)

    def test_segmented_prefix_cte_attention_config_is_forwarded(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(
                    prefix_cte_attention_backend="segmented_cte",
                    prefix_cte_attention_segment_size=32768,
                ),
            )

        self.assertEqual(
            config.neuron_config.prefix_cte_attention_backend,
            "segmented_cte",
        )
        self.assertEqual(
            config.neuron_config.prefix_cte_attention_segment_size,
            32768,
        )

    def test_sparse_context_encoding_bucket_pairs_can_omit_zero_pair(self):
        pairs = _COMPILE._context_encoding_bucket_pairs(
            _args(
                cte_buckets=["3072"],
                prefix_buckets=["131072"],
                context_encoding_bucket_pairs=["3072:131072"],
                omit_zero_prefix_pair=True,
            ),
            cte_buckets=[3072],
            prefix_buckets=[131072],
        )

        self.assertEqual(pairs, [[3072, 131072]])

    def test_sparse_context_encoding_bucket_pairs_validate_config_buckets(self):
        with self.assertRaisesRegex(ValueError, "active bucket"):
            _COMPILE._context_encoding_bucket_pairs(
                _args(context_encoding_bucket_pairs=["768:256"]),
                cte_buckets=[512],
                prefix_buckets=[256],
            )

        with self.assertRaisesRegex(ValueError, "prefix bucket"):
            _COMPILE._context_encoding_bucket_pairs(
                _args(context_encoding_bucket_pairs=["512:1024"]),
                cte_buckets=[512],
                prefix_buckets=[256],
            )

    def test_pa_num_blocks_rejects_user_blocks_below_sequence_requirement(self):
        with self.assertRaisesRegex(ValueError, "need at least 8"):
            _COMPILE._pa_num_blocks(_args(pa_num_blocks=7))

    def test_pa_headroom_blocks_extend_default_pa_capacity(self):
        args = _args(
            seq_len=4096,
            block_size=32,
            max_num_seqs=2,
            pa_num_blocks=None,
            pa_headroom_blocks=32,
        )

        self.assertEqual(_COMPILE._pa_min_blocks(args), 256)
        self.assertEqual(_COMPILE._pa_requested_blocks(args), 288)
        self.assertEqual(_COMPILE._pa_num_blocks(args), 288)

    def test_base_compile_work_dir_defaults_next_to_artifacts(self):
        with self.subTest("default"), patch.dict(os.environ, {}, clear=True):
            work_dir = _COMPILE._configure_base_compile_work_dir(
                Path("/tmp/qwen_artifacts/model_a"),
                None,
            )

            self.assertEqual(
                work_dir,
                Path("/tmp/qwen_artifacts/_nxd_model_workdir").resolve(),
            )
            self.assertEqual(os.environ["BASE_COMPILE_WORK_DIR"], str(work_dir))

        with self.subTest("existing env"), patch.dict(
            os.environ,
            {"BASE_COMPILE_WORK_DIR": "/tmp/existing_nxd_workdir"},
            clear=True,
        ):
            work_dir = _COMPILE._configure_base_compile_work_dir(
                Path("/tmp/qwen_artifacts/model_a"),
                None,
            )

            self.assertEqual(work_dir, Path("/tmp/existing_nxd_workdir").resolve())
            self.assertEqual(os.environ["BASE_COMPILE_WORK_DIR"], str(work_dir))

        with self.subTest("explicit override"), patch.dict(
            os.environ,
            {"BASE_COMPILE_WORK_DIR": "/tmp/existing_nxd_workdir"},
            clear=True,
        ):
            work_dir = _COMPILE._configure_base_compile_work_dir(
                Path("/tmp/qwen_artifacts/model_a"),
                "/tmp/explicit_nxd_workdir",
            )

            self.assertEqual(work_dir, Path("/tmp/explicit_nxd_workdir").resolve())
            self.assertEqual(os.environ["BASE_COMPILE_WORK_DIR"], str(work_dir))

    def test_deltanet_cte_backend_preserves_environment_by_default(self):
        with patch.dict(
            os.environ,
            {
                "USE_NKI_FUSED": "custom",
                "USE_NKI_CHUNKED": "custom",
            },
            clear=True,
        ):
            _COMPILE._configure_deltanet_cte_backend("env")

            self.assertEqual(os.environ["USE_NKI_FUSED"], "custom")
            self.assertEqual(os.environ["USE_NKI_CHUNKED"], "custom")

    def test_deltanet_cte_backend_can_force_nki_chunked(self):
        with patch.dict(
            os.environ,
            {
                "USE_NKI_FUSED": "1",
                "USE_PYTORCH_CHUNK": "1",
                "DELTANET_SEQUENTIAL": "1",
            },
            clear=True,
        ):
            _COMPILE._configure_deltanet_cte_backend("nki_chunked")

            self.assertEqual(os.environ["USE_NKI_FUSED"], "0")
            self.assertEqual(os.environ["USE_NKI_CHUNKED"], "1")
            self.assertNotIn("USE_PYTORCH_CHUNK", os.environ)
            self.assertNotIn("DELTANET_SEQUENTIAL", os.environ)

    def test_deltanet_cte_backend_can_force_pytorch_chunk(self):
        with patch.dict(os.environ, {"USE_NKI_CHUNKED": "1"}, clear=True):
            _COMPILE._configure_deltanet_cte_backend("pytorch_chunk")

            self.assertEqual(os.environ["USE_NKI_FUSED"], "0")
            self.assertEqual(os.environ["USE_PYTORCH_CHUNK"], "1")
            self.assertNotIn("USE_NKI_CHUNKED", os.environ)

    def test_backed_prefix_read_compile_flag_is_forwarded(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(hybrid_apc_enable_backed_prefix_reads=True),
            )

        self.assertTrue(config.config_dict["hybrid_apc_enable_backed_prefix_reads"])

    def test_vllm_chunked_prefill_uses_qwen_flags_not_nxdi_chunked_prefill(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(enable_vllm_chunked_prefill=True),
            )

        self.assertTrue(config.neuron_config.is_block_kv_layout)
        self.assertIsNone(getattr(config.neuron_config, "chunked_prefill_config", None))
        self.assertTrue(config.config_dict["use_qwen_hybrid_chunked_prefill"])
        self.assertTrue(config.config_dict["use_qwen_hybrid_chunked_prefill_nki"])

    def test_deltanet_decode_nki_compile_flag_is_forwarded(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(enable_deltanet_decode_nki=True),
            )

        self.assertTrue(config.config_dict["use_qwen_deltanet_decode_nki"])

    def test_checkpoint_bank_weights_are_added_for_reload(self):
        from safetensors import safe_open
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            compiled_path = Path(tmpdir)
            weights_dir = compiled_path / "weights"
            weights_dir.mkdir()
            shard_path = weights_dir / "tp0_sharded_checkpoint.safetensors"
            save_file(
                {"existing.weight": _COMPILE.torch.ones(1)},
                shard_path,
                metadata={"format": "pt"},
            )

            inf_config = types.SimpleNamespace(
                layer_types=["linear_attention", "full_attention", "linear_attention"],
                linear_num_value_heads=48,
                linear_num_key_heads=16,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_conv_kernel_dim=4,
                max_gdn_checkpoint_slots=64,
                hybrid_recurrent_cache_dtype="float32",
                hybrid_conv_cache_dtype="bfloat16",
                neuron_config=types.SimpleNamespace(
                    tp_degree=4,
                    torch_dtype=_COMPILE.torch.bfloat16,
                ),
            )

            _COMPILE._ensure_hybrid_checkpoint_weights(compiled_path, inf_config)

            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                recurrent = handle.get_tensor(
                    "hybrid_gdn_checkpoint_cache.recurrent_slots.0",
                )
                conv = handle.get_tensor("hybrid_gdn_checkpoint_cache.conv_slots.0")

            self.assertIn("existing.weight", keys)
            self.assertIn("hybrid_gdn_checkpoint_cache.recurrent_slots.1", keys)
            self.assertIn("hybrid_gdn_checkpoint_cache.conv_slots.1", keys)
            self.assertEqual(recurrent.dtype, _COMPILE.torch.float32)
            self.assertEqual(tuple(recurrent.shape), (64, 12, 128, 128))
            self.assertEqual(conv.dtype, _COMPILE.torch.bfloat16)
            self.assertEqual(tuple(conv.shape), (64, 2560, 3))


if __name__ == "__main__":
    unittest.main()
