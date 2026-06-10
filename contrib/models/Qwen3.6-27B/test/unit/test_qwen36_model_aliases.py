# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import inspect
import math
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


_REPO_ROOT = Path(__file__).resolve().parents[5]
_QWEN_MODEL_PATH = (
    _REPO_ROOT / "contrib" / "models" / "Qwen3.6-27B" / "src" / "modeling_qwen35.py"
)


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _jit(*_args, **_kwargs):
    def decorator(fn):
        return fn

    return decorator


class _FakeDecoderModelInstance:
    def get(self, bucket_rank, **kwargs):
        del bucket_rank, kwargs
        num_outputs = 1 if not self.neuron_config.output_logits else 2
        kvs = self.module.kv_mgr.past_key_values
        aliases = {param: num_outputs + i for i, param in enumerate(kvs)}
        self.input_output_aliases = aliases
        return self.module, aliases


class _FakeModelWrapper:
    def input_generator(self):
        return self._base_inputs

    def pad_inputs(self, *args, pad_type="first_fit"):
        del pad_type
        return args


def _fake_modules():
    return {
        "nki": _module("nki", jit=_jit),
        "neuronxcc": _package("neuronxcc"),
        "neuronxcc.nki": _package("neuronxcc.nki"),
        "neuronxcc.nki._private_kernels": _package("neuronxcc.nki._private_kernels"),
        "neuronxcc.nki._private_kernels.attention": _module(
            "neuronxcc.nki._private_kernels.attention",
            attention_isa_kernel=lambda *args, **kwargs: None,
        ),
        "neuronx_distributed": _package("neuronx_distributed"),
        "neuronx_distributed.parallel_layers": _package(
            "neuronx_distributed.parallel_layers"
        ),
        "neuronx_distributed.parallel_layers.parallel_state": _module(
            "neuronx_distributed.parallel_layers.parallel_state",
            get_tensor_model_parallel_rank=lambda: 0,
        ),
        "neuronx_distributed.parallel_layers.layers": _module(
            "neuronx_distributed.parallel_layers.layers",
            ColumnParallelLinear=nn.Linear,
            ParallelEmbedding=nn.Embedding,
            RowParallelLinear=nn.Linear,
        ),
        "neuronx_distributed.parallel_layers.mappings": _module(
            "neuronx_distributed.parallel_layers.mappings",
            _gather_along_dim=lambda tensor, *_args, **_kwargs: tensor,
        ),
        "neuronx_distributed.utils": _module(
            "neuronx_distributed.utils",
            cpu_mode=lambda: True,
        ),
        "transformers": _package("transformers"),
        "transformers.models": _package("transformers.models"),
        "transformers.models.qwen3_moe": _package("transformers.models.qwen3_moe"),
        "transformers.models.qwen3_moe.modeling_qwen3_moe": _module(
            "transformers.models.qwen3_moe.modeling_qwen3_moe",
            Qwen3MoeRMSNorm=nn.LayerNorm,
        ),
        "src": _package("src"),
        "src.nki_kernels": _package("src.nki_kernels"),
        "src.nki_kernels.nki_deltanet": _module(
            "src.nki_kernels.nki_deltanet",
            deltanet_recurrent_fwd=lambda *args, **kwargs: None,
            deltanet_recurrent_fwd_state=lambda *args, **kwargs: None,
            deltanet_recurrent_step_batched=lambda *args, **kwargs: None,
        ),
        "src.nki_kernels.nki_deltanet_chunked": _module(
            "src.nki_kernels.nki_deltanet_chunked",
            deltanet_chunk_step=lambda *args, **kwargs: None,
        ),
        "src.nki_kernels.nki_deltanet_fused": _module(
            "src.nki_kernels.nki_deltanet_fused",
            deltanet_autocp_affine_sequence=lambda *args, **kwargs: None,
            deltanet_autocp_apply_output=lambda *args, **kwargs: None,
            deltanet_autocp_prefix_apply_output=lambda *args, **kwargs: None,
            deltanet_autocp_state_summary_sequence=lambda *args, **kwargs: None,
            deltanet_autocp_state_prefix=lambda *args, **kwargs: None,
            deltanet_fused_chunked_fwd=lambda *args, **kwargs: None,
            deltanet_fused_chunked_fwd_multihead=lambda *args, **kwargs: None,
            _make_lower_mask=lambda *args, **kwargs: None,
            _make_lower_mask_diag=lambda *args, **kwargs: None,
            _make_identity=lambda *args, **kwargs: None,
        ),
        "src.nki_kernels.nki_deltanet_fused_legacy": _module(
            "src.nki_kernels.nki_deltanet_fused_legacy",
            deltanet_fused_chunked_fwd=lambda *args, **kwargs: None,
        ),
        "src.hybrid_apc": _module(
            "src.hybrid_apc",
            HybridAPCMetadataStore=object,
            HybridAPCSchedulerBridge=object,
            HybridAPCSlotAllocator=object,
        ),
        "neuronx_distributed_inference": _package("neuronx_distributed_inference"),
        "neuronx_distributed_inference.models": _package(
            "neuronx_distributed_inference.models"
        ),
        "neuronx_distributed_inference.models.config": _module(
            "neuronx_distributed_inference.models.config",
            InferenceConfig=object,
            NeuronConfig=object,
        ),
        "neuronx_distributed_inference.models.llama": _package(
            "neuronx_distributed_inference.models.llama"
        ),
        "neuronx_distributed_inference.models.llama.modeling_llama": _module(
            "neuronx_distributed_inference.models.llama.modeling_llama",
            NeuronLlamaMLP=nn.Module,
        ),
        "neuronx_distributed_inference.models.model_base": _module(
            "neuronx_distributed_inference.models.model_base",
            NeuronBaseForCausalLM=object,
            NeuronBaseModel=nn.Module,
            mask_padded_logits=lambda logits, *_args, **_kwargs: logits,
        ),
        "neuronx_distributed_inference.models.model_wrapper": _module(
            "neuronx_distributed_inference.models.model_wrapper",
            CONTEXT_ENCODING_MODEL_TAG="context_encoding_model",
            TOKEN_GENERATION_MODEL_TAG="token_generation_model",
            DecoderModelInstance=_FakeDecoderModelInstance,
            ModelWrapper=_FakeModelWrapper,
        ),
        "neuronx_distributed_inference.utils": _package(
            "neuronx_distributed_inference.utils"
        ),
        "neuronx_distributed_inference.utils.distributed": _module(
            "neuronx_distributed_inference.utils.distributed",
            get_tp_group=lambda *_args, **_kwargs: None,
        ),
        "neuronx_distributed_inference.modules": _package(
            "neuronx_distributed_inference.modules"
        ),
        "neuronx_distributed_inference.modules.async_execution": _module(
            "neuronx_distributed_inference.modules.async_execution",
            cancel_hybrid_apc_request=lambda *args, **kwargs: None,
            finish_hybrid_apc_request=lambda *args, **kwargs: None,
            prepare_hybrid_apc_model_inputs=lambda *args, **kwargs: (),
            prepare_hybrid_apc_request_for_execution=lambda *args, **kwargs: None,
        ),
        "neuronx_distributed_inference.modules.custom_calls": _module(
            "neuronx_distributed_inference.modules.custom_calls",
            CustomRMSNorm=nn.LayerNorm,
        ),
        "neuronx_distributed_inference.modules.attention": _package(
            "neuronx_distributed_inference.modules.attention"
        ),
        "neuronx_distributed_inference.modules.attention.attention_base": _module(
            "neuronx_distributed_inference.modules.attention.attention_base",
            NeuronAttentionBase=nn.Module,
        ),
        "neuronx_distributed_inference.modules.attention.utils": _module(
            "neuronx_distributed_inference.modules.attention.utils",
            RotaryEmbedding=object,
            move_heads_front=lambda tensor, *_args, **_kwargs: tensor,
            transpose_parallel_linear_layer=lambda weight: weight,
        ),
        "neuronx_distributed_inference.modules.kvcache": _package(
            "neuronx_distributed_inference.modules.kvcache"
        ),
        "neuronx_distributed_inference.modules.kvcache.block_kv_cache_manager": _module(
            "neuronx_distributed_inference.modules.kvcache.block_kv_cache_manager",
            BlockKVCacheManager=object,
        ),
        "neuronx_distributed_inference.modules.kvcache.kv_cache_manager": _module(
            "neuronx_distributed_inference.modules.kvcache.kv_cache_manager",
            KVCacheManager=object,
        ),
        "neuronx_distributed_inference.models.layer_boundary_marker": _module(
            "neuronx_distributed_inference.models.layer_boundary_marker",
            ModuleMarkerEndWrapper=object,
            ModuleMarkerStartWrapper=object,
        ),
    }


def _load_qwen_module():
    spec = importlib.util.spec_from_file_location(
        "qwen36_model_aliases_under_test",
        _QWEN_MODEL_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with patch.dict(sys.modules, _fake_modules()):
        spec.loader.exec_module(module)
    return module


def _make_instance(qwen_module, *, output_logits, on_device_sampling_config=None):
    kv0, kv1, state, checkpoint = (torch.nn.Parameter(torch.zeros(1)) for _ in range(4))
    module = SimpleNamespace(
        kv_mgr=SimpleNamespace(past_key_values=[kv0, kv1]),
        config=SimpleNamespace(use_hybrid_cache_manager=False),
        _deltanet_state_params=[state],
        _hybrid_gdn_checkpoint_params=[checkpoint],
    )
    instance = qwen_module.Qwen35DecoderModelInstance.__new__(
        qwen_module.Qwen35DecoderModelInstance
    )
    instance.neuron_config = SimpleNamespace(
        output_logits=output_logits,
        on_device_sampling_config=on_device_sampling_config,
    )
    instance.module = module
    return instance, (kv0, kv1, state, checkpoint)


def _make_wrapper(qwen_module, *, tag, use_hybrid_apc_manager=True):
    wrapper = qwen_module.Qwen35ModelWrapper.__new__(qwen_module.Qwen35ModelWrapper)
    wrapper.tag = tag
    wrapper.config = SimpleNamespace(
        hidden_size=8,
        neuron_config=SimpleNamespace(torch_dtype=torch.bfloat16),
        use_text_only_cte_inputs=True,
        use_hybrid_apc_manager=use_hybrid_apc_manager,
    )
    wrapper._base_inputs = [
        (
            torch.ones((1, 1), dtype=torch.int32),  # input_ids
            torch.ones((1, 1), dtype=torch.int32),  # attention_mask
            torch.ones((1, 1), dtype=torch.int32),  # position_ids
            torch.zeros((1,), dtype=torch.int32),  # seq_ids
            torch.ones((1, 3), dtype=torch.float32),  # sampling_params
            torch.empty(0),
            torch.zeros((1,), dtype=torch.int32),  # adapter_ids
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.zeros((1, 1), dtype=torch.int32),  # slot_mapping
            torch.zeros((1, 1), dtype=torch.int32),  # block_table
            torch.ones((1, 1), dtype=torch.int32),  # num_queries
            torch.zeros((1, 1), dtype=torch.int32),  # computed_context_lens
        )
    ]
    return wrapper


class _IdentityMarker:
    def __call__(self, tensor):
        return tensor


class _RecordingNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, tensor):
        self.calls += 1
        return tensor + 1


class _RecordingMlp:
    def __init__(self):
        self.calls = []

    def __call__(self, tensor, rmsnorm=None):
        self.calls.append((tensor.clone(), rmsnorm))
        return tensor + 2, None


class _FakeDeltaNetAttention:
    def __call__(self, hidden_states, **_kwargs):
        return torch.zeros_like(hidden_states), ("k", "v"), None, None


def _make_decoder_layer_for_mlp_test(qwen_module):
    qwen_module.ModuleMarkerStartWrapper = _IdentityMarker
    qwen_module.ModuleMarkerEndWrapper = _IdentityMarker
    layer = qwen_module.NeuronQwen35DecoderLayer.__new__(
        qwen_module.NeuronQwen35DecoderLayer
    )
    nn.Module.__init__(layer)
    layer.layer_type = "linear_attention"
    layer.config = SimpleNamespace(use_hybrid_cache_manager=False)
    layer.linear_attn = _FakeDeltaNetAttention()
    layer.input_layernorm = nn.Identity()
    layer.post_attention_layernorm = _RecordingNorm()
    layer.mlp = _RecordingMlp()
    layer.mlp_kernel_enabled = True
    layer.mlp_kernel_fused_rmsnorm = True
    return layer


def _expanded_prefix_attention_reference(
    Q,
    K_cache,
    V_cache,
    query_positions,
    cache_positions,
    key_valid_mask=None,
):
    B, q_heads, q_len, head_dim = Q.shape
    kv_heads = K_cache.shape[1]
    kv_rep = q_heads // kv_heads
    K_full = (
        K_cache.unsqueeze(2)
        .expand(-1, -1, kv_rep, -1, -1)
        .reshape(B, q_heads, K_cache.shape[2], head_dim)
    )
    V_full = (
        V_cache.unsqueeze(2)
        .expand(-1, -1, kv_rep, -1, -1)
        .reshape(B, q_heads, V_cache.shape[2], head_dim)
    )
    if cache_positions.ndim == 4:
        cache_positions = cache_positions.reshape(B, -1)
    if key_valid_mask is not None and key_valid_mask.ndim == 4:
        key_valid_mask = key_valid_mask.reshape(B, -1)

    attn_weights = torch.matmul(Q, K_full.transpose(-1, -2)) / math.sqrt(head_dim)
    causal_mask = cache_positions[:, None, None, :] <= query_positions[
        :, None, :, None
    ]
    if key_valid_mask is not None:
        causal_mask = causal_mask & key_valid_mask[:, None, None, :]
    attn_weights = attn_weights.masked_fill(~causal_mask, -65504.0)
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        Q.dtype
    )
    return torch.matmul(attn_weights, V_full)


class TestQwen36ModelAliases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qwen_module = _load_qwen_module()

    def test_deltanet_multihead_group_defaults_to_lnc2_when_available(self):
        with patch.dict(
            os.environ,
            {"NEURON_CC_FLAGS": "--target trn2 --lnc 2"},
            clear=True,
        ):
            self.assertEqual(
                self.qwen_module._resolve_deltanet_multihead_group_size(4),
                2,
            )

    def test_deltanet_multihead_group_clamps_to_lnc1_by_default(self):
        with patch.dict(
            os.environ,
            {"NEURON_CC_FLAGS": "--target trn2 --lnc 1"},
            clear=True,
        ):
            self.assertEqual(
                self.qwen_module._resolve_deltanet_multihead_group_size(4),
                1,
            )

    def test_deltanet_multihead_group_rejects_explicit_size_above_lnc(self):
        with patch.dict(
            os.environ,
            {
                "NEURON_CC_FLAGS": "--target trn2 --lnc 1",
                "QWEN36_DELTANET_MULTIHEAD_GROUP_SIZE": "2",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "requires NEURON_CC_FLAGS --lnc"):
                self.qwen_module._resolve_deltanet_multihead_group_size(4)

    def test_deltanet_autocp_lnc_defaults_to_lnc2_for_even_chunks(self):
        with patch.dict(
            os.environ,
            {"NEURON_CC_FLAGS": "--target trn2 --lnc 2"},
            clear=True,
        ):
            self.assertEqual(self.qwen_module._resolve_deltanet_autocp_lnc(128), 2)

    def test_deltanet_autocp_lnc_falls_back_to_lnc1_for_odd_chunks(self):
        with patch.dict(
            os.environ,
            {"NEURON_CC_FLAGS": "--target trn2 --lnc 2"},
            clear=True,
        ):
            self.assertEqual(self.qwen_module._resolve_deltanet_autocp_lnc(3), 1)

    def test_deltanet_autocp_lnc_rejects_explicit_uneven_chunks(self):
        with patch.dict(
            os.environ,
            {
                "NEURON_CC_FLAGS": "--target trn2 --lnc 2",
                "QWEN36_DELTANET_AUTOCP_LNC": "2",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "chunks to be divisible"):
                self.qwen_module._resolve_deltanet_autocp_lnc(3)

    def test_grouped_prefix_attention_matches_expanded_gqa_reference(self):
        torch.manual_seed(123)
        batch_size = 2
        q_heads = 6
        kv_heads = 2
        q_len = 5
        cache_len = 12
        head_dim = 8
        Q = torch.randn(batch_size, q_heads, q_len, head_dim)
        K_cache = torch.randn(batch_size, kv_heads, cache_len, head_dim)
        V_cache = torch.randn(batch_size, kv_heads, cache_len, head_dim)
        query_positions = (
            torch.arange(cache_len - q_len, cache_len)
            .view(1, q_len)
            .expand(batch_size, -1)
        )
        cache_positions = torch.arange(cache_len).view(1, cache_len).expand(
            batch_size,
            -1,
        )
        key_valid_mask = torch.ones(batch_size, cache_len, dtype=torch.bool)
        key_valid_mask[1, -2:] = False

        actual = self.qwen_module._qwen35_grouped_prefix_attention(
            Q,
            K_cache,
            V_cache,
            query_positions,
            cache_positions.view(batch_size, 1, 1, cache_len),
            key_valid_mask.view(batch_size, 1, 1, cache_len),
        )
        expected = _expanded_prefix_attention_reference(
            Q,
            K_cache,
            V_cache,
            query_positions,
            cache_positions,
            key_valid_mask,
        )

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_grouped_prefix_attention_matches_mha_reference(self):
        torch.manual_seed(456)
        batch_size = 1
        q_heads = 4
        q_len = 4
        cache_len = 7
        head_dim = 8
        Q = torch.randn(batch_size, q_heads, q_len, head_dim)
        K_cache = torch.randn(batch_size, q_heads, cache_len, head_dim)
        V_cache = torch.randn(batch_size, q_heads, cache_len, head_dim)
        query_positions = torch.arange(cache_len - q_len, cache_len).view(1, q_len)
        cache_positions = torch.arange(cache_len).view(1, cache_len)

        actual = self.qwen_module._qwen35_grouped_prefix_attention(
            Q,
            K_cache,
            V_cache,
            query_positions,
            cache_positions,
        )
        expected = _expanded_prefix_attention_reference(
            Q,
            K_cache,
            V_cache,
            query_positions,
            cache_positions,
        )

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_expanded_prefix_attention_matches_reference(self):
        torch.manual_seed(789)
        batch_size = 1
        q_heads = 8
        kv_heads = 2
        q_len = 6
        cache_len = 11
        head_dim = 8
        Q = torch.randn(batch_size, q_heads, q_len, head_dim)
        K_cache = torch.randn(batch_size, kv_heads, cache_len, head_dim)
        V_cache = torch.randn(batch_size, kv_heads, cache_len, head_dim)
        query_positions = torch.arange(cache_len - q_len, cache_len).view(1, q_len)
        cache_positions = torch.arange(cache_len).view(1, 1, 1, cache_len)
        key_valid_mask = torch.ones(batch_size, 1, 1, cache_len, dtype=torch.bool)
        key_valid_mask[:, :, :, -1] = False

        actual = self.qwen_module._qwen35_expanded_prefix_attention(
            Q,
            K_cache,
            V_cache,
            query_positions,
            cache_positions,
            key_valid_mask,
        )
        expected = _expanded_prefix_attention_reference(
            Q,
            K_cache,
            V_cache,
            query_positions,
            cache_positions,
            key_valid_mask,
        )

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_prefix_attention_impl_env_selects_legacy_expanded(self):
        with patch.dict(
            os.environ,
            {"QWEN36_PREFIX_ATTENTION_IMPL": "legacy_expanded"},
            clear=True,
        ):
            self.assertEqual(
                self.qwen_module._qwen36_prefix_attention_impl(),
                "expanded",
            )

    def test_prefix_attention_impl_rejects_unknown_value(self):
        with patch.dict(
            os.environ,
            {"QWEN36_PREFIX_ATTENTION_IMPL": "bogus"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "QWEN36_PREFIX_ATTENTION_IMPL"):
                self.qwen_module._qwen36_prefix_attention_impl()

    def test_grouped_prefix_attention_rejects_invalid_gqa_shape(self):
        Q = torch.zeros(1, 5, 2, 4)
        K_cache = torch.zeros(1, 2, 3, 4)
        V_cache = torch.zeros(1, 2, 3, 4)
        query_positions = torch.arange(2).view(1, 2)
        cache_positions = torch.arange(3).view(1, 3)

        with self.assertRaisesRegex(ValueError, "q_heads to be divisible"):
            self.qwen_module._qwen35_grouped_prefix_attention(
                Q,
                K_cache,
                V_cache,
                query_positions,
                cache_positions,
            )

    def test_host_logits_aliases_after_single_trace_output(self):
        instance, (kv0, kv1, state, checkpoint) = _make_instance(
            self.qwen_module,
            output_logits=True,
            on_device_sampling_config=None,
        )

        _module, aliases = instance.get(bucket_rank=0)

        self.assertEqual(aliases[kv0], 1)
        self.assertEqual(aliases[kv1], 2)
        self.assertEqual(aliases[state], 3)
        self.assertEqual(aliases[checkpoint], 4)

    def test_on_device_logits_aliases_after_tokens_and_logits(self):
        instance, (kv0, kv1, state, checkpoint) = _make_instance(
            self.qwen_module,
            output_logits=True,
            on_device_sampling_config=object(),
        )

        _module, aliases = instance.get(bucket_rank=0)

        self.assertEqual(aliases[kv0], 2)
        self.assertEqual(aliases[kv1], 3)
        self.assertEqual(aliases[state], 4)
        self.assertEqual(aliases[checkpoint], 5)

    def test_hybrid_checkpoint_aliases_skip_tkg_without_commit(self):
        instance, (kv0, kv1, state, checkpoint) = _make_instance(
            self.qwen_module,
            output_logits=True,
            on_device_sampling_config=object(),
        )
        instance.module.config.use_hybrid_apc_manager = True
        instance.module.config.hybrid_apc_commit_during_token_generation = False
        instance.module.n_active_tokens = 1

        _module, aliases = instance.get(bucket_rank=0)

        self.assertEqual(aliases[kv0], 2)
        self.assertEqual(aliases[kv1], 3)
        self.assertEqual(aliases[state], 4)
        self.assertNotIn(checkpoint, aliases)

    def test_hybrid_checkpoint_aliases_include_tkg_with_commit(self):
        instance, (kv0, kv1, state, checkpoint) = _make_instance(
            self.qwen_module,
            output_logits=True,
            on_device_sampling_config=object(),
        )
        instance.module.config.use_hybrid_apc_manager = True
        instance.module.config.hybrid_apc_commit_during_token_generation = True
        instance.module.n_active_tokens = 1

        _module, aliases = instance.get(bucket_rank=0)

        self.assertEqual(aliases[kv0], 2)
        self.assertEqual(aliases[kv1], 3)
        self.assertEqual(aliases[state], 4)
        self.assertEqual(aliases[checkpoint], 5)

    def test_alias_output_count_guard_rejects_shifted_deltanet_states(self):
        module = SimpleNamespace(
            kv_mgr=SimpleNamespace(past_key_values=[object(), object()]),
            config=SimpleNamespace(
                use_hybrid_cache_manager=False,
                use_hybrid_apc_manager=True,
                hybrid_apc_commit_during_token_generation=False,
            ),
            _deltanet_state_params=[object()],
            _deltanet_updated_states=[torch.zeros(1), torch.zeros(1)],
            _hybrid_gdn_checkpoint_params=[],
            _hybrid_gdn_checkpoint_updated_states=[],
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "_deltanet_updated_states has 2 tensors but _deltanet_state_params has 1",
        ):
            self.qwen_module._qwen36_validate_alias_output_counts(
                module,
                updated_kv_cache=[torch.zeros(1), torch.zeros(1)],
                is_for_context_encoding=True,
            )

    def test_gathered_logits_mask_only_actual_vocab_padding(self):
        lm_head = SimpleNamespace(pad_size=248320, gather_output=True)
        config = SimpleNamespace(vocab_size=248320)

        self.assertEqual(
            self.qwen_module._effective_lm_head_pad_size(
                lm_head, torch.empty(1, 1, 248320), config
            ),
            0,
        )
        self.assertEqual(
            self.qwen_module._effective_lm_head_pad_size(
                lm_head, torch.empty(1, 1, 248336), config
            ),
            16,
        )

    def test_sharded_logits_keep_lm_head_pad_size(self):
        lm_head = SimpleNamespace(pad_size=128, gather_output=False)
        config = SimpleNamespace(vocab_size=248320)

        self.assertEqual(
            self.qwen_module._effective_lm_head_pad_size(
                lm_head, torch.empty(1, 1, 62080), config
            ),
            128,
        )

    def test_on_device_output_logits_are_gathered_before_return(self):
        logits = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)
        gathered = torch.arange(24, dtype=torch.float32).reshape(1, 1, 24)
        lm_head = SimpleNamespace(
            gather_output=False,
            tensor_parallel_group="tp_group",
        )
        neuron_config = SimpleNamespace(
            output_logits=True,
            on_device_sampling_config=object(),
        )

        with patch.object(
            self.qwen_module,
            "_gather_along_dim",
            return_value=gathered,
        ) as gather:
            actual = self.qwen_module._qwen36_output_logits_for_return(
                logits,
                lm_head,
                neuron_config,
            )

        self.assertIs(actual, gathered)
        gather.assert_called_once_with(
            logits,
            partition_dim=2,
            process_group="tp_group",
        )

    def test_output_logits_skip_gather_when_not_vocab_sharded(self):
        logits = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)
        lm_head = SimpleNamespace(gather_output=True)
        neuron_config = SimpleNamespace(
            output_logits=True,
            on_device_sampling_config=object(),
        )

        actual = self.qwen_module._qwen36_output_logits_for_return(
            logits,
            lm_head,
            neuron_config,
        )

        self.assertIs(actual, logits)

    def test_mlp_kernel_cte_keeps_rmsnorm_separate(self):
        layer = _make_decoder_layer_for_mlp_test(self.qwen_module)
        hidden = torch.zeros((1, 4, 4), dtype=torch.float32)

        outputs = layer.forward(hidden, is_for_context_encoding=True)

        self.assertEqual(layer.post_attention_layernorm.calls, 1)
        self.assertEqual(len(layer.mlp.calls), 1)
        mlp_input, fused_rmsnorm = layer.mlp.calls[0]
        self.assertIsNone(fused_rmsnorm)
        self.assertTrue(torch.allclose(mlp_input, torch.ones_like(mlp_input)))
        self.assertEqual(outputs[0].shape, hidden.shape)

    def test_mlp_kernel_tkg_can_fuse_rmsnorm(self):
        layer = _make_decoder_layer_for_mlp_test(self.qwen_module)
        hidden = torch.zeros((1, 1, 4), dtype=torch.float32)

        layer.forward(hidden, is_for_context_encoding=False)

        self.assertEqual(layer.post_attention_layernorm.calls, 0)
        self.assertEqual(len(layer.mlp.calls), 1)
        _mlp_input, fused_rmsnorm = layer.mlp.calls[0]
        self.assertIs(fused_rmsnorm, layer.post_attention_layernorm)

    def test_fused_deltanet_does_not_clamp_cumulative_decay(self):
        self.assertFalse(
            hasattr(self.qwen_module, "_bound_fused_deltanet_log_decay")
        )

    def test_split_qkv_tkg_keeps_output_gate_on_standard_projection(self):
        init_source = inspect.getsource(self.qwen_module.NeuronQwen35Attention.__init__)
        split_tuple_source = init_source.split("split_qkv_projections = (", 1)[1].split(
            ")",
            1,
        )[0]
        self.assertNotIn("output_gate_proj", split_tuple_source)

        forward_source = inspect.getsource(self.qwen_module.NeuronQwen35Attention.forward)
        self.assertIn("gate = self.output_gate_proj(hidden_states)", forward_source)
        self.assertNotIn(
            "self._run_split_qkv_tkg_projection(\n"
            "                hidden_states,\n"
            "                self.output_gate_proj,",
            forward_source,
        )

    def test_hybrid_checkpoint_commit_ignores_inactive_duplicate_slot_rows(self):
        config = SimpleNamespace(
            layer_types=["linear_attention"],
            max_gdn_checkpoint_slots=3,
            linear_num_value_heads=1,
            linear_num_key_heads=1,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_conv_kernel_dim=3,
            hybrid_recurrent_cache_dtype="float32",
            hybrid_conv_cache_dtype="bfloat16",
            neuron_config=SimpleNamespace(tp_degree=1),
        )
        cache = self.qwen_module.HybridGDNCheckpointCache(config)
        with torch.no_grad():
            cache.recurrent_slots[0].copy_(
                torch.arange(12, dtype=torch.float32).reshape(3, 1, 2, 2)
            )
            cache.conv_slots[0].copy_(
                torch.arange(36, dtype=torch.bfloat16).reshape(3, 6, 2)
            )

        old_recurrent = cache.recurrent_slots[0].detach().clone()
        old_conv = cache.conv_slots[0].detach().clone()
        recurrent_state = torch.stack(
            [
                torch.full((1, 2, 2), 101.0),
                torch.full((1, 2, 2), 999.0),
            ]
        )
        conv_state = torch.stack(
            [
                torch.full((6, 2), 11.0, dtype=torch.bfloat16),
                torch.full((6, 2), 99.0, dtype=torch.bfloat16),
            ]
        )

        recurrent_out, conv_out = cache.commit_from_active_rows(
            layer_state_pairs=[(0, recurrent_state, conv_state)],
            seq_ids=torch.tensor([0, 1], dtype=torch.int32),
            checkpoint_slot_ids=torch.tensor([0, 0], dtype=torch.int32),
            commit_mask=torch.tensor([1, 0], dtype=torch.int32),
        )

        self.assertTrue(torch.equal(recurrent_out[0], recurrent_state[0]))
        self.assertTrue(torch.equal(conv_out[0], conv_state[0]))
        self.assertTrue(torch.equal(recurrent_out[1:], old_recurrent[1:]))
        self.assertTrue(torch.equal(conv_out[1:], old_conv[1:]))

    def test_hybrid_checkpoint_bank_reasserts_configured_dtype_after_global_cast(self):
        config = SimpleNamespace(
            layer_types=["linear_attention"],
            max_gdn_checkpoint_slots=3,
            linear_num_value_heads=1,
            linear_num_key_heads=1,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_conv_kernel_dim=3,
            hybrid_recurrent_cache_dtype="float32",
            hybrid_conv_cache_dtype="bfloat16",
            neuron_config=SimpleNamespace(tp_degree=1),
        )
        cache = self.qwen_module.HybridGDNCheckpointCache(config).to(torch.bfloat16)
        linear_attn = SimpleNamespace(
            recurrent_state_buffer=nn.Parameter(
                torch.zeros((1, 1, 2, 2), dtype=torch.bfloat16),
                requires_grad=False,
            ),
            conv_state_buffer=nn.Parameter(
                torch.zeros((1, 6, 2), dtype=torch.float32),
                requires_grad=False,
            ),
        )
        module = SimpleNamespace(
            config=config,
            layers=[SimpleNamespace(linear_attn=linear_attn)],
            hybrid_gdn_checkpoint_cache=cache,
        )

        self.assertEqual(linear_attn.recurrent_state_buffer.dtype, torch.bfloat16)
        self.assertEqual(linear_attn.conv_state_buffer.dtype, torch.float32)
        self.assertEqual(cache.recurrent_slots[0].dtype, torch.bfloat16)

        self.qwen_module._reassert_hybrid_gdn_checkpoint_param_dtypes(module)

        self.assertEqual(linear_attn.recurrent_state_buffer.dtype, torch.float32)
        self.assertEqual(linear_attn.conv_state_buffer.dtype, torch.bfloat16)
        self.assertEqual(cache.recurrent_slots[0].dtype, torch.float32)
        self.assertEqual(cache.conv_slots[0].dtype, torch.bfloat16)
        self.assertEqual(cache.recurrent_dtype, torch.float32)
        self.assertEqual(cache.conv_dtype, torch.bfloat16)

    def test_hybrid_checkpoint_restore_clamps_slots_and_ignores_inactive_rows(self):
        config = SimpleNamespace(
            layer_types=["linear_attention"],
            max_gdn_checkpoint_slots=3,
            linear_num_value_heads=1,
            linear_num_key_heads=1,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_conv_kernel_dim=3,
            hybrid_recurrent_cache_dtype="float32",
            hybrid_conv_cache_dtype="bfloat16",
            neuron_config=SimpleNamespace(tp_degree=1),
        )
        cache = self.qwen_module.HybridGDNCheckpointCache(config)
        with torch.no_grad():
            cache.recurrent_slots[0][0].fill_(10)
            cache.recurrent_slots[0][1].fill_(20)
            cache.recurrent_slots[0][2].fill_(30)
            cache.conv_slots[0][0].fill_(1)
            cache.conv_slots[0][1].fill_(2)
            cache.conv_slots[0][2].fill_(3)

        recurrent_state_buffer = torch.stack(
            [
                torch.full((1, 2, 2), 101.0),
                torch.full((1, 2, 2), 202.0),
            ]
        )
        conv_state_buffer = torch.stack(
            [
                torch.full((6, 2), 11.0, dtype=torch.bfloat16),
                torch.full((6, 2), 22.0, dtype=torch.bfloat16),
            ]
        )
        layers = [
            SimpleNamespace(
                linear_attn=SimpleNamespace(
                    recurrent_state_buffer=recurrent_state_buffer,
                    conv_state_buffer=conv_state_buffer,
                )
            )
        ]

        restored = cache.restore_to_active_rows(
            layers=layers,
            seq_ids=torch.tensor([1, -1], dtype=torch.int32),
            checkpoint_slot_ids=torch.tensor([999, 999], dtype=torch.int32),
            restore_mask=torch.tensor([1, 0], dtype=torch.int32),
        )
        recurrent_out, conv_out = restored[0]

        self.assertTrue(torch.equal(recurrent_out[0], cache.recurrent_slots[0][2]))
        self.assertTrue(torch.equal(conv_out[0], cache.conv_slots[0][2]))
        self.assertTrue(torch.equal(recurrent_out[1], recurrent_state_buffer[0]))
        self.assertTrue(torch.equal(conv_out[1], conv_state_buffer[0]))

    def test_hybrid_checkpoint_restore_zeroes_inactive_rows_for_context_prefill(self):
        config = SimpleNamespace(
            layer_types=["linear_attention"],
            max_gdn_checkpoint_slots=2,
            linear_num_value_heads=1,
            linear_num_key_heads=1,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_conv_kernel_dim=3,
            hybrid_recurrent_cache_dtype="float32",
            hybrid_conv_cache_dtype="bfloat16",
            neuron_config=SimpleNamespace(tp_degree=1),
        )
        cache = self.qwen_module.HybridGDNCheckpointCache(config)
        recurrent_state_buffer = torch.full((1, 1, 2, 2), 101.0)
        conv_state_buffer = torch.full((1, 6, 2), 11.0, dtype=torch.bfloat16)
        layers = [
            SimpleNamespace(
                linear_attn=SimpleNamespace(
                    recurrent_state_buffer=recurrent_state_buffer,
                    conv_state_buffer=conv_state_buffer,
                )
            )
        ]

        restored = cache.restore_to_active_rows(
            layers=layers,
            seq_ids=torch.tensor([0], dtype=torch.int32),
            checkpoint_slot_ids=torch.tensor([0], dtype=torch.int32),
            restore_mask=torch.tensor([0], dtype=torch.int32),
            zero_inactive=True,
        )
        recurrent_out, conv_out = restored[0]

        self.assertTrue(torch.equal(recurrent_out, torch.zeros_like(recurrent_out)))
        self.assertTrue(torch.equal(conv_out, torch.zeros_like(conv_out)))

    def test_legacy_tkg_args_are_env_gated(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(self.qwen_module._use_legacy_tkg_args())
        with patch.dict(os.environ, {"QWEN36_TKG_LEGACY_ARGS": "1"}, clear=True):
            self.assertTrue(self.qwen_module._use_legacy_tkg_args())

    def test_legacy_tkg_uses_prefix_contract_for_cte_trace_args(self):
        wrapper = _make_wrapper(
            self.qwen_module,
            tag=self.qwen_module.CONTEXT_ENCODING_MODEL_TAG,
        )

        with patch.dict(os.environ, {"QWEN36_TKG_LEGACY_ARGS": "1"}, clear=True):
            generated = wrapper.input_generator()[0]

        self.assertEqual(len(generated), 24)
        self.assertEqual(generated[11].shape, (1, 1))
        self.assertEqual(generated[12].shape, (1, 1))
        self.assertEqual(generated[13].shape, (1, 1))
        self.assertEqual(generated[14].shape, (1, 1))

    def test_legacy_tkg_trace_args_keep_prefix_metadata(self):
        wrapper = _make_wrapper(
            self.qwen_module,
            tag=self.qwen_module.TOKEN_GENERATION_MODEL_TAG,
        )

        with patch.dict(os.environ, {"QWEN36_TKG_LEGACY_ARGS": "1"}, clear=True):
            generated = wrapper.input_generator()[0]

        self.assertEqual(len(generated), 24)
        self.assertEqual(generated[11].shape, (1, 1))
        self.assertEqual(generated[12].shape, (1, 1))
        self.assertEqual(generated[13].shape, (1, 1))
        self.assertEqual(generated[14].shape, (1, 1))

    def test_prefix_cache_pad_inputs_expands_minimal_runtime_args(self):
        wrapper = _make_wrapper(
            self.qwen_module,
            tag=self.qwen_module.CONTEXT_ENCODING_MODEL_TAG,
            use_hybrid_apc_manager=False,
        )
        wrapper.is_prefix_caching = True
        wrapper.neuron_config = SimpleNamespace(
            enable_fused_speculation=False,
            enable_eagle_speculation=False,
        )

        padded = wrapper.pad_inputs(*wrapper._base_inputs[0])

        self.assertEqual(len(padded), 24)
        self.assertEqual(padded[15].numel(), 0)
        self.assertEqual(padded[21].shape, (3, 1, 1))

    def test_hybrid_prefix_cache_pad_inputs_expands_minimal_runtime_args(self):
        wrapper = _make_wrapper(
            self.qwen_module,
            tag=self.qwen_module.CONTEXT_ENCODING_MODEL_TAG,
        )
        wrapper.is_prefix_caching = True
        wrapper.neuron_config = SimpleNamespace(
            enable_fused_speculation=False,
            enable_eagle_speculation=False,
        )

        padded = wrapper.pad_inputs(*wrapper._base_inputs[0])

        self.assertEqual(len(padded), 29)
        self.assertEqual(padded[15].numel(), 0)
        self.assertEqual(padded[24].shape, (1,))

    def test_nonlegacy_tkg_trace_args_keep_prefix_and_hybrid_metadata(self):
        wrapper = _make_wrapper(
            self.qwen_module,
            tag=self.qwen_module.TOKEN_GENERATION_MODEL_TAG,
        )

        with patch.dict(os.environ, {}, clear=True):
            generated = wrapper.input_generator()[0]

        self.assertEqual(len(generated), 29)
        self.assertEqual(generated[11].shape, (1, 1))
        self.assertEqual(generated[14].shape, (1, 1))
        self.assertEqual(generated[24].shape, (1,))

    def test_tkg_token_guard_rejects_out_of_vocab_id(self):
        with self.assertRaisesRegex(ValueError, "out-of-vocab token id"):
            self.qwen_module._validate_qwen36_tkg_input_ids(
                torch.tensor([[2143289344]], dtype=torch.int32),
                248320,
            )

    def test_tkg_token_guard_accepts_valid_vocab_id(self):
        self.qwen_module._validate_qwen36_tkg_input_ids(
            torch.tensor([[42]], dtype=torch.int32),
            248320,
        )

    def test_prefill_detection_keeps_nonzero_multi_token_suffix_on_cte(self):
        self.assertTrue(
            self.qwen_module._qwen36_is_prefill_request(
                torch.ones((1, 207), dtype=torch.int32),
                torch.arange(207, 414, dtype=torch.int32).reshape(1, -1),
            )
        )

    def test_prefill_detection_keeps_one_token_nonzero_decode_on_tkg(self):
        self.assertFalse(
            self.qwen_module._qwen36_is_prefill_request(
                torch.ones((1, 1), dtype=torch.int32),
                torch.tensor([[207]], dtype=torch.int32),
            )
        )

    def test_prefill_detection_routes_packed_batched_decode_to_tkg(self):
        self.assertFalse(
            self.qwen_module._qwen36_is_prefill_request(
                torch.ones((1, 2), dtype=torch.int32),
                torch.tensor([[272, 272]], dtype=torch.int32),
                full_context_lens=torch.tensor([273, 273], dtype=torch.int32),
                computed_context_lens=torch.tensor([272, 272], dtype=torch.int32),
                prefill_completion_state=torch.tensor([True, True]),
            )
        )

    def test_prefill_detection_keeps_packed_suffix_prefill_on_cte(self):
        self.assertTrue(
            self.qwen_module._qwen36_is_prefill_request(
                torch.ones((1, 32), dtype=torch.int32),
                torch.arange(256, 288, dtype=torch.int32).reshape(1, -1),
                full_context_lens=torch.tensor([272, 272], dtype=torch.int32),
                computed_context_lens=torch.tensor([256, 256], dtype=torch.int32),
                prefill_completion_state=torch.tensor([True, True]),
            )
        )

    def test_prefill_detection_keeps_incomplete_one_token_prefill_on_cte(self):
        self.assertTrue(
            self.qwen_module._qwen36_is_prefill_request(
                torch.ones((1, 2), dtype=torch.int32),
                torch.tensor([[272, 49]], dtype=torch.int32),
                full_context_lens=torch.tensor([273, 50], dtype=torch.int32),
                computed_context_lens=torch.tensor([272, 49], dtype=torch.int32),
                prefill_completion_state=torch.tensor([True, False]),
            )
        )

    def test_hybrid_apc_controls_need_prepare_for_missing_or_inert_masks(self):
        self.assertTrue(
            self.qwen_module._qwen36_hybrid_apc_controls_need_prepare(None, None)
        )
        self.assertTrue(
            self.qwen_module._qwen36_hybrid_apc_controls_need_prepare(
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
            )
        )

    def test_hybrid_apc_controls_skip_prepare_for_active_masks(self):
        self.assertFalse(
            self.qwen_module._qwen36_hybrid_apc_controls_need_prepare(
                torch.tensor([1], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
            )
        )
        self.assertFalse(
            self.qwen_module._qwen36_hybrid_apc_controls_need_prepare(
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
            )
        )

    def test_hybrid_apc_controls_materialized_for_zeroed_restore_mask(self):
        self.assertTrue(
            self.qwen_module._qwen36_hybrid_apc_controls_materialized(
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([2048], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
            )
        )
        self.assertFalse(
            self.qwen_module._qwen36_hybrid_apc_controls_materialized(
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
            )
        )

    def test_hybrid_apc_pad_prepare_preserves_full_prefix_tail_contract(self):
        wrapper = _make_wrapper(
            self.qwen_module,
            tag=self.qwen_module.CONTEXT_ENCODING_MODEL_TAG,
        )
        wrapper.neuron_config = SimpleNamespace(
            enable_fused_speculation=False,
            enable_eagle_speculation=False,
        )
        wrapper.is_prefix_caching = True

        empty = torch.empty(0)
        base_args = list(wrapper._base_inputs[0])
        original_tail = [
            empty,  # tile_q_indices
            empty,  # tile_block_tables
            empty,  # tile_masks
            empty,  # inputs_embeds
            empty,  # kv_cache
            empty,  # active_mask
            torch.empty(0, dtype=torch.int32),  # rotary_position_id
            torch.empty(0, dtype=torch.bfloat16),  # vision_embeddings
            torch.empty(0, dtype=torch.int32),  # vision_mask
            torch.zeros((1,), dtype=torch.int32),  # restore slot
            torch.zeros((1,), dtype=torch.int32),  # restore mask
            torch.zeros((1,), dtype=torch.int32),  # restore prefix len
            torch.zeros((1,), dtype=torch.int32),  # commit slot
            torch.zeros((1,), dtype=torch.int32),  # commit mask
        ]
        prepared_tail = [
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.bfloat16),
            torch.empty(0, dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([7], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
        ]

        def _prepare_request(_wrapper, input_dict):
            return input_dict

        with (
            patch.object(
                self.qwen_module,
                "prepare_hybrid_apc_request_for_execution",
                side_effect=_prepare_request,
            ),
            patch.object(
                self.qwen_module,
                "prepare_hybrid_apc_model_inputs",
                return_value=prepared_tail,
            ),
        ):
            padded = wrapper.pad_inputs(*(base_args + original_tail))

        self.assertEqual(len(padded), 29)
        self.assertEqual(int(padded[24].item()), 3)
        self.assertEqual(int(padded[27].item()), 7)
        self.assertEqual(int(padded[28].item()), 1)

    def test_hybrid_apc_pad_prepare_skips_materialized_restore_prefix(self):
        wrapper = _make_wrapper(
            self.qwen_module,
            tag=self.qwen_module.CONTEXT_ENCODING_MODEL_TAG,
        )
        wrapper.neuron_config = SimpleNamespace(
            enable_fused_speculation=False,
            enable_eagle_speculation=False,
        )
        wrapper.is_prefix_caching = True

        empty = torch.empty(0)
        base_args = list(wrapper._base_inputs[0])
        materialized_tail = [
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.bfloat16),
            torch.empty(0, dtype=torch.int32),
            torch.tensor([5], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([2048], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
        ]

        with patch.object(
            self.qwen_module,
            "prepare_hybrid_apc_request_for_execution",
            side_effect=AssertionError("materialized controls must not re-prepare"),
        ):
            padded = wrapper.pad_inputs(*(base_args + materialized_tail))

        self.assertEqual(len(padded), 29)
        self.assertEqual(int(padded[24].item()), 5)
        self.assertEqual(int(padded[25].item()), 0)
        self.assertEqual(int(padded[26].item()), 2048)

    def test_restored_suffix_deltanet_mask_uses_token_padding(self):
        input_ids = torch.tensor([[11, 12, 0, 0]], dtype=torch.int64)
        inputs_embeds = torch.ones((1, 4, 2), dtype=torch.float32)
        attention_mask = torch.ones((1, 4), dtype=torch.int32)

        mask = self.qwen_module._qwen36_deltanet_padding_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            padding_idx=0,
            is_for_context_encoding=True,
            hybrid_restore_mask=torch.tensor([1], dtype=torch.int32),
        )

        self.assertEqual(mask.squeeze(-1).tolist(), [[1.0, 1.0, 0.0, 0.0]])

    def test_non_restored_deltanet_mask_keeps_attention_mask(self):
        input_ids = torch.tensor([[11, 12, 0, 0]], dtype=torch.int64)
        inputs_embeds = torch.ones((1, 4, 2), dtype=torch.float32)
        attention_mask = torch.ones((1, 4), dtype=torch.int32)

        mask = self.qwen_module._qwen36_deltanet_padding_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            padding_idx=0,
            is_for_context_encoding=True,
            hybrid_restore_mask=torch.tensor([0], dtype=torch.int32),
        )

        self.assertEqual(mask.squeeze(-1).tolist(), [[1.0, 1.0, 1.0, 1.0]])

    def test_mixed_restore_deltanet_mask_uses_token_padding_per_restored_row(self):
        input_ids = torch.tensor(
            [[11, 12, 0, 0], [21, 22, 23, 0]], dtype=torch.int64
        )
        inputs_embeds = torch.ones((2, 4, 2), dtype=torch.float32)
        attention_mask = torch.tensor(
            [[1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.int32
        )

        mask = self.qwen_module._qwen36_deltanet_padding_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            padding_idx=0,
            is_for_context_encoding=True,
            hybrid_restore_mask=torch.tensor([1, 0], dtype=torch.int32),
        )

        self.assertEqual(
            mask.squeeze(-1).tolist(),
            [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        )

    def test_deltanet_mask_uses_num_queries_when_attention_mask_is_full_context(self):
        input_ids = torch.tensor(
            [[11, 12, 13, 248044, 248044], [21, 248044, 248044, 248044, 248044]],
            dtype=torch.int64,
        )
        inputs_embeds = torch.ones((2, 5, 2), dtype=torch.float32)
        attention_mask = torch.ones((2, 16), dtype=torch.int32)

        mask = self.qwen_module._qwen36_deltanet_padding_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            padding_idx=None,
            is_for_context_encoding=True,
            hybrid_restore_mask=torch.tensor([0, 0], dtype=torch.int32),
            num_queries=torch.tensor([[3], [1]], dtype=torch.int32),
        )

        self.assertEqual(
            mask.squeeze(-1).tolist(),
            [[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]],
        )

    def test_negative_dummy_seq_ids_mark_inactive_state_rows(self):
        active_rows = self.qwen_module._qwen36_active_state_rows(
            torch.ones((2, 4, 1), dtype=torch.float32),
            torch.tensor([0, -1], dtype=torch.int32),
        )

        self.assertTrue(
            torch.equal(
                active_rows,
                torch.tensor([True, False]),
            )
        )

    def test_inactive_dummy_rows_preserve_previous_state(self):
        previous = torch.tensor(
            [[[1.0, 2.0]], [[3.0, 4.0]]],
            dtype=torch.float32,
        )
        updated = torch.tensor(
            [[[10.0, 20.0]], [[30.0, 40.0]]],
            dtype=torch.float32,
        )

        preserved = self.qwen_module._qwen36_preserve_inactive_state_rows(
            updated,
            previous,
            torch.tensor([True, False]),
        )

        self.assertTrue(
            torch.equal(
                preserved,
                torch.tensor(
                    [[[10.0, 20.0]], [[3.0, 4.0]]],
                    dtype=torch.float32,
                ),
            )
        )

    def test_state_rows_update_by_seq_ids(self):
        previous = torch.tensor(
            [[[1.0]], [[2.0]], [[3.0]]],
            dtype=torch.float32,
        )
        updated_rows = torch.tensor(
            [[[10.0]], [[20.0]]],
            dtype=torch.float32,
        )

        updated = self.qwen_module._qwen36_update_state_rows_by_seq_ids(
            previous,
            updated_rows,
            torch.tensor([2, 0], dtype=torch.int32),
        )

        self.assertTrue(
            torch.equal(
                updated,
                torch.tensor(
                    [[[20.0]], [[2.0]], [[10.0]]],
                    dtype=torch.float32,
                ),
            )
        )

    def test_negative_seq_id_state_row_is_noop(self):
        previous = torch.tensor(
            [[[1.0]], [[2.0]]],
            dtype=torch.float32,
        )
        updated_rows = torch.tensor(
            [[[10.0]], [[20.0]]],
            dtype=torch.float32,
        )

        updated = self.qwen_module._qwen36_update_state_rows_by_seq_ids(
            previous,
            updated_rows,
            torch.tensor([1, -1], dtype=torch.int32),
        )

        self.assertTrue(
            torch.equal(
                updated,
                torch.tensor(
                    [[[1.0]], [[10.0]]],
                    dtype=torch.float32,
                ),
            )
        )

    def test_request_ids_keep_stable_seq_slots_when_rows_reorder(self):
        model = SimpleNamespace(
            context_encoding_model=SimpleNamespace(
                neuron_config=SimpleNamespace(batch_size=2)
            ),
            token_generation_model=SimpleNamespace(
                neuron_config=SimpleNamespace(batch_size=2)
            ),
        )

        first = self.qwen_module._qwen36_stable_seq_ids_for_request_ids(
            model,
            torch.tensor([0], dtype=torch.int32),
            ("req-a",),
        )
        mixed = self.qwen_module._qwen36_stable_seq_ids_for_request_ids(
            model,
            torch.tensor([0], dtype=torch.int32),
            ("req-b", "req-a"),
        )
        next_mixed = self.qwen_module._qwen36_stable_seq_ids_for_request_ids(
            model,
            torch.tensor([0], dtype=torch.int32),
            ("req-c", "req-b"),
        )

        self.assertTrue(torch.equal(first, torch.tensor([0], dtype=torch.int32)))
        self.assertTrue(torch.equal(mixed, torch.tensor([1, 0], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(next_mixed, torch.tensor([0, 1], dtype=torch.int32))
        )

    def test_single_new_request_reuses_first_stale_seq_slot(self):
        model = SimpleNamespace(
            context_encoding_model=SimpleNamespace(
                neuron_config=SimpleNamespace(batch_size=2)
            ),
            token_generation_model=SimpleNamespace(
                neuron_config=SimpleNamespace(batch_size=2)
            ),
        )

        first = self.qwen_module._qwen36_stable_seq_ids_for_request_ids(
            model,
            torch.tensor([0], dtype=torch.int32),
            ("req-a",),
        )
        second = self.qwen_module._qwen36_stable_seq_ids_for_request_ids(
            model,
            torch.tensor([0], dtype=torch.int32),
            ("req-b",),
        )

        self.assertTrue(torch.equal(first, torch.tensor([0], dtype=torch.int32)))
        self.assertTrue(torch.equal(second, torch.tensor([0], dtype=torch.int32)))

    def test_checkpoint_cache_active_rows_follow_seq_slots(self):
        state = torch.tensor(
            [[[1.0]], [[2.0]]],
            dtype=torch.float32,
        )

        active = self.qwen_module.HybridGDNCheckpointCache._active_rows(
            state,
            torch.tensor([1, -1], dtype=torch.int32),
            2,
        )

        self.assertTrue(
            torch.equal(
                active,
                torch.tensor(
                    [[[2.0]], [[1.0]]],
                    dtype=torch.float32,
                ),
            )
        )

    def _make_tiny_deltanet_for_carry_test(self, recurrent_dtype=torch.float32):
        layer = self.qwen_module.NeuronGatedDeltaNet.__new__(
            self.qwen_module.NeuronGatedDeltaNet
        )
        nn.Module.__init__(layer)

        hidden_size = 4
        key_dim = 2
        value_dim = 2
        conv_kernel_size = 3
        conv_dim = key_dim * 2 + value_dim

        layer.hidden_size = hidden_size
        layer.tp_degree = 1
        layer.global_num_v_heads = 1
        layer.global_num_k_heads = 1
        layer.head_k_dim = key_dim
        layer.head_v_dim = value_dim
        layer.num_v_heads = 1
        layer.num_k_heads = 1
        layer.global_key_dim = key_dim
        layer.global_value_dim = value_dim
        layer.key_dim = key_dim
        layer.value_dim = value_dim
        layer.conv_kernel_size = conv_kernel_size
        layer.conv_dim = conv_dim
        layer.layer_idx = 0
        layer.use_hybrid_cache_manager = False
        layer.use_hybrid_apc_manager = True
        layer.use_qwen_hybrid_chunked_prefill = True
        layer.use_qwen_hybrid_chunked_prefill_nki = False
        layer.use_qwen_deltanet_decode_nki = False
        layer.use_cold_zero_conv_fast_path = False
        layer.head_dim = key_dim
        layer.kv_heads_per_rank = 1

        layer.conv1d_weight = nn.Linear(conv_kernel_size, conv_dim, bias=False)
        layer.in_proj_qkv = nn.Linear(hidden_size, conv_dim, bias=False)
        layer.in_proj_z = nn.Linear(hidden_size, value_dim, bias=False)
        layer.in_proj_b = nn.Linear(hidden_size, 1, bias=False)
        layer.in_proj_a = nn.Linear(hidden_size, 1, bias=False)
        layer.dt_bias_weight = nn.Linear(1, 1, bias=False)
        layer.A_log_weight = nn.Linear(1, 1, bias=False)
        layer.norm = nn.Identity()
        layer.out_proj = nn.Linear(value_dim, hidden_size, bias=False)
        layer.recurrent_state_buffer = nn.Parameter(
            torch.zeros((1, 1, key_dim, value_dim), dtype=recurrent_dtype),
            requires_grad=False,
        )
        layer.conv_state_buffer = nn.Parameter(
            torch.zeros((1, conv_dim, conv_kernel_size - 1), dtype=torch.bfloat16),
            requires_grad=False,
        )

        with torch.no_grad():
            for module in (
                layer.conv1d_weight,
                layer.in_proj_qkv,
                layer.in_proj_z,
                layer.in_proj_b,
                layer.in_proj_a,
                layer.out_proj,
            ):
                module.weight.uniform_(-0.04, 0.04)
            layer.dt_bias_weight.weight.fill_(-1.0)
            layer.A_log_weight.weight.fill_(-2.0)

        return layer

    def _assert_hybrid_gdn_checkpoint_carry_matches_full_prefill_on_cpu(
        self, seq_len
    ):
        torch.manual_seed(36 + seq_len)
        chunk = 512
        full_chunks = seq_len // chunk
        suffix = seq_len - full_chunks * chunk
        self.assertGreater(full_chunks, 0)
        self.assertGreater(suffix, 0)
        hidden = torch.randn((1, seq_len, 4), dtype=torch.float32) * 0.05

        for recurrent_dtype in (torch.float32, torch.bfloat16):
            layer = self._make_tiny_deltanet_for_carry_test(recurrent_dtype)
            cache_config = SimpleNamespace(
                layer_types=["linear_attention"],
                max_gdn_checkpoint_slots=3,
                linear_num_value_heads=1,
                linear_num_key_heads=1,
                linear_key_head_dim=2,
                linear_value_head_dim=2,
                linear_conv_kernel_dim=3,
                hybrid_recurrent_cache_dtype=(
                    "bfloat16" if recurrent_dtype is torch.bfloat16 else "float32"
                ),
                hybrid_conv_cache_dtype="bfloat16",
                neuron_config=SimpleNamespace(tp_degree=1),
            )
            cache = self.qwen_module.HybridGDNCheckpointCache(cache_config)
            layers = [SimpleNamespace(linear_attn=layer)]
            seq_ids = torch.tensor([0], dtype=torch.int32)

            def run_cte(tokens, start_pos, past):
                positions = torch.arange(
                    start_pos,
                    start_pos + tokens.shape[1],
                    dtype=torch.int64,
                ).unsqueeze(0)
                mask = torch.ones((1, tokens.shape[1], 1), dtype=torch.float32)
                with patch.dict(
                    os.environ,
                    {"USE_PYTORCH_CHUNK": "1", "USE_NKI_FUSED": "0"},
                    clear=False,
                ):
                    output, _kv, recurrent, conv = layer(
                        tokens,
                        position_ids=positions,
                        past_key_value=past,
                        seq_ids=seq_ids,
                        is_for_context_encoding=True,
                        deltanet_padding_mask=mask,
                    )
                return output, recurrent, conv

            def commit(slot, recurrent, conv):
                recurrent_out, conv_out = cache.commit_from_active_rows(
                    layer_state_pairs=[(0, recurrent, conv)],
                    seq_ids=seq_ids,
                    checkpoint_slot_ids=torch.tensor([slot], dtype=torch.int32),
                    commit_mask=torch.tensor([1], dtype=torch.int32),
                )
                with torch.no_grad():
                    cache.recurrent_slots[0].copy_(recurrent_out)
                    cache.conv_slots[0].copy_(conv_out)

            def restore(slot):
                return cache.restore_to_active_rows(
                    layers=layers,
                    seq_ids=seq_ids,
                    checkpoint_slot_ids=torch.tensor([slot], dtype=torch.int32),
                    restore_mask=torch.tensor([1], dtype=torch.int32),
                )[0]

            zero_past = (
                torch.zeros_like(layer.recurrent_state_buffer),
                torch.zeros_like(layer.conv_state_buffer),
            )
            full_output, full_recurrent, full_conv = run_cte(hidden, 0, zero_past)

            split_outputs = []
            past = zero_past
            for chunk_idx in range(full_chunks):
                start = chunk_idx * chunk
                output, recurrent, conv = run_cte(
                    hidden[:, start : start + chunk], start, past
                )
                split_outputs.append(output)
                commit(chunk_idx, recurrent, conv)
                past = restore(chunk_idx)

            padded_tail = torch.zeros((1, chunk, 4), dtype=hidden.dtype)
            tail_start = full_chunks * chunk
            padded_tail[:, :suffix] = hidden[:, tail_start:]
            positions = torch.cat(
                [
                    torch.arange(tail_start, seq_len, dtype=torch.int64),
                    torch.ones((chunk - suffix,), dtype=torch.int64),
                ]
            ).unsqueeze(0)
            tail_mask = torch.zeros((1, chunk, 1), dtype=torch.float32)
            tail_mask[:, :suffix] = 1
            with patch.dict(
                os.environ,
                {"USE_PYTORCH_CHUNK": "1", "USE_NKI_FUSED": "0"},
                clear=False,
            ):
                out2, _kv, rec2, conv2 = layer(
                    padded_tail,
                    position_ids=positions,
                    past_key_value=past,
                    seq_ids=seq_ids,
                    is_for_context_encoding=True,
                    deltanet_padding_mask=tail_mask,
                )

            split_outputs.append(out2[:, :suffix])
            split_output = torch.cat(split_outputs, dim=1)
            max_diff = (split_output - full_output).abs().max().item()
            rec_diff = (rec2.float() - full_recurrent.float()).abs().max().item()
            conv_diff = (conv2.float() - full_conv.float()).abs().max().item()

            tolerance = 1e-5 if recurrent_dtype is torch.float32 else 2e-3
            msg = (seq_len, recurrent_dtype)
            self.assertLessEqual(max_diff, tolerance, msg)
            self.assertLessEqual(rec_diff, tolerance, msg)
            self.assertLessEqual(conv_diff, tolerance, msg)

    def test_hybrid_gdn_checkpoint_carry_matches_full_prefill_on_cpu(self):
        self._assert_hybrid_gdn_checkpoint_carry_matches_full_prefill_on_cpu(1225)

    def test_hybrid_gdn_checkpoint_carry_matches_full_prefill_on_cpu_at_cliff(self):
        self._assert_hybrid_gdn_checkpoint_carry_matches_full_prefill_on_cpu(526)

    def test_dummy_cte_rows_zero_restore_controls(self):
        restore_slots, restore_mask, restore_prefix = (
            self.qwen_module._qwen36_pad_hybrid_restore_controls_for_dummy_cte_rows(
                torch.tensor([7], dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
                torch.tensor([256], dtype=torch.int32),
                2,
            )
        )

        self.assertTrue(
            torch.equal(restore_slots, torch.tensor([7, 0], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(restore_mask, torch.tensor([1, 0], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(restore_prefix, torch.tensor([256, 0], dtype=torch.int32))
        )

    def test_packed_decode_batch_is_unpacked_for_tkg(self):
        input_ids, attention_mask, position_ids, seq_ids, adapter_ids, slot_mapping = (
            self.qwen_module._qwen36_unpack_packed_decode_batch(
                input_ids=torch.tensor([[271, 198]], dtype=torch.int32),
                attention_mask=torch.tensor([[1, 1]], dtype=torch.int32),
                position_ids=torch.tensor([[272, 272]], dtype=torch.int32),
                seq_ids=torch.tensor([0], dtype=torch.int32),
                adapter_ids=torch.tensor([0], dtype=torch.int32),
                slot_mapping=torch.tensor([1552, 1553], dtype=torch.int32),
                full_context_lens=torch.tensor([273, 273], dtype=torch.int32),
                computed_context_lens=torch.tensor([272, 272], dtype=torch.int32),
            )
        )

        self.assertTrue(
            torch.equal(input_ids, torch.tensor([[271], [198]], dtype=torch.int32))
        )
        self.assertEqual(attention_mask.shape, (2, 272))
        self.assertTrue(
            torch.equal(
                attention_mask[:, :272],
                torch.ones((2, 272), dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                position_ids,
                torch.tensor([[272], [272]], dtype=torch.int32),
            )
        )
        self.assertTrue(torch.equal(seq_ids, torch.tensor([0, 1], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(adapter_ids, torch.tensor([0, 0], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(
                slot_mapping,
                torch.tensor([[1552], [1553]], dtype=torch.int32),
            )
        )

    def test_request_scoped_vllm_metadata_is_added_for_hybrid_apc(self):
        request_dict = {}

        self.qwen_module._qwen36_add_vllm_hybrid_apc_metadata(
            request_dict,
            request_ids=("req-a", "req-b"),
            metadata_by_request_id={
                "req-a": {
                    "cumulative_hashes_by_prefix_len": {256: b"a"},
                    "attention_block_refs_by_prefix_len": {256: (1,)},
                    "request_prefix_len": 256,
                    "vllm_attention_hit_len": 0,
                    "active_suffix_len": 256,
                },
                "req-b": {
                    "cumulative_hashes_by_prefix_len": {256: b"b"},
                    "attention_block_refs_by_prefix_len": {256: (2,)},
                    "request_prefix_len": 272,
                    "vllm_attention_hit_len": 256,
                    "active_suffix_len": 16,
                },
            },
        )

        self.assertEqual(
            request_dict["cumulative_hashes_by_prefix_len"],
            ({256: b"a"}, {256: b"b"}),
        )
        self.assertEqual(
            request_dict["attention_block_refs_by_prefix_len"],
            ({256: (1,)}, {256: (2,)}),
        )
        self.assertEqual(request_dict["request_prefix_len"], (256, 272))
        self.assertEqual(request_dict["vllm_attention_hit_len"], (0, 256))
        self.assertEqual(request_dict["active_suffix_len"], (256, 16))

    def test_request_scoped_vllm_metadata_tensorizes_full_input_ids(self):
        request_dict = {
            "input_ids": torch.empty((1, 0), dtype=torch.int32),
        }

        self.qwen_module._qwen36_add_vllm_hybrid_apc_metadata(
            request_dict,
            request_ids=("req-2049",),
            metadata_by_request_id={
                "req-2049": {
                    "request_prefix_len": 2049,
                    "full_input_ids": tuple(range(2049)),
                    "vllm_attention_hit_len": 2048,
                    "active_suffix_len": 1,
                },
            },
        )

        self.assertIsInstance(request_dict["full_input_ids"], torch.Tensor)
        self.assertEqual(request_dict["full_input_ids"].dtype, torch.int32)
        self.assertEqual(tuple(request_dict["full_input_ids"].shape), (1, 2049))
        self.assertEqual(int(request_dict["full_input_ids"][0, -1].item()), 2048)

    def test_vllm_metadata_request_ids_prefer_scheduler_new_request_ids(self):
        selected = self.qwen_module._qwen36_select_vllm_hybrid_apc_request_ids(
            {
                "new-a": {"vllm_attention_hit_len": 256},
                "new-b": {"vllm_attention_hit_len": 256},
            },
            ("new-a", "new-b"),
            ("model-a", "model-b"),
        )

        self.assertEqual(selected, ("new-a", "new-b"))

    def test_vllm_metadata_request_ids_use_model_order_for_packed_chunked_batch(self):
        selected = (
            self.qwen_module._qwen36_select_vllm_hybrid_apc_request_ids_for_input(
                {
                    "new-a": {"vllm_attention_hit_len": 0},
                    "cached-a": {"vllm_attention_hit_len": 271},
                },
                all_request_ids=("new-a", "cached-a"),
                new_request_ids=("new-a",),
                full_context_lens=torch.tensor([271, 272], dtype=torch.int32),
                computed_context_lens=torch.tensor([0, 271], dtype=torch.int32),
                prefill_completion_state=torch.tensor([True, True]),
            )
        )

        self.assertEqual(selected, ("new-a", "cached-a"))

    def test_vllm_metadata_request_ids_keep_model_order_when_new_ids_are_subset(self):
        selected = (
            self.qwen_module._qwen36_select_vllm_hybrid_apc_request_ids_for_input(
                {
                    "new-a": {"vllm_attention_hit_len": 0},
                },
                all_request_ids=("cached-a", "new-a"),
                new_request_ids=("new-a",),
                full_context_lens=torch.tensor([1, 272], dtype=torch.int32),
                computed_context_lens=torch.tensor([0, 0], dtype=torch.int32),
                prefill_completion_state=torch.tensor([True, True]),
            )
        )

        self.assertEqual(selected, ("cached-a", "new-a"))

    def test_flattened_slot_mapping_is_normalized_before_batch_chunking(self):
        flattened = torch.arange(256, 719, dtype=torch.int32)

        normalized = self.qwen_module._normalize_qwen36_slot_mapping(
            flattened,
            batch_size=1,
            active_tokens=463,
        )

        self.assertEqual(normalized.shape, (1, 463))
        self.assertTrue(torch.equal(normalized[0], flattened))

    def test_flattened_decode_slot_mapping_is_normalized_by_batch(self):
        flattened = torch.tensor([1488, 1489], dtype=torch.int32)

        normalized = self.qwen_module._normalize_qwen36_slot_mapping(
            flattened,
            batch_size=2,
            active_tokens=1,
        )

        self.assertTrue(
            torch.equal(
                normalized,
                torch.tensor([[1488], [1489]], dtype=torch.int32),
            )
        )

    def test_stage_builders_keep_cte_and_tkg_contracts_explicit(self):
        wrapper = _make_wrapper(
            self.qwen_module,
            tag=self.qwen_module.TOKEN_GENERATION_MODEL_TAG,
        )
        prefix_args = wrapper._base_inputs[0]
        mrope = torch.zeros((0,), dtype=torch.int32)
        vision_embeddings = torch.zeros((0,), dtype=torch.bfloat16)
        vision_mask = torch.zeros((0,), dtype=torch.int32)

        with patch.dict(os.environ, {"QWEN36_TKG_LEGACY_ARGS": "1"}, clear=True):
            cte_args = self.qwen_module.build_cte_args(
                wrapper.config,
                prefix_args,
                mrope,
                vision_embeddings,
                vision_mask,
            )
            tkg_args = self.qwen_module.build_tkg_args(
                wrapper.config,
                prefix_args,
                mrope,
                vision_embeddings,
                vision_mask,
            )

        self.assertEqual(len(cte_args), 24)
        self.assertEqual(len(tkg_args), 24)
        self.assertEqual(cte_args[13].shape, (1, 1))
        self.assertEqual(tkg_args[13].shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
