# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import sys
import types
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch


_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PATCH_PATH = os.path.join(
    _CONTRIB_ROOT,
    "vllm",
    "qwen36_hybrid_apc_scheduler_patch.py",
)
_SCHEDULER_MODULE = "vllm.v1.core.sched.scheduler"
_VLLM_NEURON_RUNNER_MODULE = "vllm_neuron.worker.neuronx_distributed_model_runner"


@dataclass(frozen=True)
class FullAttentionSpec:
    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: str
    sliding_window: int | None = None


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("qwen36_scheduler_patch", _PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scheduler(
    *,
    use_hybrid_apc=True,
    disable_unbacked_prefix_reads=False,
    enable_backed_prefix_reads=False,
    use_qwen_hybrid_chunked_prefill=False,
    block_size=2,
    model_revision="rev-a",
    additional_config=None,
    max_num_seqs=1,
):
    hf_config_kwargs = dict(
        use_hybrid_apc_manager=use_hybrid_apc,
        hybrid_apc_disable_unbacked_prefix_reads=disable_unbacked_prefix_reads,
        hybrid_apc_enable_backed_prefix_reads=enable_backed_prefix_reads,
        use_qwen_hybrid_chunked_prefill=use_qwen_hybrid_chunked_prefill,
        hybrid_apc_layout_version=1,
        hybrid_recurrent_cache_dtype="float32",
        hybrid_conv_cache_dtype="bfloat16",
        tp_rank=0,
    )
    if model_revision is not None:
        hf_config_kwargs["hybrid_apc_model_revision"] = model_revision
    hf_config = types.SimpleNamespace(**hf_config_kwargs)
    model_config = types.SimpleNamespace(hf_config=hf_config)
    vllm_config = types.SimpleNamespace(
        model_config=model_config,
        additional_config=additional_config or {},
    )
    cache_config = types.SimpleNamespace(block_size=block_size)
    scheduler_config = types.SimpleNamespace(max_num_seqs=max_num_seqs)
    return types.SimpleNamespace(
        vllm_config=vllm_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
    )


class TestQwen36HybridAPCSchedulerPatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patch = _load_patch_module()

    def tearDown(self):
        self.patch.clear_hybrid_apc_gdn_checkpoint_registry()
        sys.modules.pop(_SCHEDULER_MODULE, None)
        sys.modules.pop(_VLLM_NEURON_RUNNER_MODULE, None)
        sys.meta_path = [
            finder
            for finder in sys.meta_path
            if not getattr(finder, "_qwen36_hybrid_apc_import_hook", False)
        ]

    def test_config_flag_disables_prefix_reads_for_hybrid_apc(self):
        scheduler = _scheduler(disable_unbacked_prefix_reads=True)

        self.assertTrue(self.patch.should_disable_unbacked_prefix_reads(scheduler))

    def test_env_flag_disables_prefix_reads_for_hybrid_apc(self):
        scheduler = _scheduler()

        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            self.assertTrue(self.patch.should_disable_unbacked_prefix_reads(scheduler))

    def test_env_flag_wins_when_artifact_config_is_stale(self):
        scheduler = _scheduler(use_hybrid_apc=False)

        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            self.assertTrue(self.patch.should_disable_unbacked_prefix_reads(scheduler))

    def test_reject_unbacked_mode_disables_unbacked_prefix_reads(self):
        scheduler = _scheduler(
            additional_config={"hybrid_apc_reject_unbacked_attention_hits": True},
        )

        self.assertTrue(self.patch.should_disable_unbacked_prefix_reads(scheduler))

    def test_require_metadata_mode_disables_unbacked_prefix_reads(self):
        scheduler = _scheduler(
            additional_config={"hybrid_apc_require_vllm_metadata": True},
        )

        self.assertTrue(self.patch.should_disable_unbacked_prefix_reads(scheduler))

    def test_non_hybrid_apc_model_is_not_changed(self):
        scheduler = _scheduler(
            use_hybrid_apc=False,
            disable_unbacked_prefix_reads=True,
        )

        self.assertFalse(self.patch.should_disable_unbacked_prefix_reads(scheduler))

    def test_patch_marks_request_skip_reading_prefix_cache(self):
        calls = []

        class FakeScheduler:
            def __init__(self):
                self.vllm_config = _scheduler(
                    disable_unbacked_prefix_reads=True
                ).vllm_config

            def add_request(self, request):
                calls.append(request.skip_reading_prefix_cache)

        installed = self.patch.patch_scheduler_class(FakeScheduler)
        request = types.SimpleNamespace(skip_reading_prefix_cache=False)

        FakeScheduler().add_request(request)

        self.assertTrue(installed)
        self.assertEqual(calls, [True])
        self.assertTrue(request.skip_reading_prefix_cache)

    def test_registered_gdn_checkpoint_keeps_prefix_read_disabled_without_cte_support(self):
        scheduler = _scheduler(block_size=2)
        token_ids = [10, 11, 12, 13, 14]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=2,
            max_prefix_len=4,
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[2],
                prefix_len=2,
                block_size=2,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[4],
                prefix_len=4,
                block_size=2,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        request = types.SimpleNamespace(
            prompt_token_ids=token_ids,
            num_tokens=len(token_ids),
            cache_salt=None,
        )

        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            self.assertEqual(
                self.patch.backed_gdn_prefix_hit_len(scheduler, request),
                4,
            )
            self.assertTrue(
                self.patch.should_disable_unbacked_prefix_reads(scheduler, request)
            )

    def test_registered_gdn_checkpoint_allows_prefix_read_when_cte_supports_it(self):
        scheduler = _scheduler(
            block_size=2,
            enable_backed_prefix_reads=True,
            use_qwen_hybrid_chunked_prefill=True,
        )
        token_ids = [10, 11, 12, 13, 14]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=2,
            max_prefix_len=4,
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[2],
                prefix_len=2,
                block_size=2,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[4],
                prefix_len=4,
                block_size=2,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        request = types.SimpleNamespace(
            prompt_token_ids=token_ids,
            num_tokens=len(token_ids),
            cache_salt=None,
        )

        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            self.assertEqual(
                self.patch.backed_gdn_prefix_hit_len(scheduler, request),
                4,
            )
            self.assertFalse(
                self.patch.should_disable_unbacked_prefix_reads(scheduler, request)
            )

    def test_largest_backed_prefix_read_does_not_require_lower_checkpoint(self):
        scheduler = _scheduler(
            block_size=2,
            enable_backed_prefix_reads=True,
            use_qwen_hybrid_chunked_prefill=True,
        )
        token_ids = [10, 11, 12, 13, 14]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=2,
            max_prefix_len=4,
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[4],
                prefix_len=4,
                block_size=2,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        request = types.SimpleNamespace(
            request_id="req-largest-backed",
            prompt_token_ids=token_ids,
            num_tokens=len(token_ids),
            cache_salt=None,
        )

        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            self.assertEqual(
                self.patch.backed_gdn_prefix_hit_len(scheduler, request),
                4,
            )
            self.assertFalse(
                self.patch.should_disable_unbacked_prefix_reads(scheduler, request)
            )

        authorized = self.patch.pop_hybrid_apc_authorized_prefix_key(
            prefix_len=4,
            request_id="req-largest-backed",
            cache_salt=None,
            model_revision="rev-a",
            layout_version=1,
            tp_rank=0,
            recurrent_dtype="float32",
            conv_dtype="bfloat16",
        )
        self.assertIsNotNone(authorized)
        self.assertEqual(authorized.cumulative_prefix_hash, hashes[4])
        self.assertIsNone(
            self.patch.pop_hybrid_apc_authorized_prefix_key(
                prefix_len=2,
                request_id="req-largest-backed",
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )

    def test_partial_gdn_coverage_caps_prefix_read_to_backed_checkpoint(self):
        scheduler = _scheduler(
            block_size=2,
            enable_backed_prefix_reads=True,
            use_qwen_hybrid_chunked_prefill=True,
        )
        token_ids = [10, 11, 12, 13, 14, 15, 16]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=2,
            max_prefix_len=6,
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[4],
                prefix_len=4,
                block_size=2,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        request = types.SimpleNamespace(
            request_id="req-partial",
            prompt_token_ids=token_ids,
            num_tokens=len(token_ids),
            cache_salt=None,
        )

        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            self.assertEqual(
                self.patch.backed_gdn_prefix_hit_len(scheduler, request),
                4,
            )
            self.assertFalse(
                self.patch.should_disable_unbacked_prefix_reads(scheduler, request)
            )

        self.assertEqual(
            getattr(request, self.patch._MAX_PREFIX_CACHE_HIT_LEN_ATTR),
            4,
        )
        self.assertEqual(
            getattr(request, self.patch._MAX_PREFIX_CACHE_BLOCKS_ATTR),
            2,
        )
        self.assertIsNotNone(
            self.patch.pop_hybrid_apc_authorized_prefix_key(
                prefix_len=4,
                request_id="req-partial",
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )

    def test_max_backed_prefix_cap_selects_largest_backed_prefix_under_cap(self):
        scheduler = _scheduler(
            block_size=2,
            enable_backed_prefix_reads=True,
            use_qwen_hybrid_chunked_prefill=True,
            additional_config={"hybrid_apc_max_backed_prefix_read_len": 2},
        )
        token_ids = [10, 11, 12, 13, 14]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=2,
            max_prefix_len=4,
        )
        for prefix_len in (2, 4):
            self.patch.register_hybrid_apc_gdn_checkpoint(
                self.patch.HybridGDNPrefixKey(
                    cumulative_prefix_hash=hashes[prefix_len],
                    prefix_len=prefix_len,
                    block_size=2,
                    cache_salt=None,
                    model_revision="rev-a",
                    layout_version=1,
                    tp_rank=0,
                    recurrent_dtype="float32",
                    conv_dtype="bfloat16",
                )
            )
        request = types.SimpleNamespace(
            request_id="req-capped",
            prompt_token_ids=token_ids,
            num_tokens=len(token_ids),
            cache_salt=None,
        )

        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            self.assertFalse(
                self.patch.should_disable_unbacked_prefix_reads(scheduler, request)
            )

        self.assertEqual(
            getattr(request, self.patch._MAX_PREFIX_CACHE_HIT_LEN_ATTR),
            2,
        )
        self.assertEqual(
            getattr(request, self.patch._MAX_PREFIX_CACHE_BLOCKS_ATTR),
            1,
        )
        self.assertIsNotNone(
            self.patch.pop_hybrid_apc_authorized_prefix_key(
                prefix_len=2,
                request_id="req-capped",
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )

    def test_kv_cache_manager_caps_prefix_hash_lookup_to_backed_len(self):
        seen_hashes = []

        class FakeKVCacheManager:
            empty_kv_cache_blocks = "empty"

            def get_computed_blocks(self, request):
                seen_hashes.append(tuple(request.block_hashes))
                return "blocks", len(request.block_hashes) * 2

        fake_module = types.SimpleNamespace(KVCacheManager=FakeKVCacheManager)
        self.assertTrue(self.patch._patch_kv_cache_manager_module(fake_module))

        request = types.SimpleNamespace(block_hashes=[b"a", b"b", b"c", b"d"])
        setattr(request, self.patch._MAX_PREFIX_CACHE_BLOCKS_ATTR, 2)

        result = FakeKVCacheManager().get_computed_blocks(request)

        self.assertEqual(result, ("blocks", 4))
        self.assertEqual(seen_hashes, [(b"a", b"b")])
        self.assertEqual(request.block_hashes, [b"a", b"b", b"c", b"d"])

    def test_additional_config_allows_prefix_read_when_hf_config_is_stale(self):
        scheduler = _scheduler(
            block_size=2,
            enable_backed_prefix_reads=False,
            use_qwen_hybrid_chunked_prefill=False,
            additional_config={
                "use_hybrid_apc_manager": True,
                "hybrid_apc_disable_unbacked_prefix_reads": True,
                "hybrid_apc_enable_backed_prefix_reads": True,
                "use_qwen_hybrid_chunked_prefill": True,
            },
        )
        token_ids = [10, 11, 12, 13, 14]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=2,
            max_prefix_len=4,
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[2],
                prefix_len=2,
                block_size=2,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[4],
                prefix_len=4,
                block_size=2,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        request = types.SimpleNamespace(
            prompt_token_ids=token_ids,
            num_tokens=len(token_ids),
            cache_salt=None,
        )

        self.assertFalse(
            self.patch.should_disable_unbacked_prefix_reads(scheduler, request)
        )
        authorized = self.patch.pop_hybrid_apc_authorized_prefix_key(
            prefix_len=4,
            cache_salt=None,
            model_revision="rev-a",
            layout_version=1,
            tp_rank=0,
            recurrent_dtype="float32",
            conv_dtype="bfloat16",
        )
        self.assertIsNotNone(authorized)
        self.assertEqual(authorized.cumulative_prefix_hash, hashes[4])

    def test_authorized_prefix_read_can_be_request_scoped(self):
        key = self.patch.HybridGDNPrefixKey(
            cumulative_prefix_hash="hash-a",
            prefix_len=4,
            block_size=2,
            cache_salt=None,
            model_revision="rev-a",
            layout_version=1,
            tp_rank=0,
            recurrent_dtype="float32",
            conv_dtype="bfloat16",
        )

        self.patch.authorize_hybrid_apc_prefix_read(key, request_id="req-a")

        self.assertIsNone(
            self.patch.pop_hybrid_apc_authorized_prefix_key(
                prefix_len=4,
                request_id="req-b",
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        self.assertEqual(
            self.patch.pop_hybrid_apc_authorized_prefix_key(
                prefix_len=4,
                request_id="req-a",
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            ),
            key,
        )

    def test_scheduler_authorizes_backed_prefix_read_by_request_id(self):
        scheduler = _scheduler(
            block_size=2,
            enable_backed_prefix_reads=True,
            use_qwen_hybrid_chunked_prefill=True,
        )
        token_ids = [10, 11, 12, 13, 14]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=2,
            max_prefix_len=4,
        )
        key = self.patch.HybridGDNPrefixKey(
            cumulative_prefix_hash=hashes[4],
            prefix_len=4,
            block_size=2,
            cache_salt=None,
            model_revision="rev-a",
            layout_version=1,
            tp_rank=0,
            recurrent_dtype="float32",
            conv_dtype="bfloat16",
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[2],
                prefix_len=2,
                block_size=2,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(key)
        request = types.SimpleNamespace(
            request_id="req-a",
            prompt_token_ids=token_ids,
            num_tokens=len(token_ids),
            cache_salt=None,
        )

        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            self.assertFalse(
                self.patch.should_disable_unbacked_prefix_reads(scheduler, request)
            )

        self.assertIsNone(
            self.patch.pop_hybrid_apc_authorized_prefix_key(
                prefix_len=4,
                request_id="req-b",
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        self.assertEqual(
            self.patch.pop_hybrid_apc_authorized_prefix_key(
                prefix_len=4,
                request_id="req-a",
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            ),
            key,
        )

    def test_backed_prefix_hit_uses_vllm_block_hashes_when_available(self):
        scheduler = _scheduler(block_size=2)
        key = self.patch.HybridGDNPrefixKey(
            cumulative_prefix_hash=b"vllm-hash-4",
            prefix_len=4,
            block_size=2,
            cache_salt=None,
            model_revision="rev-a",
            layout_version=1,
            tp_rank=0,
            recurrent_dtype="float32",
            conv_dtype="bfloat16",
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(key)
        request = types.SimpleNamespace(
            prompt_token_ids=[10, 11, 12, 13, 14],
            block_hashes=[b"vllm-hash-2", b"vllm-hash-4"],
            num_tokens=5,
            cache_salt=None,
        )

        self.assertEqual(self.patch.backed_gdn_prefix_hit(scheduler, request), key)

    def test_scheduler_output_carries_vllm_hashes_and_block_refs(self):
        class FakeScheduler:
            def __init__(self):
                base = _scheduler(block_size=2)
                self.vllm_config = base.vllm_config
                self.cache_config = base.cache_config
                self.scheduler_config = base.scheduler_config
                self.requests = {
                    "req-a": types.SimpleNamespace(
                        prompt_token_ids=[10, 11, 12, 13],
                        block_hashes=[b"hash-2", b"hash-4"],
                        num_tokens=4,
                        cache_salt=None,
                    )
                }

            def add_request(self, request):
                del request

            def schedule(self):
                return types.SimpleNamespace(
                    scheduled_new_reqs=[
                        types.SimpleNamespace(
                            req_id="req-a",
                            block_ids=([11, 12],),
                            num_computed_tokens=0,
                        )
                    ],
                    scheduled_cached_reqs=types.SimpleNamespace(
                        req_ids=[],
                        new_block_ids=[],
                        num_computed_tokens=[],
                    ),
                    num_scheduled_tokens={"req-a": 4},
                )

        self.patch.patch_scheduler_class(FakeScheduler)
        scheduler_output = FakeScheduler().schedule()
        metadata = getattr(
            scheduler_output,
            "_qwen36_hybrid_apc_metadata_by_request_id",
        )

        self.assertEqual(
            metadata["req-a"]["cumulative_hashes_by_prefix_len"],
            {2: b"hash-2", 4: b"hash-4"},
        )
        self.assertEqual(
            metadata["req-a"]["attention_block_refs_by_prefix_len"],
            {2: (11,), 4: (11, 12)},
        )
        self.assertEqual(metadata["req-a"]["request_prefix_len"], 4)
        self.assertEqual(metadata["req-a"]["vllm_attention_hit_len"], 0)
        self.assertEqual(metadata["req-a"]["active_suffix_len"], 4)

    def test_scheduler_output_caps_cached_request_prefix_to_current_chunk(self):
        class FakeScheduler:
            def __init__(self):
                base = _scheduler(block_size=256)
                self.vllm_config = base.vllm_config
                self.cache_config = base.cache_config
                self.scheduler_config = base.scheduler_config
                self.requests = {
                    "req-a": types.SimpleNamespace(
                        request_id="req-a",
                        prompt_token_ids=list(range(8193)),
                        all_token_ids=list(range(8193)),
                        block_hashes=[f"hash-{idx}".encode() for idx in range(32)],
                        num_tokens=8193,
                        cache_salt=None,
                    )
                }

            def add_request(self, request):
                del request

            def schedule(self):
                return types.SimpleNamespace(
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=types.SimpleNamespace(
                        req_ids=["req-a"],
                        new_block_ids=[(list(range(13, 25)),)],
                        num_computed_tokens=[3072],
                    ),
                    num_scheduled_tokens={"req-a": 3072},
                )

        self.patch.patch_scheduler_class(FakeScheduler)
        scheduler_output = FakeScheduler().schedule()
        metadata = getattr(
            scheduler_output,
            "_qwen36_hybrid_apc_metadata_by_request_id",
        )["req-a"]

        self.assertEqual(metadata["request_prefix_len"], 6144)
        self.assertEqual(metadata["vllm_attention_hit_len"], 3072)
        self.assertEqual(metadata["active_suffix_len"], 3072)
        self.assertEqual(len(metadata["full_input_ids"]), 6144)
        self.assertIn(6144, metadata["cumulative_hashes_by_prefix_len"])
        self.assertNotIn(8192, metadata["cumulative_hashes_by_prefix_len"])

    def test_scheduler_output_excludes_generated_tokens_from_prompt_metadata(self):
        class FakeScheduler:
            def __init__(self):
                base = _scheduler(block_size=256)
                self.vllm_config = base.vllm_config
                self.cache_config = base.cache_config
                self.scheduler_config = base.scheduler_config
                self.requests = {
                    "req-a": types.SimpleNamespace(
                        request_id="req-a",
                        prompt_token_ids=list(range(8192)),
                        all_token_ids=list(range(8194)),
                        block_hashes=[f"hash-{idx}".encode() for idx in range(32)],
                        num_prompt_tokens=8192,
                        num_tokens=8194,
                        cache_salt=None,
                    )
                }

            def add_request(self, request):
                del request

            def schedule(self):
                return types.SimpleNamespace(
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=types.SimpleNamespace(
                        req_ids=["req-a"],
                        new_block_ids=[(list(range(25, 33)),)],
                        num_computed_tokens=[6144],
                    ),
                    num_scheduled_tokens={"req-a": 2050},
                )

        self.patch.patch_scheduler_class(FakeScheduler)
        scheduler_output = FakeScheduler().schedule()
        metadata = getattr(
            scheduler_output,
            "_qwen36_hybrid_apc_metadata_by_request_id",
        )["req-a"]

        self.assertEqual(metadata["request_prefix_len"], 8192)
        self.assertEqual(metadata["vllm_attention_hit_len"], 6144)
        self.assertEqual(metadata["active_suffix_len"], 2050)
        self.assertEqual(len(metadata["full_input_ids"]), 8192)
        self.assertEqual(metadata["full_input_ids"][-1], 8191)
        self.assertIn(8192, metadata["cumulative_hashes_by_prefix_len"])
        self.assertNotIn(8448, metadata["cumulative_hashes_by_prefix_len"])

    def test_scheduler_output_authorizes_backed_cached_continuation(self):
        class FakeScheduler:
            def __init__(self):
                base = _scheduler(
                    block_size=2,
                    enable_backed_prefix_reads=True,
                    use_qwen_hybrid_chunked_prefill=True,
                )
                self.vllm_config = base.vllm_config
                self.cache_config = base.cache_config
                self.scheduler_config = base.scheduler_config
                self.requests = {
                    "req-a": types.SimpleNamespace(
                        request_id="req-a",
                        prompt_token_ids=[10, 11, 12, 13],
                        block_hashes=[b"hash-2", b"hash-4"],
                        num_tokens=4,
                        cache_salt=None,
                    )
                }

            def add_request(self, request):
                del request

            def schedule(self):
                return types.SimpleNamespace(
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=types.SimpleNamespace(
                        req_ids=["req-a"],
                        new_block_ids=[([12],)],
                        num_computed_tokens=[2],
                        num_output_tokens=[0],
                    ),
                )

        key = self.patch.HybridGDNPrefixKey(
            cumulative_prefix_hash=b"hash-2",
            prefix_len=2,
            block_size=2,
            cache_salt=None,
            model_revision="rev-a",
            layout_version=1,
            tp_rank=0,
            recurrent_dtype="float32",
            conv_dtype="bfloat16",
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(key)
        self.patch.patch_scheduler_class(FakeScheduler)

        FakeScheduler().schedule()

        self.assertEqual(
            self.patch.pop_hybrid_apc_authorized_prefix_key(
                prefix_len=2,
                request_id="req-a",
                model_revision="rev-a",
            ),
            key,
        )

    def test_scheduler_preserves_backed_and_cold_prefix_read_decisions_in_mixed_batch(self):
        class FakeScheduler:
            def __init__(self):
                base = _scheduler(
                    block_size=4,
                    disable_unbacked_prefix_reads=True,
                    enable_backed_prefix_reads=True,
                    use_qwen_hybrid_chunked_prefill=True,
                    max_num_seqs=2,
                )
                self.vllm_config = base.vllm_config
                self.cache_config = base.cache_config
                self.scheduler_config = base.scheduler_config
                self.warm = types.SimpleNamespace(
                    request_id="warm",
                    prompt_token_ids=[10, 11, 12, 13, 14, 15],
                    num_tokens=6,
                    cache_salt=None,
                    skip_reading_prefix_cache=False,
                )
                self.cold = types.SimpleNamespace(
                    request_id="cold",
                    prompt_token_ids=[20, 21, 22, 23, 24, 25],
                    num_tokens=6,
                    cache_salt=None,
                    skip_reading_prefix_cache=False,
                )
                self.waiting = [self.warm, self.cold]
                self.requests = {"warm": self.warm, "cold": self.cold}
                self.schedule_seen_skip_flags = None

            def add_request(self, request):
                del request

            def schedule(self):
                self.schedule_seen_skip_flags = [
                    (request.request_id, request.skip_reading_prefix_cache)
                    for request in self.waiting
                ]
                computed_tokens = [
                    0 if request.skip_reading_prefix_cache else 4
                    for request in self.waiting
                ]
                self.waiting = []
                return types.SimpleNamespace(
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=types.SimpleNamespace(
                        req_ids=["warm", "cold"],
                        new_block_ids=[([1, 2],), ([3, 4],)],
                        num_computed_tokens=computed_tokens,
                        num_output_tokens=[0, 0],
                    ),
                    num_scheduled_tokens={
                        "warm": 6 - computed_tokens[0],
                        "cold": 6 - computed_tokens[1],
                    },
                    total_num_scheduled_tokens=12 - sum(computed_tokens),
                )

        warm_token_ids = [10, 11, 12, 13, 14, 15]
        hashes = self.patch._local_cumulative_prefix_hashes(
            warm_token_ids,
            block_size=4,
            max_prefix_len=4,
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[4],
                prefix_len=4,
                block_size=4,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )

        self.patch.patch_scheduler_class(FakeScheduler)
        scheduler = FakeScheduler()
        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            self.assertFalse(
                self.patch.should_disable_unbacked_prefix_reads(
                    scheduler,
                    scheduler.warm,
                )
            )
            if self.patch.should_disable_unbacked_prefix_reads(
                scheduler,
                scheduler.cold,
            ):
                scheduler.cold.skip_reading_prefix_cache = True
            scheduler_output = scheduler.schedule()

        metadata = getattr(
            scheduler_output,
            "_qwen36_hybrid_apc_metadata_by_request_id",
        )

        self.assertEqual(
            scheduler.schedule_seen_skip_flags,
            [("warm", False), ("cold", True)],
        )
        self.assertEqual(
            scheduler_output.scheduled_cached_reqs.num_computed_tokens,
            [4, 0],
        )
        self.assertEqual(scheduler_output.num_scheduled_tokens["warm"], 2)
        self.assertEqual(scheduler_output.num_scheduled_tokens["cold"], 6)
        self.assertEqual(metadata["warm"]["vllm_attention_hit_len"], 4)
        self.assertEqual(metadata["cold"]["vllm_attention_hit_len"], 0)
        self.assertIsNotNone(
            self.patch.pop_hybrid_apc_authorized_prefix_key(
                prefix_len=4,
                request_id="warm",
                model_revision="rev-a",
            )
        )

    def test_scheduler_preserves_all_backed_context_batch(self):
        class FakeScheduler:
            def __init__(self):
                base = _scheduler(
                    block_size=4,
                    disable_unbacked_prefix_reads=True,
                    enable_backed_prefix_reads=True,
                    use_qwen_hybrid_chunked_prefill=True,
                    max_num_seqs=2,
                )
                self.vllm_config = base.vllm_config
                self.cache_config = base.cache_config
                self.scheduler_config = base.scheduler_config
                self.req_a = types.SimpleNamespace(
                    request_id="req-a",
                    prompt_token_ids=[10, 11, 12, 13, 14, 15],
                    num_tokens=6,
                    cache_salt=None,
                    skip_reading_prefix_cache=False,
                )
                self.req_b = types.SimpleNamespace(
                    request_id="req-b",
                    prompt_token_ids=[30, 31, 32, 33, 34, 35],
                    num_tokens=6,
                    cache_salt=None,
                    skip_reading_prefix_cache=False,
                )
                self.waiting = [self.req_a, self.req_b]
                self.requests = {"req-a": self.req_a, "req-b": self.req_b}
                self.schedule_seen_skip_flags = None

            def add_request(self, request):
                del request

            def schedule(self):
                self.schedule_seen_skip_flags = [
                    (request.request_id, request.skip_reading_prefix_cache)
                    for request in self.waiting
                ]
                self.waiting = []
                return types.SimpleNamespace(
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=types.SimpleNamespace(
                        req_ids=["req-a", "req-b"],
                        new_block_ids=[([1, 2],), ([3, 4],)],
                        num_computed_tokens=[4, 4],
                        num_output_tokens=[0, 0],
                    ),
                    num_scheduled_tokens={"req-a": 2, "req-b": 2},
                    total_num_scheduled_tokens=4,
                )

        for token_ids in ([10, 11, 12, 13, 14, 15], [30, 31, 32, 33, 34, 35]):
            hashes = self.patch._local_cumulative_prefix_hashes(
                token_ids,
                block_size=4,
                max_prefix_len=4,
            )
            self.patch.register_hybrid_apc_gdn_checkpoint(
                self.patch.HybridGDNPrefixKey(
                    cumulative_prefix_hash=hashes[4],
                    prefix_len=4,
                    block_size=4,
                    cache_salt=None,
                    model_revision="rev-a",
                    layout_version=1,
                    tp_rank=0,
                    recurrent_dtype="float32",
                    conv_dtype="bfloat16",
                )
            )

        self.patch.patch_scheduler_class(FakeScheduler)
        scheduler = FakeScheduler()
        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            self.assertFalse(
                self.patch.should_disable_unbacked_prefix_reads(
                    scheduler,
                    scheduler.req_a,
                )
            )
            self.assertFalse(
                self.patch.should_disable_unbacked_prefix_reads(
                    scheduler,
                    scheduler.req_b,
                )
            )
            scheduler_output = scheduler.schedule()

        metadata = getattr(
            scheduler_output,
            "_qwen36_hybrid_apc_metadata_by_request_id",
        )

        self.assertEqual(
            scheduler.schedule_seen_skip_flags,
            [("req-a", False), ("req-b", False)],
        )
        self.assertEqual(
            scheduler_output.scheduled_cached_reqs.num_computed_tokens,
            [4, 4],
        )
        self.assertEqual(metadata["req-a"]["vllm_attention_hit_len"], 4)
        self.assertEqual(metadata["req-b"]["vllm_attention_hit_len"], 4)

    def test_scheduler_defers_waiting_prefills_while_decode_running(self):
        class FakeScheduler:
            def __init__(self):
                base = _scheduler(
                    use_hybrid_apc=True,
                    use_qwen_hybrid_chunked_prefill=True,
                    max_num_seqs=2,
                )
                self.vllm_config = base.vllm_config
                self.cache_config = base.cache_config
                self.scheduler_config = base.scheduler_config
                self.running = [types.SimpleNamespace(request_id="decode")]
                self.waiting = [types.SimpleNamespace(request_id="prefill")]
                self.waiting_seen_by_schedule = None

            def add_request(self, request):
                self.waiting.append(request)

            def schedule(self):
                self.waiting_seen_by_schedule = [
                    request.request_id for request in self.waiting
                ]
                return types.SimpleNamespace(
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=None,
                )

        self.patch.patch_scheduler_class(FakeScheduler)
        scheduler = FakeScheduler()
        scheduler.schedule()

        self.assertEqual(scheduler.waiting_seen_by_schedule, [])
        self.assertEqual(
            [request.request_id for request in scheduler.waiting],
            ["prefill"],
        )

    def test_scheduler_allows_mixed_prefill_decode_when_configured(self):
        class FakeScheduler:
            def __init__(self):
                base = _scheduler(
                    use_hybrid_apc=True,
                    use_qwen_hybrid_chunked_prefill=True,
                    max_num_seqs=2,
                    additional_config={
                        "hybrid_apc_allow_mixed_prefill_decode": True,
                    },
                )
                self.vllm_config = base.vllm_config
                self.cache_config = base.cache_config
                self.scheduler_config = base.scheduler_config
                self.running = [types.SimpleNamespace(request_id="decode")]
                self.waiting = [types.SimpleNamespace(request_id="prefill")]
                self.waiting_seen_by_schedule = None

            def add_request(self, request):
                self.waiting.append(request)

            def schedule(self):
                self.waiting_seen_by_schedule = [
                    request.request_id for request in self.waiting
                ]
                self.waiting = []
                return types.SimpleNamespace(
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=None,
                )

        self.patch.patch_scheduler_class(FakeScheduler)
        scheduler = FakeScheduler()
        scheduler.schedule()

        self.assertEqual(scheduler.waiting_seen_by_schedule, ["prefill"])
        self.assertEqual(scheduler.waiting, [])

    def test_scheduler_keeps_waiting_prefills_when_no_decode_running(self):
        class FakeScheduler:
            def __init__(self):
                base = _scheduler(
                    use_hybrid_apc=True,
                    use_qwen_hybrid_chunked_prefill=True,
                    max_num_seqs=2,
                )
                self.vllm_config = base.vllm_config
                self.cache_config = base.cache_config
                self.scheduler_config = base.scheduler_config
                self.running = []
                self.waiting = [types.SimpleNamespace(request_id="prefill")]
                self.waiting_seen_by_schedule = None

            def add_request(self, request):
                self.waiting.append(request)

            def schedule(self):
                self.waiting_seen_by_schedule = [
                    request.request_id for request in self.waiting
                ]
                return types.SimpleNamespace(
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=None,
                )

        self.patch.patch_scheduler_class(FakeScheduler)
        scheduler = FakeScheduler()
        scheduler.schedule()

        self.assertEqual(scheduler.waiting_seen_by_schedule, ["prefill"])

    def test_scheduler_output_metadata_does_not_rewrite_cached_context_hit(self):
        class FakeScheduler:
            def __init__(self):
                base = _scheduler(
                    block_size=4,
                    enable_backed_prefix_reads=True,
                    use_qwen_hybrid_chunked_prefill=True,
                )
                self.vllm_config = base.vllm_config
                self.cache_config = base.cache_config
                self.scheduler_config = base.scheduler_config
                self.requests = {
                    "req-a": types.SimpleNamespace(
                        request_id="req-a",
                        prompt_token_ids=[10, 11, 12, 13, 14, 15],
                        num_tokens=6,
                        cache_salt=None,
                    )
                }

            def add_request(self, request):
                del request

            def schedule(self):
                return types.SimpleNamespace(
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=types.SimpleNamespace(
                        req_ids=["req-a"],
                        new_block_ids=[([11, 12],)],
                        num_computed_tokens=[6],
                        num_output_tokens=[0],
                    ),
                    num_scheduled_tokens={"req-a": 1},
                    total_num_scheduled_tokens=1,
                )

        token_ids = [10, 11, 12, 13, 14, 15]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=4,
            max_prefix_len=4,
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[4],
                prefix_len=4,
                block_size=4,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )

        self.patch.patch_scheduler_class(FakeScheduler)
        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            scheduler_output = FakeScheduler().schedule()

        cached_reqs = scheduler_output.scheduled_cached_reqs
        metadata = getattr(
            scheduler_output,
            "_qwen36_hybrid_apc_metadata_by_request_id",
        )

        self.assertEqual(cached_reqs.num_computed_tokens, [6])
        self.assertEqual(scheduler_output.num_scheduled_tokens["req-a"], 1)
        self.assertEqual(scheduler_output.total_num_scheduled_tokens, 1)
        self.assertEqual(metadata["req-a"]["vllm_attention_hit_len"], 6)
        self.assertEqual(metadata["req-a"]["request_prefix_len"], 6)

    def test_scheduler_output_does_not_cap_decode_rows(self):
        class FakeScheduler:
            def __init__(self):
                base = _scheduler(
                    block_size=4,
                    enable_backed_prefix_reads=True,
                    use_qwen_hybrid_chunked_prefill=True,
                )
                self.vllm_config = base.vllm_config
                self.cache_config = base.cache_config
                self.scheduler_config = base.scheduler_config
                self.requests = {
                    "req-a": types.SimpleNamespace(
                        request_id="req-a",
                        prompt_token_ids=[10, 11, 12, 13, 14, 15],
                        num_tokens=6,
                        cache_salt=None,
                    )
                }

            def add_request(self, request):
                del request

            def schedule(self):
                return types.SimpleNamespace(
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=types.SimpleNamespace(
                        req_ids=["req-a"],
                        new_block_ids=[([11, 12],)],
                        num_computed_tokens=[6],
                        num_output_tokens=[1],
                    ),
                    num_scheduled_tokens={"req-a": 1},
                    total_num_scheduled_tokens=1,
                )

        token_ids = [10, 11, 12, 13, 14, 15]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=4,
            max_prefix_len=4,
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[4],
                prefix_len=4,
                block_size=4,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )

        self.patch.patch_scheduler_class(FakeScheduler)
        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            scheduler_output = FakeScheduler().schedule()

        metadata = getattr(
            scheduler_output,
            "_qwen36_hybrid_apc_metadata_by_request_id",
        )

        self.assertEqual(scheduler_output.scheduled_cached_reqs.num_computed_tokens, [6])
        self.assertEqual(scheduler_output.num_scheduled_tokens["req-a"], 1)
        self.assertEqual(scheduler_output.total_num_scheduled_tokens, 1)
        self.assertEqual(metadata["req-a"]["vllm_attention_hit_len"], 6)

    def test_backed_prefix_read_allows_batched_scheduler_when_configured(self):
        scheduler = _scheduler(
            block_size=2,
            enable_backed_prefix_reads=True,
            use_qwen_hybrid_chunked_prefill=True,
            max_num_seqs=2,
        )
        token_ids = [10, 11, 12, 13, 14]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=2,
            max_prefix_len=4,
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[2],
                prefix_len=2,
                block_size=2,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[4],
                prefix_len=4,
                block_size=2,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        request = types.SimpleNamespace(
            prompt_token_ids=token_ids,
            num_tokens=len(token_ids),
            cache_salt=None,
        )

        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            self.assertFalse(
                self.patch.should_disable_unbacked_prefix_reads(scheduler, request)
            )
        self.assertIsNotNone(
            self.patch.pop_hybrid_apc_authorized_prefix_key(
                prefix_len=4,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )

    def test_env_backed_prefix_override_allows_batched_scheduler(self):
        scheduler = _scheduler(
            block_size=2,
            enable_backed_prefix_reads=False,
            use_qwen_hybrid_chunked_prefill=False,
            max_num_seqs=2,
        )
        token_ids = [10, 11, 12, 13, 14]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=2,
            max_prefix_len=4,
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[2],
                prefix_len=2,
                block_size=2,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[4],
                prefix_len=4,
                block_size=2,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        request = types.SimpleNamespace(
            prompt_token_ids=token_ids,
            num_tokens=len(token_ids),
            cache_salt=None,
        )

        with patch.dict(
            os.environ,
            {
                "QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1",
                "QWEN36_HYBRID_APC_ENABLE_BACKED_PREFIX_READS": "1",
            },
        ):
            self.assertFalse(
                self.patch.should_disable_unbacked_prefix_reads(scheduler, request)
            )
        self.assertIsNotNone(
            self.patch.pop_hybrid_apc_authorized_prefix_key(
                prefix_len=4,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )

    def test_additional_config_overrides_scheduler_registry_key_metadata(self):
        scheduler = _scheduler(
            block_size=2,
            model_revision="stale-rev",
            additional_config={
                "hybrid_apc_model_revision": "runtime-rev",
                "hybrid_apc_layout_version": 2,
                "tp_rank": 3,
                "hybrid_recurrent_cache_dtype": "bf16",
                "hybrid_conv_cache_dtype": "float32",
            },
        )
        token_ids = [10, 11, 12, 13, 14]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=2,
            max_prefix_len=4,
        )
        key = self.patch.HybridGDNPrefixKey(
            cumulative_prefix_hash=hashes[4],
            prefix_len=4,
            block_size=2,
            cache_salt=None,
            model_revision="runtime-rev",
            layout_version=2,
            tp_rank=3,
            recurrent_dtype="bfloat16",
            conv_dtype="float32",
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(key)
        request = types.SimpleNamespace(
            prompt_token_ids=token_ids,
            num_tokens=len(token_ids),
            cache_salt=None,
        )

        self.assertEqual(self.patch.backed_gdn_prefix_hit(scheduler, request), key)

    def test_mismatched_gdn_checkpoint_keeps_prefix_read_disabled(self):
        scheduler = _scheduler(block_size=2)
        token_ids = [10, 11, 12, 13, 14]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=2,
            max_prefix_len=4,
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[4],
                prefix_len=4,
                block_size=2,
                cache_salt="tenant-a",
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        request = types.SimpleNamespace(
            prompt_token_ids=token_ids,
            num_tokens=len(token_ids),
            cache_salt="tenant-b",
        )

        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            self.assertEqual(
                self.patch.backed_gdn_prefix_hit_len(scheduler, request),
                0,
            )
            self.assertTrue(
                self.patch.should_disable_unbacked_prefix_reads(scheduler, request)
            )

    def test_missing_model_revision_defaults_to_unknown(self):
        scheduler = _scheduler(block_size=2, model_revision=None)
        token_ids = [10, 11, 12, 13, 14]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=2,
            max_prefix_len=4,
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[4],
                prefix_len=4,
                block_size=2,
                cache_salt=None,
                model_revision="unknown",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        request = types.SimpleNamespace(
            prompt_token_ids=token_ids,
            num_tokens=len(token_ids),
            cache_salt=None,
        )

        self.assertEqual(
            self.patch.backed_gdn_prefix_hit_len(scheduler, request),
            4,
        )

    def test_import_hook_does_not_import_scheduler_immediately(self):
        installed = self.patch.install_import_hook()

        self.assertFalse(installed)
        self.assertNotIn(_SCHEDULER_MODULE, sys.modules)
        self.assertTrue(
            any(
                getattr(finder, "_qwen36_hybrid_apc_import_hook", False)
                for finder in sys.meta_path
            )
        )

    def test_import_hook_patches_already_loaded_scheduler_module(self):
        calls = []

        class FakeScheduler:
            def __init__(self):
                self.vllm_config = _scheduler(
                    disable_unbacked_prefix_reads=True
                ).vllm_config

            def add_request(self, request):
                calls.append(request.skip_reading_prefix_cache)

        module = types.SimpleNamespace(Scheduler=FakeScheduler)
        sys.modules[_SCHEDULER_MODULE] = module

        installed = self.patch.install_import_hook()
        request = types.SimpleNamespace(skip_reading_prefix_cache=False)
        FakeScheduler().add_request(request)

        self.assertTrue(installed)
        self.assertEqual(calls, [True])
        self.assertTrue(request.skip_reading_prefix_cache)

    def test_runner_patch_exposes_request_ids_during_model_execution(self):
        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(model=types.SimpleNamespace())

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                self.seen_request_ids = getattr(
                    self.model.model,
                    "_qwen36_vllm_request_ids",
                    None,
                )
                self.seen_cached_request_ids = getattr(
                    self.model.model,
                    "_qwen36_vllm_cached_request_ids",
                    None,
                )
                self.seen_prefill_completion_state = getattr(
                    self.model.model,
                    "_qwen36_vllm_prefill_completion_state",
                    None,
                )
                self.seen_metadata = getattr(
                    self.model.model,
                    "_qwen36_vllm_hybrid_apc_metadata_by_request_id",
                    None,
                )
                self.seen_request_records = getattr(
                    self.model.model,
                    "_qwen36_vllm_hybrid_apc_request_records",
                    None,
                )
                return self.seen_request_ids

        installed = self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()
        result = runner._execute_model_for_text(
            types.SimpleNamespace(
                request_ids=["req-a"],
                _qwen36_cached_request_ids=("req-a",),
                prefill_completion_state="done",
                _qwen36_hybrid_apc_metadata_by_request_id={
                    "req-a": {"cumulative_hashes_by_prefix_len": {4: b"h4"}}
                },
                _qwen36_hybrid_apc_request_records=(
                    {
                        "request_id": "req-a",
                        "cumulative_hashes_by_prefix_len": {4: b"h4"},
                    },
                ),
            )
        )

        self.assertTrue(installed)
        self.assertEqual(result, ("req-a",))
        self.assertEqual(runner.seen_request_ids, ("req-a",))
        self.assertEqual(runner.seen_cached_request_ids, ("req-a",))
        self.assertEqual(runner.seen_prefill_completion_state, "done")
        self.assertEqual(
            runner.seen_metadata,
            {"req-a": {"cumulative_hashes_by_prefix_len": {4: b"h4"}}},
        )
        self.assertEqual(
            runner.seen_request_records,
            (
                {
                    "request_id": "req-a",
                    "cumulative_hashes_by_prefix_len": {4: b"h4"},
                },
            ),
        )
        self.assertFalse(hasattr(runner.model, "_qwen36_vllm_request_ids"))
        self.assertFalse(hasattr(runner.model.model, "_qwen36_vllm_request_ids"))
        self.assertFalse(hasattr(runner.model.model, "_qwen36_vllm_cached_request_ids"))
        self.assertFalse(
            hasattr(runner.model.model, "_qwen36_vllm_hybrid_apc_request_records")
        )

    def test_runner_patch_applies_runtime_hybrid_apc_config_during_execution(self):
        class FakeRunner:
            def __init__(self):
                self.vllm_config = types.SimpleNamespace(
                    additional_config={
                        "hybrid_apc_require_vllm_metadata": True,
                        "hybrid_apc_enable_backed_prefix_reads": True,
                    }
                )
                self.model = types.SimpleNamespace(
                    model=types.SimpleNamespace(
                        config=types.SimpleNamespace(
                            hybrid_apc_require_vllm_metadata=False,
                            hybrid_apc_allow_local_hash_fallback=True,
                            hybrid_apc_require_attention_block_refs=False,
                            hybrid_apc_reject_unbacked_attention_hits=False,
                            hybrid_apc_enable_backed_prefix_reads=False,
                        ),
                        hybrid_apc_bridge=types.SimpleNamespace(
                            allow_local_hash_fallback=True,
                            require_attention_block_refs=False,
                            reject_unbacked_attention_hits=False,
                        ),
                    )
                )

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del model_input, intermediate_tensors
                config = self.model.model.config
                bridge = self.model.model.hybrid_apc_bridge
                return (
                    config.hybrid_apc_require_vllm_metadata,
                    config.hybrid_apc_allow_local_hash_fallback,
                    config.hybrid_apc_require_attention_block_refs,
                    config.hybrid_apc_reject_unbacked_attention_hits,
                    config.hybrid_apc_enable_backed_prefix_reads,
                    bridge.allow_local_hash_fallback,
                    bridge.require_attention_block_refs,
                    bridge.reject_unbacked_attention_hits,
                )

        installed = self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()
        result = runner._execute_model_for_text(types.SimpleNamespace())

        self.assertTrue(installed)
        self.assertEqual(result, (True, False, True, True, True, False, True, True))
        config = runner.model.model.config
        self.assertFalse(config.hybrid_apc_require_vllm_metadata)
        self.assertTrue(config.hybrid_apc_allow_local_hash_fallback)
        self.assertFalse(config.hybrid_apc_require_attention_block_refs)
        self.assertFalse(config.hybrid_apc_reject_unbacked_attention_hits)
        self.assertFalse(config.hybrid_apc_enable_backed_prefix_reads)
        bridge = runner.model.model.hybrid_apc_bridge
        self.assertTrue(bridge.allow_local_hash_fallback)
        self.assertFalse(bridge.require_attention_block_refs)
        self.assertFalse(bridge.reject_unbacked_attention_hits)

    def test_runner_patch_attaches_scheduler_request_sources_to_model_input(self):
        @dataclass(frozen=True)
        class FrozenModelInput:
            request_ids: list[str]

        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(model=types.SimpleNamespace())

            def _prepare_model_input(self, scheduler_output):
                del scheduler_output
                return FrozenModelInput(request_ids=["cached-1", "new-1"])

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

        installed = self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()
        scheduler_output = types.SimpleNamespace(
            scheduled_cached_reqs=types.SimpleNamespace(req_ids=["cached-1"]),
            scheduled_new_reqs=[types.SimpleNamespace(req_id="new-1")],
            _qwen36_hybrid_apc_metadata_by_request_id={
                "new-1": {"attention_block_refs_by_prefix_len": {4: (3, 4)}}
            },
        )
        model_input = runner._prepare_model_input(scheduler_output)

        self.assertTrue(installed)
        self.assertEqual(model_input._qwen36_cached_request_ids, ("cached-1",))
        self.assertEqual(model_input._qwen36_new_request_ids, ("new-1",))
        self.assertEqual(
            model_input._qwen36_hybrid_apc_metadata_by_request_id,
            {"new-1": {"attention_block_refs_by_prefix_len": {4: (3, 4)}}},
        )
        self.assertEqual(
            model_input._qwen36_hybrid_apc_request_records,
            (
                {"request_id": "cached-1"},
                {
                    "request_id": "new-1",
                    "attention_block_refs_by_prefix_len": {4: (3, 4)},
                },
            ),
        )

    def test_runner_patch_builds_ordered_hybrid_apc_request_records(self):
        @dataclass(frozen=True)
        class FrozenModelInput:
            request_ids: list[str]

        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(model=types.SimpleNamespace())

            def _prepare_model_input(self, scheduler_output):
                del scheduler_output
                return FrozenModelInput(request_ids=["warm", "cold"])

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

        installed = self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()
        scheduler_output = types.SimpleNamespace(
            scheduled_cached_reqs=types.SimpleNamespace(req_ids=["warm"]),
            scheduled_new_reqs=[types.SimpleNamespace(req_id="cold")],
            num_scheduled_tokens={"warm": 2, "cold": 6},
            _qwen36_hybrid_apc_metadata_by_request_id={
                "warm": {
                    "vllm_attention_hit_len": 4,
                    "request_prefix_len": 6,
                    "cumulative_hashes_by_prefix_len": {4: b"warm-h4"},
                    "attention_block_refs_by_prefix_len": {4: (1, 2)},
                },
                "cold": {
                    "vllm_attention_hit_len": 0,
                    "request_prefix_len": 6,
                },
            },
        )
        model_input = runner._prepare_model_input(scheduler_output)

        self.assertTrue(installed)
        self.assertEqual(
            model_input._qwen36_hybrid_apc_request_records,
            (
                {
                    "request_id": "warm",
                    "cumulative_hashes_by_prefix_len": {4: b"warm-h4"},
                    "attention_block_refs_by_prefix_len": {4: (1, 2)},
                    "request_prefix_len": 6,
                    "vllm_attention_hit_len": 4,
                    "active_suffix_len": 2,
                },
                {
                    "request_id": "cold",
                    "request_prefix_len": 6,
                    "vllm_attention_hit_len": 0,
                    "active_suffix_len": 6,
                },
            ),
        )

    def test_runner_patch_uses_scheduler_ids_when_model_input_has_no_ids(self):
        @dataclass(frozen=True)
        class FrozenModelInput:
            pass

        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(model=types.SimpleNamespace())

            def _prepare_model_input(self, scheduler_output):
                del scheduler_output
                return FrozenModelInput()

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

        installed = self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()
        scheduler_output = types.SimpleNamespace(
            scheduled_cached_reqs=types.SimpleNamespace(req_ids=["warm"]),
            scheduled_new_reqs=[],
            num_scheduled_tokens={"warm": 2},
            _qwen36_hybrid_apc_metadata_by_request_id={
                "warm": {
                    "vllm_attention_hit_len": 2,
                    "request_prefix_len": 4,
                    "active_suffix_len": 2,
                },
            },
        )
        model_input = runner._prepare_model_input(scheduler_output)

        self.assertTrue(installed)
        self.assertEqual(model_input._qwen36_cached_request_ids, ("warm",))
        self.assertEqual(
            model_input._qwen36_hybrid_apc_request_records,
            (
                {
                    "request_id": "warm",
                    "request_prefix_len": 4,
                    "vllm_attention_hit_len": 2,
                    "active_suffix_len": 2,
                },
            ),
        )

    def test_runner_patch_expands_completed_only_prefill_logits(self):
        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(model=types.SimpleNamespace())

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

            def _prepare_logits_for_sampling(self, hidden_states, model_input):
                hidden_states = hidden_states.clone()
                for idx, state in enumerate(model_input.prefill_completion_state):
                    if not state.item():
                        hidden_states[idx] = float("-inf")
                return hidden_states

        self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()
        logits = runner._prepare_logits_for_sampling(
            torch.tensor([[1.0, 2.0, 3.0]]),
            types.SimpleNamespace(
                request_ids=["req-a", "req-b"],
                prefill_completion_state=torch.tensor([True, False]),
            ),
        )

        self.assertEqual(tuple(logits.shape), (2, 3))
        torch.testing.assert_close(logits[0], torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(torch.isneginf(logits[1]).all())

    def test_runner_patch_leaves_full_prefill_logits_unchanged(self):
        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(model=types.SimpleNamespace())

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

            def _prepare_logits_for_sampling(self, hidden_states, model_input):
                return hidden_states

        self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()
        hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        logits = runner._prepare_logits_for_sampling(
            hidden_states,
            types.SimpleNamespace(
                request_ids=["req-a", "req-b"],
                prefill_completion_state=torch.tensor([True, False]),
            ),
        )

        torch.testing.assert_close(logits, hidden_states)

    def test_runner_patch_clones_inference_tensor_before_on_device_prefill_mask(self):
        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(model=types.SimpleNamespace())

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

            def _sample_on_device(self, hidden_states, model_input):
                for idx, state in enumerate(model_input.prefill_completion_state):
                    if not state.item():
                        hidden_states[idx] = -1
                return hidden_states

        self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()
        with torch.inference_mode():
            hidden_states = torch.tensor([11, 22], dtype=torch.int32)

        sampled = runner._sample_on_device(
            hidden_states,
            types.SimpleNamespace(
                prefill_completion_state=torch.tensor([True, False]),
            ),
        )

        torch.testing.assert_close(sampled, torch.tensor([11, -1], dtype=torch.int32))
        torch.testing.assert_close(hidden_states, torch.tensor([11, 22], dtype=torch.int32))

    def test_runner_patch_masks_sampled_tokens_for_incomplete_prefill_rows(self):
        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(model=types.SimpleNamespace())

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

            def _sample_on_device(self, hidden_states, model_input):
                del hidden_states, model_input
                return types.SimpleNamespace(
                    sampled_token_ids=torch.tensor([[0], [33]], dtype=torch.int32),
                    logprobs_tensors=None,
                )

        self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()
        sampled = runner._sample_on_device(
            torch.tensor([[11], [22]], dtype=torch.int32),
            types.SimpleNamespace(
                prefill_completion_state=torch.tensor([False, True]),
            ),
        )

        torch.testing.assert_close(
            sampled.sampled_token_ids,
            torch.tensor([[-1], [33]], dtype=torch.int32),
        )

    def test_runner_patch_repairs_invalid_completed_prefill_sampled_token_from_logits(self):
        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(
                    model=types.SimpleNamespace(
                        config=types.SimpleNamespace(vocab_size=3)
                    )
                )

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

            def _sample_on_device(self, hidden_states, model_input):
                del hidden_states, model_input
                return types.SimpleNamespace(
                    sampled_token_ids=torch.tensor(
                        [[2147483647]], dtype=torch.int32
                    ),
                    logprobs_tensors=None,
                )

        self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()

        sampled = runner._sample_on_device(
            [
                torch.tensor([[2147483647]], dtype=torch.int32),
                torch.tensor([[[0.1, 3.0, 0.2]]], dtype=torch.float32),
            ],
            types.SimpleNamespace(
                prefill_completion_state=torch.tensor([True]),
            ),
        )

        torch.testing.assert_close(
            sampled.sampled_token_ids,
            torch.tensor([[1]], dtype=torch.int32),
        )

    def test_runner_patch_repairs_completed_only_output_row_from_logits(self):
        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(
                    model=types.SimpleNamespace(
                        config=types.SimpleNamespace(vocab_size=4)
                    )
                )

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

            def _sample_on_device(self, hidden_states, model_input):
                del hidden_states, model_input
                return types.SimpleNamespace(
                    sampled_token_ids=torch.tensor(
                        [[2147483647]], dtype=torch.int32
                    ),
                    logprobs_tensors=None,
                )

        self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()

        sampled = runner._sample_on_device(
            [
                torch.tensor([[2147483647]], dtype=torch.int32),
                torch.tensor([[[0.1, 0.2, 0.3, 4.0]]], dtype=torch.float32),
            ],
            types.SimpleNamespace(
                prefill_completion_state=torch.tensor([False, True]),
            ),
        )

        torch.testing.assert_close(
            sampled.sampled_token_ids,
            torch.tensor([[3]], dtype=torch.int32),
        )

    def test_runner_patch_rejects_invalid_completed_prefill_without_logits(self):
        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(
                    model=types.SimpleNamespace(
                        config=types.SimpleNamespace(vocab_size=248320)
                    )
                )

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

            def _sample_on_device(self, hidden_states, model_input):
                del hidden_states, model_input
                return types.SimpleNamespace(
                    sampled_token_ids=torch.tensor(
                        [[2147483647]], dtype=torch.int32
                    ),
                    logprobs_tensors=None,
                )

        self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()

        with self.assertRaisesRegex(
            ValueError,
            "--output-logits-with-on-device-sampling",
        ):
            runner._sample_on_device(
                torch.tensor([[2147483647]], dtype=torch.int32),
                types.SimpleNamespace(
                    prefill_completion_state=torch.tensor([True]),
                ),
            )

    def test_runner_patch_rejects_invalid_completed_prefill_with_sharded_logits(self):
        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(
                    model=types.SimpleNamespace(
                        config=types.SimpleNamespace(vocab_size=16)
                    )
                )

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

            def _sample_on_device(self, hidden_states, model_input):
                del hidden_states, model_input
                return types.SimpleNamespace(
                    sampled_token_ids=torch.tensor(
                        [[2147483647]], dtype=torch.int32
                    ),
                    logprobs_tensors=None,
                )

        self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()

        with self.assertRaisesRegex(
            ValueError,
            "gathers vocab-parallel output logits",
        ):
            runner._sample_on_device(
                [
                    torch.tensor([[2147483647]], dtype=torch.int32),
                    torch.tensor([[[0.1, 0.2, 4.0, 0.3]]], dtype=torch.float32),
                ],
                types.SimpleNamespace(
                    prefill_completion_state=torch.tensor([True]),
                ),
            )

    def test_runner_patch_masks_cpu_sampled_tokens_before_output_update(self):
        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(model=types.SimpleNamespace())

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

            def _generate_model_runner_output(self, sampler_output):
                return sampler_output.sampled_token_ids

        self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()
        setattr(
            runner,
            self.patch._RUNNER_PREFILL_STATE_FOR_OUTPUT_ATTR,
            torch.tensor([False, True]),
        )
        sampled = runner._generate_model_runner_output(
            types.SimpleNamespace(
                sampled_token_ids=torch.tensor([[0], [44]], dtype=torch.int32),
                logprobs_tensors=None,
            )
        )

        torch.testing.assert_close(
            sampled,
            torch.tensor([[-1], [44]], dtype=torch.int32),
        )

    def test_runner_patch_captures_prefill_state_during_sample_tokens(self):
        seen = []

        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(model=types.SimpleNamespace())
                self._cached_logits = torch.tensor([[1.0]])
                self._cached_model_input = types.SimpleNamespace(
                    prefill_completion_state=torch.tensor([False, True]),
                )

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

            def sample_tokens(self, grammar_output):
                del grammar_output
                seen.append(
                    getattr(
                        self,
                        self_patch._RUNNER_PREFILL_STATE_FOR_OUTPUT_ATTR,
                    )
                )
                return "sampled"

        self_patch = self.patch
        self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()
        result = runner.sample_tokens(None)

        self.assertEqual(result, "sampled")
        torch.testing.assert_close(seen[0], torch.tensor([False, True]))
        self.assertFalse(
            hasattr(runner, self.patch._RUNNER_PREFILL_STATE_FOR_OUTPUT_ATTR)
        )

    def test_runner_patch_returns_no_output_for_initial_async_sample(self):
        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(model=types.SimpleNamespace())
                self._cached_logits = None
                self._cached_model_input = None

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

            def sample_tokens(self, grammar_output):
                del grammar_output
                raise RuntimeError(
                    "sample_tokens() called without prior execute_model(). "
                    "Logits must be cached first."
                )

        self.patch.patch_neuron_model_runner_class(FakeRunner)

        self.assertIsNone(FakeRunner().sample_tokens(None))

    def test_runner_patch_uses_hybrid_attention_layers_for_kv_cache_spec(self):
        class FakeModelConfig:
            dtype = "bfloat16"

            def __init__(self):
                self.hf_config = types.SimpleNamespace(
                    num_hidden_layers=8,
                    num_attention_heads=24,
                    num_key_value_heads=4,
                    layer_types=[
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention",
                    ]
                    * 2,
                )

            def get_sliding_window(self):
                return None

        class FakeRunner:
            block_size = 256

            def __init__(self):
                self.model = types.SimpleNamespace(head_dim=256)
                self.model_config = FakeModelConfig()
                self.parallel_config = types.SimpleNamespace(tensor_parallel_size=4)

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

            def get_kv_cache_spec(self):
                return {
                    f"layers.{idx}.self_attn": FullAttentionSpec(
                        block_size=1,
                        num_kv_heads=4,
                        head_size=1,
                        dtype="original",
                    )
                    for idx in range(8)
                }

        installed = self.patch.patch_neuron_model_runner_class(FakeRunner)
        spec = FakeRunner().get_kv_cache_spec()

        self.assertTrue(installed)
        self.assertEqual(list(spec), ["layers.3.self_attn", "layers.7.self_attn"])
        self.assertEqual(spec["layers.3.self_attn"].block_size, 256)
        self.assertEqual(spec["layers.3.self_attn"].num_kv_heads, 1)
        self.assertEqual(spec["layers.3.self_attn"].head_size, 256)
        self.assertEqual(spec["layers.3.self_attn"].dtype, "bfloat16")

    def test_runner_patch_uses_full_attention_interval_for_kv_cache_spec(self):
        class FakeModelConfig:
            dtype = "bfloat16"

            def __init__(self):
                self.hf_config = types.SimpleNamespace(
                    num_hidden_layers=8,
                    num_attention_heads=24,
                    num_key_value_heads=4,
                    full_attention_interval=4,
                )

            def get_sliding_window(self):
                return None

        class FakeRunner:
            block_size = 128

            def __init__(self):
                self.model = types.SimpleNamespace(head_dim=256)
                self.model_config = FakeModelConfig()
                self.parallel_config = types.SimpleNamespace(tensor_parallel_size=4)

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

            def get_kv_cache_spec(self):
                return {"original": FullAttentionSpec(1, 4, 1, "original")}

        self.patch.patch_neuron_model_runner_class(FakeRunner)
        spec = FakeRunner().get_kv_cache_spec()

        self.assertEqual(list(spec), ["layers.3.self_attn", "layers.7.self_attn"])
        self.assertEqual(spec["layers.7.self_attn"].block_size, 128)
        self.assertEqual(spec["layers.7.self_attn"].num_kv_heads, 1)

    def test_runner_patch_keeps_original_kv_cache_spec_for_dense_attention(self):
        class FakeModelConfig:
            dtype = "bfloat16"

            def __init__(self):
                self.hf_config = types.SimpleNamespace(
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=4,
                    layer_types=["full_attention", "full_attention"],
                )

            def get_sliding_window(self):
                return None

        class FakeRunner:
            block_size = 128

            def __init__(self):
                self.original_kv_cache_spec_called = False
                self.model = types.SimpleNamespace(head_dim=256)
                self.model_config = FakeModelConfig()
                self.parallel_config = types.SimpleNamespace(tensor_parallel_size=4)

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

            def get_kv_cache_spec(self):
                self.original_kv_cache_spec_called = True
                return {"original": FullAttentionSpec(1, 4, 1, "original")}

        self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()
        spec = runner.get_kv_cache_spec()

        self.assertTrue(runner.original_kv_cache_spec_called)
        self.assertEqual(list(spec), ["original"])

    def test_import_hook_patches_already_loaded_neuron_runner_module(self):
        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(model=types.SimpleNamespace())

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                return getattr(self.model.model, "_qwen36_vllm_request_ids", None)

        module = types.SimpleNamespace(NeuronxDistributedModelRunner=FakeRunner)
        sys.modules[_VLLM_NEURON_RUNNER_MODULE] = module

        installed = self.patch.install_import_hook()
        runner = FakeRunner()
        result = runner._execute_model_for_text(
            types.SimpleNamespace(request_ids=("req-a",))
        )

        self.assertTrue(installed)
        self.assertEqual(result, ("req-a",))
        self.assertFalse(hasattr(runner.model, "_qwen36_vllm_request_ids"))
        self.assertFalse(hasattr(runner.model.model, "_qwen36_vllm_request_ids"))


if __name__ == "__main__":
    unittest.main()
