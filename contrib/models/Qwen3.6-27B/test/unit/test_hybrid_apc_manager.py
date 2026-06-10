# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import importlib.util
import types
from unittest.mock import patch

import torch


_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)
_VLLM_ROOT = os.path.join(_CONTRIB_ROOT, "vllm")
if _VLLM_ROOT not in sys.path:
    sys.path.insert(0, _VLLM_ROOT)

_HYBRID_APC_PATH = os.path.join(_CONTRIB_ROOT, "src", "hybrid_apc.py")
_SPEC = importlib.util.spec_from_file_location("qwen36_hybrid_apc", _HYBRID_APC_PATH)
_HYBRID_APC = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HYBRID_APC
_SPEC.loader.exec_module(_HYBRID_APC)

HybridAPCMetadataStore = _HYBRID_APC.HybridAPCMetadataStore
HybridAPCHitPlan = _HYBRID_APC.HybridAPCHitPlan
HybridAPCSchedulerBridge = _HYBRID_APC.HybridAPCSchedulerBridge
HybridAPCSlotAllocator = _HYBRID_APC.HybridAPCSlotAllocator
apply_hybrid_apc_prefill_plan = _HYBRID_APC.apply_hybrid_apc_prefill_plan
apply_hybrid_apc_suffix_prefill_plan = (
    _HYBRID_APC.apply_hybrid_apc_suffix_prefill_plan
)
build_cumulative_prefix_hashes = _HYBRID_APC.build_cumulative_prefix_hashes
estimate_qwen_gdn_checkpoint_bytes_per_rank = (
    _HYBRID_APC.estimate_qwen_gdn_checkpoint_bytes_per_rank
)
estimate_qwen_hybrid_cache_bytes_per_rank = (
    _HYBRID_APC.estimate_qwen_hybrid_cache_bytes_per_rank
)
import qwen36_hybrid_apc_scheduler_patch as _SCHEDULER_PATCH  # noqa: E402
from neuronx_distributed_inference.modules.async_execution import (  # noqa: E402
    _combine_vectorized_hybrid_apc_inputs,
    finish_hybrid_apc_request,
    prepare_hybrid_apc_request_for_execution,
)


def _store(**overrides):
    defaults = dict(
        required_gdn_layers=[0, 1, 2],
        block_size=128,
        checkpoint_interval=128,
        model_revision="rev-a",
        layout_version=1,
        tp_rank=0,
        recurrent_dtype="float32",
        conv_dtype="bfloat16",
    )
    defaults.update(overrides)
    return HybridAPCMetadataStore(**defaults)


def _insert(store, prefix_len, prefix_hash=None, **overrides):
    key = store.make_key(
        cumulative_prefix_hash=prefix_hash or f"h{prefix_len}",
        prefix_len=prefix_len,
        cache_salt=overrides.pop("cache_salt", "tenant-a"),
        model_revision=overrides.pop("model_revision", "rev-a"),
        layout_version=overrides.pop("layout_version", 1),
        tp_rank=overrides.pop("tp_rank", 0),
        recurrent_dtype=overrides.pop("recurrent_dtype", "float32"),
        conv_dtype=overrides.pop("conv_dtype", "bfloat16"),
    )
    checkpoint = store.insert(
        key=key,
        attention_block_refs=overrides.pop("attention_block_refs", range(prefix_len // 128)),
        gdn_checkpoint_slot=overrides.pop("gdn_checkpoint_slot", prefix_len // 128),
        **overrides,
    )
    return key, checkpoint


class TestHybridAPCMetadataStore(unittest.TestCase):
    def test_same_prefix_hash_and_salt_hits(self):
        store = _store()
        key, _checkpoint = _insert(store, 256)

        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={128: "h128", 256: "h256"},
            attention_hit_len=256,
            request_prefix_len=300,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        self.assertEqual(plan.checkpoint_key, key)
        self.assertEqual(plan.restore_checkpoint_prefix_len, 256)
        self.assertEqual(plan.usable_hit_len, 256)
        self.assertEqual(plan.residual_replay_len, 0)
        self.assertEqual(plan.suffix_len, 44)

    def test_same_tokens_with_different_salt_misses(self):
        store = _store()
        _insert(store, 128, prefix_hash="same")

        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={128: "same"},
            attention_hit_len=128,
            request_prefix_len=128,
            cache_salt="tenant-b",
            model_revision="rev-a",
        )

        self.assertIsNone(plan.checkpoint_key)
        self.assertEqual(plan.usable_hit_len, 0)

    def test_same_last_block_with_different_cumulative_hash_misses(self):
        store = _store()
        _insert(store, 256, prefix_hash="parent-a+block-z")

        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={256: "parent-b+block-z"},
            attention_hit_len=256,
            request_prefix_len=256,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        self.assertIsNone(plan.checkpoint_key)

    def test_missing_recurrent_layer_invalidates_hit(self):
        store = _store()
        key, _checkpoint = _insert(store, 128)
        store.mark_invalid(key, state_kind="recurrent", layer_id=1)

        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={128: "h128"},
            attention_hit_len=128,
            request_prefix_len=128,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        self.assertIsNone(plan.checkpoint_key)

    def test_missing_conv_layer_invalidates_hit(self):
        store = _store()
        key, _checkpoint = _insert(store, 128)
        store.mark_invalid(key, state_kind="conv", layer_id=2)

        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={128: "h128"},
            attention_hit_len=128,
            request_prefix_len=128,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        self.assertIsNone(plan.checkpoint_key)

    def test_dtype_layout_and_revision_are_identity(self):
        store = _store()
        _insert(store, 128)

        for kwargs in (
            {"recurrent_dtype": "bfloat16"},
            {"conv_dtype": "float32"},
            {"layout_version": 2},
            {"model_revision": "rev-b"},
        ):
            with self.subTest(kwargs=kwargs):
                plan = store.compute_hit_plan(
                    cumulative_hashes_by_prefix_len={128: "h128"},
                    attention_hit_len=128,
                    request_prefix_len=128,
                    cache_salt="tenant-a",
                    **kwargs,
                )
                self.assertIsNone(plan.checkpoint_key)

    def test_refcount_blocks_lru_eviction(self):
        store = _store(max_checkpoints=2)
        key128, _ = _insert(store, 128)
        key256, _ = _insert(store, 256)
        store.inc_ref(key128)
        key384, _ = _insert(store, 384)

        self.assertIsNotNone(store.lookup(key128))
        self.assertIsNone(store.lookup(key256))
        self.assertIsNotNone(store.lookup(key384))

    def test_evicting_gdn_checkpoint_makes_hybrid_hit_fallback(self):
        store = _store()
        _key, checkpoint = _insert(store, 128)
        store.on_gdn_checkpoint_evicted(checkpoint.gdn_checkpoint_slot)

        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={128: "h128"},
            attention_hit_len=128,
            request_prefix_len=128,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        self.assertIsNone(plan.checkpoint_key)

    def test_evicting_attention_block_makes_hybrid_hit_fallback(self):
        store = _store()
        key, _checkpoint = _insert(store, 256, attention_block_refs=(7, 8))
        invalidated = store.on_attention_block_evicted(8)

        self.assertEqual(invalidated, [key])
        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={256: "h256"},
            attention_hit_len=256,
            request_prefix_len=256,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )
        self.assertIsNone(plan.checkpoint_key)

    def test_non_block_aligned_prompt_uses_checkpoint_boundary_in_v0(self):
        store = _store(allow_residual_replay=False)
        _insert(store, 256)

        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={256: "h256"},
            attention_hit_len=300,
            request_prefix_len=384,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        self.assertEqual(plan.usable_hit_len, 256)
        self.assertEqual(plan.restore_checkpoint_prefix_len, 256)
        self.assertEqual(plan.residual_replay_len, 0)
        self.assertEqual(plan.suffix_len, 128)

    def test_residual_replay_requires_explicit_enablement(self):
        store = _store(allow_residual_replay=True)
        _insert(store, 256)

        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={256: "h256"},
            attention_hit_len=300,
            request_prefix_len=384,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        self.assertEqual(plan.usable_hit_len, 300)
        self.assertEqual(plan.restore_checkpoint_prefix_len, 256)
        self.assertEqual(plan.residual_replay_len, 44)
        self.assertEqual(plan.suffix_len, 84)

    def test_request_lifecycle_releases_restored_ref_on_finish(self):
        store = _store()
        key, _checkpoint = _insert(store, 128)

        record = store.on_request_restore(request_id="req-1", checkpoint_key=key)
        self.assertEqual(record.state, "RESTORED_FROM_HYBRID_APC")
        self.assertEqual(store.lookup(key).refcount, 1)

        store.on_prefill_running("req-1")
        store.on_decode_running("req-1")
        finished = store.on_request_finish("req-1")

        self.assertEqual(finished.state, "FINISHED")
        self.assertEqual(store.lookup(key).refcount, 0)

    def test_request_cancel_releases_ref_and_drops_pending_commit(self):
        store = _store()
        restored_key, _checkpoint = _insert(store, 128)
        committed_key, _checkpoint = _insert(store, 256)

        store.on_request_restore(request_id="req-1", checkpoint_key=restored_key)
        store.on_checkpoint_committed(
            request_id="req-1",
            checkpoint_key=committed_key,
        )
        cancelled = store.on_request_cancel("req-1")

        self.assertEqual(cancelled.state, "CANCELLED")
        self.assertEqual(store.lookup(restored_key).refcount, 0)
        self.assertIsNone(store.lookup(committed_key))

    def test_qwen_hbm_estimator_uses_checkpoint_slots_not_token_slots(self):
        per_checkpoint = estimate_qwen_gdn_checkpoint_bytes_per_rank()
        totals = estimate_qwen_hybrid_cache_bytes_per_rank(
            max_context_len=1024,
            checkpoint_interval=256,
        )

        self.assertEqual(totals["num_gdn_checkpoints"], 4)
        self.assertEqual(totals["gdn_checkpoint_bytes"], per_checkpoint * 4)
        self.assertGreater(totals["gdn_checkpoint_bytes"], totals["attention_kv_bytes"])


class TestHybridAPCPrefillPlanInputs(unittest.TestCase):
    def test_prefill_plan_materializes_suffix_restore_and_commit(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=2,
            recurrent_hit_len=2,
            conv_hit_len=2,
            usable_hit_len=2,
            restore_checkpoint_prefix_len=2,
            residual_replay_len=0,
            suffix_len=3,
            checkpoint_slot=5,
            checkpoint_key=None,
        )
        input_dict = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.int32),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.int32),
            "position_ids": torch.arange(5, dtype=torch.int32).unsqueeze(0),
            "slot_mapping": torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int32),
        }

        output = apply_hybrid_apc_prefill_plan(
            input_dict,
            plan=plan,
            commit_slot=7,
        )

        self.assertTrue(
            torch.equal(output["input_ids"], torch.tensor([[12, 13, 14]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["attention_mask"], torch.tensor([[1, 1, 1]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["position_ids"], torch.tensor([[2, 3, 4]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["slot_mapping"], torch.tensor([[2, 3, 4]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["computed_context_lens"], torch.tensor([[2]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["full_context_lens"], torch.tensor([[5]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["num_queries"], torch.tensor([[3]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["hybrid_restore_slot_ids"], torch.tensor([5], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["hybrid_restore_mask"], torch.tensor([1], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["hybrid_restore_prefix_lens"], torch.tensor([2], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["hybrid_commit_slot_ids"], torch.tensor([7], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["hybrid_commit_mask"], torch.tensor([1], dtype=torch.int32))
        )

    def test_prefill_plan_does_not_restore_without_checkpoint_slot(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=0,
            recurrent_hit_len=0,
            conv_hit_len=0,
            usable_hit_len=0,
            restore_checkpoint_prefix_len=0,
            residual_replay_len=0,
            suffix_len=3,
            checkpoint_slot=None,
            checkpoint_key=None,
        )
        input_dict = {
            "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.int32),
            "position_ids": torch.arange(3, dtype=torch.int32).unsqueeze(0),
        }

        output = apply_hybrid_apc_prefill_plan(input_dict, plan=plan)

        self.assertTrue(
            torch.equal(output["input_ids"], torch.tensor([[10, 11, 12]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["hybrid_restore_slot_ids"], torch.tensor([0], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["hybrid_restore_mask"], torch.tensor([0], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["hybrid_commit_mask"], torch.tensor([0], dtype=torch.int32))
        )

    def test_prefill_plan_rejects_restore_boundary_without_checkpoint_slot(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=2,
            recurrent_hit_len=0,
            conv_hit_len=0,
            usable_hit_len=0,
            restore_checkpoint_prefix_len=2,
            residual_replay_len=0,
            suffix_len=1,
            checkpoint_slot=None,
            checkpoint_key=None,
        )

        with self.assertRaisesRegex(ValueError, "requires a checkpoint slot"):
            apply_hybrid_apc_prefill_plan(
                {"input_ids": torch.tensor([[10, 11, 12]], dtype=torch.int32)},
                plan=plan,
            )

    def test_prefill_plan_uses_restore_boundary_for_residual_replay(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=5,
            recurrent_hit_len=4,
            conv_hit_len=4,
            usable_hit_len=5,
            restore_checkpoint_prefix_len=4,
            residual_replay_len=1,
            suffix_len=2,
            checkpoint_slot=9,
            checkpoint_key=None,
        )
        input_dict = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15, 16]], dtype=torch.int32),
            "position_ids": torch.arange(7, dtype=torch.int32).unsqueeze(0),
        }

        output = apply_hybrid_apc_prefill_plan(
            input_dict,
            plan=plan,
            commit_slot=10,
        )

        self.assertTrue(
            torch.equal(output["input_ids"], torch.tensor([[14, 15, 16]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["position_ids"], torch.tensor([[4, 5, 6]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["computed_context_lens"], torch.tensor([[4]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["num_queries"], torch.tensor([[3]], dtype=torch.int32))
        )

    def test_prefill_plan_slices_flattened_slot_mapping_with_suffix(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=5,
            recurrent_hit_len=4,
            conv_hit_len=4,
            usable_hit_len=5,
            restore_checkpoint_prefix_len=4,
            residual_replay_len=1,
            suffix_len=2,
            checkpoint_slot=9,
            checkpoint_key=None,
        )
        input_dict = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15, 16]], dtype=torch.int32),
            "position_ids": torch.arange(7, dtype=torch.int32).unsqueeze(0),
            "slot_mapping": torch.arange(100, 107, dtype=torch.int32),
        }

        output = apply_hybrid_apc_prefill_plan(
            input_dict,
            plan=plan,
            commit_slot=10,
        )

        self.assertTrue(
            torch.equal(output["slot_mapping"], torch.tensor([104, 105, 106], dtype=torch.int32))
        )

    def test_prefill_plan_synthesizes_padding_suffix_slots_from_block_table(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=4,
            recurrent_hit_len=4,
            conv_hit_len=4,
            usable_hit_len=4,
            restore_checkpoint_prefix_len=4,
            residual_replay_len=0,
            suffix_len=3,
            checkpoint_slot=9,
            checkpoint_key=None,
        )
        input_dict = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15, 16]], dtype=torch.int32),
            "position_ids": torch.arange(7, dtype=torch.int32).unsqueeze(0),
            "slot_mapping": torch.full((1, 7), -1, dtype=torch.int32),
            "block_table": torch.tensor([[1, 3, 4, 5]], dtype=torch.int32),
        }

        output = apply_hybrid_apc_prefill_plan(
            input_dict,
            plan=plan,
            commit_slot=10,
            block_size=4,
        )

        self.assertTrue(
            torch.equal(output["slot_mapping"], torch.tensor([[12, 13, 14]], dtype=torch.int32))
        )

    def test_prefill_plan_repairs_negative_active_slots_from_block_table(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=4,
            recurrent_hit_len=0,
            conv_hit_len=0,
            usable_hit_len=0,
            restore_checkpoint_prefix_len=0,
            residual_replay_len=0,
            suffix_len=6,
            checkpoint_slot=None,
            checkpoint_key=None,
        )
        input_dict = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.int32),
            "position_ids": torch.arange(6, dtype=torch.int32).unsqueeze(0),
            "slot_mapping": torch.tensor([[-1, -1, 10, 11, 12, 13]], dtype=torch.int32),
            "block_table": torch.tensor([[2, 3]], dtype=torch.int32),
        }

        output = apply_hybrid_apc_prefill_plan(
            input_dict,
            plan=plan,
            block_size=4,
        )

        self.assertTrue(
            torch.equal(
                output["slot_mapping"],
                torch.tensor([[8, 9, 10, 11, 12, 13]], dtype=torch.int32),
            )
        )

    def test_prefill_plan_repairs_too_short_active_slots_from_block_table(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=0,
            recurrent_hit_len=0,
            conv_hit_len=0,
            usable_hit_len=0,
            restore_checkpoint_prefix_len=0,
            residual_replay_len=0,
            suffix_len=6,
            checkpoint_slot=None,
            checkpoint_key=None,
        )
        input_dict = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.int32),
            "position_ids": torch.arange(6, dtype=torch.int32).unsqueeze(0),
            "slot_mapping": torch.tensor([12], dtype=torch.int32),
            "block_table": torch.tensor([[2, 3]], dtype=torch.int32),
        }

        output = apply_hybrid_apc_prefill_plan(
            input_dict,
            plan=plan,
            block_size=4,
        )

        self.assertTrue(
            torch.equal(
                output["slot_mapping"],
                torch.tensor([[8, 9, 10, 11, 12, 13]], dtype=torch.int32),
            )
        )

    def test_prefill_plan_rebuilds_unbacked_attention_hit_slots(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=4,
            recurrent_hit_len=0,
            conv_hit_len=0,
            usable_hit_len=0,
            restore_checkpoint_prefix_len=0,
            residual_replay_len=0,
            suffix_len=6,
            checkpoint_slot=None,
            checkpoint_key=None,
        )
        input_dict = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.int32),
            "position_ids": torch.arange(6, dtype=torch.int32).unsqueeze(0),
            "slot_mapping": torch.tensor([[12, 13, 14, 15, 16, 17]], dtype=torch.int32),
            "block_table": torch.tensor([[2, 3]], dtype=torch.int32),
        }

        output = apply_hybrid_apc_prefill_plan(
            input_dict,
            plan=plan,
            block_size=4,
        )

        self.assertTrue(
            torch.equal(
                output["slot_mapping"],
                torch.tensor([[8, 9, 10, 11, 12, 13]], dtype=torch.int32),
            )
        )

    def test_suffix_prefill_plan_preserves_active_block_table(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=4,
            recurrent_hit_len=4,
            conv_hit_len=4,
            usable_hit_len=4,
            restore_checkpoint_prefix_len=4,
            residual_replay_len=0,
            suffix_len=2,
            checkpoint_slot=1,
            checkpoint_key=None,
        )
        block_table = torch.tensor([[7, 8, 9, 10]], dtype=torch.int32)

        output = apply_hybrid_apc_suffix_prefill_plan(
            {
                "input_ids": torch.tensor([[14, 15]], dtype=torch.int32),
                "block_table": block_table,
            },
            plan=plan,
            request_prefix_len=6,
            attention_block_refs=(7,),
        )

        self.assertTrue(torch.equal(output["block_table"], block_table))


class TestHybridAPCVectorizedInputCombiner(unittest.TestCase):
    def test_backed_restore_uses_full_attention_mask_with_bucketed_suffix(self):
        neuron_config = types.SimpleNamespace(
            context_encoding_buckets=[256, 512, 1024, 2048, 4096],
            pa_block_size=256,
            seq_len=4096,
        )
        model = types.SimpleNamespace(
            neuron_config=neuron_config,
            config=types.SimpleNamespace(neuron_config=neuron_config),
        )
        rows = []
        for row_idx in range(2):
            rows.append(
                {
                    "input_ids": torch.arange(16, dtype=torch.int32).unsqueeze(0),
                    "attention_mask": torch.ones((1, 16), dtype=torch.int32),
                    "position_ids": torch.arange(
                        256,
                        272,
                        dtype=torch.int32,
                    ).unsqueeze(0),
                    "slot_mapping": torch.arange(
                        row_idx * 100,
                        row_idx * 100 + 16,
                        dtype=torch.int32,
                    ).unsqueeze(0),
                    "block_table": torch.tensor([[17 + row_idx]], dtype=torch.int32),
                    "full_context_lens": torch.tensor([[272]], dtype=torch.int32),
                    "computed_context_lens": torch.tensor([[256]], dtype=torch.int32),
                    "num_queries": torch.tensor([[16]], dtype=torch.int32),
                    "hybrid_restore_mask": torch.tensor([1], dtype=torch.int32),
                }
            )

        combined = _combine_vectorized_hybrid_apc_inputs(model, {}, rows)

        self.assertEqual(tuple(combined["input_ids"].shape), (2, 256))
        self.assertEqual(tuple(combined["position_ids"].shape), (2, 256))
        self.assertEqual(tuple(combined["slot_mapping"].shape), (2, 256))
        self.assertEqual(tuple(combined["block_table"].shape), (2, 16))
        self.assertEqual(tuple(combined["attention_mask"].shape), (2, 4096))
        self.assertTrue(
            torch.equal(
                combined["attention_mask"][:, :272],
                torch.ones((2, 272), dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                combined["attention_mask"][:, 272:],
                torch.zeros((2, 3824), dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                combined["slot_mapping"][0, :16],
                torch.arange(16, dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                combined["slot_mapping"][1, :16],
                torch.arange(100, 116, dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                combined["slot_mapping"][:, 16:],
                torch.full((2, 240), -1, dtype=torch.int32),
            )
        )


class TestHybridAPCSchedulerBridge(unittest.TestCase):
    def tearDown(self):
        _SCHEDULER_PATCH.clear_hybrid_apc_gdn_checkpoint_registry()

    def test_slot_allocator_validates_lifecycle(self):
        allocator = HybridAPCSlotAllocator(num_slots=2)

        with self.assertRaisesRegex(ValueError, "outside"):
            allocator.validate_slot_range(2)
        with self.assertRaisesRegex(ValueError, "not reserved"):
            allocator.mark_committed(1)

        slot = allocator.reserve()
        allocator.mark_committed(slot)

        self.assertEqual(allocator.committed_slots, (slot,))

    def test_cumulative_prefix_hash_includes_parent_prefix(self):
        tokens_a = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        tokens_b = torch.tensor([[9, 8, 3, 4]], dtype=torch.int32)

        hashes_a = build_cumulative_prefix_hashes(tokens_a, block_size=2)
        hashes_b = build_cumulative_prefix_hashes(tokens_b, block_size=2)

        self.assertNotEqual(hashes_a[4], hashes_b[4])

    def test_bridge_prepares_warm_suffix_and_commits_checkpoint(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=4)
        input_ids = torch.arange(256, dtype=torch.int32).unsqueeze(0)
        hashes = build_cumulative_prefix_hashes(input_ids, block_size=128)
        restored_key, _checkpoint = _insert(
            store,
            128,
            prefix_hash=hashes[128],
            gdn_checkpoint_slot=3,
        )
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
            reject_unbacked_attention_hits=False,
        )

        prepared = bridge.prepare_request(
            request_id="req-warm",
            input_dict={
                "input_ids": input_ids,
                "attention_mask": torch.ones((1, 256), dtype=torch.int32),
                "position_ids": torch.arange(256, dtype=torch.int32).unsqueeze(0),
            },
            attention_hit_len=128,
            cumulative_hashes_by_prefix_len=hashes,
            attention_block_refs_by_prefix_len={256: (11, 12)},
        )

        self.assertEqual(prepared.plan.restore_checkpoint_prefix_len, 128)
        self.assertEqual(prepared.commit_prefix_len, 256)
        self.assertEqual(prepared.commit_slot, 0)
        self.assertTrue(
            torch.equal(
                prepared.input_dict["input_ids"],
                torch.arange(128, 256, dtype=torch.int32).unsqueeze(0),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared.input_dict["position_ids"],
                torch.arange(128, 256, dtype=torch.int32).unsqueeze(0),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared.input_dict["hybrid_restore_mask"],
                torch.tensor([1], dtype=torch.int32),
            )
        )
        self.assertEqual(store.lookup(restored_key).refcount, 1)

        committed = bridge.commit_prefill(prepared)
        self.assertIsNotNone(committed)
        self.assertEqual(committed.gdn_checkpoint_slot, 0)
        self.assertEqual(committed.attention_block_refs, (11, 12))
        self.assertEqual(allocator.committed_slots, (0,))
        self.assertIsNotNone(store.lookup(prepared.commit_key))

        fake_scheduler = types.SimpleNamespace(
            cache_config=types.SimpleNamespace(block_size=128),
            vllm_config=types.SimpleNamespace(
                model_config=types.SimpleNamespace(
                    hf_config=types.SimpleNamespace(
                        hybrid_apc_model_revision="rev-a",
                        hybrid_apc_layout_version=1,
                        hybrid_recurrent_cache_dtype="float32",
                        hybrid_conv_cache_dtype="bfloat16",
                        tp_rank=0,
                    )
                )
            ),
        )
        fake_request = types.SimpleNamespace(
            prompt_token_ids=list(range(300)),
            num_tokens=300,
            cache_salt="tenant-a",
        )
        self.assertEqual(
            _SCHEDULER_PATCH.backed_gdn_prefix_hit_len(fake_scheduler, fake_request),
            256,
        )

        store.mark_invalid(prepared.commit_key, state_kind="conv")
        self.assertEqual(
            _SCHEDULER_PATCH.backed_gdn_prefix_hit_len(fake_scheduler, fake_request),
            0,
        )

        bridge.finish_request("req-warm")
        self.assertEqual(store.lookup(restored_key).refcount, 0)

    def test_bridge_full_input_commit_combines_restored_and_suffix_refs(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=4)
        input_ids = torch.arange(256, dtype=torch.int32).unsqueeze(0)
        hashes = build_cumulative_prefix_hashes(input_ids, block_size=128)
        restored_key, _checkpoint = _insert(
            store,
            128,
            prefix_hash=hashes[128],
            attention_block_refs=(7,),
            gdn_checkpoint_slot=3,
        )
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
            require_attention_block_refs=True,
        )

        prepared = bridge.prepare_request(
            request_id="req-full-offset-refs",
            input_dict={
                "input_ids": input_ids,
                "attention_mask": torch.ones((1, 256), dtype=torch.int32),
                "position_ids": torch.arange(256, dtype=torch.int32).unsqueeze(0),
            },
            attention_hit_len=128,
            cumulative_hashes_by_prefix_len=hashes,
            attention_block_refs_by_prefix_len={128: (9,)},
        )

        self.assertEqual(prepared.plan.checkpoint_key, restored_key)
        self.assertEqual(prepared.attention_block_refs, (7, 9))
        committed = bridge.commit_prefill(prepared)
        self.assertIsNotNone(committed)
        self.assertEqual(committed.attention_block_refs, (7, 9))

    def test_bridge_misses_without_gdn_checkpoint_and_cancels_reserved_slot(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=2)
        input_ids = torch.arange(256, dtype=torch.int32).unsqueeze(0)
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
            reject_unbacked_attention_hits=False,
        )

        prepared = bridge.prepare_request(
            request_id="req-cold",
            input_dict={"input_ids": input_ids},
            attention_hit_len=128,
        )

        self.assertEqual(prepared.plan.restore_checkpoint_prefix_len, 0)
        self.assertEqual(prepared.commit_slot, 0)
        self.assertTrue(
            torch.equal(prepared.input_dict["input_ids"], input_ids)
        )
        self.assertTrue(
            torch.equal(
                prepared.input_dict["hybrid_restore_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )

        cancelled = bridge.cancel_request(prepared)
        self.assertEqual(cancelled.state, "CANCELLED")
        self.assertEqual(allocator.reserved_slots, ())
        self.assertEqual(allocator.free_slots, (1, 0))
        self.assertEqual(len(store), 0)

    def test_bridge_rejects_attention_hit_without_gdn_checkpoint_by_default(self):
        bridge = HybridAPCSchedulerBridge(
            store=_store(),
            slot_allocator=HybridAPCSlotAllocator(num_slots=2),
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        with self.assertRaisesRegex(ValueError, "without a matching GDN checkpoint"):
            bridge.prepare_request(
                request_id="req-unbacked-hit",
                input_dict={
                    "input_ids": torch.arange(256, dtype=torch.int32).unsqueeze(0)
                },
                attention_hit_len=128,
            )

    def test_bridge_env_can_allow_unbacked_attention_fallback(self):
        bridge = HybridAPCSchedulerBridge(
            store=_store(),
            slot_allocator=HybridAPCSlotAllocator(num_slots=2),
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        with patch.dict(
            os.environ,
            {"QWEN36_ALLOW_UNBACKED_HYBRID_APC_FALLBACK": "1"},
        ):
            prepared = bridge.prepare_request(
                request_id="req-unbacked-fallback",
                input_dict={
                    "input_ids": torch.arange(256, dtype=torch.int32).unsqueeze(0)
                },
                attention_hit_len=128,
            )

        self.assertIsNone(prepared.plan.checkpoint_key)
        self.assertEqual(prepared.input_dict["hybrid_restore_mask"].item(), 0)

    def test_bridge_suffix_only_restore_is_explicit_and_unambiguous(self):
        store = _store()
        _insert(store, 128, gdn_checkpoint_slot=1)
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=HybridAPCSlotAllocator(num_slots=2),
            cache_salt="tenant-a",
            model_revision="rev-a",
        )
        suffix_ids = torch.arange(128, 256, dtype=torch.int32).unsqueeze(0)

        with self.assertRaisesRegex(ValueError, "without scheduler-authorized"):
            bridge.prepare_suffix_only_request(
                request_id="req-suffix-disabled",
                input_dict={"input_ids": suffix_ids},
                attention_hit_len=128,
                request_prefix_len=256,
            )

        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_ALLOW_UNHASHED_SINGLE_PREFIX_RESTORE": "1"},
        ):
            prepared = bridge.prepare_suffix_only_request(
                request_id="req-suffix",
                input_dict={"input_ids": suffix_ids},
                attention_hit_len=128,
                request_prefix_len=256,
            )

        self.assertIsNotNone(prepared)
        self.assertTrue(torch.equal(prepared.input_dict["input_ids"], suffix_ids))
        self.assertTrue(
            torch.equal(
                prepared.input_dict["position_ids"],
                torch.arange(128, 256, dtype=torch.int64).unsqueeze(0),
            )
        )
        self.assertEqual(prepared.input_dict["computed_context_lens"].item(), 128)
        self.assertEqual(prepared.input_dict["num_queries"].item(), 128)
        self.assertEqual(prepared.input_dict["hybrid_restore_mask"].item(), 1)
        self.assertEqual(prepared.input_dict["hybrid_restore_slot_ids"].item(), 1)
        self.assertTrue(
            torch.equal(
                prepared.input_dict["rotary_position_ids"],
                torch.arange(128, 256, dtype=torch.int32)
                .view(1, 1, 128)
                .expand(3, 1, 128),
            )
        )

    def test_bridge_suffix_only_restore_uses_scheduler_authorized_key(self):
        store = _store()
        _insert(store, 128, prefix_hash="h128-a", gdn_checkpoint_slot=0)
        key_b, _checkpoint_b = _insert(
            store,
            128,
            prefix_hash="h128-b",
            gdn_checkpoint_slot=1,
        )
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=HybridAPCSlotAllocator(num_slots=2),
            cache_salt="tenant-a",
            model_revision="rev-a",
        )
        _SCHEDULER_PATCH.authorize_hybrid_apc_prefix_read(key_b)

        prepared = bridge.prepare_suffix_only_request(
            request_id="req-authorized",
            input_dict={
                "input_ids": torch.arange(128, 256, dtype=torch.int32).unsqueeze(0)
            },
            attention_hit_len=128,
            request_prefix_len=256,
        )

        self.assertIsNotNone(prepared)
        self.assertEqual(prepared.plan.checkpoint_key, key_b)
        self.assertEqual(prepared.input_dict["hybrid_restore_slot_ids"].item(), 1)

    def test_same_request_suffix_uses_active_gdn_carry(self):
        store = _store()
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=HybridAPCSlotAllocator(num_slots=4),
            cache_salt="tenant-a",
            model_revision="rev-a",
        )
        first = bridge.prepare_request(
            request_id="req-live",
            input_dict={
                "input_ids": torch.arange(128, dtype=torch.int32).unsqueeze(0)
            },
            attention_hit_len=0,
            cumulative_hashes_by_prefix_len={128: "h128"},
        )
        bridge.commit_prefill(first)
        bridge.finish_request("req-live")
        _SCHEDULER_PATCH.authorize_hybrid_apc_prefix_read(
            first.commit_key,
            request_id="req-live",
        )

        same_request = bridge.prepare_suffix_only_request(
            request_id="req-live",
            input_dict={
                "input_ids": torch.arange(128, 192, dtype=torch.int32).unsqueeze(0)
            },
            attention_hit_len=128,
            request_prefix_len=192,
        )

        self.assertIsNotNone(same_request)
        self.assertEqual(same_request.input_dict["computed_context_lens"].item(), 128)
        self.assertEqual(
            same_request.input_dict["hybrid_restore_prefix_lens"].item(),
            128,
        )
        self.assertEqual(same_request.input_dict["hybrid_restore_mask"].item(), 0)
        self.assertEqual(same_request.input_dict["hybrid_restore_slot_ids"].item(), 0)

        _SCHEDULER_PATCH.authorize_hybrid_apc_prefix_read(
            first.commit_key,
            request_id="req-other",
        )
        other_request = bridge.prepare_suffix_only_request(
            request_id="req-other",
            input_dict={
                "input_ids": torch.arange(128, 192, dtype=torch.int32).unsqueeze(0)
            },
            attention_hit_len=128,
            request_prefix_len=192,
        )

        self.assertIsNotNone(other_request)
        self.assertEqual(other_request.input_dict["hybrid_restore_mask"].item(), 1)

    def test_same_request_full_prompt_slice_uses_active_gdn_carry(self):
        store = _store()
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=HybridAPCSlotAllocator(num_slots=4),
            cache_salt="tenant-a",
            model_revision="rev-a",
        )
        first = bridge.prepare_request(
            request_id="req-live-full",
            input_dict={
                "input_ids": torch.arange(128, dtype=torch.int32).unsqueeze(0)
            },
            attention_hit_len=0,
            cumulative_hashes_by_prefix_len={128: "h128"},
        )
        bridge.commit_prefill(first)
        bridge.finish_request("req-live-full")

        same_request = bridge.prepare_request(
            request_id="req-live-full",
            input_dict={
                "input_ids": torch.arange(192, dtype=torch.int32).unsqueeze(0)
            },
            attention_hit_len=128,
            request_prefix_len=192,
            cumulative_hashes_by_prefix_len={128: "h128", 192: "h192"},
        )

        self.assertEqual(
            same_request.input_dict["input_ids"].shape,
            torch.Size([1, 64]),
        )
        self.assertEqual(same_request.input_dict["computed_context_lens"].item(), 128)
        self.assertEqual(
            same_request.input_dict["hybrid_restore_prefix_lens"].item(),
            128,
        )
        self.assertEqual(same_request.input_dict["hybrid_restore_mask"].item(), 0)

        other_request = bridge.prepare_request(
            request_id="req-other-full",
            input_dict={
                "input_ids": torch.arange(192, dtype=torch.int32).unsqueeze(0)
            },
            attention_hit_len=128,
            request_prefix_len=192,
            cumulative_hashes_by_prefix_len={128: "h128", 192: "h192"},
        )

        self.assertEqual(other_request.input_dict["hybrid_restore_mask"].item(), 1)

    def test_bridge_suffix_only_restore_uses_checkpoint_attention_block_refs(self):
        store = _store()
        key, _checkpoint = _insert(
            store,
            256,
            prefix_hash="h256",
            gdn_checkpoint_slot=1,
            attention_block_refs=(4, 5),
        )
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=HybridAPCSlotAllocator(num_slots=3),
            cache_salt="tenant-a",
            model_revision="rev-a",
        )
        _SCHEDULER_PATCH.authorize_hybrid_apc_prefix_read(
            key,
            request_id="req-suffix-blocks",
        )

        prepared = bridge.prepare_suffix_only_request(
            request_id="req-suffix-blocks",
            input_dict={
                "input_ids": torch.arange(256, 272, dtype=torch.int32).unsqueeze(0),
                "block_table": torch.tensor([[99]], dtype=torch.int32),
                "rotary_position_id": torch.arange(
                    16,
                    dtype=torch.int32,
                ).unsqueeze(0),
                "rotary_position_ids": torch.arange(
                    16,
                    dtype=torch.int32,
                ).view(1, 1, 16).expand(3, 1, 16),
            },
            attention_hit_len=256,
            request_prefix_len=272,
        )

        self.assertIsNotNone(prepared)
        self.assertTrue(
            torch.equal(
                prepared.input_dict["block_table"],
                torch.tensor([[4, 5, 99]], dtype=torch.int32),
            )
        )
        expected_positions = torch.arange(256, 272, dtype=torch.int32)
        self.assertTrue(
            torch.equal(
                prepared.input_dict["rotary_position_id"],
                expected_positions.unsqueeze(0),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared.input_dict["rotary_position_ids"],
                expected_positions.view(1, 1, 16).expand(3, 1, 16),
            )
        )

    def test_bridge_suffix_only_restore_replaces_prefix_block_refs(self):
        store = _store()
        key, _checkpoint = _insert(
            store,
            256,
            prefix_hash="h256",
            gdn_checkpoint_slot=1,
            attention_block_refs=(4,),
        )
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=HybridAPCSlotAllocator(num_slots=3),
            cache_salt="tenant-a",
            model_revision="rev-a",
        )
        _SCHEDULER_PATCH.authorize_hybrid_apc_prefix_read(
            key,
            request_id="req-suffix-blocks",
        )

        prepared = bridge.prepare_suffix_only_request(
            request_id="req-suffix-blocks",
            input_dict={
                "input_ids": torch.arange(256, 272, dtype=torch.int32).unsqueeze(0),
                "block_table": torch.tensor([[99, 8]], dtype=torch.int32),
            },
            attention_hit_len=256,
            request_prefix_len=272,
        )

        self.assertIsNotNone(prepared)
        self.assertTrue(
            torch.equal(
                prepared.input_dict["block_table"],
                torch.tensor([[4, 8]], dtype=torch.int32),
            )
        )

    def test_bridge_suffix_only_restore_can_commit_boundary_checkpoint(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=3)
        input_ids = torch.arange(256, dtype=torch.int32).unsqueeze(0)
        hashes = build_cumulative_prefix_hashes(input_ids, block_size=128)
        restored_key, _checkpoint = _insert(
            store,
            128,
            prefix_hash=hashes[128],
            gdn_checkpoint_slot=2,
        )
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )
        _SCHEDULER_PATCH.authorize_hybrid_apc_prefix_read(
            restored_key,
            request_id="req-suffix-commit",
        )

        prepared = bridge.prepare_suffix_only_request(
            request_id="req-suffix-commit",
            input_dict={
                "input_ids": torch.arange(128, 256, dtype=torch.int32).unsqueeze(0)
            },
            attention_hit_len=128,
            request_prefix_len=256,
            cumulative_hashes_by_prefix_len=hashes,
            attention_block_refs_by_prefix_len={256: (8, 9)},
        )

        self.assertIsNotNone(prepared)
        self.assertEqual(prepared.plan.checkpoint_key, restored_key)
        self.assertEqual(prepared.commit_prefix_len, 256)
        self.assertEqual(prepared.commit_slot, 0)
        self.assertEqual(prepared.input_dict["hybrid_commit_mask"].item(), 1)
        self.assertEqual(prepared.input_dict["hybrid_commit_slot_ids"].item(), 0)
        committed = bridge.commit_prefill(prepared)
        self.assertIsNotNone(committed)
        self.assertEqual(committed.attention_block_refs, (8, 9))
        self.assertIsNotNone(store.lookup(prepared.commit_key))

    def test_bridge_suffix_only_commit_combines_restored_and_suffix_refs(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=3)
        input_ids = torch.arange(256, dtype=torch.int32).unsqueeze(0)
        hashes = build_cumulative_prefix_hashes(input_ids, block_size=128)
        restored_key, _checkpoint = _insert(
            store,
            128,
            prefix_hash=hashes[128],
            attention_block_refs=(7,),
            gdn_checkpoint_slot=2,
        )
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
            require_attention_block_refs=True,
        )
        _SCHEDULER_PATCH.authorize_hybrid_apc_prefix_read(
            restored_key,
            request_id="req-suffix-offset-refs",
        )

        prepared = bridge.prepare_suffix_only_request(
            request_id="req-suffix-offset-refs",
            input_dict={
                "input_ids": torch.arange(128, 256, dtype=torch.int32).unsqueeze(0)
            },
            attention_hit_len=128,
            request_prefix_len=256,
            cumulative_hashes_by_prefix_len=hashes,
            attention_block_refs_by_prefix_len={128: (9,)},
        )

        self.assertIsNotNone(prepared)
        self.assertEqual(prepared.attention_block_refs, (7, 9))
        committed = bridge.commit_prefill(prepared)
        self.assertIsNotNone(committed)
        self.assertEqual(committed.attention_block_refs, (7, 9))

    def test_bridge_suffix_only_restore_rejects_ambiguous_prefix_len(self):
        store = _store()
        _insert(store, 128, prefix_hash="h128-a", gdn_checkpoint_slot=0)
        _insert(store, 128, prefix_hash="h128-b", gdn_checkpoint_slot=1)
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=HybridAPCSlotAllocator(num_slots=2),
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_ALLOW_UNHASHED_SINGLE_PREFIX_RESTORE": "1"},
        ):
            with self.assertRaisesRegex(ValueError, "ambiguous unhashed"):
                bridge.prepare_suffix_only_request(
                    request_id="req-ambiguous",
                    input_dict={
                        "input_ids": torch.arange(128, 256, dtype=torch.int32)
                        .unsqueeze(0)
                    },
                    attention_hit_len=128,
                    request_prefix_len=256,
                )

    def test_bridge_does_not_commit_mid_prompt_checkpoint_boundary(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=2)
        input_ids = torch.arange(192, dtype=torch.int32).unsqueeze(0)
        hashes = build_cumulative_prefix_hashes(input_ids[:, :128], block_size=128)
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        prepared = bridge.prepare_request(
            request_id="req-mid-boundary",
            input_dict={"input_ids": input_ids},
            attention_hit_len=0,
            cumulative_hashes_by_prefix_len=hashes,
        )

        self.assertEqual(prepared.commit_prefix_len, 128)
        self.assertIsNone(prepared.commit_slot)
        self.assertEqual(prepared.input_dict["hybrid_commit_mask"].item(), 0)
        self.assertEqual(allocator.reserved_slots, ())
        self.assertEqual(allocator.free_slots, (0, 1))
        self.assertIsNone(bridge.commit_prefill(prepared))

    def test_bridge_can_require_scheduler_prefix_hashes(self):
        bridge = HybridAPCSchedulerBridge(
            store=_store(),
            slot_allocator=HybridAPCSlotAllocator(num_slots=2),
            cache_salt="tenant-a",
            model_revision="rev-a",
            allow_local_hash_fallback=False,
        )

        with self.assertRaisesRegex(ValueError, "requires vLLM cumulative prefix hashes"):
            bridge.prepare_request(
                request_id="req-strict",
                input_dict={"input_ids": torch.arange(128, dtype=torch.int32).unsqueeze(0)},
                attention_hit_len=0,
            )

    def test_bridge_can_require_attention_refs_on_commit(self):
        store = _store()
        input_ids = torch.arange(128, dtype=torch.int32).unsqueeze(0)
        hashes = build_cumulative_prefix_hashes(input_ids, block_size=128)
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=HybridAPCSlotAllocator(num_slots=2),
            cache_salt="tenant-a",
            model_revision="rev-a",
            allow_local_hash_fallback=False,
            require_attention_block_refs=True,
        )

        prepared = bridge.prepare_request(
            request_id="req-refs",
            input_dict={"input_ids": input_ids},
            attention_hit_len=0,
            cumulative_hashes_by_prefix_len=hashes,
        )

        with self.assertRaisesRegex(ValueError, "requires real attention block refs"):
            bridge.commit_prefill(prepared)

        committed = bridge.commit_prefill(prepared, attention_block_refs=(31,))

        self.assertEqual(committed.attention_block_refs, (31,))

    def test_bridge_salt_mismatch_does_not_restore_slot_zero(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=2)
        input_ids = torch.arange(128, dtype=torch.int32).unsqueeze(0)
        hashes = build_cumulative_prefix_hashes(input_ids, block_size=128)
        _insert(store, 128, prefix_hash=hashes[128], cache_salt="tenant-a")
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-b",
            model_revision="rev-a",
            reject_unbacked_attention_hits=False,
        )

        prepared = bridge.prepare_request(
            request_id="req-salt",
            input_dict={"input_ids": input_ids},
            attention_hit_len=128,
            cumulative_hashes_by_prefix_len=hashes,
        )

        self.assertIsNone(prepared.plan.checkpoint_key)
        self.assertEqual(prepared.plan.restore_checkpoint_prefix_len, 0)
        self.assertTrue(
            torch.equal(
                prepared.input_dict["hybrid_restore_slot_ids"],
                torch.tensor([0], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared.input_dict["hybrid_restore_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )

    def test_bridge_env_can_disable_restore_and_commit(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=2)
        input_ids = torch.arange(128, dtype=torch.int32).unsqueeze(0)
        hashes = build_cumulative_prefix_hashes(input_ids, block_size=128)
        restored_key, _checkpoint = _insert(
            store,
            128,
            prefix_hash=hashes[128],
            gdn_checkpoint_slot=1,
        )
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        with patch.dict(
            os.environ,
            {
                "QWEN36_DISABLE_HYBRID_GDN_RESTORE": "1",
                "QWEN36_DISABLE_HYBRID_GDN_COMMIT": "1",
            },
        ):
            prepared = bridge.prepare_request(
                request_id="req-disabled",
                input_dict={"input_ids": input_ids},
                attention_hit_len=128,
                cumulative_hashes_by_prefix_len=hashes,
            )

        self.assertIsNone(prepared.plan.checkpoint_key)
        self.assertIsNone(prepared.commit_slot)
        self.assertEqual(store.lookup(restored_key).refcount, 0)
        self.assertEqual(allocator.reserved_slots, ())
        self.assertTrue(torch.equal(prepared.input_dict["input_ids"], input_ids))
        self.assertEqual(prepared.input_dict["hybrid_restore_mask"].item(), 0)
        self.assertEqual(prepared.input_dict["hybrid_commit_mask"].item(), 0)
        self.assertTrue(
            torch.equal(
                prepared.input_dict["hybrid_restore_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )

    def test_bridge_skips_commit_when_checkpoint_already_exists(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=2)
        input_ids = torch.arange(128, dtype=torch.int32).unsqueeze(0)
        hashes = build_cumulative_prefix_hashes(input_ids, block_size=128)
        _insert(store, 128, prefix_hash=hashes[128], gdn_checkpoint_slot=1)
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        prepared = bridge.prepare_request(
            request_id="req-existing",
            input_dict={"input_ids": input_ids},
            attention_hit_len=128,
            cumulative_hashes_by_prefix_len=hashes,
        )

        self.assertIsNone(prepared.commit_slot)
        self.assertEqual(prepared.input_dict["hybrid_commit_mask"].item(), 0)
        self.assertEqual(allocator.free_slots, (0, 1))
        self.assertIsNone(bridge.commit_prefill(prepared))

    def test_bridge_finish_releases_uncommitted_reserved_slot(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=2)
        input_ids = torch.arange(128, dtype=torch.int32).unsqueeze(0)
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        prepared = bridge.prepare_request(
            request_id="req-no-commit",
            input_dict={"input_ids": input_ids},
            attention_hit_len=0,
        )

        self.assertEqual(prepared.commit_slot, 0)
        self.assertEqual(allocator.reserved_slots, (0,))
        finished = bridge.finish_request("req-no-commit")

        self.assertEqual(finished.state, "FINISHED")
        self.assertEqual(allocator.reserved_slots, ())
        self.assertEqual(allocator.free_slots, (1, 0))

    def test_prepare_with_request_record_keeps_lifecycle_on_original_dict(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=2)
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
            require_attention_block_refs=True,
        )
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
                pad_token_id=0,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_ids = torch.arange(128, dtype=torch.int32).unsqueeze(0)
        hashes = build_cumulative_prefix_hashes(input_ids, block_size=128)
        original_input = {
            "input_ids": input_ids,
            "hybrid_request_records": (
                {
                    "request_id": "req-record",
                    "vllm_attention_hit_len": 0,
                    "request_prefix_len": 128,
                    "cumulative_hashes_by_prefix_len": hashes,
                    "attention_block_refs_by_prefix_len": {128: (11,)},
                },
            ),
        }

        prepared_inputs = prepare_hybrid_apc_request_for_execution(
            model,
            original_input,
        )

        self.assertIsNot(prepared_inputs, original_input)
        self.assertIn("_hybrid_apc_prepared", original_input)
        self.assertEqual(allocator.reserved_slots, (0,))

        finish_hybrid_apc_request(original_input)

        self.assertEqual(allocator.reserved_slots, ())
        self.assertEqual(allocator.committed_slots, (0,))

    def test_scheduler_metadata_carries_prompt_tokens_only(self):
        scheduler = types.SimpleNamespace(
            cache_config=types.SimpleNamespace(block_size=128),
        )
        request = types.SimpleNamespace(
            prompt_token_ids=[11, 12],
            all_token_ids=[11, 12, 13],
            num_tokens=3,
            block_hashes=[],
        )

        metadata = _SCHEDULER_PATCH._scheduler_request_metadata(
            scheduler,
            request,
            num_computed_tokens=2,
        )

        self.assertEqual(metadata["request_prefix_len"], 2)
        self.assertEqual(metadata["full_input_ids"], (11, 12))

    def test_scheduler_metadata_omits_full_tokens_for_cold_chunk(self):
        scheduler = types.SimpleNamespace(
            cache_config=types.SimpleNamespace(block_size=128),
        )
        request = types.SimpleNamespace(
            prompt_token_ids=[11, 12, 13],
            all_token_ids=[11, 12, 13],
            num_tokens=3,
            block_hashes=[],
        )

        metadata = _SCHEDULER_PATCH._scheduler_request_metadata(
            scheduler,
            request,
            num_computed_tokens=0,
        )

        self.assertNotIn("full_input_ids", metadata)

    def test_scheduler_request_records_preserve_full_input_ids(self):
        scheduler_output = types.SimpleNamespace(
            num_scheduled_tokens={"req-a": 16},
        )
        setattr(
            scheduler_output,
            _SCHEDULER_PATCH._SCHEDULER_OUTPUT_METADATA_ATTR,
            {
                "req-a": {
                    "request_prefix_len": 144,
                    "full_input_ids": tuple(range(144)),
                    "vllm_attention_hit_len": 128,
                },
            },
        )
        model_input = types.SimpleNamespace(request_ids=("req-a",))

        records = _SCHEDULER_PATCH._hybrid_apc_request_records_from_model_input(
            model_input,
            scheduler_output,
        )

        self.assertIsNotNone(records)
        self.assertEqual(records[0]["full_input_ids"], tuple(range(144)))
        self.assertEqual(records[0]["active_suffix_len"], 16)

    def test_prepare_with_request_record_full_input_ids_restores_suffix(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=2)
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
            require_attention_block_refs=True,
        )
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
                pad_token_id=0,
            ),
            hybrid_apc_bridge=bridge,
        )
        full_input_ids = torch.arange(144, dtype=torch.int32).unsqueeze(0)
        suffix_input_ids = full_input_ids[:, 128:144]
        hashes = build_cumulative_prefix_hashes(full_input_ids, block_size=128)
        _insert(
            store,
            128,
            prefix_hash=hashes[128],
            attention_block_refs=(11,),
            gdn_checkpoint_slot=0,
        )
        original_input = {
            "input_ids": suffix_input_ids,
            "hybrid_request_records": (
                {
                    "request_id": "req-record-full-ids",
                    "vllm_attention_hit_len": 128,
                    "request_prefix_len": 144,
                    "full_input_ids": tuple(int(item) for item in full_input_ids[0]),
                    "cumulative_hashes_by_prefix_len": hashes,
                    "attention_block_refs_by_prefix_len": {128: (11,)},
                    "active_suffix_len": 16,
                },
            ),
        }

        prepared_inputs = prepare_hybrid_apc_request_for_execution(
            model,
            original_input,
        )

        self.assertTrue(
            torch.equal(prepared_inputs["input_ids"], suffix_input_ids),
        )
        self.assertEqual(prepared_inputs["computed_context_lens"].item(), 128)
        self.assertEqual(prepared_inputs["full_context_lens"].item(), 144)
        self.assertEqual(prepared_inputs["num_queries"].item(), 16)

    def test_bridge_evicts_lru_checkpoint_before_reserving_when_slots_full(self):
        store = _store(max_checkpoints=2)
        allocator = HybridAPCSlotAllocator(num_slots=2)
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        committed_keys = []
        for request_index in range(3):
            input_ids = (
                torch.arange(128, dtype=torch.int32).unsqueeze(0)
                + request_index * 1000
            )
            hashes = build_cumulative_prefix_hashes(input_ids, block_size=128)
            prepared = bridge.prepare_request(
                request_id=f"req-{request_index}",
                input_dict={"input_ids": input_ids},
                attention_hit_len=0,
                cumulative_hashes_by_prefix_len=hashes,
            )
            self.assertIsNotNone(prepared.commit_slot)
            bridge.commit_prefill(prepared)
            bridge.finish_request(prepared.request_id)
            committed_keys.append(prepared.commit_key)

        self.assertIsNone(store.lookup(committed_keys[0]))
        self.assertIsNotNone(store.lookup(committed_keys[1]))
        self.assertIsNotNone(store.lookup(committed_keys[2]))
        self.assertEqual(len(allocator.committed_slots), 2)
        self.assertEqual(allocator.reserved_slots, ())
        self.assertEqual(allocator.free_slots, ())

    def test_store_releases_old_slot_when_replacing_same_key(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=2)
        store.set_checkpoint_slot_releaser(allocator.release_committed)

        key = store.make_key(
            cumulative_prefix_hash="same-prefix",
            prefix_len=128,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        first_slot = allocator.reserve()
        first = store.insert(
            key=key,
            attention_block_refs=(1,),
            gdn_checkpoint_slot=first_slot,
        )
        allocator.mark_committed(first.gdn_checkpoint_slot)

        second_slot = allocator.reserve()
        second = store.insert(
            key=key,
            attention_block_refs=(2,),
            gdn_checkpoint_slot=second_slot,
        )
        allocator.mark_committed(second.gdn_checkpoint_slot)

        self.assertEqual(store.lookup(key).gdn_checkpoint_slot, second_slot)
        self.assertEqual(allocator.committed_slots, (second_slot,))
        self.assertEqual(allocator.free_slots, (first_slot,))

    def test_attention_block_eviction_unpublishes_scheduler_checkpoint(self):
        store = _store()
        key, _checkpoint = _insert(
            store,
            128,
            attention_block_refs=(7,),
            gdn_checkpoint_slot=0,
        )
        _SCHEDULER_PATCH.register_hybrid_apc_gdn_checkpoint(key)
        self.assertTrue(_SCHEDULER_PATCH.unregister_hybrid_apc_gdn_checkpoint(key))

        _SCHEDULER_PATCH.register_hybrid_apc_gdn_checkpoint(key)
        self.assertEqual(store.on_attention_block_evicted(7), [key])
        self.assertFalse(_SCHEDULER_PATCH.unregister_hybrid_apc_gdn_checkpoint(key))


if __name__ == "__main__":
    unittest.main()
