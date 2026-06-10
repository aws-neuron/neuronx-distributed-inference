# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from neuronx_distributed_inference.modules.kvcache.hybrid_prefix_cache import (
    HybridPrefixCheckpointCache,
)


def _states(value):
    return {0: f"l0-{value}", 1: f"l1-{value}", 2: f"l2-{value}"}


class TestHybridPrefixCheckpointCache(unittest.TestCase):
    def test_reuses_deepest_cumulative_prefix_checkpoint(self):
        cache = HybridPrefixCheckpointCache(
            required_gdn_layers=[0, 1, 2],
            checkpoint_interval=256,
        )
        cache.put_checkpoint(
            cumulative_prefix_hash="h256",
            prefix_len=256,
            recurrent_states=_states("r256"),
            conv_states=_states("c256"),
        )
        key512 = cache.put_checkpoint(
            cumulative_prefix_hash="h512",
            prefix_len=512,
            recurrent_states=_states("r512"),
            conv_states=_states("c512"),
        )

        plan = cache.compute_reuse_plan(
            cumulative_hashes_by_prefix_len={
                256: "h256",
                512: "h512",
                768: "h768",
                1024: "h1024",
            },
            attention_hit_len=1024,
            request_prefix_len=1200,
        )

        self.assertEqual(plan.checkpoint_key, key512)
        self.assertEqual(plan.restore_checkpoint_prefix_len, 512)
        self.assertEqual(plan.residual_replay_len, 512)
        self.assertEqual(plan.suffix_len, 176)

    def test_missing_gdn_family_state_is_not_accepted(self):
        cache = HybridPrefixCheckpointCache(
            required_gdn_layers=[0, 1, 2],
            checkpoint_interval=256,
        )
        with self.assertRaisesRegex(ValueError, "every required GDN layer"):
            cache.put_checkpoint(
                cumulative_prefix_hash="h512",
                prefix_len=512,
                recurrent_states={0: "r0", 1: "r1", 2: "r2"},
                conv_states={0: "c0", 1: "c1"},
            )

    def test_hash_salt_and_revision_are_part_of_identity(self):
        cache = HybridPrefixCheckpointCache(
            required_gdn_layers=[0, 1, 2],
            checkpoint_interval=256,
        )
        cache.put_checkpoint(
            cumulative_prefix_hash="same-hash",
            prefix_len=256,
            recurrent_states=_states("r"),
            conv_states=_states("c"),
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        miss = cache.compute_reuse_plan(
            cumulative_hashes_by_prefix_len={256: "same-hash"},
            attention_hit_len=256,
            request_prefix_len=300,
            cache_salt="tenant-b",
            model_revision="rev-a",
        )
        hit = cache.compute_reuse_plan(
            cumulative_hashes_by_prefix_len={256: "same-hash"},
            attention_hit_len=256,
            request_prefix_len=300,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        self.assertIsNone(miss.checkpoint_key)
        self.assertIsNotNone(hit.checkpoint_key)

    def test_refcount_blocks_eviction(self):
        cache = HybridPrefixCheckpointCache(
            required_gdn_layers=[0, 1, 2],
            checkpoint_interval=256,
            max_checkpoints=2,
        )
        key1 = cache.put_checkpoint(
            cumulative_prefix_hash="h256",
            prefix_len=256,
            recurrent_states=_states("r256"),
            conv_states=_states("c256"),
        )
        key2 = cache.put_checkpoint(
            cumulative_prefix_hash="h512",
            prefix_len=512,
            recurrent_states=_states("r512"),
            conv_states=_states("c512"),
        )
        cache.inc_ref(key1)
        key3 = cache.put_checkpoint(
            cumulative_prefix_hash="h768",
            prefix_len=768,
            recurrent_states=_states("r768"),
            conv_states=_states("c768"),
        )

        self.assertIsNotNone(cache.get_checkpoint(key1))
        self.assertIsNone(cache.get_checkpoint(key2))
        self.assertIsNotNone(cache.get_checkpoint(key3))

    def test_checkpoint_length_must_align_to_interval(self):
        cache = HybridPrefixCheckpointCache(
            required_gdn_layers=[0, 1, 2],
            checkpoint_interval=256,
        )

        with self.assertRaisesRegex(ValueError, "checkpoint_interval"):
            cache.put_checkpoint(
                cumulative_prefix_hash="h300",
                prefix_len=300,
                recurrent_states=_states("r300"),
                conv_states=_states("c300"),
            )


if __name__ == "__main__":
    unittest.main()
