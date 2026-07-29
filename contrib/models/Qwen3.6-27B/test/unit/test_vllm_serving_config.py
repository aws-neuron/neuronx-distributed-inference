# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import os
import unittest


_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNNER_PATH = os.path.join(_CONTRIB_ROOT, "vllm", "run_offline_inference.py")


def _load_runner():
    spec = importlib.util.spec_from_file_location("qwen36_run_offline_inference", _RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    defaults = dict(
        cte_bucket=512,
        cte_buckets=None,
        cte_bucket_profile="single",
        context_encoding_bucket_pairs=None,
        seq_len=2048,
        tensor_parallel_size=4,
        max_num_seqs=1,
        ctx_batch_size=1,
        logical_nc_config=2,
        token_generation_buckets=None,
        token_generation_batches=None,
        async_mode=False,
        block_size=128,
        enable_prefix_caching=False,
        enable_hybrid_apc=False,
        enable_vllm_chunked_prefill=True,
        kernel_q_tile_size=128,
        kernel_kv_tile_size=1024,
        hybrid_gdn_recurrent_cache_dtype=None,
        gdn_recurrent_cache_dtype="float32",
        hybrid_gdn_conv_cache_dtype=None,
        gdn_conv_cache_dtype="bfloat16",
        gdn_checkpoint_interval=256,
        max_gdn_checkpoint_slots=8,
        hybrid_cache_mode="all",
        hybrid_cache_prefix_boundary_only=True,
        hybrid_cache_validate_exact=False,
        hybrid_apc_require_vllm_metadata=False,
        hybrid_apc_reject_unbacked_attention_hits=True,
        hybrid_apc_disable_unbacked_prefix_reads=False,
        hybrid_apc_enable_backed_prefix_reads=False,
        hybrid_apc_prefill_chunk_tokens=0,
        text_only_cte=True,
        compact_cte_attention_mask=True,
        cold_zero_conv_fast_path=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestVllmServingConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = _load_runner()

    def test_cte_bucket_list_is_sorted_unique_and_128_aligned(self):
        args = _args(cte_buckets=["512,128", "256", "256"])

        self.assertEqual(self.runner._cte_buckets(args), [128, 256, 512])

    def test_cte_bucket_rejects_non_128_aligned_value(self):
        with self.assertRaisesRegex(ValueError, "128-aligned"):
            self.runner._cte_buckets(_args(cte_buckets=["192"]))

    def test_short_profile_builds_dynamic_bucket_config(self):
        config = self.runner._override_config(_args(cte_bucket_profile="short"))
        neuron_config = config["override_neuron_config"]

        self.assertEqual(neuron_config["context_encoding_buckets"], [128, 256, 512, 1024])
        self.assertEqual(neuron_config["max_context_length"], 1024)
        self.assertTrue(neuron_config["enable_bucketing"])
        self.assertEqual(config["max_prompt_length"], 1024)

    def test_text_only_and_compact_mask_flags_are_forwarded(self):
        config = self.runner._override_config(
            _args(
                text_only_cte=False,
                compact_cte_attention_mask=False,
                cold_zero_conv_fast_path=True,
            )
        )

        self.assertFalse(config["use_text_only_cte_inputs"])
        self.assertFalse(config["use_compact_cte_attention_mask"])
        self.assertTrue(config["use_cold_zero_conv_fast_path"])

    def test_sparse_context_encoding_bucket_pairs_are_forwarded(self):
        config = self.runner._override_config(
            _args(
                enable_hybrid_apc=True,
                block_size=256,
                gdn_checkpoint_interval=256,
                context_encoding_bucket_pairs=["512:0,512:32768", "3072:131072"],
            )
        )

        self.assertEqual(
            config["override_neuron_config"]["context_encoding_bucket_pairs"],
            [[512, 0], [512, 32768], [3072, 131072]],
        )

    def test_sparse_context_pairs_keep_prefix_cte_contract_without_vllm_prefix_cache(self):
        config = self.runner._override_config(
            _args(
                enable_prefix_caching=False,
                enable_hybrid_apc=False,
                context_encoding_bucket_pairs=["512:0", "3072:16384"],
            )
        )
        neuron_config = config["override_neuron_config"]

        self.assertFalse(config["use_hybrid_apc_manager"])
        self.assertTrue(neuron_config["is_prefix_caching"])
        self.assertEqual(
            neuron_config["context_encoding_bucket_pairs"],
            [[512, 0], [3072, 16384]],
        )

    def test_hybrid_apc_requires_checkpoint_interval_equal_block_size(self):
        with self.assertRaisesRegex(ValueError, "gdn-checkpoint-interval"):
            self.runner._override_config(
                _args(
                    enable_hybrid_apc=True,
                    enable_prefix_caching=True,
                    block_size=128,
                    gdn_checkpoint_interval=256,
                )
            )

    def test_hybrid_apc_enables_prefix_caching_and_slots(self):
        args = _args(
            enable_hybrid_apc=True,
            enable_prefix_caching=False,
            block_size=256,
            gdn_checkpoint_interval=256,
            max_gdn_checkpoint_slots=3,
        )

        config = self.runner._override_config(args)

        self.assertTrue(args.enable_prefix_caching)
        self.assertTrue(config["use_hybrid_apc_manager"])
        self.assertEqual(config["max_gdn_checkpoint_slots"], 3)

    def test_hybrid_apc_rejects_bfloat16_recurrent_checkpoint_cache(self):
        with self.assertRaisesRegex(ValueError, "requires float32 recurrent GDN"):
            self.runner._override_config(
                _args(
                    enable_hybrid_apc=True,
                    block_size=256,
                    gdn_checkpoint_interval=256,
                    gdn_recurrent_cache_dtype="bfloat16",
                )
            )

    def test_hybrid_apc_can_require_vllm_metadata(self):
        config = self.runner._override_config(
            _args(
                enable_hybrid_apc=True,
                block_size=256,
                gdn_checkpoint_interval=256,
                hybrid_apc_require_vllm_metadata=True,
            )
        )

        self.assertTrue(config["hybrid_apc_require_vllm_metadata"])
        self.assertFalse(config["hybrid_apc_allow_local_hash_fallback"])
        self.assertTrue(config["hybrid_apc_require_attention_block_refs"])
        self.assertTrue(config["hybrid_apc_reject_unbacked_attention_hits"])

    def test_hybrid_apc_can_disable_unbacked_prefix_reads(self):
        config = self.runner._override_config(
            _args(
                enable_hybrid_apc=True,
                block_size=256,
                gdn_checkpoint_interval=256,
                hybrid_apc_disable_unbacked_prefix_reads=True,
            )
        )

        self.assertTrue(config["hybrid_apc_disable_unbacked_prefix_reads"])

    def test_hybrid_apc_can_enable_backed_prefix_reads(self):
        config = self.runner._override_config(
            _args(
                enable_hybrid_apc=True,
                block_size=256,
                gdn_checkpoint_interval=256,
                hybrid_apc_enable_backed_prefix_reads=True,
            )
        )

        self.assertTrue(config["hybrid_apc_enable_backed_prefix_reads"])

    def test_chunked_prefill_runtime_flags_are_forwarded(self):
        config = self.runner._override_config(_args(enable_vllm_chunked_prefill=True))

        self.assertTrue(config["use_qwen_hybrid_chunked_prefill"])
        self.assertTrue(config["use_qwen_hybrid_chunked_prefill_nki"])

    def test_grouped_prefill_defaults_to_largest_compiled_bucket(self):
        config = self.runner._override_config(
            _args(
                cte_buckets=["512,1024"],
                seq_len=2048,
                enable_hybrid_apc=True,
                enable_vllm_chunked_prefill=True,
                block_size=256,
                gdn_checkpoint_interval=256,
            )
        )

        self.assertEqual(config["qwen_prefill_group_size"], 1024)
        self.assertEqual(config["hybrid_apc_prefill_chunk_tokens"], 1024)

    def test_grouped_prefill_records_explicit_four_chunk_group(self):
        config = self.runner._override_config(
            _args(
                cte_buckets=["512,1024,2048"],
                seq_len=4096,
                enable_hybrid_apc=True,
                enable_vllm_chunked_prefill=True,
                block_size=256,
                gdn_checkpoint_interval=256,
                hybrid_apc_prefill_chunk_tokens=2048,
            )
        )

        self.assertEqual(config["qwen_prefill_group_size"], 2048)
        self.assertEqual(config["hybrid_apc_prefill_chunk_tokens"], 2048)

    def test_hybrid_apc_chunked_prefill_defaults_to_largest_aligned_bucket(self):
        args = _args(
            cte_buckets=["256,512"],
            enable_hybrid_apc=True,
            enable_vllm_chunked_prefill=True,
            block_size=256,
            gdn_checkpoint_interval=256,
        )

        self.assertEqual(
            self.runner._max_num_batched_tokens(
                args,
                self.runner._cte_buckets(args),
            ),
            512,
        )

    def test_hybrid_apc_chunked_prefill_uses_largest_checkpoint_aligned_bucket(self):
        args = _args(
            cte_buckets=["512,768,1536,3072"],
            seq_len=3072,
            enable_hybrid_apc=True,
            enable_vllm_chunked_prefill=True,
            block_size=256,
            gdn_checkpoint_interval=256,
        )

        self.assertEqual(
            self.runner._max_num_batched_tokens(
                args,
                self.runner._cte_buckets(args),
            ),
            3072,
        )

    def test_hybrid_apc_chunked_prefill_requires_checkpoint_aligned_cte_bucket(self):
        args = _args(
            cte_buckets=["384"],
            enable_hybrid_apc=True,
            enable_vllm_chunked_prefill=True,
            block_size=256,
            gdn_checkpoint_interval=256,
        )

        with self.assertRaisesRegex(ValueError, "multiple"):
            self.runner._max_num_batched_tokens(
                args,
                self.runner._cte_buckets(args),
            )

    def test_hybrid_apc_can_use_safe_non_power_of_two_prefill_chunk(self):
        args = _args(
            cte_buckets=["512,768,1536,3072"],
            seq_len=8192,
            enable_hybrid_apc=True,
            enable_vllm_chunked_prefill=True,
            block_size=256,
            gdn_checkpoint_interval=256,
            hybrid_apc_prefill_chunk_tokens=3072,
        )

        self.assertEqual(
            self.runner._max_num_batched_tokens(
                args,
                self.runner._cte_buckets(args),
            ),
            3072,
        )

    def test_hybrid_apc_can_use_explicit_larger_prefill_chunk(self):
        args = _args(
            cte_buckets=["256,512,1024,2048,4096,8192"],
            seq_len=8192,
            enable_hybrid_apc=True,
            enable_vllm_chunked_prefill=True,
            block_size=256,
            gdn_checkpoint_interval=256,
            hybrid_apc_prefill_chunk_tokens=8192,
        )

        self.assertEqual(
            self.runner._max_num_batched_tokens(
                args,
                self.runner._cte_buckets(args),
            ),
            8192,
        )

    def test_hybrid_apc_larger_prefill_chunk_must_be_compiled_bucket(self):
        args = _args(
            cte_buckets=["256,512,1024,2048,4096"],
            seq_len=8192,
            enable_hybrid_apc=True,
            enable_vllm_chunked_prefill=True,
            block_size=256,
            gdn_checkpoint_interval=256,
            hybrid_apc_prefill_chunk_tokens=8192,
        )

        with self.assertRaisesRegex(ValueError, "compiled CTE bucket"):
            self.runner._max_num_batched_tokens(
                args,
                self.runner._cte_buckets(args),
            )

    def test_hybrid_apc_larger_prefill_chunk_must_align_to_checkpoint(self):
        args = _args(
            cte_buckets=["256,384,512,1024"],
            seq_len=1024,
            enable_hybrid_apc=True,
            enable_vllm_chunked_prefill=True,
            block_size=256,
            gdn_checkpoint_interval=256,
            hybrid_apc_prefill_chunk_tokens=384,
        )

        with self.assertRaisesRegex(ValueError, "multiple"):
            self.runner._max_num_batched_tokens(
                args,
                self.runner._cte_buckets(args),
            )


if __name__ == "__main__":
    unittest.main()
