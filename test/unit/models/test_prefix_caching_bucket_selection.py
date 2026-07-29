import pytest
import torch

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    ModelWrapper,
)

class TestPrefixCachingBucketSelection:
    def setup_context_encoding(self):
        self.model_cls = NeuronBaseModel
        self.buckets = [
            [8, 0], [8, 8], [8, 16],
            [16, 0], [16, 8], [16, 16],
            [32, 0], [32, 8], [32, 16],
        ]
        self.config = InferenceConfig(
            neuron_config=NeuronConfig(
                is_prefix_caching=True,
                pa_block_size=4,
                buckets=self.buckets,
            ),
        )
        model_wrapper = ModelWrapper(config=self.config, model_cls=self.model_cls)
        model_wrapper.tag = CONTEXT_ENCODING_MODEL_TAG
        model_wrapper.async_mode = False
        return model_wrapper

    def setup_token_generation(self):
        self.model_cls = NeuronBaseModel
        self.buckets = [[1, 8], [1, 16], [1, 32]]
        self.config = InferenceConfig(
            neuron_config=NeuronConfig(
                is_prefix_caching=True,
                pa_block_size=4,
                buckets=self.buckets,
            ),
        )
        model_wrapper = ModelWrapper(config=self.config, model_cls=self.model_cls)
        model_wrapper.tag = TOKEN_GENERATION_MODEL_TAG
        model_wrapper.async_mode = False
        return model_wrapper

    @pytest.mark.parametrize(
        "inp_args, prefill_bucket, prefix_bucket",
        [
            # No Prefix
            [[torch.tensor(0)]*13 + [torch.tensor([[7]])] + [torch.tensor([[0]])], 8, 0,],
            [[torch.tensor(0)]*13 + [torch.tensor([[8]])] + [torch.tensor([[0]])], 8, 0,],
            [[torch.tensor(0)]*13 + [torch.tensor([[9]])] + [torch.tensor([[0]])], 16, 0,],
            [[torch.tensor(0)]*13 + [torch.tensor([[16]])] + [torch.tensor([[0]])], 16, 0,],
            [[torch.tensor(0)]*13 + [torch.tensor([[17]])] + [torch.tensor([[0]])], 32, 0,],
            [[torch.tensor(0)]*13 + [torch.tensor([[31]])] + [torch.tensor([[0]])], 32, 0,],

            # All Prefix moved to Prefill
            [[torch.tensor(0)]*13 + [torch.tensor([[4]])] + [torch.tensor([[4]])], 8, 0,],
            [[torch.tensor(0)]*13 + [torch.tensor([[9]])] + [torch.tensor([[4]])], 16, 0,],
            [[torch.tensor(0)]*13 + [torch.tensor([[17]])] + [torch.tensor([[8]])], 32, 0,],

            # Prefix = 8
            [[torch.tensor(0)]*13 + [torch.tensor([[7]])] + [torch.tensor([[4]])], 8, 8,],
            [[torch.tensor(0)]*13 + [torch.tensor([[4]])] + [torch.tensor([[12]])], 8, 8,],
            [[torch.tensor(0)]*13 + [torch.tensor([[10]])] + [torch.tensor([[8]])], 16, 8,],
            [[torch.tensor(0)]*13 + [torch.tensor([[10]])] + [torch.tensor([[12]])], 16, 8,],
            [[torch.tensor(0)]*13 + [torch.tensor([[17]])] + [torch.tensor([[16]])], 32, 8,],
            [[torch.tensor(0)]*13 + [torch.tensor([[30]])] + [torch.tensor([[4]])], 32, 8,],

            # Prefix = 16
            [[torch.tensor(0)]*13 + [torch.tensor([[7]])] + [torch.tensor([[12]])], 8, 16,],
            [[torch.tensor(0)]*13 + [torch.tensor([[4]])] + [torch.tensor([[16]])], 8, 16,],
            [[torch.tensor(0)]*13 + [torch.tensor([[10]])] + [torch.tensor([[16]])], 16, 16,],
            [[torch.tensor(0)]*13 + [torch.tensor([[10]])] + [torch.tensor([[20]])], 16, 16,],
            [[torch.tensor(0)]*13 + [torch.tensor([[17]])] + [torch.tensor([[24]])], 32, 16,],
            [[torch.tensor(0)]*13 + [torch.tensor([[30]])] + [torch.tensor([[12]])], 32, 16,],
                                                                                    
        ]
    )
    def test_cte_no_spec(self, inp_args, prefill_bucket, prefix_bucket):
        model_wrapper = self.setup_context_encoding()
        computed_prefill_bucket, computed_prefix_bucket = model_wrapper.get_target_2d_bucket_for_prefix_caching(*inp_args)
        assert computed_prefill_bucket == prefill_bucket
        assert computed_prefix_bucket == prefix_bucket

    def test_cte_sparse_grid_selects_next_compiled_pair(self):
        model_wrapper = self.setup_context_encoding()
        model_wrapper.neuron_config.buckets = [
            [512, 0],
            [512, 32768],
            [1536, 0],
            [1536, 65536],
            [3072, 0],
            [3072, 131072],
        ]
        inp_args = (
            [torch.tensor(0)] * 13
            + [torch.tensor([[512]])]
            + [torch.tensor([[131072]])]
        )

        computed_prefill_bucket, computed_prefix_bucket = (
            model_wrapper.get_target_2d_bucket_for_prefix_caching(*inp_args)
        )

        assert computed_prefill_bucket == 3072
        assert computed_prefix_bucket == 131072

    @pytest.mark.parametrize(
        "inp_args, prefix_bucket",
        [
            [[torch.tensor(0)]*13 + [torch.tensor([[1]])] + [torch.tensor([[7]])], 8,],
            [[torch.tensor(0)]*13 + [torch.tensor([[1]])] + [torch.tensor([[8]])], 16,],
            [[torch.tensor(0)]*13 + [torch.tensor([[1]])] + [torch.tensor([[15]])], 16,],
            [[torch.tensor(0)]*13 + [torch.tensor([[1]])] + [torch.tensor([[16]])], 32,],
            [[torch.tensor(0)]*13 + [torch.tensor([[1]])] + [torch.tensor([[31]])], 32,],
        ]
    )
    def test_tkg_no_spec(self, inp_args, prefix_bucket):
        model_wrapper = self.setup_token_generation()
        computed_prefill_bucket, computed_prefix_bucket = model_wrapper.get_target_2d_bucket_for_prefix_caching(*inp_args)
        assert computed_prefill_bucket == 1
        assert computed_prefix_bucket == prefix_bucket

    def test_tkg_bucket_selection_uses_decode_active_len_not_bad_num_queries(self):
        model_wrapper = self.setup_token_generation()
        inp_args = [
            torch.ones((1, 1), dtype=torch.int32),  # input_ids
            torch.ones((1, 15), dtype=torch.int32),  # attention_mask
            torch.tensor([[15]], dtype=torch.int32),  # position_ids
            torch.zeros((1,), dtype=torch.int32),  # seq_ids
            torch.ones((1, 3), dtype=torch.float32),  # sampling_params
            torch.empty(0),
            torch.zeros((1,), dtype=torch.int32),  # adapter_ids
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.zeros((1, 1), dtype=torch.int32),  # slot_mapping
            torch.zeros((1, 4), dtype=torch.int32),  # block_table
            torch.tensor([[15]], dtype=torch.int32),  # bad num_queries
            torch.tensor([[15]], dtype=torch.int32),  # computed_context_lens
        ]

        computed_prefill_bucket, computed_prefix_bucket = (
            model_wrapper.get_target_2d_bucket_for_prefix_caching(*inp_args)
        )

        assert computed_prefill_bucket == 1
        assert computed_prefix_bucket == 16

    def test_tkg_padding_rewrites_bad_num_queries_to_decode_active_len(self):
        model_wrapper = self.setup_token_generation()
        inp_args = [
            torch.ones((1, 1), dtype=torch.int32),  # input_ids
            torch.ones((1, 15), dtype=torch.int32),  # attention_mask
            torch.tensor([[15]], dtype=torch.int32),  # position_ids
            torch.zeros((1,), dtype=torch.int32),  # seq_ids
            torch.ones((1, 3), dtype=torch.float32),  # sampling_params
            torch.empty(0),
            torch.zeros((1,), dtype=torch.int32),  # adapter_ids
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.zeros((1, 1), dtype=torch.int32),  # slot_mapping
            torch.zeros((1, 4), dtype=torch.int32),  # block_table
            torch.tensor([[15]], dtype=torch.int32),  # bad num_queries
            torch.tensor([[15]], dtype=torch.int32),  # computed_context_lens
        ]

        padded_args = model_wrapper._pad_prefix_caching_inputs(*inp_args)

        assert torch.equal(padded_args[13], torch.tensor([[1]], dtype=torch.int32))
        assert torch.equal(padded_args[14], torch.tensor([[15]], dtype=torch.int32))
        assert padded_args[0].shape[-1] == 1
        assert padded_args[1].shape[-1] == 16

    def test_cte_hybrid_apc_restore_padding_keeps_suffix_and_prefix_bucket(self):
        model_wrapper = self.setup_context_encoding()
        model_wrapper.neuron_config.buckets = [
            [256, 0],
            [256, 256],
            [512, 0],
            [512, 256],
        ]
        model_wrapper.neuron_config.pa_block_size = 256

        suffix_len = 16
        restore_len = 256
        inp_args = [
            torch.arange(suffix_len, dtype=torch.int32).reshape(1, suffix_len),
            torch.ones((1, suffix_len), dtype=torch.int32),
            torch.arange(
                restore_len,
                restore_len + suffix_len,
                dtype=torch.int32,
            ).reshape(1, suffix_len),
            torch.zeros((1,), dtype=torch.int32),
            torch.ones((1, 3), dtype=torch.float32),
            torch.empty(0),
            torch.zeros((1,), dtype=torch.int32),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.arange(1792, 1792 + suffix_len, dtype=torch.int32).reshape(
                1,
                suffix_len,
            ),
            torch.arange(8, dtype=torch.int32).reshape(1, 8),
            torch.tensor([[suffix_len]], dtype=torch.int32),
            torch.tensor([[restore_len]], dtype=torch.int32),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.tensor([0], dtype=torch.int32),  # restore slot
            torch.tensor([1], dtype=torch.int32),  # restore mask
            torch.tensor([restore_len], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),  # commit slot
            torch.tensor([0], dtype=torch.int32),  # commit mask
        ]

        prefill_bucket, prefix_bucket = (
            model_wrapper.get_target_2d_bucket_for_prefix_caching(*inp_args)
        )
        padded_args = model_wrapper._pad_prefix_caching_inputs(*inp_args)

        assert int(prefill_bucket) == 256
        assert int(prefix_bucket) == 256
        assert padded_args[0].shape == (1, 256)
        assert padded_args[1].shape == (1, 256)
        assert padded_args[2].shape == (1, 256)
        assert torch.equal(padded_args[0][:, :suffix_len], inp_args[0])
        assert torch.equal(padded_args[1][:, :suffix_len], inp_args[1])
        assert torch.equal(padded_args[2][:, :suffix_len], inp_args[2])
        assert torch.equal(padded_args[11][:, :suffix_len], inp_args[11])
        assert torch.equal(
            padded_args[11][:, suffix_len:],
            torch.full((1, 256 - suffix_len), -1, dtype=torch.int32),
        )
        assert padded_args[12].shape == (1, 1)
        assert torch.equal(padded_args[12], torch.tensor([[0]], dtype=torch.int32))
        assert torch.equal(padded_args[13], torch.tensor([[suffix_len]], dtype=torch.int32))
        assert torch.equal(padded_args[14], torch.tensor([[restore_len]], dtype=torch.int32))

    def test_cte_suffix_only_continuation_keeps_prefix_bucket(self):
        model_wrapper = self.setup_context_encoding()
        model_wrapper.neuron_config.buckets = [
            [256, 0],
            [256, 256],
            [512, 0],
            [512, 256],
        ]
        model_wrapper.neuron_config.pa_block_size = 256

        suffix_len = 256
        prefix_len = 256
        inp_args = [
            torch.arange(suffix_len, dtype=torch.int32).reshape(1, suffix_len),
            torch.ones((1, suffix_len), dtype=torch.int32),
            torch.arange(
                prefix_len,
                prefix_len + suffix_len,
                dtype=torch.int32,
            ).reshape(1, suffix_len),
            torch.zeros((1,), dtype=torch.int32),
            torch.ones((1, 3), dtype=torch.float32),
            torch.empty(0),
            torch.zeros((1,), dtype=torch.int32),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.arange(512, 512 + suffix_len, dtype=torch.int32).reshape(
                1,
                suffix_len,
            ),
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([[suffix_len]], dtype=torch.int32),
            torch.tensor([[prefix_len]], dtype=torch.int32),
        ]

        prefill_bucket, prefix_bucket = (
            model_wrapper.get_target_2d_bucket_for_prefix_caching(*inp_args)
        )
        padded_args = model_wrapper._pad_prefix_caching_inputs(*inp_args)

        assert int(prefill_bucket) == 256
        assert int(prefix_bucket) == 256
        assert padded_args[0].shape == (1, 256)
        assert padded_args[1].shape == (1, 256)
        assert padded_args[2].shape == (1, 256)
        assert torch.equal(padded_args[0], inp_args[0])
        assert torch.equal(padded_args[1], torch.ones((1, 256), dtype=torch.int32))
        assert torch.equal(padded_args[2], inp_args[2])
        assert torch.equal(padded_args[11], inp_args[11])
        assert torch.equal(padded_args[12], torch.tensor([[0]], dtype=torch.int32))
        assert torch.equal(
            padded_args[13], torch.tensor([[suffix_len]], dtype=torch.int32)
        )
        assert torch.equal(
            padded_args[14], torch.tensor([[prefix_len]], dtype=torch.int32)
        )

    def test_cte_suffix_only_partial_continuation_does_not_left_pad_slots(self):
        model_wrapper = self.setup_context_encoding()
        model_wrapper.neuron_config.buckets = [
            [256, 0],
            [256, 256],
            [256, 512],
            [256, 1024],
        ]
        model_wrapper.neuron_config.pa_block_size = 256

        suffix_len = 48
        prefix_len = 768
        inp_args = [
            torch.arange(suffix_len, dtype=torch.int32).reshape(1, suffix_len),
            torch.ones((1, suffix_len), dtype=torch.int32),
            torch.arange(
                prefix_len,
                prefix_len + suffix_len,
                dtype=torch.int32,
            ).reshape(1, suffix_len),
            torch.zeros((1,), dtype=torch.int32),
            torch.ones((1, 3), dtype=torch.float32),
            torch.empty(0),
            torch.zeros((1,), dtype=torch.int32),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.arange(1024, 1024 + suffix_len, dtype=torch.int32).reshape(
                1,
                suffix_len,
            ),
            torch.tensor([[0, 1, 2]], dtype=torch.int32),
            torch.tensor([[suffix_len]], dtype=torch.int32),
            torch.tensor([[prefix_len]], dtype=torch.int32),
        ]

        prefill_bucket, prefix_bucket = (
            model_wrapper.get_target_2d_bucket_for_prefix_caching(*inp_args)
        )
        padded_args = model_wrapper._pad_prefix_caching_inputs(*inp_args)

        assert int(prefill_bucket) == 256
        assert int(prefix_bucket) == 1024
        assert torch.equal(padded_args[0][:, :suffix_len], inp_args[0])
        assert torch.equal(padded_args[2][:, :suffix_len], inp_args[2])
        assert torch.equal(padded_args[11][:, :suffix_len], inp_args[11])
        assert torch.equal(
            padded_args[11][:, suffix_len:],
            torch.full((1, 256 - suffix_len), -1, dtype=torch.int32),
        )
        assert padded_args[1].shape == (1, 1024)
        assert torch.equal(
            padded_args[1][:, :prefix_len],
            torch.ones((1, prefix_len), dtype=torch.int32),
        )
        assert torch.equal(
            padded_args[1][:, prefix_len:],
            torch.zeros((1, 1024 - prefix_len), dtype=torch.int32),
        )
        assert torch.equal(
            padded_args[12], torch.tensor([[0, 1, 2, 0]], dtype=torch.int32)
        )
        assert torch.equal(
            padded_args[13], torch.tensor([[suffix_len]], dtype=torch.int32)
        )
        assert torch.equal(
            padded_args[14], torch.tensor([[prefix_len]], dtype=torch.int32)
        )

    def test_segmented_cte_padding_fills_active_block_table_from_slots(self):
        model_wrapper = self.setup_context_encoding()
        model_wrapper.neuron_config.buckets = [
            [256, 1024],
        ]
        model_wrapper.neuron_config.pa_block_size = 256
        model_wrapper.neuron_config.max_context_length = 1024
        model_wrapper.neuron_config.prefix_cte_attention_backend = "segmented_cte"

        suffix_len = 48
        prefix_len = 768
        inp_args = [
            torch.arange(suffix_len, dtype=torch.int32).reshape(1, suffix_len),
            torch.ones((1, suffix_len), dtype=torch.int32),
            torch.arange(
                prefix_len,
                prefix_len + suffix_len,
                dtype=torch.int32,
            ).reshape(1, suffix_len),
            torch.zeros((1,), dtype=torch.int32),
            torch.ones((1, 3), dtype=torch.float32),
            torch.empty(0),
            torch.zeros((1,), dtype=torch.int32),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.arange(1024, 1024 + suffix_len, dtype=torch.int32).reshape(
                1,
                suffix_len,
            ),
            torch.tensor([[0, 1, 2, -1]], dtype=torch.int32),
            torch.tensor([[suffix_len]], dtype=torch.int32),
            torch.tensor([[prefix_len]], dtype=torch.int32),
        ]

        padded_args = model_wrapper._pad_prefix_caching_inputs(*inp_args)

        assert torch.equal(
            padded_args[12],
            torch.tensor([[0, 1, 2, 4]], dtype=torch.int32),
        )

    def test_segmented_cte_padding_fills_short_suffix_block_after_prefix_hit(self):
        model_wrapper = self.setup_context_encoding()
        model_wrapper.neuron_config.buckets = [
            [512, 512],
        ]
        model_wrapper.neuron_config.pa_block_size = 256
        model_wrapper.neuron_config.max_context_length = 1024
        model_wrapper.neuron_config.prefix_cte_attention_backend = "segmented_cte"

        suffix_len = 14
        prefix_len = 512
        suffix_physical_block = 9
        inp_args = [
            torch.arange(suffix_len, dtype=torch.int32).reshape(1, suffix_len),
            torch.ones((1, suffix_len), dtype=torch.int32),
            torch.arange(
                prefix_len,
                prefix_len + suffix_len,
                dtype=torch.int32,
            ).reshape(1, suffix_len),
            torch.zeros((1,), dtype=torch.int32),
            torch.ones((1, 3), dtype=torch.float32),
            torch.empty(0),
            torch.zeros((1,), dtype=torch.int32),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.arange(
                suffix_physical_block * 256,
                suffix_physical_block * 256 + suffix_len,
                dtype=torch.int32,
            ).reshape(1, suffix_len),
            torch.tensor([[0, 1, -1]], dtype=torch.int32),
            torch.tensor([[suffix_len]], dtype=torch.int32),
            torch.tensor([[prefix_len]], dtype=torch.int32),
        ]

        padded_args = model_wrapper._pad_prefix_caching_inputs(*inp_args)

        assert torch.equal(
            padded_args[12],
            torch.tensor([[0, 1, suffix_physical_block, 0]], dtype=torch.int32),
        )

    def test_segmented_cte_cold_cte2048_keeps_active_block_table_when_batched(self):
        model_wrapper = self.setup_context_encoding()
        model_wrapper.neuron_config.buckets = [
            [2048, 0],
        ]
        model_wrapper.neuron_config.pa_block_size = 256
        model_wrapper.neuron_config.max_context_length = 32768
        model_wrapper.neuron_config.prefix_cte_attention_backend = "segmented_cte"

        active_len = 2048
        input_ids = torch.arange(active_len, dtype=torch.int32).repeat(2, 1)
        attention_mask = torch.ones((2, active_len), dtype=torch.int32)
        position_ids = torch.arange(active_len, dtype=torch.int32).repeat(2, 1)
        slot_mapping = torch.stack(
            (
                torch.arange(0, active_len, dtype=torch.int32),
                torch.arange(4096, 4096 + active_len, dtype=torch.int32),
            )
        )
        inp_args = [
            input_ids,
            attention_mask,
            position_ids,
            torch.zeros((2,), dtype=torch.int32),
            torch.ones((2, 3), dtype=torch.float32),
            torch.empty(0),
            torch.zeros((2,), dtype=torch.int32),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            slot_mapping,
            torch.full((2, 8), -1, dtype=torch.int32),
            torch.full((2, 1), active_len, dtype=torch.int32),
            torch.zeros((2, 1), dtype=torch.int32),
        ]

        padded_args = model_wrapper._pad_prefix_caching_inputs(*inp_args)

        assert padded_args[12].shape == (2, 8)
        assert torch.equal(
            padded_args[12][0],
            torch.arange(8, dtype=torch.int32),
        )
        assert torch.equal(
            padded_args[12][1],
            torch.arange(16, 24, dtype=torch.int32),
        )

    def test_cte_batched_hybrid_apc_restore_padding_uses_full_attention_mask(self):
        model_wrapper = self.setup_context_encoding()
        model_wrapper.neuron_config.buckets = [
            [256, 0],
            [256, 4096],
        ]
        model_wrapper.neuron_config.pa_block_size = 256

        suffix_len = 16
        restore_len = 256
        attention_mask = torch.cat(
            [
                torch.ones((1, suffix_len), dtype=torch.int32),
                torch.cat(
                    [
                        torch.ones((1, 12), dtype=torch.int32),
                        torch.zeros((1, suffix_len - 12), dtype=torch.int32),
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        )
        inp_args = [
            torch.arange(2 * suffix_len, dtype=torch.int32).reshape(2, suffix_len),
            attention_mask,
            torch.arange(
                restore_len,
                restore_len + suffix_len,
                dtype=torch.int32,
            ).reshape(1, suffix_len).expand(2, -1),
            torch.arange(2, dtype=torch.int32),
            torch.ones((2, 3), dtype=torch.float32),
            torch.empty(0),
            torch.zeros((2,), dtype=torch.int32),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.arange(2 * suffix_len, dtype=torch.int32).reshape(2, suffix_len),
            torch.tensor([[8], [9]], dtype=torch.int32),
            torch.tensor([[suffix_len], [12]], dtype=torch.int32),
            torch.tensor([[restore_len], [restore_len]], dtype=torch.int32),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.tensor([0, 1], dtype=torch.int32),  # restore slots
            torch.tensor([1, 1], dtype=torch.int32),  # restore mask
            torch.tensor([restore_len, restore_len], dtype=torch.int32),
            torch.tensor([0, 0], dtype=torch.int32),  # commit slots
            torch.tensor([0, 0], dtype=torch.int32),  # commit mask
        ]

        prefill_bucket, prefix_bucket = (
            model_wrapper.get_target_2d_bucket_for_prefix_caching(*inp_args)
        )
        padded_args = model_wrapper._pad_prefix_caching_inputs(*inp_args)

        assert int(prefill_bucket) == 256
        assert int(prefix_bucket) == 4096
        assert padded_args[1].shape == (2, 4096)
        assert torch.equal(
            padded_args[1].sum(dim=1),
            torch.tensor([restore_len + suffix_len, restore_len + 12]),
        )
        assert torch.equal(
            padded_args[1][0, : restore_len + suffix_len],
            torch.ones((restore_len + suffix_len,), dtype=torch.int32),
        )
        assert torch.equal(
            padded_args[1][0, restore_len + suffix_len :],
            torch.zeros((4096 - restore_len - suffix_len,), dtype=torch.int32),
        )
        assert padded_args[12].shape == (2, 16)

    def test_cte_batched_hybrid_apc_restore_routes_mixed_warm_cold_to_compiled_shape(self):
        model_wrapper = self.setup_context_encoding()
        model_wrapper.neuron_config.buckets = [
            [256, 0],
            [256, 256],
            [256, 512],
            [512, 0],
            [512, 256],
            [512, 512],
        ]
        model_wrapper.neuron_config.pa_block_size = 256

        active_len = 272
        warm_suffix_len = 16
        restore_len = 256
        attention_mask = torch.zeros((2, active_len), dtype=torch.int32)
        attention_mask[0, :warm_suffix_len] = 1
        attention_mask[1, :active_len] = 1
        inp_args = [
            torch.arange(2 * active_len, dtype=torch.int32).reshape(2, active_len),
            attention_mask,
            torch.arange(active_len, dtype=torch.int32).reshape(1, active_len).expand(2, -1),
            torch.arange(2, dtype=torch.int32),
            torch.ones((2, 3), dtype=torch.float32),
            torch.empty(0),
            torch.zeros((2,), dtype=torch.int32),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.arange(2 * active_len, dtype=torch.int32).reshape(2, active_len),
            torch.tensor([[8, 9], [10, 11]], dtype=torch.int32),
            torch.tensor([[warm_suffix_len], [active_len]], dtype=torch.int32),
            torch.tensor([[restore_len], [0]], dtype=torch.int32),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.tensor([0, 0], dtype=torch.int32),  # restore slots
            torch.tensor([1, 0], dtype=torch.int32),  # restore mask
            torch.tensor([restore_len, 0], dtype=torch.int32),
            torch.tensor([1, 2], dtype=torch.int32),  # commit slots
            torch.tensor([1, 0], dtype=torch.int32),  # commit mask
        ]

        prefill_bucket, prefix_bucket = (
            model_wrapper.get_target_2d_bucket_for_prefix_caching(*inp_args)
        )
        padded_args = model_wrapper._pad_prefix_caching_inputs(*inp_args)

        assert int(prefill_bucket) == 512
        assert int(prefix_bucket) == 512
        assert padded_args[0].shape == (2, 512)
        assert padded_args[1].shape == (2, 512)
        assert padded_args[2].shape == (2, 512)
        assert padded_args[11].shape == (2, 512)
        assert padded_args[12].shape == (2, 2)
        assert torch.equal(
            padded_args[12],
            torch.tensor([[8, 0], [0, 0]], dtype=torch.int32),
        )
        assert torch.equal(
            padded_args[13],
            torch.tensor([[warm_suffix_len], [active_len]], dtype=torch.int32),
        )
        assert torch.equal(
            padded_args[14],
            torch.tensor([[restore_len], [0]], dtype=torch.int32),
        )
