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
