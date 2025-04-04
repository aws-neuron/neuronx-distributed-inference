"""
Unit tests for multinomial non-deterministic sampling.
"""

import unittest

import torch
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace import parallel_model_trace

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.modules.generation.sampling import (
    Sampler,
    prepare_sampling_params,
)


class TestSampler(unittest.TestCase):
    def get_sampling_model(self):
        # NeuronConfig for on-device execution
        on_device_sampling_config_kwargs = {
            "do_sample": True,
            "dynamic": True,
            "deterministic": False,
            "global_topk": 10,
        }

        config_kwargs = {
            "batch_size": 1,
            "seq_len": 1,
            "on_cpu": False,
            "on_device_sampling_config": OnDeviceSamplingConfig(**on_device_sampling_config_kwargs),
        }

        neuron_config = NeuronConfig(**config_kwargs)

        sampler = Sampler(neuron_config)

        return sampler, {}

    def test_sampler_output_on_device(self):
        torch.random.manual_seed(5)

        batch_size = 1
        vocab_size = 1000
        token_logits = torch.rand(batch_size, vocab_size)
        sampling_params = prepare_sampling_params(
            batch_size=batch_size, top_k=[5], top_p=[0.9], temperature=[1.0]
        )

        # compile the forward method for Neuron
        compiled_forward = parallel_model_trace(
            self.get_sampling_model,
            (token_logits, sampling_params),
            tp_degree=1,
            compiler_args="--auto-cast=none --optlevel=1 --enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1",
            compiler_workdir="/tmp/torch_top_k_non_deterministic/",
        )

        # perform sampling using the compiled forward method
        output = compiled_forward(token_logits, sampling_params)

        self.assertEqual(output.shape, (batch_size,))

        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()

        print("test_sampler_output_on_device passed")

    def test_sampler_output_cpu(self):
        torch.random.manual_seed(5)

        # NeuronConfig for CPU execution
        # NOTE: the current implementation
        on_device_sampling_config_kwargs = {
            "top_k": 1,
            "deterministic": False,
        }

        config_kwargs = {
            "batch_size": 1,
            "seq_len": 1,
            "on_cpu": True,
            "on_device_sampling_config": OnDeviceSamplingConfig(**on_device_sampling_config_kwargs),
        }

        neuron_config = NeuronConfig(**config_kwargs)

        sampler = Sampler(neuron_config, do_sample=True)
        batch_size = 1
        vocab_size = 1000
        token_logits = torch.rand(batch_size, vocab_size)
        sampling_params = prepare_sampling_params(
            batch_size=batch_size, top_k=[5], top_p=[0.9], temperature=[1.0]
        )

        # perform sampling
        output = sampler.forward(token_logits, sampling_params)

        self.assertEqual(output.shape, (batch_size,))

        print("test_sampler_output_cpu passed")


if __name__ == "__main__":
    unittest.main()
