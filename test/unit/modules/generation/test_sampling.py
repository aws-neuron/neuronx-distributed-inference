# Standard Library
import unittest

import pytest

# Third Party
import torch
import torch_neuronx
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace import parallel_model_trace

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.modules.generation.sampling import (
    Sampler,
    cumsum,
    prepare_sampling_params,
)


class TestSampling(unittest.TestCase):
    def __init__(self, parent_args):
        super().__init__(parent_args)
        self.sampling_config = OnDeviceSamplingConfig()
        self.sampling_config.do_sample = True
        self.sampling_config.dynamic = False
        self.sampling_config.deterministic = True
        self.sampling_config.global_topk = 50
        self.sampling_config.topk = 1

    def update_generation_configs(self, dynamic, deterministic, global_top_k):
        self.sampling_config.dynamic = dynamic
        self.sampling_config.deterministic = deterministic
        self.sampling_config.global_topk = global_top_k

    def get_sampling_model(self, on_cpu=False):
        # run_on_cpu should be true for testing porpuses only

        neuron_kwargs = {}

        config_kwargs = {
            "dynamic": self.sampling_config.dynamic,
            "deterministic": self.sampling_config.deterministic,
            "global_topk": self.sampling_config.global_topk,
            "do_sample": self.sampling_config.do_sample,
        }
        config = OnDeviceSamplingConfig(**config_kwargs)
        neuron_kwargs["on_device_sampling_config"] = config
        neuron_config = NeuronConfig(**neuron_kwargs)

        neuron_config.on_cpu = True if on_cpu else False

        model = Sampler(neuron_config)
        return model, {}

    def get_neuron_sampling_model(self):
        return self.get_sampling_model(on_cpu=False)

    def get_static_test_cases(self):
        return [
            # vocab_size, seq_len, batch_size, top_k, top_p, temperature, dynamic, deterministic, global_top_k
            # (1000, 1, 1, [1], [1.0], [1.0], False, False, 256),  # greedy batch 1 # currently failing for non deterministic with miss match against cpu
            # (1000, 1, 2, [1], [1.0], [1.0], False, False, 256),  # greedy batch 2 # currently failing for non deterministic with miss match against cpu
            # (1000, 1, 8, [1], [1.0], [1.0], False, False, 256),  # greedy batch 8 # currently failing for non deterministic with miss match against cpu
            (1000, 1, 1, [5], [0.5], [0.9], False, True, 256),  # mulinomial batch 1
            (1000, 1, 2, [5], [0.9], [0.9], False, True, 256),  # mulinomial batch 2
            (1000, 1, 8, [5], [0.5], [0.5], False, True, 256),  # mulinomial batch 8
            (1000, 5, 8, [5], [0.5], [0.5], False, True, 256),  # mulinomial batch 8, spec len 5
        ]

    def test_static_sampling(self):
        torch.random.manual_seed(5)
        test_cases = self.get_static_test_cases()
        for tc in test_cases:
            (
                vocab_size,
                seq_len,
                batch_size,
                top_k,
                top_p,
                temperature,
                dynamic,
                deterministic,
                global_top_k,
            ) = tc
            self.update_generation_configs(
                dynamic=dynamic, deterministic=deterministic, global_top_k=global_top_k
            )
            x = torch.rand(vocab_size)
            if seq_len > 1:
                x = x.broadcast_to(batch_size, seq_len, vocab_size)
            else:
                x = x.broadcast_to(batch_size, vocab_size)
            # sample on cpu
            cpu_sampler, _ = self.get_sampling_model(on_cpu=True)
            sampling_params = prepare_sampling_params(
                batch_size=batch_size, top_k=top_k, top_p=top_p, temperature=temperature
            )
            cpu_output = cpu_sampler(x, sampling_params)
            # sample on Neuron
            compiler_args="--auto-cast=none  --optlevel=1 --enable-saturate-infinity" \
                            " --enable-mixed-precision-accumulation --model-type transformer -O1" \
                            " --internal-hlo2tensorizer-options='--verify-hlo=true'"
            neuron_sampler = parallel_model_trace(
                self.get_neuron_sampling_model,
                (x, sampling_params),
                tp_degree=1,
                compiler_args=compiler_args,
                compiler_workdir="/tmp/torch_top_k/",
            )
            neuron_output = neuron_sampler(x, sampling_params)
            assert torch.equal(
                cpu_output, neuron_output
            ), f"failed test case: {tc} \n \
            cpu_output: {cpu_output}, neuron_output: {neuron_output}"
            # Reset groups
            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()

    def get_dynamic_test_cases(self):
        return [
            # vocab_size, seq_len, batch_size, top_k, top_p, temperature, dynamic, deterministic, global_top_k
            (100, 1, 1, [5], [0.5], [0.9], True, True, 50),  # mulinomial batch 1
            (10000, 1, 2, [5], [0.9], [0.5], True, True, 256),  # mulinomial batch 2
            (10000, 1, 8, [5], [0.5], [0.5], True, True, 500),  # mulinomial batch 8
            (10000, 7, 8, [5], [0.5], [0.5], True, True, 500),  # mulinomial batch 8, spec len 7
            (
                32000,
                1,
                4,
                [1, 5, 1, 5],
                [0.5, 0.9, 0.5, 0.9],
                [0.5, 0.5, 0.8, 0.9],
                True,
                True,
                500,
            ),  # mulinomial batch 8 + per-batch-line
            (
                130000,
                1,
                4,
                [1, 5, 1, 5],
                [0.5, 0.9, 0.5, 0.9],
                [0.5, 0.5, 0.8, 0.9],
                True,
                True,
                500,
            ),  # mulinomial batch 8 + per-batch-line
        ]

    def test_dynamic_sampling(self):
        torch.random.manual_seed(5)
        test_cases = self.get_dynamic_test_cases()
        for tc in test_cases:
            (
                vocab_size,
                seq_len,
                batch_size,
                top_k,
                top_p,
                temperature,
                dynamic,
                deterministic,
                global_top_k,
            ) = tc
            self.update_generation_configs(
                dynamic=dynamic, deterministic=deterministic, global_top_k=global_top_k
            )
            x = torch.rand(vocab_size)
            if seq_len > 1:
                x = x.broadcast_to(batch_size, seq_len, vocab_size)
            else:
                x = x.broadcast_to(batch_size, vocab_size)
            # sample on cpu
            cpu_sampler, _ = self.get_sampling_model(on_cpu=True)
            sampling_params = prepare_sampling_params(
                batch_size=batch_size, top_k=top_k, top_p=top_p, temperature=temperature
            )
            cpu_output = cpu_sampler(x, sampling_params)
            # sample on Neuron
            compiler_args="--auto-cast=none  --optlevel=1 --enable-saturate-infinity" \
                            " --enable-mixed-precision-accumulation --model-type transformer -O1" \
                            " --internal-hlo2tensorizer-options='--verify-hlo=true'"
            neuron_sampler = parallel_model_trace(
                self.get_neuron_sampling_model,
                (x, sampling_params),
                tp_degree=1,
                compiler_args=compiler_args,
                compiler_workdir="/tmp/torch_top_k/",
            )
            neuron_output = neuron_sampler(x, sampling_params)
            assert torch.equal(
                cpu_output, neuron_output
            ), f"failed test case (top_k, top_p, temperature): {tc} \n \
            cpu_output: {cpu_output}, neuron_output: {neuron_output}"

            # Test new sampling params passed dynamically to the model
            dynamic_sampling_params = [
                # top_k, top_p, temperature
                ([1], [0.9], [0.5]),
                ([5], [0.9], [0.5]),
                ([10], [0.5], [0.9]),
                ([20], [0.9], [0.5]),
                ([50], [0.5], [0.9]),
            ]
            for dynamic_tc in dynamic_sampling_params:
                top_k, top_p, temperature = dynamic_tc
                sampling_params = prepare_sampling_params(
                    batch_size=batch_size, top_k=top_k, top_p=top_p, temperature=temperature
                )
                cpu_output = cpu_sampler(x, sampling_params)
                neuron_output = neuron_sampler(x, sampling_params)

                assert torch.equal(
                    cpu_output, neuron_output
                ), f"failed dynamic test case: {dynamic_tc} \n \
            cpu_output: {cpu_output}, neuron_output: {neuron_output}"

            # Reset groups
            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()


def get_sampler(topk, num_beams, on_device=True):
    neuron_kwargs = {}
    if on_device:
        neuron_kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig(top_k=topk)
    neuron_config = NeuronConfig(**neuron_kwargs)

    sampler_kwargs = {}
    if not on_device:
        sampler_kwargs["top_k"] = topk
    return Sampler(neuron_config, **sampler_kwargs)


def run_sampler_accuracy_test(batch_size, topk, num_beams=1):
    torch.manual_seed(0)
    torch.distributed.init_process_group("xla", init_method="pjrt://")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=32)
    vocab_size = 128
    logits = torch.rand(batch_size, vocab_size)
    device = xm.xla_device()
    logits_device = logits.to(device=device)

    neuron_sampler = get_sampler(topk, num_beams, on_device=True)
    cpu_sampler = get_sampler(topk, num_beams, on_device=False)
    print(neuron_sampler.sample(logits_device).cpu(), cpu_sampler.sample(logits))
    torch_neuronx.testing.assert_close(
        neuron_sampler.sample(logits_device).cpu(), cpu_sampler.sample(logits), check_dtype=False
    )
    # Reset groups
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("shape", [(13,), (11, 7), (5, 3, 2), (7, 5, 3, 2)])
def test_cumsum(dim, shape, dtype):
    if dim > len(shape) - 1:
        pytest.skip(f"Dim {dim} outside of shape {shape}")
    tensor_in = torch.rand(shape).to(dtype)
    expected_output = torch.cumsum(tensor_in, dim=dim)
    tensor_in_neuron = tensor_in.to(xm.xla_device())
    actual_output = cumsum(tensor_in_neuron, dim=dim, on_cpu=False)
    torch_neuronx.testing.assert_close(actual_output.cpu(), expected_output)


if __name__ == "__main__":
    unittest.main()
