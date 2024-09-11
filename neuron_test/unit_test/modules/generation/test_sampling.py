import torch
import torch_xla.core.xla_model as xm

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.modules.generation.sampling import Sampler


def test_neuron_sampling_accuracy_bs1():
    run_sampler_accuracy_test(batch_size=1, topk=1)


def test_neuron_sampling_accuracy_bs4():
    run_sampler_accuracy_test(batch_size=4, topk=1)


def test_greedy_sampling_cpu():
    sampler = get_sampler(topk=1, num_beams=1, on_device=False)
    x = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sampled = sampler.sample(x)
    assert torch.equal(sampled, torch.tensor([0, 1, 2]))


def test_multinomial_sampling_cpu():
    """
    To test multinomial sampling, we fix the seed to 5 and compare
    to previously collected token ids (goldens for) [9, 71]
    """
    sampler = get_sampler(topk=3, num_beams=1, on_device=False)
    torch.random.manual_seed(5)
    x = torch.rand((2, 100))
    sampled = sampler.sample(x)
    assert torch.equal(sampled, torch.tensor([9, 71]))


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
    vocab_size = 128
    logits = torch.rand(batch_size, vocab_size)
    device = xm.xla_device()
    logits_device = logits.to(device=device)

    neuron_sampler = get_sampler(topk, num_beams, on_device=True)
    cpu_sampler = get_sampler(topk, num_beams, on_device=False)
    print(neuron_sampler.sample(logits_device).cpu(), cpu_sampler.sample(logits))
    torch.testing.assert_close(
        neuron_sampler.sample(logits_device).cpu(), cpu_sampler.sample(logits), check_dtype=False
    )
