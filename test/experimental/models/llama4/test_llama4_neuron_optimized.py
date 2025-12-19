import torch

from neuronx_distributed_inference.experimental.models.llama4.examples.llama4_neuron_optimized import main


# TODO add more tests

def test_main():
    # Validate execution is complete
    assert main() is None
