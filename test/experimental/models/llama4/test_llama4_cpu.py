from neuronx_distributed_inference.experimental.models.llama4.examples.llama4_on_cpu import main

def test_main():
    # Validate execution is complete
    assert main() is None

