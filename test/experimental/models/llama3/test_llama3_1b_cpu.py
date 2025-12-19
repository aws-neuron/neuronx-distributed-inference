from neuronx_distributed_inference.experimental.models.llama3.run_cpu import main

# TODO add more tests


def test_main():
    # Validate execution is complete
    assert main() is None
