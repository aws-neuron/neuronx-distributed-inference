import torch
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel

# Test on NeuronBaseModel
def test_attn_mask_for_chunked_prefill_mostly_decode():
    neuron_config = NeuronConfig(is_chunked_prefill=True)
    config = InferenceConfig(
        neuron_config=neuron_config,
        vocab_size=1,
    )
    neuron_base = MockNeuronBaseModel(config, optimize_inference=False)

    actual = neuron_base._create_chunked_prefill_attn_mask(
        query_lens=torch.tensor([2,3,1,0]),
        key_lens=torch.tensor([4,5,4,0]),
        max_query_len=8,
        max_key_len=16,
    )

    expected = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 3 attend to 1st sequence
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 4 attend to 1st sequence
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 3 attend to 2nd sequence
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # At position 4 attend to 2nd sequence
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], # At position 5 attend to 2nd sequence
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], # At position 3 attend to 3rd sequence
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # padding
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # padding
        ]
    )

    assert torch.equal(expected, actual)


def test_attn_mask_for_chunked_prefill_mostly_prefill():
    neuron_config = NeuronConfig(is_chunked_prefill=True)
    config = InferenceConfig(
        neuron_config=neuron_config,
        vocab_size=1,
    )
    neuron_base = MockNeuronBaseModel(config, optimize_inference=False)

    actual = neuron_base._create_chunked_prefill_attn_mask(
        query_lens=torch.tensor([2,3,1]),
        key_lens=torch.tensor([2,3,4]),
        max_query_len=6,
        max_key_len=12,
    )

    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 1 attend to 1st sequence
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 2 attend to 1st sequence
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 1 attend to 2nd sequence
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # At position 2 attend to 2nd sequence
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], # At position 3 attend to 2nd sequence
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], # At position 4 attend to 3rd sequence
        ]
    )

    assert torch.equal(expected, actual)



class MockNeuronBaseModel(NeuronBaseModel):

    def setup_attr_for_model(self, config: InferenceConfig):
        pass
    
    def init_model(self, config: InferenceConfig):
        pass

    def init_inference_optimization(self, config: InferenceConfig):
        pass


# Test on NeuronBaseForCausalLM