import torch
from torch_neuronx.testing import neuron_allclose

from neuronx_distributed_inference.utils.testing import build_function
from neuronx_distributed_inference.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding


def gpt_oss_rotary_embedding_test_fn(V, position_ids):
    rotary_embedding = GptOssRotaryEmbedding(dim=2880,
                                            base=150000.0,
                                            initial_context_length=4096,
                                            scaling_factor=32.0,
                                            ntk_alpha=1.0,
                                            ntk_beta=32.0)
    return rotary_embedding(V, position_ids)


def test_gpt_oss_rotary_embedding():
    batch_size = 4
    num_attn_heads = 1
    sequence_length = 2048
    head_size = 64

    example_inputs = [(
        torch.zeros((batch_size, num_attn_heads, sequence_length, head_size), dtype=torch.bfloat16),
        torch.zeros((batch_size, sequence_length), dtype=torch.int32),
    )]
    device_fn = build_function(gpt_oss_rotary_embedding_test_fn, example_inputs)

    V = torch.rand(batch_size, num_attn_heads, sequence_length, head_size, dtype=torch.bfloat16)
    position_ids = torch.tile(torch.arange(sequence_length, dtype=torch.int32), (batch_size, 1))

    cpu_cos_output, cpu_sin_output = gpt_oss_rotary_embedding_test_fn(V, position_ids)
    device_cos_output, device_sin_output = device_fn(V, position_ids)

    assert cpu_cos_output.shape == device_cos_output.shape
    assert cpu_cos_output.dtype == device_cos_output.dtype
    assert neuron_allclose(cpu_cos_output, device_cos_output)

    assert cpu_sin_output.shape == device_sin_output.shape
    assert cpu_sin_output.dtype == device_sin_output.dtype
    assert neuron_allclose(cpu_sin_output, device_sin_output)

if __name__ == '__main__':
    test_gpt_oss_rotary_embedding()

