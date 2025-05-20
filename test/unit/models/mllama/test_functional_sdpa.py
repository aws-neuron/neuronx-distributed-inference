import itertools
from typing import List

import torch
import math
from neuronx_distributed_inference.modules.attention.utils import neuron_scaled_dot_product_attention
from .test_utils import logger, setup_debug_env, trace_nxd_model

import torch.nn.functional as F

TORCH_DTYPE = torch.float32


class TorchFunctionalSDPA(torch.nn.Module):
    def __init__(self, max_num_chunks, torch_dtype) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, q, k, v, attn_mask):
        res = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        return res


class NXD_SDPA(torch.nn.Module):
    def __init__(self, max_num_chunks, torch_dtype) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, q, k, v, attn_mask):
        return neuron_scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)


def get_example_inputs():
    Q = torch.zeros([1, 1, 128, 128], dtype=TORCH_DTYPE)
    K = torch.zeros([1, 1, 6404, 128], dtype=TORCH_DTYPE)
    V = torch.zeros([1, 1, 6404, 128], dtype=TORCH_DTYPE)
    xattn_mask = torch.zeros([1, 1, 128, 6404], dtype=TORCH_DTYPE) #(torch.rand([1, 1, 128, 6404]) > 0.5).to(dtype=TORCH_DTYPE)

    return Q, K, V, xattn_mask


def test_llama_mm_cross_attention_mask():
    setup_debug_env()

    init_args = dict(
        max_num_chunks=4,
        torch_dtype=TORCH_DTYPE,
    )

    cpu_model_meta = TorchFunctionalSDPA(**init_args)

    example_inputs = get_example_inputs()
   
    # this following line works:
    nxd_atten_calculator_model_cls = NXD_SDPA
    neuron_model = trace_nxd_model(
        nxd_atten_calculator_model_cls, example_inputs, tp_degree=1, **init_args
    )

    # Test across multiple configurations
    
    # Get Meta's output on CPU
    for i in range(10):
        new_example_inputs = get_example_inputs()
        atten_output_cpu = cpu_model_meta(*new_example_inputs)

        # Get NxD's output on CPU
        atten_output_nxd = neuron_model(*new_example_inputs)

        assert torch.allclose(atten_output_cpu, atten_output_nxd)
    logger.info("Correctness test passing on CPU.")



if __name__ == "__main__":
    test_llama_mm_cross_attention_mask()
