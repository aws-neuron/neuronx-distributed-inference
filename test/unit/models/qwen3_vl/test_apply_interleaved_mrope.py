import torch
import pytest

from neuronx_distributed_inference.utils.testing import build_function
from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import NeuronQwen3VLRotaryEmbedding

import os
os.environ["BASE_COMPILE_WORK_DIR"] = "./compiler_workdir"

def hf_reference_impl(freqs, mrope_section):
    """Apply interleaved MRoPE to 3D rotary embeddings.
    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THWTHWTHW...TT], preserving frequency continuity.
    args:
        x: (3, bs, seq_len, head_dim // 2)
        mrope_section: (3,)
    returns:
        x_t: (bs, seq_len, head_dim // 2)
    """
    freqs_t = freqs[0]  # just overwrite the first dimension T
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t

torch.manual_seed(0)

@pytest.mark.parametrize("freqs, mrope_section", [
    (torch.rand([3, 1, 2765, 64]), torch.tensor([24, 20, 20])),
])
def test_original_vs_neuron(freqs, mrope_section):
    # Golden - Run HF impl on CPU
    # We need to clone the inputs as they will be modified in-place inside hf_reference_impl
    hf_output_ref = hf_reference_impl(freqs.clone(), mrope_section.clone())
    print("hf_output_ref:", hf_output_ref)

    # Run on CP
    cp_func = NeuronQwen3VLRotaryEmbedding.neuron_compute_freqs_mrope(freqs.clone(), mrope_section.clone())
    torch.testing.assert_close(hf_output_ref, cp_func, rtol=1e-2, atol=1e-2), "cp_func output does not match HF reference"
    print("cp_func vs hf_output_ref passed!")


    # Prepare zero inputs for Neuron run
    freqs_all_zeros = torch.zeros_like(freqs)
    mrope_section_all_zeros = torch.zeros_like(mrope_section)

    # Run on Neuron device
    neuron_func = build_function(
        func=NeuronQwen3VLRotaryEmbedding.neuron_compute_freqs_mrope,
        example_inputs=[(freqs_all_zeros, mrope_section_all_zeros)],
        tp_degree=2)
    
    neuron_output = neuron_func(freqs, mrope_section)
    torch.testing.assert_close(hf_output_ref, neuron_output, rtol=1e-2, atol=1e-2), "Neuron output does not match HF reference"
    print("neuron_output vs hf_output_ref passed!")