import torch
import pytest

from neuronx_distributed_inference.utils.testing import build_function
from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import NeuronQwen3VLTextModel
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
    generate_positions_from_mask,
    pad_positions,
    pad_vision_embeddings,
)
import os
os.environ["BASE_COMPILE_WORK_DIR"] = "./compiler_workdir"

def hf_reference_impl(
            hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

torch.manual_seed(0)

@pytest.mark.parametrize("hidden_dim, seq_length, num_vision_tokens, vision_start_idx", [
    (4096, 2048, 2000, 40),
    (4096, 2765, 2752, 13),
    (4096, 5120, 5100, 10),
])
def test_original_vs_neuron(hidden_dim, seq_length, num_vision_tokens, vision_start_idx):
    
    assert seq_length >= (num_vision_tokens + vision_start_idx), "seq_length must be >= num_vision_tokens + vision_start_idx"

    hidden_states = torch.rand([1, seq_length, hidden_dim])
    visual_embeds = torch.rand([num_vision_tokens, hidden_dim])
    visual_pos_masks = torch.zeros([1, seq_length], dtype=torch.bool)
    visual_pos_masks[0, vision_start_idx:(num_vision_tokens+vision_start_idx)] = True  # simulate 1 image with num_vision_tokens tokens

    # Golden - Run HF impl on CPU
    # We need to clone the inputs as they will be modified in-place inside hf_reference_impl
    hf_output_ref = hf_reference_impl(hidden_states.clone(), visual_pos_masks.clone(), visual_embeds.clone())

    # Prepare inputs for CP and Neuron
    vision_mask_positions = generate_positions_from_mask(visual_pos_masks.squeeze(0))
    
    # Pad vision embeddings and positions to match seq_length (hidden_states.shape[1])
    pad_limit = hidden_states.shape[1]
    pad_value = 0 # dummy vision embeddings 
    vision_mask_positions = pad_positions(vision_mask_positions, pad_limit, pad_value)
    visual_embeds_padded = pad_vision_embeddings(visual_embeds.unsqueeze(0), pad_limit)

    # Run on CP
    cp_func = NeuronQwen3VLTextModel.deepstack_process_xla(hidden_states, visual_embeds_padded, vision_mask_positions)
    torch.testing.assert_close(hf_output_ref, cp_func, rtol=1e-2, atol=1e-2), "cp_func output does not match HF reference"
    print("cp_func vs hf_output_ref passed!")

    vision_mask_positions_all_zeros = torch.zeros_like(vision_mask_positions)
    hidden_states_all_zeros = torch.zeros_like(hidden_states)
    visual_embeds_all_zeros = torch.zeros_like(visual_embeds_padded)

    # Run on Neuron device
    neuron_func = build_function(
        func=NeuronQwen3VLTextModel.deepstack_process_xla,
        example_inputs=[(hidden_states_all_zeros, visual_embeds_all_zeros, vision_mask_positions_all_zeros)],
        tp_degree=2)
    
    neuron_output = neuron_func(hidden_states, visual_embeds_padded, vision_mask_positions)
    torch.testing.assert_close(hf_output_ref, neuron_output, rtol=1e-2, atol=1e-2), "Neuron output does not match HF reference"
    print("neuron_output vs hf_output_ref passed!")