import logging
import os
import pytest
import tempfile

import torch
import torch.nn as nn

from transformers.models.pixtral.modeling_pixtral import PixtralVisionConfig, PixtralAttention, PixtralRotaryEmbedding
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import build_module, build_cpu_model
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig as SmplConfig
from neuronx_distributed_inference.models.pixtral.modeling_pixtral import PixtralInferenceConfig
from neuronx_distributed_inference.models.pixtral.modeling_pixtral_vision import NeuronPixtralImageAttention
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from test_config import get_pixtral_config

TEXT_TP_DEGREE = 64
VISION_TP_DEGREE = 16
WORLD_SIZE = 64
BATCH_SIZE = 1
CP_DEGREE = 1
TEXT_SEQ_LENGTH = 10 * 1024
VISION_SEQ_LENGTH = 10 * 1024
DTYPE = torch.float16
BUCKETS = [2*1024, 4*1024, 10*1024]
MAX_CONTEXT_LENGTH = 10*1024
MAX_PATCHES_PER_IMAGE = 4096 # Pixtral supports 1024x1024 size image, which is 4096 token.
RTOL=1.6e-3

# We need to increase SCRATCHPAD_PAGE_SIZE to support 16K sequence lenghts
os.environ['NEURON_SCRATCHPAD_PAGE_SIZE'] = '1024'

# Set random seed for reproducibility
set_random_seed(0)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PixtralAttentionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = PixtralAttention(config=config)
        self.rotary_emb = PixtralRotaryEmbedding(config)

    def forward(self, hidden_states, attention_mask, position_ids):
        # Compute position embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        # Run attention and return all outputs
        outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings
        )

        return outputs[0]

class NeuronPixtralImageAttentionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = NeuronPixtralImageAttention(config=config)
        
    def forward(self, hidden_states, attention_mask, position_ids):
        outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return outputs[0]


def check_results(test_name, actual_output, expected_output, plot_outputs=False, rtol=1e-5):
    print("-" * 20)
    print(f"Test result of {test_name}:")
    print("actual_output shape:", actual_output.shape)
    print("expected_output shape:", expected_output.shape)
    passed, _ = check_accuracy_embeddings(
            actual_output, expected_output, plot_outputs=plot_outputs, rtol=rtol, atol=1e-5
        )
    assert(passed)
    print("-" * 20)


@pytest.mark.parametrize(
        "text_tp_degree, vision_tp_degree, world_size, text_seq_length, vision_seq_length, text_buckets, vision_buckets, dtype, batch_size, max_context_length, rtol",
        [(TEXT_TP_DEGREE, VISION_TP_DEGREE, WORLD_SIZE, TEXT_SEQ_LENGTH, VISION_SEQ_LENGTH, BUCKETS, BUCKETS, DTYPE, BATCH_SIZE, MAX_CONTEXT_LENGTH, RTOL)]
)
def test_attention(text_tp_degree, vision_tp_degree, world_size, text_seq_length, vision_seq_length, text_buckets, vision_buckets, dtype, batch_size, max_context_length, rtol):
    logger.info("Running PixtralImageAttention test ...")

    # Create config using get_pixtral_config
    pixtral_config = get_pixtral_config(
        dtype=dtype,
        text_tp_degree=text_tp_degree,
        vision_tp_degree=vision_tp_degree,
        world_size=world_size,
        text_seq_length=text_seq_length,
        vision_seq_len=vision_seq_length,
        text_buckets=text_buckets,
        vision_buckets=vision_buckets,
    )

    # Create configs
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/config_4layer.json"
    model_config = PixtralVisionConfig.from_pretrained(
        config_path,
        _attn_implementation="eager",  # Necessary when initialized config without model
    )

    #############################################
    # Context/Prefill Phase
    #############################################
    logger.info("Testing context/prefill phase...")

    # Create context phase test inputs
    context_hidden_states = (torch.randn((
        batch_size,
        max_context_length,
        pixtral_config.vision_config.hidden_size,
        )) * 0.05).to(dtype)

    # Context phase masks and positions
    context_attention_mask = torch.zeros((
        batch_size,
        1,
        max_context_length,
        max_context_length,
    ), dtype=torch.int32)

    for i in range(0, max_context_length, MAX_PATCHES_PER_IMAGE):
        context_attention_mask[:,:,i:i+MAX_PATCHES_PER_IMAGE, :] = 1
    
    #chunking images
    context_position_ids = [torch.arange(0, min(MAX_PATCHES_PER_IMAGE, max_context_length-i), dtype=torch.int32) for i in range(0, max_context_length, MAX_PATCHES_PER_IMAGE)]
    context_position_ids = torch.cat(context_position_ids).repeat(batch_size)

    # Build CPU model and save random checkpoint
    model_tempdir = tempfile.TemporaryDirectory()
    model_path_temp = model_tempdir.name
    cpu_model, ckpt_path = build_cpu_model(PixtralAttentionModule, model_config, dtype=dtype, checkpoint_dir=model_path_temp)

    # Run CPU inference for context phase
    logger.info("Running inference on CPU model - context phase")
    with torch.no_grad():
        cpu_output = cpu_model(
            hidden_states=context_hidden_states,
            attention_mask=context_attention_mask,
            position_ids=context_position_ids,
        )

    # Create example inputs tuple for Neuron model - context phase
    context_example_inputs = [(
        torch.ones_like(context_hidden_states),
        torch.ones_like(context_attention_mask),
        torch.ones_like(context_position_ids)
    )]

    # Build and trace Neuron model for context phase
    context_neuron_model = build_module(
        module_cls=NeuronPixtralImageAttentionModule,
        example_inputs=context_example_inputs,
        module_init_kwargs={"config": pixtral_config.vision_config},
        tp_degree=vision_tp_degree,
        checkpoint_path=ckpt_path,
        logical_nc_config=2,
    )

    # Run Neuron inference for context phase
    logger.info("Running inference on Neuron model - context phase")
    neuron_context_output = context_neuron_model(
        context_hidden_states,
        context_attention_mask,
        context_position_ids
    )

    # Check context phase results
    check_results("context_phase", neuron_context_output, cpu_output, plot_outputs=True, rtol=rtol)


if __name__ == "__main__":
    test_attention(TEXT_TP_DEGREE, VISION_TP_DEGREE, WORLD_SIZE, TEXT_SEQ_LENGTH, VISION_SEQ_LENGTH, BUCKETS, BUCKETS, DTYPE, BATCH_SIZE, MAX_CONTEXT_LENGTH, RTOL)