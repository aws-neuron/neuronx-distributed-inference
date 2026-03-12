import logging
import pytest
import time
import torch
from torch import nn
from typing import List

from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    apply_multimodal_rotary_pos_emb as hf_apply_multimodal_rotary_pos_emb
)

from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_text import (
    apply_multimodal_rotary_pos_emb as neuron_apply_multimodal_rotary_pos_emb
)

logger = logging.getLogger("Test")
logger.setLevel(logging.INFO)

def load_dummy_checkpoint():
    return {}

class TestModelBuilderInstance(BaseModelInstance):
    def __init__(self, module_cls, input_output_aliases, **module_init_kwargs):
        self.module_init_kwargs = module_init_kwargs
        super().__init__(module_cls, input_output_aliases)

    def load_module(self):
        self.module = self.module_cls(**self.module_init_kwargs)
        self.module.eval()

class MropeKernel(nn.Module):
    def __init__(self, mrope_section: List[int]):
        super().__init__()
        self.mrope_section = mrope_section

    def forward(self, q, k, cos, sin):
        return neuron_apply_multimodal_rotary_pos_emb(
            q, k, cos, sin, self.mrope_section
        )

def trace_nxd_model(
    model_class, example_inputs, checkpoint_loader, tp_degree=1, **model_init_kwargs
):
    model_builder = ModelBuilder(
        router=None,
        tp_degree=tp_degree,
        checkpoint_loader=checkpoint_loader,
        compiler_workdir=f"/tmp/compiler_workdir_tp{tp_degree}/",
    )
    logger.info("Initialized model builder")

    model_builder.add(
        key="test_nxd_model",
        model_instance=TestModelBuilderInstance(model_class, {}, **model_init_kwargs),
        example_inputs=[example_inputs],
        compiler_args="--enable-saturate-infinity --enable-mixed-precision-accumulation --auto-cast=none --model-type=transformer -O1",
    )
    logger.info("Added models. Starting trace.")

    start_time = time.time()
    traced_model = model_builder.trace()
    start_rank_tensor = torch.tensor([0], dtype=torch.int32, device="cpu")
    traced_model.nxd_model.initialize_with_saved_weights(start_rank_tensor)
    elapsed_time = time.time() - start_time
    logger.info(f"Done with trace in {elapsed_time}s!")
    return traced_model

class TestMultimodalRotaryPosEmb:
    
    @pytest.mark.parametrize("batch_size,seq_len,num_heads,head_dim", [
        (1, 1024, 28, 128),
        (4, 256, 28, 128),
        (8, 128, 28, 128),
    ])
    def test_apply_multimodal_rotary_pos_emb_on_cpu(self, batch_size, seq_len, num_heads, head_dim):
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        cos = torch.randn(3, 1, seq_len, head_dim)
        sin = torch.randn(3, 1, seq_len, head_dim)
        mrope_section = [16, 24, 24]
        
        hf_q_embed, hf_k_embed = hf_apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
        neuron_q_embed, neuron_k_embed = neuron_apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
        
        torch.testing.assert_close(hf_q_embed, neuron_q_embed, rtol=0, atol=0)
        torch.testing.assert_close(hf_k_embed, neuron_k_embed, rtol=0, atol=0)

    @pytest.mark.parametrize("batch_size,seq_len,num_heads,head_dim", [
        (1, 1024, 28, 128),
        (4, 256, 28, 128),
        (8, 128, 28, 128),
    ])
    def test_apply_multimodal_rotary_pos_emb_on_neuron(self, batch_size, seq_len, num_heads, head_dim):
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        cos = torch.randn(3, 1, seq_len, head_dim)
        sin = torch.randn(3, 1, seq_len, head_dim)
        mrope_section = [16, 24, 24]
        
        hf_q_embed, hf_k_embed = hf_apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)

        example_inputs = (q,k,cos,sin)

        neuron_model = trace_nxd_model(
            MropeKernel, example_inputs, load_dummy_checkpoint, tp_degree=1, mrope_section=mrope_section
        )

        neuron_q_embed, neuron_k_embed = neuron_model(q, k, cos, sin)
        
        torch.testing.assert_close(hf_q_embed, neuron_q_embed, rtol=0, atol=0)
        torch.testing.assert_close(hf_k_embed, neuron_k_embed, rtol=0, atol=0)
