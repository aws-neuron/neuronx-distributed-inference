"""
GPT-BigCode (SantaCoder) NeuronX Port

This module provides a NeuronX implementation of the GPT-BigCode model
(SantaCoder) for inference on AWS Trainium/Inferentia hardware.

Model Features:
- Multi-Query Attention (MQA): 1 KV head shared across all query heads
- LayerNorm normalization
- Absolute position embeddings (learned, not RoPE)
- GELU activation (tanh approximation)

Usage:
    from neuronx_port.modeling_gpt_bigcode import (
        NeuronGPTBigCodeForCausalLM,
        GPTBigCodeInferenceConfig,
    )
    from neuronx_distributed_inference.models.config import NeuronConfig
    from transformers import AutoTokenizer

    # Create config
    neuron_config = NeuronConfig(
        tp_degree=1,
        batch_size=1,
        seq_len=512,
        torch_dtype=torch.bfloat16,
    )
    config = GPTBigCodeInferenceConfig.from_pretrained(
        "/path/to/hf_model",
        neuron_config=neuron_config,
    )

    # Load model
    model = NeuronGPTBigCodeForCausalLM.from_pretrained(
        "/path/to/compiled_model",
        config=config,
    )

    # Generate
    tokenizer = AutoTokenizer.from_pretrained("/path/to/hf_model")
    inputs = tokenizer("def hello():", return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_new_tokens=50)
    print(tokenizer.decode(outputs[0]))

Version: v1
Port ID: 1188
"""

from .modeling_gpt_bigcode import (
    NeuronGPTBigCodeForCausalLM,
    NeuronGPTBigCodeModel,
    GPTBigCodeInferenceConfig,
    NeuronGPTBigCodeAttention,
    NeuronGPTBigCodeMLP,
    NeuronGPTBigCodeBlock,
    GPTBigCodeEmbedding,
)

__version__ = "1.0.0"
__all__ = [
    "NeuronGPTBigCodeForCausalLM",
    "NeuronGPTBigCodeModel",
    "GPTBigCodeInferenceConfig",
    "NeuronGPTBigCodeAttention",
    "NeuronGPTBigCodeMLP",
    "NeuronGPTBigCodeBlock",
    "GPTBigCodeEmbedding",
]
