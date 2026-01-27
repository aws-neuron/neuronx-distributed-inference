"""
Gemma-2B-IT NeuronX Port

This package contains the NeuronX implementation of Google's Gemma-2B-IT model
for AWS Trainium/Inferentia hardware.

Usage:
    from neuronx_port.modeling_gemma import (
        NeuronGemmaForCausalLM,
        GemmaInferenceConfig,
        GemmaNeuronConfig,
    )

    # Load compiled model
    model = NeuronGemmaForCausalLM.from_pretrained("./compiled_model_pt")

    # Or create new model for compilation
    neuron_config = GemmaNeuronConfig(tp_degree=1, batch_size=1, seq_len=512)
    config = GemmaInferenceConfig.from_pretrained("/path/to/hf_model", neuron_config=neuron_config)
    model = NeuronGemmaForCausalLM(config)
    model.load_weights("/path/to/hf_model")
    model.compile()
    model.save("./compiled_model_pt")

Key Classes:
    - NeuronGemmaForCausalLM: Main model class for inference
    - GemmaInferenceConfig: Configuration class with from_pretrained
    - GemmaNeuronConfig: Neuron-specific configuration
    - GemmaRMSNorm: Custom RMSNorm with (1 + weight) scaling
    - GemmaNormalizedEmbedding: Embedding with sqrt(hidden_size) normalization
"""

from .modeling_gemma import (
    NeuronGemmaForCausalLM,
    NeuronGemmaModel,
    GemmaInferenceConfig,
    GemmaNeuronConfig,
    GemmaRMSNorm,
    GemmaNormalizedEmbedding,
    NeuronGemmaAttention,
    NeuronGemmaMLP,
    NeuronGemmaDecoderLayer,
)

__all__ = [
    "NeuronGemmaForCausalLM",
    "NeuronGemmaModel",
    "GemmaInferenceConfig",
    "GemmaNeuronConfig",
    "GemmaRMSNorm",
    "GemmaNormalizedEmbedding",
    "NeuronGemmaAttention",
    "NeuronGemmaMLP",
    "NeuronGemmaDecoderLayer",
]

__version__ = "1.0.0"
__model__ = "gemma-2b-it"
