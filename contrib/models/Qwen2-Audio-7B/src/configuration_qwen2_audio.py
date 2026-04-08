"""
Qwen2-Audio configuration for NeuronX Distributed Inference.

Contains:
  - Qwen2AudioEncoderNeuronConfig  — NeuronConfig for the audio encoder
  - Qwen2AudioMultimodalConfig     — Full model config (audio encoder + text LM)

NOTE ON NAMING: The NXDI framework base class (ImageToTextInferenceConfig)
requires fields named 'vision_config' internally. We expose 'audio_config'
as a public property that aliases it.
"""

import json, os
from typing import List, Type, Dict, Optional

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.image_to_text_model_base import ImageToTextInferenceConfig
from neuronx_distributed_inference.models.qwen2.modeling_qwen2 import Qwen2NeuronConfig


# ── Audio encoder NeuronConfig ─────────────────────────────────────────────────

class Qwen2AudioEncoderNeuronConfig(NeuronConfig):
    """NeuronConfig for the audio encoder — no fused_qkv, no KV cache."""
    def __init__(self, **kwargs):
        kwargs.setdefault("fused_qkv", False)
        super().__init__(**kwargs)


# ── Full multimodal config ─────────────────────────────────────────────────────

_TEXT_CONFIG_KEYS = [
    "hidden_size", "num_attention_heads", "num_hidden_layers",
    "num_key_value_heads", "pad_token_id", "vocab_size",
    "intermediate_size", "max_position_embeddings", "rms_norm_eps",
    "rope_theta", "hidden_act", "bos_token_id", "eos_token_id",
]


class Qwen2AudioMultimodalConfig(ImageToTextInferenceConfig):
    """
    Config for the full Qwen2-Audio model on Neuron.

    Public API:
        audio_config  — audio encoder configuration (property)
        text_config   — language model configuration
    """

    def __init__(self, text_neuron_config, audio_neuron_config,
                 fused_spec_config=None, load_config=None,
                 metadata: Optional[Dict] = None, **kwargs):
        # NXDI framework expects 'vision_neuron_config'
        super().__init__(
            text_neuron_config=text_neuron_config,
            vision_neuron_config=audio_neuron_config,
            fused_spec_config=fused_spec_config,
            load_config=load_config,
            metadata=metadata,
            **kwargs,
        )
        self._add_derived_config()

    @property
    def audio_config(self):
        """Audio encoder config (stored as vision_config by NXDI framework)."""
        return self.vision_config

    def _add_derived_config(self):
        self.num_cores_per_group = 1
        self.qkv_bias = True
        self.o_bias = False

        for key in _TEXT_CONFIG_KEYS:
            if hasattr(self, key):
                setattr(self.text_config, key, getattr(self, key))

        self.text_config.qkv_bias = True
        self.text_config.o_bias = False
        self.text_config.num_cores_per_group = 1
        self.text_config.output_attentions = False
        self.text_config.output_hidden_states = False
        self.pad_token_id = getattr(self.text_config, "pad_token_id", 151643)
        self.audio_token_index = getattr(self, "audio_token_index", 151646)

    def get_required_attributes(self) -> List[str]:
        return [
            "text_config", "vision_config",
            "text_config.hidden_size", "text_config.num_attention_heads",
            "text_config.num_hidden_layers", "text_config.num_key_value_heads",
            "text_config.pad_token_id", "text_config.vocab_size",
            "text_config.max_position_embeddings", "text_config.rope_theta",
            "text_config.rms_norm_eps", "text_config.hidden_act",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return Qwen2NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path, text_neuron_config=None,
                        audio_neuron_config=None, **kwargs):
        """Load from Qwen2-Audio config.json."""
        with open(os.path.join(model_path, "config.json")) as f:
            cd = json.load(f)

        tc = cd.get("text_config", {})
        ac = cd.get("audio_config", {})

        tc.setdefault("hidden_size", 4096)
        tc.setdefault("num_attention_heads", 32)
        tc.setdefault("num_hidden_layers", 32)
        tc.setdefault("num_key_value_heads", 32)
        tc.setdefault("intermediate_size", 11008)
        tc.setdefault("max_position_embeddings", 8192)
        tc.setdefault("rope_theta", 10000)
        tc.setdefault("rms_norm_eps", 1e-5)
        tc.setdefault("hidden_act", "silu")
        tc.setdefault("vocab_size", 156032)
        tc.setdefault("pad_token_id", 151643)
        tc.setdefault("bos_token_id", 151643)
        tc.setdefault("eos_token_id", 151645)

        ac.setdefault("d_model", 1280)
        ac.setdefault("encoder_layers", 32)
        ac.setdefault("encoder_attention_heads", 20)
        ac.setdefault("encoder_ffn_dim", 5120)
        ac.setdefault("num_mel_bins", 128)
        ac.setdefault("max_source_positions", 1500)

        kwargs.pop("neuron_config", None)
        return cls(
            text_neuron_config=text_neuron_config,
            audio_neuron_config=audio_neuron_config,
            text_config=tc,
            vision_config=ac,  # NXDI framework field name
            audio_token_index=cd.get("audio_token_index", 151646),
            _name_or_path=model_path,
            **kwargs,
        )


__all__ = ["Qwen2AudioEncoderNeuronConfig", "Qwen2AudioMultimodalConfig"]
