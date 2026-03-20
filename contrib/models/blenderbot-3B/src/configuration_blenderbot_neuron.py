"""
Blenderbot NeuronX Configuration

Maps HuggingFace BlenderbotConfig fields to the NeuronX InferenceConfig interface.
Based on: transformers/src/transformers/models/blenderbot/configuration_blenderbot.py
"""

import json
import os

import torch
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig


class BlenderbotNeuronConfig(NeuronConfig):
    """NeuronConfig subclass for Blenderbot encoder-decoder model."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BlenderbotInferenceConfig(InferenceConfig):
    """InferenceConfig for Blenderbot that bridges HF config fields to framework expectations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_neuron_config_cls(cls):
        return BlenderbotNeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, neuron_config=None, **kwargs):
        """Load config from HF pretrained model directory."""
        config_file = os.path.join(model_path, "config.json")
        with open(config_file, "r") as f:
            hf_config = json.load(f)

        if neuron_config is None:
            neuron_config = BlenderbotNeuronConfig()

        def _load(self):
            for k, v in hf_config.items():
                setattr(self, k, v)

        return cls(neuron_config=neuron_config, load_config=_load, **kwargs)

    def add_derived_config(self):
        # CRITICAL: HF sets num_hidden_layers = encoder_layers (2), but decoder has 24 layers.
        # Force num_hidden_layers = decoder_layers for the decoder port.
        self.num_hidden_layers = getattr(self, 'decoder_layers', 24)
        self.hidden_size = getattr(self, 'd_model', 2560)
        self.num_attention_heads = getattr(self, 'decoder_attention_heads', 32)
        self.num_key_value_heads = getattr(self, 'decoder_attention_heads', 32)
        self.intermediate_size = getattr(self, 'decoder_ffn_dim', 10240)
        self.head_dim = self.hidden_size // self.num_attention_heads
        if not hasattr(self, "tie_word_embeddings"):
            self.tie_word_embeddings = True
        self.num_cores_per_group = 1
