# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NeuronX Distributed Inference implementation of Qwen3-ASR-1.7B.

This implements the full ASR pipeline:
1. Audio encoder (traced separately, Whisper-like architecture)
2. Text decoder (NeuronQwen3VL pattern with multimodal scatter)

Architecture:
- Audio encoder: 24-layer transformer, d_model=1024, Conv2D frontend
- Text decoder: 28-layer Qwen3, hidden_size=2048, GQA 16/8, QK-norm, mRoPE
- Pipeline: mel spectrogram -> encoder -> scatter into text embeddings -> autoregressive decode

The text decoder reuses NeuronQwen3VLForCausalLM (multimodal scatter mechanism)
with audio embeddings placed at audio_token positions.
"""

import copy
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl import (
    NeuronQwen3VLForCausalLM,
    Qwen3VLInferenceConfig,
    Qwen3VLNeuronConfig,
)
from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import (
    NeuronQwen3VLTextForCausalLM,
)
from neuronx_distributed_inference.models.application_base import (
    load_state_dict as nxdi_load_sd,
)
from neuronx_distributed_inference.models.image_to_text_model_base import (
    normalize_path,
)
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
    generate_positions_from_mask,
    pad_positions,
    pad_vision_embeddings,
)


# Special token IDs for Qwen3-ASR
AUDIO_PAD_ID = 151676  # Placeholder token for audio embeddings in text sequence
AUDIO_START_ID = 151669  # Marks beginning of audio segment
AUDIO_END_ID = 151670  # Marks end of audio segment
IM_START_ID = 151644  # <|im_start|>
IM_END_ID = 151645  # <|im_end|> (also used as EOS for generation)
EOS_ID = 151643  # End of sequence / pad token
ASR_TEXT_TOKEN_ID = (
    151704  # <asr_text> separator between language tag and transcription
)


def get_encoder_output_length(T_mel: int) -> int:
    """Compute number of encoder output tokens from mel frame count.

    The encoder uses chunked processing with Conv2D stride-2 frontend.
    Each 100 mel frames (1 second) produces 13 output tokens.

    Args:
        T_mel: Number of mel spectrogram frames (100 per second of audio)

    Returns:
        Number of encoder output tokens (audio embeddings)
    """
    input_lengths_leave = T_mel % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (T_mel // 100) * 13
    return output_lengths


class NeuronQwen3ASRForCausalLM(NeuronQwen3VLForCausalLM):
    """Qwen3-ASR text decoder on Neuron, using Qwen3-VL's multimodal scatter.

    This class handles:
    - Loading text decoder weights (excluding audio encoder weights)
    - Converting HuggingFace state dict to NxDI format
    - Prefill with audio embedding scatter at AUDIO_PAD positions
    - Autoregressive decode with mRoPE position tracking

    The audio encoder is handled separately via StaticQwen3ASREncoder (traced).
    """

    vision_model_cls = None
    vision_model_wrapper = None

    def enable_vision_encoder(self, **kwargs):
        """No-op: encoder is traced separately."""
        pass

    def load(
        self,
        compiled_model_path: str,
        start_rank_id: int = 0,
        debug: bool = False,
        **kwargs,
    ):
        """Load compiled text model only (no vision model)."""
        text_path = normalize_path(compiled_model_path) + "text_model/"
        self.text_traced_model = torch.jit.load(text_path + "model.pt")

        text_weights = self.get_text_builder(debug).shard_checkpoint()
        start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32)
        self.text_traced_model.nxd_model.initialize(text_weights, start_rank_tensor)

        for model_wrapper in self.text_models:
            model_wrapper.model = self.text_traced_model

        self.is_loaded_to_neuron = True

    def compile(
        self,
        compiled_model_path: str,
        debug: bool = False,
        pre_shard_weights_hook=None,
        dry_run: bool = False,
    ):
        """Compile text model only (skip vision model trace)."""
        from neuronx_distributed_inference.models.application_base import (
            NeuronApplicationBase,
        )

        NeuronApplicationBase.compile(
            self,
            compiled_model_path,
            debug=debug,
            pre_shard_weights_hook=pre_shard_weights_hook,
            dry_run=dry_run,
        )

    @classmethod
    def get_state_dict(cls, model_name_or_path: str, config):
        """Convert HuggingFace Qwen3-ASR weights to NxDI format.

        Key mappings:
        - thinker.model.* -> language_model.* (text decoder layers)
        - thinker.lm_head.* -> lm_head.* (output projection)
        - thinker.audio_tower.* -> excluded (handled by traced encoder)
        """
        raw_sd = nxdi_load_sd(model_name_or_path)
        converted_sd = {}

        for key, value in raw_sd.items():
            if key.startswith("thinker.audio_tower."):
                continue  # Encoder handled separately
            if key.startswith("thinker.model."):
                new_key = "language_model." + key[len("thinker.model.") :]
                converted_sd[new_key] = value
            elif key.startswith("thinker.lm_head."):
                new_key = key[len("thinker.") :]
                converted_sd[new_key] = value
            else:
                converted_sd[key] = value

        model_sd = NeuronQwen3VLTextForCausalLM.convert_hf_to_neuron_state_dict(
            converted_sd, config.text_config
        )

        # Handle tied embeddings
        if getattr(config.text_config, "tie_word_embeddings", False):
            if "embed_tokens.weight" in model_sd and "lm_head.weight" not in model_sd:
                model_sd["lm_head.weight"] = model_sd["embed_tokens.weight"]

        return model_sd

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        seq_ids: Optional[torch.Tensor] = None,
        sampling_params: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass with audio embedding scatter.

        During prefill (input_ids seq_len > 1):
            - Scatters audio_embeddings at AUDIO_PAD_ID positions
            - Computes mRoPE position_ids (3 axes, all same for ASR)

        During decode (input_ids seq_len == 1):
            - Uses dummy vision inputs (no scatter)
            - Increments mRoPE positions using stored rope_deltas
        """
        pad_limit = self.get_padding_length(input_ids)

        # Determine if we're in prefill with audio or decode/text-only
        if (
            audio_embeddings is not None
            and input_ids.shape[-1] > 1
            and audio_embeddings.sum() != 0
        ):
            # Prefill with audio: scatter embeddings at AUDIO_PAD positions
            vision_mask = (input_ids == AUDIO_PAD_ID).unsqueeze(-1).to(torch.bool)
            vision_mask = generate_positions_from_mask(vision_mask.squeeze())
            vision_mask = pad_positions(vision_mask, pad_limit, (pad_limit - 1))

            vision_embeddings = audio_embeddings.to(
                self.text_config.neuron_config.torch_dtype
            )
            embedding_dim = vision_embeddings.shape[-1]
            vision_embeddings = vision_embeddings.view(-1, embedding_dim).unsqueeze(0)
            vision_embeddings = pad_vision_embeddings(vision_embeddings, pad_limit)
        else:
            # Text-only or decode phase: use dummy inputs
            vision_embeddings, vision_mask, _ = (
                self.text_model_wrapper.get_dummy_vision_inputs(
                    config=self.text_config,
                    input_ids=input_ids,
                    n_active_tokens=pad_limit,
                    fill_value=(pad_limit - 1),
                )
            )

        # Compute mRoPE position IDs (3 axes, all identical for ASR)
        if input_ids.shape[-1] > 1:
            # Prefill: compute positions from attention mask
            if attention_mask is not None:
                pos = attention_mask.long().cumsum(-1) - 1
                pos.masked_fill_(attention_mask == 0, 1)
            else:
                seq_len = input_ids.shape[1]
                pos = torch.arange(seq_len).unsqueeze(0)
            # Expand to 3 mRoPE axes [temporal, height, width] - all same for ASR
            rotary_position_ids = pos.unsqueeze(0).expand(3, -1, -1)

            # Store rope_deltas for decode phase
            if attention_mask is not None:
                max_pos = pos.max(-1, keepdim=True)[0]
                self.rope_deltas = (
                    max_pos + 1 - attention_mask.sum(-1, keepdim=True)
                ).long()
            else:
                self.rope_deltas = torch.zeros(1, 1, dtype=torch.long)
        else:
            # Decode: increment position based on stored delta
            batch_size = input_ids.shape[0]
            if self.rope_deltas is not None:
                delta = self.rope_deltas.to(input_ids.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            else:
                delta = 0
            rotary_position_ids = copy.deepcopy(position_ids)
            rotary_position_ids = rotary_position_ids.view(1, -1).expand(batch_size, -1)
            rotary_position_ids = rotary_position_ids.add(delta)
            rotary_position_ids = rotary_position_ids.unsqueeze(0).expand(3, -1, -1)

        deepstack_vision_embeds = torch.zeros(0)

        # Call grandparent forward (bypasses NeuronQwen3VLForCausalLM's vision handling)
        output_token = super(NeuronQwen3VLForCausalLM, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            rotary_position_ids=rotary_position_ids,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
            deepstack_vision_embeds=deepstack_vision_embeds,
        )
        return output_token


def create_inference_config(
    model_path: str,
    tp_degree: int = 4,
    batch_size: int = 1,
    n_positions: int = 1024,
) -> Qwen3VLInferenceConfig:
    """Create Qwen3VLInferenceConfig for Qwen3-ASR text decoder.

    Args:
        model_path: Path to HuggingFace model directory
        tp_degree: Tensor parallel degree (2 or 4 for trn2.3xlarge)
        batch_size: Inference batch size
        n_positions: KV cache length (1024 sufficient for most ASR)

    Returns:
        Qwen3VLInferenceConfig configured for Qwen3-ASR
    """
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = hf_config.thinker_config.text_config

    text_neuron_config = Qwen3VLNeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        n_positions=n_positions,
        seq_len=n_positions,
        max_context_length=n_positions,
        on_device_sampling_config=None,
        torch_dtype=torch.bfloat16,
    )
    # Dummy vision config (required by Qwen3VLInferenceConfig but not used)
    vision_neuron_config = Qwen3VLNeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        seq_len=512,
        n_positions=512,
        on_device_sampling_config=None,
        torch_dtype=torch.bfloat16,
    )

    text_config_dict = {
        "hidden_size": text_config.hidden_size,
        "num_hidden_layers": text_config.num_hidden_layers,
        "num_attention_heads": text_config.num_attention_heads,
        "num_key_value_heads": text_config.num_key_value_heads,
        "head_dim": text_config.head_dim,
        "intermediate_size": text_config.intermediate_size,
        "vocab_size": text_config.vocab_size,
        "max_position_embeddings": text_config.max_position_embeddings,
        "rope_theta": text_config.rope_theta,
        "rms_norm_eps": text_config.rms_norm_eps,
        "tie_word_embeddings": text_config.tie_word_embeddings,
        "attention_bias": getattr(text_config, "attention_bias", False),
        "hidden_act": "silu",
        "rope_scaling": {
            "type": "mrope",
            "rope_type": "default",
            "mrope_section": [24, 20, 20],
        },
        "pad_token_id": EOS_ID,
        "attention_dropout": 0.0,
        "bos_token_id": EOS_ID,
        "dtype": "bfloat16",
        "eos_token_id": IM_END_ID,
        "initializer_range": 0.02,
        "output_attentions": False,
        "output_hidden_states": False,
    }

    vision_config_dict = {
        "hidden_size": 1024,
        "num_hidden_layers": 1,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 64,
        "intermediate_size": 4096,
        "image_size": 224,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "deepstack_visual_indexes": [],
        "vocab_size": text_config.vocab_size,
        "max_position_embeddings": 512,
        "depth": 1,
        "hidden_act": "gelu",
        "in_channels": 3,
        "initializer_range": 0.02,
        "num_heads": 16,
        "num_position_embeddings": 256,
        "out_hidden_size": text_config.hidden_size,
        "temporal_patch_size": 2,
    }

    config = Qwen3VLInferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        text_config=text_config_dict,
        vision_config=vision_config_dict,
        _name_or_path=model_path,
        image_token_id=AUDIO_PAD_ID,
    )
    return config
