# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Qwen3-Omni vision encoder for NxD Inference.

Reuses the Qwen3-VL vision model code since the architecture is identical.
Only difference: state dict key mapping (thinker.visual.* → visual.*) and
the PatchMerger naming (ln_q + mlp.0/mlp.2 vs norm + linear_fc1/linear_fc2).

The vision model is compiled and run separately from the text model, with
outputs passed through the ImageToText framework.
"""
import logging
from typing import List
from unittest.mock import patch as mock_patch

import torch
import torch.nn as nn

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.modules.checkpoint import load_state_dict
from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
    NeuronQwen3VLForImageEncoding,
    NeuronQwen3VLVisionModel,
    NeuronQwen3VLVisionModelWrapper,
)

logger = logging.getLogger(__name__)


def _load_state_dict_with_thinker_alias(model_path, *args, **kwargs):
    """Wraps load_state_dict to add model.visual.* aliases for thinker.visual.* keys."""
    sd = load_state_dict(model_path, *args, **kwargs)
    if "thinker.visual.pos_embed.weight" in sd and "model.visual.pos_embed.weight" not in sd:
        sd["model.visual.pos_embed.weight"] = sd["thinker.visual.pos_embed.weight"]
    return sd


class NeuronQwen3OmniVisionModel(NeuronQwen3VLVisionModel):
    """
    Qwen3-Omni vision model — architecturally identical to Qwen3-VL.
    The only code change is in state dict conversion (done externally).
    """
    pass


class NeuronQwen3OmniVisionModelWrapper(NeuronQwen3VLVisionModelWrapper):
    """Wraps NeuronQwen3OmniVisionModel for Neuron compilation.

    Patches load_state_dict during __init__ to handle Qwen3-Omni's
    pos_embed key prefix (thinker.visual.* instead of model.visual.*).
    """

    def __init__(self, *args, **kwargs):
        with mock_patch(
            "neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision.load_state_dict",
            _load_state_dict_with_thinker_alias,
        ):
            super().__init__(*args, **kwargs)


class NeuronQwen3OmniForImageEncoding(NeuronQwen3VLForImageEncoding):
    """Standalone vision encoder application for Qwen3-Omni."""

    _model_cls = NeuronQwen3OmniVisionModel

    def get_model_wrapper_cls(self):
        return NeuronQwen3OmniVisionModelWrapper

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        """
        Converts HF Qwen3-Omni vision state dict to Neuron format.

        Key mappings:
          thinker.visual.* → visual.*  (then same as Qwen3-VL)
          visual.merger.ln_q.* → merger.norm.*
          visual.merger.mlp.0.* → merger.linear_fc1.*
          visual.merger.mlp.2.* → merger.linear_fc2.*
          visual.merger_list.N.ln_q.* → deepstack_merger_list.N.norm.*
          visual.merger_list.N.mlp.0.* → deepstack_merger_list.N.linear_fc1.*
          visual.merger_list.N.mlp.2.* → deepstack_merger_list.N.linear_fc2.*
          visual.blocks.*.attn.qkv.* → blocks.*.attn.qkv_proj.Wqkv.*
          visual.blocks.*.attn.proj.* → blocks.*.attn.o_proj.*
        """
        new_state_dict = {}

        for key, value in state_dict.items():
            if "visual." not in key:
                continue

            new_key = key
            # Strip thinker prefix
            if new_key.startswith("thinker.visual."):
                new_key = new_key[len("thinker."):]
            elif new_key.startswith("model.thinker.visual."):
                new_key = new_key[len("model.thinker."):]

            # Strip visual. prefix (NxD vision model doesn't use it)
            new_key = new_key.replace("visual.", "", 1)

            # Qwen3-Omni uses merger_list; NxD Qwen3-VL uses deepstack_merger_list
            new_key = new_key.replace("merger_list.", "deepstack_merger_list.")

            # PatchMerger key renaming: ln_q → norm, mlp.0 → linear_fc1, mlp.2 → linear_fc2
            new_key = new_key.replace(".ln_q.", ".norm.")
            new_key = new_key.replace(".mlp.0.", ".linear_fc1.")
            new_key = new_key.replace(".mlp.2.", ".linear_fc2.")

            # Attention key renaming (same as Qwen3-VL)
            if ".attn.qkv." in new_key:
                new_key = new_key.replace(".attn.qkv.", ".attn.qkv_proj.Wqkv.")
            elif ".attn.proj." in new_key:
                new_key = new_key.replace(".attn.proj.", ".attn.o_proj.")

            new_state_dict[new_key] = value.clone().detach().contiguous()

        return new_state_dict
