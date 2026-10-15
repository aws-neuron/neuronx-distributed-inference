#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for InternVL3-8B-Instruct NxDI contrib model.

Validates Neuron model accuracy against CPU reference using logit_validation.

Usage:
    pytest test_model.py -v --tb=short

Prerequisites:
    - Model downloaded to MODEL_PATH
    - Compiled model at COMPILED_MODEL_PATH (run compile_internvl3_vlm.py first)
    - Neuron runtime available (trn2.3xlarge, LNC=2, TP=4)
"""

import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch
from torch_neuronx.testing.validation import logit_validation
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.accuracy import generate_expected_logits
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

# Import from src directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_internvl3 import (
    NeuronInternVL3ForCausalLM,
    InternVL3InferenceConfig,
)


# Test configuration — override via environment variables if needed
MODEL_PATH = os.environ.get(
    "INTERNVL3_MODEL_PATH", "/mnt/models/InternVL3-8B-Instruct/"
)
COMPILED_MODEL_PATH = os.environ.get(
    "INTERNVL3_COMPILED_PATH", "/mnt/models/neuron_models/InternVL3-8B-Instruct/"
)
NUM_TOKENS_TO_CHECK = 16
TOKEN_DIVERGENCE_ATOL = 0.02


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_compiled_model(model_path: str, compiled_path: str):
    """Load pre-compiled InternVL3 model for inference."""
    config_path = Path(compiled_path) / "neuron_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"neuron_config.json not found at {config_path}")

    with open(config_path) as f:
        config_data = json.load(f)
    nc_dict = config_data.get("neuron_config", config_data)

    dtype_str = nc_dict.get("torch_dtype", "torch.bfloat16")
    if isinstance(dtype_str, str):
        dtype = (
            getattr(torch, dtype_str.split(".")[-1])
            if "torch" in dtype_str
            else torch.bfloat16
        )
    else:
        dtype = dtype_str

    text_neuron_config = NeuronConfig(
        tp_degree=nc_dict.get("tp_degree", 4),
        max_batch_size=nc_dict.get("batch_size", nc_dict.get("max_batch_size", 1)),
        seq_len=nc_dict.get("seq_len", 2048),
        torch_dtype=dtype,
        on_device_sampling_config=None,
        save_sharded_checkpoint=True,
    )

    vision_neuron_config = NeuronConfig(
        tp_degree=nc_dict.get("tp_degree", 4),
        max_batch_size=1,
        seq_len=256,
        torch_dtype=dtype,
        on_device_sampling_config=None,
        buckets=[1],
        fused_qkv=True,
        save_sharded_checkpoint=True,
    )

    config = InternVL3InferenceConfig.from_pretrained(
        model_path,
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
    )

    model = NeuronInternVL3ForCausalLM(model_path, config=config)
    model.load(compiled_path)
    return model


def get_tp_aligned_vocab_size(vocab_size: int, tp_degree: int) -> int:
    """Get the largest vocab index that's valid (not TP padding).

    When vocab_size is not divisible by tp_degree, lm_head pads up to the
    next multiple of tp_degree. The padded positions contain -FLT_MAX in
    the Neuron output. Truncate to this boundary to avoid false failures.
    """
    return (vocab_size // tp_degree) * tp_degree


# ---------------------------------------------------------------------------
# Integration tests (requires pre-compiled Neuron model)
# ---------------------------------------------------------------------------


class TestInternVL3Integration:
    """Integration tests for InternVL3-8B-Instruct on Neuron."""

    @pytest.fixture(scope="class")
    def neuron_model(self):
        """Load pre-compiled Neuron model (shared across tests in this class)."""
        compiled_path = Path(COMPILED_MODEL_PATH)
        if not compiled_path.exists():
            pytest.skip(f"Compiled model not found at {compiled_path}")
        return load_compiled_model(MODEL_PATH, COMPILED_MODEL_PATH)

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Load tokenizer."""
        return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    def test_config(self, neuron_model):
        """Validate InternVL3 config matches expected Qwen2.5-7B architecture."""
        config = neuron_model.config
        assert config.hidden_size == 3584
        assert config.num_attention_heads == 28
        assert config.num_key_value_heads == 4
        assert config.num_hidden_layers == 28
        assert config.intermediate_size == 18944
        assert config.vocab_size == 151674
        assert config.rope_theta == 1000000.0

    def test_text_logit_validation(self, neuron_model, tokenizer):
        """
        Validate text-only logit accuracy: Neuron vs CPU reference.

        Uses generate_expected_logits() for CPU golden logits and
        logit_validation() for multi-tier logit comparison.

        InternVL3 vocab_size (151674) is not divisible by TP degree (4),
        so lm_head pads to the next multiple (151676). The 2 padding
        positions get -FLT_MAX in Neuron output. We truncate both CPU
        and Neuron logits to the TP-aligned boundary (151672) to avoid
        false failures from these padding artifacts.
        """
        prompt = "The capital of France is"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        attention_mask = torch.ones_like(input_ids, dtype=torch.int32)

        generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=NUM_TOKENS_TO_CHECK,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or 0,
        )

        # 1. Generate CPU reference logits
        expected_logits = generate_expected_logits(
            neuron_model=neuron_model,
            input_ids=input_ids,
            inputs_attention_mask=attention_mask,
            generation_config=generation_config,
            num_tokens=NUM_TOKENS_TO_CHECK,
        )

        # 2. Truncate expected logits to TP-aligned vocab boundary
        tp_degree = neuron_model.config.neuron_config.tp_degree
        vocab_size = neuron_model.config.vocab_size
        aligned_vocab = get_tp_aligned_vocab_size(vocab_size, tp_degree)
        expected_logits = expected_logits[..., :aligned_vocab]

        # 3. Build generate_fn that returns Neuron logits (also truncated)
        adapter = HuggingFaceGenerationAdapter(neuron_model)

        expected_attention_mask = torch.ones(
            (attention_mask.shape[0], expected_logits.shape[0]),
            dtype=torch.int32,
        )
        extrapolated_attention_mask = torch.cat(
            (attention_mask, expected_attention_mask), dim=1
        )

        def generate_fn(input_ids_tensor):
            neuron_model.reset()
            input_length = input_ids_tensor.shape[1]
            attn_mask = extrapolated_attention_mask[:, :input_length]
            new_tokens = NUM_TOKENS_TO_CHECK + input_ids.shape[1] - input_length

            outputs = adapter.generate(
                input_ids=input_ids_tensor,
                attention_mask=attn_mask,
                max_new_tokens=new_tokens,
                min_new_tokens=new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                generation_config=generation_config,
            )
            actual_logits = torch.stack(outputs.scores)
            # Truncate to TP-aligned vocab boundary
            actual_logits = actual_logits[..., :aligned_vocab]
            return actual_logits

        # 4. Run logit_validation with BF16-appropriate tolerances.
        # Default tolerances (top-5: 0.01, top-50: 0.02) are calibrated for
        # FP16/FP32 models. BF16 has lower mantissa precision (7 bits vs 10),
        # so normalized logit errors of 0.03-0.06 are expected.
        bf16_tol_map = {
            5: (1e-5, 0.05),
            50: (1e-5, 0.06),
            1000: (1e-5, 0.06),
            None: (1e-5, 0.08),
        }

        passed, results, status_msg = logit_validation(
            input_ids=input_ids,
            generate_fn=generate_fn,
            expected_logits=expected_logits,
            tol_map=bf16_tol_map,
            divergence_difference_tol=TOKEN_DIVERGENCE_ATOL,
        )

        print(f"\n{status_msg}")
        assert passed, f"Logit validation failed:\n{status_msg}"
