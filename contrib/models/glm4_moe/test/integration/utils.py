# coding=utf-8
"""Integration test utilities for GLM-4.5 MoE."""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from glm4_moe.modeling_glm4_moe import Glm4MoeInferenceConfig, NeuronGlm4MoeForCausalLM
from neuronx_distributed_inference.models.config import (
    MoENeuronConfig,
    OnDeviceSamplingConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config


LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))


def create_neuron_config(
    tp_degree: int = 2,
    seq_len: int = 128,
    batch_size: int = 1,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> MoENeuronConfig:
    """Create MoENeuronConfig for GLM-4.5 MoE integration tests.

    Args:
        tp_degree: Tensor parallelism degree.
        seq_len: Maximum sequence length.
        batch_size: Batch size for inference.
        torch_dtype: Dtype for model weights.

    Returns:
        Configured MoENeuronConfig.
    """
    return MoENeuronConfig(
        tp_degree=tp_degree,
        moe_tp_degree=tp_degree,
        moe_ep_degree=1,
        batch_size=batch_size,
        ctx_batch_size=batch_size,
        tkg_batch_size=batch_size,
        seq_len=seq_len,
        max_context_length=seq_len - 8,
        torch_dtype=torch_dtype,
        on_device_sampling_config=OnDeviceSamplingConfig(do_sample=False, top_k=1),
        output_logits=True,  # Required for check_accuracy_logits_v2
        enable_bucketing=False,
        flash_decoding_enabled=False,
        fused_qkv=True,
        sequence_parallel_enabled=False,
        qkv_kernel_enabled=False,
        attn_kernel_enabled=False,
    )


def save_hf_checkpoint(config_dict: dict, save_dir: str) -> str:
    """Create a tiny random-weight HF checkpoint from a config dict.

    Args:
        config_dict: HuggingFace model config dictionary.
        save_dir: Directory to save the checkpoint.

    Returns:
        Path to the saved checkpoint directory.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    os.makedirs(save_dir, exist_ok=True)

    # Remove integration test internal note
    config_dict = {k: v for k, v in config_dict.items() if not k.startswith("_")}

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    config = AutoConfig.from_pretrained(save_dir)
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(torch.bfloat16)
    model.save_pretrained(save_dir)

    return save_dir


def prepare_inputs(
    batch_size: int,
    seq_len: int,
    vocab_size: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare random input tensors for inference.

    Args:
        batch_size: Number of sequences in the batch.
        seq_len: Sequence length.
        vocab_size: Vocabulary size for token sampling.

    Returns:
        Tuple of (input_ids, attention_mask).
    """
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return input_ids, attention_mask


def get_test_name_suffix(tp_degree: int, dtype: torch.dtype, seq_len: int) -> str:
    """Generate a descriptive suffix for test artifact file names.

    Args:
        tp_degree: Tensor parallelism degree.
        dtype: Model dtype.
        seq_len: Sequence length.

    Returns:
        String suffix, e.g. 'tp2_bf16_s128'.
    """
    dtype_str = {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
    }.get(dtype, "unk")
    return f"tp{tp_degree}_{dtype_str}_s{seq_len}"
