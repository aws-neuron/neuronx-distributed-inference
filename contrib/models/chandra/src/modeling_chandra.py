#!/usr/bin/env python3
"""
Chandra OCR VLM on NeuronX Distributed Inference

Chandra (datalab-to/chandra) is a Qwen3-VL-8B fine-tune for OCR with layout
preservation. It uses NxDI's built-in Qwen3-VL multimodal pipeline -- no
custom modeling code is needed. This module provides tested configurations
and helper functions.

Supported configurations:
  - trn2.3xlarge: LNC=2, tp_degree=4, batch_size=1 or 4
  - vLLM serving via vllm-neuron 0.4.1+

Usage:
  See README.md for full examples, or run test/integration/test_model.py.
"""

import os
import time
from typing import Dict, Any, Optional, List

# These imports require the Neuron SDK environment
# (e.g., /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate)

# Default image resize limit: images with long side > this are resized down.
# Prevents exceeding the vision encoder bucket (4096 patches).
MAX_LONG_SIDE = 1024


def get_chandra_neuron_configs(
    batch_size: int = 1,
    seq_len: int = 8192,
    tp_degree: int = 4,
    logical_neuron_cores: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """
    Return tested NxDI neuron configs for Chandra on trn2.3xlarge.

    Args:
        batch_size: Text model batch size (1 or 4 tested).
        seq_len: Maximum sequence length for text model.
        tp_degree: Tensor parallel degree (4 for trn2.3xlarge LNC=2).
        logical_neuron_cores: LNC setting (2 for trn2.3xlarge default).

    Returns:
        Dict with 'text_neuron_config' and 'vision_neuron_config' keys.
    """
    text_neuron_config = {
        "batch_size": batch_size,
        "ctx_batch_size": 1,
        "tkg_batch_size": batch_size,
        "seq_len": seq_len,
        "max_context_length": seq_len,
        "enable_bucketing": True,
        "context_encoding_buckets": [1024, 4096, seq_len],
        "token_generation_buckets": [1024, 4096, seq_len],
        "world_size": tp_degree,
        "tp_degree": tp_degree,
        "torch_dtype": "bfloat16",
        "rpl_reduce_dtype": "bfloat16",
        "attention_dtype": "bfloat16",
        "cast_type": "as-declared",
        "logical_neuron_cores": logical_neuron_cores,
        "cc_pipeline_tiling_factor": logical_neuron_cores,
        "fused_qkv": True,
        "qkv_kernel_enabled": True,
        "mlp_kernel_enabled": False,  # SBUF OOM with LNC=2
        "attn_kernel_enabled": True,
    }
    vision_neuron_config = {
        "batch_size": 1,
        "seq_len": 4096,
        "max_context_length": 4096,
        "enable_bucketing": True,
        "buckets": [1024, 4096],
        "world_size": tp_degree,
        "tp_degree": tp_degree,
        "torch_dtype": "bfloat16",
        "rpl_reduce_dtype": "bfloat16",
        "cast_type": "as-declared",
        "logical_neuron_cores": logical_neuron_cores,
        "cc_pipeline_tiling_factor": logical_neuron_cores,
        "fused_qkv": True,
        "attn_kernel_enabled": False,  # Not supported for Qwen3-VL vision
        "mlp_kernel_enabled": False,  # Not supported for Qwen3-VL vision
    }
    return {
        "text_neuron_config": text_neuron_config,
        "vision_neuron_config": vision_neuron_config,
    }


def load_chandra_vllm(
    model_path: str,
    batch_size: int = 1,
    seq_len: int = 8192,
    tp_degree: int = 4,
    logical_neuron_cores: int = 2,
):
    """
    Load Chandra via vLLM-neuron with tested Neuron configs.

    Requires:
      - VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference (set automatically)
      - vllm-neuron 0.4.1+ with vLLM 0.13.0+
      - Neuron SDK 2.28+

    Args:
        model_path: Path to downloaded datalab-to/chandra model.
        batch_size: Max concurrent requests (1 or 4 tested).
        seq_len: Maximum sequence length.
        tp_degree: Tensor parallel degree.
        logical_neuron_cores: LNC setting.

    Returns:
        vllm.LLM instance ready for generate().
    """
    os.environ["VLLM_NEURON_FRAMEWORK"] = "neuronx-distributed-inference"
    from vllm import LLM

    configs = get_chandra_neuron_configs(
        batch_size=batch_size,
        seq_len=seq_len,
        tp_degree=tp_degree,
        logical_neuron_cores=logical_neuron_cores,
    )

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tp_degree,
        max_num_seqs=batch_size,
        max_model_len=seq_len,
        additional_config=dict(override_neuron_config=configs),
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    return llm


def resize_image(image, max_long_side: int = MAX_LONG_SIDE):
    """
    Resize an image so its long side does not exceed max_long_side.

    This prevents exceeding the vision encoder's 4096-patch bucket limit.
    A 1024px long side produces ~1092 patches for typical document images.

    Args:
        image: PIL.Image.Image instance.
        max_long_side: Maximum pixel dimension for the long side.

    Returns:
        PIL.Image.Image, possibly resized.
    """
    from PIL import Image

    w, h = image.size
    if max(w, h) > max_long_side:
        scale = max_long_side / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image


def run_chandra_ocr(
    llm,
    image,
    processor,
    prompt_text: str = "Convert the document in this image to markdown. Preserve the layout and formatting.",
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Run OCR on a single image using a loaded Chandra vLLM instance.

    Args:
        llm: vllm.LLM instance from load_chandra_vllm().
        image: PIL.Image.Image (will be resized if needed).
        processor: AutoProcessor from transformers.
        prompt_text: OCR instruction prompt.
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature (0.0 for greedy).

    Returns:
        Dict with 'text', 'num_tokens', 'latency_s', 'tokens_per_sec'.
    """
    from vllm import SamplingParams

    image = resize_image(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = {"prompt": prompt, "multi_modal_data": {"image": [image]}}

    t0 = time.time()
    outputs = llm.generate(
        [inputs],
        SamplingParams(top_k=1, max_tokens=max_tokens, temperature=temperature),
    )
    latency = time.time() - t0

    text = outputs[0].outputs[0].text
    ntok = len(outputs[0].outputs[0].token_ids)
    return {
        "text": text,
        "num_tokens": ntok,
        "latency_s": latency,
        "tokens_per_sec": ntok / latency if latency > 0 else 0,
    }
