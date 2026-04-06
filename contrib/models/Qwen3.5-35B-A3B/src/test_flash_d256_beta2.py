#!/usr/bin/env python3
"""
Integration test for Beta 2 nki_flash_attn_d256 kernel.

Tests that the upgraded kernel produces correct output when integrated
into the full Qwen3.5-35B-A3B model. Uses seq_len=640, max_context=512
to ensure the kernel activates (head_dim > 128 and seq_len >= 512).

Usage:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    cd /home/ubuntu/nxdi-qwen35/contrib/models/Qwen3.5-35B-A3B/src
    python3 test_flash_d256_beta2.py
"""

import json
import os
import sys
import time

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from modeling_qwen35_moe import NeuronQwen35MoeForCausalLM, Qwen35MoeInferenceConfig
from neuronx_distributed_inference.models.config import MoENeuronConfig

MODEL_PATH = "/mnt/models/Qwen3.5-35B-A3B"
COMPILED_PATH = "/mnt/models/compiled_flash_d256_beta2"


def create_config():
    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        full_config = json.load(f)
    text_config = full_config.get("text_config", full_config)

    neuron_config = MoENeuronConfig(
        tp_degree=4,
        max_batch_size=1,
        max_context_length=512,
        max_new_tokens=128,
        seq_len=640,
        on_device_sampling_config=None,
        torch_dtype=torch.bfloat16,
        fused_qkv=True,
        moe_tp_degree=4,
        moe_ep_degree=1,
        blockwise_matmul_config={"block_size": 256},
        context_encoding_buckets=[512],
    )

    config_dict = dict(text_config)
    config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
    if "rope_parameters" in text_config:
        config_dict["rope_theta"] = text_config["rope_parameters"].get(
            "rope_theta", 10000000
        )
    if config_dict.get("tie_word_embeddings") is None:
        config_dict["tie_word_embeddings"] = False

    return Qwen35MoeInferenceConfig(neuron_config=neuron_config, **config_dict)


def greedy_generate(model, input_ids, max_new_tokens=10, eos_token_id=None):
    """Simple greedy generation loop for NxDI models."""
    bs, prompt_len = input_ids.shape
    seq_ids = torch.arange(bs, dtype=torch.int32)

    # Context encoding (prefill): send full prompt
    attention_mask = torch.ones((bs, prompt_len), dtype=torch.int32)
    position_ids = (
        torch.arange(prompt_len, dtype=torch.int32).unsqueeze(0).expand(bs, -1)
    )

    with torch.no_grad():
        model.reset()
        output = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
        )
        logits = output.logits if hasattr(output, "logits") else output[0]

    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True).to(torch.int64)
    generated_tokens = [next_token]

    # Token generation (autoregressive)
    for step in range(1, max_new_tokens):
        cur_pos = prompt_len + step - 1
        position_ids = torch.tensor([[cur_pos]], dtype=torch.int32).expand(bs, -1)
        attention_mask = torch.ones((bs, cur_pos + 1), dtype=torch.int32)

        with torch.no_grad():
            output = model.forward(
                input_ids=next_token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                seq_ids=seq_ids,
            )
            logits = output.logits if hasattr(output, "logits") else output[0]

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True).to(
            torch.int64
        )
        generated_tokens.append(next_token)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return torch.cat([input_ids] + generated_tokens, dim=-1)


def main():
    print("=== Compat API flash_attn_d256 Integration Test ===\n")

    config = create_config()
    model = NeuronQwen35MoeForCausalLM(model_path=MODEL_PATH, config=config)

    compiled_path = Path(COMPILED_PATH)
    if not (compiled_path / "model.pt").exists():
        print(f"Compiling model to {COMPILED_PATH}...")
        os.makedirs(COMPILED_PATH, exist_ok=True)
        t0 = time.time()
        model.compile(COMPILED_PATH)
        print(f"Compilation complete in {time.time() - t0:.1f}s")
    else:
        print(f"Loading from cache: {COMPILED_PATH}")

    model.load(COMPILED_PATH)
    print("Model loaded\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "The capital of France is"
    expected = "Paris"

    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids
    print(f"Prompt: {prompt}")
    print(f"Input IDs shape: {input_ids.shape}")

    t0 = time.time()
    output = greedy_generate(
        model, input_ids, max_new_tokens=10, eos_token_id=tokenizer.eos_token_id
    )
    t1 = time.time()

    generated_ids = output[0, input_ids.shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print(f"\nGenerated: {generated_text}")
    print(f"Time: {t1 - t0:.1f}s")

    if expected.lower() in generated_text.lower():
        print(f"\nPASS -- '{expected}' found in output")
    else:
        print(f"\nFAIL -- expected '{expected}' in output, got: {generated_text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
