#!/usr/bin/env python3
"""Integration test for Laguna-XS.2 NxDI contrib model.

Tests:
1. Config loading from model checkpoint
2. Model construction (CPU)
3. Compilation (Neuron)
4. Weight loading
5. Basic generation

Usage:
    # Set environment variables
    export LAGUNA_MODEL_PATH=/mnt/models/Laguna-XS.2
    export LAGUNA_COMPILED_PATH=/mnt/models/laguna-compiled
    export LAGUNA_TP_DEGREE=4

    # Run
    python test/integration/test_laguna.py
"""

import json
import os
import sys
import time

import torch

# Add src to path
test_dir = os.path.dirname(os.path.abspath(__file__))
contrib_dir = os.path.dirname(os.path.dirname(test_dir))
sys.path.insert(0, contrib_dir)

from src.modeling_laguna import (
    NeuronLagunaForCausalLM,
    LagunaInferenceConfig,
)

# Defaults
MODEL_PATH = os.environ.get("LAGUNA_MODEL_PATH", "/mnt/models/Laguna-XS.2")
COMPILED_PATH = os.environ.get("LAGUNA_COMPILED_PATH", "/mnt/models/laguna-compiled")
TP_DEGREE = int(os.environ.get("LAGUNA_TP_DEGREE", "4"))
SEQ_LEN = int(os.environ.get("LAGUNA_SEQ_LEN", "512"))
BATCH_SIZE = 1


def load_config():
    """Load Laguna config from model path."""
    from neuronx_distributed_inference.models.config import MoENeuronConfig

    neuron_config = MoENeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=BATCH_SIZE,
        max_batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        on_device_sampling_config=None,
        torch_dtype=torch.bfloat16,
        fused_qkv=False,
    )

    config = LagunaInferenceConfig.from_pretrained(
        MODEL_PATH,
        neuron_config=neuron_config,
    )

    return config


def test_config():
    """Test config loading."""
    print("=" * 60)
    print("TEST: Config Loading")
    print("=" * 60)

    config = load_config()
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  head_dim: {config.head_dim}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  num_experts: {config.num_experts}")
    print(f"  sliding_window: {config.sliding_window}")
    print(f"  layer_types[0:4]: {config.layer_types[:4]}")
    print(f"  heads_per_layer[0:4]: {config.num_attention_heads_per_layer[:4]}")
    print(f"  mlp_layer_types[0:3]: {config.mlp_layer_types[:3]}")

    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 40
    assert config.vocab_size == 100352
    assert config.head_dim == 128
    assert config.num_experts == 256
    print("  PASS\n")
    return config


def test_compile_and_load(config):
    """Test model compilation and weight loading."""
    print("=" * 60)
    print("TEST: Compile + Load")
    print("=" * 60)

    # Compile
    print(f"  Compiling with TP={TP_DEGREE}, seq_len={SEQ_LEN}...")
    start = time.time()
    model = NeuronLagunaForCausalLM(MODEL_PATH, config)
    model.compile(COMPILED_PATH)
    compile_time = time.time() - start
    print(f"  Compilation time: {compile_time:.1f}s")

    # Load
    print("  Loading compiled model...")
    start = time.time()
    model = NeuronLagunaForCausalLM(MODEL_PATH, config)
    model.load(COMPILED_PATH)
    load_time = time.time() - start
    print(f"  Load time: {load_time:.1f}s")

    print("  PASS\n")
    return model


def test_generation(model, config):
    """Test basic text generation."""
    print("=" * 60)
    print("TEST: Basic Generation")
    print("=" * 60)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt = "The capital of France is"
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_len = len(ids)

    # Prepare inputs
    input_ids = torch.zeros(1, SEQ_LEN, dtype=torch.int32)
    input_ids[0, :prompt_len] = torch.tensor(ids, dtype=torch.int32)
    attention_mask = torch.zeros(1, SEQ_LEN, dtype=torch.int32)
    attention_mask[0, :prompt_len] = 1
    position_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long)
    position_ids[0, :prompt_len] = torch.arange(prompt_len, dtype=torch.long)

    # Prefill (CTE)
    print(f"  Prompt: '{prompt}' ({prompt_len} tokens)")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    # Without on-device sampling, outputs.logits contains full logits.
    # CTE returns shape [batch, 1, vocab] (last position only). Extract via argmax.
    logits = outputs.logits if hasattr(outputs, "logits") else outputs.tokens
    if logits.dim() == 3:
        token_id = logits[0, -1, :].argmax().item()
    elif logits.dim() == 2:
        token_id = logits[0].argmax().item()
    else:
        token_id = logits.argmax().item()
    generated = [token_id]
    cur_pos = prompt_len

    # Token generation loop
    for step in range(19):
        tkg_in = torch.tensor([[token_id]], dtype=torch.long)
        am_len = cur_pos + 1
        tkg_mask = torch.cat(
            [
                torch.ones(1, am_len, dtype=torch.long),
                torch.zeros(1, SEQ_LEN - am_len, dtype=torch.long),
            ],
            dim=1,
        )
        out = model(
            input_ids=tkg_in,
            attention_mask=tkg_mask,
            position_ids=torch.tensor([[cur_pos]], dtype=torch.long),
        )
        cur_pos += 1
        out_logits = out.logits if hasattr(out, "logits") else out.tokens
        if out_logits.dim() == 3:
            token_id = out_logits[0, -1, :].argmax().item()
        elif out_logits.dim() == 2:
            token_id = out_logits[0].argmax().item()
        else:
            token_id = out_logits.argmax().item()
        generated.append(token_id)
        eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        if token_id in eos_ids:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"  Generated: '{text}'")
    print("  PASS\n")


if __name__ == "__main__":
    config = test_config()

    if "--config-only" in sys.argv:
        print("Config test only. Exiting.")
        sys.exit(0)

    model = test_compile_and_load(config)
    test_generation(model, config)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
