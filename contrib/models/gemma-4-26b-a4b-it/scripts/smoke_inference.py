#!/usr/bin/env python3
"""Smoke inference for Gemma-4-26B-A4B-it on Trainium 2.

Goal: confirm the compiled+loaded model produces non-trivial output for a
short prompt. We do NOT validate accuracy here — that's downstream.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd contrib/models/gemma-4-26b-a4b-it
    PYTHONPATH=src python scripts/smoke_inference.py 2>&1 | tee ~/inference.log

Environment overrides (must match the values used by smoke_compile.py):
    GEMMA4_MODEL_PATH       (default: /home/ubuntu/gemma4-26b-a4b)
    GEMMA4_COMPILED_PATH    (default: /home/ubuntu/gemma4-compiled)
    GEMMA4_TP_DEGREE        (default: 8)
    GEMMA4_BATCH_SIZE       (default: 1)
    GEMMA4_SEQ_LEN          (default: 256)
    GEMMA4_MAX_NEW_TOKENS   (default: 8)
    GEMMA4_PROMPT           (default: "Hello, my name is")
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

import ndxi_patch  # noqa: E402

ndxi_patch.apply_patch()

from modeling_gemma4_neuron import (  # noqa: E402
    Gemma4InferenceConfig,
    Gemma4NeuronConfig,
    NeuronGemma4ForCausalLM,
)


MODEL_PATH = os.environ.get("GEMMA4_MODEL_PATH", "/home/ubuntu/gemma4-26b-a4b")
COMPILED_PATH = os.environ.get("GEMMA4_COMPILED_PATH", "/home/ubuntu/gemma4-compiled")
TP_DEGREE = int(os.environ.get("GEMMA4_TP_DEGREE", "8"))
BATCH_SIZE = int(os.environ.get("GEMMA4_BATCH_SIZE", "1"))
SEQ_LEN = int(os.environ.get("GEMMA4_SEQ_LEN", "256"))
MAX_NEW_TOKENS = int(os.environ.get("GEMMA4_MAX_NEW_TOKENS", "8"))
PROMPT = os.environ.get("GEMMA4_PROMPT", "Hello, my name is")
MOE_EP_DEGREE = int(os.environ.get("GEMMA4_MOE_EP_DEGREE", "1"))
MOE_TP_DEGREE = int(os.environ.get("GEMMA4_MOE_TP_DEGREE", str(TP_DEGREE)))


def create_config(model_path: str) -> Gemma4InferenceConfig:
    neuron_config = Gemma4NeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=BATCH_SIZE,
        max_batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        on_device_sampling_config=None,
        torch_dtype=torch.bfloat16,
        fused_qkv=False,
        attn_kernel_enabled=False,
        moe_ep_degree=MOE_EP_DEGREE,
        moe_tp_degree=MOE_TP_DEGREE,
        glu_mlp=True,
        glu_type="glu",
        router_act_fn="softmax",
        router_dtype="float32",
        disable_normalize_top_k_affinities=True,
    )

    def load_config_fn(config_obj):
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        for k, v in config_dict.items():
            setattr(config_obj, k, v)

    cfg = Gemma4InferenceConfig(
        neuron_config=neuron_config, load_config=load_config_fn
    )
    if os.environ.get("GEMMA4_DISABLE_MOE", "0") == "1":
        cfg.disable_moe_for_smoke_compile = True
    return cfg


def _load_tokenizer(model_path):
    """Try several backends — installed transformers may break on gemma-4."""
    # 1. AutoTokenizer (preferred but may fail on gemma-4 special tokens).
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"AutoTokenizer failed: {e}; falling back to tokenizers backend")
    # 2. Raw tokenizers (HF Rust). Reads tokenizer.json directly.
    from tokenizers import Tokenizer

    class _Wrapped:
        def __init__(self, t, eos_ids):
            self._t = t
            self.eos_token_id = eos_ids

        def __call__(self, text, return_tensors=None):
            enc = self._t.encode(text)
            ids = torch.tensor([enc.ids], dtype=torch.long)
            return {"input_ids": ids}

        def decode(self, ids, skip_special_tokens=True):
            return self._t.decode(ids, skip_special_tokens=skip_special_tokens)

    t = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
    # gemma-4 generation_config eos: <end_of_turn> 106 + <eos> 1.
    return _Wrapped(t, [1, 106])


def generate(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    n_positions = SEQ_LEN

    if seq_len > n_positions:
        raise RuntimeError(f"prompt length {seq_len} exceeds compiled seq_len {n_positions}")

    pad_len = n_positions - seq_len
    input_ids_padded = torch.cat(
        [input_ids, torch.zeros(1, pad_len, dtype=torch.long)], dim=1
    )
    attention_mask = torch.cat(
        [
            torch.ones(1, seq_len, dtype=torch.long),
            torch.zeros(1, pad_len, dtype=torch.long),
        ],
        dim=1,
    )
    position_ids = torch.zeros(1, n_positions, dtype=torch.long)
    position_ids[0, :seq_len] = torch.arange(seq_len)

    timing = {}
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_padded,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    timing["ttft_ms"] = (time.perf_counter() - t0) * 1000

    if hasattr(outputs, "logits") and outputs.logits is not None:
        logits = outputs.logits
        next_token_logits = logits[:, -1, :] if logits.dim() == 3 else logits
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    elif hasattr(outputs, "tokens") and outputs.tokens is not None:
        next_token = outputs.tokens[:, -1:]
    else:
        raise RuntimeError(f"Unexpected output type: {type(outputs)}")

    generated = [int(next_token.item())]
    cur_pos = seq_len

    t_gen = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        attention_mask[0, cur_pos] = 1
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                position_ids=torch.tensor([[cur_pos]]),
            )
        cur_pos += 1
        if hasattr(outputs, "logits") and outputs.logits is not None:
            logits = outputs.logits
            next_token_logits = logits[:, -1, :] if logits.dim() == 3 else logits
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        elif hasattr(outputs, "tokens") and outputs.tokens is not None:
            next_token = outputs.tokens[:, -1:]
        else:
            break
        generated.append(int(next_token.item()))
        eos = tokenizer.eos_token_id
        if isinstance(eos, list):
            if next_token.item() in eos:
                break
        elif next_token.item() == eos:
            break

    t_end = time.perf_counter()
    n_decode = len(generated) - 1
    if n_decode > 0:
        timing["tpot_ms"] = (t_end - t_gen) / n_decode * 1000
        timing["throughput_tps"] = n_decode / (t_end - t_gen)
    timing["total_tokens"] = len(generated)
    return generated, tokenizer.decode(generated, skip_special_tokens=True), timing


def main() -> int:
    print("=" * 80)
    print("Gemma-4-26B-A4B-it smoke inference")
    print(f"  model_path:    {MODEL_PATH}")
    print(f"  compiled_path: {COMPILED_PATH}")
    print(f"  tp_degree:     {TP_DEGREE}")
    print(f"  seq_len:       {SEQ_LEN}")
    print(f"  prompt:        {PROMPT!r}")
    print("=" * 80)

    if not Path(COMPILED_PATH).exists():
        print(f"ERROR: compiled path {COMPILED_PATH} does not exist", file=sys.stderr)
        return 1

    config = create_config(MODEL_PATH)
    print("Loading compiled model ...")
    t0 = time.perf_counter()
    model = NeuronGemma4ForCausalLM(MODEL_PATH, config)
    model.load(COMPILED_PATH)
    print(f"Load took {time.perf_counter() - t0:.1f}s")

    print("Loading tokenizer ...")
    tokenizer = _load_tokenizer(MODEL_PATH)

    print(f"\nGenerating {MAX_NEW_TOKENS} tokens ...")
    tokens, text, timing = generate(model, tokenizer, PROMPT, MAX_NEW_TOKENS)
    print(f"\nGenerated tokens: {tokens}")
    print(f"Decoded:          {text!r}")
    print(f"Timing:           {timing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
