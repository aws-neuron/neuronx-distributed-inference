#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeltaNet path parity probe for Qwen3.5-4B.

Run on a Neuron instance after weights are available. This intentionally is not
part of normal pytest collection because it can compile NKI kernels and requires
the full checkpoint.

Example:
    cd contrib/models/Qwen3.5-4B
    QWEN35_MODEL_PATH=/mnt/models/Qwen3.5-4B \\
      python test/parity/deltanet_path_probe.py --layer-idx 0 --seq-len 128
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager

import torch
import torch.nn.functional as F

_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)

from neuronx_distributed_inference.models.config import NeuronConfig
from src.modeling_qwen35 import Qwen35InferenceConfig, NeuronGatedDeltaNet


@contextmanager
def patched_env(**updates):
    old = {k: os.environ.get(k) for k in updates}
    for k, v in updates.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def cosine(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def load_config(model_path, tp_degree):
    with open(os.path.join(model_path, "config.json")) as f:
        full_config = json.load(f)
    text_config = full_config.get("text_config", full_config)
    config_dict = dict(text_config)
    config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
    config_dict["tie_word_embeddings"] = full_config.get(
        "tie_word_embeddings",
        text_config.get("tie_word_embeddings", False),
    )
    if "rope_parameters" in text_config:
        config_dict["rope_theta"] = text_config["rope_parameters"].get(
            "rope_theta", 10000000
        )

    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=1,
        max_batch_size=1,
        seq_len=128,
        torch_dtype=torch.bfloat16,
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=2,
    )
    return Qwen35InferenceConfig(neuron_config=neuron_config, **config_dict)


def strip_prefix_state_dict(state_dict):
    stripped = {}
    for k, v in state_dict.items():
        if k.startswith("language_model."):
            stripped[k.replace("language_model.", "", 1)] = v
        elif k.startswith("model.language_model."):
            stripped[k.replace("model.language_model.", "", 1)] = v
        elif k.startswith("model."):
            stripped[k.replace("model.", "", 1)] = v
        else:
            stripped[k] = v
    return stripped


def load_deltanet_layer_weights(module, model_path, layer_idx):
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    state_dict = strip_prefix_state_dict(hf_model.state_dict())
    prefix = f"layers.{layer_idx}.linear_attn."
    layer_sd = {}
    for name in module.state_dict().keys():
        key = prefix + name
        if key in state_dict:
            layer_sd[name] = state_dict[key]

    missing, unexpected = module.load_state_dict(layer_sd, strict=False)
    missing = [m for m in missing if not m.endswith("_buffer")]
    if missing or unexpected:
        raise RuntimeError(f"weight load mismatch: missing={missing}, unexpected={unexpected}")
    del hf_model


def run_path(module, hidden_states, mode):
    env = {
        "USE_NKI_FUSED": "0",
        "USE_NKI_CHUNKED": None,
        "USE_NKI": None,
        "DELTANET_SEQUENTIAL": None,
        "USE_PYTORCH_CHUNK": None,
    }
    if mode == "sequential":
        env["DELTANET_SEQUENTIAL"] = "1"
    elif mode == "fused":
        env["USE_NKI_FUSED"] = "1"
    elif mode == "chunk":
        env["USE_PYTORCH_CHUNK"] = "1"
    elif mode == "nki_recurrent":
        env["USE_NKI"] = "1"
    else:
        raise ValueError(f"unknown mode: {mode}")

    with patched_env(**env):
        with torch.no_grad():
            out, _dummy_kv, rec_state, conv_state = module(hidden_states)
    return out.detach().cpu(), rec_state.detach().cpu(), conv_state.detach().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.environ.get("QWEN35_MODEL_PATH"))
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument(
        "--compare",
        nargs="+",
        default=["fused"],
        choices=["fused", "chunk", "nki_recurrent"],
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "xla"])
    args = parser.parse_args()

    if not args.model_path:
        raise SystemExit("Set QWEN35_MODEL_PATH or pass --model-path")

    device = torch.device("cpu")
    if args.device == "xla":
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()

    config = load_config(args.model_path, args.tp_degree)
    if config.layer_types[args.layer_idx] != "linear_attention":
        raise SystemExit(f"layer {args.layer_idx} is {config.layer_types[args.layer_idx]}, not DeltaNet")

    module = NeuronGatedDeltaNet(config, args.layer_idx).to(device)
    load_deltanet_layer_weights(module, args.model_path, args.layer_idx)
    module = module.to(device=device, dtype=torch.bfloat16).eval()

    torch.manual_seed(0)
    hidden_states = torch.randn(
        1,
        args.seq_len,
        config.hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    ref = run_path(module, hidden_states, "sequential")
    print(f"reference=sequential layer={args.layer_idx} seq_len={args.seq_len}")
    for mode in args.compare:
        cur = run_path(module, hidden_states, mode)
        print(f"\nmode={mode}")
        for label, ref_t, cur_t in zip(("output", "recurrent_state", "conv_state"), ref, cur):
            print(
                f"{label}: cosine={cosine(ref_t, cur_t):.6f} "
                f"max_abs={max_abs(ref_t, cur_t):.6f} "
                f"shape={tuple(cur_t.shape)}"
            )


if __name__ == "__main__":
    main()
