#!/usr/bin/env python3
"""
Integration smoke for Gemma-4-26B-A4B-it NeuronX Distributed Inference port.

This is a Stage-1 / Stage-2 / Stage-3 smoke runner mirrored on PR #106's
test layout. It is invoked via the helper scripts under ``scripts/`` —
the body here is the same flow as those scripts but importable.

Usage:
    # Stage 1 — dense path only (fast)
    GEMMA4_DISABLE_MOE=1 PYTHONPATH=src \
        python test/integration/test_model.py compile

    # Stage 2 — MoE on
    PYTHONPATH=src python test/integration/test_model.py compile

    # Stage 3 — generate
    PYTHONPATH=src python test/integration/test_model.py generate
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

# Apply NxDI runtime patches (NKI kernel for d>128, get_last_kv_window fix).
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
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


def cmd_compile() -> int:
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: model path {MODEL_PATH} does not exist", file=sys.stderr)
        return 1
    config = create_config(MODEL_PATH)
    print(
        f"Config: hidden_size={config.hidden_size}, "
        f"num_layers={config.num_hidden_layers}, "
        f"num_experts={getattr(config, 'num_experts', None)}, "
        f"top_k={getattr(config, 'top_k_experts', None)}"
    )
    t0 = time.perf_counter()
    model = NeuronGemma4ForCausalLM(MODEL_PATH, config)
    model.compile(COMPILED_PATH)
    print(f"Compile finished in {(time.perf_counter() - t0)/60:.1f} min")
    model = NeuronGemma4ForCausalLM(MODEL_PATH, config)
    model.load(COMPILED_PATH)
    print("Smoke compile + load OK")
    return 0


def cmd_generate() -> int:
    # Defer to scripts/smoke_inference.py — this entrypoint is just a
    # pytest-friendly thin wrapper. The smoke_inference.py script in
    # scripts/ handles tokenizer fallback for gemma-4 special-tokens.
    from subprocess import run

    here = Path(__file__).resolve().parent.parent.parent
    smoke = here / "scripts" / "smoke_inference.py"
    return run([sys.executable, str(smoke)]).returncode


def main(argv) -> int:
    if len(argv) < 2 or argv[1] not in {"compile", "generate"}:
        print("usage: test_model.py {compile|generate}", file=sys.stderr)
        return 2
    return cmd_compile() if argv[1] == "compile" else cmd_generate()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
