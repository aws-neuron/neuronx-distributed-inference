#!/usr/bin/env python3
"""Smoke compile script for Gemma-4-26B-A4B-it on Trainium 2.

Following PR #106's `test/integration/test_model.py` shape. Run on a trn2
host with the NxDI venv activated.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd contrib/models/gemma-4-26b-a4b-it
    # IMPORTANT: trn2.3xlarge defaults to LNC=2 (4 logical cores). If you
    # want TP=8 you MUST set NEURON_LOGICAL_NC_CONFIG=1 first; otherwise
    # the runtime fails with c10::Error inside NeuronAllocators::Get when
    # the spmd state initializer tries to copy zeros to ranks 4..7.
    NEURON_LOGICAL_NC_CONFIG=1 PYTHONPATH=src python scripts/smoke_compile.py 2>&1 | tee ~/compile.log

Environment overrides:
    GEMMA4_MODEL_PATH       (default: /home/ubuntu/gemma4-26b-a4b)
    GEMMA4_COMPILED_PATH    (default: /home/ubuntu/gemma4-compiled)
    GEMMA4_TP_DEGREE        (default: 8; set to 4 if you do not want to set LNC=1)
    GEMMA4_BATCH_SIZE       (default: 1)
    GEMMA4_SEQ_LEN          (default: 256, kept short for first compile)
    GEMMA4_DISABLE_MOE      (default: 0; set to 1 for dense-only smoke)
    GEMMA4_MOE_EP_DEGREE    (default: 1)
    GEMMA4_MOE_TP_DEGREE    (default: <TP_DEGREE>)
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

# Apply NxDI runtime patches (NKI kernel for d>128, get_last_kv_window fix).
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
        # MoE knobs (consumed by MoENeuronConfig __init__):
        moe_ep_degree=MOE_EP_DEGREE,
        moe_tp_degree=MOE_TP_DEGREE,
        glu_mlp=True,
        glu_type="glu",
        # Gemma's router runs softmax in FP32. We fold this into the
        # custom NeuronGemma4Router, but the underlying NxDI RouterConfig
        # is still consulted by `initialize_moe_module` for typing of any
        # internal router-state buffers, so set sensible values here.
        router_act_fn="softmax",
        router_dtype="float32",
        # Gemma renormalizes top-k weights INSIDE the custom router and
        # bakes per_expert_scale in there. Disable NxDI's renorm so it
        # uses our pre-computed expert_affinities verbatim.
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
    # Smoke-compile flag: set GEMMA4_DISABLE_MOE=1 to validate the rest of
    # the architecture without the MoE branch (dense MLP only). MoE
    # integration with NxDI moe_v2 has separate process-group bring-up
    # work — see README "Known limitations".
    if os.environ.get("GEMMA4_DISABLE_MOE", "0") == "1":
        cfg.disable_moe_for_smoke_compile = True
    return cfg


def main() -> int:
    print("=" * 80)
    print(f"Gemma-4-26B-A4B-it smoke compile")
    print(f"  model_path:     {MODEL_PATH}")
    print(f"  compiled_path:  {COMPILED_PATH}")
    print(f"  tp_degree:      {TP_DEGREE}")
    print(f"  batch_size:     {BATCH_SIZE}")
    print(f"  seq_len:        {SEQ_LEN}")
    print("=" * 80)

    if not Path(MODEL_PATH).exists():
        print(f"ERROR: model path {MODEL_PATH} does not exist", file=sys.stderr)
        return 1

    config = create_config(MODEL_PATH)
    print(
        f"Config loaded: hidden_size={config.hidden_size}, "
        f"num_layers={config.num_hidden_layers}, "
        f"num_experts={getattr(config, 'num_experts', None)}, "
        f"top_k={getattr(config, 'top_k_experts', None)}"
    )

    print(f"\nCompiling to {COMPILED_PATH} ...")
    t0 = time.perf_counter()
    model = NeuronGemma4ForCausalLM(MODEL_PATH, config)
    model.compile(COMPILED_PATH)
    elapsed = time.perf_counter() - t0
    print(f"\nCompile finished in {elapsed/60:.1f} min")

    print("\nLoading compiled model ...")
    model = NeuronGemma4ForCausalLM(MODEL_PATH, config)
    model.load(COMPILED_PATH)
    print("Smoke compile + load OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
