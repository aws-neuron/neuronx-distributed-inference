#!/usr/bin/env python3
"""Compile the Qwen3-Omni talker MoE on Neuron with ``norm`` tensor capture.

Why tensor_capture_config: HF talker's per-step generate loop reads the
transformer's final-layer hidden (pre-lm_head) via ``output.hidden_states[-1]``
and feeds it to ``code_predictor`` to produce 15 residual codes. Our previous
compile had ``tensor_capture_config=None`` and the runtime shim fabricated the
hidden by re-embedding argmax'd tokens. That lossy stand-in drifts and never
lets the talker emit ``codec_eos_token_id`` — every bench sample maxed out at
``max_new_tokens``. Adding a capture on ``norm`` (the RMSNorm applied right
before lm_head) gives us the real hidden through the NEFF at negligible cost.

Output path: ``/tmp/qwen3_omni_compiled/talker_tp8_capnorm``.

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  NEURON_RT_VISIBLE_CORES=0-7 python compile_talker.py
"""
import os
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0-7")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import _upstream_compat  # noqa: F401 — installs TensorRegistry.clear fix

import argparse
import json
import time

import torch

from neuronx_distributed_inference.models.config import (
    MoENeuronConfig, TensorCaptureConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from modeling_qwen3_omni_talker import (
    NeuronTalkerForCausalLM, TalkerInferenceConfig,
)

MODEL_PATH = "/home/ubuntu/models/Qwen3-Omni-30B-A3B-Instruct"
# Buckets mirror the existing talker_tp8 compile.
TALKER_BUCKETS = [64, 128, 256, 512, 1024, 2048, 4096]
TP_DEGREE = 8


def _talker_load_config():
    """Return a load_config hook that populates the config from
    talker.text_config (the 20-layer MoE inside Qwen3-Omni).

    We reuse ``load_pretrained_config`` by handing it the nested
    ``talker.text_config`` (which is itself a ``PretrainedConfig``).
    """
    from transformers import AutoConfig
    full_cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return load_pretrained_config(hf_config=full_cfg.talker_config.text_config)


def build_config():
    neuron_config = MoENeuronConfig(
        batch_size=1,
        seq_len=4096,
        max_context_length=4096,
        ctx_batch_size=1,
        tp_degree=TP_DEGREE,
        torch_dtype=torch.bfloat16,
        fused_qkv=False,
        sequence_parallel_enabled=False,
        flash_decoding_enabled=False,
        qkv_kernel_enabled=False,
        qkv_nki_kernel_enabled=False,
        attn_kernel_enabled=False,
        enable_bucketing=True,
        context_encoding_buckets=TALKER_BUCKETS,
        token_generation_buckets=TALKER_BUCKETS,
        # Capture the talker's final RMSNorm output (pre-lm_head hidden). The
        # underlying NeuronBaseModel attribute is ``norm``. The NEFF emits an
        # extra output tensor of shape [B, S_bucket, hidden_size=1024] per
        # forward.
        tensor_capture_config=TensorCaptureConfig(
            modules_to_capture=["norm"],
            capture_inputs=False,
        ),
        output_logits=True,
        blockwise_matmul_config={"use_torch_block_wise": True},
    )

    cfg = TalkerInferenceConfig(
        neuron_config=neuron_config,
        load_config=_talker_load_config(),
    )
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/qwen3_omni_compiled/talker_tp8_capnorm")
    args = parser.parse_args()

    cfg = build_config()
    print(f"Creating NeuronTalkerForCausalLM (tp={TP_DEGREE}, buckets={TALKER_BUCKETS})")
    app = NeuronTalkerForCausalLM(model_path=MODEL_PATH, config=cfg)

    print(f"Compiling to {args.out} ...")
    t0 = time.time()
    app.compile(args.out)
    print(f"Compile took {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
