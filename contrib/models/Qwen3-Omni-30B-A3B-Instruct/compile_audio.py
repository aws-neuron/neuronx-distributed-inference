#!/usr/bin/env python3
"""Compile Qwen3-Omni audio encoder transformer to a single Neuron core.

Conv2d frontend stays on CPU. Transformer layers + postprocessor are traced
per bucket size via torch_neuronx.trace (no TP).

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    NEURON_RT_VISIBLE_CORES=16 python compile_audio.py
"""
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)

MODEL_PATH = "/home/ubuntu/models/Qwen3-Omni-30B-A3B-Instruct"
COMPILED_PATH = "/home/ubuntu/traced_model/Qwen3-Omni-audio"

from modeling_qwen3_omni_audio import Qwen3OmniAudioEncoder

config_path = Path(MODEL_PATH) / "config.json"
with open(config_path) as f:
    full_config = json.load(f)

audio_config = full_config.get("thinker_config", {}).get("audio_config", {})
print(f"Audio config: d_model={audio_config.get('d_model')}, "
      f"layers={audio_config.get('encoder_layers', audio_config.get('num_hidden_layers'))}, "
      f"heads={audio_config.get('encoder_attention_heads')}")

print("Loading audio encoder weights...")
t0 = time.perf_counter()
encoder = Qwen3OmniAudioEncoder.from_pretrained(MODEL_PATH, audio_config)
print(f"Weights loaded in {time.perf_counter() - t0:.1f}s")

print(f"Compiling audio encoder to Neuron (buckets: {encoder.__class__.__name__})...")
t0 = time.perf_counter()
encoder.compile_neuron(COMPILED_PATH)
elapsed = time.perf_counter() - t0
print(f"Audio encoder compilation complete in {elapsed:.1f}s")
print(f"Compiled audio encoder saved to: {COMPILED_PATH}")
