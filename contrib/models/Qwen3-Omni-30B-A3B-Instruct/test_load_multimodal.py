#!/usr/bin/env python3
"""Quick test: load both text and vision models."""
import sys, os, time, torch, logging, traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s: %(message)s')

sys.path.insert(0, str(Path(__file__).parent / "src"))

from modeling_qwen3_omni import (
    NeuronQwen3OmniForCausalLM,
    Qwen3OmniInferenceConfig,
    load_qwen3_omni_multimodal_config,
)
from neuronx_distributed_inference.models.config import MoENeuronConfig, NeuronConfig

model_path = '/home/ubuntu/models/Qwen3-Omni-30B-A3B-Instruct'
compiled_path = '/home/ubuntu/traced_model/Qwen3-Omni-multimodal'

text_neuron_config = MoENeuronConfig(
    tp_degree=16, batch_size=1, seq_len=4096,
    max_context_length=2048, torch_dtype=torch.bfloat16,
    on_device_sampling_config={'top_k': 1, 'do_sample': False},
    blockwise_matmul_config={'use_torch_block_wise': True},
)

vision_neuron_config = NeuronConfig(
    tp_degree=16, batch_size=1, seq_len=4096, torch_dtype=torch.bfloat16,
)

config = Qwen3OmniInferenceConfig(
    text_neuron_config=text_neuron_config,
    vision_neuron_config=vision_neuron_config,
    load_config=load_qwen3_omni_multimodal_config(model_path),
)

model = NeuronQwen3OmniForCausalLM(model_path, config)

print('=== Loading model ===')
t0 = time.perf_counter()
try:
    model.load(compiled_path)
    print(f'SUCCESS: Model loaded in {time.perf_counter() - t0:.1f}s')
except Exception as e:
    print(f'LOAD FAILED after {time.perf_counter() - t0:.1f}s')
    traceback.print_exc()
