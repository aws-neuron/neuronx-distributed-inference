#!/usr/bin/env python3
"""Token-level matching validation for Baichuan2-7B-Base contrib model.

Compares Neuron model output against HuggingFace CPU reference token-by-token
using the NeuroborosFoundations validator framework.
"""
import sys
sys.path.insert(0, '/home/dhwanw/workplace/NeuroborosFoundations/src')

from pathlib import Path

# Point model_class/config_class at the contrib src
CONTRIB_SRC = str(Path(__file__).parent.parent.parent / "src")
MODELING_FILE = f"{CONTRIB_SRC}/modeling_baichuan2.py"

from amzn.neuron.neuroboros.model_validation.validator import test_accuracy

config = {
    "model_name": "Baichuan2-7B-Base",
    "model_path": "/shared/dhwanw2/models/Baichuan2-7B-Base",
    "compiled_model_path": "/home/dhwanw/workplace/port_fixes/baichuan2/compiled_model",
    "model_class": f"{MODELING_FILE}:NeuronBaichuan2ForCausalLM",
    "config_class": f"{MODELING_FILE}:Baichuan2InferenceConfig",
    "num_tokens_to_check": 64,
}

test_params = {
    "batch_size": 1,
    "seq_len": 128,
}

passed, details = test_accuracy(config, test_params)

print(f"\n{'='*60}")
print(f"FINAL RESULT: {'PASSED' if passed else 'FAILED'}")
print(f"Greedy match rate: {details['match_rate']*100:.2f}%")
print(f"Avg teacher-forced: {details['avg_teacher_forced_match_rate']*100:.2f}%")
print(f"Avg best rate: {details['avg_best_match_rate']*100:.2f}%")
print(f"{'='*60}")

sys.exit(0 if passed else 1)
