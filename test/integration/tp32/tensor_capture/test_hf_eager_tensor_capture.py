from argparse import Namespace
import os
import pytest
import tempfile
import torch
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig, AutoTokenizer

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeInferenceConfig, NeuronQwen3MoeForCausalLM
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.constants import TEST_PROMPT
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from neuronx_distributed.utils.tensor_capture.model_modification import (
    modify_hf_eager_model_for_tensor_capture,
    find_available_modules)

import torch.nn as nn
import types
from typing import List, Dict, Any, Optional, Union, Callable


def save_checkpoint(config_path):
    hf_config = AutoConfig.from_pretrained(config_path)
    hf_model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch.float32)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir



def get_1_layer_qwen_model_config_and_checkpoint(tp_degree, batch_size, max_context_length, seq_len, torch_dtype, fused_qkv):
    # Load model from config, and save with random weights.
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/config.json"

    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name

    neuron_config = MoENeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        max_context_length=max_context_length,
        seq_len=seq_len, 
        torch_dtype=torch_dtype,
        fused_qkv=fused_qkv,
    )
    config = Qwen3MoeInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    return model_tempdir, config


def test_hf_tensor_capture():
    m , c = get_1_layer_qwen_model_config_and_checkpoint(32,1,512,5120,"bfloat16",True)

    model = NeuronQwen3MoeForCausalLM(m.name, c)
    hf_model = model.load_hf_model(m.name)
    modules_to_capture=['layers.0.mlp.gate']
    tensor_capture_save_dir= os.path.dirname(os.path.abspath(__file__)) + "/tensor_capture_hf/"
    hf_model, hooks, _ = modify_hf_eager_model_for_tensor_capture(hf_model, modules_to_capture, tensor_capture_save_dir)

    prompts = [TEST_PROMPT]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B-Instruct-2507")
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    num_tokens_to_check=512
    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)

    hf_model.generate(
        inputs.input_ids,
        max_new_tokens=num_tokens_to_check,
        min_new_tokens=num_tokens_to_check,
        do_sample=False,
        attention_mask=inputs.attention_mask,
        return_dict_in_generate=True,
        output_scores=True,
        generation_config=generation_config
    )

    tensor_files = sorted([
        f for f in os.listdir(tensor_capture_save_dir)
        if f.startswith("captured_tensors_") and f.endswith(".pt")
    ])

    print(f"Number of captured tensors: {len(tensor_files)}")
    assert len(tensor_files) == 512, "Expected 512 tensor files"

    for fname in tensor_files:
        step = int(fname.split("_step_")[1].split("_")[0])
        path = os.path.join(tensor_capture_save_dir, fname)
        tensor = torch.load(path)
        expected_shape = (15, 128) if step == 1 else (1, 128)
        assert tensor.shape == expected_shape, f"{fname}: got shape {tensor.shape}, expected {expected_shape}"

    print("All tensor files have the expected shape.")

    # Optionally clean up
    import shutil
    shutil.rmtree(tensor_capture_save_dir)
    print(f"Deleted directory: {tensor_capture_save_dir}")


if __name__ == "__main__":
    test_hf_tensor_capture()