import copy
import os
import json
import shutil
import hashlib
import subprocess
import pytest
import shlex
import time
import tempfile
from contextlib import contextmanager
from typing import Iterable, List, Dict, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from neuronx_distributed_inference.models.config import (
    MoENeuronConfig,
    TensorCaptureConfig,
    OnDeviceSamplingConfig,
    TensorReplacementConfig,
)
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeInferenceConfig,
    NeuronQwen3MoeForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.tensor_capture_utils import get_tensor_capture_hook
from neuronx_distributed.utils.tensor_capture.model_modification import (
    modify_hf_eager_model_for_tensor_capture,
)
from torch_neuronx.testing.validation import DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE

# -------------------------
# Constants / Defaults
# -------------------------

LOGITS_DIVERGENCE_TOLERANCE = 0.005
CAPTURE_DIR = "~/tensor_capture_4layer"

# Reading neuron_config test cases from jsons
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

# BS16 baseline
with open(os.path.join(CURR_DIR, "neuron_configs/bs16_sl10k_baseline_tp64.json"), "r") as f:
    baseline_json = json.load(f)
BASELINE_NEURON_CONFIG = MoENeuronConfig(**baseline_json)

# BS16 TP4/CP16 for CTE, TP8/DP8 for TKG, EP32/TP2 for MOE
with open(os.path.join(CURR_DIR, "neuron_configs/bs16_sl10k_optimized.json"), "r") as f:
    optimized_json = json.load(f)
OPTIMIZED_NEURON_CONFIG = MoENeuronConfig(**optimized_json)

DEFAULT_PROMPTS = [
    "A pencil cost $0.50, and an eraser cost $0.25. If you bought 6 pencils and 8 erasers and paid $10, how much change would you get?"
]

# Map CPU module names → Neuron canonical names (optional for replacement)
MODULE_NAME_EQUIV = {"mlp.gate": "mlp.router.linear_router"}


def dir_missing_or_empty(path: str) -> bool:
    return (not os.path.exists(path)) or (not os.listdir(path))


@pytest.mark.tp64
@pytest.mark.parametrize(
    "model_path, neuron_config, num_tokens_to_check, divergence_tolerance",
    [
        pytest.param(
            os.getenv("QWEN3_MOE_MODEL_PATH"),
            BASELINE_NEURON_CONFIG,
            15,
            LOGITS_DIVERGENCE_TOLERANCE,
        ),  # BS16 baseline
        # pytest.param(OPTIMIZED_NEURON_CONFIG, 16, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE), # BS16 MoE EP2/TP2, TP4/CP16 for prefill, TP8/DP8 for decode attention
    ],
)
def test_4_layer_accuracy_with_tf(
    model_path, neuron_config, num_tokens_to_check, divergence_tolerance
):
    if not model_path or dir_missing_or_empty(model_path):
        # Use pytest.xfail instead of raising error to mark test as expected failure
        # when model is not available, allowing test suite to continue running
        pytest.xfail(f"Model path {model_path} does not exist")

    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    num_of_layers = int(cfg["num_hidden_layers"])

    BASELINE_NEURON_CONFIG.batch_size = 1
    BASELINE_NEURON_CONFIG.ctx_batch_size = 1
    BASELINE_NEURON_CONFIG.output_logits = True
    BASELINE_NEURON_CONFIG.on_device_sampling_config = OnDeviceSamplingConfig()
    neuron_cfg = BASELINE_NEURON_CONFIG

    router_modules_to_capture = []
    for i in range(num_of_layers):
        router_modules_to_capture.append(f"layers.{i}.mlp.router.linear_router")

    modules_to_capture = []
    modules_to_capture.extend(router_modules_to_capture)

    # capture reference tensors
    num_tokens_to_check = 512
    cpu_dir, neu_dir = run_tensor_capture(
        model_path=model_path,
        neuron_config=neuron_cfg,
        num_tokens_to_check=num_tokens_to_check,
        prompts=DEFAULT_PROMPTS,
        modules_to_capture=modules_to_capture,
    )

    # need to +1 due to 1-indexed system in tensor capture
    tf_map = {i: router_modules_to_capture for i in range(1, num_tokens_to_check + 1)}
    # 3) run tensor-replacement using those captures
    run_accuracy_with_tensor_replacement(
        model_path=model_path,
        neuron_config=neuron_cfg,
        num_tokens_to_check=512,
        prompts=DEFAULT_PROMPTS,
        cpu_dir=cpu_dir,
        neuron_dir=neu_dir,
        tf_map=tf_map,  # use default all-steps mapping
        module_equiv=MODULE_NAME_EQUIV,
    )


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def ensure_empty_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def build_qwen3_neuron(model_path: str, neuron_config: MoENeuronConfig):
    """
    Compile once, load once, wrap with HF generation adapter.
    Returns (raw_neuron_model, generation_adapter, tokenizer).
    """
    config = Qwen3MoeInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B")

    neuron_model = NeuronQwen3MoeForCausalLM(model_path, config)
    compiled_path = os.path.join(model_path, "compiled_checkpoint_accuracy")
    neuron_model.compile(compiled_path)
    neuron_model.load(compiled_path)

    adapter_model = HuggingFaceGenerationAdapter(neuron_model)
    return neuron_model, adapter_model, tokenizer


@contextmanager
def patched_cpu_model_for_capture(
    base_hf_model,
    modules_to_capture: List[str],
    save_dir: str,
):
    """
    Context manager that patches the eager HF model for tensor capture
    and guarantees clean unpatching.
    """
    model, hooks, original_forward = modify_hf_eager_model_for_tensor_capture(
        base_hf_model,
        modules_to_capture=modules_to_capture,
        tensor_capture_save_dir=save_dir,
    )
    try:
        yield model
    finally:
        model.forward = original_forward  # restore
        for h in hooks:
            h.remove()


def capture_for_prompt(
    prompt: str,
    model,
    tokenizer,
    neuron_config: MoENeuronConfig,
    generation_config: GenerationConfig,
    capture_dir: str,
    use_tensor_hook: bool = False,
    num_tokens_to_check: int = 5,
):
    ensure_empty_dir(capture_dir)
    prompts = [prompt] * neuron_config.batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=num_tokens_to_check,
        min_new_tokens=num_tokens_to_check,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        generation_config=generation_config,
    )
    if use_tensor_hook:
        kwargs["tensor_capture_hook"] = get_tensor_capture_hook(tensor_capture_save_dir=capture_dir)
    outputs = model.generate(**kwargs)
    return outputs


def run_tensor_capture(
    model_path: str,
    neuron_config: MoENeuronConfig,
    num_tokens_to_check: int,
    modules_to_capture: List[str],
    prompts: Iterable[str] = DEFAULT_PROMPTS,
):
    """
    - Compiles + loads Neuron once.
    - Loads HF eager model once
    - For each prompt, captures CPU and Neuron tensors into separate dirs.
    """

    base_capture_root = os.path.expanduser(CAPTURE_DIR)
    gen_cfg = GenerationConfig(do_sample=False, pad_token_id=0)
    # enable capture in Neuron config
    neuron_config.tensor_capture_config = TensorCaptureConfig(
        modules_to_capture=modules_to_capture, max_intermediate_tensors=3
    )

    # model build
    neuron_model, hf_adapter, tokenizer = build_qwen3_neuron(model_path, neuron_config)
    # also get a CPU eager model using the same architecture
    cpu_hf_model = neuron_model.load_hf_model(model_path)

    cpu_dir, neu_dir = None, None
    for idx, prompt in enumerate(prompts):
        print(f"[{idx+1}] Prompt: {prompt[:80]}...")
        p_hash = hash_prompt(prompt)

        cpu_dir = os.path.join(base_capture_root, "tensor_capture", "cpu", p_hash)
        neu_dir = os.path.join(base_capture_root, "tensor_capture", "neuron", p_hash)

        # CPU eager capture (temporary patch)
        with patched_cpu_model_for_capture(
            cpu_hf_model,
            modules_to_capture=[
                m.replace("mlp.router.linear_router", "mlp.gate")
                if "mlp.router.linear_router" in m
                else m
                for m in modules_to_capture
            ],
            save_dir=cpu_dir,
        ) as cpu_model_patched:
            capture_for_prompt(
                prompt,
                cpu_model_patched,
                tokenizer,
                neuron_config,
                gen_cfg,
                cpu_dir,
                use_tensor_hook=False,
                num_tokens_to_check=num_tokens_to_check,
            )

        # Neuron capture (adapter path uses tensor_capture_hook)
        capture_for_prompt(
            prompt,
            hf_adapter,
            tokenizer,
            neuron_config,
            gen_cfg,
            neu_dir,
            use_tensor_hook=True,
            num_tokens_to_check=num_tokens_to_check,
        )

        # Clean per-prompt state if Neuron model stashed step counters
        if hasattr(neuron_model, "_tensor_capture_step"):
            delattr(neuron_model, "_tensor_capture_step")
    # torch.save(neu_outs, "/home/ubuntu/ref_neuron.pt")
    print("Tensor capture complete.")
    return cpu_dir, neu_dir


def run_accuracy_with_tensor_replacement(
    model_path: str,
    neuron_config: MoENeuronConfig,
    num_tokens_to_check: int,
    tf_map: Dict[int, List[str]],
    prompts: Iterable[str] = DEFAULT_PROMPTS,
    # Replacement sources + mapping
    cpu_dir: str = "",
    neuron_dir: str = "",
    module_equiv: Optional[Dict[str, str]] = None,
):
    """
    - Enables tensor replacement using pre-captured CPU/Neuron dirs.
    - Captures Neuron outputs under a `tensor_replace/.../neuron/...` folder.
    """
    base_replace_root = os.path.expanduser(CAPTURE_DIR)
    gen_cfg = GenerationConfig(do_sample=False, pad_token_id=0)

    # configure replacement
    neuron_config.tensor_replacement_config = TensorReplacementConfig(
        ref_dir=os.path.expanduser(cpu_dir),
        neuron_dir=os.path.expanduser(neuron_dir),
        tf_map=tf_map,
        module_map=module_equiv or MODULE_NAME_EQUIV,
        neuron_config=copy.deepcopy(neuron_config),
    )

    # model build (compile + load once)
    neuron_model, _, tokenizer = build_qwen3_neuron(model_path, neuron_config)

    for idx, prompt in enumerate(prompts):
        print(f"[{idx+1}] Prompt: {prompt[:80]}...")
        p_hash = hash_prompt(prompt)
        neu_dir_out = os.path.join(base_replace_root, "tensor_replace", "neuron", p_hash)

        check_accuracy_logits(
            neuron_model,
            generation_config=gen_cfg,
            num_tokens_to_check=num_tokens_to_check,
            prompt=prompt,
            tokenizer=tokenizer,
            divergence_difference_tol=LOGITS_DIVERGENCE_TOLERANCE,
            generate_fn_divergence=True,
            tensor_capture_hook=get_tensor_capture_hook(tensor_capture_save_dir=neu_dir_out),
        )

        if hasattr(neuron_model, "_tensor_capture_step"):
            delattr(neuron_model, "_tensor_capture_step")

    return neu_dir_out


if __name__ == "__main__":
    qwen3_moe_model_local_path = "~/models/qwen-3-235b-4layers/"

    test_4_layer_accuracy_with_tf(
        qwen3_moe_model_local_path,
        BASELINE_NEURON_CONFIG,
        512,
        LOGITS_DIVERGENCE_TOLERANCE,
    )  # BS16 baseline
