import os
import json
import shutil
import hashlib
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
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.tensor_capture_utils import get_tensor_capture_hook
from neuronx_distributed.utils.tensor_capture.model_modification import (
    modify_hf_eager_model_for_tensor_capture,
)

from tf_viz import compare_and_plot

# -------------------------
# Constants / Defaults
# -------------------------

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPTS = [
    "A pencil cost $0.50, and an eraser cost $0.25. If you bought 6 pencils and 8 erasers and paid $10, how much change would you get?"
]

# Map CPU module names → Neuron canonical names (optional for replacement)
MODULE_NAME_EQUIV = {"mlp.gate": "mlp.router.linear_router"}

set_random_seed(42)

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

    model = NeuronQwen3MoeForCausalLM(model_path, config)
    compiled_path = os.path.join(model_path, "compiled_checkpoint_accuracy")
    model.compile(compiled_path)
    model.load(compiled_path)

    hf_adapter_model = HuggingFaceGenerationAdapter(model)
    return model, hf_adapter_model, tokenizer


def build_random_hf_model(config_path: str) -> str:
    """
    Saves a tiny HF model with random weights and returns a TemporaryDirectory path.
    The caller owns the lifecycle of the tempdir (call .cleanup()).
    """
    hf_config = AutoConfig.from_pretrained(config_path)
    hf_model = AutoModelForCausalLM.from_config(hf_config, torch_dtype="float32")
    model_tmpdir = tempfile.TemporaryDirectory()
    model_path = model_tmpdir.name
    print(f"Model path {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tmpdir


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
    model,                     # HF-like .generate(...) model (CPU eager or Neuron adapter)
    tokenizer,
    generation_config: GenerationConfig,
    capture_dir: str,
    use_tensor_hook: bool = False,
    num_tokens_to_check: int = 5,
):
    ensure_empty_dir(capture_dir)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
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
        kwargs["tensor_capture_hook"] = get_tensor_capture_hook(
            tensor_capture_save_dir=capture_dir
        )
    model.generate(**kwargs)


def run_tensor_capture(
    model_path: str,
    neuron_config: MoENeuronConfig,
    num_tokens_to_check: int,
    prompts: Iterable[str] = DEFAULT_PROMPTS,
    modules_to_capture: Optional[List[str]] = None,
):
    """
    - Compiles + loads Neuron once.
    - Loads HF eager model once
    - For each prompt, captures CPU and Neuron tensors into separate dirs.
    """
    modules_to_capture = modules_to_capture or [
        "layers.0.mlp.router.linear_router",
        "layers.1.mlp.router.linear_router",
        "layers.2.mlp.router.linear_router",
        "layers.3.mlp.router.linear_router",
    ]
    base_capture_root = os.path.expanduser("~/tensor_replace_example")
    gen_cfg = GenerationConfig(do_sample=False, pad_token_id=0)
    # enable capture in Neuron config
    neuron_config.tensor_capture_config = TensorCaptureConfig(
        modules_to_capture=modules_to_capture
    )

    # model build
    raw_neuron, neuron_model, tokenizer = build_qwen3_neuron(model_path, neuron_config)
    # also get a CPU eager model using the same architecture
    cpu_hf_model = raw_neuron.load_hf_model(model_path)

    cpu_dir, neu_dir = None, None
    for idx, prompt in enumerate(prompts):
        print(f"[{idx+1}] Prompt: {prompt[:80]}...")
        p_hash = hash_prompt(prompt)

        cpu_dir = os.path.join(
            base_capture_root, "tensor_capture", "cpu", p_hash
        )
        neu_dir = os.path.join(
            base_capture_root, "tensor_capture", "neuron", p_hash
        )

        # CPU eager capture (temporary patch)
        with patched_cpu_model_for_capture(
            cpu_hf_model,
            modules_to_capture=[m.replace("mlp.router.linear_router", "mlp.gate") for m in modules_to_capture],
            save_dir=cpu_dir,
        ) as cpu_model_patched:
            capture_for_prompt(
                prompt, cpu_model_patched, tokenizer, gen_cfg, cpu_dir,
                use_tensor_hook=False, num_tokens_to_check=num_tokens_to_check
            )

        # Neuron capture (adapter path uses tensor_capture_hook)
        capture_for_prompt(
            prompt, neuron_model, tokenizer, gen_cfg, neu_dir,
            use_tensor_hook=True, num_tokens_to_check=num_tokens_to_check
        )

        # Clean per-prompt state if Neuron model stashed step counters
        if hasattr(raw_neuron, "_tensor_capture_step"):
            delattr(raw_neuron, "_tensor_capture_step")

    print("Tensor capture complete.")
    return cpu_dir, neu_dir


def run_tensor_replacement(
    model_path: str,
    neuron_config: MoENeuronConfig,
    num_tokens_to_check: int,
    prompts: Iterable[str] = DEFAULT_PROMPTS,
    # Replacement sources + mapping
    cpu_dir: str = "",
    neuron_dir: str = "",
    tf_map: Optional[Dict[int, List[str]]] = None,
    module_equiv: Optional[Dict[str, str]] = None
):
    """
    - Enables tensor replacement using pre-captured CPU/Neuron dirs.
    - Captures Neuron outputs under a `tensor_replace/.../neuron/...` folder.
    """
    base_replace_root = os.path.expanduser("~/tensor_replace_example")
    gen_cfg = GenerationConfig(do_sample=False, pad_token_id=0)

    # reasonable default: replace the same 4 routers for all steps (1..512 as example)
    if tf_map is None:
        tf_map = {i: [
            "layers.0.mlp.router.linear_router",
            "layers.1.mlp.router.linear_router",
            "layers.2.mlp.router.linear_router",
            "layers.3.mlp.router.linear_router",
        ] for i in range(1, 513)}

    # configure replacement
    neuron_config.tensor_replacement_config = TensorReplacementConfig(
        ref_dir=os.path.expanduser(cpu_dir),
        neuron_dir=os.path.expanduser(neuron_dir),
        tf_map=tf_map,
        module_map=module_equiv or MODULE_NAME_EQUIV,
    )

    modules_to_capture = [
        "layers.0.mlp.router.linear_router",
        "layers.1.mlp.router.linear_router",
        "layers.2.mlp.router.linear_router",
        "layers.3.mlp.router.linear_router",
    ]

    neuron_config.tensor_capture_config = TensorCaptureConfig(
        modules_to_capture=modules_to_capture
    )

    # model build (compile + load once)
    raw_neuron, neuron_model, tokenizer = build_qwen3_neuron(model_path, neuron_config)

    for idx, prompt in enumerate(prompts):
        print(f"[{idx+1}] Prompt: {prompt[:80]}...")
        p_hash = hash_prompt(prompt)
        neu_dir_out = os.path.join(
            base_replace_root, "tensor_replace", "neuron", p_hash
        )

        capture_for_prompt(
            prompt, neuron_model, tokenizer, gen_cfg, neu_dir_out,
            use_tensor_hook=True, num_tokens_to_check=num_tokens_to_check
        )

        if hasattr(raw_neuron, "_tensor_capture_step"):
            delattr(raw_neuron, "_tensor_capture_step")

    print("Tensor replacement run complete.")
    return neu_dir_out


# -------------------------
# Example script usage
# -------------------------

if __name__ == "__main__":

    try:
        # 1) create a tiny random HF model to compile against 
        cfg_path = os.path.join(CURR_DIR, "config.json")
        model_cpu = build_random_hf_model(cfg_path)  # TemporaryDirectory
        model_path = model_cpu.name

        neuron_cfg = MoENeuronConfig(
            tp_degree=64,
            batch_size=1,
            max_context_length=512,
            seq_len=5120,
            torch_dtype="bfloat16",
            fused_qkv=True,
            output_logits=True,
            on_device_sampling_config=OnDeviceSamplingConfig(),
        )

        # capture reference tensors
        cpu_dir, neu_dir = run_tensor_capture(
            model_path=model_path,
            neuron_config=neuron_cfg,
            num_tokens_to_check=512,
            prompts=DEFAULT_PROMPTS,
            modules_to_capture=[
                "layers.0.mlp.router.linear_router",
                "layers.1.mlp.router.linear_router",
                "layers.2.mlp.router.linear_router",
                "layers.3.mlp.router.linear_router",
            ]
        )
        steps_to_plot = [4, 25, 40, 100, 254, 409]
        compare_and_plot(cpu_dir, neu_dir, tkg_steps_to_plot=steps_to_plot, max_points=4096)

        # 3) run tensor-replacement using those captures
        neu_tr_dir = run_tensor_replacement(
            model_path=model_path,
            neuron_config=neuron_cfg,
            num_tokens_to_check=512,
            prompts=DEFAULT_PROMPTS,
            cpu_dir=cpu_dir,
            neuron_dir=neu_dir,
            tf_map=None,                 # use default all-steps mapping
            module_equiv=MODULE_NAME_EQUIV
        )
        
        compare_and_plot(cpu_dir, neu_tr_dir, tkg_steps_to_plot=steps_to_plot, max_points=4096)
    finally:
        model_cpu.cleanup()
