import datetime
import os
from typing import Dict, List, Optional, Tuple
import torch
import hashlib
import re
from glob import glob
import tempfile
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset

from neuronx_distributed_inference.models.config import MoENeuronConfig, TensorCaptureConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeInferenceConfig, NeuronQwen3MoeForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter
from neuronx_distributed.utils.tensor_capture.model_modification import modify_hf_eager_model_for_tensor_capture
from neuronx_distributed_inference.utils.tensor_capture_utils import get_tensor_capture_hook
import json
import shutil
from pathlib import Path
import torch.nn.functional as F


torch.manual_seed(42)

# Map functional equivalents between CPU and Neuron modules
MODULE_NAME_EQUIV = {
    "mlp.gate": "mlp.router.linear_router"
}
prompt_margins = {}

def normalize_module_name(name):
    # Apply equivalence mapping on suffix
    for cpu_suffix, neuron_suffix in MODULE_NAME_EQUIV.items():
        if name.endswith(cpu_suffix):
            return name[:-len(cpu_suffix)] + neuron_suffix
    return name


def extract_step_and_module(filename: str):
    match = re.search(r"step_(\d+)_module_(.*)_output.pt", filename)
    if not match:
        return None, None
    step = int(match.group(1))
    module = match.group(2)
    return step, module


def compute_topk_margin(tensor: torch.Tensor, k=8, top1_base=False):
    """
    Returns per-token relative difference between top-(k-1) and top-k logits.

    Args:
        tensor: [..., num_experts] tensor (e.g., [batch, seq_len, num_experts])
        k: number of experts per token (top-k)

    Returns:
        A tensor of shape [...], same as token shape (e.g., [batch, seq_len])
    """
    if tensor.dim() < 2:
        raise ValueError("Tensor must have at least 2 dimensions to sort across experts.")

    sorted_logits, _ = torch.sort(tensor, dim=-1, descending=True)
    topk_1 = sorted_logits[..., k]
    topk = sorted_logits[..., k - 1]
    top1 = sorted_logits[..., 0]
    return ((topk - topk_1).abs() / (topk.abs() + 1e-8)) if not top1_base else ((topk - topk_1).abs() / (top1.abs() + 1e-8))


def compute_tensor_rel_diff(t1: torch.Tensor, t2: torch.Tensor):
    return (torch.norm(t1 - t2) / (torch.norm(t1) + 1e-8)).item()


def same_experts(cpu_logits, neuron_logits, top_k, strict_order=False):
    cpu_aff = F.softmax(cpu_logits, dim=-1, dtype=torch.float64)
    neuron_aff = F.softmax(neuron_logits, dim=-1, dtype=torch.float64)
    cpu_experts = torch.topk(cpu_aff, top_k, dim=-1).indices
    neuron_experts = torch.topk(neuron_aff, top_k, dim=-1).indices

    if strict_order:
        return torch.equal(cpu_experts, neuron_experts)
    else:
        return torch.equal(torch.sort(cpu_experts, dim=-1).values, torch.sort(neuron_experts, dim=-1).values)


def load_and_compare_experts(cpu_dir: str, neuron_dir: str, top_k: int = 8):
    cpu_files = glob(os.path.join(cpu_dir, "*.pt"))
    neuron_files = glob(os.path.join(neuron_dir, "*.pt"))

    # Build index: (step, normalized_module_name) → filepath
    def build_index(file_list, for_cpu=True):
        index = {}
        for f in file_list:
            step, module = extract_step_and_module(os.path.basename(f))
            if step is not None:
                norm_module = module
                if for_cpu:
                    norm_module = normalize_module_name(module)
                index[(step, norm_module)] = f
        return index

    cpu_index = build_index(cpu_files, for_cpu=True)
    neuron_index = build_index(neuron_files, for_cpu=False)

    common_keys = set(cpu_index.keys()) & set(neuron_index.keys())
    if not common_keys:
        raise ValueError("Cpu and Neuron have no common steps and modules")

    for key in sorted(common_keys):
        step, module = key
        cpu_tensor = torch.load(cpu_index[key])
        neuron_tensor = torch.load(neuron_index[key])

        cpu_shape = cpu_tensor.shape
        neuron_shape = neuron_tensor.shape

        # Squeeze batch dim if it's size 1 and there's one extra dim
        if neuron_shape[0] == 1 and len(neuron_shape) == len(cpu_shape) + 1:
            neuron_tensor = neuron_tensor.squeeze(0)
            neuron_shape = neuron_tensor.shape  # update shape
            print(f"Squeezed batch dim from Neuron tensor {neuron_shape}")

        if cpu_shape != neuron_shape:
            if step == 1 and all(n >= c for n, c in zip(neuron_shape, cpu_shape)):
                slices = tuple(slice(0, c) for c in cpu_shape)
                neuron_tensor = neuron_tensor[slices]
                print(f"Sliced Neuron tensor for module '{module}' at step {step} from {neuron_shape} → {cpu_shape}")
            else:
                raise ValueError(f"CPU and Neuron shapes do not match cpu shape: {cpu_shape}, neuron shape: {neuron_shape}")

        def maybe_slice_last_token(tensor):
            if tensor.dim() == 2:
                print("Sliced experts for last token")
                return tensor[-1, :].unsqueeze(0)  # [1, E]
            else:
                raise ValueError(f"Unexpected tensor shape for step_1: {tensor.shape}")

        if step == 1:
            cpu_tensor = maybe_slice_last_token(cpu_tensor)
            neuron_tensor = maybe_slice_last_token(neuron_tensor)


        if not same_experts(cpu_tensor, neuron_tensor, top_k):
            return False  # mismatch found

    return True  # all matching


def find_prompts_with_same_experts(cpu_base_dir, neuron_base_dir):
    prompt_hashes = os.listdir(neuron_base_dir)
    stable_prompts = []

    print(f"Number of prompts {len(prompt_hashes)}")

    for prompt_hash in prompt_hashes:
        cpu_dir = os.path.join(cpu_base_dir, prompt_hash)
        neuron_dir = os.path.join(neuron_base_dir, prompt_hash)
        if not os.path.isdir(cpu_dir):
            raise ValueError(f"Missing hash in cpu dir {prompt_hash}")

        if load_and_compare_experts(cpu_dir, neuron_dir, top_k=8):
            stable_prompts.append(prompt_hash)

    return stable_prompts


def compute_rel_diff(t1: torch.Tensor, t2: torch.Tensor):
    return ((t1 - t2).abs() / (t1.abs() + 1e-8)).mean().item()


def precompute_margins(cpu_dir, neuron_dir, top_k=8, max_steps=512):

    if not os.path.isdir(cpu_dir):
        raise ValueError(f"CPU dir is expected to be present {cpu_dir}")
    if not os.path.isdir(neuron_dir):
        raise ValueError(f"Neuron dir is expected to be present {neuron_dir}")
    if not (os.path.basename(cpu_dir) == os.path.basename(neuron_dir)):
        raise ValueError(f"Prompt hashes do not match between cpu and neuron cpu hash: {os.path.basename(cpu_dir)}, neuron hash: {os.path.basename(neuron_dir)}")
    

    prompt_hash = os.path.basename(cpu_dir)
    cpu_files = glob(os.path.join(cpu_dir, "*.pt"))
    neuron_files = glob(os.path.join(neuron_dir, "*.pt"))

    def build_index(file_list, for_cpu=True):
        index = {}
        for f in file_list:
            step, module = extract_step_and_module(os.path.basename(f))
            if step is not None and step <= max_steps:
                norm_module = normalize_module_name(module) if for_cpu else module
                index[(step, norm_module)] = f
        return index

    cpu_index = build_index(cpu_files, for_cpu=True)
    neuron_index = build_index(neuron_files, for_cpu=False)
    common_keys = sorted(set(cpu_index.keys()) & set(neuron_index.keys()))
    if not common_keys:
        raise ValueError("No common keys found between cpu and neuron")

    # Compute per-step min margin
    margins_by_step = {}

    for (step, module) in common_keys:
        print(f"Running prompt hash {prompt_hash} for step {step}")
        cpu_tensor = torch.load(cpu_index[(step, module)])
        neuron_tensor = torch.load(neuron_index[(step, module)])

        if neuron_tensor.dim() == cpu_tensor.dim() + 1 and neuron_tensor.shape[0] == 1:
            neuron_tensor = neuron_tensor.squeeze(0)

        if cpu_tensor.shape != neuron_tensor.shape:
            if step == 1:
                slices = tuple(slice(0, s) for s in cpu_tensor.shape)
                neuron_tensor = neuron_tensor[slices]
            else:
                raise ValueError(f"Shape mismatch at {step}, {module}")

        if step == 1:
            cpu_tensor = cpu_tensor[-1, :].unsqueeze(0)
            neuron_tensor = neuron_tensor[-1, :].unsqueeze(0)
        
        neuron_margin = compute_topk_margin(neuron_tensor, top_k)
        margin = neuron_margin.item() #torch.minimum(cpu_margin, neuron_margin).min().item()

        if step not in margins_by_step:
            margins_by_step[step] = margin
        else:
            raise ValueError(f"Margin for step {step} already exists, we should not have reached this condition")
    prompt_margins[prompt_hash] = margins_by_step

    return prompt_margins



def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def save_checkpoint(config_path, torch_dtype):
    hf_config = AutoConfig.from_pretrained(config_path)
    hf_model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch_dtype)
    model_tempdir = tempfile.TemporaryDirectory()
    hf_model.save_pretrained(model_tempdir.name)
    return model_tempdir


def find_prompt_by_hash(
    target_hash: str,
    dataset_name="openai/gsm8k",  # or any HF dataset
) -> Optional[str]:
    """
    Stream a Hugging Face dataset and finds the first prompt that hashes (SHA-256) to `target_hash`.

    Returns:
        Prompt else None.
    """

    for prompt in stream_hf_prompts(dataset_name):
        if not isinstance(prompt, str):
            raise ValueError(f"Found prompt that is not a string {prompt}")
        if hash_prompt(prompt) == target_hash:
            return prompt
    return None


def stream_hf_prompts(dataset_name: str, prompt_key: str = "question", max_prompts: int = None):
    """Yields prompts from a Hugging Face dataset."""
    dataset = load_dataset(dataset_name, "main", split="test")
    count = 0
    for example in dataset:
        if max_prompts and count >= max_prompts:
            break
        prompt = example[prompt_key]
        yield prompt
        count += 1


def capture_for_prompt(
    prompt: str,
    model,
    tokenizer,
    generation_config,
    capture_dir: str,
    use_tensor_hook: bool = False,
    num_tokens_to_check=5
):
    
    # Delete only this specific prompt dir (not others)
    if os.path.exists(capture_dir):
        shutil.rmtree(capture_dir)

    os.makedirs(capture_dir, exist_ok=True)
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
        kwargs["tensor_capture_hook"] = get_tensor_capture_hook(tensor_capture_save_dir=capture_dir)

    model.generate(**kwargs)



def get_top_n_prompts_upto_step(step: int, n: int = 5):

    if not prompt_margins:
        return {}
    
    # build cumulative min per prompt
    cumulative_margins: Dict[str, Dict[int, float]] = {}
    for prompt, step_margin_map in prompt_margins.items():
        if not step_margin_map:
            raise ValueError(f"Expected step map for prompt {prompt} not present : {step_margin_map}")
        
        cumulative_margins[prompt] = {}
        running_min = float("inf")
        argmin_step = -1
        for s in sorted(step_margin_map):
            if s <= step:
                if step_margin_map[s] < running_min:
                    running_min = min(running_min, step_margin_map[s])
                    argmin_step = s
                cumulative_margins[prompt][s] = (running_min, argmin_step)

    leaderboards: Dict[int, List[Tuple[str, float, int]]] = {}

    # Find, per step, n prompts with highest top-k margins at said step
    for s in range(1, step+1):
        row: List[Tuple[str, float, int]] = []
        for prompt, cum_map  in cumulative_margins.items():
            cum_min, argmin_s = cum_map[s]
            row.append((prompt, cum_min, argmin_s))

        if not row:
            raise ValueError(f"Prompt {prompt} dpes not have row to append at step {s}")
        
        row.sort(key=lambda t: t[1], reverse=True)
        leaderboards[s] = row[:n]
    return leaderboards


def save_prompt_margins(prompt_margins, filename_prefix="neuron_prompt_topk_margins"):
    """
    Serialize and save prompt_margins dictionary as JSON in the same directory
    as the current script.
    """
    # Ensure everything is JSON serializable (convert tensors/numpy to float)
    serializable = {
        prompt: {step: float(margin) for step, margin in step_map.items()}
        for prompt, step_map in prompt_margins.items()
    }

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Build timestamped filename
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{filename_prefix}_{ts}.json"
    out_path = os.path.join(script_dir, filename)

    # Save JSON
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"prompt_margins saved to {out_path}")
    return out_path


def run_dataset_capture(tp_degree, batch_size, max_context_length, seq_len, torch_dtype, fused_qkv, dataset_name: str, prompt_key: str, max_prompts=1, num_tokens_to_check=512):

    # Init model, config, tokenizer
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    model_tempdir = save_checkpoint(config_path, torch_dtype)
    model_path = model_tempdir.name

    tensor_capture_config = TensorCaptureConfig(modules_to_capture=["layers.0.mlp.router.linear_router"])
    neuron_config = MoENeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        max_context_length=max_context_length,
        seq_len=seq_len,
        torch_dtype=torch_dtype,
        fused_qkv=fused_qkv,
        tensor_capture_config=tensor_capture_config,
        on_device_sampling_config=OnDeviceSamplingConfig()
    )
    config = Qwen3MoeInferenceConfig(neuron_config, load_config=load_pretrained_config(model_path))
    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B")

    # Compile and load Neuron model ONCE
    neuron_raw_model = NeuronQwen3MoeForCausalLM(model_path, config)
    compiled_model_path = model_path + "/compiled_checkpoint_accuracy"
    neuron_raw_model.compile(compiled_model_path)
    neuron_raw_model.load(compiled_model_path)
    neuron_model = HuggingFaceGenerationAdapter(neuron_raw_model)

    base_capture_root = os.path.expanduser("~/qwen3")
    base_hf_model = neuron_raw_model.load_hf_model(model_path)

    try:
        # Iterate over streamed prompts
        for idx, prompt in enumerate(stream_hf_prompts(dataset_name, prompt_key=prompt_key, max_prompts=max_prompts)):
            print(f"[{idx + 1}] Processing prompt: {prompt[:80]}...")

            # Hash and setup dirs
            prompt_hash = hash_prompt(prompt)
            cpu_capture_path = os.path.join(base_capture_root, "tensor_capture", "cpu", f"{torch_dtype}", prompt_hash)
            neuron_capture_path = os.path.join(base_capture_root, "tensor_capture", "neuron", f"{torch_dtype}", prompt_hash)

            # Clone and hook HF model with prompt-specific tensor_capture_save_dir
            hf_model, hooks, original_forward = modify_hf_eager_model_for_tensor_capture(
                base_hf_model,  # rewrap the base model for this prompt
                modules_to_capture=["layers.0.mlp.gate"],
                tensor_capture_save_dir=cpu_capture_path,
            )

            # Capture for cpu
            try:
                capture_for_prompt(prompt, hf_model, tokenizer, generation_config, cpu_capture_path, use_tensor_hook=False, num_tokens_to_check=num_tokens_to_check)
            finally:
                # Remove hooks and patched forward after capture
                hf_model.forward = original_forward
                for h in hooks:
                    print(f"Removing forward hook {h}")
                    h.remove()

            # Capture for neuron
            capture_for_prompt(prompt, neuron_model, tokenizer, generation_config, neuron_capture_path, use_tensor_hook=True, num_tokens_to_check=num_tokens_to_check)
            # Remove capture step for next iteration
            if hasattr(neuron_raw_model, '_tensor_capture_step'):
                print(f"Removing attr _tensor_capture_step from {neuron_raw_model._get_name()}")
                delattr(neuron_raw_model, '_tensor_capture_step')

            # Compute relative error
            precompute_margins(cpu_capture_path, neuron_capture_path, max_steps=num_tokens_to_check)
        save_prompt_margins(prompt_margins, filename_prefix=f'neuron_prompt_topk_margins_{torch_dtype}')
        leaderboards = get_top_n_prompts_upto_step(num_tokens_to_check)

        serializable = {
        step: [
            {"prompt": prompt, "margin": margin, "argmin_step": argmin_step}
            for prompt, margin, argmin_step in entries
        ]
            for step, entries in leaderboards.items()
        }

        # Build timestamped filename
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"topk_margin_leaderboards_1layer_{torch_dtype}_{ts}.json"
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(script_dir, filename)

        # Write to file
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
                
    finally:
        model_tempdir.cleanup()


if __name__ == "__main__":
    # bf16 per prompt per step margins saved to - s3://yellv-neuron-integration/qwen3-stable-prompts/neuron_prompt_topk_margins_bf16_2025-08-20_00-29-56.json
    # bf16 leaderboard prompts per step saved to - s3://yellv-neuron-integration/qwen3-stable-prompts/topk_margin_leaderboards_1layer_bf16_2025-08-20_00-32-21.json
    run_dataset_capture(
        tp_degree=32,
        batch_size=1,
        max_context_length=512,
        seq_len=5120,
        torch_dtype="float16",
        fused_qkv=True,
        dataset_name="openai/gsm8k",  # or any HF dataset
        prompt_key="question",         # adjust to dataset schema
        max_prompts=None,
        num_tokens_to_check=512
    )
 