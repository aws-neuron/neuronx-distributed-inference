# Copyright 2025 © Amazon.com and Affiliates
"""Validate Isaac text-only logits on Neuron against CPU reference.

Loads the compiled Isaac model, runs all 5 text reference prompts,
and compares first-token logit distributions against saved CPU reference .pt files.

Metrics:
- Top-1 token match
- Top-5 / Top-10 overlap
- Cosine similarity of full logit vectors
- Max absolute error

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    export PYTHONPATH=/mnt/models/neuronx-distributed-inference/contrib/models/Isaac-0.2-2B/src:$PYTHONPATH
    python validate_text_logits.py
"""

from isaac_neuron.ndxi_patch import apply_patch

apply_patch()

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, GenerationConfig  # noqa: E402

from neuronx_distributed_inference.models.config import (  # noqa: E402
    NeuronConfig,
    OnDeviceSamplingConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import (  # noqa: E402
    load_pretrained_config,
    HuggingFaceGenerationAdapter,
)
from neuronx_distributed_inference.modules.generation.sampling import (  # noqa: E402
    prepare_sampling_params,
)

from isaac_neuron.modeling_isaac import (  # noqa: E402
    NeuronIsaacForConditionalGeneration,
    IsaacInferenceConfig,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = os.getenv("DATA_HOME", "/mnt/models")
REFERENCE_DIR = f"{DATA_PATH}/reference_outputs"
MODEL_PATH = f"{DATA_PATH}/Isaac-0.2-2B-Preview"
TRACED_MODEL_PATH = f"{DATA_PATH}/traced_model/Isaac-0.2-2B"

# Same prompts as capture_reference.py
TEXT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain quantum entanglement in simple terms:",
    "The meaning of life is",
    "List three primary colors:",
]

# Thresholds
COSINE_SIM_THRESHOLD = 0.99  # BF16 quantization on Neuron vs FP32 CPU
TOP1_MUST_MATCH = True
TOP5_MIN_OVERLAP = 3  # At least 3 of 5 should match
TOP10_MIN_OVERLAP = 5  # At least 5 of 10 should match

# Environment
os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"
torch.manual_seed(42)


def create_neuron_configs():
    """Create text and vision neuron configurations (must match compilation)."""
    text_config = NeuronConfig(
        batch_size=1,
        seq_len=1024,
        torch_dtype=torch.bfloat16,
        tp_degree=1,
        cp_degree=1,
        save_sharded_checkpoint=True,
        skip_sharding=False,
        is_continuous_batching=True,
        ctx_batch_size=1,
        enable_bucketing=True,
        context_encoding_buckets=[1024],
        token_generation_buckets=[1024],
        async_mode=False,
        on_device_sampling_config=OnDeviceSamplingConfig(
            dynamic=True,
            do_sample=True,
            deterministic=True,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            global_topk=256,
            top_k_kernel_enabled=True,
        ),
        output_logits=True,
        fused_qkv=False,
        sequence_parallel_enabled=False,
        attn_kernel_enabled=False,
        attn_tkg_nki_kernel_enabled=False,
        attn_tkg_builtin_kernel_enabled=False,
        qkv_kernel_enabled=False,
        mlp_kernel_enabled=False,
    )

    vision_config = NeuronConfig(
        batch_size=1,
        seq_len=1024,
        torch_dtype=torch.bfloat16,
        tp_degree=1,
        world_size=1,
        save_sharded_checkpoint=True,
        is_continuous_batching=True,
        ctx_batch_size=1,
        enable_bucketing=True,
        buckets=[1],
        fused_qkv=False,
        attn_kernel_enabled=False,
        qkv_kernel_enabled=False,
        mlp_kernel_enabled=False,
    )

    return text_config, vision_config


def load_compiled_model():
    """Load the pre-compiled Isaac model from traced checkpoint."""
    text_config, vision_config = create_neuron_configs()

    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    config = IsaacInferenceConfig(
        text_neuron_config=text_config,
        vision_neuron_config=vision_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )

    print(f"Loading compiled model from {TRACED_MODEL_PATH}...")
    model = NeuronIsaacForConditionalGeneration(TRACED_MODEL_PATH, config)
    model.load(TRACED_MODEL_PATH, skip_warmup=True)
    print("Model loaded successfully.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, padding_side="right", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def compare_logits(neuron_logits, ref_logits, prompt_name):
    """Compare Neuron vs CPU reference logit vectors.

    Args:
        neuron_logits: [vocab_size] float tensor from Neuron
        ref_logits: [vocab_size] float tensor from CPU reference
        prompt_name: string for logging

    Returns:
        dict with all comparison metrics, and bool pass/fail
    """
    neuron_f = neuron_logits.float()
    ref_f = ref_logits.float()

    # Top-1 match
    neuron_top1 = neuron_f.argmax().item()
    ref_top1 = ref_f.argmax().item()
    top1_match = neuron_top1 == ref_top1

    # Top-5 overlap
    neuron_top5 = set(torch.topk(neuron_f, 5).indices.tolist())
    ref_top5 = set(torch.topk(ref_f, 5).indices.tolist())
    top5_overlap = len(neuron_top5 & ref_top5)

    # Top-10 overlap
    neuron_top10 = set(torch.topk(neuron_f, 10).indices.tolist())
    ref_top10 = set(torch.topk(ref_f, 10).indices.tolist())
    top10_overlap = len(neuron_top10 & ref_top10)

    # Cosine similarity
    cosine_sim = F.cosine_similarity(neuron_f.unsqueeze(0), ref_f.unsqueeze(0)).item()

    # Max absolute error
    max_abs_err = (neuron_f - ref_f).abs().max().item()

    # Mean absolute error
    mean_abs_err = (neuron_f - ref_f).abs().mean().item()

    # Pass/fail
    passed = True
    failures = []
    if TOP1_MUST_MATCH and not top1_match:
        passed = False
        failures.append(f"Top-1 mismatch: Neuron={neuron_top1}, CPU={ref_top1}")
    if top5_overlap < TOP5_MIN_OVERLAP:
        passed = False
        failures.append(f"Top-5 overlap {top5_overlap} < {TOP5_MIN_OVERLAP}")
    if top10_overlap < TOP10_MIN_OVERLAP:
        passed = False
        failures.append(f"Top-10 overlap {top10_overlap} < {TOP10_MIN_OVERLAP}")
    if cosine_sim < COSINE_SIM_THRESHOLD:
        passed = False
        failures.append(f"Cosine sim {cosine_sim:.6f} < {COSINE_SIM_THRESHOLD}")

    result = {
        "prompt": prompt_name,
        "passed": passed,
        "top1_match": top1_match,
        "neuron_top1": neuron_top1,
        "ref_top1": ref_top1,
        "top5_overlap": top5_overlap,
        "top10_overlap": top10_overlap,
        "cosine_sim": cosine_sim,
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "failures": failures,
        "neuron_top10_ids": sorted(neuron_top10),
        "ref_top10_ids": sorted(ref_top10),
    }

    return result, passed


def run_validation():
    """Main validation loop."""
    model, tokenizer = load_compiled_model()
    generation_model = HuggingFaceGenerationAdapter(model)

    # Load reference results metadata
    with open(os.path.join(REFERENCE_DIR, "reference_results.json")) as f:
        ref_metadata = json.load(f)

    print(f"\n{'=' * 70}")
    print("TEXT-ONLY LOGIT VALIDATION: Neuron vs CPU Reference")
    print(f"{'=' * 70}")
    print(f"  Reference dir: {REFERENCE_DIR}")
    print(
        f"  Thresholds: cosine>{COSINE_SIM_THRESHOLD}, top1_must_match={TOP1_MUST_MATCH}"
    )
    print(f"  Prompts: {len(TEXT_PROMPTS)}")

    results = []
    all_passed = True

    for i, prompt in enumerate(TEXT_PROMPTS):
        print(f'\n--- Prompt {i}: "{prompt}" ---')

        # Load CPU reference logits
        ref_path = os.path.join(REFERENCE_DIR, f"text_logits_{i:03d}.pt")
        if not os.path.exists(ref_path):
            print(f"  SKIP: Reference file not found: {ref_path}")
            continue
        ref_logits = torch.load(ref_path, map_location="cpu")  # [151936] float32
        print(
            f"  CPU ref: top-1={ref_logits.argmax().item()}, shape={ref_logits.shape}"
        )

        # Tokenize with chat template (matching capture_reference.py)
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        attention_mask = torch.ones_like(input_ids)
        seq_len = input_ids.shape[1]
        print(f"  Input seq_len: {seq_len}")

        # Generate with logit collection
        # We only need 1 new token to get the first-token logits (CTE pass)
        sampling_params = prepare_sampling_params(
            batch_size=1,
            top_k=[1],
            top_p=[1.0],
            temperature=[1.0],  # temperature=1.0 so scores == raw logits
        )

        generation_config = GenerationConfig(
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=1,  # Only need first token
        )

        outputs = generation_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=model.config.neuron_config.max_length,
            sampling_params=sampling_params,
            generation_config=generation_config,
            max_new_tokens=1,
        )

        # Extract first-token logits from scores
        # outputs.scores is a tuple of tensors, one per generated token
        # outputs.scores[0] shape: [batch_size, vocab_size]
        if (
            hasattr(outputs, "scores")
            and outputs.scores is not None
            and len(outputs.scores) > 0
        ):
            neuron_logits = outputs.scores[0][0].float().cpu()  # [vocab_size]
            print(
                f"  Neuron: top-1={neuron_logits.argmax().item()}, shape={neuron_logits.shape}"
            )
        else:
            print(
                "  ERROR: No scores in output. Check output_logits=True in NeuronConfig."
            )
            print(f"  Output type: {type(outputs)}")
            if hasattr(outputs, "__dict__"):
                print(f"  Output attrs: {list(outputs.__dict__.keys())}")
            all_passed = False
            continue

        # Compare
        result, passed = compare_logits(neuron_logits, ref_logits, prompt)
        results.append(result)
        if not passed:
            all_passed = False

        # Print result
        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] cosine={result['cosine_sim']:.6f}, "
            f"top1={'match' if result['top1_match'] else 'MISMATCH'}, "
            f"top5={result['top5_overlap']}/5, top10={result['top10_overlap']}/10, "
            f"max_abs_err={result['max_abs_err']:.4f}"
        )
        if not passed:
            for f in result["failures"]:
                print(f"    FAILURE: {f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"  Passed: {passed_count}/{total}")

    if results:
        avg_cosine = sum(r["cosine_sim"] for r in results) / len(results)
        avg_top5 = sum(r["top5_overlap"] for r in results) / len(results)
        avg_top10 = sum(r["top10_overlap"] for r in results) / len(results)
        print(f"  Avg cosine sim: {avg_cosine:.6f}")
        print(f"  Avg top-5 overlap: {avg_top5:.1f}/5")
        print(f"  Avg top-10 overlap: {avg_top10:.1f}/10")

    if all_passed:
        print("\n  ALL TEXT PROMPTS PASSED")
    else:
        print("\n  SOME PROMPTS FAILED — see details above")
        sys.exit(1)

    # Save results
    out_path = os.path.join(REFERENCE_DIR, "neuron_text_validation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run_validation()
