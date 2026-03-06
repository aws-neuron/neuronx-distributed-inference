"""Integration test utilities for Solar Open MoE.

Uses transformers.SolarOpenForCausalLM (available since transformers 5.0.0) as the
CPU reference model for logit accuracy checks against NeuronSolarOpenForCausalLM.

Public API:
- create_tiny_solar_open_model(): write a minimal HF checkpoint via SolarOpenForCausalLM
- get_neuron_config(): return MoENeuronConfig for integration tests
- check_text_accuracy(): compare last-token logits (MAE) between CPU and Neuron model
"""

import os
import sys
from pathlib import Path

import torch

# Add contrib src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from solar_open.modeling_solar_open import (
    NeuronSolarOpenForCausalLM,
    SolarOpenInferenceConfig,
    load_solar_open_config,
)

from neuronx_distributed_inference.models.config import MoENeuronConfig


# ---------------------------------------------------------------------------
# Neuron config for integration tests
# ---------------------------------------------------------------------------


def get_neuron_config() -> MoENeuronConfig:
    """Return MoENeuronConfig for Solar Open integration tests.

    Uses tp_degree=2, moe_tp_degree=2, moe_ep_degree=1 — compatible with
    trn2.3xlarge (2 NeuronCores) and smaller test instances.
    """
    return MoENeuronConfig(
        tp_degree=2,
        moe_tp_degree=2,
        moe_ep_degree=1,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=128,
        max_context_length=112,
        torch_dtype=torch.bfloat16,
        fused_qkv=True,
        flash_decoding_enabled=False,
        output_logits=True,
        enable_bucketing=False,
        sequence_parallel_enabled=False,
        qkv_kernel_enabled=False,
        attn_kernel_enabled=False,
    )


# ---------------------------------------------------------------------------
# Tiny random model factory (uses transformers SolarOpenForCausalLM)
# ---------------------------------------------------------------------------


def create_tiny_solar_open_model(model_dir: str) -> None:
    """Create a tiny random-weight Solar Open checkpoint using transformers 5.0.0.

    Calls SolarOpenForCausalLM(config).save_pretrained(model_dir) which writes:
      - config.json  (model_type="solar_open", all HF fields)
      - model.safetensors  (random weights in Format B: pre-fused 3D expert tensors)
      - generation_config.json  (with transformers_version="5.0.0")

    The weight format is automatically detected by
    convert_solar_open_hf_to_neuron_state_dict via the presence of
    "layers.0.mlp.experts.gate_up_proj" keys (Format B path).

    Args:
        model_dir: Directory to write the checkpoint to.
    """
    from transformers import SolarOpenConfig, SolarOpenForCausalLM

    os.makedirs(model_dir, exist_ok=True)

    config = SolarOpenConfig(
        hidden_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        moe_intermediate_size=256,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        vocab_size=1024,
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=2,
    )

    torch.manual_seed(42)
    model = SolarOpenForCausalLM(config)
    model.save_pretrained(model_dir)


# ---------------------------------------------------------------------------
# Logit accuracy check (CPU transformers reference vs Neuron)
# ---------------------------------------------------------------------------


def check_text_accuracy(
    model_dir: str,
    traced_dir: str,
    neuron_config: MoENeuronConfig,
    tol: float = 0.1,
) -> bool:
    """Compare last-token logits from CPU reference and compiled Neuron model.

    Uses transformers.SolarOpenForCausalLM as the CPU reference (available since
    transformers 5.0.0).  Compares last-token logit vectors via mean absolute error.

    Exact token-ID matching is not used because Neuron's bfloat16 hardware
    arithmetic can produce a different argmax on borderline logits, even when the
    overall logit distribution is very close to the CPU reference.

    Args:
        model_dir: Path to the tiny random model checkpoint.
        traced_dir: Path to the compiled Neuron model.
        neuron_config: MoENeuronConfig used for compilation.
        tol: Maximum allowed mean absolute error on last-token logits.

    Returns:
        True if logit MAE is within tolerance, False otherwise.
    """
    import shutil
    from transformers import SolarOpenForCausalLM

    torch.manual_seed(0)
    input_ids = torch.randint(0, 500, (1, 16), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    # --- CPU Reference (transformers SolarOpenForCausalLM) ---
    hf_model = SolarOpenForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16
    )
    hf_model.eval()

    with torch.no_grad():
        cpu_output = hf_model(input_ids, attention_mask=attention_mask)
    ref_logits = cpu_output.logits.float()  # [1, seq, vocab]
    ref_last = ref_logits[:, -1, :]  # [1, vocab]

    # --- Neuron model ---
    # Copy model.safetensors to traced_dir so load() can find checkpoint weights
    src = os.path.join(model_dir, "model.safetensors")
    dst = os.path.join(traced_dir, "model.safetensors")
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)

    neuron_model = NeuronSolarOpenForCausalLM(traced_dir)
    neuron_model.load(traced_dir)

    with torch.no_grad():
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        output = neuron_model(input_ids, position_ids=position_ids)
        raw_logits = output.logits if hasattr(output, "logits") else output[0]
        if isinstance(raw_logits, (list, tuple)):
            raw_logits = raw_logits[0]
        neuron_logits = raw_logits.float()  # [1, seq, vocab]
    neuron_last = neuron_logits[:, -1, :]  # [1, vocab]

    mae = (ref_last - neuron_last).abs().mean().item()
    cpu_tok = ref_last.argmax(dim=-1).item()
    nrn_tok = neuron_last.argmax(dim=-1).item()
    print(f"  Logit MAE (last token): {mae:.6f} (tol={tol})")
    print(f"  CPU argmax token:    {cpu_tok}")
    print(f"  Neuron argmax token: {nrn_tok}")
    return mae < tol
