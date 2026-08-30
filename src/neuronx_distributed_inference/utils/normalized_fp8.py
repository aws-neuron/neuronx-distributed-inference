"""Validation for explicitly normalized static-FP8 checkpoints."""

from __future__ import annotations

from typing import Any

from neuronx_distributed_inference.modules.checkpoint import (
    inspect_safetensors_sharded,
)


def _expected_names(num_layers: int) -> tuple[set[str], set[str], set[str], set[str]]:
    expected = {
        "embed_tokens.weight",
        "lm_head.weight",
        "norm.weight",
        "norm.weight_unpadded",
        "rank_util.rank",
    }
    fp8_weights: set[str] = set()
    scalar_scales: set[str] = set()
    expert_scales: set[str] = set()

    for layer in range(num_layers):
        prefix = f"layers.{layer}"
        expected.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.input_layernorm.weight_unpadded",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight_unpadded",
                f"{prefix}.self_attn.learned_sinks.sink",
                f"{prefix}.self_attn.rank_util.rank",
                f"{prefix}.feed_forward.moe.router.linear_router.weight",
                f"{prefix}.feed_forward.moe.router.linear_router.bias",
            }
        )
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weight = f"{prefix}.self_attn.{projection}.weight"
            scale = weight.removesuffix(".weight") + ".scale"
            fp8_weights.add(weight)
            scalar_scales.add(scale)
            expected.update(
                {
                    weight,
                    f"{prefix}.self_attn.{projection}.bias",
                    scale,
                }
            )
        for projection in ("down_proj", "gate_up_proj"):
            base = f"{prefix}.feed_forward.moe.expert_mlps.mlp_op.{projection}"
            weight = f"{base}.weight"
            scale = f"{base}.scale"
            fp8_weights.add(weight)
            expert_scales.add(scale)
            expected.update({weight, f"{base}.bias", scale})

    return expected, fp8_weights, scalar_scales, expert_scales


def validate_gpt_oss_120b_normalized_fp8_metadata(
    metadata: dict[str, dict[str, Any]], config: Any
) -> None:
    """Fail closed unless metadata matches the normalized GPT-OSS-120B ABI."""
    neuron_config = config.neuron_config
    if not neuron_config.quantized:
        raise ValueError("normalized_fp8 requires quantized=true")
    if neuron_config.quantization_dtype != "f8e4m3":
        raise ValueError("normalized_fp8 requires quantization_dtype=f8e4m3")
    if neuron_config.fused_qkv:
        raise ValueError("normalized_fp8 requires separate q/k/v projections")
    if neuron_config.vocab_parallel:
        raise ValueError("normalized_fp8 does not support vocab-parallel embedding")
    if config.num_hidden_layers != 36 or config.num_local_experts != 128:
        raise ValueError("normalized_fp8 supports the GPT-OSS-120B layout only")
    if neuron_config.attention_dp_degree != neuron_config.cp_degree:
        raise ValueError("normalized_fp8 requires matching attention DP and CP")

    expected, fp8_weights, scalar_scales, expert_scales = _expected_names(
        config.num_hidden_layers
    )
    actual = set(metadata)
    if actual != expected:
        raise ValueError(
            "normalized_fp8 tensor namespace mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )

    for name in fp8_weights:
        details = metadata[name]
        if details["dtype"] != "F8_E4M3":
            raise ValueError(f"normalized_fp8 weight is not F8_E4M3: {name}")
        if not details["shape"] or any(
            dimension <= 0 for dimension in details["shape"]
        ):
            raise ValueError(f"normalized_fp8 weight has invalid shape: {name}")

    for name in scalar_scales:
        details = metadata[name]
        if details["dtype"] != "F32" or details["shape"] != (1,):
            raise ValueError(
                f"normalized_fp8 scalar scale shape/dtype mismatch: {name}"
            )
        if details["filename"] == "expert-scales.safetensors":
            raise ValueError(f"non-expert scale is in expert scale sidecar: {name}")

    for name in expert_scales:
        expected_shape = (
            (128, 1, 1) if name.endswith("down_proj.scale") else (128, 2, 1)
        )
        details = metadata[name]
        if details["dtype"] != "F32" or details["shape"] != expected_shape:
            raise ValueError(
                f"normalized_fp8 expert scale shape/dtype mismatch: {name}"
            )
        if details["filename"] != "expert-scales.safetensors":
            raise ValueError(f"expert scale is not in expert sidecar: {name}")

    for name, details in metadata.items():
        if name.endswith(".scale"):
            continue
        if details["dtype"] == "F8_E4M3":
            if name not in fp8_weights:
                raise ValueError(f"unexpected FP8 tensor: {name}")
        elif name.endswith(".rank"):
            if details["dtype"] != "I32":
                raise ValueError(f"rank tensor is not I32: {name}")
        elif details["dtype"] not in {"BF16", "F32"}:
            raise ValueError(f"unexpected normalized tensor dtype: {name}")


def validate_gpt_oss_120b_normalized_fp8_checkpoint(
    model_path: str, config: Any
) -> None:
    """Validate the indexed checkpoint before loading any tensor payload."""
    metadata = inspect_safetensors_sharded(model_path)
    filenames = {details["filename"] for details in metadata.values()}
    if "expert-scales.safetensors" in filenames and "model.safetensors" in filenames:
        raise ValueError(
            "normalized_fp8 requires indexed model shards when the expert scale sidecar is present"
        )
    validate_gpt_oss_120b_normalized_fp8_metadata(metadata, config)
