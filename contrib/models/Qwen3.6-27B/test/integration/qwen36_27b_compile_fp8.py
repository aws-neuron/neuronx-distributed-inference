#!/usr/bin/env python3
"""Compile Qwen3.6-27B 64K with scoped weight-mode ablations.

This script intentionally starts from the validated 64K hybrid/chunked-prefill
baseline and changes only weight quantization. Supported modes:

* ``fp8_mlp_only``: MLP linear weights are converted to FP8 while attention,
  DeltaNet, normalization, embeddings, lm_head, KV cache, and recurrent state
  remain BF16.
* ``fp8_full``: all supported linear/matmul weights are converted to FP8 while
  embeddings, normalization, rotary state, DeltaNet recurrent/conv state, KV
  cache, and lm_head remain BF16 by default.
* ``bf16_control``: no FP8 conversion; this is the real-token host-logits
  control for separating FP8 conversion failures from serving/logits failures.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch


_FP8_ENV_DEFAULTS = {
    "XLA_HANDLE_SPECIAL_SCALAR": "1",
    "UNSAFE_FP8FNCAST": "1",
}

_WEIGHT_DTYPE_FP8_MLP_ONLY = "fp8_mlp_only"
_WEIGHT_DTYPE_FP8_FULL = "fp8_full"
_FP8_EXCLUDE_GROUPS = {
    "linear_attn",
    "linear_attn_qkv",
    "linear_attn_z",
    "linear_attn_out_proj",
    "mlp",
    "self_attn",
    "self_attn_qkv",
    "self_attn_o_proj",
}
_WEIGHT_DTYPE_BF16_CONTROL = "bf16_control"
_FP8_WLO_SKIP_PATTERNS = [
    r".*\.scale$",
    r".*\.weight_scale$",
    r".*linear_attn\.conv1d_weight\.weight$",
]
_DISABLE_TOKEN_GENERATION_WLO_ENV = "QWEN36_DISABLE_TOKEN_GENERATION_WLO"
_DELTANET_CTE_BACKEND_ENV = {
    "USE_NKI_FUSED",
    "USE_NKI_CHUNKED",
    "USE_NKI",
    "DELTANET_SEQUENTIAL",
    "USE_PYTORCH_CHUNK",
}


def _ensure_fp8_environment() -> None:
    for name, value in _FP8_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)


def _repo_root(path: str | None) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    return Path(__file__).resolve().parents[5]


def _load_text_config(model_path: Path) -> dict:
    with (model_path / "config.json").open() as f:
        full_config = json.load(f)
    text_config = full_config.get("text_config", full_config)
    config_dict = dict(text_config)
    config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
    if "rope_parameters" in text_config:
        config_dict["rope_theta"] = text_config["rope_parameters"].get(
            "rope_theta", 10000000
        )
    config_dict.setdefault("tie_word_embeddings", False)
    return config_dict


def _sanitize_reloadable_neuron_config(compiled_path: Path) -> None:
    """Keep direct-cast KV quant config reloadable after JSON serialization."""
    config_path = compiled_path / "neuron_config.json"
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text())
    neuron_config = config.get("neuron_config", config)
    kv_quant_config = neuron_config.get("kv_quant_config")
    if not isinstance(kv_quant_config, dict):
        return
    if not kv_quant_config.get("direct_cast", True):
        return

    # Neuron serializes QuantizationType enum defaults as nested JSON objects.
    # KVQuantizationConfig expects real enum values on reload, so omit those
    # fields and let the constructor restore its per-tensor symmetric defaults.
    neuron_config["kv_quant_config"] = {"direct_cast": True}
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def _compiled_parameter_dtype(inf_config) -> torch.dtype:
    dtype = getattr(inf_config.neuron_config, "torch_dtype", torch.bfloat16)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        dtype_name = dtype.removeprefix("torch.")
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype_name, torch.bfloat16)
    return torch.bfloat16


def _hybrid_cache_torch_dtype(value, default: torch.dtype) -> torch.dtype:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    normalized = str(value).lower().removeprefix("torch.")
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return default


def _ensure_hybrid_checkpoint_weights(compiled_path: Path, inf_config) -> None:
    """Add reloadable zero checkpoint-bank tensors when NxD omits them."""
    gdn_layer_ids = [
        idx
        for idx, layer_type in enumerate(getattr(inf_config, "layer_types", ()))
        if layer_type == "linear_attention"
    ]
    weights_dir = compiled_path / "weights"
    if not gdn_layer_ids or not weights_dir.exists():
        return

    from safetensors import safe_open  # noqa: WPS433
    from safetensors.torch import load_file, save_file  # noqa: WPS433

    tp_degree = int(inf_config.neuron_config.tp_degree)
    local_num_value_heads = int(inf_config.linear_num_value_heads) // tp_degree
    local_num_key_heads = int(inf_config.linear_num_key_heads) // tp_degree
    key_dim = int(inf_config.linear_key_head_dim)
    value_dim = int(inf_config.linear_value_head_dim)
    slots = int(inf_config.max_gdn_checkpoint_slots)
    conv_dim = 2 * local_num_key_heads * key_dim + local_num_value_heads * value_dim
    conv_state_len = int(inf_config.linear_conv_kernel_dim) - 1
    default_param_dtype = _compiled_parameter_dtype(inf_config)
    recurrent_param_dtype = _hybrid_cache_torch_dtype(
        getattr(
            inf_config,
            "hybrid_recurrent_cache_dtype",
            getattr(inf_config, "gdn_recurrent_cache_dtype", None),
        ),
        torch.float32,
    )
    conv_param_dtype = _hybrid_cache_torch_dtype(
        getattr(
            inf_config,
            "hybrid_conv_cache_dtype",
            getattr(inf_config, "gdn_conv_cache_dtype", None),
        ),
        default_param_dtype,
    )

    recurrent_shape = (slots, local_num_value_heads, key_dim, value_dim)
    conv_shape = (slots, conv_dim, conv_state_len)
    recurrent_keys = [
        f"hybrid_gdn_checkpoint_cache.recurrent_slots.{idx}"
        for idx in range(len(gdn_layer_ids))
    ]
    conv_keys = [
        f"hybrid_gdn_checkpoint_cache.conv_slots.{idx}"
        for idx in range(len(gdn_layer_ids))
    ]

    for shard in sorted(weights_dir.glob("tp*_sharded_checkpoint.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            existing = set(handle.keys())
            metadata = handle.metadata()
        missing_recurrent = [key for key in recurrent_keys if key not in existing]
        missing_conv = [key for key in conv_keys if key not in existing]
        if not missing_recurrent and not missing_conv:
            continue

        tensors = load_file(shard, device="cpu")
        for key in missing_recurrent:
            tensors[key] = torch.zeros(recurrent_shape, dtype=recurrent_param_dtype)
        for key in missing_conv:
            tensors[key] = torch.zeros(conv_shape, dtype=conv_param_dtype)

        tmp_path = shard.with_suffix(shard.suffix + ".tmp")
        save_file(tensors, tmp_path, metadata=metadata)
        os.replace(tmp_path, shard)
        print(
            "CHECKPOINT_BANK_WEIGHTS_ADDED",
            shard.name,
            len(missing_recurrent),
            len(missing_conv),
            str(recurrent_param_dtype),
            str(conv_param_dtype),
            flush=True,
        )


def _parse_int_list(values: list[str] | None) -> list[int] | None:
    if values is None:
        return None
    tokens: list[str] = []
    for value in values:
        tokens.extend(value.replace(",", " ").split())
    return [int(token) for token in tokens]


def _parse_bucket_pairs(values: list[str] | None) -> list[tuple[int, int]] | None:
    if values is None:
        return None
    pairs: list[tuple[int, int]] = []
    for value in values:
        for token in value.replace(",", " ").split():
            if ":" in token:
                active, prefix = token.split(":", 1)
            elif "x" in token:
                active, prefix = token.split("x", 1)
            else:
                raise ValueError(
                    "--context-encoding-bucket-pairs entries must use "
                    f"ACTIVE:PREFIX syntax, got {token!r}"
                )
            pairs.append((int(active), int(prefix)))
    return pairs


def _cte_buckets(args: argparse.Namespace) -> list[int]:
    buckets = _parse_int_list(args.cte_buckets) or [args.cte_bucket]
    buckets = sorted(set(buckets))
    if not buckets:
        raise ValueError("At least one CTE bucket is required")
    for bucket in buckets:
        if bucket <= 0:
            raise ValueError(f"CTE buckets must be positive, got {bucket}")
        if bucket % 128 != 0:
            raise ValueError(
                f"CTE bucket {bucket} is not 128-aligned; DeltaNet CTE uses 128-token chunks"
            )
    if buckets[-1] > args.seq_len:
        raise ValueError(
            f"Largest CTE bucket {buckets[-1]} exceeds --seq-len {args.seq_len}"
        )
    return buckets


def _prefix_buckets(args: argparse.Namespace, cte_buckets: list[int]) -> list[int]:
    buckets = _parse_int_list(args.prefix_buckets) or cte_buckets
    buckets = sorted(set(buckets))
    if not buckets:
        raise ValueError("At least one prefix bucket is required")
    for bucket in buckets:
        if bucket <= 0:
            raise ValueError(f"Prefix buckets must be positive, got {bucket}")
        if bucket % args.block_size != 0:
            raise ValueError(
                f"Prefix bucket {bucket} must be divisible by block size {args.block_size}"
            )
    if buckets[-1] > args.seq_len:
        raise ValueError(
            f"Largest prefix bucket {buckets[-1]} exceeds --seq-len {args.seq_len}"
        )
    return buckets


def _context_encoding_bucket_pairs(
    args: argparse.Namespace,
    cte_buckets: list[int],
    prefix_buckets: list[int],
) -> list[list[int]] | None:
    raw_pairs = _parse_bucket_pairs(args.context_encoding_bucket_pairs)
    if raw_pairs is None:
        return None

    cte_bucket_set = set(cte_buckets)
    prefix_bucket_set = set(prefix_buckets)
    pairs = set()
    if not getattr(args, "omit_zero_prefix_pair", False):
        pairs.update((cte_bucket, 0) for cte_bucket in cte_buckets)
    for active_tokens, prefix_tokens in raw_pairs:
        if active_tokens not in cte_bucket_set:
            raise ValueError(
                "--context-encoding-bucket-pairs active bucket must be present "
                f"in --cte-buckets, got {active_tokens} with {cte_buckets}"
            )
        if prefix_tokens < 0:
            raise ValueError(
                "--context-encoding-bucket-pairs prefix bucket must be "
                f"non-negative, got {prefix_tokens}"
            )
        if prefix_tokens > 0 and prefix_tokens not in prefix_bucket_set:
            raise ValueError(
                "--context-encoding-bucket-pairs prefix bucket must be 0 or "
                f"present in --prefix-buckets, got {prefix_tokens} with "
                f"{prefix_buckets}"
            )
        pairs.add((active_tokens, prefix_tokens))

    prefix_order = {0: 0}
    prefix_order.update(
        {prefix_bucket: index + 1 for index, prefix_bucket in enumerate(prefix_buckets)}
    )
    cte_order = {cte_bucket: index for index, cte_bucket in enumerate(cte_buckets)}
    return [
        [active_tokens, prefix_tokens]
        for active_tokens, prefix_tokens in sorted(
            pairs,
            key=lambda pair: (cte_order[pair[0]], prefix_order[pair[1]]),
        )
    ]


def _token_generation_buckets(args: argparse.Namespace) -> list[int]:
    buckets = _parse_int_list(args.token_generation_buckets) or [args.seq_len]
    buckets = sorted(set(buckets))
    if not buckets:
        raise ValueError("At least one token-generation bucket is required")
    for bucket in buckets:
        if bucket <= 0:
            raise ValueError(
                f"Token-generation buckets must be positive, got {bucket}"
            )
        if bucket > args.seq_len:
            raise ValueError(
                f"Token-generation bucket {bucket} exceeds --seq-len {args.seq_len}"
            )
    return buckets


def _token_generation_batches(args: argparse.Namespace) -> list[int] | None:
    batches = _parse_int_list(args.token_generation_batches)
    if batches is None:
        return None
    batches = sorted(set(batches))
    if not batches:
        raise ValueError("Token-generation batches cannot be empty")
    for batch in batches:
        if batch <= 0:
            raise ValueError(
                f"Token-generation batches must be positive, got {batch}"
            )
        if batch > args.max_num_seqs:
            raise ValueError(
                f"Token-generation batch {batch} exceeds --max-num-seqs "
                f"{args.max_num_seqs}"
            )
    return batches


def _weights_to_skip_layout_optimization(args: argparse.Namespace) -> list[str]:
    patterns: list[str] = []
    if args.weight_dtype in (_WEIGHT_DTYPE_FP8_MLP_ONLY, _WEIGHT_DTYPE_FP8_FULL):
        patterns.extend(_FP8_WLO_SKIP_PATTERNS)
    patterns.extend(getattr(args, "weights_to_skip_layout_optimization", None) or [])
    return list(dict.fromkeys(patterns))


def _disable_token_generation_wlo(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "disable_token_generation_wlo", False)) or (
        os.environ.get(_DISABLE_TOKEN_GENERATION_WLO_ENV) == "1"
    )


def _validate_prefix_buckets_fit_context(
    args: argparse.Namespace,
    max_context_length: int,
    prefix_buckets: list[int],
) -> None:
    if not (args.enable_prefix_caching or args.enable_hybrid_apc):
        return
    if prefix_buckets[-1] > max_context_length:
        raise ValueError(
            f"Largest prefix bucket {prefix_buckets[-1]} exceeds "
            f"--max-context-length {max_context_length}. Long-context APC needs "
            "--max-context-length to cover the largest reusable prefix bucket."
        )


def _max_context_length(args: argparse.Namespace, cte_buckets: list[int]) -> int:
    max_context_length = args.max_context_length or cte_buckets[-1]
    if max_context_length < cte_buckets[-1]:
        raise ValueError(
            f"--max-context-length {max_context_length} is smaller than largest "
            f"CTE bucket {cte_buckets[-1]}"
        )
    if max_context_length > args.seq_len:
        raise ValueError(
            f"--max-context-length {max_context_length} exceeds --seq-len {args.seq_len}"
        )
    return max_context_length


def _pa_min_blocks(args: argparse.Namespace) -> int:
    return max(
        1,
        ((args.seq_len + args.block_size - 1) // args.block_size)
        * args.max_num_seqs,
    )


def _pa_requested_blocks(args: argparse.Namespace) -> int:
    min_blocks = _pa_min_blocks(args)
    if args.pa_num_blocks is None:
        requested_blocks = min_blocks + max(0, int(args.pa_headroom_blocks))
    else:
        requested_blocks = int(args.pa_num_blocks)
    if requested_blocks < min_blocks:
        raise ValueError(
            f"--pa-num-blocks {requested_blocks} is too small for seq_len="
            f"{args.seq_len} and block_size={args.block_size}; need at least {min_blocks}"
        )
    return requested_blocks


def _pa_num_blocks(args: argparse.Namespace) -> int:
    # Keep this value identical to vLLM's --num-gpu-blocks-override /
    # NeuronConfig.pa_num_blocks contract. NxDI's BlockKVCacheManager accounts
    # for its own reserved internal block when prefix caching is enabled.
    return _pa_requested_blocks(args)


def _configure_base_compile_work_dir(
    compiled_path: Path,
    requested_work_dir: str | None,
) -> Path:
    if requested_work_dir:
        work_dir = Path(requested_work_dir).expanduser().resolve()
    else:
        existing_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR")
        if existing_work_dir:
            work_dir = Path(existing_work_dir).expanduser().resolve()
        else:
            work_dir = (compiled_path.parent / "_nxd_model_workdir").resolve()

    work_dir.mkdir(parents=True, exist_ok=True)
    os.environ["BASE_COMPILE_WORK_DIR"] = str(work_dir)
    return work_dir


def _configure_deltanet_cte_backend(backend: str) -> None:
    """Select the DeltaNet CTE implementation used while tracing the artifact."""
    if backend == "env":
        return

    for name in _DELTANET_CTE_BACKEND_ENV:
        os.environ.pop(name, None)

    if backend == "fused":
        os.environ["USE_NKI_FUSED"] = "1"
    elif backend == "nki_chunked":
        os.environ["USE_NKI_FUSED"] = "0"
        os.environ["USE_NKI_CHUNKED"] = "1"
    elif backend == "pytorch_chunk":
        os.environ["USE_NKI_FUSED"] = "0"
        os.environ["USE_PYTORCH_CHUNK"] = "1"
    elif backend == "sequential":
        os.environ["USE_NKI_FUSED"] = "0"
        os.environ["DELTANET_SEQUENTIAL"] = "1"
    elif backend == "nki_recurrent":
        os.environ["USE_NKI_FUSED"] = "0"
        os.environ["USE_NKI"] = "1"
    else:
        raise ValueError(f"Unsupported DeltaNet CTE backend: {backend}")


def _mlp_only_modules_to_not_convert(num_layers: int) -> list[str]:
    """Exclude numerically sensitive or unsupported modules from FP8 conversion."""
    modules = [
        "embed_tokens",
        "model.embed_tokens",
        "lm_head",
        "norm",
        "model.norm",
        "rotary_emb",
        "model.rotary_emb",
    ]
    for layer_idx in range(num_layers):
        for prefix in ("layers", "model.layers"):
            modules.extend(
                [
                    f"{prefix}.{layer_idx}.self_attn",
                    f"{prefix}.{layer_idx}.linear_attn",
                    f"{prefix}.{layer_idx}.input_layernorm",
                    f"{prefix}.{layer_idx}.post_attention_layernorm",
                ]
            )
    return modules


def _full_fp8_modules_to_not_convert(
    num_layers: int,
    *,
    quantize_lm_head: bool,
    quantize_linear_attn_gates: bool = False,
    fp8_exclude_groups: set[str] | None = None,
) -> list[str]:
    """Exclude non-linear or sensitive modules from full FP8 conversion.

    This follows the common NVIDIA/vLLM policy: quantize eligible Linear
    matmuls, keep lm_head in higher precision unless explicitly requested, and
    keep normalization/cache/state tensors unquantized.
    """
    fp8_exclude_groups = fp8_exclude_groups or set()
    modules = [
        "embed_tokens",
        "model.embed_tokens",
        "norm",
        "model.norm",
        "rotary_emb",
        "model.rotary_emb",
        "mrope_emb",
        "model.mrope_emb",
    ]
    if not quantize_lm_head:
        modules.extend(["lm_head", "model.lm_head"])

    modules.extend(
        [
            "hybrid_gdn_checkpoint_cache.recurrent_slots",
            "hybrid_gdn_checkpoint_cache.conv_slots",
            "model.hybrid_gdn_checkpoint_cache.recurrent_slots",
            "model.hybrid_gdn_checkpoint_cache.conv_slots",
        ]
    )

    for layer_idx in range(num_layers):
        for prefix in ("layers", "model.layers"):
            layer_prefix = f"{prefix}.{layer_idx}"
            modules.extend(
                [
                    f"{layer_prefix}.input_layernorm",
                    f"{layer_prefix}.post_attention_layernorm",
                    f"{layer_prefix}.self_attn.q_norm",
                    f"{layer_prefix}.self_attn.k_norm",
                    f"{layer_prefix}.self_attn.q_layernorm",
                    f"{layer_prefix}.self_attn.k_layernorm",
                    f"{layer_prefix}.self_attn.rotary_emb",
                    f"{layer_prefix}.self_attn.mrope_emb",
                    f"{layer_prefix}.linear_attn.conv1d",
                    f"{layer_prefix}.linear_attn.conv1d_weight",
                    f"{layer_prefix}.linear_attn.A_log",
                    f"{layer_prefix}.linear_attn.A_log_weight",
                    f"{layer_prefix}.linear_attn.dt_bias",
                    f"{layer_prefix}.linear_attn.dt_bias_weight",
                    f"{layer_prefix}.linear_attn.norm",
                    f"{layer_prefix}.linear_attn.recurrent_state_buffer",
                    f"{layer_prefix}.linear_attn.conv_state_buffer",
                ]
            )
            if not quantize_linear_attn_gates:
                modules.extend(
                    [
                        f"{layer_prefix}.linear_attn.in_proj_a",
                        f"{layer_prefix}.linear_attn.in_proj_b",
                        f"{layer_prefix}.linear_attn.in_proj_ba",
                    ]
                )
            if "linear_attn" in fp8_exclude_groups:
                modules.append(f"{layer_prefix}.linear_attn")
            else:
                if "linear_attn_qkv" in fp8_exclude_groups:
                    modules.append(f"{layer_prefix}.linear_attn.in_proj_qkv")
                if "linear_attn_z" in fp8_exclude_groups:
                    modules.append(f"{layer_prefix}.linear_attn.in_proj_z")
                if "linear_attn_out_proj" in fp8_exclude_groups:
                    modules.append(f"{layer_prefix}.linear_attn.out_proj")
            if "mlp" in fp8_exclude_groups:
                modules.append(f"{layer_prefix}.mlp")
            if "self_attn" in fp8_exclude_groups:
                modules.append(f"{layer_prefix}.self_attn")
            else:
                if "self_attn_qkv" in fp8_exclude_groups:
                    for proj_name in ("q_proj", "k_proj", "v_proj"):
                        modules.append(f"{layer_prefix}.self_attn.{proj_name}")
                if "self_attn_o_proj" in fp8_exclude_groups:
                    modules.append(f"{layer_prefix}.self_attn.o_proj")
    return modules


def _quantized_checkpoint_ready(path: Path) -> bool:
    if path.is_file():
        return True
    if path.is_dir():
        return any(path.iterdir())
    return False


def _mlp_layer_idx(name: str) -> int | None:
    parts = name.split(".")
    if len(parts) < 4:
        return None
    for idx, part in enumerate(parts[:-3]):
        if part == "layers" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None


def _is_mlp_weight(
    name: str,
    *,
    num_layers: int,
    quantize_edge_mlp_layers: bool,
) -> bool:
    parts = name.split(".")
    if not (
        len(parts) >= 4
        and parts[-3] == "mlp"
        and parts[-2] in {"gate_proj", "up_proj", "down_proj"}
        and parts[-1] == "weight"
    ):
        return False
    if quantize_edge_mlp_layers:
        return True
    layer_idx = _mlp_layer_idx(name)
    if layer_idx is None:
        return True
    return layer_idx not in {0, num_layers - 1}


def _is_full_fp8_weight(
    name: str,
    *,
    quantize_lm_head: bool,
    quantize_linear_attn_gates: bool = False,
    fp8_exclude_groups: set[str] | None = None,
) -> bool:
    fp8_exclude_groups = fp8_exclude_groups or set()
    if not name.endswith(".weight"):
        return False
    parts = name.split(".")
    if len(parts) >= 2 and parts[-2] == "lm_head":
        return quantize_lm_head
    if len(parts) < 4:
        return False

    module_name = parts[-3]
    projection_name = parts[-2]
    if module_name == "mlp" and "mlp" in fp8_exclude_groups:
        return False
    if module_name == "self_attn":
        if "self_attn" in fp8_exclude_groups:
            return False
        if projection_name in {"q_proj", "k_proj", "v_proj"} and (
            "self_attn_qkv" in fp8_exclude_groups
        ):
            return False
        if projection_name == "o_proj" and "self_attn_o_proj" in fp8_exclude_groups:
            return False
    if module_name == "linear_attn":
        if "linear_attn" in fp8_exclude_groups:
            return False
        if projection_name in {"in_proj_a", "in_proj_b"}:
            return quantize_linear_attn_gates
        if projection_name == "in_proj_qkv" and (
            "linear_attn_qkv" in fp8_exclude_groups
        ):
            return False
        if projection_name == "in_proj_z" and "linear_attn_z" in fp8_exclude_groups:
            return False
        if (
            projection_name == "out_proj"
            and "linear_attn_out_proj" in fp8_exclude_groups
        ):
            return False
    supported_projection_names = {
        "mlp": {"gate_proj", "up_proj", "down_proj"},
        "self_attn": {"q_proj", "k_proj", "v_proj", "o_proj"},
        "linear_attn": {
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "out_proj",
        },
    }
    return projection_name in supported_projection_names.get(module_name, set())


def _scale_name(weight_name: str) -> str:
    return weight_name[: -len(".weight")] + ".weight_scale"


def _clear_quantized_checkpoint_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.name.endswith(".safetensors") or child.name.endswith(".json"):
            child.unlink()


def _save_manual_fp8_state_dict(
    model_path: Path,
    output_path: Path,
    *,
    weight_dtype: str,
    quantize_edge_mlp_layers: bool,
    quantize_lm_head: bool,
    quantize_linear_attn_gates: bool = False,
    fp8_exclude_groups: set[str] | None = None,
) -> None:
    """Create a sharded FP8 checkpoint directly from HF safetensors.

    Loading the HF architecture requires a newer Transformers than the Neuron
    venv uses internally. For these FP8 ablations, we do not need model
    execution: the checkpoint transform is a direct tensor rewrite.
    """
    from safetensors.torch import load_file, save_file  # noqa: WPS433
    from neuronx_distributed.quantization.quantization_utils import (  # noqa: WPS433
        quantize_fp8_per_channel,
    )

    num_layers = int(_load_text_config(model_path)["num_hidden_layers"])
    fp8_exclude_groups = fp8_exclude_groups or set()
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            source_index = json.load(f)
        source_weight_map = source_index["weight_map"]
        filenames = sorted(set(source_weight_map.values()))
    elif (model_path / "model.safetensors").exists():
        source_weight_map = None
        filenames = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No safetensors checkpoint found in {model_path}")

    _clear_quantized_checkpoint_dir(output_path)
    output_weight_map: dict[str, str] = {}
    total_size = 0
    quantized_count = 0

    for filename in filenames:
        shard = load_file(str(model_path / filename))
        output_shard = {}
        for name, tensor in shard.items():
            if weight_dtype == _WEIGHT_DTYPE_FP8_MLP_ONLY:
                should_quantize = _is_mlp_weight(
                    name,
                    num_layers=num_layers,
                    quantize_edge_mlp_layers=quantize_edge_mlp_layers,
                )
            elif weight_dtype == _WEIGHT_DTYPE_FP8_FULL:
                should_quantize = _is_full_fp8_weight(
                    name,
                    quantize_lm_head=quantize_lm_head,
                    quantize_linear_attn_gates=quantize_linear_attn_gates,
                    fp8_exclude_groups=fp8_exclude_groups,
                )
            else:
                raise ValueError(f"Unsupported FP8 weight dtype: {weight_dtype}")

            if should_quantize:
                weight, scale = quantize_fp8_per_channel(
                    tensor,
                    torch.float8_e4m3fn,
                    channel_axis=0,
                )
                output_shard[name] = weight
                output_shard[_scale_name(name)] = scale
                output_weight_map[_scale_name(name)] = filename
                total_size += weight.numel() * weight.element_size()
                total_size += scale.numel() * scale.element_size()
                quantized_count += 1
            else:
                output_shard[name] = tensor
                total_size += tensor.numel() * tensor.element_size()
            output_weight_map[name] = filename

        save_file(output_shard, str(output_path / filename), metadata={"format": "pt"})
        del shard
        del output_shard
        gc.collect()

    if source_weight_map is not None:
        with (output_path / "model.safetensors.index.json").open("w") as f:
            json.dump(
                {
                    "metadata": {"total_size": total_size},
                    "weight_map": output_weight_map,
                },
                f,
                indent=2,
                sort_keys=True,
            )

    print("MANUAL_FP8_WEIGHT_COUNT", quantized_count, flush=True)


def _build_config(args: argparse.Namespace):
    from neuronx_distributed_inference.models.config import (  # noqa: WPS433
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from src.modeling_qwen35 import Qwen35InferenceConfig  # noqa: WPS433

    model_path = Path(args.model_path).expanduser().resolve()
    config_dict = _load_text_config(model_path)
    num_layers = int(config_dict["num_hidden_layers"])
    fp8_exclude_groups = set(getattr(args, "fp8_exclude_groups", []) or [])
    if args.weight_dtype == _WEIGHT_DTYPE_FP8_FULL:
        modules_to_not_convert = _full_fp8_modules_to_not_convert(
            num_layers,
            quantize_lm_head=args.quantize_lm_head,
            quantize_linear_attn_gates=args.fp8_quantize_linear_attn_gates,
            fp8_exclude_groups=fp8_exclude_groups,
        )
    else:
        modules_to_not_convert = _mlp_only_modules_to_not_convert(num_layers)
    if (
        args.weight_dtype == _WEIGHT_DTYPE_FP8_MLP_ONLY
        and not args.quantize_edge_mlp_layers
    ):
        for layer_idx in (0, num_layers - 1):
            for prefix in ("layers", "model.layers"):
                modules_to_not_convert.append(f"{prefix}.{layer_idx}.mlp")
    cte_buckets = _cte_buckets(args)
    max_context_length = _max_context_length(args, cte_buckets)
    prefix_buckets = _prefix_buckets(args, cte_buckets)
    context_encoding_bucket_pairs = _context_encoding_bucket_pairs(
        args,
        cte_buckets,
        prefix_buckets,
    )
    token_generation_buckets = _token_generation_buckets(args)
    token_generation_batches = _token_generation_batches(args)
    _validate_prefix_buckets_fit_context(args, max_context_length, prefix_buckets)

    neuron_config_kwargs = {
        "tp_degree": args.tp_degree,
        "batch_size": args.max_num_seqs,
        "ctx_batch_size": args.ctx_batch_size,
        "tkg_batch_size": args.max_num_seqs,
        "seq_len": args.seq_len,
        "max_context_length": max_context_length,
        "max_length": args.seq_len,
        "context_encoding_buckets": cte_buckets,
        "token_generation_buckets": token_generation_buckets,
        "torch_dtype": torch.bfloat16,
        "enable_bucketing": len(cte_buckets) > 1
        or len(token_generation_buckets) > 1,
        "logical_nc_config": args.logical_nc_config,
        "save_sharded_checkpoint": True,
        "skip_warmup": args.skip_warmup,
    }
    if args.async_mode:
        neuron_config_kwargs["async_mode"] = True
    if token_generation_batches is not None:
        neuron_config_kwargs["token_generation_batches"] = token_generation_batches
    if (
        args.enable_fused_qkv
        or args.enable_qkv_nki_kernels
        or args.enable_attn_block_tkg_nki_kernel
    ):
        neuron_config_kwargs["fused_qkv"] = True
    if (
        args.enable_qkv_nki_kernels
        or args.enable_attn_block_tkg_nki_kernel
    ):
        neuron_config_kwargs["qkv_kernel_enabled"] = True
        neuron_config_kwargs["qkv_nki_kernel_enabled"] = True
    if args.enable_qkv_cte_nki_kernel_fuse_rope:
        rope_dim = config_dict.get("rope_dim")
        head_dim = config_dict.get("head_dim")
        if rope_dim is not None and head_dim is not None and int(rope_dim) != int(head_dim):
            raise ValueError(
                "--enable-qkv-cte-nki-kernel-fuse-rope is not valid for "
                f"partial-RoPE Qwen3.6 configs: rope_dim={rope_dim}, "
                f"head_dim={head_dim}. The stock fused-RoPE QKV kernel expects "
                "cos/sin to cover the full head dimension."
            )
        neuron_config_kwargs["qkv_cte_nki_kernel_fuse_rope"] = True
    if args.enable_split_qkv_tkg_nki_kernel:
        neuron_config_kwargs["qkv_tkg_nki_kernel_enabled"] = True
    if args.enable_attn_block_tkg_nki_kernel:
        neuron_config_kwargs["attn_block_tkg_nki_kernel_enabled"] = True
    if args.enable_attn_block_tkg_cascaded_attention:
        neuron_config_kwargs["attn_block_tkg_nki_kernel_cascaded_attention"] = True
    if args.enable_attn_block_tkg_cache_update:
        neuron_config_kwargs["attn_block_tkg_nki_kernel_cache_update"] = True
    if args.enable_out_proj_nki_kernel:
        neuron_config_kwargs["out_proj_kernel_enabled"] = True
    if args.enable_mlp_cte_nki_kernel or args.enable_mlp_tkg_nki_kernel:
        neuron_config_kwargs["mlp_kernel_enabled"] = True
    if args.enable_mlp_tkg_nki_kernel:
        neuron_config_kwargs["mlp_tkg_nki_kernel_enabled"] = True
    if args.enable_quantized_mlp_kernel:
        neuron_config_kwargs["quantized_mlp_kernel_enabled"] = True
    if args.enable_k_cache_transposed:
        neuron_config_kwargs["k_cache_transposed"] = True
    if args.weight_dtype in (_WEIGHT_DTYPE_FP8_MLP_ONLY, _WEIGHT_DTYPE_FP8_FULL):
        neuron_config_kwargs.update(
            {
                "quantized": True,
                "quantized_checkpoints_path": str(
                    Path(args.quantized_checkpoints_path).expanduser().resolve()
                ),
                "quantization_type": "per_channel_symmetric",
                "quantization_dtype": "f8e4m3",
                "modules_to_not_convert": modules_to_not_convert,
                "kv_cache_quant": False,
                "quantized_mlp_kernel_enabled": bool(
                    args.enable_quantized_mlp_kernel
                ),
                "activation_quantization_type": None,
            }
        )
    else:
        neuron_config_kwargs["quantized"] = False
    wlo_skip_patterns = _weights_to_skip_layout_optimization(args)
    if wlo_skip_patterns:
        neuron_config_kwargs["weights_to_skip_layout_optimization"] = wlo_skip_patterns
    if args.enable_kv_cache_quant:
        neuron_config_kwargs["kv_cache_quant"] = True
        neuron_config_kwargs["kv_quant_config"] = {"direct_cast": True}
    if args.disable_on_device_sampling:
        # vLLM/host-side sampling consumes logits from the Neuron trace. Without
        # logits, the serving path can only surface placeholder token ids.
        neuron_config_kwargs["output_logits"] = True
    else:
        neuron_config_kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig(
            do_sample=False,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
        )
        # Qwen's LM head is vocab-sharded when on-device sampling is enabled
        # (gather_output=False). The sampler must do distributed argmax/top-k
        # across vocab shards instead of sampling only from rank 0's shard.
        neuron_config_kwargs["vocab_parallel"] = True
        if args.output_logits_with_on_device_sampling:
            neuron_config_kwargs["output_logits"] = True
    if args.disable_argmax_kernel:
        neuron_config_kwargs["disable_argmax_kernel"] = True
    if args.disable_context_encoding_argmax_kernel:
        neuron_config_kwargs["disable_context_encoding_argmax_kernel"] = True
    if args.enable_prefix_caching or args.enable_hybrid_apc or args.enable_vllm_chunked_prefill:
        neuron_config_kwargs["is_block_kv_layout"] = True
        neuron_config_kwargs["pa_block_size"] = args.block_size
        neuron_config_kwargs["pa_num_blocks"] = _pa_num_blocks(args)
    if args.enable_prefix_caching or args.enable_hybrid_apc:
        neuron_config_kwargs["is_prefix_caching"] = True
        neuron_config_kwargs["prefix_buckets"] = prefix_buckets
        if context_encoding_bucket_pairs is not None:
            neuron_config_kwargs["context_encoding_bucket_pairs"] = (
                context_encoding_bucket_pairs
            )
        if args.prefix_cte_attention_chunk_size is not None:
            neuron_config_kwargs["prefix_cte_attention_chunk_size"] = (
                args.prefix_cte_attention_chunk_size
            )
        neuron_config_kwargs["prefix_cte_attention_backend"] = (
            args.prefix_cte_attention_backend
        )
        if args.prefix_cte_attention_segment_size is not None:
            neuron_config_kwargs["prefix_cte_attention_segment_size"] = (
                args.prefix_cte_attention_segment_size
            )
    if args.enable_vllm_chunked_prefill:
        # This flag selects Qwen's custom vLLM/Hybrid APC CTE prefix path.
        # Do not set NeuronConfig.chunked_prefill_config here: NxDI's generic
        # chunked-prefill feature is still rejected by NeuronBaseForCausalLM.
        neuron_config_kwargs["is_block_kv_layout"] = True

    neuron_config = NeuronConfig(**neuron_config_kwargs)

    if args.disable_static_hybrid_cache or args.enable_prefix_caching or args.enable_hybrid_apc:
        config_dict["use_hybrid_cache_manager"] = False
    else:
        config_dict.setdefault("use_hybrid_cache_manager", True)
    config_dict["use_hybrid_apc_manager"] = args.enable_hybrid_apc
    config_dict["gdn_checkpoint_interval"] = args.gdn_checkpoint_interval
    config_dict["max_gdn_checkpoint_slots"] = args.max_gdn_checkpoint_slots
    config_dict["gdn_recurrent_cache_dtype"] = args.gdn_recurrent_cache_dtype
    config_dict["gdn_conv_cache_dtype"] = args.gdn_conv_cache_dtype
    config_dict["hybrid_recurrent_cache_dtype"] = args.gdn_recurrent_cache_dtype
    config_dict["hybrid_conv_cache_dtype"] = args.gdn_conv_cache_dtype
    config_dict["hybrid_cache_mode"] = args.hybrid_cache_mode
    config_dict["hybrid_apc_require_vllm_metadata"] = args.hybrid_apc_require_vllm_metadata
    config_dict["hybrid_apc_allow_local_hash_fallback"] = (
        not args.hybrid_apc_require_vllm_metadata
    )
    config_dict["hybrid_apc_require_attention_block_refs"] = (
        args.hybrid_apc_require_vllm_metadata
    )
    config_dict["hybrid_apc_enable_backed_prefix_reads"] = getattr(
        args,
        "hybrid_apc_enable_backed_prefix_reads",
        False,
    )
    config_dict["hybrid_apc_commit_during_token_generation"] = (
        args.hybrid_apc_commit_during_token_generation
    )
    config_dict["use_qwen_hybrid_chunked_prefill"] = args.enable_vllm_chunked_prefill
    config_dict["use_qwen_hybrid_chunked_prefill_nki"] = args.enable_vllm_chunked_prefill
    config_dict["use_qwen_deltanet_decode_nki"] = getattr(
        args, "enable_deltanet_decode_nki", False
    )
    config_dict["use_text_only_cte_inputs"] = args.text_only_cte
    config_dict["use_compact_cte_attention_mask"] = args.compact_cte_attention_mask
    config_dict["use_cold_zero_conv_fast_path"] = args.cold_zero_conv_fast_path
    config_dict["use_qwen_qk_norm_rope_nki"] = args.enable_qwen_qk_norm_rope_nki_kernel
    config_dict["use_qwen_output_gate_nki"] = args.enable_qwen_output_gate_nki_kernel
    config_dict["use_qwen_qkv_gate_packed"] = args.enable_qwen_qkv_gate_packed_kernel
    config_dict["use_qwen_gated_o_proj_nki"] = args.enable_qwen_gated_o_proj_nki_kernel
    config_dict["disable_token_generation_wlo"] = _disable_token_generation_wlo(args)
    inf_config = Qwen35InferenceConfig(neuron_config=neuron_config, **config_dict)
    return inf_config, modules_to_not_convert


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-path", required=True)
    parser.add_argument("--quantized-checkpoints-path")
    parser.add_argument(
        "--base-compile-work-dir",
        default=None,
        help=(
            "NxDI compiler work directory. Defaults next to --compiled-path "
            "instead of /tmp so large compiles do not fill the root volume."
        ),
    )
    parser.add_argument(
        "--weight-dtype",
        choices=[
            _WEIGHT_DTYPE_FP8_MLP_ONLY,
            _WEIGHT_DTYPE_FP8_FULL,
            _WEIGHT_DTYPE_BF16_CONTROL,
        ],
        default=_WEIGHT_DTYPE_FP8_MLP_ONLY,
        help=(
            "Weight mode to compile. Use bf16_control for the non-FP8 "
            "host-logits real-token control."
        ),
    )
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=None,
        help=(
            "Maximum total context length in NeuronConfig. Defaults to the "
            "largest CTE bucket; set higher when chunked prefill serves long "
            "contexts with smaller active chunks."
        ),
    )
    parser.add_argument("--cte-bucket", type=int, default=512)
    parser.add_argument("--cte-buckets", nargs="+", default=None)
    parser.add_argument("--prefix-buckets", nargs="+", default=None)
    parser.add_argument(
        "--context-encoding-bucket-pairs",
        nargs="+",
        default=None,
        help=(
            "Optional sparse context-encoding 2D buckets as ACTIVE:PREFIX "
            "pairs. Prefix 0 pairs for every CTE bucket are added "
            "automatically unless --omit-zero-prefix-pair is set."
        ),
    )
    parser.add_argument(
        "--omit-zero-prefix-pair",
        action="store_true",
        help=(
            "Do not automatically add ACTIVE:0 dense context-encoding pairs. "
            "Use for long-prefix fallback artifacts that should not load the "
            "dense cold-prefill NEFF."
        ),
    )
    parser.add_argument("--token-generation-buckets", nargs="+", default=None)
    parser.add_argument("--token-generation-batches", nargs="+", default=None)
    parser.add_argument(
        "--disable-token-generation-wlo",
        action="store_true",
        help=(
            "Disable NxDI token-generation weight layout optimization. Use this "
            "when the generated layout_opt graph fails runtime validation."
        ),
    )
    parser.add_argument(
        "--weights-to-skip-layout-optimization",
        nargs="+",
        default=None,
        help=(
            "Regex patterns for checkpoint tensors that must not go through "
            "weight layout optimization. FP8 modes always add Qwen3.6-safe "
            "defaults for per-channel scale tensors and the tiny DeltaNet "
            "conv1d weight."
        ),
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--pa-num-blocks", type=int, default=None)
    parser.add_argument(
        "--pa-headroom-blocks",
        type=int,
        default=0,
        help=(
            "Extra usable PA blocks above the minimum seq_len/max_num_seqs "
            "capacity. Ignored when --pa-num-blocks is set. The final value is "
            "the NeuronConfig.pa_num_blocks value and should match vLLM "
            "--num-gpu-blocks-override."
        ),
    )
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--ctx-batch-size", type=int, default=1)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--async-mode", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-hybrid-apc", action="store_true")
    parser.add_argument("--enable-vllm-chunked-prefill", action="store_true")
    parser.add_argument(
        "--text-only-cte",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compact-cte-attention-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cold-zero-conv-fast-path",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Trace the DeltaNet conv path for cold context encoding that always "
            "starts at position 0. Do not use for APC or partial-prefix suffix CTE."
        ),
    )
    parser.add_argument(
        "--enable-deltanet-decode-nki",
        action="store_true",
        help=(
            "Trace token generation with the stateful one-token DeltaNet NKI "
            "decode step instead of the default Torch/XLA recurrent step."
        ),
    )
    parser.add_argument(
        "--deltanet-cte-backend",
        choices=[
            "env",
            "fused",
            "nki_chunked",
            "pytorch_chunk",
            "sequential",
            "nki_recurrent",
        ],
        default="env",
        help=(
            "DeltaNet CTE backend to force during tracing. The default preserves "
            "the caller's USE_NKI_* environment. Use nki_chunked or "
            "pytorch_chunk to compile controls for fused-CTE NaNs."
        ),
    )
    parser.add_argument("--disable-on-device-sampling", action="store_true")
    parser.add_argument(
        "--disable-argmax-kernel",
        action="store_true",
        help=(
            "Use the non-custom distributed argmax path for on-device greedy "
            "sampling. This is slower but avoids the NKI argmax output path "
            "when validating sampled-token correctness."
        ),
    )
    parser.add_argument(
        "--disable-context-encoding-argmax-kernel",
        action="store_true",
        help=(
            "Use the non-custom distributed argmax path only for context-encoding "
            "greedy sampling. Token generation keeps the configured argmax path, "
            "which limits decode-performance impact while isolating prefill "
            "sampled-token correctness."
        ),
    )
    parser.add_argument(
        "--output-logits-with-on-device-sampling",
        action="store_true",
        help=(
            "Debug mode: keep on-device greedy sampling enabled but also return "
            "logits from the trace so sampled token IDs can be compared with "
            "host argmax."
        ),
    )
    parser.add_argument("--kernel-q-tile-size", type=int, default=128)
    parser.add_argument("--kernel-kv-tile-size", type=int, default=1024)
    parser.add_argument(
        "--enable-fused-qkv",
        action="store_true",
        help=(
            "Fuse Q/K/V projection weights in the NxDI attention module. This "
            "is required by the QKV NKI kernels and the block TKG decode kernel."
        ),
    )
    parser.add_argument(
        "--enable-qkv-nki-kernels",
        action="store_true",
        help=(
            "Enable NxDI QKV kernels required by the block token-generation "
            "attention kernel."
        ),
    )
    parser.add_argument(
        "--enable-qkv-cte-nki-kernel-fuse-rope",
        action="store_true",
        help=(
            "Pass CTE RoPE cos/sin into the NxDI QKV NKI kernel so Q/K RoPE "
            "is fused into the projection kernel. For Qwen3.6 this must be "
            "validated with partial-RoPE coverage before compiling a perf artifact."
        ),
    )
    parser.add_argument(
        "--enable-qwen-qk-norm-rope-nki-kernel",
        action="store_true",
        help=(
            "Use the Qwen3.6-specific NKI kernel that fuses Q/K per-head "
            "RMSNorm with partial RoPE during multi-token context encoding."
        ),
    )
    parser.add_argument(
        "--enable-qwen-output-gate-nki-kernel",
        action="store_true",
        help=(
            "Use the Qwen3.6-specific output-gate projection path that routes "
            "the multi-token attention gate matmul through the NKI QKV CTE "
            "projection kernel."
        ),
    )
    parser.add_argument(
        "--enable-qwen-qkv-gate-packed-kernel",
        action="store_true",
        help=(
            "Use the Qwen3.6-specific packed QKV+gate projection path. This "
            "packs full-attention Wqkv as [Q | output_gate | K | V] and "
            "splits the gate from the QKV NKI output instead of running a "
            "separate output_gate_proj."
        ),
    )
    parser.add_argument(
        "--enable-qwen-gated-o-proj-nki-kernel",
        action="store_true",
        help=(
            "Use the Qwen3.6-specific ROW FP8 output-projection kernel that "
            "applies sigmoid(output_gate) to attention output inside the "
            "projection kernel for multi-token context encoding."
        ),
    )
    parser.add_argument(
        "--enable-split-qkv-tkg-nki-kernel",
        action="store_true",
        help=(
            "Enable Qwen's split Q/K/V token-generation NKI projection path. "
            "This is TKG-only and intentionally does not enable fused_qkv or "
            "the stock QKV CTE wrapper."
        ),
    )
    parser.add_argument(
        "--enable-attn-block-tkg-nki-kernel",
        action="store_true",
        help=(
            "Enable the NxDI token-generation attention NKI kernel for block "
            "KV layout. This targets decode speed when prefix caching is used."
        ),
    )
    parser.add_argument(
        "--enable-attn-block-tkg-cascaded-attention",
        action="store_true",
        help=(
            "Enable cascaded attention for the block token-generation NKI "
            "attention kernel."
        ),
    )
    parser.add_argument(
        "--enable-attn-block-tkg-cache-update",
        action="store_true",
        help=(
            "Update KV cache inside the block token-generation attention "
            "kernel instead of through the separate cache update path."
        ),
    )
    parser.add_argument(
        "--enable-out-proj-nki-kernel",
        action="store_true",
        help=(
            "Enable NxDI's NKI output-projection kernel for attention output. "
            "Block TKG enables this internally; this flag exposes it for "
            "non-block-TKG decode experiments."
        ),
    )
    parser.add_argument(
        "--enable-mlp-tkg-nki-kernel",
        action="store_true",
        help=(
            "Use NxDI/NKILib's MLP kernel for token generation. The Qwen3.6 "
            "custom decoder keeps this behind a flag because it changes the "
            "dense FFN lowering path."
        ),
    )
    parser.add_argument(
        "--enable-mlp-cte-nki-kernel",
        action="store_true",
        help=(
            "Use NxDI/NKILib's MLP kernel for context encoding. This targets "
            "cold-prefill dense SwiGLU cost and keeps Qwen CTE RMSNorm on the "
            "separate high-precision path before FP8 GEMM quantization."
        ),
    )
    parser.add_argument(
        "--enable-quantized-mlp-kernel",
        action="store_true",
        help=(
            "Enable the quantized FP8 MLP kernel path. Pair this with "
            "--enable-mlp-tkg-nki-kernel for FP8 full-weight decode "
            "experiments."
        ),
    )
    parser.add_argument(
        "--enable-k-cache-transposed",
        action="store_true",
        help=(
            "Store the K cache in the transposed layout used by the Neuron "
            "decode attention path. Best paired with block TKG cache update."
        ),
    )
    parser.add_argument(
        "--enable-kv-cache-quant",
        action="store_true",
        help=(
            "Use the NxDI FP8 direct-cast KV cache quantization path to reduce "
            "decode KV-cache HBM traffic."
        ),
    )
    parser.add_argument(
        "--prefix-cte-attention-chunk-size",
        type=int,
        default=None,
        help=(
            "When set, long prefix-cache CTE attention streams cached prefix KV "
            "in chunks of this size using online softmax instead of compiling "
            "one monolithic [active_tokens, prefix_tokens] attention score "
            "tensor. This is intended for 256K prefix buckets that exceed "
            "Neuron HBM scratchpad when compiled as a single prefix-attention "
            "shape."
        ),
    )
    parser.add_argument(
        "--prefix-cte-attention-backend",
        choices=["attention_cte", "segmented_cte"],
        default="attention_cte",
        help=(
            "Prefix-cache CTE attention backend. attention_cte is the existing "
            "flat-prior kernel. segmented_cte uses the Neuron 2.30 block-KV "
            "segmented CTE kernel to stream long cached prefixes by segment."
        ),
    )
    parser.add_argument(
        "--prefix-cte-attention-segment-size",
        type=int,
        default=None,
        help=(
            "Prior segment size for --prefix-cte-attention-backend segmented_cte. "
            "Must be positive and divisible by --block-size."
        ),
    )
    parser.add_argument("--disable-static-hybrid-cache", action="store_true")
    parser.add_argument("--gdn-checkpoint-interval", type=int, default=256)
    parser.add_argument("--max-gdn-checkpoint-slots", type=int, default=8)
    parser.add_argument("--gdn-recurrent-cache-dtype", default="float32")
    parser.add_argument("--gdn-conv-cache-dtype", default="bfloat16")
    parser.add_argument("--hybrid-cache-mode", default="all")
    parser.add_argument("--hybrid-apc-require-vllm-metadata", action="store_true")
    parser.add_argument(
        "--hybrid-apc-enable-backed-prefix-reads",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--hybrid-apc-commit-during-token-generation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep the legacy Hybrid APC checkpoint-bank commit outputs on "
            "token generation traces. The default commits checkpoint banks only "
            "during context encoding."
        ),
    )
    parser.add_argument(
        "--quantize-edge-mlp-layers",
        action="store_true",
        help=(
            "Quantize layer-0 and final-layer MLP weights too. By default they "
            "stay BF16, matching the AWS Trn2 FP8 tutorial's conservative "
            "edge-layer policy."
        ),
    )
    parser.add_argument(
        "--quantize-lm-head",
        action="store_true",
        help=(
            "Also quantize lm_head in fp8_full mode. Default keeps lm_head BF16, "
            "matching common NVIDIA/vLLM FP8 policy."
        ),
    )
    parser.add_argument(
        "--fp8-quantize-linear-attn-gates",
        action="store_true",
        help=(
            "Use the older coherent FP8 policy for Qwen3.6 linear-attention "
            "gate projections: leave in_proj_a/in_proj_b out of "
            "modules_to_not_convert and manually quantize their weights to "
            "FP8. This is an isolation flag because public Qwen FP8 configs "
            "usually keep gate/control projections higher precision."
        ),
    )
    parser.add_argument(
        "--fp8-exclude-groups",
        nargs="*",
        choices=sorted(_FP8_EXCLUDE_GROUPS),
        default=[],
        help=(
            "Extra fp8_full module groups to leave BF16 for targeted coherence "
            "isolation. Useful values are linear_attn, mlp, self_attn, and the "
            "finer-grained linear_attn_qkv/linear_attn_z/linear_attn_out_proj/"
            "self_attn_qkv/self_attn_o_proj groups."
        ),
    )
    parser.add_argument("--force-quantize", action="store_true")
    parser.add_argument("--quantize-only", action="store_true")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help=(
            "Run post-compile artifact fixes on an existing compiled-path "
            "without regenerating FP8 checkpoints or invoking model.compile(). "
            "Useful after an interrupted checkpoint-bank insertion."
        ),
    )
    parser.add_argument("--load-after-compile", action="store_true")
    args = parser.parse_args()
    if (
        args.weight_dtype in (_WEIGHT_DTYPE_FP8_MLP_ONLY, _WEIGHT_DTYPE_FP8_FULL)
        and not args.quantized_checkpoints_path
    ):
        parser.error("--quantized-checkpoints-path is required for FP8 weight modes")
    if args.quantize_lm_head and args.weight_dtype != _WEIGHT_DTYPE_FP8_FULL:
        parser.error("--quantize-lm-head is only valid with --weight-dtype fp8_full")
    if args.enable_split_qkv_tkg_nki_kernel and (
        args.enable_fused_qkv
        or args.enable_qkv_nki_kernels
        or args.enable_attn_block_tkg_nki_kernel
    ):
        parser.error(
            "--enable-split-qkv-tkg-nki-kernel cannot be combined with "
            "--enable-fused-qkv, --enable-qkv-nki-kernels, "
            "or --enable-attn-block-tkg-nki-kernel"
        )
    if (
        args.context_encoding_bucket_pairs is not None
        and not (args.enable_prefix_caching or args.enable_hybrid_apc)
    ):
        parser.error(
            "--context-encoding-bucket-pairs requires prefix caching or Hybrid APC"
        )
    if args.max_num_seqs <= 0:
        parser.error("--max-num-seqs must be positive")
    if args.ctx_batch_size <= 0:
        parser.error("--ctx-batch-size must be positive")
    if args.pa_headroom_blocks < 0:
        parser.error("--pa-headroom-blocks must be non-negative")
    if args.pa_num_blocks is not None and args.pa_headroom_blocks:
        parser.error("--pa-headroom-blocks cannot be combined with --pa-num-blocks")
    if (
        args.prefix_cte_attention_chunk_size is not None
        and args.prefix_cte_attention_chunk_size <= 0
    ):
        parser.error("--prefix-cte-attention-chunk-size must be positive")
    if (
        args.prefix_cte_attention_segment_size is not None
        and args.prefix_cte_attention_segment_size <= 0
    ):
        parser.error("--prefix-cte-attention-segment-size must be positive")
    if (
        args.prefix_cte_attention_segment_size is not None
        and args.prefix_cte_attention_segment_size % args.block_size != 0
    ):
        parser.error(
            "--prefix-cte-attention-segment-size must be divisible by --block-size"
        )
    if args.enable_hybrid_apc and args.gdn_checkpoint_interval != args.block_size:
        parser.error(
            "--enable-hybrid-apc v0 requires --gdn-checkpoint-interval to "
            "equal --block-size"
        )

    repo = _repo_root(args.repo_root)
    contrib_model_dir = repo / "contrib" / "models" / "Qwen3.6-27B"
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(contrib_model_dir))
    if args.weight_dtype in (_WEIGHT_DTYPE_FP8_MLP_ONLY, _WEIGHT_DTYPE_FP8_FULL):
        _ensure_fp8_environment()
    _configure_deltanet_cte_backend(args.deltanet_cte_backend)

    from src.modeling_qwen35 import NeuronQwen35ForCausalLM  # noqa: WPS433

    model_path = Path(args.model_path).expanduser().resolve()
    compiled_path = Path(args.compiled_path).expanduser().resolve()
    quantized_path = (
        Path(args.quantized_checkpoints_path).expanduser().resolve()
        if args.quantized_checkpoints_path
        else None
    )
    base_compile_work_dir = _configure_base_compile_work_dir(
        compiled_path,
        args.base_compile_work_dir,
    )

    inf_config, modules_to_not_convert = _build_config(args)

    print("WEIGHT_DTYPE_MODE", args.weight_dtype, flush=True)
    if args.weight_dtype == _WEIGHT_DTYPE_FP8_MLP_ONLY:
        print("FP8_MODE mlp_only", flush=True)
    elif args.weight_dtype == _WEIGHT_DTYPE_FP8_FULL:
        print("FP8_MODE full", flush=True)
    else:
        print("FP8_MODE disabled_bf16_control", flush=True)
    print("QUANTIZE_LM_HEAD", bool(args.quantize_lm_head), flush=True)
    print(
        "FP8_QUANTIZE_LINEAR_ATTN_GATES",
        bool(args.fp8_quantize_linear_attn_gates),
        flush=True,
    )
    print(
        "FP8_EXCLUDE_GROUPS",
        ",".join(sorted(set(args.fp8_exclude_groups))) or "none",
        flush=True,
    )
    print("MODEL_PATH", str(model_path), flush=True)
    print("COMPILED_PATH", str(compiled_path), flush=True)
    print("BASE_COMPILE_WORK_DIR", str(base_compile_work_dir), flush=True)
    print("DELTANET_CTE_BACKEND", args.deltanet_cte_backend, flush=True)
    for env_name in sorted(_DELTANET_CTE_BACKEND_ENV):
        print(env_name, os.environ.get(env_name), flush=True)
    if quantized_path is not None:
        print("QUANTIZED_CHECKPOINTS_PATH", str(quantized_path), flush=True)
    for env_name in _FP8_ENV_DEFAULTS:
        print(env_name, os.environ.get(env_name), flush=True)
    print(
        "WEIGHTS_TO_SKIP_LAYOUT_OPTIMIZATION",
        json.dumps(inf_config.neuron_config.weights_to_skip_layout_optimization),
        flush=True,
    )
    print(
        "DISABLE_TOKEN_GENERATION_WLO",
        bool(inf_config.disable_token_generation_wlo),
        flush=True,
    )
    print("MODULES_TO_NOT_CONVERT_COUNT", len(modules_to_not_convert), flush=True)
    print(
        "CONTEXT_TRACE_SHAPE",
        json.dumps(
            {
                "seq_len": args.seq_len,
                "max_context_length": _max_context_length(args, _cte_buckets(args)),
                "context_encoding_buckets": _cte_buckets(args),
                "prefix_buckets": _prefix_buckets(args, _cte_buckets(args)),
                "prefix_cte_attention_backend": args.prefix_cte_attention_backend,
                "prefix_cte_attention_segment_size": (
                    args.prefix_cte_attention_segment_size
                ),
                "prefix_cte_attention_chunk_size": args.prefix_cte_attention_chunk_size,
                "context_encoding_bucket_pairs": _context_encoding_bucket_pairs(
                    args,
                    _cte_buckets(args),
                    _prefix_buckets(args, _cte_buckets(args)),
                ),
                "token_generation_buckets": _token_generation_buckets(args),
                "token_generation_batches": _token_generation_batches(args),
                "max_num_seqs": args.max_num_seqs,
                "ctx_batch_size": args.ctx_batch_size,
                "tkg_batch_size": args.max_num_seqs,
                "async_mode": args.async_mode,
                "skip_warmup": args.skip_warmup,
                "enable_prefix_caching": args.enable_prefix_caching,
                "enable_hybrid_apc": args.enable_hybrid_apc,
                "enable_vllm_chunked_prefill": args.enable_vllm_chunked_prefill,
                "enable_deltanet_decode_nki": args.enable_deltanet_decode_nki,
                "enable_fused_qkv": args.enable_fused_qkv,
                "enable_qkv_nki_kernels": args.enable_qkv_nki_kernels,
                "enable_qwen_qk_norm_rope_nki_kernel": (
                    args.enable_qwen_qk_norm_rope_nki_kernel
                ),
                "enable_qwen_qkv_gate_packed_kernel": (
                    args.enable_qwen_qkv_gate_packed_kernel
                ),
                "enable_qwen_gated_o_proj_nki_kernel": (
                    args.enable_qwen_gated_o_proj_nki_kernel
                ),
                "enable_split_qkv_tkg_nki_kernel": (
                    args.enable_split_qkv_tkg_nki_kernel
                ),
                "enable_attn_block_tkg_nki_kernel": (
                    args.enable_attn_block_tkg_nki_kernel
                ),
                "enable_attn_block_tkg_cascaded_attention": (
                    args.enable_attn_block_tkg_cascaded_attention
                ),
                "enable_attn_block_tkg_cache_update": (
                    args.enable_attn_block_tkg_cache_update
                ),
                "enable_out_proj_nki_kernel": args.enable_out_proj_nki_kernel,
                "enable_mlp_cte_nki_kernel": args.enable_mlp_cte_nki_kernel,
                "enable_mlp_tkg_nki_kernel": args.enable_mlp_tkg_nki_kernel,
                "enable_quantized_mlp_kernel": args.enable_quantized_mlp_kernel,
                "enable_k_cache_transposed": args.enable_k_cache_transposed,
                "enable_kv_cache_quant": args.enable_kv_cache_quant,
                "fp8_quantize_linear_attn_gates": bool(
                    args.fp8_quantize_linear_attn_gates
                ),
                "block_size": args.block_size,
                "pa_min_blocks": _pa_min_blocks(args),
                "pa_requested_blocks": _pa_requested_blocks(args),
                "pa_usable_headroom_blocks": (
                    _pa_requested_blocks(args) - _pa_min_blocks(args)
                ),
                "pa_headroom_blocks": (
                    _pa_requested_blocks(args) - _pa_min_blocks(args)
                ),
                "pa_num_blocks": _pa_num_blocks(args),
                "gdn_checkpoint_interval": args.gdn_checkpoint_interval,
                "max_gdn_checkpoint_slots": args.max_gdn_checkpoint_slots,
                "hybrid_apc_commit_during_token_generation": (
                    args.hybrid_apc_commit_during_token_generation
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if args.postprocess_only:
        if not compiled_path.exists():
            raise FileNotFoundError(f"--postprocess-only missing artifact: {compiled_path}")
        print("POSTPROCESS_ONLY_START", flush=True)
        _ensure_hybrid_checkpoint_weights(compiled_path, inf_config)
        _sanitize_reloadable_neuron_config(compiled_path)
        print("COMPILE_DONE", flush=True)
        return 0

    if args.weight_dtype == _WEIGHT_DTYPE_BF16_CONTROL:
        print("QUANTIZE_SKIP bf16_control", flush=True)
    elif args.force_quantize or not _quantized_checkpoint_ready(quantized_path):
        print("QUANTIZE_START manual_fp8", flush=True)
        _save_manual_fp8_state_dict(
            model_path,
            quantized_path,
            weight_dtype=args.weight_dtype,
            quantize_edge_mlp_layers=args.quantize_edge_mlp_layers,
            quantize_lm_head=args.quantize_lm_head,
            quantize_linear_attn_gates=args.fp8_quantize_linear_attn_gates,
            fp8_exclude_groups=set(args.fp8_exclude_groups),
        )
        print("QUANTIZE_DONE", flush=True)
    else:
        print("QUANTIZE_SKIP existing checkpoint found", flush=True)

    if args.quantize_only:
        return 0

    print("COMPILE_START", flush=True)
    model = NeuronQwen35ForCausalLM(str(model_path), inf_config)
    model.compile(str(compiled_path))
    _ensure_hybrid_checkpoint_weights(compiled_path, inf_config)
    _sanitize_reloadable_neuron_config(compiled_path)
    del model
    gc.collect()
    print("COMPILE_DONE", flush=True)

    if args.load_after_compile:
        model = NeuronQwen35ForCausalLM(str(compiled_path))
        model.load(str(compiled_path))
        print("LOAD_AFTER_COMPILE_OK", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
