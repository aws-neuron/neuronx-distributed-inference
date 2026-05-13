#!/usr/bin/env python3
"""Compile Qwen3.6-27B 64K with a scoped FP8 weight-quantization ablation.

This script intentionally starts from the validated 64K hybrid/chunked-prefill
baseline and changes only weight quantization. The first supported mode is
``mlp_only``: MLP linear weights are converted to FP8 while attention, DeltaNet,
normalization, embeddings, lm_head, KV cache, and recurrent state remain BF16.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch


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


def _quantized_checkpoint_ready(path: Path) -> bool:
    if path.is_file():
        return True
    if path.is_dir():
        return any(path.iterdir())
    return False


def _is_mlp_weight(name: str) -> bool:
    parts = name.split(".")
    return (
        len(parts) >= 4
        and parts[-3] == "mlp"
        and parts[-2] in {"gate_proj", "up_proj", "down_proj"}
        and parts[-1] == "weight"
    )


def _scale_name(weight_name: str) -> str:
    return weight_name[: -len(".weight")] + ".weight_scale"


def _clear_quantized_checkpoint_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.name.endswith(".safetensors") or child.name.endswith(".json"):
            child.unlink()


def _save_mlp_only_fp8_state_dict(model_path: Path, output_path: Path) -> None:
    """Create a sharded FP8 checkpoint directly from HF safetensors.

    Loading the HF architecture requires a newer Transformers than the Neuron
    venv uses internally. For this MLP-only ablation, we do not need model
    execution: the checkpoint transform is a direct tensor rewrite.
    """
    from safetensors.torch import load_file, save_file  # noqa: WPS433
    from neuronx_distributed.quantization.quantization_utils import (  # noqa: WPS433
        quantize_fp8_per_channel,
    )

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
            if _is_mlp_weight(name):
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

    print("MANUAL_FP8_MLP_WEIGHT_COUNT", quantized_count, flush=True)


def _build_config(args: argparse.Namespace):
    from neuronx_distributed_inference.models.config import (  # noqa: WPS433
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from src.modeling_qwen35 import Qwen35InferenceConfig  # noqa: WPS433

    model_path = Path(args.model_path).expanduser().resolve()
    config_dict = _load_text_config(model_path)
    num_layers = int(config_dict["num_hidden_layers"])
    modules_to_not_convert = _mlp_only_modules_to_not_convert(num_layers)

    neuron_config = NeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=args.seq_len,
        max_context_length=args.cte_bucket,
        max_length=args.seq_len,
        context_encoding_buckets=[args.cte_bucket],
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(
            do_sample=False,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
        ),
        enable_bucketing=False,
        logical_nc_config=args.logical_nc_config,
        save_sharded_checkpoint=True,
        quantized=True,
        quantized_checkpoints_path=str(
            Path(args.quantized_checkpoints_path).expanduser().resolve()
        ),
        quantization_type="per_channel_symmetric",
        quantization_dtype="f8e4m3",
        modules_to_not_convert=modules_to_not_convert,
        kv_cache_quant=False,
        quantized_mlp_kernel_enabled=False,
        activation_quantization_type=None,
    )

    config_dict.setdefault("use_hybrid_cache_manager", True)
    config_dict.setdefault("use_qwen_hybrid_chunked_prefill", True)
    config_dict.setdefault("use_qwen_hybrid_chunked_prefill_nki", True)

    inf_config = Qwen35InferenceConfig(neuron_config=neuron_config, **config_dict)
    return inf_config, modules_to_not_convert


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-path", required=True)
    parser.add_argument("--quantized-checkpoints-path", required=True)
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--cte-bucket", type=int, default=512)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--force-quantize", action="store_true")
    parser.add_argument("--quantize-only", action="store_true")
    parser.add_argument("--load-after-compile", action="store_true")
    args = parser.parse_args()

    repo = _repo_root(args.repo_root)
    contrib_model_dir = repo / "contrib" / "models" / "Qwen3.6-27B"
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(contrib_model_dir))

    from src.modeling_qwen35 import NeuronQwen35ForCausalLM  # noqa: WPS433

    model_path = Path(args.model_path).expanduser().resolve()
    compiled_path = Path(args.compiled_path).expanduser().resolve()
    quantized_path = Path(args.quantized_checkpoints_path).expanduser().resolve()

    inf_config, modules_to_not_convert = _build_config(args)

    print("FP8_MODE mlp_only", flush=True)
    print("MODEL_PATH", str(model_path), flush=True)
    print("COMPILED_PATH", str(compiled_path), flush=True)
    print("QUANTIZED_CHECKPOINTS_PATH", str(quantized_path), flush=True)
    print("MODULES_TO_NOT_CONVERT_COUNT", len(modules_to_not_convert), flush=True)
    print(
        "CONTEXT_TRACE_SHAPE",
        json.dumps(
            {
                "seq_len": args.seq_len,
                "max_context_length": args.cte_bucket,
                "context_encoding_buckets": [args.cte_bucket],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if args.force_quantize or not _quantized_checkpoint_ready(quantized_path):
        print("QUANTIZE_START manual_mlp_only", flush=True)
        _save_mlp_only_fp8_state_dict(model_path, quantized_path)
        print("QUANTIZE_DONE", flush=True)
    else:
        print("QUANTIZE_SKIP existing checkpoint found", flush=True)

    if args.quantize_only:
        return 0

    print("COMPILE_START", flush=True)
    model = NeuronQwen35ForCausalLM(str(model_path), inf_config)
    model.compile(str(compiled_path))
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
