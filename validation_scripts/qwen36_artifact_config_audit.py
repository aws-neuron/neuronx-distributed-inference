#!/usr/bin/env python3
"""Audit Qwen3.6 Neuron artifact config for APC/prefill A/B experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_config(path: Path) -> dict[str, Any]:
    config_path = path
    if path.is_dir():
        config_path = path / "neuron_config.json"
    with config_path.open() as handle:
        return json.load(handle)


def _first_config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    override = config.get("override_neuron_config")
    if isinstance(override, dict):
        for key in keys:
            if key in override:
                return override[key]
    nested = config.get("neuron_config")
    if isinstance(nested, dict):
        for key in keys:
            if key in nested:
                return nested[key]
        nested_override = nested.get("override_neuron_config")
        if isinstance(nested_override, dict):
            for key in keys:
                if key in nested_override:
                    return nested_override[key]
    return default


def _bool_config(config: dict[str, Any], *keys: str) -> bool:
    return bool(_first_config_value(config, *keys, default=False))


def _compile_backend_from_log(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    backend = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("DELTANET_CTE_BACKEND "):
            backend = line.split(maxsplit=1)[1].strip()
        elif " --deltanet-cte-backend " in line:
            backend = line.rsplit(" --deltanet-cte-backend ", 1)[1].split()[0]
    return backend


def _warning(
    warnings: list[dict[str, Any]],
    *,
    code: str,
    message: str,
    value: Any = None,
):
    warnings.append({"code": code, "message": message, "value": value})


def audit(
    *,
    artifact: Path,
    compile_log: Path | None,
    recommended_block_size: int,
    min_usable_headroom_blocks: int,
    strict_hybrid_gate: bool,
) -> dict[str, Any]:
    config = _load_config(artifact)
    seq_len = int(_first_config_value(config, "seq_len", "max_length", default=0) or 0)
    max_num_seqs = int(_first_config_value(config, "batch_size", default=1) or 1)
    ctx_batch_size = int(_first_config_value(config, "ctx_batch_size", default=1) or 1)
    block_size = int(_first_config_value(config, "pa_block_size", default=0) or 0)
    pa_num_blocks = int(_first_config_value(config, "pa_num_blocks", default=0) or 0)
    max_gdn_slots = int(
        _first_config_value(config, "max_gdn_checkpoint_slots", default=0) or 0
    )
    cte_buckets = _first_config_value(config, "context_encoding_buckets", default=[])
    token_generation_buckets = _first_config_value(
        config,
        "token_generation_buckets",
        default=[],
    )
    prefix_buckets = _first_config_value(config, "prefix_buckets", default=[])
    tkg_batch_size = int(_first_config_value(config, "tkg_batch_size", default=1) or 1)
    async_mode = _bool_config(config, "async_mode")
    output_logits = _bool_config(config, "output_logits")
    on_device_sampling_config = _first_config_value(
        config,
        "on_device_sampling_config",
        default=None,
    )
    min_blocks = (
        max(1, math.ceil(seq_len / block_size) * max_num_seqs)
        if seq_len > 0 and block_size > 0
        else 0
    )
    usable_headroom_blocks = pa_num_blocks - min_blocks if pa_num_blocks else None
    usable_headroom_blocks = (
        max(0, usable_headroom_blocks)
        if usable_headroom_blocks is not None
        else None
    )
    required_full_prompt_boundaries = (
        math.ceil(seq_len / block_size) if seq_len > 0 and block_size > 0 else 0
    )
    compile_backend = _compile_backend_from_log(compile_log)
    if compile_backend is None and "nki_chunked" in str(artifact):
        compile_backend = "nki_chunked_from_artifact_name"

    warnings: list[dict[str, Any]] = []
    if block_size and recommended_block_size and block_size != recommended_block_size:
        _warning(
            warnings,
            code="non_recommended_block_size",
            message=(
                "Artifact PA block size differs from the configured Neuron "
                "performance recommendation."
            ),
            value={"pa_block_size": block_size, "recommended": recommended_block_size},
        )
    if (
        usable_headroom_blocks is not None
        and usable_headroom_blocks < min_usable_headroom_blocks
    ):
        _warning(
            warnings,
            code="low_pa_headroom",
            message=(
                "PA block capacity has little usable residency headroom after "
                "minimum sequence capacity."
            ),
            value={
                "pa_num_blocks": pa_num_blocks,
                "min_blocks": min_blocks,
                "usable_headroom_blocks": usable_headroom_blocks,
                "minimum_expected": min_usable_headroom_blocks,
            },
        )
    if strict_hybrid_gate and max_gdn_slots and required_full_prompt_boundaries > max_gdn_slots:
        _warning(
            warnings,
            code="strict_gate_boundary_slots_exceed_gdn_slots",
            message=(
                "With the current disable-unbacked-prefix-reads gate, a full "
                "prompt can require more backed prefix boundaries than the GDN "
                "checkpoint slot budget can hold unless boundary chunk commits "
                "or a less conservative gate are used."
            ),
            value={
                "required_full_prompt_boundaries": required_full_prompt_boundaries,
                "max_gdn_checkpoint_slots": max_gdn_slots,
            },
        )
    if compile_backend and "nki_chunked" in compile_backend:
        _warning(
            warnings,
            code="nki_chunked_deltanet_cte",
            message=(
                "Compile log or artifact name indicates the nki_chunked DeltaNet "
                "CTE backend; compare against a fused-control artifact."
            ),
            value=compile_backend,
        )
    if (
        seq_len >= 32768
        and isinstance(token_generation_buckets, list)
        and token_generation_buckets == [seq_len]
    ):
        _warning(
            warnings,
            code="single_full_length_tkg_bucket",
            message=(
                "Decode has only a full-length token-generation bucket. Short "
                "generations will still use the largest TKG trace shape; compare "
                "against an artifact compiled with smaller TKG buckets such as "
                "8192,32768,seq_len."
            ),
            value=token_generation_buckets,
        )
    if not async_mode:
        _warning(
            warnings,
            code="sync_neuron_runtime_decode",
            message=(
                "Neuron async_mode is disabled. The previous fast decode control "
                "path used async runtime execution for token generation."
            ),
            value=False,
        )
    if tkg_batch_size <= 1:
        _warning(
            warnings,
            code="single_sequence_tkg_batch",
            message=(
                "tkg_batch_size is 1, so decode cannot amortize per-token runner "
                "overhead across concurrent sequences."
            ),
            value=tkg_batch_size,
        )

    summary = {
        "artifact": str(artifact),
        "compile_log": str(compile_log) if compile_log is not None else None,
        "seq_len": seq_len,
        "max_num_seqs": max_num_seqs,
        "ctx_batch_size": ctx_batch_size,
        "pa_block_size": block_size,
        "pa_num_blocks": pa_num_blocks,
        "pa_min_blocks": min_blocks,
        "pa_usable_headroom_blocks": usable_headroom_blocks,
        "max_gdn_checkpoint_slots": max_gdn_slots,
        "required_full_prompt_boundaries": required_full_prompt_boundaries,
        "context_encoding_buckets": cte_buckets,
        "token_generation_buckets": token_generation_buckets,
        "tkg_batch_size": tkg_batch_size,
        "async_mode": async_mode,
        "output_logits": output_logits,
        "on_device_sampling": on_device_sampling_config is not None,
        "prefix_buckets": prefix_buckets,
        "is_prefix_caching": _bool_config(config, "is_prefix_caching"),
        "use_hybrid_apc_manager": _bool_config(config, "use_hybrid_apc_manager"),
        "use_qwen_hybrid_chunked_prefill": _bool_config(
            config,
            "use_qwen_hybrid_chunked_prefill",
        ),
        "deltanet_cte_backend": compile_backend,
        "warnings": warnings,
    }
    summary["warning_count"] = len(warnings)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", help="Artifact directory or neuron_config.json path")
    parser.add_argument("--compile-log", type=Path, default=None)
    parser.add_argument("--recommended-block-size", type=int, default=32)
    parser.add_argument("--min-usable-headroom-blocks", type=int, default=8)
    parser.add_argument(
        "--no-strict-hybrid-gate",
        action="store_true",
        help="Do not warn when full-prompt boundary count exceeds GDN slot count.",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    summary = audit(
        artifact=Path(args.artifact).expanduser().resolve(),
        compile_log=args.compile_log.expanduser().resolve()
        if args.compile_log is not None
        else None,
        recommended_block_size=args.recommended_block_size,
        min_usable_headroom_blocks=args.min_usable_headroom_blocks,
        strict_hybrid_gate=not args.no_strict_hybrid_gate,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if args.strict and summary["warnings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
