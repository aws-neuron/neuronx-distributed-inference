"""vLLM scheduler patch for Qwen Hybrid APC fallback.

This module is intentionally opt-in. The safe fallback for current Hybrid APC
validation is to make vLLM skip attention-prefix reads before slot allocation
when the GDN checkpoint side is not integrated with the scheduler yet.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import json
import logging
import os
import struct
import sys
from typing import Any, Hashable, NamedTuple

import torch


logger = logging.getLogger(__name__)
_SCHEDULER_MODULE = "vllm.v1.core.sched.scheduler"
_KV_CACHE_MANAGER_MODULE = "vllm.v1.core.kv_cache_manager"
_VLLM_NEURON_RUNNER_MODULE = "vllm_neuron.worker.neuronx_distributed_model_runner"
_VLLM_NEURON_LOADER_MODULE = "vllm_neuron.worker.neuronx_distributed_model_loader"
_PATCHED_MODULES = {
    _SCHEDULER_MODULE,
    _KV_CACHE_MANAGER_MODULE,
    _VLLM_NEURON_RUNNER_MODULE,
    _VLLM_NEURON_LOADER_MODULE,
}


class HybridGDNPrefixKey(NamedTuple):
    cumulative_prefix_hash: Hashable
    prefix_len: int
    block_size: int
    cache_salt: Hashable | None
    model_revision: str
    layout_version: int
    tp_rank: int
    recurrent_dtype: str
    conv_dtype: str


_GDN_PREFIX_KEYS: set[HybridGDNPrefixKey] = set()
_AUTHORIZED_PREFIX_READS: dict[int, list[HybridGDNPrefixKey]] = {}
_AUTHORIZED_PREFIX_READS_BY_REQUEST: dict[Hashable, list[HybridGDNPrefixKey]] = {}
_SCHEDULER_OUTPUT_METADATA_ATTR = "_qwen36_hybrid_apc_metadata_by_request_id"
_SCHEDULER_OUTPUT_REQUEST_RECORDS_ATTR = "_qwen36_hybrid_apc_request_records"
_MAX_PREFIX_CACHE_HIT_LEN_ATTR = "_qwen36_hybrid_apc_max_prefix_cache_hit_len"
_MAX_PREFIX_CACHE_BLOCKS_ATTR = "_qwen36_hybrid_apc_max_prefix_cache_blocks"
_RUNNER_PREFILL_STATE_FOR_OUTPUT_ATTR = (
    "_qwen36_hybrid_apc_prefill_completion_state_for_output"
)
_HYBRID_APC_RUNTIME_CONFIG_KEYS = (
    "use_hybrid_apc_manager",
    "use_qwen_hybrid_chunked_prefill",
    "use_qwen_hybrid_chunked_prefill_nki",
    "gdn_checkpoint_interval",
    "max_gdn_checkpoint_slots",
    "gdn_recurrent_cache_dtype",
    "gdn_conv_cache_dtype",
    "hybrid_recurrent_cache_dtype",
    "hybrid_conv_cache_dtype",
    "hybrid_cache_mode",
    "hybrid_cache_prefix_boundary_only",
    "hybrid_cache_block_boundary_only",
    "hybrid_cache_validate_exact",
    "hybrid_apc_layout_version",
    "hybrid_apc_allow_residual_replay",
    "hybrid_apc_cache_salt",
    "hybrid_apc_model_revision",
    "hybrid_apc_require_vllm_metadata",
    "hybrid_apc_allow_local_hash_fallback",
    "hybrid_apc_require_attention_block_refs",
    "hybrid_apc_reject_unbacked_attention_hits",
    "hybrid_apc_disable_unbacked_prefix_reads",
    "hybrid_apc_enable_backed_prefix_reads",
    "hybrid_apc_max_backed_prefix_read_len",
    "hybrid_apc_allow_mixed_prefill_decode",
    "hybrid_apc_prefill_chunk_tokens",
    "qwen_prefill_group_size",
)
_HYBRID_APC_BRIDGE_CONFIG_ATTRS = {
    "hybrid_apc_allow_local_hash_fallback": "allow_local_hash_fallback",
    "hybrid_apc_require_attention_block_refs": "require_attention_block_refs",
    "hybrid_apc_reject_unbacked_attention_hits": "reject_unbacked_attention_hits",
    "hybrid_apc_cache_salt": "cache_salt",
    "hybrid_apc_model_revision": "model_revision",
    "hybrid_apc_layout_version": "layout_version",
    "hybrid_recurrent_cache_dtype": "recurrent_dtype",
    "hybrid_conv_cache_dtype": "conv_dtype",
}
_KV_CACHE_ATTENTION_LAYER_TYPES = {
    "attention",
    "full_attention",
    "self_attention",
    "sliding_attention",
}


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def _get_hf_config(vllm_config: Any) -> Any:
    model_config = getattr(vllm_config, "model_config", None)
    return getattr(model_config, "hf_config", None)


def _get_additional_config(vllm_config: Any) -> dict[str, Any]:
    additional_config = getattr(vllm_config, "additional_config", None)
    return additional_config if isinstance(additional_config, dict) else {}


def _config_flag(config: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(config, name, default))


def _config_value(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _num_layers_from_hf_config(
    hf_config: Any,
    original_get_kv_cache_spec: Any | None = None,
) -> int | None:
    if hf_config is None:
        return None
    original_globals = getattr(original_get_kv_cache_spec, "__globals__", {})
    get_num_layers = original_globals.get("get_num_layers_from_hf_config")
    if get_num_layers is not None:
        try:
            return int(get_num_layers(hf_config))
        except Exception:
            pass
    for attr in ("num_hidden_layers", "num_layers", "n_layer"):
        value = getattr(hf_config, attr, None)
        if value is not None:
            return int(value)
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is not None:
        try:
            return len(layer_types)
        except TypeError:
            return None
    return None


def _hybrid_kv_attention_layer_indices(
    hf_config: Any,
    num_layers: int,
) -> list[int] | None:
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is not None:
        try:
            layer_types = list(layer_types)
        except TypeError:
            layer_types = None
    if layer_types is not None and len(layer_types) == num_layers:
        attention_indices = [
            idx
            for idx, layer_type in enumerate(layer_types)
            if str(layer_type).lower() in _KV_CACHE_ATTENTION_LAYER_TYPES
        ]
        if 0 < len(attention_indices) < num_layers:
            return attention_indices
        return None

    full_attention_interval = getattr(hf_config, "full_attention_interval", None)
    if full_attention_interval:
        interval = int(full_attention_interval)
        if interval > 1:
            attention_indices = [
                idx for idx in range(num_layers) if (idx + 1) % interval == 0
            ]
            if attention_indices and len(attention_indices) < num_layers:
                return attention_indices
    return None


def _local_num_kv_heads(hf_config: Any, parallel_config: Any) -> int:
    tp_size = max(1, int(getattr(parallel_config, "tensor_parallel_size", 1) or 1))
    total_kv_heads = getattr(hf_config, "num_key_value_heads", None)
    if total_kv_heads is None:
        total_kv_heads = getattr(hf_config, "num_attention_heads", None)
    if total_kv_heads is None:
        return tp_size
    return max(1, int(total_kv_heads) // tp_size)


def _full_attention_spec_class(original_get_kv_cache_spec: Any) -> Any | None:
    original_globals = getattr(original_get_kv_cache_spec, "__globals__", {})
    spec_cls = original_globals.get("FullAttentionSpec")
    if spec_cls is not None:
        return spec_cls
    try:
        from vllm.v1.kv_cache_interface import FullAttentionSpec  # noqa: WPS433

        return FullAttentionSpec
    except Exception:
        return None


def _scheduler_config_flag(
    scheduler: Any,
    name: str,
    default: bool = False,
) -> bool:
    vllm_config = getattr(scheduler, "vllm_config", None)
    additional_config = _get_additional_config(vllm_config)
    if name in additional_config:
        return bool(additional_config[name])
    return _config_flag(_get_hf_config(vllm_config), name, default)


def _scheduler_config_value(
    scheduler: Any,
    name: str,
    default: Any,
) -> Any:
    vllm_config = getattr(scheduler, "vllm_config", None)
    additional_config = _get_additional_config(vllm_config)
    if name in additional_config:
        return additional_config[name]
    return _config_value(_get_hf_config(vllm_config), name, default)


def _max_num_seqs_for_scheduler(scheduler: Any) -> int:
    scheduler_config = getattr(scheduler, "scheduler_config", None)
    max_num_seqs = getattr(scheduler_config, "max_num_seqs", 1)
    return int(max_num_seqs or 1)


def _should_defer_waiting_prefills_while_running(scheduler: Any) -> bool:
    if _env_flag("QWEN36_HYBRID_APC_ALLOW_MIXED_PREFILL_DECODE"):
        return False
    if _scheduler_config_flag(
        scheduler,
        "hybrid_apc_allow_mixed_prefill_decode",
    ):
        return False
    if _env_flag("QWEN36_HYBRID_APC_DEFER_WAITING_WHILE_RUNNING"):
        return True
    return (
        _scheduler_config_flag(scheduler, "use_hybrid_apc_manager")
        and _scheduler_config_flag(scheduler, "use_qwen_hybrid_chunked_prefill")
        and _max_num_seqs_for_scheduler(scheduler) > 1
    )


def _new_empty_queue_like(queue: Any):
    try:
        return type(queue)()
    except Exception:
        return None


def _queue_add(queue: Any, request: Any) -> None:
    add_request = getattr(queue, "add_request", None)
    if add_request is not None:
        add_request(request)
    else:
        queue.append(request)


def _merge_waiting_queues(front: Any, back: Any):
    merged = _new_empty_queue_like(front)
    if merged is None:
        return back
    for request in front:
        _queue_add(merged, request)
    for request in back:
        _queue_add(merged, request)
    return merged


def _normalize_dtype(value: Any, default: str) -> str:
    if value is None:
        value = default
    normalized = str(value).lower()
    aliases = {
        "fp32": "float32",
        "float32": "float32",
        "torch.float32": "float32",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "torch.bfloat16": "bfloat16",
    }
    return aliases.get(normalized, normalized)


def _normalize_request_id(request_id: Any) -> Hashable | None:
    if request_id is None:
        return None
    if isinstance(request_id, list):
        return tuple(request_id)
    try:
        hash(request_id)
    except TypeError:
        return repr(request_id)
    return request_id


def _to_registry_key(key: Any) -> HybridGDNPrefixKey:
    return HybridGDNPrefixKey(
        cumulative_prefix_hash=getattr(key, "cumulative_prefix_hash"),
        prefix_len=int(getattr(key, "prefix_len")),
        block_size=int(getattr(key, "block_size")),
        cache_salt=getattr(key, "cache_salt", None),
        model_revision=str(getattr(key, "model_revision", "unknown")),
        layout_version=int(getattr(key, "layout_version", 1)),
        tp_rank=int(getattr(key, "tp_rank", 0)),
        recurrent_dtype=_normalize_dtype(
            getattr(key, "recurrent_dtype", None),
            "float32",
        ),
        conv_dtype=_normalize_dtype(getattr(key, "conv_dtype", None), "bfloat16"),
    )


def register_hybrid_apc_gdn_checkpoint(key: Any) -> HybridGDNPrefixKey:
    """Publish a committed GDN checkpoint boundary to the scheduler process."""

    registry_key = _to_registry_key(key)
    _GDN_PREFIX_KEYS.add(registry_key)
    if _env_flag("QWEN36_HYBRID_APC_DEBUG"):
        print(
            "[hybrid_apc_debug] scheduler-register "
            f"prefix_len={registry_key.prefix_len} "
            f"model_revision={registry_key.model_revision} "
            f"registry_size={len(_GDN_PREFIX_KEYS)}",
            flush=True,
        )
    return registry_key


def unregister_hybrid_apc_gdn_checkpoint(key: Any) -> bool:
    registry_key = _to_registry_key(key)
    if registry_key not in _GDN_PREFIX_KEYS:
        return False
    _GDN_PREFIX_KEYS.remove(registry_key)
    if _env_flag("QWEN36_HYBRID_APC_DEBUG"):
        print(
            "[hybrid_apc_debug] scheduler-unregister "
            f"prefix_len={registry_key.prefix_len} "
            f"model_revision={registry_key.model_revision} "
            f"registry_size={len(_GDN_PREFIX_KEYS)}",
            flush=True,
        )
    return True


def clear_hybrid_apc_gdn_checkpoint_registry() -> None:
    _GDN_PREFIX_KEYS.clear()
    _AUTHORIZED_PREFIX_READS.clear()
    _AUTHORIZED_PREFIX_READS_BY_REQUEST.clear()


def authorize_hybrid_apc_prefix_read(
    key: Any,
    *,
    request_id: Hashable | None = None,
) -> HybridGDNPrefixKey:
    """Publish a scheduler-approved prefix read for suffix-only model prep."""

    registry_key = _to_registry_key(key)
    normalized_request_id = _normalize_request_id(request_id)
    if normalized_request_id is None:
        _AUTHORIZED_PREFIX_READS.setdefault(registry_key.prefix_len, []).append(
            registry_key
        )
    else:
        _AUTHORIZED_PREFIX_READS_BY_REQUEST.setdefault(
            normalized_request_id,
            [],
        ).append(registry_key)
    return registry_key


def _pop_matching_authorized_key(
    candidates: list[HybridGDNPrefixKey],
    *,
    prefix_len: int,
    cache_salt: Hashable | None,
    model_revision: str,
    layout_version: int,
    tp_rank: int,
    recurrent_dtype: str,
    conv_dtype: str,
) -> HybridGDNPrefixKey | None:
    for idx, key in enumerate(candidates):
        if key.prefix_len != prefix_len:
            continue
        if key.cache_salt != cache_salt:
            continue
        if key.model_revision != str(model_revision):
            continue
        if key.layout_version != int(layout_version):
            continue
        if key.tp_rank != int(tp_rank):
            continue
        if key.recurrent_dtype != recurrent_dtype or key.conv_dtype != conv_dtype:
            continue
        return candidates.pop(idx)
    return None


def pop_hybrid_apc_authorized_prefix_key(
    *,
    prefix_len: int,
    request_id: Hashable | None = None,
    cache_salt: Hashable | None = None,
    model_revision: str = "unknown",
    layout_version: int = 1,
    tp_rank: int = 0,
    recurrent_dtype: str = "float32",
    conv_dtype: str = "bfloat16",
) -> HybridGDNPrefixKey | None:
    """Consume the exact key for a prefix read allowed by the scheduler."""

    prefix_len = int(prefix_len)
    recurrent_dtype = _normalize_dtype(recurrent_dtype, "float32")
    conv_dtype = _normalize_dtype(conv_dtype, "bfloat16")
    normalized_request_id = _normalize_request_id(request_id)
    if normalized_request_id is not None:
        candidates = _AUTHORIZED_PREFIX_READS_BY_REQUEST.get(normalized_request_id)
        if candidates:
            matched = _pop_matching_authorized_key(
                candidates,
                prefix_len=prefix_len,
                cache_salt=cache_salt,
                model_revision=model_revision,
                layout_version=layout_version,
                tp_rank=tp_rank,
                recurrent_dtype=recurrent_dtype,
                conv_dtype=conv_dtype,
            )
            if matched is not None:
                if not candidates:
                    _AUTHORIZED_PREFIX_READS_BY_REQUEST.pop(
                        normalized_request_id,
                        None,
                    )
                return matched

    candidates = _AUTHORIZED_PREFIX_READS.get(prefix_len)
    if not candidates:
        return None
    matched = _pop_matching_authorized_key(
        candidates,
        prefix_len=prefix_len,
        cache_salt=cache_salt,
        model_revision=model_revision,
        layout_version=layout_version,
        tp_rank=tp_rank,
        recurrent_dtype=recurrent_dtype,
        conv_dtype=conv_dtype,
    )
    if matched is not None and not candidates:
        _AUTHORIZED_PREFIX_READS.pop(prefix_len, None)
    if matched is not None:
        return matched
    return None


def _block_size_for_scheduler(scheduler: Any) -> int:
    cache_config = getattr(scheduler, "cache_config", None)
    block_size = getattr(cache_config, "block_size", None)
    if block_size is None:
        hf_config = _get_hf_config(getattr(scheduler, "vllm_config", None))
        block_size = _config_value(
            hf_config,
            "gdn_checkpoint_interval",
            0,
        )
    return int(block_size or 0)


def _local_cumulative_prefix_hashes(
    token_ids: list[int] | tuple[int, ...],
    *,
    block_size: int,
    max_prefix_len: int,
) -> dict[int, str]:
    max_prefix_len = max(0, int(max_prefix_len))
    max_prefix_len = max_prefix_len // block_size * block_size
    parent_digest = b""
    hashes: dict[int, str] = {}
    for block_start in range(0, max_prefix_len, block_size):
        block_end = block_start + block_size
        block = [int(token) for token in token_ids[block_start:block_end]]
        digest = hashlib.blake2b(digest_size=16)
        digest.update(parent_digest)
        digest.update(struct.pack("<QQ", block_size, block_end))
        digest.update(struct.pack("<" + "q" * len(block), *block))
        parent_digest = digest.digest()
        hashes[block_end] = parent_digest.hex()
    return hashes


def _vllm_cumulative_prefix_hashes(
    request: Any,
    *,
    block_size: int,
    max_prefix_len: int | None = None,
) -> dict[int, Hashable]:
    block_hashes = list(getattr(request, "block_hashes", ()) or ())
    if not block_hashes:
        return {}
    if max_prefix_len is None:
        max_prefix_len = len(block_hashes) * block_size
    max_prefix_len = max(0, int(max_prefix_len))
    max_prefix_len = max_prefix_len // block_size * block_size
    hashes: dict[int, Hashable] = {}
    for index, block_hash in enumerate(block_hashes):
        prefix_len = (index + 1) * block_size
        if prefix_len > max_prefix_len:
            break
        hashes[prefix_len] = block_hash
    return hashes


def _candidate_cumulative_prefix_hashes(
    scheduler: Any,
    request: Any,
    *,
    max_prefix_len: int,
) -> list[dict[int, Hashable]]:
    block_size = _block_size_for_scheduler(scheduler)
    if block_size <= 0:
        return []
    candidates = []
    vllm_hashes = _vllm_cumulative_prefix_hashes(
        request,
        block_size=block_size,
        max_prefix_len=max_prefix_len,
    )
    if vllm_hashes:
        candidates.append(vllm_hashes)
    token_ids = getattr(request, "prompt_token_ids", None)
    if token_ids:
        local_hashes = _local_cumulative_prefix_hashes(
            token_ids,
            block_size=block_size,
            max_prefix_len=max_prefix_len,
        )
        if local_hashes:
            candidates.append(local_hashes)
    return candidates


def _request_registry_key(
    *,
    scheduler: Any,
    request: Any,
    cumulative_prefix_hash: Hashable,
    prefix_len: int,
    block_size: int,
) -> HybridGDNPrefixKey:
    return HybridGDNPrefixKey(
        cumulative_prefix_hash=cumulative_prefix_hash,
        prefix_len=int(prefix_len),
        block_size=int(block_size),
        cache_salt=getattr(request, "cache_salt", None),
        model_revision=str(
            _scheduler_config_value(
                scheduler,
                "hybrid_apc_model_revision",
                "unknown",
            )
        ),
        layout_version=int(
            _scheduler_config_value(scheduler, "hybrid_apc_layout_version", 1)
        ),
        tp_rank=int(_scheduler_config_value(scheduler, "tp_rank", 0)),
        recurrent_dtype=_normalize_dtype(
            _scheduler_config_value(
                scheduler,
                "hybrid_recurrent_cache_dtype",
                _scheduler_config_value(
                    scheduler,
                    "gdn_recurrent_cache_dtype",
                    "float32",
                ),
            ),
            "float32",
        ),
        conv_dtype=_normalize_dtype(
            _scheduler_config_value(
                scheduler,
                "hybrid_conv_cache_dtype",
                _scheduler_config_value(
                    scheduler,
                    "gdn_conv_cache_dtype",
                    "bfloat16",
                ),
            ),
            "bfloat16",
        ),
    )


def _request_max_cache_hit_len(scheduler: Any, request: Any) -> int:
    if request is None:
        return 0
    block_size = _block_size_for_scheduler(scheduler)
    if block_size <= 0:
        return 0
    token_ids = getattr(request, "prompt_token_ids", None)
    token_count = int(getattr(request, "num_tokens", len(token_ids or ())))
    max_cache_hit_len = max(0, token_count - 1)
    if token_ids:
        max_cache_hit_len = min(max_cache_hit_len, len(token_ids))
    return max_cache_hit_len


def backed_gdn_prefix_hits(scheduler: Any, request: Any) -> dict[int, HybridGDNPrefixKey]:
    """Return request prefix lengths with registered GDN checkpoints."""

    if request is None:
        return {}
    block_size = _block_size_for_scheduler(scheduler)
    max_cache_hit_len = _request_max_cache_hit_len(scheduler, request)
    if block_size <= 0 or max_cache_hit_len <= 0:
        return {}
    hits: dict[int, HybridGDNPrefixKey] = {}
    for hashes in _candidate_cumulative_prefix_hashes(
        scheduler,
        request,
        max_prefix_len=max_cache_hit_len,
    ):
        for prefix_len in sorted(hashes, reverse=True):
            if prefix_len in hits:
                continue
            key = _request_registry_key(
                scheduler=scheduler,
                request=request,
                cumulative_prefix_hash=hashes[prefix_len],
                prefix_len=prefix_len,
                block_size=block_size,
            )
            if key in _GDN_PREFIX_KEYS:
                hits[prefix_len] = key
    return hits


def backed_gdn_prefix_hit(scheduler: Any, request: Any) -> HybridGDNPrefixKey | None:
    """Return the largest request prefix with a registered GDN checkpoint."""

    hits = backed_gdn_prefix_hits(scheduler, request)
    if not hits:
        return None
    return hits[max(hits)]


def _required_backed_prefix_lens(scheduler: Any, request: Any) -> tuple[int, ...]:
    if request is None:
        return ()
    block_size = _block_size_for_scheduler(scheduler)
    max_cache_hit_len = _request_max_cache_hit_len(scheduler, request)
    if block_size <= 0 or max_cache_hit_len <= 0:
        return ()
    required: set[int] = set()
    for hashes in _candidate_cumulative_prefix_hashes(
        scheduler,
        request,
        max_prefix_len=max_cache_hit_len,
    ):
        required.update(int(prefix_len) for prefix_len in hashes)
    return tuple(sorted(required))


def _block_id_groups(block_ids: Any) -> list[list[int]]:
    if block_ids is None:
        return []
    if isinstance(block_ids, tuple):
        groups = block_ids
    elif (
        isinstance(block_ids, list)
        and block_ids
        and all(isinstance(item, (list, tuple)) for item in block_ids)
    ):
        groups = tuple(block_ids)
    else:
        groups = (block_ids,)
    normalized = []
    for group in groups:
        try:
            normalized.append([int(block_id) for block_id in group])
        except TypeError:
            continue
    return normalized


def _attention_block_refs_by_prefix_len(
    block_ids: Any,
    *,
    block_size: int,
) -> dict[int, tuple[int, ...]]:
    groups = _block_id_groups(block_ids)
    if not groups:
        return {}
    max_blocks = max(len(group) for group in groups)
    refs_by_prefix_len: dict[int, tuple[int, ...]] = {}
    for block_count in range(1, max_blocks + 1):
        refs: list[int] = []
        for group in groups:
            refs.extend(group[:block_count])
        if refs:
            refs_by_prefix_len[block_count * block_size] = tuple(refs)
    return refs_by_prefix_len


def _scheduler_request_metadata(
    scheduler: Any,
    request: Any,
    *,
    block_ids: Any = None,
    num_computed_tokens: int | None = None,
    active_suffix_len: int | None = None,
) -> dict[str, Any]:
    block_size = _block_size_for_scheduler(scheduler)
    if request is None or block_size <= 0:
        return {}
    token_ids = getattr(request, "prompt_token_ids", None)
    prompt_token_count = int(
        getattr(request, "num_prompt_tokens", len(token_ids or ())) or 0
    )
    if token_ids is not None:
        prompt_token_count = min(prompt_token_count, len(token_ids))
    full_request_prefix_len = prompt_token_count
    request_prefix_len = full_request_prefix_len
    if num_computed_tokens is not None and active_suffix_len is not None:
        scheduled_prefix_len = int(num_computed_tokens) + int(active_suffix_len)
        request_prefix_len = max(0, min(full_request_prefix_len, scheduled_prefix_len))
    cumulative_hashes = _vllm_cumulative_prefix_hashes(
        request,
        block_size=block_size,
        max_prefix_len=request_prefix_len,
    )
    metadata: dict[str, Any] = {}
    if cumulative_hashes:
        metadata["cumulative_hashes_by_prefix_len"] = cumulative_hashes
    refs_by_prefix_len = _attention_block_refs_by_prefix_len(
        block_ids,
        block_size=block_size,
    )
    if refs_by_prefix_len:
        metadata["attention_block_refs_by_prefix_len"] = refs_by_prefix_len
    metadata["request_prefix_len"] = request_prefix_len
    full_token_ids = token_ids or getattr(request, "all_token_ids", None)
    has_computed_prefix = (
        num_computed_tokens is not None and int(num_computed_tokens) > 0
    )
    if has_computed_prefix and full_token_ids is not None:
        full_token_ids = list(full_token_ids)
        if len(full_token_ids) >= request_prefix_len:
            metadata["full_input_ids"] = tuple(
                int(token_id)
                for token_id in full_token_ids[:request_prefix_len]
            )
    if num_computed_tokens is not None:
        metadata["vllm_attention_hit_len"] = int(num_computed_tokens)
    if active_suffix_len is not None:
        metadata["active_suffix_len"] = int(active_suffix_len)
    return metadata


def _unbacked_prefix_reads_disabled_requested(scheduler: Any) -> bool:
    if _env_flag("QWEN36_HYBRID_APC_ENABLE_PREFIX_READS"):
        return False

    disable_requested = _env_flag("QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS")
    if disable_requested:
        return True
    if not _scheduler_config_flag(scheduler, "use_hybrid_apc_manager"):
        return False
    if _scheduler_config_flag(
        scheduler,
        "hybrid_apc_reject_unbacked_attention_hits",
    ) or _scheduler_config_flag(
        scheduler,
        "hybrid_apc_require_vllm_metadata",
    ):
        return True
    return _scheduler_config_flag(
        scheduler,
        "hybrid_apc_disable_unbacked_prefix_reads",
    )


def _request_prefix_len(request: Any) -> int:
    if request is None:
        return 0
    token_ids = getattr(request, "prompt_token_ids", None)
    return int(getattr(request, "num_tokens", len(token_ids or ())) or 0)


def _backed_prefix_read_decision(scheduler: Any, request: Any) -> dict[str, Any]:
    backed_hits = backed_gdn_prefix_hits(scheduler, request)
    required_prefix_lens = _required_backed_prefix_lens(scheduler, request)
    backed_hit_len = max(backed_hits) if backed_hits else 0
    max_readable_prefix_len = max(required_prefix_lens) if required_prefix_lens else 0
    missing_higher_backed_lens = [
        prefix_len
        for prefix_len in required_prefix_lens
        if prefix_len > backed_hit_len and prefix_len not in backed_hits
    ]
    supports_backed = _supports_backed_prefix_reads(scheduler)
    max_backed_prefix_read_len = _max_backed_prefix_read_len(scheduler)
    exceeds_backed_prefix_cap = (
        max_backed_prefix_read_len > 0
        and backed_hit_len > max_backed_prefix_read_len
    )
    capped_backed_hits = [
        prefix_len
        for prefix_len in backed_hits
        if max_backed_prefix_read_len <= 0 or prefix_len <= max_backed_prefix_read_len
    ]
    prefix_read_len = max(capped_backed_hits) if capped_backed_hits else 0
    allowed = (
        prefix_read_len > 0
        and supports_backed
    )
    return {
        "allowed": allowed,
        "backed_hits": backed_hits,
        "required_prefix_lens": required_prefix_lens,
        "backed_hit_len": backed_hit_len,
        "prefix_read_len": prefix_read_len,
        "missing_backed_lens": tuple(missing_higher_backed_lens),
        "supports_backed": supports_backed,
        "max_backed_prefix_read_len": max_backed_prefix_read_len,
        "exceeds_backed_prefix_cap": exceeds_backed_prefix_cap,
    }


def _request_from_scheduler(scheduler: Any, req_id: Any) -> Any:
    requests = getattr(scheduler, "requests", None)
    if isinstance(requests, dict):
        return requests.get(req_id)
    return None


def _attach_scheduler_output_metadata(scheduler: Any, scheduler_output: Any) -> None:
    metadata_by_request_id: dict[Hashable, dict[str, Any]] = {}
    active_suffix_lens = _num_scheduled_tokens_by_request_id(scheduler_output)
    for req_data in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
        req_id = getattr(req_data, "req_id", None)
        request = _request_from_scheduler(scheduler, req_id)
        metadata = _scheduler_request_metadata(
            scheduler,
            request,
            block_ids=getattr(req_data, "block_ids", None),
            num_computed_tokens=getattr(req_data, "num_computed_tokens", None),
            active_suffix_len=active_suffix_lens.get(_normalize_request_id(req_id)),
        )
        if metadata:
            metadata_by_request_id[_normalize_request_id(req_id)] = metadata
            _authorize_scheduled_prefix_read(
                scheduler,
                request,
                request_id=req_id,
                prefix_len=metadata.get("vllm_attention_hit_len"),
            )

    cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
    req_ids = list(getattr(cached_reqs, "req_ids", ()) or ())
    new_block_ids = list(getattr(cached_reqs, "new_block_ids", ()) or ())
    num_computed_tokens = list(
        getattr(cached_reqs, "num_computed_tokens", ()) or ()
    )
    for index, req_id in enumerate(req_ids):
        request = _request_from_scheduler(scheduler, req_id)
        block_ids = new_block_ids[index] if index < len(new_block_ids) else None
        computed = (
            int(num_computed_tokens[index])
            if index < len(num_computed_tokens)
            else None
        )
        metadata = _scheduler_request_metadata(
            scheduler,
            request,
            block_ids=block_ids,
            num_computed_tokens=computed,
            active_suffix_len=active_suffix_lens.get(_normalize_request_id(req_id)),
        )
        if metadata:
            metadata_by_request_id[_normalize_request_id(req_id)] = metadata
            _authorize_scheduled_prefix_read(
                scheduler,
                request,
                request_id=req_id,
                prefix_len=metadata.get("vllm_attention_hit_len"),
            )

    if metadata_by_request_id:
        setattr(
            scheduler_output,
            _SCHEDULER_OUTPUT_METADATA_ATTR,
            metadata_by_request_id,
        )


def backed_gdn_prefix_hit_len(scheduler: Any, request: Any) -> int:
    hit = backed_gdn_prefix_hit(scheduler, request)
    if hit is None:
        return 0
    return hit.prefix_len


def _request_id_for_scheduler_request(request: Any) -> Hashable | None:
    if request is None:
        return None
    for attr in ("request_id", "req_id", "id"):
        request_id = getattr(request, attr, None)
        if request_id is not None:
            return _normalize_request_id(request_id)
    return None


def _supports_backed_prefix_reads(scheduler: Any) -> bool:
    """Return whether this artifact can consume a backed Hybrid APC prefix."""

    if _env_flag("QWEN36_HYBRID_APC_ENABLE_BACKED_PREFIX_READS"):
        return True

    if not _scheduler_config_flag(scheduler, "hybrid_apc_enable_backed_prefix_reads"):
        return False

    # A backed GDN checkpoint is not enough on its own. The CTE graph must also
    # consume attention KV prefix state; otherwise warm requests restore GDN
    # state but full-attention layers still see only the suffix.
    return _scheduler_config_flag(scheduler, "use_qwen_hybrid_chunked_prefill")


def _max_backed_prefix_read_len(scheduler: Any) -> int:
    env_value = _env_int("QWEN36_HYBRID_APC_MAX_BACKED_PREFIX_READ_LEN")
    if env_value is not None:
        return max(0, env_value)
    return max(
        0,
        int(
            _scheduler_config_value(
                scheduler,
                "hybrid_apc_max_backed_prefix_read_len",
                0,
            )
            or 0
        ),
    )


def _set_request_prefix_cache_cap(
    request: Any,
    *,
    prefix_len: int,
    block_size: int,
) -> None:
    if request is None:
        return
    prefix_len = max(0, int(prefix_len))
    block_size = max(1, int(block_size))
    try:
        setattr(request, _MAX_PREFIX_CACHE_HIT_LEN_ATTR, prefix_len)
        setattr(
            request,
            _MAX_PREFIX_CACHE_BLOCKS_ATTR,
            prefix_len // block_size,
        )
    except Exception:
        return


def should_disable_unbacked_prefix_reads(scheduler: Any, request: Any = None) -> bool:
    """Return whether this scheduler should avoid vLLM APC reads.

    The current Qwen Hybrid APC control plane can prove an attention hit is
    invalid only inside model request prep. That is too late for allocation.
    This opt-in fallback makes vLLM allocate the request as no-prefix unless
    the scheduler process has a registered matching GDN checkpoint boundary and
    the compiled artifact can consume the matching attention KV prefix in CTE.
    """

    disable_requested = _unbacked_prefix_reads_disabled_requested(scheduler)
    if not disable_requested:
        return False
    decision = _backed_prefix_read_decision(scheduler, request)
    if _env_flag("QWEN36_HYBRID_APC_DEBUG"):
        prompt_len = len(getattr(request, "prompt_token_ids", ()) or ())
        print(
            "[hybrid_apc_debug] scheduler-decision "
            f"disable_requested={disable_requested} "
            f"backed_hit_len={decision['backed_hit_len']} "
            f"prefix_read_len={decision['prefix_read_len']} "
            f"supports_backed={decision['supports_backed']} "
            f"max_num_seqs={_max_num_seqs_for_scheduler(scheduler)} "
            f"prompt_len={prompt_len} "
            f"required_backed_lens={decision['required_prefix_lens']} "
            f"missing_backed_lens={decision['missing_backed_lens']} "
            f"max_backed_prefix_read_len={decision['max_backed_prefix_read_len']} "
            f"exceeds_backed_prefix_cap={decision['exceeds_backed_prefix_cap']} "
            f"registry_size={len(_GDN_PREFIX_KEYS)}",
            flush=True,
        )
    if decision["allowed"]:
        request_id = _request_id_for_scheduler_request(request)
        prefix_len = decision["prefix_read_len"]
        _set_request_prefix_cache_cap(
            request,
            prefix_len=prefix_len,
            block_size=_block_size_for_scheduler(scheduler),
        )
        authorize_hybrid_apc_prefix_read(
            decision["backed_hits"][prefix_len],
            request_id=request_id,
        )
        return False
    return True


def patch_scheduler_class(scheduler_cls: type) -> bool:
    """Patch a vLLM Scheduler class in-place.

    Returns True if this call installed the patch, False if the class was
    already patched.
    """

    original_add_request = getattr(scheduler_cls, "add_request", None)
    if original_add_request is None:
        raise AttributeError(f"{scheduler_cls!r} has no add_request method")
    installed = False

    if not getattr(original_add_request, "_qwen36_hybrid_apc_patched", False):

        def add_request_with_hybrid_apc_fallback(self, request):
            if should_disable_unbacked_prefix_reads(self, request):
                request.skip_reading_prefix_cache = True
            return original_add_request(self, request)

        add_request_with_hybrid_apc_fallback._qwen36_hybrid_apc_patched = True
        add_request_with_hybrid_apc_fallback._qwen36_original_add_request = (
            original_add_request
        )
        scheduler_cls.add_request = add_request_with_hybrid_apc_fallback
        installed = True

    original_schedule = getattr(scheduler_cls, "schedule", None)
    if original_schedule is not None and not getattr(
        original_schedule,
        "_qwen36_hybrid_apc_metadata_patched",
        False,
    ):

        def schedule_with_hybrid_apc_metadata(self, *args, **kwargs):
            deferred_waiting = None
            temporary_waiting = None
            waiting = getattr(self, "waiting", None)
            running = getattr(self, "running", None)
            if (
                waiting
                and running
                and _should_defer_waiting_prefills_while_running(self)
            ):
                temporary_waiting = _new_empty_queue_like(waiting)
                if temporary_waiting is not None:
                    deferred_waiting = waiting
                    self.waiting = temporary_waiting
            try:
                scheduler_output = original_schedule(self, *args, **kwargs)
            finally:
                if deferred_waiting is not None:
                    current_waiting = getattr(self, "waiting", temporary_waiting)
                    if current_waiting:
                        self.waiting = _merge_waiting_queues(
                            current_waiting,
                            deferred_waiting,
                        )
                    else:
                        self.waiting = deferred_waiting
            _attach_scheduler_output_metadata(self, scheduler_output)
            return scheduler_output

        schedule_with_hybrid_apc_metadata._qwen36_hybrid_apc_metadata_patched = True
        schedule_with_hybrid_apc_metadata._qwen36_original_schedule = (
            original_schedule
        )
        scheduler_cls.schedule = schedule_with_hybrid_apc_metadata
        installed = True

    return installed


def _patch_scheduler_module(module: Any) -> bool:
    scheduler_cls = getattr(module, "Scheduler", None)
    if scheduler_cls is None:
        return False
    installed = patch_scheduler_class(scheduler_cls)
    if installed:
        logger.info("Installed Qwen Hybrid APC scheduler fallback patch")
    return installed


def _patch_kv_cache_manager_module(module: Any) -> bool:
    kv_cache_manager_cls = getattr(module, "KVCacheManager", None)
    if kv_cache_manager_cls is None:
        return False
    original_get_computed_blocks = getattr(
        kv_cache_manager_cls,
        "get_computed_blocks",
        None,
    )
    if original_get_computed_blocks is None or getattr(
        original_get_computed_blocks,
        "_qwen36_hybrid_apc_prefix_cap_patched",
        False,
    ):
        return False

    def get_computed_blocks_with_hybrid_apc_cap(self, request, *args, **kwargs):
        cap_blocks = getattr(request, _MAX_PREFIX_CACHE_BLOCKS_ATTR, None)
        try:
            cap_blocks = None if cap_blocks is None else max(0, int(cap_blocks))
        except (TypeError, ValueError):
            cap_blocks = None
        if cap_blocks is None:
            return original_get_computed_blocks(self, request, *args, **kwargs)
        if cap_blocks <= 0:
            return getattr(self, "empty_kv_cache_blocks"), 0

        block_hashes = getattr(request, "block_hashes", None)
        if not block_hashes or len(block_hashes) <= cap_blocks:
            return original_get_computed_blocks(self, request, *args, **kwargs)

        original_block_hashes = block_hashes
        if isinstance(block_hashes, tuple):
            capped_block_hashes = block_hashes[:cap_blocks]
        else:
            capped_block_hashes = list(block_hashes[:cap_blocks])
        try:
            request.block_hashes = capped_block_hashes
            return original_get_computed_blocks(self, request, *args, **kwargs)
        finally:
            request.block_hashes = original_block_hashes

    get_computed_blocks_with_hybrid_apc_cap._qwen36_hybrid_apc_prefix_cap_patched = (
        True
    )
    get_computed_blocks_with_hybrid_apc_cap._qwen36_original_get_computed_blocks = (
        original_get_computed_blocks
    )
    kv_cache_manager_cls.get_computed_blocks = get_computed_blocks_with_hybrid_apc_cap
    logger.info("Installed Qwen Hybrid APC KV prefix cap patch")
    return True


def _request_ids_from_model_input(model_input: Any) -> tuple[Hashable, ...] | None:
    request_ids = getattr(model_input, "request_ids", None)
    return _as_request_id_tuple(request_ids)


def _as_request_id_tuple(request_ids: Any) -> tuple[Hashable, ...] | None:
    if request_ids is None:
        return None
    if isinstance(request_ids, tuple):
        return request_ids
    if isinstance(request_ids, list):
        return tuple(request_ids)
    if isinstance(request_ids, (str, bytes)):
        return (request_ids,)
    try:
        return tuple(request_ids)
    except TypeError:
        return (request_ids,)


def _request_ids_from_model_input_or_scheduler_output(
    model_input: Any,
    scheduler_output: Any,
) -> tuple[Hashable, ...] | None:
    request_ids = _request_ids_from_model_input(model_input)
    if request_ids:
        return request_ids
    cached_request_ids = _request_ids_from_scheduler_output(
        scheduler_output,
        kind="cached",
    )
    new_request_ids = _request_ids_from_scheduler_output(
        scheduler_output,
        kind="new",
    )
    combined = tuple(cached_request_ids or ()) + tuple(new_request_ids or ())
    return combined or None


def _request_ids_from_scheduler_output(
    scheduler_output: Any,
    *,
    kind: str,
) -> tuple[Hashable, ...] | None:
    if kind == "cached":
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        return _as_request_id_tuple(getattr(cached_reqs, "req_ids", None))
    if kind == "new":
        new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
        if new_reqs is None:
            return None
        return tuple(getattr(req, "req_id") for req in new_reqs)
    raise ValueError(f"unknown scheduler request kind: {kind}")


def _num_scheduled_tokens_by_request_id(scheduler_output: Any) -> dict[Hashable, int]:
    values = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(values, dict):
        return {}
    scheduled_tokens: dict[Hashable, int] = {}
    for req_id, value in values.items():
        normalized = _normalize_request_id(req_id)
        if normalized is None:
            continue
        try:
            scheduled_tokens[normalized] = int(value)
        except (TypeError, ValueError):
            continue
    return scheduled_tokens


def _scheduler_metadata_for_request_id(
    metadata_by_request_id: Any,
    request_id: Any,
) -> dict[str, Any]:
    if not isinstance(metadata_by_request_id, dict):
        return {}
    normalized = _normalize_request_id(request_id)
    metadata = metadata_by_request_id.get(normalized)
    if metadata is None and request_id is not None:
        metadata = metadata_by_request_id.get(str(request_id))
    return metadata if isinstance(metadata, dict) else {}


def _authorize_scheduled_prefix_read(
    scheduler: Any,
    request: Any,
    *,
    request_id: Any,
    prefix_len: int | None,
) -> None:
    """Authorize a vLLM prefix hit that is backed by a committed GDN checkpoint."""

    if request is None or prefix_len is None:
        return
    prefix_len = int(prefix_len)
    if prefix_len <= 0 or not _supports_backed_prefix_reads(scheduler):
        return
    max_backed_prefix_read_len = _max_backed_prefix_read_len(scheduler)
    if max_backed_prefix_read_len > 0 and prefix_len > max_backed_prefix_read_len:
        return
    key = backed_gdn_prefix_hits(scheduler, request).get(prefix_len)
    if key is None:
        return
    authorize_hybrid_apc_prefix_read(
        key,
        request_id=_request_id_for_scheduler_request(request) or request_id,
    )


def _hybrid_apc_request_records_from_model_input(
    model_input: Any,
    scheduler_output: Any,
) -> tuple[dict[str, Any], ...] | None:
    request_ids = _request_ids_from_model_input_or_scheduler_output(
        model_input,
        scheduler_output,
    )
    if not request_ids:
        return None
    metadata_by_request_id = getattr(
        scheduler_output,
        _SCHEDULER_OUTPUT_METADATA_ATTR,
        None,
    )
    if not isinstance(metadata_by_request_id, dict):
        return None

    num_scheduled_tokens = _num_scheduled_tokens_by_request_id(scheduler_output)
    records: list[dict[str, Any]] = []
    found_metadata = False
    for request_id in request_ids:
        metadata = _scheduler_metadata_for_request_id(metadata_by_request_id, request_id)
        record: dict[str, Any] = {"request_id": request_id}
        for key in (
            "cumulative_hashes_by_prefix_len",
            "attention_block_refs_by_prefix_len",
            "request_prefix_len",
            "full_input_ids",
            "vllm_attention_hit_len",
        ):
            if key in metadata:
                record[key] = metadata[key]
                found_metadata = True
        normalized = _normalize_request_id(request_id)
        if normalized in num_scheduled_tokens:
            record["active_suffix_len"] = num_scheduled_tokens[normalized]
        records.append(record)
    return tuple(records) if found_metadata else None


def _request_id_target_models(model: Any) -> list[Any]:
    targets = []
    seen = set()
    current = model
    for _ in range(4):
        if current is None:
            break
        current_id = id(current)
        if current_id in seen:
            break
        seen.add(current_id)
        targets.append(current)
        for attr in (
            "context_encoding_model",
            "token_generation_model",
            "fused_spec_model",
        ):
            wrapper = getattr(current, attr, None)
            if wrapper is None:
                continue
            wrapper_id = id(wrapper)
            if wrapper_id in seen:
                continue
            seen.add(wrapper_id)
            targets.append(wrapper)
        current = getattr(current, "model", None)
    return targets


def _runner_hybrid_apc_runtime_config(runner: Any) -> dict[str, Any]:
    additional_config = _get_additional_config(getattr(runner, "vllm_config", None))
    runtime_config = {
        key: additional_config[key]
        for key in _HYBRID_APC_RUNTIME_CONFIG_KEYS
        if key in additional_config
    }
    if runtime_config.get("hybrid_apc_require_vllm_metadata"):
        runtime_config["hybrid_apc_allow_local_hash_fallback"] = False
        runtime_config["hybrid_apc_require_attention_block_refs"] = True
        runtime_config["hybrid_apc_reject_unbacked_attention_hits"] = True
    return runtime_config


def _config_targets_for_model(model: Any) -> list[Any]:
    targets = []
    seen = set()
    for target in _request_id_target_models(model):
        config = getattr(target, "config", None)
        if config is not None:
            config_id = id(config)
            if config_id not in seen:
                seen.add(config_id)
                targets.append(config)
        if any(hasattr(target, key) for key in _HYBRID_APC_RUNTIME_CONFIG_KEYS):
            target_id = id(target)
            if target_id not in seen:
                seen.add(target_id)
                targets.append(target)
    return targets


def _apply_runtime_config_values(
    *,
    target: Any,
    values: dict[str, Any],
    previous_values: list[tuple[Any, str, Any]],
    missing: Any,
) -> None:
    for attr, value in values.items():
        previous_values.append((target, attr, getattr(target, attr, missing)))
        setattr(target, attr, value)


def _apply_hybrid_apc_runtime_config(
    model: Any,
    values: dict[str, Any],
    *,
    previous_values: list[tuple[Any, str, Any]],
    missing: Any,
) -> None:
    if not values:
        return
    for target in _config_targets_for_model(model):
        _apply_runtime_config_values(
            target=target,
            values=values,
            previous_values=previous_values,
            missing=missing,
        )
    bridge_values = {
        bridge_attr: values[config_attr]
        for config_attr, bridge_attr in _HYBRID_APC_BRIDGE_CONFIG_ATTRS.items()
        if config_attr in values
    }
    if not bridge_values:
        return
    for target in _request_id_target_models(model):
        bridge = getattr(target, "hybrid_apc_bridge", None)
        if bridge is None:
            continue
        _apply_runtime_config_values(
            target=bridge,
            values=bridge_values,
            previous_values=previous_values,
            missing=missing,
        )


def _debug_logits_tensor(stage: str, tensor: Any) -> None:
    if not _env_flag("QWEN36_VLLM_LOGITS_DEBUG"):
        return
    if tensor is None or not hasattr(tensor, "numel"):
        print(f"[qwen36_vllm_logits_debug] stage={stage} tensor=none", flush=True)
        return
    try:
        import torch  # noqa: WPS433

        if tensor.numel() == 0:
            print(
                "[qwen36_vllm_logits_debug] "
                f"stage={stage} shape={tuple(tensor.shape)} dtype={tensor.dtype} empty",
                flush=True,
            )
            return
        flat = tensor.detach().reshape(-1)
        if torch.is_floating_point(flat):
            finite_mask = torch.isfinite(flat)
            finite_count = int(finite_mask.sum().item())
            nan_count = int(torch.isnan(flat).sum().item())
            posinf_count = int(
                torch.logical_and(torch.isinf(flat), flat > 0).sum().item()
            )
            neginf_count = int(
                torch.logical_and(torch.isinf(flat), flat < 0).sum().item()
            )
            if finite_count:
                finite_flat = flat[finite_mask].float()
                finite_min = float(finite_flat.min().item())
                finite_max = float(finite_flat.max().item())
            else:
                finite_min = "none"
                finite_max = "none"
            row_argmax = []
            row_argmax_values = []
            if tensor.ndim >= 2:
                rows = tensor.detach().float().reshape(tensor.shape[0], -1)
                argmax = rows.argmax(dim=-1)
                row_argmax = [int(item) for item in argmax[:8].cpu().tolist()]
                row_argmax_values = [
                    float(rows[row, argmax[row]].item())
                    for row in range(min(rows.shape[0], 8))
                ]
            print(
                "[qwen36_vllm_logits_debug] "
                f"stage={stage} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                f"finite={finite_count} nan={nan_count} posinf={posinf_count} "
                f"neginf={neginf_count} finite_min={finite_min} "
                f"finite_max={finite_max} row_argmax={row_argmax} "
                f"row_argmax_values={row_argmax_values}",
                flush=True,
            )
        else:
            flat_i64 = flat.to(torch.int64)
            print(
                "[qwen36_vllm_logits_debug] "
                f"stage={stage} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                f"min={int(flat_i64.min().item())} max={int(flat_i64.max().item())}",
                flush=True,
            )
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(
            "[qwen36_vllm_logits_debug] "
            f"stage={stage} summary_error={type(exc).__name__}: {exc}",
            flush=True,
        )


def _expand_completed_prefill_logits(hidden_states: Any, model_input: Any) -> Any:
    """Restore completed-only CTE logits to vLLM's scheduled request rows."""
    prefill_state = getattr(model_input, "prefill_completion_state", None)
    if prefill_state is None or not hasattr(hidden_states, "shape"):
        return hidden_states
    if len(getattr(hidden_states, "shape", ())) == 0:
        return hidden_states

    try:
        import torch  # noqa: WPS433

        if hasattr(prefill_state, "detach"):
            state_values = [
                bool(item)
                for item in prefill_state.detach().cpu().reshape(-1).tolist()
            ]
        else:
            state_values = [bool(item) for item in prefill_state]
        scheduled_rows = len(state_values)
        output_rows = int(hidden_states.shape[0])
        if output_rows == scheduled_rows:
            return hidden_states

        completed_rows = [idx for idx, is_done in enumerate(state_values) if is_done]
        if output_rows != len(completed_rows):
            return hidden_states
        if not torch.is_floating_point(hidden_states):
            return hidden_states

        expanded = hidden_states.new_full(
            (scheduled_rows, *tuple(hidden_states.shape[1:])),
            float("-inf"),
        )
        for src_row, dst_row in enumerate(completed_rows):
            expanded[dst_row] = hidden_states[src_row]
        if _env_flag("QWEN36_VLLM_LOGITS_DEBUG"):
            print(
                "[qwen36_vllm_logits_debug] "
                f"expanded_completed_prefill_logits output_rows={output_rows} "
                f"scheduled_rows={scheduled_rows} completed_rows={completed_rows}",
                flush=True,
            )
        return expanded
    except Exception as exc:  # pragma: no cover - defensive shim only
        if _env_flag("QWEN36_VLLM_LOGITS_DEBUG"):
            print(
                "[qwen36_vllm_logits_debug] "
                f"expand_completed_prefill_logits_error={type(exc).__name__}: {exc}",
                flush=True,
            )
        return hidden_states


def _prefill_completion_state_values(prefill_completion_state: Any) -> list[bool] | None:
    if prefill_completion_state is None:
        return None
    try:
        if hasattr(prefill_completion_state, "numel"):
            if prefill_completion_state.numel() == 0:
                return None
            values = prefill_completion_state.reshape(-1)
            normalized = []
            for value in values:
                try:
                    normalized.append(bool(value.item()))
                except AttributeError:
                    normalized.append(bool(value))
            return normalized
        values = list(prefill_completion_state)
    except Exception:
        return None
    if not values:
        return None
    normalized = []
    for value in values:
        try:
            normalized.append(bool(value.item()))
        except AttributeError:
            normalized.append(bool(value))
    return normalized


def _prefill_completion_has_incomplete_row(prefill_completion_state: Any) -> bool:
    values = _prefill_completion_state_values(prefill_completion_state)
    return bool(values) and not all(values)


def _runner_vocab_size(runner: Any) -> int | None:
    owners = [
        runner,
        getattr(runner, "model", None),
        getattr(getattr(runner, "model", None), "model", None),
        getattr(getattr(getattr(runner, "model", None), "model", None), "config", None),
        getattr(runner, "model_config", None),
    ]
    for owner in owners:
        vocab_size = getattr(owner, "vocab_size", None)
        if vocab_size is not None:
            try:
                return int(vocab_size)
            except (TypeError, ValueError):
                return None
        config = getattr(owner, "config", None)
        vocab_size = getattr(config, "vocab_size", None)
        if vocab_size is not None:
            try:
                return int(vocab_size)
            except (TypeError, ValueError):
                return None
    return None


def _format_token_id(value: int) -> str:
    if value < 0:
        return str(value)
    return f"{value} (0x{value & 0xFFFFFFFF:08x})"


def _prefill_state_for_output_rows(
    values: list[bool],
    output_rows: int,
) -> list[bool]:
    if output_rows <= 0:
        return []
    if output_rows == len(values):
        return values
    completed_count = sum(1 for value in values if value)
    if completed_count > 0 and output_rows == completed_count:
        return [True] * output_rows
    return values[:output_rows]


def _validate_completed_prefill_sampled_tokens(
    sampled_token_ids: Any,
    prefill_completion_state: Any,
    *,
    vocab_size: int | None,
    stage: str,
) -> None:
    values = _prefill_completion_state_values(prefill_completion_state)
    if not values or sampled_token_ids is None or not hasattr(sampled_token_ids, "shape"):
        return
    if not hasattr(sampled_token_ids, "dtype"):
        return
    if sampled_token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "Qwen3.6 sampled token ids must be int32 or int64 before vLLM "
            f"publishes completed prefill rows; stage={stage}; "
            f"dtype={sampled_token_ids.dtype}"
        )
    shape = getattr(sampled_token_ids, "shape", ())
    if not shape:
        return
    row_values = _prefill_state_for_output_rows(values, int(shape[0]))
    row_count = min(len(row_values), int(shape[0]))
    for row_idx, is_done in enumerate(row_values[:row_count]):
        if not is_done:
            continue
        row = sampled_token_ids[row_idx].reshape(-1)
        if row.numel() == 0:
            continue
        invalid_id, reason = _sampled_token_invalid_id_and_reason(
            row,
            vocab_size=vocab_size,
        )
        if invalid_id is None:
            continue
        raise ValueError(
            "Qwen3.6 sampled token id contract violated before vLLM output "
            f"update: {reason}; stage={stage}; row={row_idx}; "
            f"token_id={_format_token_id(invalid_id)}; "
            f"prefill_completion_state={values}; "
            f"sampled_shape={tuple(sampled_token_ids.shape)}"
        )


def _sampled_token_invalid_id_and_reason(
    row: Any,
    *,
    vocab_size: int | None,
) -> tuple[int | None, str | None]:
    if row is None or not hasattr(row, "numel") or row.numel() == 0:
        return None, None
    min_id = int(row.min().item())
    max_id = int(row.max().item())
    if min_id < 0:
        return min_id, "negative"
    if vocab_size is not None and max_id >= vocab_size:
        return max_id, f"out-of-vocab for vocab_size={vocab_size}"
    return None, None


def _logits_argmax_token_ids_for_sample_shape(
    logits_source: Any,
    sampled_token_ids: Any,
    *,
    vocab_size: int | None = None,
) -> Any:
    logits_tensor = _first_tensor_like(logits_source)
    if logits_tensor is None or not hasattr(logits_tensor, "dim"):
        return None
    if not torch.is_floating_point(logits_tensor):
        return None
    if logits_tensor.dim() >= 3:
        logits_for_argmax = logits_tensor[:, -1, :]
    elif logits_tensor.dim() == 2:
        logits_for_argmax = logits_tensor
    elif logits_tensor.dim() == 1:
        logits_for_argmax = logits_tensor.reshape(1, -1)
    else:
        return None
    if vocab_size is not None and int(logits_for_argmax.shape[-1]) < int(vocab_size):
        return None

    argmax = logits_for_argmax.detach().float().argmax(dim=-1)
    target_shape = tuple(getattr(sampled_token_ids, "shape", ()))
    if len(target_shape) <= 1:
        shaped = argmax.reshape(-1)
    else:
        shaped = argmax.reshape(-1, *([1] * (len(target_shape) - 1)))
    return shaped.to(
        device=sampled_token_ids.device,
        dtype=sampled_token_ids.dtype,
    )


def _summarize_logits_for_fallback(logits_source: Any) -> str:
    logits_tensor = _first_tensor_like(logits_source)
    if logits_tensor is None or not hasattr(logits_tensor, "dim"):
        return "logits=unavailable"
    if not torch.is_floating_point(logits_tensor):
        return (
            f"logits_shape={tuple(getattr(logits_tensor, 'shape', ())) } "
            f"logits_dtype={getattr(logits_tensor, 'dtype', None)} non_float"
        )
    try:
        logits_float = logits_tensor.detach().float()
        flat = logits_float.reshape(-1)
        finite_mask = torch.isfinite(flat)
        finite_count = int(finite_mask.sum().item())
        nan_count = int(torch.isnan(flat).sum().item())
        posinf_count = int(
            torch.logical_and(torch.isinf(flat), flat > 0).sum().item()
        )
        neginf_count = int(
            torch.logical_and(torch.isinf(flat), flat < 0).sum().item()
        )
        finite_min = finite_max = None
        if finite_count:
            finite_values = flat[finite_mask]
            finite_min = float(finite_values.min().item())
            finite_max = float(finite_values.max().item())
        logits_for_argmax = (
            logits_float[:, -1, :] if logits_float.dim() >= 3 else logits_float
        )
        argmax = logits_for_argmax.argmax(dim=-1).detach().cpu().reshape(-1)
        argmax_values = (
            logits_for_argmax.gather(
                dim=-1,
                index=logits_for_argmax.argmax(dim=-1, keepdim=True),
            )
            .detach()
            .cpu()
            .reshape(-1)
        )
        return (
            f"logits_shape={tuple(logits_tensor.shape)} logits_dtype={logits_tensor.dtype} "
            f"finite={finite_count}/{int(flat.numel())} nan={nan_count} "
            f"posinf={posinf_count} neginf={neginf_count} "
            f"finite_min={finite_min} finite_max={finite_max} "
            f"argmax={argmax[:4].tolist()} "
            f"argmax_values={[float(item) for item in argmax_values[:4].tolist()]}"
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"logits_summary_error={type(exc).__name__}: {exc}"


def _mask_incomplete_prefill_sampled_tokens(
    sampler_output: Any,
    prefill_completion_state: Any,
    *,
    vocab_size: int | None = None,
    stage: str = "sample",
    logits_source: Any = None,
) -> Any:
    values = _prefill_completion_state_values(prefill_completion_state)
    if not values:
        if _env_flag("QWEN36_HYBRID_APC_DEBUG"):
            print(
                "[hybrid_apc_debug] sample-mask skip "
                f"prefill_completion_state={values}",
                flush=True,
            )
        return sampler_output

    sampled_token_ids = getattr(sampler_output, "sampled_token_ids", None)
    if sampled_token_ids is None or not hasattr(sampled_token_ids, "clone"):
        if _env_flag("QWEN36_HYBRID_APC_DEBUG"):
            print(
                "[hybrid_apc_debug] sample-mask missing-sampled-token-ids "
                f"prefill_completion_state={values} "
                f"sampler_output_type={type(sampler_output).__name__}",
                flush=True,
            )
        return sampler_output
    shape = getattr(sampled_token_ids, "shape", ())
    if not shape:
        if _env_flag("QWEN36_HYBRID_APC_DEBUG"):
            print(
                "[hybrid_apc_debug] sample-mask scalar-sampled-token-ids "
                f"prefill_completion_state={values}",
                flush=True,
            )
        return sampler_output
    row_count = min(len(values), int(shape[0]))
    if row_count <= 0:
        return sampler_output

    row_values = _prefill_state_for_output_rows(values, int(shape[0]))
    row_count = min(len(row_values), int(shape[0]))
    fallback_token_ids = None
    masked_token_ids = None
    repaired_completed_rows: list[dict[str, Any]] = []
    for row_idx, is_done in enumerate(row_values[:row_count]):
        if not is_done:
            if masked_token_ids is None:
                masked_token_ids = sampled_token_ids.clone()
            masked_token_ids[row_idx] = -1
            continue

        row = sampled_token_ids[row_idx].reshape(-1)
        invalid_id, reason = _sampled_token_invalid_id_and_reason(
            row,
            vocab_size=vocab_size,
        )
        if invalid_id is None:
            continue

        if fallback_token_ids is None:
            fallback_token_ids = _logits_argmax_token_ids_for_sample_shape(
                logits_source,
                sampled_token_ids,
                vocab_size=vocab_size,
            )
        if (
            fallback_token_ids is None
            or not hasattr(fallback_token_ids, "shape")
            or int(fallback_token_ids.shape[0]) <= row_idx
        ):
            raise ValueError(
                "Qwen3.6 completed prefill sampled token is invalid and logits "
                "are unavailable for host fallback. Compile the artifact with "
                "--output-logits-with-on-device-sampling from a build that "
                "gathers vocab-parallel output logits, or use "
                "--disable-on-device-sampling for host sampling. "
                f"{reason}; stage={stage}; row={row_idx}; "
                f"token_id={_format_token_id(invalid_id)}; "
                f"prefill_completion_state={values}; "
                f"effective_output_state={row_values}; "
                f"sampled_shape={tuple(sampled_token_ids.shape)}"
            )

        if masked_token_ids is None:
            masked_token_ids = sampled_token_ids.clone()
        masked_token_ids[row_idx] = fallback_token_ids[row_idx]
        repaired_completed_rows.append(
            {
                "row": row_idx,
                "reason": reason,
                "token_id": invalid_id,
                "fallback": int(fallback_token_ids[row_idx].reshape(-1)[0].item()),
                "logits_summary": _summarize_logits_for_fallback(logits_source),
            }
        )

    if masked_token_ids is None:
        _validate_completed_prefill_sampled_tokens(
            sampled_token_ids,
            values,
            vocab_size=vocab_size,
            stage=stage,
        )
        if _env_flag("QWEN36_HYBRID_APC_DEBUG"):
            print(
                "[hybrid_apc_debug] sample-mask skip "
                f"prefill_completion_state={values}",
                flush=True,
            )
        return sampler_output

    _validate_completed_prefill_sampled_tokens(
        masked_token_ids,
        values,
        vocab_size=vocab_size,
        stage=stage,
    )
    try:
        sampler_output.sampled_token_ids = masked_token_ids
    except Exception:
        return sampler_output
    for row in repaired_completed_rows:
        logger.warning(
            "Replacing invalid completed-prefill sampled token with logits "
            "argmax before vLLM output update: stage=%s row=%s %s "
            "token_id=%s fallback_token_id=%s prefill_completion_state=%s",
            stage,
            row["row"],
            row["reason"],
            _format_token_id(int(row["token_id"])),
            row["fallback"],
            values,
        )
        logger.warning(
            "Qwen3.6 fallback logits summary: stage=%s row=%s %s",
            stage,
            row["row"],
            row["logits_summary"],
        )
    if _env_flag("QWEN36_HYBRID_APC_DEBUG"):
        try:
            before = sampled_token_ids.detach().cpu().reshape(-1).tolist()
            after = masked_token_ids.detach().cpu().reshape(-1).tolist()
        except Exception:
            before = "unavailable"
            after = "unavailable"
        print(
            "[hybrid_apc_debug] sample-mask applied "
            f"prefill_completion_state={values} before={before} after={after}",
            flush=True,
        )
    return sampler_output


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(item) for item in shape]


def _flatten_int_sample(value: Any, *, limit: int = 8) -> list[int] | None:
    detach = getattr(value, "detach", None)
    if detach is None:
        return None
    try:
        tensor = detach().cpu().reshape(-1)
        return [int(item) for item in tensor[:limit].tolist()]
    except Exception:
        return None


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "shape")


def _first_tensor_like(value: Any) -> Any:
    if _is_tensor_like(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor_like(item)
            if found is not None:
                return found
    return None


def _describe_sample_logits_value(value: Any, *, depth: int = 0) -> dict[str, Any]:
    row: dict[str, Any] = {"type": type(value).__name__}
    shape = _shape_of(value)
    if shape is not None:
        row["shape"] = shape
    if isinstance(value, (list, tuple)):
        row["len"] = len(value)
        if depth < 3:
            row["items"] = [
                _describe_sample_logits_value(item, depth=depth + 1)
                for item in value[:4]
            ]
    return row


def _split_sample_logits_output(value: Any) -> tuple[Any, Any, str]:
    """Return token IDs and logits from sample+logits debug model outputs."""

    tokens = getattr(value, "tokens", None)
    logits = getattr(value, "logits", None)
    if tokens is not None or logits is not None:
        return tokens, logits, type(value).__name__

    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            nested_tokens, nested_logits, nested_kind = _split_sample_logits_output(
                value[0]
            )
            return nested_tokens, nested_logits, f"{type(value).__name__}[{nested_kind}]"
        if len(value) >= 2:
            return value[0], value[1], type(value).__name__

    return value, None, type(value).__name__


def _json_float_value(value: float) -> float | str:
    if value != value:
        return "nan"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return float(value)


def _log_sample_logits_comparison(
    hidden_states: Any,
    model_input: Any,
    sampler_output: Any,
) -> None:
    """Debug-only compare traced sampled tokens with returned logits argmax."""

    path = os.environ.get("QWEN36_SAMPLE_LOGITS_COMPARE_JSONL")
    if not path:
        return
    try:
        tokens, logits, hidden_state_kind = _split_sample_logits_output(hidden_states)
        token_tensor = _first_tensor_like(tokens)
        logits_tensor = _first_tensor_like(logits)
        row: dict[str, Any] = {
            "request_ids": list(getattr(model_input, "request_ids", ()) or ()),
            "hidden_state_type": hidden_state_kind,
            "hidden_state_structure": _describe_sample_logits_value(hidden_states),
            "tokens_shape": _shape_of(token_tensor),
            "logits_shape": _shape_of(logits_tensor),
            "sampler_output_type": type(sampler_output).__name__,
        }
        if token_tensor is not None and logits_tensor is not None:
            row["sampled_dtype"] = str(token_tensor.dtype)
            row["logits_dtype"] = str(logits_tensor.dtype)
            logits_tensor = logits_tensor.detach().float()
            flat_logits = logits_tensor.reshape(-1)
            finite_mask = torch.isfinite(flat_logits)
            finite_count = int(finite_mask.sum().item())
            row.update(
                {
                    "logits_numel": int(flat_logits.numel()),
                    "logits_finite": finite_count,
                    "logits_nan": int(torch.isnan(flat_logits).sum().item()),
                    "logits_posinf": int(
                        torch.logical_and(torch.isinf(flat_logits), flat_logits > 0)
                        .sum()
                        .item()
                    ),
                    "logits_neginf": int(
                        torch.logical_and(torch.isinf(flat_logits), flat_logits < 0)
                        .sum()
                        .item()
                    ),
                }
            )
            if finite_count:
                finite_flat = flat_logits[finite_mask]
                row["logits_finite_min"] = float(finite_flat.min().item())
                row["logits_finite_max"] = float(finite_flat.max().item())
            else:
                row["logits_finite_min"] = None
                row["logits_finite_max"] = None
            logits_for_argmax = (
                logits_tensor[:, -1, :]
                if logits_tensor.dim() >= 3
                else logits_tensor
            )
            argmax_tokens = logits_for_argmax.argmax(dim=-1).detach().cpu().reshape(-1)
            argmax_values = (
                logits_for_argmax.gather(
                    dim=-1,
                    index=logits_for_argmax.argmax(dim=-1, keepdim=True),
                )
                .detach()
                .cpu()
                .reshape(-1)
            )
            sampled_tokens = token_tensor.detach().cpu().reshape(-1)
            count = min(int(argmax_tokens.numel()), int(sampled_tokens.numel()))
            row.update(
                {
                    "sampled_tokens": [
                        int(item) for item in sampled_tokens[: min(count, 8)].tolist()
                    ],
                    "logits_argmax_tokens": [
                        int(item) for item in argmax_tokens[: min(count, 8)].tolist()
                    ],
                    "logits_argmax_values": [
                        _json_float_value(float(item))
                        for item in argmax_values[: min(count, 8)].tolist()
                    ],
                    "num_compared": count,
                    "num_matches": int(
                        (sampled_tokens[:count] == argmax_tokens[:count]).sum().item()
                    )
                    if count
                    else 0,
                }
            )
        else:
            row["sampled_tokens"] = _flatten_int_sample(token_tensor)

        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    except Exception as exc:
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                            "hidden_state_type": type(hidden_states).__name__,
                            "stage": "sample_logits_compare",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        except Exception:
            return


def _log_sample_logits_split_error(
    hidden_states: Any,
    model_input: Any,
    exc: BaseException,
) -> None:
    path = os.environ.get("QWEN36_SAMPLE_LOGITS_COMPARE_JSONL")
    if not path:
        return
    try:
        tokens, logits, hidden_state_kind = _split_sample_logits_output(hidden_states)
        token_tensor = _first_tensor_like(tokens)
        logits_tensor = _first_tensor_like(logits)
        row = {
            "error": f"{type(exc).__name__}: {exc}",
            "hidden_state_type": hidden_state_kind,
            "hidden_state_structure": _describe_sample_logits_value(hidden_states),
            "logits_shape": _shape_of(logits_tensor),
            "request_ids": list(getattr(model_input, "request_ids", ()) or ()),
            "stage": "sample_on_device",
            "tokens_shape": _shape_of(token_tensor),
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    except Exception:
        return


def patch_neuron_model_runner_class(runner_cls: type) -> bool:
    """Patch vLLM-Neuron runner to expose scheduler row metadata."""

    original_execute = getattr(runner_cls, "_execute_model_for_text", None)
    if original_execute is None:
        raise AttributeError(
            f"{runner_cls!r} has no _execute_model_for_text method"
        )
    original_prepare = getattr(runner_cls, "_prepare_model_input", None)
    original_prepare_logits = getattr(
        runner_cls,
        "_prepare_logits_for_sampling",
        None,
    )
    original_sample_on_device = getattr(
        runner_cls,
        "_sample_on_device",
        None,
    )
    original_get_kv_cache_spec = getattr(
        runner_cls,
        "get_kv_cache_spec",
        None,
    )
    original_sample_tokens = getattr(runner_cls, "sample_tokens", None)
    original_generate_output = getattr(
        runner_cls,
        "_generate_model_runner_output",
        None,
    )

    missing = object()
    installed = False

    if original_prepare is not None and not getattr(
        original_prepare,
        "_qwen36_hybrid_apc_model_input_patched",
        False,
    ):

        def prepare_model_input_with_hybrid_apc_metadata(
            self,
            scheduler_output,
            *args,
            **kwargs,
        ):
            model_input = original_prepare(self, scheduler_output, *args, **kwargs)
            object.__setattr__(
                model_input,
                "_qwen36_cached_request_ids",
                _request_ids_from_scheduler_output(
                    scheduler_output,
                    kind="cached",
                ),
            )
            object.__setattr__(
                model_input,
                "_qwen36_new_request_ids",
                _request_ids_from_scheduler_output(
                    scheduler_output,
                    kind="new",
                ),
            )
            metadata_by_request_id = getattr(
                scheduler_output,
                _SCHEDULER_OUTPUT_METADATA_ATTR,
                None,
            )
            if metadata_by_request_id is not None:
                object.__setattr__(
                    model_input,
                    _SCHEDULER_OUTPUT_METADATA_ATTR,
                    metadata_by_request_id,
                )
            request_records = _hybrid_apc_request_records_from_model_input(
                model_input,
                scheduler_output,
            )
            if request_records is not None:
                object.__setattr__(
                    model_input,
                    _SCHEDULER_OUTPUT_REQUEST_RECORDS_ATTR,
                    request_records,
                )
            return model_input

        prepare_model_input_with_hybrid_apc_metadata._qwen36_hybrid_apc_model_input_patched = (
            True
        )
        prepare_model_input_with_hybrid_apc_metadata._qwen36_original_prepare_model_input = (
            original_prepare
        )
        runner_cls._prepare_model_input = prepare_model_input_with_hybrid_apc_metadata
        installed = True

    if original_prepare_logits is not None and not getattr(
        original_prepare_logits,
        "_qwen36_vllm_logits_debug_patched",
        False,
    ):

        def prepare_logits_for_sampling_with_debug(
            self,
            hidden_states,
            model_input,
            *args,
            **kwargs,
        ):
            if _env_flag("QWEN36_VLLM_LOGITS_DEBUG"):
                _debug_logits_tensor("runner_hidden_states_before_prepare", hidden_states)
                prefill_state = getattr(model_input, "prefill_completion_state", None)
                request_ids = getattr(model_input, "request_ids", None)
                print(
                    "[qwen36_vllm_logits_debug] "
                    f"request_ids={request_ids} prefill_completion_state={prefill_state}",
                    flush=True,
                )
            hidden_states = _expand_completed_prefill_logits(hidden_states, model_input)
            _debug_logits_tensor(
                "runner_hidden_states_after_prefill_expand",
                hidden_states,
            )
            logits = original_prepare_logits(
                self,
                hidden_states,
                model_input,
                *args,
                **kwargs,
            )
            _debug_logits_tensor("runner_logits_after_prepare", logits)
            return logits

        prepare_logits_for_sampling_with_debug._qwen36_vllm_logits_debug_patched = (
            True
        )
        prepare_logits_for_sampling_with_debug._qwen36_original_prepare_logits = (
            original_prepare_logits
        )
        runner_cls._prepare_logits_for_sampling = prepare_logits_for_sampling_with_debug
        installed = True

    if original_sample_on_device is not None and not getattr(
        original_sample_on_device,
        "_qwen36_clone_incomplete_prefill_tokens_patched",
        False,
    ):

        def sample_on_device_with_incomplete_prefill_clone(
            self,
            hidden_states,
            model_input,
            *args,
            **kwargs,
        ):
            prefill_state = getattr(model_input, "prefill_completion_state", None)
            if _prefill_completion_has_incomplete_row(prefill_state):
                clone = getattr(hidden_states, "clone", None)
                if clone is not None:
                    hidden_states = clone()
            hidden_states_for_sampling, logits_for_fallback, _ = _split_sample_logits_output(
                hidden_states
            )
            token_tensor_for_sampling = _first_tensor_like(hidden_states_for_sampling)
            if token_tensor_for_sampling is not None:
                clone = getattr(token_tensor_for_sampling, "clone", None)
                if clone is not None:
                    token_tensor_for_sampling = clone()
                hidden_states_for_sampling = token_tensor_for_sampling
            try:
                sampler_output = original_sample_on_device(
                    self,
                    hidden_states_for_sampling,
                    model_input,
                    *args,
                    **kwargs,
                )
            except Exception as exc:
                _log_sample_logits_split_error(hidden_states, model_input, exc)
                raise
            sampler_output = _mask_incomplete_prefill_sampled_tokens(
                sampler_output,
                prefill_state,
                vocab_size=_runner_vocab_size(self),
                stage="sample_on_device",
                logits_source=logits_for_fallback,
            )
            _log_sample_logits_comparison(hidden_states, model_input, sampler_output)
            return sampler_output

        sample_on_device_with_incomplete_prefill_clone._qwen36_clone_incomplete_prefill_tokens_patched = (
            True
        )
        sample_on_device_with_incomplete_prefill_clone._qwen36_original_sample_on_device = (
            original_sample_on_device
        )
        runner_cls._sample_on_device = sample_on_device_with_incomplete_prefill_clone
        installed = True

    if original_get_kv_cache_spec is not None and not getattr(
        original_get_kv_cache_spec,
        "_qwen36_hybrid_kv_cache_spec_patched",
        False,
    ):

        def get_kv_cache_spec_with_qwen36_hybrid_layers(self):
            model_config = getattr(self, "model_config", None)
            hf_config = getattr(model_config, "hf_config", None)
            num_layers = _num_layers_from_hf_config(
                hf_config,
                original_get_kv_cache_spec,
            )
            if num_layers is None:
                return original_get_kv_cache_spec(self)
            attention_layer_indices = _hybrid_kv_attention_layer_indices(
                hf_config,
                num_layers,
            )
            if attention_layer_indices is None:
                return original_get_kv_cache_spec(self)
            full_attention_spec_cls = _full_attention_spec_class(
                original_get_kv_cache_spec
            )
            if full_attention_spec_cls is None:
                return original_get_kv_cache_spec(self)

            parallel_config = getattr(self, "parallel_config", None)
            model = getattr(self, "model", None)
            get_sliding_window = getattr(model_config, "get_sliding_window", None)
            sliding_window = (
                get_sliding_window() if callable(get_sliding_window) else None
            )
            local_kv_heads = _local_num_kv_heads(hf_config, parallel_config)
            kv_cache_spec = {}
            for layer_idx in attention_layer_indices:
                layer_name = f"layers.{layer_idx}.self_attn"
                kv_cache_spec[layer_name] = full_attention_spec_cls(
                    block_size=getattr(self, "block_size"),
                    num_kv_heads=local_kv_heads,
                    head_size=getattr(model, "head_dim"),
                    dtype=getattr(model_config, "dtype"),
                    sliding_window=sliding_window,
                )
            logger.info(
                "Using Qwen hybrid KV-cache spec for %d/%d attention layers "
                "with %d local KV heads",
                len(attention_layer_indices),
                num_layers,
                local_kv_heads,
            )
            return kv_cache_spec

        get_kv_cache_spec_with_qwen36_hybrid_layers._qwen36_hybrid_kv_cache_spec_patched = (
            True
        )
        get_kv_cache_spec_with_qwen36_hybrid_layers._qwen36_original_get_kv_cache_spec = (
            original_get_kv_cache_spec
        )
        runner_cls.get_kv_cache_spec = get_kv_cache_spec_with_qwen36_hybrid_layers
        installed = True

    if original_generate_output is not None and not getattr(
        original_generate_output,
        "_qwen36_mask_incomplete_prefill_output_patched",
        False,
    ):

        def generate_model_runner_output_with_prefill_mask(
            self,
            sampler_outputs,
            *args,
            **kwargs,
        ):
            prefill_state = getattr(
                self,
                _RUNNER_PREFILL_STATE_FOR_OUTPUT_ATTR,
                None,
            )
            if prefill_state is not None:
                sampler_outputs = _mask_incomplete_prefill_sampled_tokens(
                    sampler_outputs,
                    prefill_state,
                    vocab_size=_runner_vocab_size(self),
                    stage="generate_model_runner_output",
                )
            return original_generate_output(self, sampler_outputs, *args, **kwargs)

        generate_model_runner_output_with_prefill_mask._qwen36_mask_incomplete_prefill_output_patched = (
            True
        )
        generate_model_runner_output_with_prefill_mask._qwen36_original_generate_model_runner_output = (
            original_generate_output
        )
        runner_cls._generate_model_runner_output = (
            generate_model_runner_output_with_prefill_mask
        )
        installed = True

    if original_sample_tokens is not None and not getattr(
        original_sample_tokens,
        "_qwen36_capture_prefill_state_for_output_patched",
        False,
    ):

        def sample_tokens_with_prefill_state_for_output(self, *args, **kwargs):
            model_input = getattr(self, "_cached_model_input", None)
            if getattr(self, "_cached_logits", None) is None:
                return None
            prefill_state = getattr(model_input, "prefill_completion_state", None)
            previous_value = getattr(
                self,
                _RUNNER_PREFILL_STATE_FOR_OUTPUT_ATTR,
                missing,
            )
            setattr(self, _RUNNER_PREFILL_STATE_FOR_OUTPUT_ATTR, prefill_state)
            try:
                return original_sample_tokens(self, *args, **kwargs)
            finally:
                if previous_value is missing:
                    try:
                        delattr(self, _RUNNER_PREFILL_STATE_FOR_OUTPUT_ATTR)
                    except AttributeError:
                        pass
                else:
                    setattr(
                        self,
                        _RUNNER_PREFILL_STATE_FOR_OUTPUT_ATTR,
                        previous_value,
                    )

        sample_tokens_with_prefill_state_for_output._qwen36_capture_prefill_state_for_output_patched = (
            True
        )
        sample_tokens_with_prefill_state_for_output._qwen36_original_sample_tokens = (
            original_sample_tokens
        )
        runner_cls.sample_tokens = sample_tokens_with_prefill_state_for_output
        installed = True

    if getattr(original_execute, "_qwen36_hybrid_apc_request_ids_patched", False):
        return installed

    def execute_model_for_text_with_request_ids(self, model_input, *args, **kwargs):
        model = getattr(self, "model", None)
        runtime_config = _runner_hybrid_apc_runtime_config(self)
        request_ids = _request_ids_from_model_input(model_input)
        if request_ids is None:
            request_ids = tuple(
                getattr(model_input, "_qwen36_cached_request_ids", ()) or ()
            ) + tuple(getattr(model_input, "_qwen36_new_request_ids", ()) or ())
            if not request_ids:
                request_ids = None
        metadata = {
            "_qwen36_vllm_request_ids": request_ids,
            "_qwen36_vllm_cached_request_ids": getattr(
                model_input,
                "_qwen36_cached_request_ids",
                None,
            ),
            "_qwen36_vllm_new_request_ids": getattr(
                model_input,
                "_qwen36_new_request_ids",
                None,
            ),
            "_qwen36_vllm_prefill_completion_state": getattr(
                model_input,
                "prefill_completion_state",
                None,
            ),
            "_qwen36_vllm_hybrid_apc_metadata_by_request_id": getattr(
                model_input,
                _SCHEDULER_OUTPUT_METADATA_ATTR,
                None,
            ),
            "_qwen36_vllm_hybrid_apc_request_records": getattr(
                model_input,
                _SCHEDULER_OUTPUT_REQUEST_RECORDS_ATTR,
                None,
            ),
        }
        previous_values = []
        _apply_hybrid_apc_runtime_config(
            model,
            runtime_config,
            previous_values=previous_values,
            missing=missing,
        )
        if any(value is not None for value in metadata.values()):
            for target in _request_id_target_models(model):
                for attr, value in metadata.items():
                    if value is None:
                        continue
                    previous_values.append(
                        (
                            target,
                            attr,
                            getattr(target, attr, missing),
                        )
                    )
                    setattr(target, attr, value)
        try:
            return original_execute(self, model_input, *args, **kwargs)
        finally:
            for target, attr, previous_value in reversed(previous_values):
                if previous_value is missing:
                    try:
                        delattr(target, attr)
                    except AttributeError:
                        pass
                else:
                    setattr(target, attr, previous_value)

    execute_model_for_text_with_request_ids._qwen36_hybrid_apc_request_ids_patched = (
        True
    )
    execute_model_for_text_with_request_ids._qwen36_original_execute_model_for_text = (
        original_execute
    )
    runner_cls._execute_model_for_text = execute_model_for_text_with_request_ids
    return True


def _patch_neuron_runner_module(module: Any) -> bool:
    runner_cls = getattr(module, "NeuronxDistributedModelRunner", None)
    if runner_cls is None:
        return False
    installed = patch_neuron_model_runner_class(runner_cls)
    if installed:
        logger.info("Installed Qwen Hybrid APC vLLM-Neuron runner patch")
    return installed


def _restore_nested_output(restore, value: Any) -> Any:
    if hasattr(value, "shape"):
        return restore(value)
    if isinstance(value, list):
        return [_restore_nested_output(restore, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_nested_output(restore, item) for item in value)
    return value


def _patch_neuron_loader_module(module: Any) -> bool:
    causal_lm_cls = getattr(module, "NeuronCausalLM", None)
    if causal_lm_cls is None:
        return False
    original_forward = getattr(causal_lm_cls, "forward", None)
    if original_forward is None or getattr(
        original_forward,
        "_qwen36_sample_logits_tokens_patched",
        False,
    ):
        return False

    def forward_with_sample_logits_tokens(
        self,
        input_ids,
        input_block_ids,
        **kwargs,
    ):
        import time as _time  # noqa: WPS433

        forward_start = _time.perf_counter()
        batch_size = (
            input_ids.shape[0]
            if hasattr(input_ids, "shape")
            else len(input_ids)
        )

        with self._reordered(input_block_ids, input_ids=input_ids, **kwargs) as (
            sorted_ids,
            inputs,
            restore,
        ):
            model_start = _time.perf_counter()
            output = self.model(
                inputs["input_ids"],
                attention_mask=None,
                seq_ids=sorted_ids,
                block_table=inputs["block_tables"],
                **{
                    key: value
                    for key, value in inputs.items()
                    if key
                    not in ["input_ids", "block_tables", "prefill_completion_state"]
                },
            )
            model_elapsed = (_time.perf_counter() - model_start) * 1000
            module.logger.debug("[PERF]     model_execution: %.2fms", model_elapsed)

            output_proc_start = _time.perf_counter()
            if self.model.config.neuron_config.on_device_sampling_config:
                tokens = getattr(output, "tokens", None)
                logits = getattr(output, "logits", None)
                if tokens is not None and logits is not None:
                    output = [tokens, logits]
                else:
                    output = output.hidden_states
                if getattr(
                    self.model.config.neuron_config,
                    "enable_fused_speculation",
                    False,
                ):
                    fused = output
                    output = self._remask_fused_spec_output(fused, inputs)
            else:
                if self.neuron_config.is_chunked_prefill:
                    assert kwargs.get("prefill_completion_state") is not None
                    idx_for_sampling = (
                        kwargs["prefill_completion_state"].nonzero().flatten()
                    )
                    output = output.logits[0, idx_for_sampling, :]
                else:
                    output = output.logits[:, -1, :]
            output_proc_elapsed = (_time.perf_counter() - output_proc_start) * 1000
            module.logger.debug(
                "[PERF]     output_processing: %.2fms",
                output_proc_elapsed,
            )

            restore_start = _time.perf_counter()
            result = _restore_nested_output(restore, output)
            restore_elapsed = (_time.perf_counter() - restore_start) * 1000
            module.logger.debug("[PERF]     restore: %.2fms", restore_elapsed)

            forward_elapsed = (_time.perf_counter() - forward_start) * 1000
            module.logger.debug(
                "[PERF]   forward() total: %.2fms [batch=%d]",
                forward_elapsed,
                batch_size,
            )
            return result

    forward_with_sample_logits_tokens._qwen36_sample_logits_tokens_patched = True
    forward_with_sample_logits_tokens._qwen36_original_forward = original_forward
    causal_lm_cls.forward = forward_with_sample_logits_tokens
    logger.info("Installed Qwen sample+logits vLLM-Neuron loader patch")
    return True


def _patch_module(module_name: str, module: Any) -> bool:
    if module_name == _SCHEDULER_MODULE:
        return _patch_scheduler_module(module)
    if module_name == _KV_CACHE_MANAGER_MODULE:
        return _patch_kv_cache_manager_module(module)
    if module_name == _VLLM_NEURON_RUNNER_MODULE:
        return _patch_neuron_runner_module(module)
    if module_name == _VLLM_NEURON_LOADER_MODULE:
        return _patch_neuron_loader_module(module)
    return False


class _HybridAPCSchedulerPatchLoader(importlib.abc.Loader):
    _qwen36_hybrid_apc_loader = True

    def __init__(self, wrapped_loader: importlib.abc.Loader):
        self.wrapped_loader = wrapped_loader

    def create_module(self, spec):
        create_module = getattr(self.wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module):
        self.wrapped_loader.exec_module(module)
        _patch_module(module.__name__, module)


class _HybridAPCSchedulerPatchFinder(importlib.abc.MetaPathFinder):
    _qwen36_hybrid_apc_import_hook = True

    def find_spec(self, fullname, path, target=None):
        if fullname not in _PATCHED_MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        if getattr(spec.loader, "_qwen36_hybrid_apc_loader", False):
            return spec
        spec.loader = _HybridAPCSchedulerPatchLoader(spec.loader)
        return spec


def install_import_hook() -> bool:
    """Patch vLLM components lazily, without importing vLLM at Python startup."""

    installed = False
    for module_name in _PATCHED_MODULES:
        module = sys.modules.get(module_name)
        if module is not None:
            installed = _patch_module(module_name, module) or installed
    for finder in sys.meta_path:
        if getattr(finder, "_qwen36_hybrid_apc_import_hook", False):
            return installed
    sys.meta_path.insert(0, _HybridAPCSchedulerPatchFinder())
    return installed


def install() -> bool:
    """Install the vLLM scheduler patch when vLLM is available."""

    from vllm.v1.core.sched.scheduler import Scheduler  # noqa: WPS433

    installed = False
    module = sys.modules.get(_SCHEDULER_MODULE)
    if module is not None:
        installed = _patch_scheduler_module(module)
    else:
        installed = patch_scheduler_class(Scheduler)
    kv_cache_manager_module = sys.modules.get(_KV_CACHE_MANAGER_MODULE)
    if kv_cache_manager_module is not None:
        installed = _patch_kv_cache_manager_module(kv_cache_manager_module) or installed
    runner_module = sys.modules.get(_VLLM_NEURON_RUNNER_MODULE)
    if runner_module is not None:
        installed = _patch_neuron_runner_module(runner_module) or installed
    loader_module = sys.modules.get(_VLLM_NEURON_LOADER_MODULE)
    if loader_module is not None:
        installed = _patch_neuron_loader_module(loader_module) or installed
    if installed:
        logger.info("Installed Qwen Hybrid APC scheduler fallback patch")
    return installed
