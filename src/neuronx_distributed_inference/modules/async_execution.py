import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import torch

if TYPE_CHECKING:
    from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM
    from neuronx_distributed_inference.models.model_wrapper import ModelWrapper


def _is_hybrid_apc_enabled(neuron_base_instance: "NeuronBaseForCausalLM") -> bool:
    for owner in (
        neuron_base_instance,
        getattr(neuron_base_instance, "config", None),
        getattr(neuron_base_instance, "neuron_config", None),
        getattr(getattr(neuron_base_instance, "config", None), "neuron_config", None),
    ):
        if bool(getattr(owner, "use_hybrid_apc_manager", False)):
            return True
    return False


def _async_request_ids_signature(neuron_base_instance: "NeuronBaseForCausalLM"):
    request_ids = getattr(neuron_base_instance, "_qwen36_vllm_request_ids", None)
    if request_ids is None:
        return None
    if isinstance(request_ids, torch.Tensor):
        return tuple(request_ids.detach().cpu().reshape(-1).tolist())
    if isinstance(request_ids, (str, bytes)):
        return (request_ids,)
    try:
        return tuple(request_ids)
    except TypeError:
        return (request_ids,)


def _batch_vector(
    input_dict: Dict[str, Any],
    key: str,
    *,
    batch_size: int,
    default: int = 0,
) -> torch.Tensor:
    value = input_dict.get(key)
    if value is None:
        return torch.full((batch_size,), default, dtype=torch.int32)
    value = value.reshape(-1).to(torch.int32)
    if value.shape[0] == batch_size:
        return value
    if value.shape[0] > batch_size:
        return value[:batch_size]
    pad = torch.full((batch_size - value.shape[0],), default, dtype=value.dtype)
    return torch.cat([value, pad], dim=0)


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _to_python_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    return int(value)


def _single_batch_value(value: Any):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        flat = value.reshape(-1)
        if flat.numel() != 1:
            return None
        return flat[0]
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        return value[0]
    return value


def _multi_batch_int_values(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        flat = value.reshape(-1)
        if flat.numel() <= 1:
            return None
        return [int(item.item()) for item in flat]
    if isinstance(value, (list, tuple)):
        if len(value) <= 1:
            return None
        try:
            return [int(item) for item in value]
        except (TypeError, ValueError):
            return None
    return None


def _single_batch_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == 1


def _truthy_single_value(value: Any) -> bool:
    item = _single_batch_value(value)
    if item is None:
        return False
    if isinstance(item, torch.Tensor):
        return bool(item.item())
    return bool(item)


def _request_id_matches(candidate: Any, request_id: Any) -> bool:
    if candidate == request_id:
        return True
    try:
        return str(candidate) == str(request_id)
    except Exception:
        return False


def _request_id_in_collection(request_id: Any, values: Any) -> bool:
    if request_id is None or values is None:
        return False
    if isinstance(values, torch.Tensor):
        values = values.reshape(-1).tolist()
    elif isinstance(values, (str, bytes)):
        values = (values,)
    try:
        iterator = iter(values)
    except TypeError:
        return _request_id_matches(values, request_id)
    return any(_request_id_matches(value, request_id) for value in iterator)


def _as_hybrid_apc_request_id_tuple(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return tuple(value.detach().cpu().reshape(-1).tolist())
    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _lookup_hybrid_apc_metadata_for_request(metadata_by_request_id: Any, request_id: Any):
    if not isinstance(metadata_by_request_id, dict):
        return None
    for key in (request_id, str(request_id)):
        metadata = metadata_by_request_id.get(key)
        if isinstance(metadata, dict):
            return metadata
    for key, metadata in metadata_by_request_id.items():
        if _request_id_matches(key, request_id) and isinstance(metadata, dict):
            return metadata
    return None


def _with_hybrid_apc_owner_metadata(
    input_dict: Dict[str, Any],
    owner: Any,
) -> Dict[str, Any]:
    output = input_dict
    request_records = getattr(owner, "_qwen36_vllm_hybrid_apc_request_records", None)
    if request_records is not None and "hybrid_request_records" not in output:
        output = dict(output)
        output["hybrid_request_records"] = request_records

    request_ids = _as_hybrid_apc_request_id_tuple(
        getattr(owner, "_qwen36_vllm_request_ids", None)
    )
    metadata_by_request_id = getattr(
        owner,
        "_qwen36_vllm_hybrid_apc_metadata_by_request_id",
        None,
    )
    if (
        request_records is None
        and request_ids
        and isinstance(metadata_by_request_id, dict)
        and "hybrid_request_records" not in output
    ):
        records = []
        for request_id in request_ids:
            metadata = _lookup_hybrid_apc_metadata_for_request(
                metadata_by_request_id,
                request_id,
            )
            if metadata is None:
                continue
            record = {"request_id": request_id}
            record.update(metadata)
            records.append(record)
        if records:
            output = dict(output)
            output["hybrid_request_records"] = tuple(records)

    if request_ids and "hybrid_request_id" not in output:
        output = dict(output)
        output["hybrid_request_id"] = request_ids[0] if len(request_ids) == 1 else request_ids

    for attr, key in (
        ("_qwen36_vllm_cached_request_ids", "hybrid_cached_request_ids"),
        ("_qwen36_vllm_prefill_completion_state", "hybrid_prefill_completion_state"),
    ):
        value = getattr(owner, attr, None)
        if value is not None and key not in output:
            output = dict(output)
            output[key] = value
    return output


def _with_hybrid_apc_candidate_owner_metadata(
    input_dict: Dict[str, Any],
    *owners: Any,
) -> Dict[str, Any]:
    output = input_dict
    seen: set[int] = set()
    for owner in owners:
        if owner is None:
            continue
        owner_id = id(owner)
        if owner_id in seen:
            continue
        seen.add(owner_id)
        output = _with_hybrid_apc_owner_metadata(output, owner)
    return output


def _batch_size_from_input_dict(input_dict: Dict[str, Any]) -> int:
    batch_size = 1
    for key in (
        "input_ids",
        "seq_ids",
        "computed_context_lens",
        "full_context_lens",
        "vllm_attention_hit_len",
        "hybrid_attention_hit_len",
        "attention_hit_len",
        "hybrid_prefill_completion_state",
        "hybrid_active_suffix_len",
        "active_suffix_len",
        "hybrid_request_records",
    ):
        value = input_dict.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            batch_size = max(batch_size, int(value.reshape(value.shape[0], -1).shape[0]))
        elif isinstance(value, (list, tuple)):
            batch_size = max(batch_size, len(value))
    return batch_size


def _batch_int_list(
    input_dict: Dict[str, Any],
    key: str,
    *,
    batch_size: int,
) -> list[int] | None:
    value = input_dict.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        flat = value.reshape(-1)
        if flat.numel() < batch_size:
            return None
        return [int(item.item()) for item in flat[:batch_size]]
    if isinstance(value, (list, tuple)) and len(value) >= batch_size:
        try:
            return [int(item) for item in value[:batch_size]]
        except (TypeError, ValueError):
            return None
    return None


def _vectorized_query_lengths(
    input_dict: Dict[str, Any],
    *,
    batch_size: int,
) -> list[int] | None:
    num_queries = _batch_int_list(input_dict, "num_queries", batch_size=batch_size)
    if num_queries is not None:
        return num_queries

    active_suffix_len = _batch_int_list(
        input_dict,
        "hybrid_active_suffix_len",
        batch_size=batch_size,
    )
    if active_suffix_len is None:
        active_suffix_len = _batch_int_list(
            input_dict,
            "active_suffix_len",
            batch_size=batch_size,
        )
    if active_suffix_len is not None:
        return [max(0, int(query_len)) for query_len in active_suffix_len]

    records = _hybrid_apc_request_records(input_dict, batch_size=batch_size)
    record_active_suffix_len = _hybrid_apc_record_values(records, "active_suffix_len")
    if (
        isinstance(record_active_suffix_len, (list, tuple))
        and len(record_active_suffix_len) >= batch_size
    ):
        try:
            return [
                max(0, int(query_len))
                for query_len in record_active_suffix_len[:batch_size]
            ]
        except (TypeError, ValueError):
            pass
    if batch_size == 1 and record_active_suffix_len is not None:
        try:
            return [max(0, int(record_active_suffix_len))]
        except (TypeError, ValueError):
            pass

    full_context_lens = _batch_int_list(
        input_dict,
        "full_context_lens",
        batch_size=batch_size,
    )
    computed_context_lens = _batch_int_list(
        input_dict,
        "computed_context_lens",
        batch_size=batch_size,
    )
    if full_context_lens is not None and computed_context_lens is not None:
        return [
            max(0, full_len - computed_len)
            for full_len, computed_len in zip(full_context_lens, computed_context_lens)
        ]

    input_ids = input_dict.get("input_ids")
    if (
        isinstance(input_ids, torch.Tensor)
        and input_ids.ndim >= 2
        and input_ids.shape[0] == 1
        and input_ids.shape[1] % batch_size == 0
    ):
        return [input_ids.shape[1] // batch_size] * batch_size
    return None


def _select_batch_item(
    value: Any,
    index: int,
    batch_size: int,
    *,
    key: str = "",
    query_lengths: list[int] | None = None,
):
    if isinstance(value, torch.Tensor):
        if value.numel() == 0 or value.ndim == 0:
            return value
        if (
            key in {"rotary_position_id", "rotary_position_ids"}
            and value.ndim >= 2
            and value.shape[1] == batch_size
        ):
            return value[:, index : index + 1, ...]
        if value.shape[0] == batch_size:
            return value[index : index + 1]
        if (
            value.ndim >= 2
            and value.shape[0] == 1
            and query_lengths is not None
            and key
            in {
                "input_ids",
                "attention_mask",
                "position_ids",
                "slot_mapping",
                "inputs_embeds",
            }
        ):
            offset = sum(query_lengths[:index])
            length = query_lengths[index]
            if offset + length <= value.shape[1]:
                return value[:, offset : offset + length, ...]
        if (
            key in {"seq_ids", "adapter_ids"}
            and value.ndim == 1
            and value.shape[0] == 1
            and batch_size > 1
        ):
            fill_value = index if key == "seq_ids" else int(value.reshape(-1)[0].item())
            return torch.tensor([fill_value], dtype=value.dtype, device=value.device)
        return value
    if key == "llava_args" and isinstance(value, (list, tuple)):
        return [
            _select_batch_item(
                item,
                index,
                batch_size,
                key=f"{key}[{idx}]",
                query_lengths=query_lengths,
            )
            for idx, item in enumerate(value)
        ]
    if isinstance(value, tuple) and len(value) == batch_size:
        return value[index]
    if isinstance(value, list) and len(value) == batch_size:
        return value[index]
    return value


def _hybrid_apc_request_records(
    input_dict: Dict[str, Any],
    *,
    batch_size: int,
) -> tuple[dict[str, Any], ...] | None:
    records = input_dict.get("hybrid_request_records")
    if records is None:
        return None
    if isinstance(records, dict):
        records = (records,)
    elif isinstance(records, list):
        records = tuple(records)
    if not isinstance(records, tuple):
        return None
    if len(records) != batch_size:
        raise ValueError(
            "hybrid APC request record count must match batch size: "
            f"records={len(records)} batch_size={batch_size}"
        )
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("hybrid APC request records must be dictionaries")
    return records


def _hybrid_apc_record_values(
    records: tuple[dict[str, Any], ...] | None,
    key: str,
):
    if not records:
        return None
    values = [record.get(key) for record in records]
    if not any(value is not None for value in values):
        return None
    return values[0] if len(values) == 1 else tuple(values)


def _apply_hybrid_apc_request_record(
    row_input: Dict[str, Any],
    record: dict[str, Any] | None,
) -> None:
    if not isinstance(record, dict):
        return
    for source_key, target_key in (
        ("request_id", "hybrid_request_id"),
        ("vllm_attention_hit_len", "vllm_attention_hit_len"),
        ("request_prefix_len", "request_prefix_len"),
        ("cumulative_hashes_by_prefix_len", "cumulative_hashes_by_prefix_len"),
        ("attention_block_refs_by_prefix_len", "attention_block_refs_by_prefix_len"),
        ("active_suffix_len", "hybrid_active_suffix_len"),
        ("full_input_ids", "hybrid_full_input_ids"),
    ):
        value = record.get(source_key)
        if value is not None:
            if source_key == "full_input_ids" and not isinstance(value, torch.Tensor):
                input_ids = row_input.get("input_ids")
                dtype = (
                    input_ids.dtype
                    if isinstance(input_ids, torch.Tensor)
                    else torch.int64
                )
                device = (
                    input_ids.device
                    if isinstance(input_ids, torch.Tensor)
                    else None
                )
                value = torch.tensor([list(value)], dtype=dtype, device=device)
            row_input[target_key] = value


def _pad_value_for_key(
    neuron_base_instance: "NeuronBaseForCausalLM",
    key: str,
) -> int:
    if key == "input_ids":
        return int(getattr(neuron_base_instance.config, "pad_token_id", 0) or 0)
    if key in {
        "attention_mask",
        "hybrid_restore_mask",
        "hybrid_restore_prefix_lens",
        "hybrid_commit_mask",
    }:
        return 0
    if key in {"position_ids", "rotary_position_id", "rotary_position_ids"}:
        return 1
    if key == "slot_mapping":
        return -1
    return 0


def _right_pad_dim1(tensor: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
    if tensor.ndim < 2 or tensor.shape[1] == target_len:
        return tensor
    if tensor.shape[1] > target_len:
        raise ValueError(
            f"cannot pad tensor with dim1 {tensor.shape[1]} down to {target_len}"
        )
    pad_shape = list(tensor.shape)
    pad_shape[1] = target_len - tensor.shape[1]
    pad = torch.full(
        tuple(pad_shape),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad], dim=1)


def _right_pad_last_dim(
    tensor: torch.Tensor,
    target_len: int,
    pad_value: int,
) -> torch.Tensor:
    if tensor.shape[-1] == target_len:
        return tensor
    if tensor.shape[-1] > target_len:
        raise ValueError(
            f"cannot pad tensor with last dim {tensor.shape[-1]} down to {target_len}"
        )
    pad_shape = list(tensor.shape)
    pad_shape[-1] = target_len - tensor.shape[-1]
    pad = torch.full(
        tuple(pad_shape),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad], dim=-1)


def _resize_dim1(tensor: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
    if tensor.ndim < 2 or tensor.shape[1] == target_len:
        return tensor
    if tensor.shape[1] > target_len:
        return tensor[:, :target_len, ...]
    return _right_pad_dim1(tensor, target_len, pad_value)


def _configured_cte_bucket_len(
    neuron_base_instance: "NeuronBaseForCausalLM",
    current_len: int,
) -> int:
    bucket_sources = (
        getattr(
            getattr(neuron_base_instance, "neuron_config", None),
            "context_encoding_buckets",
            None,
        ),
        getattr(
            getattr(
                getattr(neuron_base_instance, "context_encoding_model", None),
                "neuron_config",
                None,
            ),
            "context_encoding_buckets",
            None,
        ),
        getattr(
            getattr(
                getattr(neuron_base_instance, "context_encoding_model", None),
                "neuron_config",
                None,
            ),
            "buckets",
            None,
        ),
    )
    buckets: list[int] = []
    for source in bucket_sources:
        if source is None:
            continue
        for bucket in source:
            if isinstance(bucket, (list, tuple)):
                if not bucket:
                    continue
                bucket = bucket[0]
            try:
                buckets.append(int(bucket))
            except (TypeError, ValueError):
                continue
        if buckets:
            break
    for bucket in sorted(set(buckets)):
        if current_len <= bucket:
            return bucket
    return current_len


def _pa_block_size(neuron_base_instance: "NeuronBaseForCausalLM") -> int | None:
    for owner in (
        getattr(neuron_base_instance, "neuron_config", None),
        getattr(getattr(neuron_base_instance, "config", None), "neuron_config", None),
    ):
        block_size = getattr(owner, "pa_block_size", None)
        if block_size:
            return int(block_size)
    return None


def _active_block_table_target_len(
    neuron_base_instance: "NeuronBaseForCausalLM",
    target_context_len: int | None,
) -> int | None:
    if target_context_len is None:
        return None
    block_size = _pa_block_size(neuron_base_instance)
    if not block_size:
        return None
    return max(1, (int(target_context_len) + block_size - 1) // block_size)


def _restore_block_table_target_len(
    neuron_base_instance: "NeuronBaseForCausalLM",
    row_input_dicts: list[Dict[str, Any]],
) -> int | None:
    block_size = _pa_block_size(neuron_base_instance)
    if not block_size:
        return None
    target_len = 0
    for row_input in row_input_dicts:
        restore_mask = _single_batch_value(row_input.get("hybrid_restore_mask"))
        if restore_mask is None or _to_python_int(restore_mask) <= 0:
            continue
        restore_prefix_len = _single_batch_value(
            row_input.get("hybrid_restore_prefix_lens")
        )
        if restore_prefix_len is None:
            continue
        restore_blocks = (
            _to_python_int(restore_prefix_len) + block_size - 1
        ) // block_size
        target_len = max(target_len, restore_blocks)
    return target_len or None


def _configured_max_context_len(
    neuron_base_instance: "NeuronBaseForCausalLM",
) -> int | None:
    owners = (
        getattr(neuron_base_instance, "neuron_config", None),
        getattr(getattr(neuron_base_instance, "config", None), "neuron_config", None),
        getattr(
            getattr(neuron_base_instance, "context_encoding_model", None),
            "neuron_config",
            None,
        ),
        getattr(neuron_base_instance, "config", None),
    )
    for owner in owners:
        if owner is None:
            continue
        for attr in (
            "seq_len",
            "max_context_length",
            "max_model_len",
            "max_position_embeddings",
        ):
            value = getattr(owner, attr, None)
            if value is None:
                continue
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return None


def _full_block_table_target_len(
    neuron_base_instance: "NeuronBaseForCausalLM",
    row_input_dicts: list[Dict[str, Any]],
) -> int | None:
    target_len = 0
    block_size = _pa_block_size(neuron_base_instance)
    max_context_len = _configured_max_context_len(neuron_base_instance)
    if block_size and max_context_len:
        target_len = max(target_len, (max_context_len + block_size - 1) // block_size)
    for row_input in row_input_dicts:
        block_table = row_input.get("block_table")
        if not isinstance(block_table, torch.Tensor) or block_table.numel() == 0:
            continue
        if block_table.ndim >= 2:
            target_len = max(target_len, int(block_table.shape[1]))
        elif block_table.ndim == 1:
            target_len = max(target_len, int(block_table.numel()))
    return target_len or None


def _uses_block_backed_restore(row_input_dicts: list[Dict[str, Any]]) -> bool:
    for row_input in row_input_dicts:
        block_table = row_input.get("block_table")
        if not isinstance(block_table, torch.Tensor) or block_table.numel() == 0:
            continue
        restore_mask = _single_batch_value(row_input.get("hybrid_restore_mask"))
        if restore_mask is not None and _to_python_int(restore_mask) > 0:
            return True
        computed_len = _single_batch_value(row_input.get("computed_context_lens"))
        if computed_len is not None and _to_python_int(computed_len) > 0:
            return True
    return False


def _synthesize_slots_from_block_table(
    *,
    block_table_row: torch.Tensor,
    position_row: torch.Tensor,
    q_len: int,
    block_size: int,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if q_len <= 0:
        return torch.empty((0,), dtype=dtype, device=position_row.device)
    positions = position_row[:q_len].to(torch.int64)
    logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
    if logical_blocks.numel() == 0:
        return torch.empty((0,), dtype=dtype, device=position_row.device)
    block_table_i64 = block_table_row.to(torch.int64)
    nonzero_block_indices = (block_table_i64 != 0).nonzero(as_tuple=False).reshape(-1)
    if int(nonzero_block_indices.numel()) > 0:
        block_table_i64 = block_table_i64[: int(nonzero_block_indices[-1].item()) + 1]
    if int(logical_blocks.max().item()) >= int(block_table_i64.shape[0]):
        # Some vLLM cached/chunked rows carry only the active suffix/decode
        # block table, while position_ids remain absolute in the request.
        min_logical_block = int(logical_blocks.min().item())
        rebased_blocks = logical_blocks - min_logical_block
        if int(rebased_blocks.max().item()) >= int(block_table_i64.shape[0]):
            return None
        logical_blocks = rebased_blocks
    offsets = positions.remainder(block_size)
    physical_blocks = torch.index_select(
        block_table_i64,
        0,
        logical_blocks,
    )
    return (physical_blocks * block_size + offsets).to(dtype=dtype)


def _repair_vectorized_slot_mapping(
    neuron_base_instance: "NeuronBaseForCausalLM",
    combined: Dict[str, Any],
) -> None:
    input_ids = combined.get("input_ids")
    position_ids = combined.get("position_ids")
    block_table = combined.get("block_table")
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or not isinstance(position_ids, torch.Tensor)
        or position_ids.ndim != 2
        or not isinstance(block_table, torch.Tensor)
        or block_table.ndim != 2
    ):
        return

    batch_size, target_len = int(input_ids.shape[0]), int(input_ids.shape[1])
    if batch_size <= 0 or target_len <= 0:
        return
    block_size = _pa_block_size(neuron_base_instance)
    if not block_size:
        return

    query_lengths = _vectorized_query_lengths(combined, batch_size=batch_size)
    if query_lengths is None:
        return

    slot_mapping = combined.get("slot_mapping")
    slot_dtype = (
        slot_mapping.dtype
        if isinstance(slot_mapping, torch.Tensor)
        else torch.int32
    )
    slot_device = input_ids.device
    scalar_slots = None
    slot_rows = None
    if isinstance(slot_mapping, torch.Tensor) and slot_mapping.numel() > 0:
        slot_device = slot_mapping.device
        if slot_mapping.ndim == 1 and int(slot_mapping.numel()) == batch_size:
            scalar_slots = slot_mapping.to(dtype=slot_dtype)
        elif slot_mapping.ndim >= 2 and slot_mapping.shape[0] >= batch_size:
            slot_rows = _resize_dim1(
                slot_mapping[:batch_size].to(dtype=slot_dtype),
                target_len,
                -1,
            )

    repaired = torch.full(
        (batch_size, target_len),
        -1,
        dtype=slot_dtype,
        device=slot_device,
    )
    changed = slot_rows is None or tuple(slot_rows.shape[:2]) != (batch_size, target_len)
    for row_idx, query_len in enumerate(query_lengths[:batch_size]):
        q_len = max(0, min(int(query_len), target_len))
        if q_len == 0:
            continue
        if slot_rows is not None and bool((slot_rows[row_idx, :q_len] >= 0).all().item()):
            repaired[row_idx, :q_len] = slot_rows[row_idx, :q_len]
            if q_len < target_len:
                repaired[row_idx, q_len:] = -1
            continue
        if scalar_slots is not None and q_len == 1 and int(scalar_slots.numel()) > row_idx:
            repaired[row_idx, 0] = scalar_slots[row_idx]
            changed = True
            continue
        synthesized = _synthesize_slots_from_block_table(
            block_table_row=block_table[row_idx],
            position_row=position_ids[row_idx],
            q_len=q_len,
            block_size=block_size,
            dtype=slot_dtype,
        )
        if synthesized is None:
            if slot_rows is not None:
                repaired[row_idx] = slot_rows[row_idx]
            elif scalar_slots is not None and int(scalar_slots.numel()) > row_idx:
                repaired[row_idx, 0] = scalar_slots[row_idx]
            continue
        repaired[row_idx, :q_len] = synthesized
        changed = True

    if changed or slot_rows is None or bool((repaired[:, :target_len] != slot_rows).any().item()):
        combined["slot_mapping"] = repaired


def _repair_vectorized_batch_vectors(combined: Dict[str, Any]) -> None:
    input_ids = combined.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 2:
        return
    batch_size = int(input_ids.shape[0])
    if batch_size <= 1:
        return

    seq_ids = combined.get("seq_ids")
    if not isinstance(seq_ids, torch.Tensor) or seq_ids.reshape(-1).shape[0] != batch_size:
        dtype = seq_ids.dtype if isinstance(seq_ids, torch.Tensor) else torch.int32
        device = seq_ids.device if isinstance(seq_ids, torch.Tensor) else input_ids.device
        combined["seq_ids"] = torch.arange(batch_size, dtype=dtype, device=device)

    adapter_ids = combined.get("adapter_ids")
    if (
        not isinstance(adapter_ids, torch.Tensor)
        or adapter_ids.numel() == 0
        or adapter_ids.reshape(-1).shape[0] != batch_size
    ):
        dtype = adapter_ids.dtype if isinstance(adapter_ids, torch.Tensor) else torch.int32
        device = adapter_ids.device if isinstance(adapter_ids, torch.Tensor) else input_ids.device
        fill_value = (
            int(adapter_ids.reshape(-1)[0].item())
            if isinstance(adapter_ids, torch.Tensor) and adapter_ids.numel() > 0
            else 0
        )
        combined["adapter_ids"] = torch.full(
            (batch_size,),
            fill_value,
            dtype=dtype,
            device=device,
        )


def _repair_vectorized_attention_mask_for_block_table(
    neuron_base_instance: "NeuronBaseForCausalLM",
    combined: Dict[str, Any],
) -> None:
    input_ids = combined.get("input_ids")
    attention_mask = combined.get("attention_mask")
    block_table = combined.get("block_table")
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or not isinstance(attention_mask, torch.Tensor)
        or attention_mask.ndim != 2
        or not isinstance(block_table, torch.Tensor)
        or block_table.ndim != 2
    ):
        return

    batch_size = int(input_ids.shape[0])
    if (
        batch_size <= 0
        or block_table.shape[0] != batch_size
        or block_table.shape[1] <= 1
    ):
        return
    restore_mask = _batch_int_list(
        combined,
        "hybrid_restore_mask",
        batch_size=batch_size,
    )
    computed_context_lens = _batch_int_list(
        combined,
        "computed_context_lens",
        batch_size=batch_size,
    )
    if not (
        (restore_mask is not None and any(value > 0 for value in restore_mask))
        or (
            computed_context_lens is not None
            and any(value > 0 for value in computed_context_lens)
        )
    ):
        return

    block_size = _pa_block_size(neuron_base_instance)
    if not block_size:
        return
    target_len = int(block_table.shape[1]) * int(block_size)
    max_context_len = _configured_max_context_len(neuron_base_instance)
    if max_context_len is not None:
        target_len = max(target_len, max_context_len)
    if int(attention_mask.shape[1]) == target_len:
        return
    if int(attention_mask.shape[1]) > target_len:
        target_len = int(attention_mask.shape[1])

    context_lens = _batch_int_list(combined, "full_context_lens", batch_size=batch_size)
    if context_lens is None:
        num_queries = _batch_int_list(combined, "num_queries", batch_size=batch_size)
        if computed_context_lens is not None and num_queries is not None:
            context_lens = [
                computed_len + query_len
                for computed_len, query_len in zip(computed_context_lens, num_queries)
            ]
    if context_lens is None:
        context_lens = [
            int(row.to(torch.int64).sum().item())
            for row in attention_mask[:batch_size]
        ]

    repaired = torch.zeros(
        (batch_size, target_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    for row_idx, context_len in enumerate(context_lens[:batch_size]):
        active_len = max(0, min(int(context_len), target_len))
        if active_len:
            repaired[row_idx, :active_len] = 1
    combined["attention_mask"] = repaired


_VECTOR_CTE_SEQUENCE_KEYS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "slot_mapping",
    "inputs_embeds",
}


def _with_zero_hybrid_apc_slots(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(input_dict)
    seq_ids = input_dict.get("seq_ids")
    if isinstance(seq_ids, torch.Tensor) and seq_ids.ndim >= 1:
        batch_size = int(seq_ids.reshape(-1).shape[0])
        device = seq_ids.device
    else:
        input_ids = input_dict.get("input_ids")
        batch_size = (
            int(input_ids.shape[0])
            if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 1
            else 1
        )
        device = input_ids.device if isinstance(input_ids, torch.Tensor) else None
    kwargs = {"dtype": torch.int32}
    if device is not None:
        kwargs["device"] = device
    zeros = torch.zeros((batch_size,), **kwargs)
    output.setdefault("hybrid_restore_slot_ids", zeros)
    output.setdefault("hybrid_restore_mask", torch.zeros_like(zeros))
    output.setdefault("hybrid_restore_prefix_lens", torch.zeros_like(zeros))
    output.setdefault("hybrid_commit_slot_ids", torch.zeros_like(zeros))
    output.setdefault("hybrid_commit_mask", torch.zeros_like(zeros))
    if "num_queries" not in output:
        query_lengths = _vectorized_query_lengths(output, batch_size=batch_size)
        if query_lengths is None:
            input_ids = output.get("input_ids")
            active_len = (
                int(input_ids.shape[1])
                if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 2
                else 0
            )
            query_lengths = [active_len] * batch_size
        query_kwargs = {"dtype": torch.int32}
        if device is not None:
            query_kwargs["device"] = device
        output["num_queries"] = torch.tensor(
            [[max(0, int(query_len))] for query_len in query_lengths[:batch_size]],
            **query_kwargs,
        )
    input_ids = output.get("input_ids")
    if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 2:
        query_lengths = _vectorized_query_lengths(output, batch_size=batch_size)
        if query_lengths is None:
            query_lengths = [int(input_ids.shape[1])] * batch_size
        attention_mask = torch.zeros(
            input_ids.shape[:2],
            dtype=torch.int32,
            device=input_ids.device,
        )
        for row_idx, query_len in enumerate(query_lengths[:batch_size]):
            active_len = max(0, min(int(query_len), input_ids.shape[1]))
            if active_len:
                attention_mask[row_idx, :active_len] = 1
        output["attention_mask"] = attention_mask
    return output


_UNBACKED_SUFFIX_ONLY_HYBRID_APC_ERROR = (
    "suffix-only hybrid APC received an attention prefix hit "
    "without scheduler-authorized GDN checkpoint metadata"
)


def _is_unbacked_suffix_only_hybrid_apc_error(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and _UNBACKED_SUFFIX_ONLY_HYBRID_APC_ERROR in str(
        exc
    )


def _active_chunk_suffix_len(
    *,
    suffix_len: int,
    active_suffix_len: int | None,
) -> int | None:
    if active_suffix_len is None:
        return int(suffix_len)
    active_len = int(active_suffix_len)
    if active_len <= 0 or active_len > int(suffix_len):
        return None
    return active_len


def _is_seq_id_fallback_request_id(request_id: Any) -> bool:
    return (
        isinstance(request_id, tuple)
        and len(request_id) == 2
        and request_id[0] == "seq_id"
    )


def _is_same_request_chunked_prefill_continuation(
    input_dict: Dict[str, Any],
    *,
    request_id: Any,
    request_prefix_len: int,
    hit_len: int,
    suffix_len: int,
    active_suffix_len: int | None,
) -> bool:
    if suffix_len <= 0 or hit_len <= 0:
        return False
    active_len = _active_chunk_suffix_len(
        suffix_len=suffix_len,
        active_suffix_len=active_suffix_len,
    )
    if active_len is None:
        return False
    if int(request_prefix_len) - int(hit_len) != int(active_len):
        return True
    if (
        input_dict.get("hybrid_prefill_completion_state") is not None
        and not _truthy_single_value(input_dict.get("hybrid_prefill_completion_state"))
        and _request_id_in_collection(
            request_id,
            input_dict.get("hybrid_cached_request_ids"),
        )
    ):
        return True
    if _is_seq_id_fallback_request_id(request_id):
        return True
    return False


def _with_inert_hybrid_apc_chunk_continuation(
    input_dict: Dict[str, Any],
    *,
    hit_len: int,
    active_prefix_len: int,
    suffix_len: int,
) -> Dict[str, Any]:
    input_ids = input_dict.get("input_ids")
    device = input_ids.device if isinstance(input_ids, torch.Tensor) else None
    kwargs = {"dtype": torch.int32}
    if device is not None:
        kwargs["device"] = device

    output = dict(input_dict)
    output["computed_context_lens"] = torch.tensor([[max(0, int(hit_len))]], **kwargs)
    output["full_context_lens"] = torch.tensor(
        [[max(0, int(active_prefix_len))]],
        **kwargs,
    )
    output["num_queries"] = torch.tensor([[max(0, int(suffix_len))]], **kwargs)
    return _with_zero_hybrid_apc_slots(output)


def _with_same_request_gdn_active_carry(
    input_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Keep attention prefix reads but carry same-request GDN state directly."""

    output = dict(input_dict)
    _zero_mask_if_present(output, "hybrid_restore_mask")
    return output


def _replace_prepared_input_dict(prepared, input_dict: Dict[str, Any]):
    if hasattr(prepared, "_replace"):
        return prepared._replace(input_dict=input_dict)
    prepared.input_dict = input_dict
    return prepared


def _is_completed_cached_decode_row(
    input_dict: Dict[str, Any],
    *,
    request_id: Any,
    query_len: int | None,
) -> bool:
    if not _request_id_in_collection(
        request_id,
        input_dict.get("hybrid_cached_request_ids"),
    ):
        return False
    if not _truthy_single_value(input_dict.get("hybrid_prefill_completion_state")):
        return False
    return query_len is None or int(query_len) <= 1


def _combine_vectorized_hybrid_apc_inputs(
    neuron_base_instance: "NeuronBaseForCausalLM",
    original_input_dict: Dict[str, Any],
    row_input_dicts: list[Dict[str, Any]],
) -> Dict[str, Any]:
    combined = dict(original_input_dict)
    keys: set[str] = set()
    for row_input in row_input_dicts:
        keys.update(row_input.keys())

    max_sequence_dim1 = None
    for key in _VECTOR_CTE_SEQUENCE_KEYS:
        tensors = [
            row_input.get(key)
            for row_input in row_input_dicts
            if isinstance(row_input.get(key), torch.Tensor)
        ]
        if not tensors or not any(tensor.ndim >= 1 for tensor in tensors):
            continue
        key_max = max(tensor.shape[1] if tensor.ndim >= 2 else 1 for tensor in tensors)
        max_sequence_dim1 = (
            key_max
            if max_sequence_dim1 is None
            else max(max_sequence_dim1, key_max)
        )
    target_sequence_dim1 = (
        _configured_cte_bucket_len(neuron_base_instance, max_sequence_dim1)
        if max_sequence_dim1 is not None
        else None
    )
    target_block_dim1 = _active_block_table_target_len(
        neuron_base_instance,
        target_sequence_dim1,
    )
    restore_block_dim1 = _restore_block_table_target_len(
        neuron_base_instance,
        row_input_dicts,
    )
    full_block_dim1 = (
        _full_block_table_target_len(neuron_base_instance, row_input_dicts)
        if _uses_block_backed_restore(row_input_dicts)
        else None
    )
    if full_block_dim1 is not None:
        target_block_dim1 = max(target_block_dim1 or 0, full_block_dim1)

    for key in keys:
        if key.startswith("_hybrid_apc"):
            continue
        values = [row_input.get(key) for row_input in row_input_dicts]
        if not all(isinstance(value, torch.Tensor) for value in values):
            continue

        tensors = [value for value in values if isinstance(value, torch.Tensor)]
        if all(tensor.numel() == 0 for tensor in tensors):
            combined[key] = tensors[0]
            continue
        if all(tensor.ndim == 0 for tensor in tensors):
            combined[key] = torch.stack(tensors)
            continue
        if (
            key in {"rotary_position_id", "rotary_position_ids"}
            and all(tensor.ndim == 3 and tensor.shape[1] == 1 for tensor in tensors)
        ):
            target_dim = max(tensor.shape[-1] for tensor in tensors)
            combined[key] = torch.cat(
                [
                    _right_pad_last_dim(
                        tensor,
                        target_dim,
                        _pad_value_for_key(neuron_base_instance, key),
                    )
                    for tensor in tensors
                ],
                dim=1,
            )
            continue
        if all(tensor.ndim >= 1 and tensor.shape[0] == 1 for tensor in tensors):
            max_dim1 = None
            if any(tensor.ndim >= 2 for tensor in tensors):
                max_dim1 = max(tensor.shape[1] if tensor.ndim >= 2 else 1 for tensor in tensors)
            target_dim1 = max_dim1
            if key in _VECTOR_CTE_SEQUENCE_KEYS and target_sequence_dim1 is not None:
                target_dim1 = target_sequence_dim1
            elif key == "block_table" and target_block_dim1 is not None:
                target_dim1 = target_block_dim1
                if restore_block_dim1 is not None:
                    target_dim1 = max(target_dim1, restore_block_dim1)
            padded = []
            for tensor in tensors:
                current = (
                    tensor.reshape(1, -1)
                    if target_dim1 is not None and tensor.ndim == 1
                    else tensor
                )
                if target_dim1 is not None and current.ndim >= 2:
                    resize = _resize_dim1 if key == "block_table" else _right_pad_dim1
                    current = resize(
                        current,
                        target_dim1,
                        _pad_value_for_key(neuron_base_instance, key),
                    )
                padded.append(current)
            try:
                combined[key] = torch.cat(padded, dim=0)
            except RuntimeError as exc:
                raise ValueError(
                    f"cannot combine vectorized hybrid APC tensor {key!r}: "
                    f"{[tuple(tensor.shape) for tensor in padded]}"
                ) from exc
            continue
        if all(tuple(tensor.shape) == tuple(tensors[0].shape) for tensor in tensors):
            combined[key] = tensors[0]

    _repair_vectorized_batch_vectors(combined)
    _repair_vectorized_attention_mask_for_block_table(neuron_base_instance, combined)
    _repair_vectorized_slot_mapping(neuron_base_instance, combined)
    return combined


def _prepare_vectorized_hybrid_apc_requests(
    neuron_base_instance: "NeuronBaseForCausalLM",
    input_dict: Dict[str, Any],
    *,
    bridge: Any,
    batch_size: int,
) -> Dict[str, Any]:
    row_outputs: list[Dict[str, Any]] = []
    prepared_requests = []
    query_lengths = _vectorized_query_lengths(input_dict, batch_size=batch_size)
    request_records = _hybrid_apc_request_records(
        input_dict,
        batch_size=batch_size,
    )
    try:
        for index in range(batch_size):
            row_input = {
                key: _select_batch_item(
                    value,
                    index,
                    batch_size,
                    key=key,
                    query_lengths=query_lengths,
                )
                for key, value in input_dict.items()
                if not key.startswith("_hybrid_apc")
            }
            if request_records is not None:
                _apply_hybrid_apc_request_record(row_input, request_records[index])
            row_input["hybrid_apc_bridge"] = bridge
            row_output = prepare_hybrid_apc_request_for_execution(
                neuron_base_instance,
                row_input,
            )
            row_outputs.append(row_output)
            prepared = row_input.get("_hybrid_apc_prepared")
            if prepared is not None:
                prepared_requests.append(prepared)
    except Exception:
        for prepared in prepared_requests:
            bridge.cancel_request(prepared)
        raise

    if prepared_requests:
        input_dict["_hybrid_apc_bridge"] = bridge
        input_dict["_hybrid_apc_prepared"] = prepared_requests

    combined = _combine_vectorized_hybrid_apc_inputs(
        neuron_base_instance,
        input_dict,
        row_outputs,
    )
    if os.environ.get("QWEN36_HYBRID_APC_DEBUG") == "1":
        print(
            "[hybrid_apc_debug] prepare-vectorized "
            f"batch_size={batch_size} prepared={len(prepared_requests)} "
            f"input_shape={tuple(input_dict['input_ids'].shape)} "
            f"prepared_shape={tuple(combined['input_ids'].shape)} "
            f"computed={combined.get('computed_context_lens')} "
            f"num_queries={combined.get('num_queries')} "
            f"restore_mask={combined.get('hybrid_restore_mask')} "
            f"commit_mask={combined.get('hybrid_commit_mask')}",
            flush=True,
        )
    return combined


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _hybrid_gdn_restore_disabled() -> bool:
    return _env_flag("QWEN36_DISABLE_HYBRID_GDN_RESTORE") or _env_flag(
        "QWEN36_DISABLE_HYBRID_GDN_RESTORE_COMMIT"
    )


def _hybrid_gdn_restore_mask_zeroed() -> bool:
    return _env_flag("QWEN36_ZERO_HYBRID_GDN_RESTORE_MASK")


def _hybrid_gdn_commit_disabled() -> bool:
    return _env_flag("QWEN36_DISABLE_HYBRID_GDN_COMMIT") or _env_flag(
        "QWEN36_DISABLE_HYBRID_GDN_RESTORE_COMMIT"
    )


def _zero_mask_if_present(input_dict: Dict[str, Any], key: str):
    value = input_dict.get(key)
    if isinstance(value, torch.Tensor):
        input_dict[key] = torch.zeros_like(value)


def _apply_hybrid_gdn_debug_switches(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    if _hybrid_gdn_restore_disabled() or _hybrid_gdn_restore_mask_zeroed():
        _zero_mask_if_present(input_dict, "hybrid_restore_mask")
    if _hybrid_gdn_commit_disabled():
        _zero_mask_if_present(input_dict, "hybrid_commit_mask")
    return input_dict


def _get_hybrid_apc_bridge(
    neuron_base_instance: "NeuronBaseForCausalLM",
    input_dict: Dict[str, Any],
):
    bridge = _first_present(
        input_dict.get("hybrid_apc_bridge"),
        getattr(neuron_base_instance, "hybrid_apc_bridge", None),
        getattr(neuron_base_instance, "_hybrid_apc_last_bridge", None),
    )
    if bridge is None:
        ensure_bridge = getattr(
            neuron_base_instance,
            "ensure_hybrid_apc_scheduler_bridge",
            None,
        )
        if ensure_bridge is not None:
            bridge = ensure_bridge()
    return bridge


def _select_hybrid_apc_owner(
    neuron_base_instance: "NeuronBaseForCausalLM",
    model_to_execute: "ModelWrapper",
    input_dict: Dict[str, Any],
):
    candidates = (
        neuron_base_instance,
        model_to_execute,
        getattr(neuron_base_instance, "context_encoding_model", None),
        getattr(neuron_base_instance, "token_generation_model", None),
    )
    fallback = neuron_base_instance
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        if fallback is neuron_base_instance and _is_hybrid_apc_enabled(candidate):
            fallback = candidate
        if not _is_hybrid_apc_enabled(candidate):
            continue
        if _get_hybrid_apc_bridge(candidate, input_dict) is not None:
            return candidate
    return fallback


def _requires_external_hybrid_apc_metadata(
    neuron_base_instance: "NeuronBaseForCausalLM",
    bridge: Any,
) -> bool:
    for owner in (
        getattr(neuron_base_instance, "config", None),
        getattr(neuron_base_instance, "neuron_config", None),
        getattr(getattr(neuron_base_instance, "config", None), "neuron_config", None),
    ):
        if bool(getattr(owner, "hybrid_apc_require_vllm_metadata", False)):
            return True
    return bool(
        getattr(bridge, "requires_external_metadata", False)
    )


def prepare_hybrid_apc_request_for_execution(
    neuron_base_instance: "NeuronBaseForCausalLM",
    input_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Run scheduler-side hybrid APC request preparation when metadata exists.

    The concrete bridge lives with the Qwen contrib model. This function is
    intentionally duck-typed so the core async path does not import contrib
    modules. vLLM/NxDI request prep can opt in by attaching a bridge object plus
    attention-hit metadata to ``input_dict``.
    """

    if not _is_hybrid_apc_enabled(neuron_base_instance):
        return input_dict

    bridge = _get_hybrid_apc_bridge(neuron_base_instance, input_dict)
    requires_external_metadata = _requires_external_hybrid_apc_metadata(
        neuron_base_instance,
        bridge,
    )
    if bridge is None:
        if requires_external_metadata:
            raise ValueError(
                "hybrid APC requires a scheduler bridge attached to the model "
                "or input_dict"
            )
        return input_dict

    lifecycle_input_dict = input_dict
    batch_size = _batch_size_from_input_dict(input_dict)
    request_records = _hybrid_apc_request_records(
        input_dict,
        batch_size=batch_size,
    )
    if batch_size == 1 and request_records is not None:
        input_dict = dict(input_dict)
        _apply_hybrid_apc_request_record(input_dict, request_records[0])

    request_id = _first_present(
        input_dict.get("hybrid_request_id"),
        input_dict.get("request_id"),
        _hybrid_apc_record_values(request_records, "request_id"),
    )
    if request_id is None:
        seq_id = _single_batch_value(input_dict.get("seq_ids"))
        if seq_id is not None:
            request_id = ("seq_id", _to_python_int(seq_id))

    attention_hit_len_source = _first_present(
        _hybrid_apc_record_values(request_records, "vllm_attention_hit_len"),
        input_dict.get("vllm_attention_hit_len"),
        input_dict.get("hybrid_attention_hit_len"),
        input_dict.get("attention_hit_len"),
    )
    if attention_hit_len_source is None:
        attention_hit_len_source = input_dict.get("computed_context_lens")
    attention_hit_len = _single_batch_value(attention_hit_len_source)
    multi_attention_hit_lens = _multi_batch_int_values(attention_hit_len_source)
    if attention_hit_len is None and multi_attention_hit_lens is None:
        multi_attention_hit_lens = _multi_batch_int_values(
            input_dict.get("computed_context_lens")
        )
    if (
        attention_hit_len is None
        and multi_attention_hit_lens is not None
        and batch_size > 1
    ):
        if (
            all(hit_len == 0 for hit_len in multi_attention_hit_lens)
            and not requires_external_metadata
        ):
            return input_dict
        return _prepare_vectorized_hybrid_apc_requests(
            neuron_base_instance,
            input_dict,
            bridge=bridge,
            batch_size=batch_size,
        )
    if attention_hit_len is None and multi_attention_hit_lens is not None:
        if all(hit_len == 0 for hit_len in multi_attention_hit_lens):
            if requires_external_metadata:
                raise ValueError(
                    "hybrid APC v0 request prep supports one request at a time; "
                    "vectorized continuous-batching metadata is not wired yet"
                )
            return input_dict
        raise ValueError(
            "hybrid APC v0 request prep supports one request at a time; "
            "vectorized continuous-batching metadata is not wired yet"
        )

    if request_id is None and attention_hit_len is None and not requires_external_metadata:
        return input_dict
    if request_id is None:
        raise ValueError("hybrid APC request prep requires request_id")
    if attention_hit_len is None:
        raise ValueError("hybrid APC request prep requires attention hit length")

    request_prefix_len = _first_present(
        input_dict.get("request_prefix_len"),
        input_dict.get("hybrid_request_prefix_len"),
        input_dict.get("prompt_len"),
        _single_batch_value(input_dict.get("full_context_lens")),
    )
    if request_prefix_len is not None:
        request_prefix_len = _to_python_int(request_prefix_len)

    active_suffix_len = _first_present(
        input_dict.get("hybrid_active_suffix_len"),
        input_dict.get("active_suffix_len"),
        _hybrid_apc_record_values(request_records, "active_suffix_len"),
    )
    active_suffix_len = _single_batch_value(active_suffix_len)
    if active_suffix_len is not None:
        active_suffix_len = max(0, _to_python_int(active_suffix_len))

    query_len = _single_batch_value(input_dict.get("num_queries"))
    if query_len is None and active_suffix_len is not None:
        query_len = active_suffix_len
    elif (
        query_len is None
        and request_prefix_len is not None
        and attention_hit_len is not None
    ):
        query_len = max(0, request_prefix_len - _to_python_int(attention_hit_len))
    elif query_len is not None:
        query_len = _to_python_int(query_len)

    if _is_completed_cached_decode_row(
        input_dict,
        request_id=request_id,
        query_len=query_len,
    ):
        return _with_zero_hybrid_apc_slots(input_dict)

    cumulative_hashes_by_prefix_len = _first_present(
        input_dict.get("vllm_or_local_prefix_hashes"),
        input_dict.get("cumulative_hashes_by_prefix_len"),
        input_dict.get("hybrid_cumulative_hashes_by_prefix_len"),
    )
    attention_block_refs_by_prefix_len = _first_present(
        input_dict.get("attention_block_refs"),
        input_dict.get("attention_block_refs_by_prefix_len"),
        input_dict.get("hybrid_attention_block_refs_by_prefix_len"),
    )
    if (
        requires_external_metadata
        and not cumulative_hashes_by_prefix_len
        and _to_python_int(attention_hit_len) <= 0
    ):
        return _with_zero_hybrid_apc_slots(input_dict)
    full_input_ids = _first_present(
        input_dict.get("hybrid_full_input_ids"),
        input_dict.get("full_input_ids"),
        input_dict.get("prompt_input_ids"),
    )
    bridge_input_dict = input_dict
    prepared = None
    if full_input_ids is not None:
        if not _single_batch_tensor(full_input_ids):
            if requires_external_metadata:
                raise ValueError(
                    "hybrid APC v0 request prep supports one request at a time; "
                    "vectorized continuous-batching metadata is not wired yet"
                )
            return input_dict
        bridge_input_dict = dict(input_dict)
        bridge_input_dict["input_ids"] = full_input_ids
        for source_key, target_key in (
            ("hybrid_full_attention_mask", "attention_mask"),
            ("full_attention_mask", "attention_mask"),
            ("hybrid_full_position_ids", "position_ids"),
            ("full_position_ids", "position_ids"),
            ("hybrid_full_slot_mapping", "slot_mapping"),
            ("full_slot_mapping", "slot_mapping"),
        ):
            value = input_dict.get(source_key)
            if value is not None:
                bridge_input_dict[target_key] = value
    elif request_prefix_len is not None:
        input_ids = input_dict.get("input_ids")
        if (
            isinstance(input_ids, torch.Tensor)
            and input_ids.ndim >= 2
            and input_ids.shape[1] < request_prefix_len
        ):
            if _to_python_int(attention_hit_len) <= 0:
                active_prefix_len = min(
                    request_prefix_len,
                    int(input_ids.shape[1]),
                )
                if (
                    cumulative_hashes_by_prefix_len
                    and active_prefix_len in cumulative_hashes_by_prefix_len
                ):
                    prepared = bridge.prepare_request(
                        request_id=request_id,
                        input_dict=input_dict,
                        attention_hit_len=0,
                        request_prefix_len=active_prefix_len,
                        cumulative_hashes_by_prefix_len=cumulative_hashes_by_prefix_len,
                        attention_block_refs_by_prefix_len=attention_block_refs_by_prefix_len,
                    )
                else:
                    return _with_zero_hybrid_apc_slots(input_dict)
            # The live prefix-caching request has already been sliced to the
            # attention suffix. Without full prompt tokens the bridge cannot
            # compute or apply an exact GDN checkpoint boundary.
            else:
                prepare_suffix_only = getattr(
                    bridge,
                    "prepare_suffix_only_request",
                    None,
                )
                if prepare_suffix_only is not None:
                    suffix_len = int(input_ids.shape[1])
                    active_prefix_len = request_prefix_len
                    hit_len = _to_python_int(attention_hit_len)
                    # vLLM chunked prefill may report the final prompt length in
                    # request_prefix_len while scheduling only the next suffix
                    # chunk. Hybrid APC restore/commit must use the active chunk
                    # boundary, otherwise the suffix-only bridge rejects the row.
                    same_request_chunk_continuation = (
                        _is_same_request_chunked_prefill_continuation(
                            input_dict,
                            request_id=request_id,
                            request_prefix_len=request_prefix_len,
                            hit_len=hit_len,
                            suffix_len=suffix_len,
                            active_suffix_len=active_suffix_len,
                        )
                    )
                    if same_request_chunk_continuation:
                        active_chunk_suffix_len = _active_chunk_suffix_len(
                            suffix_len=suffix_len,
                            active_suffix_len=active_suffix_len,
                        )
                        if active_chunk_suffix_len is None:
                            active_chunk_suffix_len = suffix_len
                        active_prefix_len = min(
                            request_prefix_len,
                            hit_len + active_chunk_suffix_len,
                        )
                        if (
                            active_chunk_suffix_len != suffix_len
                            or _is_seq_id_fallback_request_id(request_id)
                        ):
                            return _with_inert_hybrid_apc_chunk_continuation(
                                input_dict,
                                hit_len=hit_len,
                                active_prefix_len=active_prefix_len,
                                suffix_len=active_chunk_suffix_len,
                            )
                    try:
                        prepared = prepare_suffix_only(
                            request_id=request_id,
                            input_dict=input_dict,
                            attention_hit_len=hit_len,
                            request_prefix_len=active_prefix_len,
                            cumulative_hashes_by_prefix_len=cumulative_hashes_by_prefix_len,
                            attention_block_refs_by_prefix_len=attention_block_refs_by_prefix_len,
                        )
                        if (
                            prepared is not None
                            and same_request_chunk_continuation
                        ):
                            prepared = _replace_prepared_input_dict(
                                prepared,
                                _with_same_request_gdn_active_carry(
                                    prepared.input_dict
                                ),
                            )
                    except ValueError as exc:
                        if os.environ.get("QWEN36_HYBRID_APC_DEBUG") == "1":
                            print(
                                "[hybrid_apc_debug] suffix-unbacked-check "
                                f"request_id={request_id!r} "
                                f"same_request_chunk_continuation={same_request_chunk_continuation} "
                                f"request_prefix_len={request_prefix_len} "
                                f"hit_len={hit_len} suffix_len={suffix_len} "
                                f"active_suffix_len={active_suffix_len} "
                                f"query_len={query_len} "
                                f"error={exc}",
                                flush=True,
                            )
                        if (
                            same_request_chunk_continuation
                            and _is_unbacked_suffix_only_hybrid_apc_error(exc)
                        ):
                            return _with_inert_hybrid_apc_chunk_continuation(
                                input_dict,
                                hit_len=hit_len,
                                active_prefix_len=active_prefix_len,
                                suffix_len=suffix_len,
                            )
                        raise
            if prepared is None:
                if requires_external_metadata:
                    raise ValueError(
                        "hybrid APC production mode received suffix-only input "
                        "without hybrid_full_input_ids/full_input_ids; request prep "
                        "must attach full prompt tokens before suffix slicing"
                    )
                return input_dict
    if not _single_batch_tensor(bridge_input_dict.get("input_ids")):
        if requires_external_metadata:
            raise ValueError(
                "hybrid APC v0 request prep supports one request at a time; "
                "vectorized continuous-batching metadata is not wired yet"
            )
        return input_dict

    if prepared is None:
        prepared = bridge.prepare_request(
            request_id=request_id,
            input_dict=bridge_input_dict,
            attention_hit_len=_to_python_int(attention_hit_len),
            request_prefix_len=request_prefix_len,
            cumulative_hashes_by_prefix_len=cumulative_hashes_by_prefix_len,
            attention_block_refs_by_prefix_len=attention_block_refs_by_prefix_len,
        )
    setattr(neuron_base_instance, "_hybrid_apc_last_bridge", bridge)
    input_dict["_hybrid_apc_bridge"] = bridge
    input_dict["_hybrid_apc_prepared"] = prepared
    prepared.input_dict["_hybrid_apc_bridge"] = bridge
    prepared.input_dict["_hybrid_apc_prepared"] = prepared
    if lifecycle_input_dict is not input_dict:
        lifecycle_input_dict["_hybrid_apc_bridge"] = bridge
        lifecycle_input_dict["_hybrid_apc_prepared"] = prepared
    _apply_hybrid_gdn_debug_switches(prepared.input_dict)
    if os.environ.get("QWEN36_HYBRID_APC_DEBUG") == "1":
        prepared_inputs = prepared.input_dict
        print(
            "[hybrid_apc_debug] prepare "
            f"request_id={request_id!r} attention_hit_len={_to_python_int(attention_hit_len)} "
            f"request_prefix_len={request_prefix_len} restore_len={prepared.plan.restore_checkpoint_prefix_len} "
            f"commit_prefix_len={prepared.commit_prefix_len} restore_slot={prepared.plan.checkpoint_slot} "
            f"commit_slot={prepared.commit_slot} input_shape={tuple(input_dict['input_ids'].shape)} "
            f"prepared_shape={tuple(prepared_inputs['input_ids'].shape)} "
            f"computed={prepared_inputs.get('computed_context_lens')} "
            f"num_queries={prepared_inputs.get('num_queries')} "
            f"restore_mask={prepared_inputs.get('hybrid_restore_mask')} "
            f"commit_mask={prepared_inputs.get('hybrid_commit_mask')}",
            flush=True,
        )
    return prepared.input_dict


def finish_hybrid_apc_request(input_dict: Dict[str, Any]):
    bridge = input_dict.pop("_hybrid_apc_bridge", None)
    prepared = input_dict.pop("_hybrid_apc_prepared", None)
    if os.environ.get("QWEN36_HYBRID_APC_DEBUG") == "1":
        print(
            "[hybrid_apc_debug] finish "
            f"has_bridge={bridge is not None} has_prepared={prepared is not None}",
            flush=True,
        )
    if bridge is None or prepared is None:
        return

    prepared_requests = prepared if isinstance(prepared, list) else [prepared]
    if _hybrid_gdn_commit_disabled():
        for prepared_request in prepared_requests:
            bridge.cancel_request(prepared_request)
        return
    for prepared_request in prepared_requests:
        actual_refs = _first_present(
            input_dict.get("actual_refs"),
            input_dict.get("actual_attention_block_refs"),
            input_dict.get("hybrid_actual_attention_block_refs"),
            getattr(prepared_request, "attention_block_refs", None),
        )
        if os.environ.get("QWEN36_HYBRID_APC_DEBUG") == "1":
            print(
                "[hybrid_apc_debug] finish-commit "
                f"request_id={prepared_request.request_id!r} "
                f"commit_prefix_len={getattr(prepared_request, 'commit_prefix_len', None)} "
                f"commit_slot={getattr(prepared_request, 'commit_slot', None)} "
                f"actual_refs={actual_refs}",
                flush=True,
            )
        try:
            bridge.commit_prefill(prepared_request, attention_block_refs=actual_refs)
        except Exception:
            bridge.cancel_request(prepared_request)
            raise
        bridge.finish_request(prepared_request.request_id)


def cancel_hybrid_apc_request(input_dict: Dict[str, Any]):
    bridge = input_dict.pop("_hybrid_apc_bridge", None)
    prepared = input_dict.pop("_hybrid_apc_prepared", None)
    if bridge is None or prepared is None:
        return
    prepared_requests = prepared if isinstance(prepared, list) else [prepared]
    for prepared_request in prepared_requests:
        bridge.cancel_request(prepared_request)


def _active_hybrid_apc_slots(
    slot_ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    name: str,
) -> list[int]:
    slot_ids = slot_ids.reshape(-1).to(torch.int64)
    mask = mask.reshape(-1).to(torch.bool)
    if slot_ids.shape != mask.shape:
        raise ValueError(f"{name} slot ids and mask must have matching shape")
    return [int(slot.item()) for slot in slot_ids[mask]]


def _validate_hybrid_apc_slot_inputs(
    neuron_base_instance: "NeuronBaseForCausalLM",
    *,
    restore_slot_ids: torch.Tensor,
    restore_mask: torch.Tensor,
    commit_slot_ids: torch.Tensor,
    commit_mask: torch.Tensor,
):
    """Validate active checkpoint slots before the traced model clamps them."""

    active_restore_slots = _active_hybrid_apc_slots(
        restore_slot_ids,
        restore_mask,
        name="hybrid restore",
    )
    active_commit_slots = _active_hybrid_apc_slots(
        commit_slot_ids,
        commit_mask,
        name="hybrid commit",
    )
    if not active_restore_slots and not active_commit_slots:
        return

    max_slots = getattr(
        getattr(neuron_base_instance, "config", None),
        "max_gdn_checkpoint_slots",
        None,
    )
    if max_slots is not None:
        max_slots = int(max_slots)
        for kind, slots in (
            ("restore", active_restore_slots),
            ("commit", active_commit_slots),
        ):
            for slot in slots:
                if slot < 0 or slot >= max_slots:
                    raise ValueError(
                        f"hybrid APC {kind} slot {slot} is outside [0, {max_slots})"
                    )

    allocator = getattr(neuron_base_instance, "hybrid_apc_slot_allocator", None)
    if allocator is None:
        return

    committed_slots = set(getattr(allocator, "committed_slots", ()))
    reserved_slots = set(getattr(allocator, "reserved_slots", ()))
    for slot in active_restore_slots:
        if slot not in committed_slots:
            raise ValueError(
                f"hybrid APC restore slot {slot} is not a committed checkpoint slot"
            )
    for slot in active_commit_slots:
        if slot not in reserved_slots:
            raise ValueError(
                f"hybrid APC commit slot {slot} is not a reserved checkpoint slot"
            )


def prepare_hybrid_apc_model_inputs(
    neuron_base_instance: "NeuronBaseForCausalLM",
    input_dict: Dict[str, Any],
) -> list[torch.Tensor]:
    """Build optional Qwen hybrid APC args for prefix-caching execution.

    The vLLM scheduler owns prefix hashes and checkpoint-slot allocation. This
    bridge only translates scheduler-provided values into the fixed traced model
    inputs. If no restore/commit slots are supplied, masks stay zero and the
    model executes without GDN checkpoint reuse.
    """

    if not _is_hybrid_apc_enabled(neuron_base_instance):
        return []

    batch_size = int(input_dict["seq_ids"].reshape(-1).shape[0])
    empty = torch.empty(0)

    computed_context_lens = input_dict.get("computed_context_lens")
    if computed_context_lens is None:
        restore_prefix_lens = torch.zeros((batch_size,), dtype=torch.int32)
    else:
        restore_prefix_lens = computed_context_lens.reshape(-1).to(torch.int32)
        if restore_prefix_lens.shape[0] != batch_size:
            restore_prefix_lens = _batch_vector(
                {"value": restore_prefix_lens},
                "value",
                batch_size=batch_size,
            )

    restore_slot_ids = _batch_vector(
        input_dict,
        "hybrid_restore_slot_ids",
        batch_size=batch_size,
        default=0,
    )
    restore_mask = _batch_vector(
        input_dict,
        "hybrid_restore_mask",
        batch_size=batch_size,
        default=0,
    )

    if "hybrid_restore_prefix_lens" in input_dict:
        restore_prefix_lens = _batch_vector(
            input_dict,
            "hybrid_restore_prefix_lens",
            batch_size=batch_size,
            default=0,
        )

    commit_slot_ids = _batch_vector(
        input_dict,
        "hybrid_commit_slot_ids",
        batch_size=batch_size,
        default=0,
    )
    if "hybrid_commit_mask" in input_dict:
        commit_mask = _batch_vector(
            input_dict,
            "hybrid_commit_mask",
            batch_size=batch_size,
            default=0,
        )
    else:
        commit_mask = torch.zeros((batch_size,), dtype=torch.int32)

    switch_inputs = {
        "hybrid_restore_mask": restore_mask,
        "hybrid_commit_mask": commit_mask,
    }
    _apply_hybrid_gdn_debug_switches(switch_inputs)
    restore_mask = switch_inputs["hybrid_restore_mask"]
    commit_mask = switch_inputs["hybrid_commit_mask"]

    _validate_hybrid_apc_slot_inputs(
        neuron_base_instance,
        restore_slot_ids=restore_slot_ids,
        restore_mask=restore_mask,
        commit_slot_ids=commit_slot_ids,
        commit_mask=commit_mask,
    )

    llava_args = input_dict.get("llava_args") or []
    rotary_position_id = _first_present(
        input_dict.get("rotary_position_id"),
        input_dict.get("rotary_position_ids"),
        llava_args[2] if len(llava_args) >= 3 else None,
        empty,
    )
    vision_embeddings = _first_present(
        input_dict.get("vision_embeddings"),
        llava_args[0] if len(llava_args) >= 1 else None,
        empty,
    )
    vision_mask = _first_present(
        input_dict.get("vision_mask"),
        llava_args[1] if len(llava_args) >= 2 else None,
        empty,
    )

    return [
        input_dict.get("tile_q_indices", empty),
        input_dict.get("tile_block_tables", empty),
        input_dict.get("tile_masks", empty),
        input_dict.get("inputs_embeds", empty),
        input_dict.get("kv_cache", empty),
        input_dict.get("active_mask", empty),
        rotary_position_id,
        vision_embeddings,
        vision_mask,
        restore_slot_ids,
        restore_mask,
        restore_prefix_lens,
        commit_slot_ids,
        commit_mask,
    ]


def prepare_disabled_hybrid_apc_model_inputs(
    neuron_base_instance: "NeuronBaseForCausalLM",
    input_dict: Dict[str, Any],
) -> list[torch.Tensor]:
    """Build inert Hybrid APC args for decode/TKG execution.

    Hybrid APC restore/commit is a prefill concern. The compiled model still has
    the fixed Hybrid APC inputs, so decode must pass the same arity, but it does
    not need request planning, slot validation, or mask/vector normalization on
    every generated token.
    """

    if not _is_hybrid_apc_enabled(neuron_base_instance):
        return []

    seq_ids = input_dict["seq_ids"].reshape(-1)
    batch_size = int(seq_ids.shape[0])
    device = seq_ids.device
    empty = torch.empty(0, device=device)
    zeros = torch.zeros((batch_size,), dtype=torch.int32, device=device)

    llava_args = input_dict.get("llava_args") or []
    rotary_position_id = _first_present(
        input_dict.get("rotary_position_id"),
        input_dict.get("rotary_position_ids"),
        llava_args[2] if len(llava_args) >= 3 else None,
        empty,
    )
    vision_embeddings = _first_present(
        input_dict.get("vision_embeddings"),
        llava_args[0] if len(llava_args) >= 1 else None,
        empty,
    )
    vision_mask = _first_present(
        input_dict.get("vision_mask"),
        llava_args[1] if len(llava_args) >= 2 else None,
        empty,
    )

    return [
        input_dict.get("tile_q_indices", empty),
        input_dict.get("tile_block_tables", empty),
        input_dict.get("tile_masks", empty),
        input_dict.get("inputs_embeds", empty),
        input_dict.get("kv_cache", empty),
        input_dict.get("active_mask", empty),
        rotary_position_id,
        vision_embeddings,
        vision_mask,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
    ]


def _is_context_encoding_execution(
    neuron_base_instance: "NeuronBaseForCausalLM",
    model_to_execute: "ModelWrapper",
    input_dict: Dict[str, Any],
) -> bool:
    if getattr(model_to_execute, "tag", None) == "context_encoding_model":
        return True
    input_ids = input_dict.get("input_ids")
    if (
        isinstance(input_ids, torch.Tensor)
        and input_ids.ndim >= 2
        and input_ids.shape[-1] > 1
        and not getattr(neuron_base_instance.neuron_config, "enable_fused_speculation", False)
        and not getattr(neuron_base_instance.neuron_config, "enable_eagle_speculation", False)
    ):
        return True
    is_prefill = getattr(neuron_base_instance, "_is_prefill", None)
    position_ids = input_dict.get("position_ids")
    if callable(is_prefill) and position_ids is not None:
        return bool(is_prefill(position_ids))
    return False


def _is_cached_chunked_prefill_continuation(inputs: Dict[str, Any]) -> bool:
    batch_size = _batch_size_from_input_dict(inputs)
    if batch_size != 1:
        return False

    request_records = _hybrid_apc_request_records(inputs, batch_size=batch_size)
    active_suffix_len = _first_present(
        inputs.get("hybrid_active_suffix_len"),
        inputs.get("active_suffix_len"),
        _hybrid_apc_record_values(request_records, "active_suffix_len"),
    )
    active_suffix_len = _single_batch_value(active_suffix_len)
    if active_suffix_len is None:
        return False
    active_suffix_len = _to_python_int(active_suffix_len)
    if active_suffix_len <= 0:
        return False

    attention_hit_len = _first_present(
        _hybrid_apc_record_values(request_records, "vllm_attention_hit_len"),
        inputs.get("vllm_attention_hit_len"),
        inputs.get("hybrid_attention_hit_len"),
        inputs.get("attention_hit_len"),
        inputs.get("computed_context_lens"),
    )
    attention_hit_len = _single_batch_value(attention_hit_len)
    if attention_hit_len is None:
        return False
    attention_hit_len = _to_python_int(attention_hit_len)
    if attention_hit_len <= 0:
        return False

    request_prefix_len = _first_present(
        _hybrid_apc_record_values(request_records, "request_prefix_len"),
        inputs.get("request_prefix_len"),
        inputs.get("hybrid_request_prefix_len"),
        inputs.get("prompt_len"),
        _single_batch_value(inputs.get("full_context_lens")),
    )
    request_prefix_len = _single_batch_value(request_prefix_len)
    if request_prefix_len is None:
        return False

    return _to_python_int(request_prefix_len) >= attention_hit_len + active_suffix_len


def _is_chunked_prefill_execution(
    neuron_base_instance: "NeuronBaseForCausalLM",
    inputs: Dict[str, Any],
    *,
    is_fused_speculation: bool,
) -> bool:
    if is_fused_speculation:
        return False
    if getattr(neuron_base_instance.neuron_config, "enable_eagle_speculation", False):
        return False
    input_ids = inputs.get("input_ids")
    if (
        isinstance(input_ids, torch.Tensor)
        and input_ids.ndim >= 2
        and input_ids.shape[-1] > 1
    ):
        return True
    return _is_cached_chunked_prefill_continuation(inputs)


def _debug_hybrid_apc_owner_metadata_summary(owner: Any) -> str:
    if owner is None:
        return "None"
    parts = [type(owner).__name__]
    for attr in (
        "_qwen36_vllm_request_ids",
        "_qwen36_vllm_cached_request_ids",
        "_qwen36_vllm_prefill_completion_state",
        "_qwen36_vllm_hybrid_apc_request_records",
        "_qwen36_vllm_hybrid_apc_metadata_by_request_id",
    ):
        value = getattr(owner, attr, None)
        if value is None:
            continue
        if isinstance(value, dict):
            parts.append(f"{attr}=dict[{len(value)}]")
        elif isinstance(value, (list, tuple)):
            parts.append(f"{attr}=seq[{len(value)}]")
        elif isinstance(value, torch.Tensor):
            parts.append(f"{attr}=tensor{tuple(value.shape)}")
        else:
            parts.append(f"{attr}={type(value).__name__}")
    return " ".join(parts)


def _format_token_id(value: int) -> str:
    if value < 0:
        return str(value)
    return f"{value} (0x{value & 0xFFFFFFFF:08x})"


def _model_vocab_size(neuron_base_instance: "NeuronBaseForCausalLM") -> int | None:
    for owner in (
        neuron_base_instance,
        getattr(neuron_base_instance, "config", None),
        getattr(neuron_base_instance, "model", None),
        getattr(getattr(neuron_base_instance, "model", None), "config", None),
    ):
        vocab_size = getattr(owner, "vocab_size", None)
        if vocab_size is not None:
            try:
                return int(vocab_size)
            except (TypeError, ValueError):
                return None
    return None


def _summarize_tensor_minmax(value: Any) -> str:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return "empty"
    try:
        flat = value.detach().reshape(-1)
        return f"{int(flat.min().item())}:{int(flat.max().item())}"
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"


def _validate_token_generation_input_ids(
    neuron_base_instance: "NeuronBaseForCausalLM",
    model_to_execute: "ModelWrapper",
    input_dict: Dict[str, Any],
) -> None:
    if getattr(model_to_execute, "tag", None) != "token_generation_model":
        return
    input_ids = input_dict.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        return
    if input_ids.numel() == 0:
        raise ValueError("Token generation input_ids must be non-empty")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "Token generation input_ids must be int32 or int64 before Neuron "
            f"execution, got {input_ids.dtype}"
        )

    min_id = int(input_ids.min().item())
    max_id = int(input_ids.max().item())
    vocab_size = _model_vocab_size(neuron_base_instance)
    invalid_id = None
    reason = None
    if min_id < 0:
        invalid_id = min_id
        reason = "negative"
    elif vocab_size is not None and max_id >= vocab_size:
        invalid_id = max_id
        reason = f"out-of-vocab for vocab_size={vocab_size}"
    if invalid_id is None:
        return

    request_ids = getattr(neuron_base_instance, "_qwen36_vllm_request_ids", None)
    if request_ids is None:
        request_ids = input_dict.get("request_ids", input_dict.get("request_id"))
    raise ValueError(
        "Token generation input_ids contract violated before Neuron execution: "
        f"{reason}; token_id={_format_token_id(invalid_id)}; "
        f"request_ids={request_ids}; "
        f"input_shape={tuple(input_ids.shape)} dtype={input_ids.dtype}; "
        f"position_minmax={_summarize_tensor_minmax(input_dict.get('position_ids'))}; "
        f"slot_minmax={_summarize_tensor_minmax(input_dict.get('slot_mapping'))}; "
        f"block_minmax={_summarize_tensor_minmax(input_dict.get('block_table'))}; "
        f"num_queries={_summarize_tensor_minmax(input_dict.get('num_queries'))}; "
        "computed_context_lens="
        f"{_summarize_tensor_minmax(input_dict.get('computed_context_lens'))}"
    )


def _with_disabled_hybrid_apc_controls(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(input_dict)
    _zero_mask_if_present(output, "hybrid_restore_mask")
    _zero_mask_if_present(output, "hybrid_commit_mask")
    return output


class AsyncTensorWrapper:
    """
    Wrapper class for tensors from models executed with async runtime.

    Attributes:

    1. `async_result`: A 2d list of tensors representing a collection of outputs from all tp ranks.
        **Exception:** If on_cpu=True, ranked_tensor is a 2d list of tensors containing  to be concatenated together
    2. `batch_padded`: A boolean indicating if the result has been padded along batch dimension.
    3. `on_cpu`: A boolean indicating if the ranked tensors have been synced to CPU.

    All 4 possiblilities are to be handled:
    1. `(batch_padded=False, on_cpu=False)` which implies request_batch_size = compiled_batch_size
    2. `(batch_padded=True, on_cpu=False)` which implies request_batch_size < compiled_batch_size
    3. `(batch_padded=True, on_cpu=True)` which implies request_batch_size > compiled_batch_size and request_batch_size % compiled_batch_size != 0
    4. `(batch_padded=False, on_cpu=True)` which implies request_batch_size > compiled_batch_size and request_batch_size % compiled_batch_size == 0
    """

    def __init__(self, async_result: List[List[torch.Tensor]], batch_padded: bool, on_cpu: bool):
        self.async_result = async_result
        self.batch_padded = batch_padded
        self.on_cpu = on_cpu

        if self.on_cpu:
            assert not is_ranked_io(
                self.async_result
            ), f"Initialized with {on_cpu=} but found that async_result is still on Neuron."
        else:
            assert is_ranked_io(
                self.async_result
            ), f"Initialized with {on_cpu=} but found that async_result is still on CPU."

    def get_ranked_tensor(self):
        assert not self.on_cpu, "Can't get ranked tensor if async_result is already on CPU."
        return self.async_result

    def sync_async_result_to_cpu(
        self, seq_ids: torch.Tensor, is_fused_speculation: bool = False, early_exit: bool = False, is_prefix_caching: bool = False
    ):
        if not self.on_cpu:  # cases 1 and 2
            synced_result = get_async_output(self.async_result)
        else:  # cases 3 and 4
            if is_fused_speculation:
                synced_result = [torch.cat(x, dim=0) for x in zip(*self.async_result)]
            else:
                synced_result = torch.cat([x[0] for x in self.async_result], dim=0)
        if early_exit:  # used for discarding results
            return
        # handle unpadding based on supplied seq_ids
        batch_size = seq_ids.shape[0]
        seq_ids = seq_ids.reshape(batch_size)  # make sure it's 1d tensor for index_select
        if is_prefix_caching:
            seq_ids = torch.arange(batch_size)
        if isinstance(synced_result, torch.Tensor):
            return torch.index_select(synced_result, 0, seq_ids)

        index_select = lambda x: torch.index_select(x, 0, seq_ids)  # noqa: E731
        try:
            return list(map(index_select, synced_result))
        except Exception as e:
            raise type(e)(f"Detected failure case: {str(e)}, tensor to select {synced_result}, seq_ids {seq_ids}") from e


def execute_model_prefix_caching(
    neuron_base_instance: "NeuronBaseForCausalLM",
    model_to_execute: "ModelWrapper",
    input_dict: Dict[str, Any],
    pad_type: str = "first_fit",
) -> Tuple[AsyncTensorWrapper, bool]:
    original_input_dict = input_dict
    hybrid_apc_owner = _select_hybrid_apc_owner(
        neuron_base_instance,
        model_to_execute,
        input_dict,
    )
    if os.environ.get("QWEN36_HYBRID_APC_DEBUG") == "1":
        print(
            "[hybrid_apc_debug] async-owner "
            f"base_enabled={_is_hybrid_apc_enabled(neuron_base_instance)} "
            f"wrapper_enabled={_is_hybrid_apc_enabled(model_to_execute)} "
            f"owner_type={type(hybrid_apc_owner).__name__} "
            f"tag={getattr(model_to_execute, 'tag', None)}",
            flush=True,
        )
    try:
        is_context_encoding = _is_context_encoding_execution(
            neuron_base_instance,
            model_to_execute,
            input_dict,
        )
        if is_context_encoding:
            input_dict = _with_hybrid_apc_owner_metadata(
                input_dict,
                hybrid_apc_owner,
            )
            input_dict = prepare_hybrid_apc_request_for_execution(
                hybrid_apc_owner,
                input_dict,
            )
            prepared_bridge = input_dict.get("_hybrid_apc_bridge")
            if prepared_bridge is not None:
                for bridge_owner in (
                    neuron_base_instance,
                    model_to_execute,
                    getattr(neuron_base_instance, "context_encoding_model", None),
                    getattr(neuron_base_instance, "token_generation_model", None),
                ):
                    if bridge_owner is not None:
                        setattr(
                            bridge_owner,
                            "_hybrid_apc_last_bridge",
                            prepared_bridge,
                        )
            if os.environ.get("QWEN36_HYBRID_APC_DEBUG") == "1":
                print(
                    "[hybrid_apc_debug] async-prepared-return "
                    f"has_bridge={'_hybrid_apc_bridge' in input_dict} "
                    f"has_prepared={'_hybrid_apc_prepared' in input_dict}",
                    flush=True,
                )
            for lifecycle_key in ("_hybrid_apc_bridge", "_hybrid_apc_prepared"):
                if lifecycle_key in input_dict:
                    original_input_dict[lifecycle_key] = input_dict[lifecycle_key]
            if "_hybrid_apc_prepared" in input_dict:
                setattr(
                    neuron_base_instance,
                    "_hybrid_apc_pending_input_dict",
                    input_dict,
                )
        else:
            input_dict = _with_disabled_hybrid_apc_controls(input_dict)
        if "num_queries" not in input_dict:
            full_context_lens = input_dict["full_context_lens"]
            computed_context_lens = input_dict["computed_context_lens"]
            num_queries = full_context_lens - computed_context_lens
            input_dict["num_queries"] = num_queries

        if (
            not neuron_base_instance.neuron_config.enable_fused_speculation
            and not neuron_base_instance.neuron_config.enable_eagle_speculation
        ):
            if is_context_encoding:
                hybrid_apc_args = prepare_hybrid_apc_model_inputs(
                    hybrid_apc_owner, input_dict
                )
            else:
                _validate_token_generation_input_ids(
                    neuron_base_instance,
                    model_to_execute,
                    input_dict,
                )
                hybrid_apc_args = prepare_disabled_hybrid_apc_model_inputs(
                    hybrid_apc_owner, input_dict
                )
            return model_to_execute(
                input_dict["input_ids"],
                input_dict["attention_mask"],
                input_dict["position_ids"],
                input_dict["seq_ids"],
                input_dict["sampling_params"],
                torch.empty(0),  # prev_hidden
                input_dict["adapter_ids"],
                torch.empty(0),  # accepted_indices
                torch.empty(0),  # current_length
                torch.empty(0),  # medusa_mask
                torch.empty(0),  # scatter_index
                input_dict["slot_mapping"],
                input_dict["block_table"],
                input_dict["num_queries"],
                input_dict["computed_context_lens"],
                *hybrid_apc_args,
                pad_type=pad_type
            ), model_to_execute.is_neuron()
        elif neuron_base_instance.neuron_config.enable_eagle_speculation:
            return model_to_execute(
                input_dict["input_ids"],
                input_dict["attention_mask"],
                input_dict["position_ids"],
                input_dict["seq_ids"],
                input_dict["sampling_params"],
                torch.empty(0),  # prev_hidden
                input_dict["adapter_ids"],
                input_dict["slot_mapping"],
                input_dict["block_table"],
                input_dict["num_queries"],
                input_dict["computed_context_lens"],
                torch.empty(0),  # target_input_ids
                torch.empty(0),  # target_attention_mask
                torch.empty(0),  # target_position_ids
                torch.empty(0),  # target_slot_mapping
                torch.empty(0),  # target_active_block_table
                pad_type=pad_type
            ), model_to_execute.is_neuron()
        else:
            raise NotImplementedError("Non-EAGLE fused speculation with prefix caching does not support async mode.")
    except Exception:
        cancel_hybrid_apc_request(original_input_dict)
        raise


def execute_model(
    neuron_base_instance: "NeuronBaseForCausalLM",
    model_to_execute: "ModelWrapper",
    input_dict: Dict[str, Any],
    hits_bucket_boundary: bool = False,
) -> Tuple[AsyncTensorWrapper, bool]:
    pad_type = "first_fit" if not hits_bucket_boundary else "second_fit"

    if neuron_base_instance.neuron_config.is_prefix_caching:
        return execute_model_prefix_caching(neuron_base_instance,
                                            model_to_execute,
                                            input_dict,
                                            pad_type)

    ordered_tuple_inputs = neuron_base_instance._convert_input_dict_to_ordered_tuple(input_dict)
    return model_to_execute(*ordered_tuple_inputs, pad_type=pad_type), model_to_execute.is_neuron()


def get_async_output(ranked_async_tensor: Any, clone: bool = False):
    if not is_ranked_io(ranked_async_tensor):
        return ranked_async_tensor

    maybe_clone = lambda x: x.clone().detach() if clone else x  # noqa: E731
    return [maybe_clone(async_tensor.cpu()) for async_tensor in ranked_async_tensor[0]]


def is_ranked_io(input_ids: Any):
    # make sure the contents are List[List[torch.Tensor]]
    # and that tensor is a privateuseone device tensor (neuron)
    return (
        isinstance(input_ids, list)
        and isinstance(input_ids[0], list)
        and isinstance(input_ids[0][0], torch.Tensor)
        and input_ids[0][0].device.type == "privateuseone"
    )


def within_bounds(inputs: Dict[str, Any], max_length: int, generation_length: int):
    return (max_length - inputs["position_ids"].max().item()) > generation_length * 2


def will_hit_bucket_boundary(known_length: int, buckets: Union[List[int], List[Tuple[int, int]]], max_num_tokens_generated=1):
    hits_bucket_boundary = False

    for bucket in buckets:
        # If buckets are 2D, use last index.
        if not isinstance(bucket, int):
            bucket = bucket[-1]
        # we do max_num_tokens_generated * 2 because we speculatively execute one step
        # ahead, before knowing the results of the current neff execution. In the worst case
        # both neffs will have full matching tokens, which is problematic near the bucket boundary
        # therefore, we will execute at a higher bucket in such cases.
        if known_length < bucket and (known_length + max_num_tokens_generated * 2) >= bucket:
            hits_bucket_boundary = True
            return hits_bucket_boundary

    return hits_bucket_boundary


def causal_lm_async_execution(
    neuron_base_instance: "NeuronBaseForCausalLM",
    inputs: Dict[str, Any],
    is_fused_speculation: bool = False,
):
    is_prefix_caching = neuron_base_instance.neuron_config.is_prefix_caching

    # PREFILL STAGE:
    is_prefill = neuron_base_instance._is_prefill(inputs["position_ids"])
    prefill_probe_inputs = inputs
    if is_prefix_caching:
        prefill_probe_inputs = _with_hybrid_apc_candidate_owner_metadata(
            inputs,
            neuron_base_instance,
            getattr(neuron_base_instance, "context_encoding_model", None),
            getattr(neuron_base_instance, "token_generation_model", None),
        )
    probe_is_chunked_prefill = (
        is_prefix_caching
        and _is_chunked_prefill_execution(
            neuron_base_instance,
            prefill_probe_inputs,
            is_fused_speculation=is_fused_speculation,
        )
    )
    if (
        is_prefix_caching
        and not is_prefill
        and os.environ.get("QWEN36_HYBRID_APC_DEBUG") == "1"
    ):
        input_ids = inputs.get("input_ids")
        records = prefill_probe_inputs.get("hybrid_request_records")
        print(
            "[hybrid_apc_debug] prefill-route-probe "
            f"input_shape={tuple(input_ids.shape) if isinstance(input_ids, torch.Tensor) else None} "
            f"probe_is_chunked_prefill={probe_is_chunked_prefill} "
            f"probe_keys={sorted(k for k in prefill_probe_inputs if k.startswith('hybrid_') or k in ('request_prefix_len', 'active_suffix_len', 'vllm_attention_hit_len'))} "
            f"records_len={len(records) if isinstance(records, tuple) else None} "
            f"base={_debug_hybrid_apc_owner_metadata_summary(neuron_base_instance)} "
            f"context={_debug_hybrid_apc_owner_metadata_summary(getattr(neuron_base_instance, 'context_encoding_model', None))} "
            f"token={_debug_hybrid_apc_owner_metadata_summary(getattr(neuron_base_instance, 'token_generation_model', None))}",
            flush=True,
        )
    if (
        is_prefix_caching
        and not is_prefill
        and probe_is_chunked_prefill
    ):
        is_prefill = True
        inputs = prefill_probe_inputs
    elif is_prefix_caching and is_prefill:
        inputs = prefill_probe_inputs
    neuron_base_instance.async_should_stop = False
    prefill_outputs = None
    is_run_on_neuron = None
    if is_prefill:
        try:
            timing_enabled = os.environ.get("QWEN36_PREFILL_TIMING") == "1"
            execute_start = time.perf_counter() if timing_enabled else None
            prefill_outputs, is_run_on_neuron = execute_model(
                neuron_base_instance, neuron_base_instance.context_encoding_model, inputs
            )
            if timing_enabled and execute_start is not None:
                input_ids = inputs.get("input_ids")
                position_ids = inputs.get("position_ids")
                computed_context_lens = inputs.get("computed_context_lens")
                num_queries = inputs.get("num_queries")
                print(
                    "[qwen36_perf] async_execute_model "
                    f"elapsed_ms={(time.perf_counter() - execute_start) * 1000.0:.3f} "
                    f"is_run_on_neuron={is_run_on_neuron} "
                    f"input_shape={tuple(input_ids.shape) if isinstance(input_ids, torch.Tensor) else None} "
                    f"position_shape={tuple(position_ids.shape) if isinstance(position_ids, torch.Tensor) else None} "
                    f"num_queries={num_queries.reshape(-1).tolist() if isinstance(num_queries, torch.Tensor) and num_queries.numel() else []} "
                    f"computed={computed_context_lens.reshape(-1).tolist() if isinstance(computed_context_lens, torch.Tensor) and computed_context_lens.numel() else []} "
                    f"request_ids={_async_request_ids_signature(neuron_base_instance)}",
                    flush=True,
                )

            # Sequence IDs from vLLM will be in sorted order, but the maximum range of sequence IDs is
            # not [0, num_requested_prefills] but [0, max_num_seqs]. To prevent out-of-bound accesses,
            # we convert the sequence IDs to their argsorted values.
            _seq_ids = torch.argsort(inputs["seq_ids"])

            sync_start = time.perf_counter() if timing_enabled else None
            outputs = prefill_outputs.sync_async_result_to_cpu(
                _seq_ids, is_fused_speculation=is_fused_speculation, is_prefix_caching=is_prefix_caching
            )
            if timing_enabled and sync_start is not None:
                print(
                    "[qwen36_perf] async_sync_result "
                    f"elapsed_ms={(time.perf_counter() - sync_start) * 1000.0:.3f} "
                    f"is_run_on_neuron={is_run_on_neuron} "
                    f"request_ids={_async_request_ids_signature(neuron_base_instance)}",
                    flush=True,
                )
            pending_hybrid_apc = getattr(
                neuron_base_instance,
                "_hybrid_apc_pending_input_dict",
                None,
            )
            neuron_base_instance._hybrid_apc_pending_input_dict = None
            finish_hybrid_apc_request(
                pending_hybrid_apc if pending_hybrid_apc is not None else inputs
            )
        except Exception:
            pending_hybrid_apc = getattr(
                neuron_base_instance,
                "_hybrid_apc_pending_input_dict",
                None,
            )
            neuron_base_instance._hybrid_apc_pending_input_dict = None
            cancel_hybrid_apc_request(
                pending_hybrid_apc if pending_hybrid_apc is not None else inputs
            )
            raise

        # clean up async state
        neuron_base_instance.prior_outputs = None
        neuron_base_instance.prior_seq_ids = None
        neuron_base_instance.prior_request_ids = None

        return outputs, is_run_on_neuron

    # GENERATION STAGE:
    generation_model = (
        neuron_base_instance.token_generation_model
        if not is_fused_speculation
        else neuron_base_instance.fused_spec_model
    )
    generation_length = (
        1 if not is_fused_speculation else neuron_base_instance.neuron_config.speculation_length
    )
    known_seqlen = inputs["attention_mask"].shape[1]
    hits_bucket_boundary = will_hit_bucket_boundary(
        known_seqlen,
        buckets=generation_model.neuron_config.buckets,
        max_num_tokens_generated=generation_length,
    )
    request_ids_signature = _async_request_ids_signature(neuron_base_instance)
    prior_request_ids = getattr(neuron_base_instance, "prior_request_ids", None)
    request_ids_changed = (
        request_ids_signature is not None
        and prior_request_ids is not None
        and request_ids_signature != prior_request_ids
    )
    force_sync_for_hybrid_apc = _is_hybrid_apc_enabled(neuron_base_instance)

    stay_in_sync_mode = (
        not torch.equal(neuron_base_instance.prior_seq_ids, inputs["seq_ids"])
        or hits_bucket_boundary
        or request_ids_changed
        or force_sync_for_hybrid_apc
    )
    start_async = not stay_in_sync_mode and neuron_base_instance.prior_outputs is None
    continue_async = not stay_in_sync_mode and not start_async

    if stay_in_sync_mode:
        # reset async state
        neuron_base_instance.prior_outputs = None
        neuron_base_instance.prior_seq_ids = None
        neuron_base_instance.prior_request_ids = None

    if stay_in_sync_mode or start_async:
        next_outputs, is_run_on_neuron = execute_model(
            neuron_base_instance,
            generation_model,
            inputs,
            hits_bucket_boundary=hits_bucket_boundary,
        )
        if start_async:
            neuron_base_instance.prior_outputs = next_outputs
            neuron_base_instance.prior_seq_ids = inputs["seq_ids"]
            neuron_base_instance.prior_request_ids = request_ids_signature

    if start_async or continue_async:
        if within_bounds(inputs, neuron_base_instance.neuron_config.seq_len, generation_length):
            next_outputs = neuron_base_instance.prior_outputs
            inputs["input_ids"] = next_outputs.get_ranked_tensor()
            if neuron_base_instance.next_cpu_inputs is not None:
                for key in neuron_base_instance.next_cpu_inputs:
                    inputs[key] = neuron_base_instance.next_cpu_inputs[key]
            elif not is_fused_speculation:
                raise RuntimeError(
                    "Expected next_cpu_inputs to be generated for a non fused_spec model."
                )
            next_outputs, is_run_on_neuron = execute_model(
                neuron_base_instance, generation_model, inputs
            )
        else:
            if neuron_base_instance.prior_outputs is not None:
                outputs = neuron_base_instance.prior_outputs.sync_async_result_to_cpu(
                    inputs["seq_ids"], is_fused_speculation=is_fused_speculation, is_prefix_caching=is_prefix_caching
                )
                neuron_base_instance.prior_outputs = None
                neuron_base_instance.prior_seq_ids = None
                neuron_base_instance.prior_request_ids = None
                neuron_base_instance.async_should_stop = True
            else:
                raise RuntimeError(
                    "The stopping criteria for fused async should have been triggered, but it wasn't."
                )

            return outputs, True  # assume async mode only runs on neuron

    # output to be returned
    outputs: AsyncTensorWrapper = neuron_base_instance.prior_outputs if not stay_in_sync_mode else next_outputs
    outputs = outputs.sync_async_result_to_cpu(
        inputs["seq_ids"], is_fused_speculation=is_fused_speculation, is_prefix_caching=is_prefix_caching
    )

    if stay_in_sync_mode:
        # make sure prior outputs is not set
        neuron_base_instance.prior_outputs = None
        neuron_base_instance.prior_seq_ids = None
        neuron_base_instance.prior_request_ids = None
        return outputs, is_run_on_neuron

    # next step
    neuron_base_instance.prior_outputs = next_outputs
    neuron_base_instance.prior_seq_ids = inputs["seq_ids"]
    neuron_base_instance.prior_request_ids = request_ids_signature

    return outputs, is_run_on_neuron
