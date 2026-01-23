# Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.

import torch


def patched_get_last_kv_window(window_size, position_ids, latest_k, latest_v, windowed_context_encoding_window_idx=-1, spec_len=0):
    """
    Replaces https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/modules/attention/utils.py#L634
    to convert the index tensor in torch.gather to a LongTensor. Otherwise, the function will error out.
    """
    batch_size, num_head, _, head_dim = latest_k.shape
    latest_pos = torch.amax(position_ids, dim=1)
    end_idx = (latest_pos + 1).clamp(min=window_size)
    start_idx = (end_idx - window_size).clamp(min=0)
    orig_indices = start_idx[:, None] + torch.arange(window_size)

    # Calculate per-batch left shifts
    left_shifts = (window_size - (end_idx % window_size)) % window_size
    base = torch.arange(window_size).expand(batch_size, window_size)
    shifted_idx = (base + left_shifts[:, None]) % window_size

    # Determine per-batch shifted gather indices
    gather_idx = torch.gather(orig_indices, dim=1, index=shifted_idx.long())
    gather_idx = gather_idx[:, None, :, None].expand(batch_size, num_head, window_size, head_dim).to(device=latest_k.device)

    # Gather to create non-physically contiguous KV cache
    latest_k = torch.gather(latest_k, dim=2, index=gather_idx.long())
    latest_v = torch.gather(latest_v, dim=2, index=gather_idx.long())
    return latest_k, latest_v


def apply_patch() -> None:
    import neuronx_distributed_inference.modules.attention.utils as u
    u.get_last_kv_window = patched_get_last_kv_window
