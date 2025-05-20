import torch
import torch.distributed
from neuronx_distributed.parallel_layers.mappings import gather_from_sequence_parallel_region

from neuronx_distributed_inference.utils.distributed import get_tp_group
from neuronx_distributed_inference.modules.custom_calls import neuron_cumsum


def seq_parallel_slice_last_token(
    hidden_states, position_ids, sequence_dimension, batch_size, hidden_size, num_queries, neuron_config, config
):
    if config.neuron_config.padding_side == "left":
        index = torch.tensor([hidden_states.shape[1] - 1], device=hidden_states.device).to(torch.int32)
    elif config.neuron_config.is_chunked_prefill:
        index = neuron_cumsum(num_queries.reshape(1, -1).float()).int() - 1
        index = index.reshape(1, -1, 1)
        index = index.expand(batch_size, -1, hidden_size).to(torch.int32)
    else:
        index = torch.max(position_ids, dim=1, keepdim=True).indices.to(torch.int32)
    sharded_seq_len = hidden_states.shape[1]
    if hasattr(neuron_config, "tile_cc") and neuron_config.tile_cc:
        # now we have round robin of seqlen_per_tile_per_core across the ranks
        seqlen_per_tile_per_core = (
            sharded_seq_len // neuron_config.cc_pipeline_tiling_factor
        )  # FIXME.
        seqlen_per_tile = neuron_config.tp_degree * seqlen_per_tile_per_core
        tile_index = torch.divide(index, seqlen_per_tile, rounding_mode="floor").to(torch.int32)
        tile_offset_within_rank = torch.mul(tile_index, seqlen_per_tile_per_core).to(torch.int32)
        pos_within_tile_within_rank = torch.remainder(index, seqlen_per_tile_per_core)
        index_local = torch.add(tile_offset_within_rank, pos_within_tile_within_rank)

        # now we have round robin of seqlen_per_tile_per_core across the ranks
        index_post_shard = torch.divide(index, seqlen_per_tile_per_core, rounding_mode="floor").to(
            torch.int32
        )
        index_post_shard = torch.remainder(index_post_shard, neuron_config.tp_degree)
    else:
        index_local = torch.remainder(index, sharded_seq_len)
        index_post_shard = torch.divide(index, sharded_seq_len, rounding_mode="floor").to(
            torch.int32
        )
    if config.neuron_config.is_chunked_prefill:
        index_local = index_local.expand(batch_size, -1, hidden_size)
        index_post_shard = index_post_shard.expand(batch_size, -1, hidden_size)
        index_post_shard = index_post_shard.unsqueeze(1).expand(-1, neuron_config.tp_degree, -1, -1)
    else:
        index_local = index_local.unsqueeze(1).expand(batch_size, 1, hidden_size)
        index_post_shard = index_post_shard.unsqueeze(1).expand(batch_size, 1, hidden_size)
    hidden_states = torch.gather(hidden_states, dim=1, index=index_local)
    if config.neuron_config.is_chunked_prefill:
        hidden_states = hidden_states.unsqueeze(1)
    hidden_states = gather_from_sequence_parallel_region(
        hidden_states, sequence_dimension, process_group=get_tp_group(config)
    )
    hidden_states = torch.gather(hidden_states, dim=1, index=index_post_shard)
    if config.neuron_config.is_chunked_prefill:
        return hidden_states[:, 0, :, :]
    return hidden_states
