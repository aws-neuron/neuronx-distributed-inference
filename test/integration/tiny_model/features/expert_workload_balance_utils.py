import json
import torch
import math

def postprocess_router_logits(neuron_output, ep_degree, n_routed_experts, seq_len, block_size, num_experts_per_tok):
    op, logits = neuron_output
    all_conditions=[]
    num_block_per_core_per_layer = torch.zeros(logits.shape[0], ep_degree)
    for i in range(logits.shape[0]):
        conditions, blocks_per_expert_output = process_expert_routing(logits[i], n_routed_experts, seq_len, block_size, num_experts_per_tok)
        all_conditions.append(conditions)
        num_expert = blocks_per_expert_output.shape[0]
        for j in range(ep_degree):
            num_block_per_core_per_layer[i,j] = torch.sum(blocks_per_expert_output[j:j + int(num_expert / ep_degree)])

def process_expert_routing(router_logits, E, T, B, TOPK):
    N = math.ceil((T * TOPK - (E - 1)) / B) + E - 1
    N = math.ceil(N / 2) * 2
    # Get top-k expert indices
    _, expert_index = torch.topk(router_logits, TOPK)
    expert_index = expert_index.detach().to(dtype=torch.long)
    
    router = expert_index

    # Create one-hot encoding
    one_hot = torch.arange(E, device=router.device)
    token_experts = torch.zeros((T, E), device=router.device)
    for i in range(TOPK):
        token_experts += (router[:, i].unsqueeze(1) == one_hot.unsqueeze(0))

    # Calculate blocks per expert
    blocks_per_expert = torch.ceil(token_experts.sum(0)/B).to(torch.int32)
    n_padding_block = N - blocks_per_expert.sum()
    
    blocks_per_expert_output = blocks_per_expert.clone()
    blocks_per_expert[E-1] += n_padding_block

    cumulative_blocks_per_expert = torch.cumsum(blocks_per_expert, dim=0)

    token_position_by_id_and_expert = torch.cumsum(token_experts, dim=0)
    expert_block_offsets = cumulative_blocks_per_expert * B
    token_position_by_id_and_expert[:, 1:] += expert_block_offsets[:-1]

    token_position_by_id_and_expert = torch.where(token_experts.bool(), 
                                                token_position_by_id_and_expert, 
                                                torch.zeros_like(token_position_by_id_and_expert))
    token_position_by_id_and_expert = token_position_by_id_and_expert.to(torch.int32)

    token_position_to_id = torch.full((int(N * B + 1),), -1, 
                                    dtype=torch.int32, 
                                    device=router.device)

    tokens_ids = torch.arange(T, device=router.device, dtype=torch.int32)
    token_position_to_id[token_position_by_id_and_expert] = tokens_ids.unsqueeze(1)
    token_position_to_id = token_position_to_id[1:]

    # Reshape into blocks
    blocks = token_position_to_id.view(N, B)

    conditions = torch.any(blocks != -1, dim=1).to(torch.int32)

    return conditions, blocks_per_expert_output

def save_core_blocks_to_json(tensor_data, filename='core_blocks.json', metadata=None):
    """
    Save the number of blocks computed by each core in each layer to a JSON file.
    
    Args:
        tensor_data: numpy array with shape [num_layer, num_core] containing block counts
        filename: Output JSON filename
        metadata: Optional metadata dictionary
    """
    num_layers, num_cores = tensor_data.shape
    
    data_dict = {
        "metadata": {
            "num_layers": num_layers,
            "num_cores": num_cores,
            **(metadata or {})
        },
        "data": {
            f"layer_{layer}": {
                f"core_{core}": int(tensor_data[layer,core])
                for core in range(num_cores)
            }
            for layer in range(num_layers)
        }
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(data_dict, f, indent=4)
        print(f"Successfully wrote data to {filename}")
    except Exception as e:
        print(f"Error writing to file: {e}")