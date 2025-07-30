import torch


class DynamicTokenTree:
    def __init__(self, dynamic_tree_params):
        """Initialize TokenTree with a tree configuration.

        Args:
            dynamic_tree_params (list): Parameters for the dynamic tree
        """
        try:
            # dynamic_tree_params has format [steps (max_depth), branching factor, # token per step, # verification token]
            self.dynamic_tree_params = dynamic_tree_params
            self.step = dynamic_tree_params['step']
            self.branching_factor = dynamic_tree_params['branching_factor']
            self.num_inputs = dynamic_tree_params['num_inputs']
            self.num_verification_token = dynamic_tree_params['num_verification_token']
            self.spec_len = self.get_spec_len()
        except Exception as e:
            raise ValueError(f"Error initializing DynamicTokenTree: {str(e)}")

    def get_spec_len(self):
        return 1 + self.branching_factor + (self.step - 1) * self.num_inputs * self.branching_factor

    @staticmethod
    def update_attention_mask(step, input_position_ids, prev_attn_mask, branching_factor, num_inputs, position_ids_offset):
        device = prev_attn_mask.device
        bs, _, spec_len, _ = prev_attn_mask.shape
        parent_position_ids = input_position_ids - position_ids_offset
        if step == 0:
            new_position_start_ids = 1
            num_new_nodes = branching_factor
        else:
            new_position_start_ids = 1 + branching_factor + (step - 1) * branching_factor * num_inputs
            num_new_nodes = branching_factor * num_inputs

        prev_attn_mask = DynamicTokenTree.set_eye_at_indices(prev_attn_mask, parent_position_ids)

        parent_mask = torch.gather(prev_attn_mask, 2, parent_position_ids.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, spec_len))

        repeated_mask = parent_mask.repeat_interleave(branching_factor, dim=2)

        new_positions = torch.arange(new_position_start_ids, new_position_start_ids + num_new_nodes, device=device)
        new_positions = new_positions.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(bs, 1, -1, spec_len)
        updated_attn_mask = prev_attn_mask.clone()

        updated_attn_mask[:, :, new_position_start_ids:new_position_start_ids + num_new_nodes, :] = repeated_mask[:, :, :num_new_nodes, :]

        return updated_attn_mask

    @staticmethod
    def set_eye_at_indices(b, a):
        bs, _, n, _ = b.shape
        device = b.device
        m = a.shape[-1]
        mask = torch.zeros_like(b, dtype=torch.bool).squeeze(1).view(bs, -1)

        rows = a.unsqueeze(-1).expand(-1, -1, m)
        cols = a.unsqueeze(1).expand(-1, m, -1)
        linear_indices = rows * n + cols
        linear_indices_flat = linear_indices.view(bs, -1)
        updates = torch.ones_like(linear_indices_flat, dtype=torch.bool, device=device)

        mask = torch.scatter(mask, 1, linear_indices_flat, updates)

        eye = torch.eye(n, device=device).unsqueeze(0).unsqueeze(0).expand(bs, 1, -1, -1)
        mask = mask.unsqueeze(1).view(bs, 1, n, n)
        b = torch.where(mask, eye, b)

        return b

    @staticmethod
    def get_draft_attention_mask(attn_mask, input_position_ids, position_ids_offset):
        bs, _, spec_len, _ = attn_mask.shape

        selected_position_ids = input_position_ids - position_ids_offset

        gathered_mask = torch.gather(attn_mask, 2, selected_position_ids.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, spec_len))

        return gathered_mask

    @staticmethod
    def get_draft_rotary_position_ids(attn_mask, input_position_ids, position_ids_offset):
        bs, _, spec_len, _ = attn_mask.shape
        selected_position_ids = input_position_ids - position_ids_offset
        gathered_mask = torch.gather(attn_mask, 2, selected_position_ids.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, spec_len))

        rotary_position_ids = gathered_mask.sum(dim=-1).squeeze(1)
        return rotary_position_ids

    @staticmethod
    def replace_with_indices(original_tensor, values_to_replace):
        device = original_tensor.device
        values_to_replace = values_to_replace.to(device)

        bs, n = original_tensor.shape
        _, k, m = values_to_replace.shape

        original_expanded = original_tensor.unsqueeze(1).unsqueeze(1).expand(-1, k, m, -1)
        values_expanded = values_to_replace.unsqueeze(-1).expand(-1, -1, -1, n)

        mask = (values_expanded == original_expanded)

        indices = torch.arange(n, device=device).expand_as(original_expanded)

        result = torch.where(mask, indices, torch.full_like(indices, n))

        result, _ = result.min(dim=-1)

        return result

    @staticmethod
    def get_path(attention_mask, verified_position_ids):
        device = attention_mask.device
        bs, _, spec_len, _ = attention_mask.shape
        num_verified = verified_position_ids.shape[1]

        trimmed_attn_mask = torch.gather(attention_mask, 2, verified_position_ids.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, spec_len))
        trimmed_attn_mask = trimmed_attn_mask.squeeze(1)

        col_indices = torch.arange(spec_len, device=device).expand_as(trimmed_attn_mask)

        masked_indices = col_indices * (trimmed_attn_mask == 1) + (trimmed_attn_mask != 1) * -1

        k = masked_indices.shape[-1]
        top_indices, _ = torch.topk(masked_indices.to(torch.float32), k)
        result = top_indices[:, :, :num_verified]

        valid_mask = result >= 0

        valid_len = valid_mask.sum(dim=-1)
        result = DynamicTokenTree.replace_with_indices(verified_position_ids, result)
        result = result * valid_mask + (~valid_mask) * num_verified

        k = result.shape[-1]
        result, _ = torch.topk(result.to(torch.float32), k, largest=False)
        fill_tensor = DynamicTokenTree.generate_and_fill_tensor(result, num_verified)

        indices = torch.arange(num_verified, device=device).unsqueeze(0).unsqueeze(0).expand(bs, -1, -1)

        fill_mask = indices < valid_len.unsqueeze(-1)
        result = torch.where(fill_mask, result, fill_tensor)
        return result

    @staticmethod
    def get_selected_path(all_path, selected_idx):
        bs, num_path, path_len = all_path.shape
        gather_index = selected_idx.unsqueeze(-1).expand(-1, -1, path_len)
        result = torch.gather(all_path, 1, gather_index).squeeze(1)
        return result

    @staticmethod
    def generate_and_fill_tensor(input_tensor, n):
        device = input_tensor.device
        input_tensor = torch.tensor(input_tensor, dtype=torch.long, device=device)
        shape = input_tensor.shape
        result = torch.arange(n, dtype=torch.long, device=device)
        result = result.expand(shape[0], shape[1], -1)

        mask = (result.unsqueeze(-1) == input_tensor.unsqueeze(-2)).any(dim=-1)
        result = torch.where(mask, torch.tensor(-1, device=device), result)

        k = result.shape[-1]
        result, _ = torch.topk(result.to(torch.float32), k, largest=False)

        return result

    @staticmethod
    def get_target_cache_scatter_index(path):
        device = path.device
        batch_size, num_path, path_length = path.shape

        path_scatter = torch.arange(path_length, device=device, dtype=path.dtype).expand_as(path)

        scatter_indices = torch.zeros_like(path)

        path_mask = torch.zeros_like(path, dtype=torch.bool)
        path_mask = torch.scatter(path_mask, 2, path, torch.ones_like(path, dtype=torch.bool))

        scatter_indices = torch.scatter(scatter_indices, 2, path, path_scatter)

        all_indices = torch.arange(path_length, device=device).expand_as(path)

        scatter_indices = torch.where(path_mask, scatter_indices, all_indices)

        return scatter_indices

    @staticmethod
    def get_draft_hidden(all_hidden, input_position_ids, position_ids_offset):

        bs, seq_len, hidden_dim = all_hidden.shape
        parent_position_ids = input_position_ids - position_ids_offset

        expanded_indices = parent_position_ids.unsqueeze(-1).expand(-1, -1, hidden_dim)

        draft_hidden = torch.gather(all_hidden, 1, expanded_indices)

        return draft_hidden

    @staticmethod
    def get_parent_position_ids(attention_mask, position_ids, position_ids_offset):
        device = attention_mask.device
        bs, _, spec_len, _ = attention_mask.shape
        child_position_ids = position_ids - position_ids_offset

        batch_indices = torch.arange(bs, device=device).unsqueeze(1).expand_as(position_ids)

        relevant_rows = attention_mask[batch_indices, 0, child_position_ids]

        rightmost_ones = torch.argmax(relevant_rows.flip(dims=[-1]), dim=-1)
        parent_position_ids = spec_len - 1 - rightmost_ones

        parent_position_ids = parent_position_ids + position_ids_offset

        return parent_position_ids

    @staticmethod
    def new_get_parent_position_ids(adj, position_ids, position_ids_offset):
        child_position_ids = position_ids - position_ids_offset
        # Use gather to select the relevant columns from adj
        parent_ids = torch.gather(adj, 2, child_position_ids.unsqueeze(1).expand(-1, adj.size(1), -1))

        # Get the argmax along the second dimension (dim=1)
        parent_ids = torch.argmax(parent_ids, dim=1)

        return parent_ids + position_ids_offset

    @staticmethod
    def update_adj_matrix(prev_adj_matrix, input_position_ids, new_position_id, position_ids_offset):
        device = prev_adj_matrix.device
        bs, spec_len, _ = prev_adj_matrix.shape

        parent_position_id = input_position_ids - position_ids_offset

        # Flatten prev_adj_matrix to (bs, spec_len*spec_len)
        flat_adj_matrix = prev_adj_matrix.clone().view(bs, -1)

        # Compute linear indices
        rows = parent_position_id.unsqueeze(-1).expand(-1, -1, new_position_id.size(1))
        cols = new_position_id.unsqueeze(1).expand(-1, parent_position_id.size(1), -1)
        linear_indices = rows * spec_len + cols

        # Flatten linear_indices
        linear_indices_flat = linear_indices.view(bs, -1)

        # Create updates tensor (all ones in this case)
        updates = torch.ones_like(linear_indices_flat, dtype=prev_adj_matrix.dtype, device=device)

        # Scatter updates into flat_adj_matrix
        # flat_adj_matrix.scatter_(1, linear_indices_flat, updates)
        flat_adj_matrix = torch.scatter(flat_adj_matrix, 1, linear_indices_flat, updates)

        # Reshape back to (bs, spec_len, spec_len)
        updated_adj_matrix = flat_adj_matrix.view(bs, spec_len, spec_len)

        return updated_adj_matrix

    @staticmethod
    def compute_cumulative_draft_prob(adj_matrix, node_values):
        device = adj_matrix.device
        bs, n = node_values.shape

        adj_matrix = adj_matrix.to(torch.float32)

        T = torch.eye(n, dtype=torch.float32, device=device).unsqueeze(0).expand(bs, -1, -1)
        A = adj_matrix.clone()

        for _ in range(int(torch.log2(torch.tensor(n, dtype=torch.float32)).item()) + 1):
            T = T + A
            A = torch.bmm(A, adj_matrix)

        log_values = torch.log(node_values.to(torch.float32) + 1e-10)

        accum_log = torch.bmm(T.transpose(1, 2), log_values.unsqueeze(2)).squeeze(2)

        accum_product = torch.exp(accum_log)

        return accum_product.to(torch.float32)

    @staticmethod
    def convert_logits_to_prob(logits):
        logits = logits.to(torch.float32)

        logits_sum = torch.sum(logits, dim=-1, keepdim=True)

        prob = logits / logits_sum
        return prob

    @staticmethod
    def get_generated_token_position_ids(step, branching_factor, num_inputs, bs):
        if step == 0:
            result = torch.arange(1, 1 + branching_factor)
        else:
            result = torch.arange(1 + branching_factor + (step - 1) * branching_factor * num_inputs,
                                  1 + branching_factor + step * branching_factor * num_inputs)

        result = result.unsqueeze(0).expand(bs, -1)

        return result

    @staticmethod
    def get_target_active_mask(matrix, input_indices, indices_offset):
        bs, _, spec_len, _ = matrix.shape
        indices = input_indices - indices_offset
        num_verified = indices.shape[1]

        row_indices = indices.unsqueeze(-1).expand(-1, -1, spec_len)
        col_indices = indices.unsqueeze(-2).expand(-1, num_verified, -1)

        matrix_gathered_rows = torch.gather(matrix, 2, row_indices.unsqueeze(1).expand(-1, 1, -1, -1))
        result = torch.gather(matrix_gathered_rows, 3, col_indices.unsqueeze(1).expand(-1, 1, -1, -1))

        return result

    @staticmethod
    def get_draft_input_ids(candidate_input_ids, input_position_ids, position_ids_offset):
        position_ids = input_position_ids - position_ids_offset
        result = torch.gather(candidate_input_ids, dim=1, index=position_ids)
        return result

    @staticmethod
    def get_comp(candidate_input_ids, target_tokens, paths):
        bs, num_path, path_len = paths.shape
        # Expand candidate_input_ids and target_tokens to match paths shape
        candidate_input_ids_exp = candidate_input_ids.unsqueeze(1).expand(-1, num_path, -1)
        target_tokens_exp = target_tokens.unsqueeze(1).expand(-1, num_path, -1)

        # Create an index for gathering
        batch_indices = torch.arange(bs).view(-1, 1, 1).expand(-1, num_path, path_len)
        path_indices = torch.arange(num_path).view(1, -1, 1).expand(bs, -1, path_len)

        batch_indices = batch_indices.to(paths.dtype)
        path_indices = path_indices.to(paths.dtype)
        # Gather the values
        candidate_input_ids_comp = candidate_input_ids_exp[batch_indices, path_indices, paths]
        target_tokens_comp = target_tokens_exp[batch_indices, path_indices, paths]

        return candidate_input_ids_comp, target_tokens_comp

    @staticmethod
    def update_attention_mask_for_leaf(attn_mask):
        # The input attn_mask has shape bs, spec_len,spec_len
        # we want to make sure the diagonal of the attn_mask is 1
        diagonal_mask = torch.eye(attn_mask.shape[-1], device=attn_mask.device).unsqueeze(0)

        # Expand the diagonal mask to match the batch size of attn_mask
        diagonal_mask = diagonal_mask.expand_as(attn_mask)

        # Use the logical OR operation to combine the original mask with the diagonal mask
        updated_mask = torch.logical_or(attn_mask, diagonal_mask)

        return updated_mask
