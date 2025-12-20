import torch


class HiddenStateRollingBuffer(torch.nn.Module):
    """
    Stores EAGLE hidden state in a rolling buffer with simple storage/access.

    The primary purpose of this class is to support batched asynchronous
    execution of EAGLE speculative decoding.

    When using EAGLE speculation, each decode step requires that the prior
    iteration hidden_state output  is used in the current iteration forward
    pass.

    During batched asynchronous execution, it is possible that a currently
    executing speculative decode step is interrupted because a new context
    encoding is scheduled to begin. This means that rather than storing a single
    hidden state, we must store at least 2. The extra hidden state allows us to
    restart the prior decode step by rolling the hidden_state back to the last
    iteration. To simplify the implementation of this, we simply store
    `buffer_length` steps which can be stored and retrieved on-demand.

    The hidden state retrieved for a particular position_id is the respective prev_hidden_state
    for EAGLE. This means that `set_state` should be called with the `next_position_ids` and `get_state`
    should be called with the current `position_ids`.

    In EAGLE speculation, `buffer_length` should be set to `k * 2` so that any
    speculated position is accessible within 2 iterations.
    """

    def __init__(
        self,
        max_batch_size: int,
        buffer_length: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
        inplace: bool = False,
        apply_seq_ids_mask: bool = False,
    ):
        super().__init__()
        self.max_batch_size = max_batch_size
        # pad to last batch position as garbage
        self.buffer_length = buffer_length
        self.inplace = inplace
        self.shape = (self.max_batch_size + 1, self.buffer_length, hidden_size)
        self.hidden_states = torch.nn.Parameter(
            torch.zeros(self.shape, dtype=dtype), requires_grad=False
        )
        self.apply_seq_ids_mask = apply_seq_ids_mask

    def set_state(
        self,
        seq_ids: torch.Tensor,  # shape: [batch_size, 1]
        position_ids: torch.Tensor,  # shape: [batch_size, n_active_tokens]
        hidden_state: torch.Tensor,  # shape: [batch_size, 1, hidden_size],
    ):
        seq_ids = seq_ids.reshape(seq_ids.shape[0])
        position_ids = position_ids.reshape(position_ids.shape[0])
        if self.apply_seq_ids_mask and seq_ids.shape[0] > 1:
            seq_ids_mask = torch.ge(seq_ids, torch.full_like(seq_ids, 0))
            pad_seq_ids = torch.full_like(seq_ids, self.max_batch_size)
            seq_ids = torch.where(seq_ids_mask, seq_ids, pad_seq_ids)
        hidden_state = hidden_state.squeeze(1)
        index = (seq_ids, position_ids % self.buffer_length)

        result = torch.index_put(self.hidden_states, index, hidden_state)
        if self.inplace:
            self.hidden_states.data = result
        return result

    def get_state(
        self,
        seq_ids: torch.Tensor,  # shape: [batch_size, 1]
        position_ids: torch.Tensor,  # shape: [batch_size, n_active_tokens]
    ):
        seq_ids = seq_ids.reshape(seq_ids.shape[0])
        position_ids = position_ids.reshape(position_ids.shape[0])
        if self.apply_seq_ids_mask and seq_ids.shape[0] > 1:
            seq_ids_mask = torch.ge(seq_ids, torch.full_like(seq_ids, 0))
            pad_seq_ids = torch.full_like(seq_ids, self.max_batch_size)
            seq_ids = torch.where(seq_ids_mask, seq_ids, pad_seq_ids)
        index = (position_ids) % self.buffer_length
        return self.hidden_states[seq_ids, index].unsqueeze(1)
