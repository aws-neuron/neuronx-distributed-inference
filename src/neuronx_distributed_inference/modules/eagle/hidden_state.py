import torch

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki as nki
import neuronxcc.nki.typing as nt


@nki.jit(experimental_flags='enable-mutable-parameter')
def rolling_buffer_set_state(
    buffer: nt.tensor[nt.mutable],  # float [max_batch_size, buffer_size, hidden_size]
    hidden_states: nt.tensor,       # float [batch_size, hidden_size]
    batch_ids: nt.tensor,           # int   [batch_size]
    position_ids: nt.tensor         # int   [batch_size]
):
    """
    Insert hidden states into a rolling buffer at the batch & token position.

    This kernel is used to ensure that large DMAs are performed for the hidden
    state update. When this kernel is not used, it is possible to encounter
    inefficient LNC sharding that uses many small DMAs in order to perform the
    state write.

    For simplicity, this kernel uses single index DMAs and a single NeuronCore
    for each scatter.

    Arguments:
        buffer: The rolling buffer to insert values into.
        hidden_states: The new hidden states to insert into the rolling buffer.
        batch_ids: The batch or sequence ids for each hidden state
        position_ids: The position of each token. This function assumes that position
            modification has been applied (positions_ids % buffer_length).

    Returns:
        The rolling buffer modified in-place.
    """
    batch_size = batch_ids.shape[0]
    max_batch_size, buffer_size, hidden_size = buffer.shape

    # Validate that inputs are of expected size
    assert hidden_states.shape[0] == batch_size
    assert position_ids.shape[0] == batch_size
    assert buffer.shape[0] >= batch_size
    assert buffer.shape[-1] == hidden_states.shape[-1]

    reshaped = buffer.reshape((max_batch_size * buffer_size, hidden_size))

    # Loop over batch to avoid issues with HBM -> HBM Vector DMAs (uCode-135)
    for i in nl.static_range(batch_size):

        # Compute positional offset into the rolling buffer
        batch_id = nl.load(batch_ids[i])
        position_id = nl.load(position_ids[i])
        base = nisa.tensor_scalar(batch_id, nl.multiply, buffer_size)
        index = nisa.tensor_tensor(base, position_id, op=nl.add)

        # Scatter into the hidden buffer on one NeuronCore
        b_i, h_i = nl.mgrid[i:i + 1, :hidden_size]
        nisa.dma_copy(src=hidden_states[b_i, h_i], dst=reshaped[index, h_i], oob_mode=nisa.oob_mode.skip)

    return buffer


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
        use_kernel: bool = False,
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
        self.use_kernel = use_kernel

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

        # Always use kernel scatter when executing on-device
        if hidden_state.device.type == 'cpu' or not self.use_kernel:
            result = torch.index_put(self.hidden_states, index, hidden_state)
        else:
            result = rolling_buffer_set_state(self.hidden_states, hidden_state, *index)

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
