import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.nccl as nccl
import numpy as np
from neuronx_distributed.parallel_layers import parallel_state
from neuronxcc.nki._pre_prod_kernels.output_proj import output_proj_kernel

# Global SPMD grid variables used throughout the code.
n_prgs, prg_id = 1, 0


@nki.jit
def __init_spmd_grid_size():
    """
    Initializes the spmd global variables n_prgs, prg_id
    """
    grid_ndim = nl.program_ndim()
    assert grid_ndim == 0 or grid_ndim == 1, \
        "llama3_transfomer_fwd_<tp|rmsnorm_sp> only supports no specialization or specialization along one axis"

    global n_prgs, prg_id
    if grid_ndim != 0 and nl.num_programs(axes=0) > 1:
        n_prgs = nl.num_programs(axes=0)
        prg_id = nl.program_id(axis=0)


@nki.jit
def matmul_o_proj_kernel(self_attn_out, W_o):
    """
    Calls the output_proj_kernel for output projection, then performs all_reduce
    :param self_attn_out: Shape is [B, N, D, S]
    :param W_o: Shape is [N*D, H]
    """
    __init_spmd_grid_size()
    batch_size, _, _, sequence_length = self_attn_out.shape
    _, output_dim = W_o.shape
    o_proj = nl.ndarray(
        shape=[batch_size, sequence_length, output_dim],
        dtype=self_attn_out.dtype,
        buffer=nl.shared_hbm,
    )
    o_proj_temp = nl.ndarray(
        shape=[batch_size, sequence_length, output_dim],
        dtype=self_attn_out.dtype,
        buffer=nl.shared_hbm,
    )
    replica_groups = parallel_state.get_tensor_model_parallel_replica_groups()
    i_n = nl.arange(output_dim)[None, None, :]
    i_m0 = nl.arange(sequence_length)[None, :, None]
    # self_attn_out: [batch_size, N * D, sequence_length]
    # W_o: [N * D, output_dim]
    output_proj_kernel(self_attn_out, W_o, o_proj_temp)
    nccl.all_reduce(
        op=np.add,
        srcs=[o_proj_temp],
        dsts=[o_proj[0, i_m0, i_n]],
        replica_groups=replica_groups,
        dtype=self_attn_out.dtype,
    )
    return o_proj
