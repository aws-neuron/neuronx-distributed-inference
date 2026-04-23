"""NKI per-chunk DeltaNet kernel for CTE (context encoding / prefill).

Single-chunk kernel: processes one chunk (128 tokens) with Neumann-series
power-doubling for intra-chunk correction. The caller loops over chunks
in PyTorch, passing state between calls.

Each kernel call:
  - Takes one chunk of data: q, k, v, beta, g_cumsum, g_last  (all 128x128)
  - Takes recurrent state_in (128x128)
  - Returns chunk output (128x128) and state_out (128x128)

No sequence-indexed DMA inside the kernel -- all inputs/outputs are full tiles.
This avoids the DMA OOB issue seen with nl.sequential_range + slice indexing
in the NxDI model compilation context.

NKI v3 (SDK 2.29, NKI 0.3.0). Uses nki.* namespace.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128


@nki.jit
def deltanet_chunk_step(
    query,  # (128, 128) float32 -- one chunk, l2-normed+scaled
    key,  # (128, 128) float32 -- one chunk, l2-normed
    value,  # (128, 128) float32 -- one chunk
    beta_broadcast,  # (128, 128) float32 -- write gate broadcast to 128
    g_cumsum,  # (128, 128) float32 -- cumsum of g within chunk, broadcast
    g_last,  # (128, 128) float32 -- g_cumsum[-1], constant in chunk, broadcast
    state_in,  # (128, 128) float32 -- recurrent state from previous chunk
    lower_mask,  # (128, 128) float32 -- strict lower triangular
    identity,  # (128, 128) float32 -- identity matrix
    lower_mask_diag,  # (128, 128) float32 -- lower tri with diagonal
):
    """Process one chunk of DeltaNet.

    Returns:
        output:    (128, 128) float32 -- chunk output
        state_out: (128, 128) float32 -- updated recurrent state
    """
    C, dim = query.shape  # C = 128, dim = 128

    # Output tensors in HBM
    output = nl.ndarray((P_MAX, dim), dtype=query.dtype, buffer=nl.shared_hbm)
    state_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    # Load all inputs into SBUF
    q_c = nl.ndarray((P_MAX, dim), dtype=query.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_c, src=query)

    k_c = nl.ndarray((P_MAX, dim), dtype=key.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_c, src=key)

    v_c = nl.ndarray((P_MAX, dim), dtype=value.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_c, src=value)

    beta_c = nl.ndarray((P_MAX, dim), dtype=beta_broadcast.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=beta_c, src=beta_broadcast)

    gc_c = nl.ndarray((P_MAX, dim), dtype=g_cumsum.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc_c, src=g_cumsum)

    gl_c = nl.ndarray((P_MAX, dim), dtype=g_last.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gl_c, src=g_last)

    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=state, src=state_in)

    # Load masks
    eye = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eye, src=identity)

    Lmask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask, src=lower_mask)

    Lmask_d = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask_d, src=lower_mask_diag)

    # ============================================================
    # k_beta = K * beta, v_beta = V * beta
    # ============================================================
    k_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=k_beta, data1=k_c, data2=beta_c, op=nl.multiply)

    v_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=v_beta, data1=v_c, data2=beta_c, op=nl.multiply)

    # ============================================================
    # exp(g_cumsum) and exp(-g_cumsum)
    # ============================================================
    exp_gc = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=exp_gc, op=nl.exp, data=gc_c, bias=None, scale=1.0)

    neg_gc = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=neg_gc,
        data=gc_c,
        op0=nl.multiply,
        operand0=-1.0,
        engine=nisa.vector_engine,
    )
    exp_neg_gc = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=exp_neg_gc, op=nl.exp, data=neg_gc, bias=None, scale=1.0)

    # exp(g_last) for state decay
    exp_gl = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=exp_gl, op=nl.exp, data=gl_c, bias=None, scale=1.0)

    # ============================================================
    # Phase 1: Build A matrix (intra-chunk correction)
    # QK = k_beta @ k^T  -- contract over features
    # ============================================================
    kb_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kb_T_psum, stationary=k_beta, moving=eye)
    kb_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kb_T, src=kb_T_psum)

    k_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=k_T_psum, stationary=k_c, moving=eye)
    k_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=k_T, src=k_T_psum)

    QK_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=QK_psum, stationary=kb_T, moving=k_T)
    QK = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=QK, src=QK_psum)

    # ============================================================
    # Decay mask: QK_decay[i,j] = QK[i,j] * exp(gc[i]) * exp(-gc[j])
    # ============================================================
    QK_row = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=QK_row, data1=QK, data2=exp_gc, op=nl.multiply)

    QK_r_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=QK_r_T_psum, stationary=QK_row, moving=eye)
    QK_r_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=QK_r_T, src=QK_r_T_psum)

    QK_r_T_col = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=QK_r_T_col, data1=QK_r_T, data2=exp_neg_gc, op=nl.multiply)

    QK_d_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=QK_d_psum, stationary=QK_r_T_col, moving=eye)
    QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=QK_decay, src=QK_d_psum)

    # A = -QK_decay * lower_mask
    neg_QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=neg_QK_decay,
        data=QK_decay,
        op0=nl.multiply,
        operand0=-1.0,
        engine=nisa.vector_engine,
    )
    A = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=A, data1=neg_QK_decay, data2=Lmask, op=nl.multiply)

    # ============================================================
    # Neumann power-doubling: N = (I+A)(I+A^2)...(I+A^{64})
    # ============================================================
    P_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=P_acc, data1=eye, data2=A, op=nl.add)

    A_pow = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=A_pow, src=A)

    for _round in nl.sequential_range(6):
        Ap_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=Ap_T_psum, stationary=A_pow, moving=eye)
        Ap_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=Ap_T, src=Ap_T_psum)

        Ap_sq_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=Ap_sq_psum, stationary=Ap_T, moving=A_pow)
        nisa.tensor_copy(dst=A_pow, src=Ap_sq_psum)

        IpA = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=IpA, data1=eye, data2=A_pow, op=nl.add)

        IpA_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=IpA_T_psum, stationary=IpA, moving=eye)
        IpA_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=IpA_T, src=IpA_T_psum)

        Pacc_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=Pacc_psum, stationary=IpA_T, moving=P_acc)
        nisa.tensor_copy(dst=P_acc, src=Pacc_psum)

    # ============================================================
    # Apply N: value_corr = N @ v_beta, k_cumdecay = N @ (k_beta * exp_gc)
    # ============================================================
    N_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=N_T_psum, stationary=P_acc, moving=eye)
    N_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=N_T, src=N_T_psum)

    vc_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=vc_psum, stationary=N_T, moving=v_beta)
    value_corr = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=value_corr, src=vc_psum)

    kb_exp_gc = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=kb_exp_gc, data1=k_beta, data2=exp_gc, op=nl.multiply)

    kcd_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kcd_psum, stationary=N_T, moving=kb_exp_gc)
    k_cumdecay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=k_cumdecay, src=kcd_psum)

    # ============================================================
    # Phase 2: Inter-chunk state propagation
    # attn_intra = (q @ k^T) * decay_mask * lower_mask_diag
    # ============================================================
    q_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=q_T_psum, stationary=q_c, moving=eye)
    q_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=q_T, src=q_T_psum)

    qk_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=qk_psum, stationary=q_T, moving=k_T)
    qk_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qk_raw, src=qk_psum)

    qk_row = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=qk_row, data1=qk_raw, data2=exp_gc, op=nl.multiply)

    qk_r_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=qk_r_T_psum, stationary=qk_row, moving=eye)
    qk_r_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qk_r_T, src=qk_r_T_psum)

    qk_r_T_col = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=qk_r_T_col, data1=qk_r_T, data2=exp_neg_gc, op=nl.multiply)

    qk_d_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=qk_d_psum, stationary=qk_r_T_col, moving=eye)
    qk_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qk_decay, src=qk_d_psum)

    attn_intra = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=attn_intra, data1=qk_decay, data2=Lmask_d, op=nl.multiply)

    # ============================================================
    # v_prime = k_cumdecay @ state
    # ============================================================
    kcd_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kcd_T_psum, stationary=k_cumdecay, moving=eye)
    kcd_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kcd_T, src=kcd_T_psum)

    vp_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=vp_psum, stationary=kcd_T, moving=state)
    v_prime = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=v_prime, src=vp_psum)

    v_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=v_new, data1=value_corr, data2=v_prime, op=nl.subtract)

    # ============================================================
    # attn_inter = (q * exp(g_cumsum)) @ state
    # ============================================================
    q_exp = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=q_exp, data1=q_c, data2=exp_gc, op=nl.multiply)

    qe_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=qe_T_psum, stationary=q_exp, moving=eye)
    qe_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qe_T, src=qe_T_psum)

    ai_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ai_psum, stationary=qe_T, moving=state)
    attn_inter = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=attn_inter, src=ai_psum)

    # ============================================================
    # attn_intra @ v_new
    # ============================================================
    ai_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ai_T_psum, stationary=attn_intra, moving=eye)
    ai_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ai_T, src=ai_T_psum)

    intra_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=intra_psum, stationary=ai_T, moving=v_new)
    intra_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=intra_out, src=intra_psum)

    # ============================================================
    # chunk_output = attn_inter + intra_out
    # ============================================================
    chunk_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=chunk_out, data1=attn_inter, data2=intra_out, op=nl.add)

    nisa.dma_copy(dst=output, src=chunk_out)

    # ============================================================
    # State update: state_new = exp(g_last) * (state + k_raw_decay^T @ v_new)
    # ============================================================
    k_raw_decay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=k_raw_decay, data1=k_c, data2=exp_neg_gc, op=nl.multiply)

    kv_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kv_psum, stationary=k_raw_decay, moving=v_new)
    kv_outer = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kv_outer, src=kv_psum)

    state_plus = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=state_plus, data1=state, data2=kv_outer, op=nl.add)

    state_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=state_new, data1=state_plus, data2=exp_gl, op=nl.multiply)

    nisa.dma_copy(dst=state_out, src=state_new)

    return output, state_out
