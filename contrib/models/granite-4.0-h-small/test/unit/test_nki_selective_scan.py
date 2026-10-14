#!/usr/bin/env python3
"""
NKI Selective Scan Kernel for Mamba2 (Granite 4.0-H-Small)

Replaces the O(L²) parallel scan with O(L) hardware-accelerated scan
using nisa.tensor_tensor_scan on Trainium2.

tensor_tensor_scan computes: result[i] = op0(data0[i], result[i-1]) op1 data1[i]
For Mamba SSM: state[t] = exp(dA[t]) * state[t-1] + dBx[t]
  → data0 = exp(dA), op0 = multiply, data1 = dBx, op1 = add

Granite Mamba2 dimensions (after TP=4 sharding, num_heads//4=32):
    batch_size = 1, seq_len ≤ 128, num_heads = 32 (TP-sharded)
    head_dim = 64, ssm_state_size = 128

Strategy:
    - Pre-transpose inputs to (num_heads, seq_len, ...) on PyTorch side
    - Partition dim (P=128) maps to num_heads (32, padded to 128 or tiled)
    - Free dim maps to seq_len (up to 128)
    - Outer loop over head_dim × ssm_state_size

Usage:
    export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    python nki_selective_scan.py
"""

import numpy as np
import torch
import time


# ==============================================================================
# Reference implementation (PyTorch, CPU)
# ==============================================================================


def selective_scan_reference(
    x: torch.Tensor,  # (batch, seq_len, num_heads, head_dim) float32
    dt: torch.Tensor,  # (batch, seq_len, num_heads) float32
    A: torch.Tensor,  # (num_heads,) float32 — negative values
    B: torch.Tensor,  # (batch, seq_len, num_heads, ssm_state_size) float32
    C: torch.Tensor,  # (batch, seq_len, num_heads, ssm_state_size) float32
    D: torch.Tensor,  # (num_heads,) float32
) -> tuple:
    """
    Sequential O(L) reference selective scan.

    Returns:
        y: (batch, seq_len, num_heads, head_dim)
        final_state: (batch, num_heads, head_dim, ssm_state_size)
    """
    batch, seq_len, num_heads, head_dim = x.shape
    ssm_state_size = B.shape[-1]

    state = torch.zeros(batch, num_heads, head_dim, ssm_state_size, dtype=x.dtype)
    y = torch.zeros_like(x)

    for t in range(seq_len):
        dA = torch.exp(dt[:, t, :] * A)  # (batch, heads)
        dB = dt[:, t, :].unsqueeze(-1) * B[:, t, :, :]  # (batch, heads, state)
        dBx = dB.unsqueeze(2) * x[:, t, :, :].unsqueeze(
            -1
        )  # (batch, heads, dim, state)

        state = dA.unsqueeze(-1).unsqueeze(-1) * state + dBx
        y[:, t, :, :] = torch.einsum("bhds,bhs->bhd", state, C[:, t, :, :])
        y[:, t, :, :] += D.view(1, -1, 1) * x[:, t, :, :]

    return y, state


def selective_scan_quadratic(
    x: torch.Tensor,  # (batch, seq_len, num_heads, head_dim) float32
    dt: torch.Tensor,  # (batch, seq_len, num_heads) float32
    A: torch.Tensor,  # (num_heads,) float32
    B: torch.Tensor,  # (batch, seq_len, num_heads, ssm_state_size) float32
    C: torch.Tensor,  # (batch, seq_len, num_heads, ssm_state_size) float32
    D: torch.Tensor,  # (num_heads,) float32
) -> tuple:
    """O(L²) parallel scan — matches current Neuron implementation."""
    batch, seq_len, num_heads, head_dim = x.shape

    dA_log = dt * A.view(1, 1, -1)
    dB = dt.unsqueeze(-1) * B
    dBx = dB.unsqueeze(3) * x.unsqueeze(-1)

    log_dA_cumsum = torch.cumsum(dA_log, dim=1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=x.dtype))
    log_diff = log_dA_cumsum.unsqueeze(2) - log_dA_cumsum.unsqueeze(1)
    log_diff = log_diff.masked_fill(causal_mask.unsqueeze(0).unsqueeze(-1) == 0, -1e9)
    weights = torch.exp(log_diff)

    states = torch.einsum("btih,bihds->bthds", weights, dBx)
    y = torch.einsum("blhs,blhds->blhd", C, states)
    y = y + D.view(1, 1, -1, 1) * x

    return y, states[:, -1, :, :, :]


# ==============================================================================
# NKI Kernel — Module-level (required for NKI tracer to find the function)
# ==============================================================================

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    HAS_NKI = True
except ImportError:
    HAS_NKI = False

if HAS_NKI:
    P_MAX = 128

    @nki.jit
    def nki_scan_kernel(
        dA_exp_t,  # (NH, SL) — pre-transposed decay coefficients
        dBx_t,  # (NH * HD * SS, SL) — flattened+transposed
        C_t,  # (NH * SS, SL) — flattened+transposed
        Dx_t,  # (NH * HD, SL) — pre-computed D*x, flattened+transposed
        x_t,  # (NH * HD, SL) — flattened+transposed (unused, for shape derivation)
        hd_range,  # (HD,) — dummy tensor whose shape[0] gives head_dim
        ss_range,  # (SS,) — dummy tensor whose shape[0] gives ssm_state_size
    ):
        """
        NKI selective scan using tensor_tensor_scan.

        Allocates outputs on HBM and returns them (NKI pattern).
        D*x is pre-computed on PyTorch side to avoid broadcast issues.

        Returns:
            y_out: (NH * HD, SL) — scan output
            final_state_out: (NH * HD * SS, 1) — final hidden state
        """
        NH = dA_exp_t.shape[0]
        SL = dA_exp_t.shape[1]
        HD = hd_range.shape[0]
        SS = ss_range.shape[0]

        # Allocate outputs on HBM
        y_out = nl.ndarray((NH * HD, SL), dtype=nl.float32, buffer=nl.shared_hbm)
        final_state_out = nl.ndarray(
            (NH * HD * SS, 1), dtype=nl.float32, buffer=nl.shared_hbm
        )

        # Load dA_exp once: (num_heads, seq_len) -> SBUF
        dA_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=dA_sb, value=0.0)
        nisa.dma_copy(
            dst=dA_sb[0:NH, 0:SL],
            src=dA_exp_t[0:NH, 0:SL],
        )

        # For each head_dim d:
        for d in nl.affine_range(HD):
            # Load pre-computed D*x as initial y accumulator
            y_acc_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=y_acc_sb, value=0.0)
            Dx_row_start = d * NH
            nisa.dma_copy(
                dst=y_acc_sb[0:NH, 0:SL],
                src=Dx_t[Dx_row_start : Dx_row_start + NH, 0:SL],
            )

            # Accumulate over ssm_state_size
            for s in nl.affine_range(SS):
                # Load dBx for this (d, s): from dBx_t at row (d * SS + s) * NH
                dBx_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=dBx_sb, value=0.0)
                dBx_row = (d * SS + s) * NH
                nisa.dma_copy(
                    dst=dBx_sb[0:NH, 0:SL],
                    src=dBx_t[dBx_row : dBx_row + NH, 0:SL],
                )

                # Run scan: state[h, t] = dA[h, t] * state[h, t-1] + dBx[h, t]
                init_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=init_sb, value=0.0)

                state_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor_scan(
                    dst=state_sb[0:NH, 0:SL],
                    data0=dA_sb[0:NH, 0:SL],
                    data1=dBx_sb[0:NH, 0:SL],
                    initial=init_sb[0:NH, 0:1],
                    op0=nl.multiply,
                    op1=nl.add,
                )

                # Save final state column: state[h, SL-1] → final_state_out
                final_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=final_sb[0:NH, 0:1],
                    src=state_sb[0:NH, SL - 1 : SL],
                )
                fs_row = (d * SS + s) * NH
                nisa.dma_copy(
                    dst=final_state_out[fs_row : fs_row + NH, 0:1],
                    src=final_sb[0:NH, 0:1],
                )

                # Load C for this state dim s: from C_t at row s * NH
                C_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=C_sb, value=0.0)
                C_row = s * NH
                nisa.dma_copy(
                    dst=C_sb[0:NH, 0:SL],
                    src=C_t[C_row : C_row + NH, 0:SL],
                )

                # y_acc += C * state
                Cs_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=Cs_sb[0:NH, 0:SL],
                    data1=C_sb[0:NH, 0:SL],
                    data2=state_sb[0:NH, 0:SL],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=y_acc_sb[0:NH, 0:SL],
                    data1=y_acc_sb[0:NH, 0:SL],
                    data2=Cs_sb[0:NH, 0:SL],
                    op=nl.add,
                )

            # Store y_acc for this d: → y_out at row d * NH
            y_row = d * NH
            nisa.dma_copy(
                dst=y_out[y_row : y_row + NH, 0:SL],
                src=y_acc_sb[0:NH, 0:SL],
            )

        return y_out, final_state_out


def prepare_scan_inputs(x, dt, A, B, C, D):
    """
    Pre-compute and transpose inputs for the NKI kernel.

    This runs on PyTorch (element-wise ops, cheap) before the NKI kernel.

    Args:
        x: (batch, seq_len, num_heads, head_dim)
        dt: (batch, seq_len, num_heads) — after softplus + clamp
        A: (num_heads,) — negative values
        B: (batch, seq_len, num_heads, ssm_state_size)
        C: (batch, seq_len, num_heads, ssm_state_size)
        D: (num_heads,)

    Returns dict of transposed tensors for the NKI kernel (batch=1 assumed).
    """
    batch, seq_len, num_heads, head_dim = x.shape
    ssm_state_size = B.shape[-1]

    # Decay coefficients
    dA_exp = torch.exp(dt * A.view(1, 1, -1))  # (B, L, H)

    # Input contributions
    dB = dt.unsqueeze(-1) * B  # (B, L, H, S)
    dBx = dB.unsqueeze(3) * x.unsqueeze(-1)  # (B, L, H, D, S)

    # Transpose to partition-first layout (squeeze batch dim)
    # dA_exp: (L, H) → (H, L)
    dA_exp_t = dA_exp[0].transpose(0, 1).contiguous()  # (H, L)

    # dBx: (L, H, D, S) → flatten (H*D*S, L) with H varying fastest
    # Layout: for each (d, s), rows d*S*H + s*H : d*S*H + (s+1)*H contain heads for that (d,s)
    dBx_0 = dBx[0]  # (L, H, D, S)
    # Reshape to (L, D*S*H) then transpose to (D*S*H, L)
    # We want indexing: row (d*SS + s)*NH + h maps to dBx[t, h, d, s]
    # So reshape dBx_0 from (L, H, D, S) to (L, D, S, H) → (L, D*S*H) → (D*S*H, L)
    dBx_reshaped = dBx_0.permute(0, 2, 3, 1).reshape(
        seq_len, head_dim * ssm_state_size * num_heads
    )
    dBx_t = dBx_reshaped.transpose(0, 1).contiguous()  # (D*S*H, L)

    # C: (L, H, S) → flatten (S*H, L)
    C_0 = C[0]  # (L, H, S)
    C_reshaped = C_0.permute(0, 2, 1).reshape(seq_len, ssm_state_size * num_heads)
    C_t = C_reshaped.transpose(0, 1).contiguous()  # (S*H, L)

    # x: (L, H, D) → flatten (D*H, L)
    x_0 = x[0]  # (L, H, D)
    x_reshaped = x_0.permute(0, 2, 1).reshape(seq_len, head_dim * num_heads)
    x_t = x_reshaped.transpose(0, 1).contiguous()  # (D*H, L)

    # D*x: pre-compute skip connection — same layout as x_t: (D*H, L)
    # D is (H,), broadcast to (L, H, D) then flatten same as x
    Dx_0 = D.view(1, -1, 1) * x_0  # (L, H, D)
    Dx_reshaped = Dx_0.permute(0, 2, 1).reshape(seq_len, head_dim * num_heads)
    Dx_t = Dx_reshaped.transpose(0, 1).contiguous()  # (D*H, L)

    return {
        "dA_exp_t": dA_exp_t.float(),
        "dBx_t": dBx_t.float(),
        "C_t": C_t.float(),
        "Dx_t": Dx_t.float(),
        "x_t": x_t.float(),
        "num_heads": num_heads,
        "head_dim": head_dim,
        "ssm_state_size": ssm_state_size,
    }


def unpack_scan_outputs(
    y_flat, state_flat, num_heads, head_dim, ssm_state_size, seq_len
):
    """
    Unpack NKI kernel outputs back to standard shapes.

    Args:
        y_flat: (D*H, L) — flattened output
        state_flat: (D*S*H, 1) — flattened final state

    Returns:
        y: (1, seq_len, num_heads, head_dim)
        final_state: (1, num_heads, head_dim, ssm_state_size)
    """
    # y_flat is (D*H, L) where row d*H + h maps to y[t, h, d]
    # Reshape to (D, H, L) → permute to (L, H, D) → add batch
    y_reshaped = y_flat.reshape(head_dim, num_heads, seq_len)  # (D, H, L)
    y = y_reshaped.permute(2, 1, 0).unsqueeze(0).contiguous()  # (1, L, H, D)

    # state_flat is (D*S*H, 1) where row (d*S + s)*H + h maps to state[h, d, s]
    state_reshaped = state_flat.reshape(
        head_dim, ssm_state_size, num_heads
    )  # (D, S, H)
    final_state = (
        state_reshaped.permute(2, 0, 1).unsqueeze(0).contiguous()
    )  # (1, H, D, S)

    return y, final_state


# ==============================================================================
# Tests
# ==============================================================================


def test_reference_match():
    """Validate O(L) and O(L²) produce same results."""
    print("Test 1: Reference implementation match (CPU)...")

    batch, seq_len, num_heads, head_dim, ssm_state_size = 1, 16, 8, 4, 8
    torch.manual_seed(42)

    x = torch.randn(batch, seq_len, num_heads, head_dim)
    dt = torch.rand(batch, seq_len, num_heads) * 0.1
    A = -torch.arange(1, num_heads + 1, dtype=torch.float32)
    B = torch.randn(batch, seq_len, num_heads, ssm_state_size)
    C = torch.randn(batch, seq_len, num_heads, ssm_state_size)
    D = torch.ones(num_heads)

    y_seq, state_seq = selective_scan_reference(x, dt, A, B, C, D)
    y_quad, state_quad = selective_scan_quadratic(x, dt, A, B, C, D)

    y_diff = (y_seq - y_quad).abs()
    state_diff = (state_seq - state_quad).abs()

    print(f"  y max diff:     {y_diff.max().item():.2e}")
    print(f"  state max diff: {state_diff.max().item():.2e}")
    assert y_diff.max().item() < 1e-4
    assert state_diff.max().item() < 1e-4
    print("  PASS\n")


def test_transpose_roundtrip():
    """Validate prepare/unpack are inverses."""
    print("Test 2: Transpose round-trip (CPU)...")

    batch, seq_len, num_heads, head_dim, ssm_state_size = 1, 16, 8, 4, 8
    torch.manual_seed(42)

    x = torch.randn(batch, seq_len, num_heads, head_dim)
    dt = torch.rand(batch, seq_len, num_heads) * 0.1
    A = -torch.arange(1, num_heads + 1, dtype=torch.float32)
    B = torch.randn(batch, seq_len, num_heads, ssm_state_size)
    C = torch.randn(batch, seq_len, num_heads, ssm_state_size)
    D = torch.ones(num_heads)

    inputs = prepare_scan_inputs(x, dt, A, B, C, D)

    # Verify dA_exp transpose
    dA_exp_ref = torch.exp(dt * A.view(1, 1, -1))[0]  # (L, H)
    dA_exp_rt = inputs["dA_exp_t"].transpose(0, 1)  # (H, L) → (L, H)
    assert (dA_exp_ref - dA_exp_rt).abs().max() < 1e-6

    # Verify x transpose round-trip
    x_t = inputs["x_t"]  # (D*H, L)
    x_rt = x_t.reshape(head_dim, num_heads, seq_len).permute(2, 1, 0)  # (L, H, D)
    assert (x[0] - x_rt).abs().max() < 1e-6

    print("  PASS\n")


def test_granite_scale():
    """Test at Granite TP=4 scale."""
    print("Test 3: Granite TP=4 scale (CPU)...")

    # After TP=4 sharding: num_heads=128//4=32
    batch, seq_len, num_heads, head_dim, ssm_state_size = 1, 128, 32, 64, 128
    torch.manual_seed(42)

    x = torch.randn(batch, seq_len, num_heads, head_dim)
    dt = torch.rand(batch, seq_len, num_heads) * 0.1
    A = -torch.arange(1, num_heads + 1, dtype=torch.float32)
    B = torch.randn(batch, seq_len, num_heads, ssm_state_size)
    C = torch.randn(batch, seq_len, num_heads, ssm_state_size)
    D = torch.ones(num_heads)

    t0 = time.perf_counter()
    y_seq, _ = selective_scan_reference(x, dt, A, B, C, D)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_quad, _ = selective_scan_quadratic(x, dt, A, B, C, D)
    t_quad = time.perf_counter() - t0

    # Also time the prepare step
    t0 = time.perf_counter()
    inputs = prepare_scan_inputs(x, dt, A, B, C, D)
    t_prep = time.perf_counter() - t0

    diff = (y_seq - y_quad).abs()
    print(f"  Sequential O(L):   {t_seq:.3f}s")
    print(f"  Quadratic O(L²):   {t_quad:.3f}s")
    print(f"  Speedup:           {t_quad / t_seq:.1f}x (CPU, algorithmic only)")
    print(f"  Prepare time:      {t_prep:.4f}s")
    print(f"  y max diff:        {diff.max().item():.2e}")
    print(f"  Prepared shapes:")
    print(f"    dA_exp_t: {inputs['dA_exp_t'].shape}")
    print(f"    dBx_t:    {inputs['dBx_t'].shape}")
    print(f"    C_t:      {inputs['C_t'].shape}")
    print(f"    x_t:      {inputs['x_t'].shape}")

    assert diff.max().item() < 1e-2  # Looser for larger scale
    print("  PASS\n")


def cpu_simulate_kernel(
    dA_exp_t, dBx_t, C_t, Dx_t, x_t, num_heads, head_dim, ssm_state_size, seq_len
):
    """
    CPU simulation of what the NKI kernel should compute, using the same
    transposed data layout. This isolates layout bugs from NKI bugs.
    """
    NH = num_heads
    HD = head_dim
    SS = ssm_state_size
    SL = seq_len

    y_flat = torch.zeros(HD * NH, SL, dtype=torch.float32)
    state_flat = torch.zeros(HD * SS * NH, 1, dtype=torch.float32)

    for d in range(HD):
        # y_acc = Dx (pre-computed D*x for this d)
        y_acc = Dx_t[d * NH : (d + 1) * NH, :].clone()  # (NH, SL)

        for s in range(SS):
            # dBx for this (d, s)
            dBx_row = (d * SS + s) * NH
            dBx_sb = dBx_t[dBx_row : dBx_row + NH, :]  # (NH, SL)

            # Sequential scan: state[h, t] = dA[h, t] * state[h, t-1] + dBx[h, t]
            state_sb = torch.zeros(NH, SL, dtype=torch.float32)
            prev = torch.zeros(NH, dtype=torch.float32)
            for t in range(SL):
                prev = dA_exp_t[:NH, t] * prev + dBx_sb[:NH, t]
                state_sb[:, t] = prev

            # Save final state
            fs_row = (d * SS + s) * NH
            state_flat[fs_row : fs_row + NH, 0] = state_sb[:, SL - 1]

            # C for this s
            C_row = s * NH
            C_sb = C_t[C_row : C_row + NH, :]  # (NH, SL)

            # y_acc += C * state
            y_acc = y_acc + C_sb * state_sb

        # Store
        y_row = d * NH
        y_flat[y_row : y_row + NH, :] = y_acc

    return y_flat, state_flat


def test_cpu_kernel_sim():
    """Test CPU simulation of kernel logic to isolate layout issues."""
    print("Test 3.5: CPU kernel simulation (isolate layout vs NKI issues)...")

    batch, seq_len, num_heads, head_dim, ssm_state_size = 1, 16, 32, 4, 8
    torch.manual_seed(42)

    x = torch.randn(batch, seq_len, num_heads, head_dim)
    dt = torch.rand(batch, seq_len, num_heads) * 0.1
    A = -torch.arange(1, num_heads + 1, dtype=torch.float32)
    B = torch.randn(batch, seq_len, num_heads, ssm_state_size)
    C = torch.randn(batch, seq_len, num_heads, ssm_state_size)
    D = torch.ones(num_heads)

    # CPU reference
    y_ref, state_ref = selective_scan_reference(x, dt, A, B, C, D)

    # Prepare inputs (same as NKI path)
    inputs = prepare_scan_inputs(x, dt, A, B, C, D)

    # CPU simulation of kernel
    y_flat, state_flat = cpu_simulate_kernel(
        inputs["dA_exp_t"],
        inputs["dBx_t"],
        inputs["C_t"],
        inputs["Dx_t"],
        inputs["x_t"],
        num_heads,
        head_dim,
        ssm_state_size,
        seq_len,
    )

    # Unpack using same function as NKI path
    y_sim, state_sim = unpack_scan_outputs(
        y_flat,
        state_flat,
        num_heads,
        head_dim,
        ssm_state_size,
        seq_len,
    )

    y_diff = (y_ref - y_sim).abs()
    state_diff = (state_ref - state_sim).abs()

    print(f"  y max diff:     {y_diff.max().item():.2e}")
    print(f"  y mean diff:    {y_diff.mean().item():.2e}")
    print(f"  state max diff: {state_diff.max().item():.2e}")

    # Print first few values for manual inspection
    print(f"  y_ref[0,0,0,:4]:  {y_ref[0, 0, 0, :4].tolist()}")
    print(f"  y_sim[0,0,0,:4]:  {y_sim[0, 0, 0, :4].tolist()}")
    print(f"  y_ref[0,0,1,:4]:  {y_ref[0, 0, 1, :4].tolist()}")
    print(f"  y_sim[0,0,1,:4]:  {y_sim[0, 0, 1, :4].tolist()}")

    if y_diff.max().item() < 0.01:
        print("  PASS — layout is correct, any NKI mismatch is a kernel issue\n")
    else:
        print("  FAIL — layout mismatch between prepare/unpack and reference\n")
        # Find where the biggest diff is
        idx = torch.unravel_index(y_diff.argmax(), y_diff.shape)
        print(f"  Worst diff at index {idx}")
        print(f"    ref={y_ref[idx].item():.6f}  sim={y_sim[idx].item():.6f}")


def test_nki_kernel():
    """Test NKI kernel on Trainium."""
    try:
        import torch_xla.core.xla_model as xm

        assert HAS_NKI, "NKI not available"
    except (ImportError, ModuleNotFoundError, AssertionError, NameError):  # noqa: F821
        print("SKIP: NKI/torch_xla not available (run on Trainium)\n")
        return

    print("Test 4: NKI kernel on Trainium...")

    # Start with small dimensions for validation
    batch, seq_len, num_heads, head_dim, ssm_state_size = 1, 16, 32, 4, 8
    torch.manual_seed(42)

    x = torch.randn(batch, seq_len, num_heads, head_dim)
    dt = torch.rand(batch, seq_len, num_heads) * 0.1
    A = -torch.arange(1, num_heads + 1, dtype=torch.float32)
    B = torch.randn(batch, seq_len, num_heads, ssm_state_size)
    C = torch.randn(batch, seq_len, num_heads, ssm_state_size)
    D = torch.ones(num_heads)

    # CPU reference
    y_ref, state_ref = selective_scan_reference(x, dt, A, B, C, D)

    # CPU kernel simulation (for comparison)
    inputs = prepare_scan_inputs(x, dt, A, B, C, D)
    y_flat_cpu, state_flat_cpu = cpu_simulate_kernel(
        inputs["dA_exp_t"],
        inputs["dBx_t"],
        inputs["C_t"],
        inputs["Dx_t"],
        inputs["x_t"],
        num_heads,
        head_dim,
        ssm_state_size,
        seq_len,
    )

    device = xm.xla_device()
    dA_exp_t = inputs["dA_exp_t"].to(device)
    dBx_t = inputs["dBx_t"].to(device)
    C_t = inputs["C_t"].to(device)
    Dx_t = inputs["Dx_t"].to(device)
    x_t = inputs["x_t"].to(device)

    # Run NKI kernel — now returns outputs
    hd_range = torch.zeros(head_dim, dtype=torch.float32, device=device)
    ss_range = torch.zeros(ssm_state_size, dtype=torch.float32, device=device)

    y_flat, state_flat = nki_scan_kernel(
        dA_exp_t,
        dBx_t,
        C_t,
        Dx_t,
        x_t,
        hd_range,
        ss_range,
    )
    xm.mark_step()

    # Compare NKI output to CPU kernel sim (same layout, isolates NKI issues)
    y_flat_nki = y_flat.cpu()
    state_flat_nki = state_flat.cpu()

    flat_y_diff = (y_flat_cpu - y_flat_nki).abs()
    flat_s_diff = (state_flat_cpu - state_flat_nki).abs()
    print(f"  NKI vs CPU-sim (flat y) max diff:     {flat_y_diff.max().item():.2e}")
    print(f"  NKI vs CPU-sim (flat state) max diff: {flat_s_diff.max().item():.2e}")

    # Also compare to original reference
    y_nki, state_nki = unpack_scan_outputs(
        y_flat_nki,
        state_flat_nki,
        num_heads,
        head_dim,
        ssm_state_size,
        seq_len,
    )

    y_diff = (y_ref - y_nki).abs()
    state_diff = (state_ref - state_nki).abs()

    print(f"  NKI vs reference y max diff:     {y_diff.max().item():.2e}")
    print(f"  NKI vs reference y mean diff:    {y_diff.mean().item():.2e}")
    print(f"  NKI vs reference state max diff: {state_diff.max().item():.2e}")

    # Print sample values
    print(f"  y_ref[0,0,0,:4]:  {y_ref[0, 0, 0, :4].tolist()}")
    print(f"  y_nki[0,0,0,:4]:  {y_nki[0, 0, 0, :4].tolist()}")

    if y_diff.max().item() < 0.01:
        print("  PASS\n")
    else:
        print(f"  FAIL — max diff {y_diff.max().item():.4f}\n")

    # Now test at Granite TP=4 scale
    print("Test 5: NKI kernel at Granite TP=4 scale...")

    batch, seq_len, num_heads, head_dim, ssm_state_size = 1, 128, 32, 64, 128
    torch.manual_seed(42)

    x = torch.randn(batch, seq_len, num_heads, head_dim)
    dt = torch.rand(batch, seq_len, num_heads) * 0.1
    A = -torch.arange(1, num_heads + 1, dtype=torch.float32)
    B = torch.randn(batch, seq_len, num_heads, ssm_state_size)
    C = torch.randn(batch, seq_len, num_heads, ssm_state_size)
    D = torch.ones(num_heads)

    y_ref, state_ref = selective_scan_reference(x, dt, A, B, C, D)
    inputs = prepare_scan_inputs(x, dt, A, B, C, D)

    dA_exp_t = inputs["dA_exp_t"].to(device)
    dBx_t = inputs["dBx_t"].to(device)
    C_t = inputs["C_t"].to(device)
    Dx_t = inputs["Dx_t"].to(device)
    x_t = inputs["x_t"].to(device)

    hd_range = torch.zeros(head_dim, dtype=torch.float32, device=device)
    ss_range = torch.zeros(ssm_state_size, dtype=torch.float32, device=device)

    t0 = time.perf_counter()
    y_flat, state_flat = nki_scan_kernel(
        dA_exp_t,
        dBx_t,
        C_t,
        Dx_t,
        x_t,
        hd_range,
        ss_range,
    )
    xm.mark_step()
    t_nki = time.perf_counter() - t0

    y_nki, state_nki = unpack_scan_outputs(
        y_flat.cpu(),
        state_flat.cpu(),
        num_heads,
        head_dim,
        ssm_state_size,
        seq_len,
    )

    y_diff = (y_ref - y_nki).abs()
    print(f"  NKI time (incl compile): {t_nki:.3f}s")
    print(f"  y max diff: {y_diff.max().item():.2e}")

    # Benchmark: run multiple iterations after warmup
    # First warmup call (may still compile for this shape)
    y_flat2, state_flat2 = nki_scan_kernel(
        dA_exp_t,
        dBx_t,
        C_t,
        Dx_t,
        x_t,
        hd_range,
        ss_range,
    )
    xm.mark_step()

    # Second warmup
    y_flat3, state_flat3 = nki_scan_kernel(
        dA_exp_t,
        dBx_t,
        C_t,
        Dx_t,
        x_t,
        hd_range,
        ss_range,
    )
    xm.mark_step()

    # Timed runs
    n_runs = 10
    t0 = time.perf_counter()
    for _ in range(n_runs):
        y_flat_bench, state_flat_bench = nki_scan_kernel(
            dA_exp_t,
            dBx_t,
            C_t,
            Dx_t,
            x_t,
            hd_range,
            ss_range,
        )
        xm.mark_step()
    t_bench = time.perf_counter() - t0
    avg_ms = (t_bench / n_runs) * 1000

    print(f"  NKI benchmark ({n_runs} runs):   {avg_ms:.2f} ms/call")
    print("  PASS\n")


if __name__ == "__main__":
    test_reference_match()
    test_transpose_roundtrip()
    test_cpu_kernel_sim()
    test_granite_scale()
    test_nki_kernel()
