#!/usr/bin/env python3
"""Test harness for the FULL PairformerLayer SPMD mega-kernel.

Validates the mega-kernel output against the CPU reference implementation
of all 7 sub-operations in a PairformerLayer:
    Step 1:  s = s + PairBiasAttn(s, z, mask)
    Step 2:  z = z + TriMulOut(z, pair_mask)
    Step 3:  z = z + TriMulIn(z, pair_mask)
    Step 4:  z = z + TriAttnStart(z)
    Step 5:  z = z + TriAttnEnd(z)
    Step 6a: s = s + Transition_s(s)
    Step 6b: z = z + Transition_z(z)

Usage:
    # On the trn2 instance:
    source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_RT_VISIBLE_CORES=0 \
        python test_full_layer_spmd.py --N 128

    # CPU-only reference (for debugging):
    python test_full_layer_spmd.py --N 128 --cpu-only

Requirements:
    - Boltz-2 pip package installed (boltz v2.2.1)
    - torch-neuronx, neuronxcc (for Neuron compilation)
    - Model weights at ~/.boltz/boltz2_conf.ckpt
"""

import argparse
import os
import time

import torch

from boltz.main import (
    Boltz2,
    PairformerArgsV2,
    Boltz2DiffusionParams,
    MSAModuleArgs,
    BoltzSteeringParams,
)
from dataclasses import asdict

P_MAX = 128


def cosine_similarity(a, b):
    """Compute cosine similarity between two tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm())).item()


def run_cpu_reference(layer, s, z, mask, pair_mask):
    """Run the full PairformerLayer forward pass on CPU as reference.

    Args:
        layer: PairformerLayer (V2) in float32
        s: [1, N, C_s=384] float32
        z: [1, N, N, C_z=128] float32
        mask: [1, N] float32
        pair_mask: [1, N, N] float32

    Returns:
        s_ref: [1, N, C_s] float32
        z_ref: [1, N, N, C_z] float32
        intermediates: list of (name, s_or_z_clone) for per-step comparison
    """
    s_ref = s.clone().float()
    z_ref = z.clone().float()
    intermediates = []

    with torch.no_grad():
        # Step 1: PairBiasAttention (s = s + PBA(s, z))
        # V2: pre_norm_s is external to the attention module
        s_normed = layer.pre_norm_s(s_ref)
        s_delta = layer.attention(s=s_normed, z=z_ref, mask=mask, k_in=s_normed)
        s_ref = s_ref + s_delta
        intermediates.append(
            ("pair_bias_attn", {"s": s_ref.clone(), "z": z_ref.clone()})
        )

        # Step 2: TriMulOut
        z_delta = layer.tri_mul_out(z_ref, mask=pair_mask, use_kernels=False)
        z_ref = z_ref + z_delta
        intermediates.append(("tri_mul_out", {"s": s_ref.clone(), "z": z_ref.clone()}))

        # Step 3: TriMulIn
        z_delta = layer.tri_mul_in(z_ref, mask=pair_mask, use_kernels=False)
        z_ref = z_ref + z_delta
        intermediates.append(("tri_mul_in", {"s": s_ref.clone(), "z": z_ref.clone()}))

        # Step 4: TriAttnStart
        z_delta = layer.tri_att_start(
            z_ref, mask=pair_mask, chunk_size=None, use_kernels=False
        )
        z_ref = z_ref + z_delta
        intermediates.append(
            ("tri_att_start", {"s": s_ref.clone(), "z": z_ref.clone()})
        )

        # Step 5: TriAttnEnd
        z_delta = layer.tri_att_end(
            z_ref, mask=pair_mask, chunk_size=None, use_kernels=False
        )
        z_ref = z_ref + z_delta
        intermediates.append(("tri_att_end", {"s": s_ref.clone(), "z": z_ref.clone()}))

        # Step 6a: Transition_s
        s_delta = layer.transition_s(s_ref)
        s_ref = s_ref + s_delta
        intermediates.append(("transition_s", {"s": s_ref.clone(), "z": z_ref.clone()}))

        # Step 6b: Transition_z
        z_delta = layer.transition_z(z_ref)
        z_ref = z_ref + z_delta
        intermediates.append(("transition_z", {"s": s_ref.clone(), "z": z_ref.clone()}))

    return s_ref, z_ref, intermediates


def load_model(layer_idx=0):
    """Load Boltz-2 model and return the specified pairformer layer."""
    print("Loading Boltz-2 model...")
    t0 = time.time()
    boltz = Boltz2.load_from_checkpoint(
        os.path.expanduser("~/.boltz/boltz2_conf.ckpt"),
        strict=True,
        predict_args={
            "recycling_steps": 3,
            "sampling_steps": 20,
            "diffusion_samples": 1,
            "max_parallel_samples": 1,
            "write_confidence_summary": True,
            "write_full_pae": True,
            "write_full_pde": True,
        },
        map_location="cpu",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False,
        use_kernels=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs(use_paired_feature=True)),
        steering_args=asdict(BoltzSteeringParams()),
    )
    boltz.eval()
    boltz = boltz.float()
    t1 = time.time()
    print(f"Model loaded in {t1 - t0:.1f}s")
    layer = boltz.pairformer_module.layers[layer_idx]
    print(f"Using pairformer layer {layer_idx}")
    return layer


def extract_all_weights(layer):
    """Extract ALL weights for the full layer mega-kernel. Same as compile script."""
    weight_map = [
        # PairBiasAttention (Step 1)
        ("pba_norm_s_w", "pre_norm_s.weight"),
        ("pba_norm_s_b", "pre_norm_s.bias"),
        ("pba_norm_z_w", "attention.proj_z.0.weight"),
        ("pba_norm_z_b", "attention.proj_z.0.bias"),
        ("pba_q_w", "attention.proj_q.weight"),
        ("pba_q_b", "attention.proj_q.bias"),
        ("pba_k_w", "attention.proj_k.weight"),
        ("pba_v_w", "attention.proj_v.weight"),
        ("pba_z_w", "attention.proj_z.1.weight"),
        ("pba_g_w", "attention.proj_g.weight"),
        ("pba_o_w", "attention.proj_o.weight"),
        # TriMulOut (Step 2)
        ("tmul_out_norm_in_w", "tri_mul_out.norm_in.weight"),
        ("tmul_out_norm_in_b", "tri_mul_out.norm_in.bias"),
        ("tmul_out_p_in_w", "tri_mul_out.p_in.weight"),
        ("tmul_out_g_in_w", "tri_mul_out.g_in.weight"),
        ("tmul_out_norm_out_w", "tri_mul_out.norm_out.weight"),
        ("tmul_out_norm_out_b", "tri_mul_out.norm_out.bias"),
        ("tmul_out_p_out_w", "tri_mul_out.p_out.weight"),
        ("tmul_out_g_out_w", "tri_mul_out.g_out.weight"),
        # TriMulIn (Step 3)
        ("tmul_in_norm_in_w", "tri_mul_in.norm_in.weight"),
        ("tmul_in_norm_in_b", "tri_mul_in.norm_in.bias"),
        ("tmul_in_p_in_w", "tri_mul_in.p_in.weight"),
        ("tmul_in_g_in_w", "tri_mul_in.g_in.weight"),
        ("tmul_in_norm_out_w", "tri_mul_in.norm_out.weight"),
        ("tmul_in_norm_out_b", "tri_mul_in.norm_out.bias"),
        ("tmul_in_p_out_w", "tri_mul_in.p_out.weight"),
        ("tmul_in_g_out_w", "tri_mul_in.g_out.weight"),
        # TriAttnStart (Step 4)
        ("tatt_s_ln_w", "tri_att_start.layer_norm.weight"),
        ("tatt_s_ln_b", "tri_att_start.layer_norm.bias"),
        ("tatt_s_bias_proj_w", "tri_att_start.linear.weight"),
        ("tatt_s_q_w", "tri_att_start.mha.linear_q.weight"),
        ("tatt_s_k_w", "tri_att_start.mha.linear_k.weight"),
        ("tatt_s_v_w", "tri_att_start.mha.linear_v.weight"),
        ("tatt_s_g_w", "tri_att_start.mha.linear_g.weight"),
        ("tatt_s_o_w", "tri_att_start.mha.linear_o.weight"),
        # TriAttnEnd (Step 5)
        ("tatt_e_ln_w", "tri_att_end.layer_norm.weight"),
        ("tatt_e_ln_b", "tri_att_end.layer_norm.bias"),
        ("tatt_e_bias_proj_w", "tri_att_end.linear.weight"),
        ("tatt_e_q_w", "tri_att_end.mha.linear_q.weight"),
        ("tatt_e_k_w", "tri_att_end.mha.linear_k.weight"),
        ("tatt_e_v_w", "tri_att_end.mha.linear_v.weight"),
        ("tatt_e_g_w", "tri_att_end.mha.linear_g.weight"),
        ("tatt_e_o_w", "tri_att_end.mha.linear_o.weight"),
        # Transition_s (Step 6a)
        ("trans_s_norm_w", "transition_s.norm.weight"),
        ("trans_s_norm_b", "transition_s.norm.bias"),
        ("trans_s_fc1_w", "transition_s.fc1.weight"),
        ("trans_s_fc2_w", "transition_s.fc2.weight"),
        ("trans_s_fc3_w", "transition_s.fc3.weight"),
        # Transition_z (Step 6b)
        ("trans_z_norm_w", "transition_z.norm.weight"),
        ("trans_z_norm_b", "transition_z.norm.bias"),
        ("trans_z_fc1_w", "transition_z.fc1.weight"),
        ("trans_z_fc2_w", "transition_z.fc2.weight"),
        ("trans_z_fc3_w", "transition_z.fc3.weight"),
    ]

    w = {}
    for buf_name, attr_path in weight_map:
        obj = layer
        for p in attr_path.split("."):
            obj = getattr(obj, p)
        v = obj.data.clone().to(torch.bfloat16)
        if v.dim() == 1:
            v = v.unsqueeze(0).expand(P_MAX, -1).contiguous()
        w[buf_name] = v

    return w


def run_full_kernel(layer, s, z, mask, pair_mask, weights_bf16, device, N, C_z, C_s):
    """Run the full PairformerLayer SPMD mega-kernel and compare to CPU reference."""
    import torch_xla.core.xla_model as xm
    from full_pairformer_layer_spmd import full_pairformer_layer_spmd

    H_z = 4
    H_s = 16
    n_flat = N * N

    # CPU reference (all 7 steps)
    print("Running CPU reference (all 7 steps)...")
    t0 = time.time()
    s_ref, z_ref, intermediates = run_cpu_reference(layer, s, z, mask, pair_mask)
    t1 = time.time()
    print(f"CPU reference: {t1 - t0:.3f}s")

    for name, vals in intermediates:
        s_v = vals["s"]
        z_v = vals["z"]
        print(
            f"  After {name}: s mean={s_v.mean():.6f} std={s_v.std():.6f} | "
            f"z mean={z_v.mean():.6f} std={z_v.std():.6f}"
        )

    # Prepare inputs for kernel (flatten z to [N*N, C_z], remove batch dim from s)
    s_flat = s[0].clone().to(torch.bfloat16).contiguous().to(device)
    z_flat = z[0].reshape(N * N, C_z).clone().to(torch.bfloat16).contiguous().to(device)
    pm_flat = pair_mask[0].reshape(N * N, 1).to(torch.bfloat16).contiguous().to(device)
    mask_flat = mask[0].unsqueeze(-1).to(torch.bfloat16).contiguous().to(device)

    # Pre-allocate scratch buffers
    scratch_buf = torch.zeros(6 * n_flat, C_z, dtype=torch.bfloat16, device=device)
    bias_buf = torch.zeros(n_flat, H_z, dtype=torch.bfloat16, device=device)
    s_scratch_q = torch.zeros(N, C_s, dtype=torch.bfloat16, device=device)
    s_scratch_k = torch.zeros(N, C_s, dtype=torch.bfloat16, device=device)
    s_scratch_v = torch.zeros(N, C_s, dtype=torch.bfloat16, device=device)
    s_scratch_gate = torch.zeros(N, C_s, dtype=torch.bfloat16, device=device)
    z_bias_scratch = torch.zeros(n_flat, H_s, dtype=torch.bfloat16, device=device)
    s_intermediate = torch.zeros(N, C_s, dtype=torch.bfloat16, device=device)

    xm.mark_step()
    xm.wait_device_ops()

    w = weights_bf16
    print("\nCompiling + running full layer SPMD mega-kernel...")
    t0 = time.time()

    s_out, z_out = full_pairformer_layer_spmd[2](
        s_flat,
        z_flat,
        pm_flat,
        mask_flat,
        # PBA weights
        w["pba_norm_s_w"],
        w["pba_norm_s_b"],
        w["pba_norm_z_w"],
        w["pba_norm_z_b"],
        w["pba_q_w"],
        w["pba_q_b"],
        w["pba_k_w"],
        w["pba_v_w"],
        w["pba_z_w"],
        w["pba_g_w"],
        w["pba_o_w"],
        # TriMulOut
        w["tmul_out_norm_in_w"],
        w["tmul_out_norm_in_b"],
        w["tmul_out_p_in_w"],
        w["tmul_out_g_in_w"],
        w["tmul_out_norm_out_w"],
        w["tmul_out_norm_out_b"],
        w["tmul_out_p_out_w"],
        w["tmul_out_g_out_w"],
        # TriMulIn
        w["tmul_in_norm_in_w"],
        w["tmul_in_norm_in_b"],
        w["tmul_in_p_in_w"],
        w["tmul_in_g_in_w"],
        w["tmul_in_norm_out_w"],
        w["tmul_in_norm_out_b"],
        w["tmul_in_p_out_w"],
        w["tmul_in_g_out_w"],
        # TriAttnStart
        w["tatt_s_ln_w"],
        w["tatt_s_ln_b"],
        w["tatt_s_bias_proj_w"],
        w["tatt_s_q_w"],
        w["tatt_s_k_w"],
        w["tatt_s_v_w"],
        w["tatt_s_g_w"],
        w["tatt_s_o_w"],
        # TriAttnEnd
        w["tatt_e_ln_w"],
        w["tatt_e_ln_b"],
        w["tatt_e_bias_proj_w"],
        w["tatt_e_q_w"],
        w["tatt_e_k_w"],
        w["tatt_e_v_w"],
        w["tatt_e_g_w"],
        w["tatt_e_o_w"],
        # Transition_s
        w["trans_s_norm_w"],
        w["trans_s_norm_b"],
        w["trans_s_fc1_w"],
        w["trans_s_fc2_w"],
        w["trans_s_fc3_w"],
        # Transition_z
        w["trans_z_norm_w"],
        w["trans_z_norm_b"],
        w["trans_z_fc1_w"],
        w["trans_z_fc2_w"],
        w["trans_z_fc3_w"],
        # Scratch buffers
        scratch_buf,
        bias_buf,
        s_scratch_q,
        s_scratch_k,
        s_scratch_v,
        s_scratch_gate,
        z_bias_scratch,
        s_intermediate,
        N=N,
    )
    xm.mark_step()
    xm.wait_device_ops()
    t1 = time.time()
    print(f"Compilation + first run: {t1 - t0:.1f}s")

    # Compare
    s_result = s_out.cpu().reshape(N, C_s).float()
    z_result = z_out.cpu().reshape(N, N, C_z).float()
    s_ref_squeezed = s_ref.squeeze(0)
    z_ref_squeezed = z_ref.squeeze(0)

    s_cos = cosine_similarity(s_result, s_ref_squeezed)
    s_max_diff = (s_result - s_ref_squeezed).abs().max().item()
    s_mean_diff = (s_result - s_ref_squeezed).abs().mean().item()

    z_cos = cosine_similarity(z_result, z_ref_squeezed)
    z_max_diff = (z_result - z_ref_squeezed).abs().max().item()
    z_mean_diff = (z_result - z_ref_squeezed).abs().mean().item()

    print()
    print(f"{'=' * 60}")
    print(f"Full PairformerLayer Mega-Kernel Results (N={N})")
    print(f"{'=' * 60}")
    print(f"\n--- s output ---")
    print(f"  Cosine similarity:  {s_cos:.6f}")
    print(f"  Max abs diff:       {s_max_diff:.6f}")
    print(f"  Mean abs diff:      {s_mean_diff:.6f}")
    print(f"  Ref mean:           {s_ref_squeezed.mean():.6f}")
    print(f"  Kernel mean:        {s_result.mean():.6f}")
    print(f"  Ref std:            {s_ref_squeezed.std():.6f}")
    print(f"  Kernel std:         {s_result.std():.6f}")

    print(f"\n--- z output ---")
    print(f"  Cosine similarity:  {z_cos:.6f}")
    print(f"  Max abs diff:       {z_max_diff:.6f}")
    print(f"  Mean abs diff:      {z_mean_diff:.6f}")
    print(f"  Ref mean:           {z_ref_squeezed.mean():.6f}")
    print(f"  Kernel mean:        {z_result.mean():.6f}")
    print(f"  Ref std:            {z_ref_squeezed.std():.6f}")
    print(f"  Kernel std:         {z_result.std():.6f}")

    # NaN check
    s_nan = torch.isnan(s_result).any().item()
    z_nan = torch.isnan(z_result).any().item()
    if s_nan:
        print(f"\n  WARNING: s_out contains NaN!")
    if z_nan:
        print(f"\n  WARNING: z_out contains NaN!")

    # Verdict
    print(f"\n--- Verdict ---")
    if s_nan or z_nan:
        print(f"  FAIL (NaN detected)")
    elif s_cos > 0.999 and z_cos > 0.999:
        print(f"  PASS (s_cos={s_cos:.4f} > 0.999, z_cos={z_cos:.4f} > 0.999)")
    elif s_cos > 0.99 and z_cos > 0.99:
        print(f"  MARGINAL PASS (s_cos={s_cos:.4f} > 0.99, z_cos={z_cos:.4f} > 0.99)")
    else:
        print(f"  FAIL (s_cos={s_cos:.4f}, z_cos={z_cos:.4f} — need > 0.99)")

    return s_cos, z_cos


def main():
    parser = argparse.ArgumentParser(
        description="Test full PairformerLayer SPMD mega-kernel"
    )
    parser.add_argument(
        "--N", type=int, default=128, help="Sequence length (multiple of 128)"
    )
    parser.add_argument(
        "--layer", type=int, default=0, help="Which pairformer layer (0-63)"
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Only run CPU reference"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    N = args.N
    C_z = 128
    C_s = 384

    assert N % 128 == 0, f"N must be a multiple of 128, got {N}"

    print(
        f"=== Full PairformerLayer SPMD Mega-Kernel Test: N={N}, layer={args.layer} ==="
    )
    print()

    # Load model
    layer = load_model(args.layer)

    # Create test inputs (with batch dim for CPU reference)
    print(f"Creating test inputs: N={N}")
    torch.manual_seed(args.seed)
    s = torch.randn(1, N, C_s, dtype=torch.float32) * 0.1
    z = torch.randn(1, N, N, C_z, dtype=torch.float32) * 0.1
    mask = torch.ones(1, N, dtype=torch.float32)
    pair_mask = torch.ones(1, N, N, dtype=torch.float32)

    if args.cpu_only:
        print("Running CPU reference...")
        t0 = time.time()
        s_ref, z_ref, intermediates = run_cpu_reference(layer, s, z, mask, pair_mask)
        t1 = time.time()
        print(f"CPU reference: {t1 - t0:.3f}s")
        for name, vals in intermediates:
            sv = vals["s"]
            zv = vals["z"]
            print(
                f"  After {name}: s mean={sv.mean():.6f} std={sv.std():.6f} | "
                f"z mean={zv.mean():.6f} std={zv.std():.6f}"
            )
        print(f"\nFinal s: mean={s_ref.mean():.6f}, std={s_ref.std():.6f}")
        print(f"Final z: mean={z_ref.mean():.6f}, std={z_ref.std():.6f}")
        print("\nCPU-only mode, exiting.")
        return

    # Import Neuron packages
    try:
        import torch_neuronx
        import torch_xla.core.xla_model as xm
    except ImportError as e:
        print(f"Cannot import Neuron packages: {e}")
        print("Run this script on the trn2 instance.")
        return

    device = xm.xla_device()
    print(f"XLA device: {device}")

    # Extract and reshape weights
    print("Extracting weights...")
    weights_bf16 = extract_all_weights(layer)

    # Move weights to device
    for key in weights_bf16:
        weights_bf16[key] = weights_bf16[key].to(device)

    run_full_kernel(layer, s, z, mask, pair_mask, weights_bf16, device, N, C_z, C_s)


if __name__ == "__main__":
    main()
