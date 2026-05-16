#!/usr/bin/env python3
"""Compile FULL PairformerLayer SPMD grid=[2] mega-kernel at a given N.

This compiles the full_pairformer_layer_spmd kernel which covers ALL 7
sub-operations of a PairformerLayer:
  Step 1:  s = s + PairBiasAttn(s, z)
  Step 2:  z = z + TriMulOut(z)
  Step 3:  z = z + TriMulIn(z)
  Step 4:  z = z + TriAttnStart(z)
  Step 5:  z = z + TriAttnEnd(z)
  Step 6a: s = s + Transition_s(s)
  Step 6b: z = z + Transition_z(z)

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_RT_VISIBLE_CORES=0 \
        python compile_full_layer_spmd.py --N 128
"""

import argparse
import glob
import os
import time

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
os.environ.setdefault("NEURON_CC_FLAGS", "--model-type transformer")

import torch
import torch_xla.core.xla_model as xm

from boltz.main import (
    Boltz2,
    PairformerArgsV2,
    Boltz2DiffusionParams,
    MSAModuleArgs,
    BoltzSteeringParams,
)
from dataclasses import asdict

from full_pairformer_layer_spmd import full_pairformer_layer_spmd

P_MAX = 128


def extract_all_weights(layer):
    """Extract ALL weights needed for the full PairformerLayer mega-kernel.

    Returns a dict of (buf_name, attr_path) that maps kernel parameter names
    to PyTorch attribute paths on the layer object.

    Attribute paths use V2 conventions:
      - PairBiasAttention: layer.pre_norm_s (s LN), layer.attention.* (projections)
      - Z LN for PBA is inside layer.attention.proj_z (Sequential: [0]=LN, [1]=Linear)
      - Transition_s/z: layer.transition_s/z.* (no bias on fc1/fc2/fc3)
    """
    weight_map = [
        # ---- PairBiasAttention (Step 1) ----
        # s LayerNorm (pre_norm_s on PairformerLayer itself)
        ("pba_norm_s_w", "pre_norm_s.weight"),
        ("pba_norm_s_b", "pre_norm_s.bias"),
        # z LayerNorm (inside attention.proj_z Sequential, index 0)
        ("pba_norm_z_w", "attention.proj_z.0.weight"),
        ("pba_norm_z_b", "attention.proj_z.0.bias"),
        # Q projection (has bias)
        ("pba_q_w", "attention.proj_q.weight"),
        ("pba_q_b", "attention.proj_q.bias"),
        # K, V, gate, output projections (no bias)
        ("pba_k_w", "attention.proj_k.weight"),
        ("pba_v_w", "attention.proj_v.weight"),
        # z bias projection (inside proj_z Sequential, index 1)
        ("pba_z_w", "attention.proj_z.1.weight"),
        # gate and output
        ("pba_g_w", "attention.proj_g.weight"),
        ("pba_o_w", "attention.proj_o.weight"),
        # ---- TriMulOut (Step 2) ----
        ("tmul_out_norm_in_w", "tri_mul_out.norm_in.weight"),
        ("tmul_out_norm_in_b", "tri_mul_out.norm_in.bias"),
        ("tmul_out_p_in_w", "tri_mul_out.p_in.weight"),
        ("tmul_out_g_in_w", "tri_mul_out.g_in.weight"),
        ("tmul_out_norm_out_w", "tri_mul_out.norm_out.weight"),
        ("tmul_out_norm_out_b", "tri_mul_out.norm_out.bias"),
        ("tmul_out_p_out_w", "tri_mul_out.p_out.weight"),
        ("tmul_out_g_out_w", "tri_mul_out.g_out.weight"),
        # ---- TriMulIn (Step 3) ----
        ("tmul_in_norm_in_w", "tri_mul_in.norm_in.weight"),
        ("tmul_in_norm_in_b", "tri_mul_in.norm_in.bias"),
        ("tmul_in_p_in_w", "tri_mul_in.p_in.weight"),
        ("tmul_in_g_in_w", "tri_mul_in.g_in.weight"),
        ("tmul_in_norm_out_w", "tri_mul_in.norm_out.weight"),
        ("tmul_in_norm_out_b", "tri_mul_in.norm_out.bias"),
        ("tmul_in_p_out_w", "tri_mul_in.p_out.weight"),
        ("tmul_in_g_out_w", "tri_mul_in.g_out.weight"),
        # ---- TriAttnStart (Step 4) ----
        ("tatt_s_ln_w", "tri_att_start.layer_norm.weight"),
        ("tatt_s_ln_b", "tri_att_start.layer_norm.bias"),
        ("tatt_s_bias_proj_w", "tri_att_start.linear.weight"),
        ("tatt_s_q_w", "tri_att_start.mha.linear_q.weight"),
        ("tatt_s_k_w", "tri_att_start.mha.linear_k.weight"),
        ("tatt_s_v_w", "tri_att_start.mha.linear_v.weight"),
        ("tatt_s_g_w", "tri_att_start.mha.linear_g.weight"),
        ("tatt_s_o_w", "tri_att_start.mha.linear_o.weight"),
        # ---- TriAttnEnd (Step 5) ----
        ("tatt_e_ln_w", "tri_att_end.layer_norm.weight"),
        ("tatt_e_ln_b", "tri_att_end.layer_norm.bias"),
        ("tatt_e_bias_proj_w", "tri_att_end.linear.weight"),
        ("tatt_e_q_w", "tri_att_end.mha.linear_q.weight"),
        ("tatt_e_k_w", "tri_att_end.mha.linear_k.weight"),
        ("tatt_e_v_w", "tri_att_end.mha.linear_v.weight"),
        ("tatt_e_g_w", "tri_att_end.mha.linear_g.weight"),
        ("tatt_e_o_w", "tri_att_end.mha.linear_o.weight"),
        # ---- Transition_s (Step 6a) — no bias on fc1/fc2/fc3 ----
        ("trans_s_norm_w", "transition_s.norm.weight"),
        ("trans_s_norm_b", "transition_s.norm.bias"),
        ("trans_s_fc1_w", "transition_s.fc1.weight"),
        ("trans_s_fc2_w", "transition_s.fc2.weight"),
        ("trans_s_fc3_w", "transition_s.fc3.weight"),
        # ---- Transition_z (Step 6b) — no bias on fc1/fc2/fc3 ----
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
            # LayerNorm weights: tile to [P_MAX, F] where each row is the same
            v = v.unsqueeze(0).expand(P_MAX, -1).contiguous()
        w[buf_name] = v

    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=128)
    args = parser.parse_args()

    N = args.N
    C_z = 128
    C_s = 384
    H_z = 4
    H_s = 16
    n_flat = N * N

    print(f"{'=' * 70}")
    print(f"Compile FULL PairformerLayer SPMD grid=[2] mega-kernel for N={N}")
    print(f"  z scratch HBM (shared): {6 * n_flat * C_z * 2 / 1024 / 1024:.1f} MB")
    print(f"  z bias HBM (shared):    {n_flat * H_z * 2 / 1024 / 1024:.1f} MB")
    print(f"  s scratch (Q/K/V/gate): {4 * N * C_s * 2 / 1024 / 1024:.1f} MB")
    print(f"  z→s bias scratch:       {n_flat * H_s * 2 / 1024 / 1024:.1f} MB")
    print(f"  s intermediate:         {N * C_s * 2 / 1024 / 1024:.3f} MB")
    print(f"  SPMD: 2 physical cores, work split per phase")
    print(f"{'=' * 70}")

    # Load model
    print("\nLoading model...")
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
    layer = boltz.pairformer_module.layers[0]

    # Extract all weights
    print("Extracting weights...")
    device = xm.xla_device()
    w = extract_all_weights(layer)

    # Move weights to device
    for key in w:
        w[key] = w[key].to(device)

    # Create inputs
    torch.manual_seed(42)
    s = (torch.randn(N, C_s) * 0.1).to(torch.bfloat16).to(device)
    z_flat = (torch.randn(n_flat, C_z) * 0.1).to(torch.bfloat16).to(device)
    pm_flat = torch.ones(n_flat, 1, dtype=torch.bfloat16, device=device)
    mask = torch.ones(N, 1, dtype=torch.bfloat16, device=device)

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

    # Record NEFFs before
    cache_base = "/var/tmp/neuron-compile-cache/"
    cache_dirs = (
        [os.path.join(cache_base, d) for d in os.listdir(cache_base)]
        if os.path.isdir(cache_base)
        else []
    )
    cache_dir = max(cache_dirs, key=os.path.getmtime) if cache_dirs else cache_base
    neffs_before = set(glob.glob(os.path.join(cache_dir, "*/model.neff")))

    print("\nCompiling FULL PairformerLayer SPMD grid=[2] mega-kernel...")
    print("  (2 physical cores, all 7 sub-operations)")
    print(f"  Total weight params passed: {len(w)}")

    t0 = time.time()

    # Launch with grid=[2]
    s_out, z_out = full_pairformer_layer_spmd[2](
        # Main tensors
        s,
        z_flat,
        pm_flat,
        mask,
        # PairBiasAttention weights (Step 1)
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
        # TriMulOut weights (Step 2)
        w["tmul_out_norm_in_w"],
        w["tmul_out_norm_in_b"],
        w["tmul_out_p_in_w"],
        w["tmul_out_g_in_w"],
        w["tmul_out_norm_out_w"],
        w["tmul_out_norm_out_b"],
        w["tmul_out_p_out_w"],
        w["tmul_out_g_out_w"],
        # TriMulIn weights (Step 3)
        w["tmul_in_norm_in_w"],
        w["tmul_in_norm_in_b"],
        w["tmul_in_p_in_w"],
        w["tmul_in_g_in_w"],
        w["tmul_in_norm_out_w"],
        w["tmul_in_norm_out_b"],
        w["tmul_in_p_out_w"],
        w["tmul_in_g_out_w"],
        # TriAttnStart weights (Step 4)
        w["tatt_s_ln_w"],
        w["tatt_s_ln_b"],
        w["tatt_s_bias_proj_w"],
        w["tatt_s_q_w"],
        w["tatt_s_k_w"],
        w["tatt_s_v_w"],
        w["tatt_s_g_w"],
        w["tatt_s_o_w"],
        # TriAttnEnd weights (Step 5)
        w["tatt_e_ln_w"],
        w["tatt_e_ln_b"],
        w["tatt_e_bias_proj_w"],
        w["tatt_e_q_w"],
        w["tatt_e_k_w"],
        w["tatt_e_v_w"],
        w["tatt_e_g_w"],
        w["tatt_e_o_w"],
        # Transition_s weights (Step 6a)
        w["trans_s_norm_w"],
        w["trans_s_norm_b"],
        w["trans_s_fc1_w"],
        w["trans_s_fc2_w"],
        w["trans_s_fc3_w"],
        # Transition_z weights (Step 6b)
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
        # Constants
        N=N,
    )
    xm.mark_step()
    xm.wait_device_ops()
    t1 = time.time()
    print(f"Compilation + execution: {t1 - t0:.1f}s")

    # Verify outputs
    s_out_cpu = s_out.cpu()
    z_out_cpu = z_out.cpu()
    print(f"\ns_out shape: {s_out_cpu.shape}, dtype: {s_out_cpu.dtype}")
    print(
        f"s_out range: [{s_out_cpu.float().min():.4f}, {s_out_cpu.float().max():.4f}]"
    )
    print(f"s_out mean:  {s_out_cpu.float().mean():.4f}")
    print(f"\nz_out shape: {z_out_cpu.shape}, dtype: {z_out_cpu.dtype}")
    print(
        f"z_out range: [{z_out_cpu.float().min():.4f}, {z_out_cpu.float().max():.4f}]"
    )
    print(f"z_out mean:  {z_out_cpu.float().mean():.4f}")

    # Check for NaN
    if torch.isnan(s_out_cpu).any():
        print("\nWARNING: s_out contains NaN values!")
    if torch.isnan(z_out_cpu).any():
        print("\nWARNING: z_out contains NaN values!")

    # Find new NEFF
    neffs_after = set(glob.glob(os.path.join(cache_dir, "*/model.neff")))
    new_neffs = neffs_after - neffs_before

    if new_neffs:
        for neff in sorted(new_neffs):
            size_mb = os.path.getsize(neff) / 1024 / 1024
            print(f"\nNew NEFF: {neff}")
            print(f"  Size: {size_mb:.1f} MB")
            import shutil

            dest = f"/tmp/full_layer_spmd_N{N}.neff"
            shutil.copy2(neff, dest)
            print(f"  Copied to: {dest}")
            print(f"\nBenchmark command:")
            print(
                f"  neuron-bench exec --enable-only-latency -w 5 -n 50 -f random --fixed-nc-count 1 {dest}"
            )
    else:
        print("\nNo new NEFFs found (may have used cached)")
        all_neffs = list(neffs_after)
        if all_neffs:
            all_neffs.sort(key=os.path.getmtime, reverse=True)
            neff = all_neffs[0]
            size_mb = os.path.getsize(neff) / 1024 / 1024
            print(f"Most recent NEFF: {neff}")
            print(f"  Size: {size_mb:.1f} MB")
            import shutil

            dest = f"/tmp/full_layer_spmd_N{N}.neff"
            shutil.copy2(neff, dest)
            print(f"  Copied to: {dest}")
            print(f"\nBenchmark command:")
            print(
                f"  neuron-bench exec --enable-only-latency -w 5 -n 50 -f random --fixed-nc-count 1 {dest}"
            )


if __name__ == "__main__":
    main()
