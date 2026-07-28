#!/usr/bin/env python3
"""
Setup patches for Mistral3-family models using the LlamaForCausalLM code path on SDK 2.29.

Applies to: Ministral 3B (text-only extraction), Mistral-Small-3.1-24B (text-only extraction).
Both models use NeuronLlamaModel (modeling_llama.py) which already has:
  - head_dim from config (via getattr)
  - rms_norm_eps pass-through
  - QKV weight fusion (convert_state_dict_to_fused_qkv)
  - Fused RMSNorm in decoder forward

Only model-agnostic library patches are needed:
  1. nkilib QKV CTE eps guard
  2. neuronxcc QKV CTE eps guard
  3. TKG kernel (mode depends on model):
     - multi-kv: For models with >1 KV head per TP rank (Ministral 3B at TP=4)
     - stock: For models with 1 KV head per TP rank (Mistral-Small-24B at TP=8)
     - none: Skip TKG entirely (baseline QKV-only testing)

Usage:
    python setup_patches.py [--tkg-mode multi-kv|stock|none] [--venv /path/to/venv] [--dry-run]
"""

import argparse
import os
import shutil
import sys

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

DEFAULT_VENV = "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16"


def resolve_paths(venv_root):
    sp = os.path.join(venv_root, "lib", "python3.12", "site-packages")
    return {
        "site_packages": sp,
        "attention_base": os.path.join(
            sp,
            "neuronx_distributed_inference",
            "modules",
            "attention",
            "attention_base.py",
        ),
        "nkilib_qkv_cte": os.path.join(sp, "nkilib", "core", "qkv", "qkv_cte.py"),
        "neuronxcc_qkv_cte": os.path.join(
            sp, "neuronxcc", "nki", "_pre_prod_kernels", "qkv_cte_impl.py"
        ),
        "nkilib_transformer": os.path.join(sp, "nkilib", "experimental", "transformer"),
    }


def backup(path):
    bak = path + ".bak_contrib"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"  Backed up {os.path.basename(path)}")


def read(path):
    with open(path) as f:
        return f.read()


def write(path, content):
    with open(path, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Patch 1: nkilib QKV CTE eps guard
# ---------------------------------------------------------------------------


def patch_nkilib_eps(paths, dry_run=False):
    """Guard nisa.memset for norm_eps when norm_eps=None."""
    fpath = paths["nkilib_qkv_cte"]
    content = read(fpath)

    if "norm_eps if norm_eps is not None else 0" in content:
        print("  [1] nkilib eps guard: already patched")
        return True

    old = "value=norm_eps)"
    new = "value=norm_eps if norm_eps is not None else 0)"

    if old in content:
        if not dry_run:
            backup(fpath)
            content = content.replace(old, new, 1)
            write(fpath, content)
        print("  [1] nkilib eps guard: PATCHED")
        return True

    print("  [1] nkilib eps guard: ERROR - target not found")
    return False


# ---------------------------------------------------------------------------
# Patch 2: neuronxcc QKV CTE eps guard
# ---------------------------------------------------------------------------


def patch_neuronxcc_eps(paths, dry_run=False):
    """Guard bias_eps[...] = eps when eps=None."""
    fpath = paths["neuronxcc_qkv_cte"]
    content = read(fpath)

    if "if eps is not None:" in content and "bias_eps" in content:
        print("  [2] neuronxcc eps guard: already patched")
        return True

    old_2sp = "  bias_eps[...] = eps"
    new_2sp = "  if eps is not None:\n    bias_eps[...] = eps"
    old_4sp = "    bias_eps[...] = eps"
    new_4sp = "    if eps is not None:\n        bias_eps[...] = eps"

    if old_2sp in content and "if eps is not None:" not in content:
        old, new = old_2sp, new_2sp
    elif old_4sp in content and "if eps is not None:" not in content:
        old, new = old_4sp, new_4sp
    else:
        old, new = None, None

    if "if eps is not None:" in content:
        print("  [2] neuronxcc eps guard: already patched")
        return True

    if old is not None:
        if not dry_run:
            backup(fpath)
            content = content.replace(old, new, 1)
            write(fpath, content)
        print("  [2] neuronxcc eps guard: PATCHED")
        return True

    print("  [2] neuronxcc eps guard: ERROR - target not found")
    return False


# ---------------------------------------------------------------------------
# Patch 3: Multi-KV TKG kernel + adapter
# ---------------------------------------------------------------------------


def _fix_nki030_kernel(fpath):
    """Fix kernel for NKI 0.3.0: remove *, and add defaults."""
    content = read(fpath)
    if "*," not in content:
        return

    content = content.replace("    *,\n", "")

    func_start = content.find("def attention_block_tkg(")
    if func_start == -1:
        return
    paren_depth = 0
    in_func = False
    func_end = func_start
    for i in range(func_start, len(content)):
        if content[i] == "(":
            paren_depth += 1
            in_func = True
        elif content[i] == ")":
            paren_depth -= 1
            if in_func and paren_depth == 0:
                func_end = i
                break

    sig = content[func_start : func_end + 1]
    lines = sig.split("\n")
    new_lines = []
    seen_default = False

    for line in lines:
        stripped = line.strip()
        if (
            stripped.startswith("#")
            or stripped == ""
            or stripped.startswith("def ")
            or stripped == ")"
            or stripped == "),"
        ):
            new_lines.append(line)
            continue

        has_default = "=" in stripped and ":" in stripped
        if has_default:
            seen_default = True
            new_lines.append(line)
            continue

        if not seen_default:
            new_lines.append(line)
            continue

        if "Optional[" in stripped or ": nl.ndarray" in stripped:
            default = "None"
        elif ": bool" in stripped:
            default = "False"
        elif ": float" in stripped:
            default = "0.0"
        elif ": int" in stripped:
            default = "0"
        else:
            default = "None"

        if stripped.endswith(","):
            line = line.rstrip().rstrip(",") + f" = {default},"
        else:
            line = line.rstrip() + f" = {default}"
        new_lines.append(line)

    new_sig = "\n".join(new_lines)
    content = content[:func_start] + new_sig + content[func_end + 1 :]
    write(fpath, content)


def patch_multi_kv_tkg(paths, dry_run=False):
    """Install Leanstral forked multi-KV TKG kernel and adapter monkeypatch."""
    kernel_src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "attention_block_tkg_multi_kv.py"
    )
    kernel_dst = os.path.join(
        paths["nkilib_transformer"], "attention_block_tkg_multi_kv.py"
    )

    if not os.path.exists(kernel_src):
        print(f"  [3a] multi-KV kernel: ERROR - source not found at {kernel_src}")
        return False

    if not dry_run:
        shutil.copy2(kernel_src, kernel_dst)
    print(f"  [3a] multi-KV kernel: copied to nkilib")

    if not dry_run:
        _fix_nki030_kernel(kernel_dst)
    print(f"  [3a] multi-KV kernel: NKI 0.3.0 fix applied")

    fpath = paths["attention_base"]
    content = read(fpath)

    PATCH_MARKER = "# MULTI_KV_TKG_PATCH_APPLIED"
    if PATCH_MARKER in content:
        print("  [3b] multi-KV adapter: already patched")
        return True

    adapter_src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "multi_kv_adapter.py"
    )

    if os.path.exists(adapter_src):
        adapter_code = read(adapter_src)
    else:
        print(f"  [3b] multi-KV adapter: ERROR - source not found at {adapter_src}")
        return False

    if not dry_run:
        backup(fpath)
        with open(fpath, "a") as f:
            f.write("\n\n" + PATCH_MARKER + "\n")
            f.write(adapter_code)
    print("  [3b] multi-KV adapter: PATCHED")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Apply Mistral3-family NKI optimization patches (Llama code path)"
    )
    parser.add_argument(
        "--venv",
        default=DEFAULT_VENV,
        help=f"Path to Neuron venv (default: {DEFAULT_VENV})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be patched"
    )
    parser.add_argument(
        "--tkg-mode",
        choices=["multi-kv", "stock", "none"],
        default="multi-kv",
        help="TKG kernel mode: multi-kv (Ministral 3B, >1 KV/rank), "
        "stock (Mistral-Small-24B, 1 KV/rank), none (no TKG)",
    )
    # Keep --skip-tkg for backwards compatibility
    parser.add_argument(
        "--skip-tkg",
        action="store_true",
        help="(Deprecated) Use --tkg-mode none instead",
    )
    args = parser.parse_args()

    if args.skip_tkg:
        args.tkg_mode = "none"

    venv = args.venv
    sp = os.path.join(venv, "lib", "python3.12", "site-packages")
    if not os.path.isdir(sp):
        print(f"ERROR: site-packages not found at {sp}")
        print("Make sure you're running on a DLAMI 20260410 (SDK 2.29) instance")
        sys.exit(1)

    paths = resolve_paths(venv)
    print(f"Patching SDK 2.29 at: {venv}")
    print(f"TKG mode: {args.tkg_mode}")
    print(f"Note: Llama code path already has rms_norm_eps, fused_qkv, fused_rmsnorm")
    if args.dry_run:
        print("(DRY RUN - no files will be modified)\n")
    else:
        print()

    results = []
    results.append(("nkilib_eps", patch_nkilib_eps(paths, args.dry_run)))
    results.append(("neuronxcc_eps", patch_neuronxcc_eps(paths, args.dry_run)))

    if args.tkg_mode == "multi-kv":
        results.append(("multi_kv_tkg", patch_multi_kv_tkg(paths, args.dry_run)))
    elif args.tkg_mode == "stock":
        print("  [3] Stock TKG: no custom kernel needed (1 KV head/rank)")
        print("      Enable via vLLM --additional-config flags only")

    print("\n--- Summary ---")
    ok = all(r[1] for r in results)
    for name, success in results:
        print(f"  {name}: {'OK' if success else 'FAILED'}")

    if ok:
        print("\nAll patches applied successfully.")
        if args.tkg_mode != "none":
            print("\nTo start vLLM with full NKI optimization:")
            print("  python -m vllm.entrypoints.openai.api_server \\")
            print("    --model /mnt/models/<extracted-text-model> \\")
            if args.tkg_mode == "multi-kv":
                print("    --tensor-parallel-size 4 --max-model-len 8192 \\")
            else:
                print("    --tensor-parallel-size 8 --max-model-len 8192 \\")
            print("    --max-num-seqs 1 --no-enable-prefix-caching \\")
            print('    --additional-config \'{"override_neuron_config": {')
            print('      "fused_qkv": true, "qkv_nki_kernel_enabled": true,')
            print('      "qkv_kernel_enabled": true,')
            print('      "attn_block_tkg_nki_kernel_enabled": true,')
            print('      "attn_block_tkg_nki_kernel_cache_update": true')
            print("    }}'")
    else:
        print("\nSome patches failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
