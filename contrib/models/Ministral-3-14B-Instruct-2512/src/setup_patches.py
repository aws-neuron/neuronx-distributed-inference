#!/usr/bin/env python3
"""
Setup patches for Ministral-3-14B-Instruct-2512 (Leanstral) on SDK 2.29.

Applies to: Ministral 14B (Mistral3ForConditionalGeneration, 32Q/8KV at TP=4).

Applies all required patches to a fresh DLAMI 20260410 (SDK 2.29) installation:
  1. Mistral rms_norm_eps pass-through (NxDI)
  2. nkilib QKV CTE eps guard
  3. neuronxcc QKV CTE eps guard
  4. QKV weight fusion in convert_hf_to_neuron_state_dict (NxDI)
  5. Fused RMSNorm in Mistral decoder forward (NxDI)
  6. Multi-KV TKG kernel + adapter (nkilib + NxDI)

Usage:
    python setup_patches.py [--venv /path/to/venv] [--dry-run]
"""

import argparse
import os
import re
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
        "modeling_mistral": os.path.join(
            sp,
            "neuronx_distributed_inference",
            "models",
            "mistral",
            "modeling_mistral.py",
        ),
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
# Patch 1: Mistral rms_norm_eps pass-through
# ---------------------------------------------------------------------------


def patch_rms_norm_eps(paths, dry_run=False):
    """Add rms_norm_eps=config.rms_norm_eps to NeuronMistralAttention super().__init__()."""
    fpath = paths["modeling_mistral"]
    content = read(fpath)

    if "rms_norm_eps=config.rms_norm_eps" in content:
        print("  [1] rms_norm_eps: already patched")
        return True

    # Find the super().__init__ call in NeuronMistralAttention and add rms_norm_eps
    idx = content.find("class NeuronMistralAttention")
    if idx == -1:
        print("  [1] rms_norm_eps: ERROR - cannot find NeuronMistralAttention class")
        return False
    # Find the super().__init__ after this class
    super_idx = content.find("super().__init__(", idx)
    if super_idx == -1:
        print("  [1] rms_norm_eps: ERROR - cannot find super().__init__ call")
        return False
    # Find the closing paren (handle nested parens like getattr())
    paren_depth = 0
    end_idx = super_idx + len("super().__init__(")
    for i in range(end_idx, len(content)):
        if content[i] == "(":
            paren_depth += 1
        elif content[i] == ")":
            if paren_depth == 0:
                end_idx = i
                break
            paren_depth -= 1

    # Insert rms_norm_eps before the closing paren
    call_content = content[super_idx:end_idx]
    if "rms_norm_eps" not in call_content:
        # Add after the last parameter, with proper formatting
        insert = ",\n            rms_norm_eps=config.rms_norm_eps"
        new_content = (
            content[:end_idx].rstrip() + insert + "\n        " + content[end_idx:]
        )
        if not dry_run:
            backup(fpath)
            write(fpath, new_content)
        print("  [1] rms_norm_eps: PATCHED")
        return True
    else:
        print("  [1] rms_norm_eps: already present in super().__init__")
        return True


# ---------------------------------------------------------------------------
# Patch 2: nkilib QKV CTE eps guard
# ---------------------------------------------------------------------------


def patch_nkilib_eps(paths, dry_run=False):
    """Guard nisa.memset for norm_eps when norm_eps=None."""
    fpath = paths["nkilib_qkv_cte"]
    content = read(fpath)

    if "norm_eps if norm_eps is not None else 0" in content:
        print("  [2] nkilib eps guard: already patched")
        return True

    # Find: nisa.memset(dst=norm_eps_sb, value=norm_eps)
    old = "value=norm_eps)"
    new = "value=norm_eps if norm_eps is not None else 0)"

    if old in content:
        if not dry_run:
            backup(fpath)
            # Replace only the first occurrence in the relevant context
            content = content.replace(old, new, 1)
            write(fpath, content)
        print("  [2] nkilib eps guard: PATCHED")
        return True

    print("  [2] nkilib eps guard: ERROR - target not found")
    return False


# ---------------------------------------------------------------------------
# Patch 3: neuronxcc QKV CTE eps guard
# ---------------------------------------------------------------------------


def patch_neuronxcc_eps(paths, dry_run=False):
    """Guard bias_eps[...] = eps when eps=None."""
    fpath = paths["neuronxcc_qkv_cte"]
    content = read(fpath)

    if "if eps is not None:" in content and "bias_eps" in content:
        print("  [3] neuronxcc eps guard: already patched")
        return True

    # Find: bias_eps[...] = eps (without an if guard)
    # Note: neuronxcc uses 2-space indentation
    # Try both 2-space and 4-space indentation patterns
    old_2sp = "  bias_eps[...] = eps"
    new_2sp = "  if eps is not None:\n    bias_eps[...] = eps"
    old_4sp = "    bias_eps[...] = eps"
    new_4sp = "    if eps is not None:\n        bias_eps[...] = eps"

    if old_2sp in content and "if eps is not None:" not in content:
        old = old_2sp
        new = new_2sp
    elif old_4sp in content and "if eps is not None:" not in content:
        old = old_4sp
        new = new_4sp
    else:
        old = None
        new = None

    if "if eps is not None:" in content:
        print("  [3] neuronxcc eps guard: already patched")
        return True

    if old is not None:
        if not dry_run:
            backup(fpath)
            content = content.replace(old, new, 1)
            write(fpath, content)
        print("  [3] neuronxcc eps guard: PATCHED")
        return True

    print("  [3] neuronxcc eps guard: ERROR - target not found")
    return False


# ---------------------------------------------------------------------------
# Patch 4: QKV weight fusion
# ---------------------------------------------------------------------------


def patch_fused_qkv(paths, dry_run=False):
    """Add QKV weight fusion to convert_hf_to_neuron_state_dict for Mistral."""
    fpath = paths["modeling_mistral"]
    content = read(fpath)

    if "Fuse Q/K/V weights into Wqkv" in content:
        print("  [4] fused_qkv: already patched")
        return True

    # Find the return state_dict in convert_hf_to_neuron_state_dict
    # We need to insert the fusion code just before "return state_dict"
    # in the convert_hf_to_neuron_state_dict function
    func_marker = "def convert_hf_to_neuron_state_dict"
    func_idx = content.find(func_marker)
    if func_idx == -1:
        print("  [4] fused_qkv: ERROR - cannot find convert_hf_to_neuron_state_dict")
        return False

    # Find the last "return state_dict" after this function
    # (there may be multiple return statements; we want the final one in this function)
    return_pattern = "        return state_dict"
    last_return_idx = content.rfind(return_pattern, func_idx)
    if last_return_idx == -1:
        print("  [4] fused_qkv: ERROR - cannot find 'return state_dict'")
        return False

    fusion_code = """
        # Fuse Q/K/V weights into Wqkv when fused_qkv is enabled
        if getattr(neuron_config, "fused_qkv", False):
            import torch as _torch_fqkv
            for i in range(num_layers):
                q_key = f"layers.{i}.self_attn.q_proj.weight"
                k_key = f"layers.{i}.self_attn.k_proj.weight"
                v_key = f"layers.{i}.self_attn.v_proj.weight"
                if q_key in state_dict and k_key in state_dict and v_key in state_dict:
                    q_w = state_dict.pop(q_key)
                    k_w = state_dict.pop(k_key)
                    v_w = state_dict.pop(v_key)
                    fused_key = f"layers.{i}.self_attn.qkv_proj.Wqkv.weight"
                    state_dict[fused_key] = _torch_fqkv.cat([q_w, k_w, v_w], dim=0)
                    # Also handle biases if present
                    q_bias_key = f"layers.{i}.self_attn.q_proj.bias"
                    k_bias_key = f"layers.{i}.self_attn.k_proj.bias"
                    v_bias_key = f"layers.{i}.self_attn.v_proj.bias"
                    if q_bias_key in state_dict:
                        fused_bias_key = f"layers.{i}.self_attn.qkv_proj.Wqkv.bias"
                        state_dict[fused_bias_key] = _torch_fqkv.cat([
                            state_dict.pop(q_bias_key),
                            state_dict.pop(k_bias_key),
                            state_dict.pop(v_bias_key),
                        ], dim=0)

"""

    new_content = content[:last_return_idx] + fusion_code + content[last_return_idx:]
    if not dry_run:
        backup(fpath)
        write(fpath, new_content)
    print("  [4] fused_qkv: PATCHED")
    return True


# ---------------------------------------------------------------------------
# Patch 5: Fused RMSNorm in decoder forward
# ---------------------------------------------------------------------------


def patch_fused_rmsnorm(paths, dry_run=False):
    """Add rmsnorm=self.input_layernorm to attention call in Mistral decoder forward."""
    fpath = paths["modeling_mistral"]
    content = read(fpath)

    if "rmsnorm=self.input_layernorm" in content:
        print("  [5] fused_rmsnorm: already patched")
        return True

    # Find the attention call in the decoder forward method that does NOT have rmsnorm
    # This varies by SDK version. We look for the self.self_attn( call in the forward method.
    # The Llama model passes rmsnorm=self.input_layernorm; Mistral does not.

    # Strategy: find "hidden_states = self.self_attn(" in a decoder forward method
    # and add rmsnorm parameter

    # Look for the pattern where self_attn is called with hidden_states
    attn_call_pattern = "self.self_attn(\n"
    idx = content.find(attn_call_pattern)
    if idx == -1:
        attn_call_pattern = "self.self_attn("
        idx = content.find(attn_call_pattern)

    if idx == -1:
        print(
            "  [5] fused_rmsnorm: WARNING - cannot find self.self_attn call, skipping"
        )
        return True  # Non-fatal -- fused_rmsnorm is optional

    # Find the closing paren of the self_attn call
    paren_depth = 0
    start = idx + len("self.self_attn(")
    for i in range(start, len(content)):
        if content[i] == "(":
            paren_depth += 1
        elif content[i] == ")":
            if paren_depth == 0:
                # Insert rmsnorm before closing paren
                call_body = content[start:i]
                if "rmsnorm" not in call_body:
                    # Find the indentation of the existing params
                    # Look back from ')' to find the indentation level
                    # We need to add rmsnorm as a kwarg at the same indent level
                    # Strategy: find the last newline before the closing ')' to get indent
                    last_nl = content.rfind("\n", start, i)
                    if last_nl != -1:
                        # Get indent of closing paren
                        close_indent = ""
                        for c in content[last_nl + 1 : i]:
                            if c in " \t":
                                close_indent += c
                            else:
                                break
                        # Param indent is typically close_indent + 4 spaces
                        param_indent = close_indent + "    "
                    else:
                        param_indent = "            "
                        close_indent = "        "
                    insert = (
                        f"{param_indent}rmsnorm=self.input_layernorm,\n{close_indent}"
                    )
                    # Replace the closing paren and its preceding whitespace
                    # Find where the whitespace before ')' starts
                    ws_start = i
                    while ws_start > start and content[ws_start - 1] in " \t\n":
                        ws_start -= 1
                    # Check if there's already a trailing comma
                    pre_ws = content[ws_start - 1] if ws_start > start else ""
                    if pre_ws == ",":
                        new_content = content[:ws_start] + "\n" + insert + content[i:]
                    else:
                        new_content = content[:ws_start] + ",\n" + insert + content[i:]
                    if not dry_run:
                        backup(fpath)
                        write(fpath, new_content)
                    print("  [5] fused_rmsnorm: PATCHED")
                else:
                    print("  [5] fused_rmsnorm: already present")
                return True
            paren_depth -= 1

    print("  [5] fused_rmsnorm: WARNING - could not find closing paren, skipping")
    return True


# ---------------------------------------------------------------------------
# Patch 6: Multi-KV TKG kernel + adapter
# ---------------------------------------------------------------------------


def _fix_nki030_kernel(fpath):
    """Fix kernel for NKI 0.3.0: remove *, and add defaults to params after first defaulted."""
    content = read(fpath)
    if "*," not in content:
        return  # Already fixed

    content = content.replace("    *,\n", "")

    # Find function signature
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

        # Add default based on type annotation
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
    # Step 6a: Copy the forked kernel to nkilib
    kernel_src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "attention_block_tkg_multi_kv.py"
    )
    kernel_dst = os.path.join(
        paths["nkilib_transformer"], "attention_block_tkg_multi_kv.py"
    )

    if not os.path.exists(kernel_src):
        print(f"  [6a] multi-KV kernel: ERROR - source not found at {kernel_src}")
        return False

    if not dry_run:
        shutil.copy2(kernel_src, kernel_dst)
    print(f"  [6a] multi-KV kernel: copied to nkilib")

    # Step 6a.1: Apply NKI 0.3.0 compatibility fix to the kernel
    # NKI 0.3.0 does not support keyword-only arguments (after *,)
    # We remove *, and add defaults to params that need them
    if not dry_run:
        _fix_nki030_kernel(kernel_dst)
    print(f"  [6a] multi-KV kernel: NKI 0.3.0 fix applied")

    # Step 6b: Apply adapter monkeypatch to attention_base.py
    fpath = paths["attention_base"]
    content = read(fpath)

    PATCH_MARKER = "# MULTI_KV_TKG_PATCH_APPLIED"
    if PATCH_MARKER in content:
        print("  [6b] multi-KV adapter: already patched")
        return True

    # Read the adapter code from our local file
    adapter_src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "multi_kv_adapter.py"
    )

    if os.path.exists(adapter_src):
        adapter_code = read(adapter_src)
    else:
        print(f"  [6b] multi-KV adapter: ERROR - source not found at {adapter_src}")
        return False

    if not dry_run:
        backup(fpath)
        with open(fpath, "a") as f:
            f.write("\n\n" + PATCH_MARKER + "\n")
            f.write(adapter_code)
    print("  [6b] multi-KV adapter: PATCHED")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Apply Mistral NKI optimization patches for Ministral-3-14B (Leanstral)"
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
        "--skip-tkg",
        action="store_true",
        help="Skip TKG kernel patches (patches 1-5 only, for baseline NKI QKV testing)",
    )
    args = parser.parse_args()

    venv = args.venv
    sp = os.path.join(venv, "lib", "python3.12", "site-packages")
    if not os.path.isdir(sp):
        print(f"ERROR: site-packages not found at {sp}")
        print("Make sure you're running on a DLAMI 20260410 (SDK 2.29) instance")
        sys.exit(1)

    paths = resolve_paths(venv)
    print(f"Patching SDK 2.29 at: {venv}")
    if args.dry_run:
        print("(DRY RUN - no files will be modified)\n")
    else:
        print()

    results = []
    results.append(("rms_norm_eps", patch_rms_norm_eps(paths, args.dry_run)))
    results.append(("nkilib_eps", patch_nkilib_eps(paths, args.dry_run)))
    results.append(("neuronxcc_eps", patch_neuronxcc_eps(paths, args.dry_run)))
    results.append(("fused_qkv", patch_fused_qkv(paths, args.dry_run)))
    results.append(("fused_rmsnorm", patch_fused_rmsnorm(paths, args.dry_run)))

    if not args.skip_tkg:
        results.append(("multi_kv_tkg", patch_multi_kv_tkg(paths, args.dry_run)))

    print("\n--- Summary ---")
    ok = all(r[1] for r in results)
    for name, success in results:
        print(f"  {name}: {'OK' if success else 'FAILED'}")

    if ok:
        print("\nAll patches applied successfully.")
        if not args.skip_tkg:
            print("\nTo start vLLM with full NKI optimization:")
            print("  python -m vllm.entrypoints.openai.api_server \\")
            print("    --model /path/to/Ministral-3-14B-Instruct-2512 \\")
            print("    --tensor-parallel-size 4 --max-model-len 4096 \\")
            print("    --max-num-seqs 1 --no-enable-prefix-caching \\")
            print("    --block-size 8 \\")
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
