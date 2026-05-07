#!/usr/bin/env python3
"""
Setup patches for sarvam-m (sarvamai/sarvam-m) on Neuron SDK 2.29.

sarvam-m is a 24B Mistral-architecture model (MistralForCausalLM) with a non-standard
head_dim configuration: hidden_size=5120, num_attention_heads=32 gives a computed
head_dim of 160, but the model config explicitly specifies head_dim=128.

NxDI's NeuronMixtralAttention (which handles Mistral models) hardcodes
head_dim = config.hidden_size // config.num_attention_heads, ignoring any explicit
head_dim in the config. This causes a shape mismatch: K/V projections output
num_kv_heads * 128 = 1024, but the KV cache is allocated for num_kv_heads * 160 = 1280.

This script applies three patches:
  1. NeuronMixtralAttention head_dim: Use getattr(config, "head_dim", ...) like Llama does
  2. nkilib QKV CTE eps guard: Handle norm_eps=None in nisa.memset
  3. neuronxcc QKV CTE eps guard: Handle eps=None in bias_eps assignment

Usage:
    python setup_patches.py [--venv /path/to/venv] [--dry-run]
"""

import argparse
import os
import shutil
import sys


DEFAULT_VENV = "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16"


def resolve_paths(venv_root):
    sp = os.path.join(venv_root, "lib", "python3.12", "site-packages")
    return {
        "site_packages": sp,
        "modeling_mixtral": os.path.join(
            sp,
            "neuronx_distributed_inference",
            "models",
            "mixtral",
            "modeling_mixtral.py",
        ),
        "nkilib_qkv_cte": os.path.join(sp, "nkilib", "core", "qkv", "qkv_cte.py"),
        "neuronxcc_qkv_cte": os.path.join(
            sp, "neuronxcc", "nki", "_pre_prod_kernels", "qkv_cte_impl.py"
        ),
    }


def backup(path):
    bak = path + ".bak_sarvam_m"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"  Backed up {os.path.basename(path)}")


def read_file(path):
    with open(path) as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Patch 1: NeuronMixtralAttention head_dim from config
# ---------------------------------------------------------------------------


def patch_mixtral_head_dim(paths, dry_run=False):
    """
    Patch NeuronMixtralAttention to read head_dim from config, matching the
    pattern used by NeuronLlamaAttention and other model implementations.

    Before (hardcoded):
        head_dim=config.hidden_size // config.num_attention_heads

    After (config-aware):
        head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    This affects the RotaryEmbedding dimension and the head_dim passed to
    NeuronAttentionBase.__init__(), which determines KV cache allocation.
    """
    fpath = paths["modeling_mixtral"]
    content = read_file(fpath)

    # Check if already patched
    if 'getattr(config, "head_dim"' in content:
        print("  [1] Mixtral head_dim: already patched")
        return True

    # The RotaryEmbedding line
    old_rope = "config.hidden_size // config.num_attention_heads,\n            max_position_embeddings=config.max_position_embeddings,"
    new_rope = 'getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),\n            max_position_embeddings=config.max_position_embeddings,'

    # The super().__init__ head_dim line
    old_super = "head_dim=config.hidden_size // config.num_attention_heads,"
    new_super = 'head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),'

    if old_rope not in content:
        print("  [1] Mixtral head_dim: ERROR - RotaryEmbedding target not found")
        return False

    if old_super not in content:
        print("  [1] Mixtral head_dim: ERROR - super().__init__ target not found")
        return False

    if not dry_run:
        backup(fpath)
        content = content.replace(old_rope, new_rope, 1)
        content = content.replace(old_super, new_super, 1)
        write_file(fpath, content)

    print("  [1] Mixtral head_dim: PATCHED (2 locations)")
    return True


# ---------------------------------------------------------------------------
# Patch 2: nkilib QKV CTE eps guard
# ---------------------------------------------------------------------------


def patch_nkilib_eps(paths, dry_run=False):
    """Guard nisa.memset for norm_eps when norm_eps=None."""
    fpath = paths["nkilib_qkv_cte"]
    content = read_file(fpath)

    if "norm_eps if norm_eps is not None else 0" in content:
        print("  [2] nkilib eps guard: already patched")
        return True

    old = "value=norm_eps)"
    new = "value=norm_eps if norm_eps is not None else 0)"

    if old in content:
        if not dry_run:
            backup(fpath)
            content = content.replace(old, new, 1)
            write_file(fpath, content)
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
    content = read_file(fpath)

    if "if eps is not None:" in content and "bias_eps" in content:
        print("  [3] neuronxcc eps guard: already patched")
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
        print("  [3] neuronxcc eps guard: already patched")
        return True

    if old is not None:
        if not dry_run:
            backup(fpath)
            content = content.replace(old, new, 1)
            write_file(fpath, content)
        print("  [3] neuronxcc eps guard: PATCHED")
        return True

    print("  [3] neuronxcc eps guard: ERROR - target not found")
    return False


# ---------------------------------------------------------------------------
# Combined apply function
# ---------------------------------------------------------------------------


def apply_patches(venv=None, dry_run=False):
    """Apply all patches. Returns True if all succeeded."""
    venv = venv or DEFAULT_VENV
    sp = os.path.join(venv, "lib", "python3.12", "site-packages")
    if not os.path.isdir(sp):
        print(f"ERROR: site-packages not found at {sp}")
        return False

    paths = resolve_paths(venv)
    print(f"Patching SDK 2.29 at: {venv}")
    if dry_run:
        print("(DRY RUN - no files will be modified)\n")
    else:
        print()

    results = []
    results.append(("mixtral_head_dim", patch_mixtral_head_dim(paths, dry_run)))
    results.append(("nkilib_eps", patch_nkilib_eps(paths, dry_run)))
    results.append(("neuronxcc_eps", patch_neuronxcc_eps(paths, dry_run)))

    print("\n--- Summary ---")
    ok = all(r[1] for r in results)
    for name, success in results:
        print(f"  {name}: {'OK' if success else 'FAILED'}")

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Apply patches for sarvam-m (Mistral head_dim + NKI eps guards)"
    )
    parser.add_argument(
        "--venv",
        default=DEFAULT_VENV,
        help=f"Path to Neuron venv (default: {DEFAULT_VENV})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be patched"
    )
    args = parser.parse_args()

    ok = apply_patches(args.venv, args.dry_run)

    if ok:
        print("\nAll patches applied successfully.")
        print("\nTo start vLLM with sarvam-m:")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print("    --model sarvamai/sarvam-m \\")
        print("    --tensor-parallel-size 8 --max-model-len 8192 \\")
        print("    --max-num-seqs 4 --no-enable-prefix-caching \\")
        print('    --additional-config \'{"override_neuron_config": {')
        print('      "qkv_nki_kernel_enabled": true,')
        print('      "qkv_kernel_enabled": true')
        print("    }}'")
    else:
        print("\nSome patches failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
