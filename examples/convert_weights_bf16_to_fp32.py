#!/usr/bin/env python3
"""Convert BF16 sharded weights to FP32, splitting across two drives.

Reads each per-rank safetensor from the BF16 source directory, casts all
tensors to float32, and writes to the output directory. Files are distributed
across two output drives to manage disk space. A unified output directory
is created with symlinks.

Usage:
    python examples/convert_weights_bf16_to_fp32.py \
        --src /scratch2/deepseek_v3_logit/weights \
        --dst /scratch4/deepseek_v3_fp32/weights \
        --overflow /scratch0/deepseek_v3_fp32_overflow \
        --overflow-after 38
"""

import argparse
import os
import time

from safetensors.torch import load_file, save_file
import torch


def convert_one(src_path, dst_path):
    """Load a BF16 safetensor, cast to FP32, save."""
    state_dict = load_file(src_path)
    fp32_dict = {}
    for k, v in state_dict.items():
        if v.dtype == torch.bfloat16:
            fp32_dict[k] = v.to(torch.float32)
        elif v.dtype == torch.float16:
            fp32_dict[k] = v.to(torch.float32)
        else:
            fp32_dict[k] = v
    del state_dict
    save_file(fp32_dict, dst_path)
    del fp32_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="BF16 sharded weights dir")
    parser.add_argument("--dst", required=True, help="Primary FP32 output dir")
    parser.add_argument("--overflow", default=None, help="Secondary dir when primary fills up")
    parser.add_argument("--overflow-after", type=int, default=38,
                        help="Switch to overflow dir after this many files")
    parser.add_argument("--delete-src", action="store_true",
                        help="Delete source BF16 file after successful conversion")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    if args.overflow:
        os.makedirs(args.overflow, exist_ok=True)

    files = sorted(f for f in os.listdir(args.src) if f.endswith(".safetensors"))
    print(f"Found {len(files)} safetensor files to convert")

    total_start = time.time()
    for i, fname in enumerate(files):
        src_path = os.path.join(args.src, fname)

        if args.overflow and i >= args.overflow_after:
            out_path = os.path.join(args.overflow, fname)
            # Create symlink in dst dir pointing to overflow
            link_path = os.path.join(args.dst, fname)
            if not os.path.exists(link_path):
                os.symlink(out_path, link_path)
        else:
            out_path = os.path.join(args.dst, fname)

        if os.path.exists(out_path):
            print(f"[{i+1}/{len(files)}] {fname} already exists, skipping")
            continue

        src_size = os.path.getsize(src_path) / 1e9
        t0 = time.time()
        print(f"[{i+1}/{len(files)}] Converting {fname} ({src_size:.1f} GB BF16 -> FP32)...",
              end="", flush=True)

        convert_one(src_path, out_path)

        dst_size = os.path.getsize(out_path) / 1e9
        elapsed = time.time() - t0
        print(f" {dst_size:.1f} GB in {elapsed:.1f}s")

        if args.delete_src:
            os.remove(src_path)
            print(f"  Deleted source: {src_path}")

    total_elapsed = time.time() - total_start
    print(f"\nDone! Converted {len(files)} files in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"Output: {args.dst}")


if __name__ == "__main__":
    main()
