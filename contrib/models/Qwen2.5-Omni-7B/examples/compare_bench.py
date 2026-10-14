#!/usr/bin/env python3
"""Compare per-module bench JSONs from Neuron and GPU.

Usage:
  python examples/compare_bench.py \\
      --neuron bench_neuron.json --gpu bench_gpu.json \\
      [--md bench_compare.md]

Reads the two JSON files written by ``bench_modules_neuron.py`` and
``bench_modules_gpu.py`` and prints a markdown table comparing the
median wall time of each top-level module (Thinker / Talker / DiT /
BigVGAN). Speedup is reported as ``GPU / Neuron`` so values >1 mean
Neuron is faster.
"""

import argparse
import json
import sys
from pathlib import Path


MODULE_ORDER = ["thinker", "talker", "dit", "bigvgan"]


def _fmt_seconds(v):
    if v is None:
        return "-"
    if v >= 1.0:
        return f"{v:.3f}s"
    return f"{v * 1000:.1f}ms"


def _fmt_ratio(num, den):
    if num is None or den is None or den == 0:
        return "-"
    return f"{num / den:.2f}x"


def _module_row(name, neuron, gpu):
    n_med = neuron.get("median_s") if neuron else None
    g_med = gpu.get("median_s") if gpu else None

    detail_parts = []
    for src, label in ((neuron, "neuron"), (gpu, "gpu")):
        if not src:
            continue
        if "tpot_ms" in src:
            detail_parts.append(f"{label} tpot={src['tpot_ms']}ms")
        if "per_step_ms" in src:
            detail_parts.append(f"{label} per_step={src['per_step_ms']}ms")
        if "generated_tokens_last_run" in src:
            detail_parts.append(f"{label} ntok={src['generated_tokens_last_run']}")
        if "bucket_used" in src and label == "neuron":
            detail_parts.append(f"bucket={src['bucket_used']}")
    detail = "; ".join(detail_parts) if detail_parts else ""

    if neuron and "skipped" in neuron:
        return [name, "SKIP", _fmt_seconds(g_med), "-", neuron["skipped"]]
    if gpu and "skipped" in gpu:
        return [name, _fmt_seconds(n_med), "SKIP", "-", gpu["skipped"]]

    return [
        name,
        _fmt_seconds(n_med),
        _fmt_seconds(g_med),
        _fmt_ratio(g_med, n_med),
        detail,
    ]


def _render_md(neuron, gpu):
    lines = []
    lines.append("# Qwen2.5-Omni Per-Module Bench Comparison\n")
    lines.append(
        f"- **Neuron:** {neuron.get('device', '?')} "
        f"(dtype={neuron.get('dtype', '?')}, "
        f"visible_cores={neuron.get('visible_cores', '?')})"
    )
    lines.append(
        f"- **GPU:** {gpu.get('device', '?')} "
        f"(dtype={gpu.get('dtype', '?')}, "
        f"attn_impl={gpu.get('attn_impl', '?')})"
    )
    lines.append("")
    lines.append("| Module | Neuron median | GPU median | GPU / Neuron | Notes |")
    lines.append("|--------|---------------|------------|--------------|-------|")
    for name in MODULE_ORDER:
        n = neuron.get("modules", {}).get(name)
        g = gpu.get("modules", {}).get(name)
        if not n and not g:
            continue
        row = _module_row(name, n, g)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--neuron", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--md", default=None,
                   help="Optional path to write the markdown table to.")
    args = p.parse_args()

    neuron_path = Path(args.neuron)
    gpu_path = Path(args.gpu)
    if not neuron_path.exists():
        sys.exit(f"Missing {neuron_path}")
    if not gpu_path.exists():
        sys.exit(f"Missing {gpu_path}")

    with open(neuron_path) as f:
        neuron = json.load(f)
    with open(gpu_path) as f:
        gpu = json.load(f)

    md = _render_md(neuron, gpu)
    print(md)
    if args.md:
        Path(args.md).write_text(md)
        print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
