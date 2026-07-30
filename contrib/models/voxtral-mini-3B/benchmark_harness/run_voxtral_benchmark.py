"""Voxtral serial latency harness.

Reads a manifest of audio clips, calls `Backend.transcribe(audio_path,
audio_wave)` on each, records `latency_sec`, writes results CSV.

Usage:
    python run_voxtral_benchmark.py \\
        --backend nxdi_neuron \\
        --manifest dataset/manifest.csv \\
        --model-dir /mnt/models/Voxtral-Mini-3B-2507 \\
        --compiled-dir /mnt/models/compiled/voxtral_mini_3b \\
        --tp-degree 4 --seq-len 512 --n-positions 768 --ods \\
        --output-dir results/ --runs 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from backends import load_backend  # noqa: E402
from common.audio import load_audio_16k  # noqa: E402
from common.manifest import read_manifest, write_results  # noqa: E402
from common.timing import neuron_sync  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="nxdi_neuron")
    p.add_argument(
        "--manifest",
        type=Path,
        default=_HERE / "dataset" / "manifest.csv",
    )
    p.add_argument("--output-dir", type=Path, default=_HERE / "results")
    p.add_argument("--runs", type=int, default=1,
                   help="Number of measurement runs.  --runs 3 writes "
                        "run_1.csv, run_2.csv, run_3.csv.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N rows (validation).")
    p.add_argument("--skip-warmup", action="store_true",
                   help="Skip the untimed warm-up transcription.")
    p.add_argument("--max-new-tokens", type=int, default=256)

    # NxDI backend args
    p.add_argument("--model-dir", default="/mnt/models/Voxtral-Mini-3B-2507")
    p.add_argument("--compiled-dir", default="/mnt/models/compiled/voxtral_mini_3b")
    p.add_argument("--tp-degree", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512,
                   help="CTE bucket size.  Use 2048 on SDK 2.31; 512 on SDK 2.30.")
    p.add_argument("--n-positions", type=int, default=768,
                   help="KV cache size.  Matches audio prefill length.")
    p.add_argument("--ods", action="store_true",
                   help="Enable on-device sampling (greedy).")
    p.add_argument("--no-move-trace", action="store_true",
                   help="Disable torch_neuronx.move_trace_to_device on the "
                        "encoder (falls back to async_load).")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def build_backend(args: argparse.Namespace):
    if args.backend == "nxdi_neuron":
        return load_backend(
            "nxdi_neuron",
            model_dir=args.model_dir,
            compiled_dir=args.compiled_dir,
            tp_degree=args.tp_degree,
            seq_len=args.seq_len,
            n_positions=args.n_positions,
            dtype=args.dtype,
            on_device_sampling=args.ods,
            move_trace_to_device=not args.no_move_trace,
            max_new_tokens=args.max_new_tokens,
        )
    raise ValueError(f"Unknown --backend {args.backend!r}")


def run_once(backend, source_fields, rows, output_path: Path, args) -> float:
    """Run the whole manifest once, write output CSV, return mean latency."""
    latencies = []
    output_rows = []
    for i, row in enumerate(rows):
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = (args.manifest.parent / audio_path).resolve()

        try:
            audio_wave = load_audio_16k(audio_path)
        except FileNotFoundError:
            print(f"  [{i + 1}/{len(rows)}] MISSING: {audio_path}", flush=True)
            output_rows.append({**row, "transcript_hyp": "",
                                "latency_sec": ""})
            write_results(output_path, source_fields, output_rows)
            continue

        neuron_sync()
        t0 = time.perf_counter()
        text = backend.transcribe(audio_path, audio_wave)
        neuron_sync()
        latency = time.perf_counter() - t0

        latencies.append(latency)
        out_row = {**row, "transcript_hyp": text,
                   "latency_sec": f"{latency:.6f}"}
        # Merge any per-phase stats the backend exposes
        for k, v in getattr(backend, "last_stats", {}).items():
            out_row[k] = f"{v:.3f}" if isinstance(v, float) else str(v)
        output_rows.append(out_row)

        # Stream progress
        write_results(output_path, source_fields, output_rows)
        print(
            f"  [{i + 1}/{len(rows)}] {audio_path.name}: "
            f"{latency * 1000:6.1f} ms",
            flush=True,
        )

    mean_ms = 1000.0 * sum(latencies) / len(latencies) if latencies else 0.0
    print(f"  Mean latency: {mean_ms:.1f} ms")
    return mean_ms


def main() -> int:
    args = parse_args()
    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        print("       See dataset/README.md for how to populate.",
              file=sys.stderr)
        return 1

    source_fields, rows = read_manifest(args.manifest)
    if not rows:
        print(f"ERROR: manifest {args.manifest} has no data rows.  "
              "See dataset/README.md.", file=sys.stderr)
        return 1
    if args.limit:
        rows = rows[: args.limit]

    print(f"Loading backend {args.backend!r}...")
    backend = build_backend(args)

    # Warmup (untimed)
    if not args.skip_warmup:
        first_audio = Path(rows[0]["audio_path"])
        if not first_audio.is_absolute():
            first_audio = (args.manifest.parent / first_audio).resolve()
        if first_audio.exists():
            print(f"Warmup: {first_audio.name}...")
            audio = load_audio_16k(first_audio)
            _ = backend.warmup(first_audio, audio)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    means_ms = []
    for run_index in range(1, args.runs + 1):
        output_path = (args.output_dir /
                       ("latency.csv" if args.runs == 1
                        else f"run_{run_index}.csv"))
        print(f"\n=== Run {run_index}/{args.runs} -> {output_path} ===")
        mean_ms = run_once(backend, source_fields, rows, output_path, args)
        means_ms.append(mean_ms)

    if args.runs > 1:
        overall = sum(means_ms) / len(means_ms)
        print(f"\nAll runs mean-of-means: {overall:.1f} ms/file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
