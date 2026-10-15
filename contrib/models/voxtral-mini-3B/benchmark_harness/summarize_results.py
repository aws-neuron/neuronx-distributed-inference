"""Summarize latency CSV(s) written by run_voxtral_benchmark.py.

Prints mean / median / p90 / p99 and a per-duration-bin breakdown.

Usage:
    python summarize_results.py results/
    python summarize_results.py results/latency.csv
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path


BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, math.inf)]


def read_latencies(csv_path: Path) -> list[tuple[float, float]]:
    """Return [(duration_sec, latency_sec), ...] for a single CSV."""
    out: list[tuple[float, float]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dur = float(row.get("duration_sec", 0) or 0)
                lat = float(row.get("latency_sec", "") or "nan")
            except ValueError:
                continue
            if math.isfinite(lat):
                out.append((dur, lat))
    return out


def collect(root: Path) -> list[tuple[float, float]]:
    if root.is_file():
        return read_latencies(root)
    latencies: list[tuple[float, float]] = []
    for p in sorted(root.glob("run_*.csv")):
        latencies += read_latencies(p)
    if not latencies:
        # Fall back to latency.csv
        latency_csv = root / "latency.csv"
        if latency_csv.exists():
            latencies += read_latencies(latency_csv)
    return latencies


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * p
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_v[lo]
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (k - lo)


def summarize(rows: list[tuple[float, float]]) -> None:
    if not rows:
        print("No latencies found.")
        return

    latencies = [lat for _, lat in rows]
    mean_ms = 1000.0 * statistics.mean(latencies)
    median_ms = 1000.0 * statistics.median(latencies)
    p90_ms = 1000.0 * pct(latencies, 0.90)
    p99_ms = 1000.0 * pct(latencies, 0.99)
    stdev_ms = 1000.0 * statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    print(f"N files:        {len(latencies)}")
    print(f"Mean latency:   {mean_ms:.1f} ms")
    print(f"Median latency: {median_ms:.1f} ms")
    print(f"P90 latency:    {p90_ms:.1f} ms")
    print(f"P99 latency:    {p99_ms:.1f} ms")
    print(f"Stdev:          {stdev_ms:.1f} ms")

    # Per-duration bin
    print("\nPer-duration bin (mean latency):")
    print(f"  {'bin':<10} {'n':>3}  {'mean ms':>9}")
    for lo, hi in BINS:
        bin_lats = [1000.0 * lat for dur, lat in rows if lo <= dur < hi]
        if not bin_lats:
            continue
        label = f"{int(lo)}-{int(hi) if hi != math.inf else '30+'} s"
        print(
            f"  {label:<10} {len(bin_lats):>3}  "
            f"{statistics.mean(bin_lats):>9.1f}"
        )


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python summarize_results.py <results/ or results/run_1.csv>")
        return 1
    root = Path(sys.argv[1])
    if not root.exists():
        print(f"ERROR: {root} does not exist.")
        return 1
    rows = collect(root)
    summarize(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
