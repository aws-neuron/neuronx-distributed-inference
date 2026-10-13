#!/usr/bin/env python3
"""Sample host RSS and Neuron device memory while a benchmark runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _sample_neuron_sysfs() -> dict[str, Any]:
    root = Path("/sys/devices/virtual/neuron_device")
    totals: dict[str, int] = {}
    cores: dict[str, dict[str, int]] = {}
    if not root.exists():
        return {"available": False, "totals_bytes": totals, "cores": cores}
    for path in root.glob("neuron*/neuron_core*/stats/memory_usage/device_mem/**/*"):
        if not path.is_file():
            continue
        value = _read_int(path)
        if value is None:
            continue
        parts = path.parts
        try:
            neuron = next(part for part in parts if part.startswith("neuron"))
            core = next(part for part in parts if part.startswith("neuron_core"))
        except StopIteration:
            continue
        category = path.parent.name if path.name == "bytes" else path.name
        key = f"{neuron}/{core}"
        cores.setdefault(key, {})[category] = value
        totals[category] = totals.get(category, 0) + value
    return {"available": True, "totals_bytes": totals, "cores": cores}


def _sample_processes(match: re.Pattern[str]) -> dict[str, Any]:
    result = subprocess.run(
        ["ps", "-eo", "pid,ppid,rss,comm,args"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    processes = []
    total_rss_kb = 0
    own_pid = os.getpid()
    for line in result.stdout.splitlines()[1:]:
        fields = line.strip().split(None, 4)
        if len(fields) < 5:
            continue
        pid, ppid, rss, command, args = fields
        if int(pid) == own_pid:
            continue
        if not match.search(args) and not match.search(command):
            continue
        rss_kb = int(rss)
        total_rss_kb += rss_kb
        processes.append(
            {
                "pid": int(pid),
                "ppid": int(ppid),
                "rss_kb": rss_kb,
                "command": command,
                "args": args,
            }
        )
    return {"processes": processes, "total_rss_kb": total_rss_kb}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument(
        "--match",
        default="VLLM::EngineCore|qwen36_.*bench|qwen36_.*sweep",
        help="Regex matched against process command names and args.",
    )
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument(
        "--stop-when-no-match",
        action="store_true",
        help="Exit once at least one matching process has been seen and later none remain.",
    )
    args = parser.parse_args()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    match = re.compile(args.match)
    start = time.time()
    stop_requested = False
    saw_process = False
    samples = 0
    peak_host_rss_kb = 0
    peak_neuron_total_bytes = 0
    peak_neuron_by_category: dict[str, int] = {}

    def _request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    with args.output_jsonl.open("a", encoding="utf-8") as handle:
        while True:
            process_sample = _sample_processes(match)
            neuron_sample = _sample_neuron_sysfs()
            now = time.time()
            row = {
                "timestamp_unix": now,
                "elapsed_seconds": now - start,
                "host": process_sample,
                "neuron": neuron_sample,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            samples += 1

            if process_sample["processes"]:
                saw_process = True
            peak_host_rss_kb = max(peak_host_rss_kb, int(process_sample["total_rss_kb"]))
            neuron_totals = neuron_sample.get("totals_bytes", {})
            current_neuron_total = sum(int(value) for value in neuron_totals.values())
            peak_neuron_total_bytes = max(peak_neuron_total_bytes, current_neuron_total)
            for category, value in neuron_totals.items():
                peak_neuron_by_category[category] = max(
                    peak_neuron_by_category.get(category, 0),
                    int(value),
                )

            if stop_requested:
                break
            if args.duration_seconds and now - start >= args.duration_seconds:
                break
            if (
                args.stop_when_no_match
                and saw_process
                and not process_sample["processes"]
            ):
                break
            time.sleep(args.interval_seconds)

    _write_json(
        args.summary_json,
        {
            "samples": samples,
            "duration_seconds": time.time() - start,
            "peak_host_rss_kb": peak_host_rss_kb,
            "peak_host_rss_gib": peak_host_rss_kb / 1024 / 1024,
            "peak_neuron_total_bytes": peak_neuron_total_bytes,
            "peak_neuron_total_gib": peak_neuron_total_bytes / 1024 / 1024 / 1024,
            "peak_neuron_by_category_bytes": peak_neuron_by_category,
            "peak_neuron_by_category_gib": {
                key: value / 1024 / 1024 / 1024
                for key, value in peak_neuron_by_category.items()
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
