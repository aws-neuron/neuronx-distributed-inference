"""Manifest reader / results-CSV writer.

Schema:  audio_path,duration_sec,transcript

The writer adds a `transcript_hyp` and `latency_sec` column (plus any
backend-specific extras via `.last_stats`).
"""

from __future__ import annotations

import csv
from pathlib import Path


def read_manifest(manifest_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read manifest.csv and return (source_fields, rows).

    Drops any pre-existing measurement columns from source_fields so we can
    add fresh ones on write.
    """
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "audio_path" not in reader.fieldnames:
            raise ValueError(f"{manifest_path} must contain an audio_path column")
        rows = list(reader)
        drop = {"latency_sec", "transcript_hyp"}
        fields = [name for name in reader.fieldnames if name not in drop]
    return fields, rows


def write_results(
    output_path: Path,
    source_fields: list[str],
    rows: list[dict[str, str]],
) -> None:
    """Write CSV atomically.  Fields = source_fields + all keys observed in rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    fields = list(source_fields)
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with tmp_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(output_path)
