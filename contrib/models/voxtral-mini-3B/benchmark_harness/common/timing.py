"""Neuron device-sync helper.

Bracket every measured `transcribe()` call with `neuron_sync()` so
`perf_counter` reflects wall-clock latency (not queued-but-not-run work).
For NxDI, the returned Python object (transcript text) forces a
device -> host transfer and thus an implicit sync; the explicit sync here
is defense-in-depth.
"""

from __future__ import annotations


def neuron_sync() -> None:
    """Block the host until pending Neuron work completes.

    Best-effort: calls `torch.neuron.synchronize()` if available and
    silently no-ops otherwise.  For NxDI applications, generation returns
    Python objects which act as an implicit sync anyway.
    """
    try:
        import torch

        sync = getattr(getattr(torch, "neuron", None), "synchronize", None)
        if sync is not None:
            sync()
    except Exception:
        pass
