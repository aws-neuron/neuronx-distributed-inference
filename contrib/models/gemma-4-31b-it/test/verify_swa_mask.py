# coding=utf-8
"""On-device verification of the SWA TKG mask fix (Task 025).

Traces the fixed _create_windowed_attn_mask_tkg on Neuron and compares against
CPU-computed expected masks for the same positions from OpenRelay's PR #106
analysis. Confirms the fix survives Neuron tracing/compilation.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python3 verify_swa_mask.py
"""

import sys

import torch
import torch.nn as nn

try:
    import torch_neuronx
    _HAS_NEURON = True
except ImportError:
    print("WARNING: torch_neuronx not available -- running CPU-only test")
    _HAS_NEURON = False


class FixedMaskModule(nn.Module):
    """Isolated copy of the Task 025 fix so we can trace it in isolation."""

    def __init__(self, uniform_cache_len: int, window_size: int):
        super().__init__()
        self.uniform_cache_len = uniform_cache_len
        self.window_size = window_size

    def forward(self, attention_mask, position_ids):
        cache_len = self.uniform_cache_len
        pos = position_ids[:, 0].unsqueeze(1)
        j = torch.arange(cache_len, device=attention_mask.device).unsqueeze(0)
        window_mask = (j < pos) & (j >= (pos - self.window_size + 1))
        return window_mask[:, None, None, :]


def expected_mask(pos_value: int, cache_len: int, window_size: int) -> torch.Tensor:
    """The mask we expect: True at [max(0, pos-window+1), pos-1]."""
    low = max(0, pos_value - window_size + 1)
    high = pos_value  # exclusive
    mask = torch.zeros(cache_len, dtype=torch.bool)
    if high > 0 and low < cache_len:
        mask[low:min(high, cache_len)] = True
    return mask


def main():
    window_size = 1024
    cache_len = 8192  # simulating max_length=8192 with sliding_window=1024

    print("=" * 80)
    print("Task 025 on-device verification: SWA TKG mask")
    print(f"window_size={window_size}, uniform_cache_len={cache_len}")
    print(f"Backend: {'Neuron' if _HAS_NEURON else 'CPU-only'}")
    print("=" * 80)

    module = FixedMaskModule(cache_len, window_size).eval()

    if _HAS_NEURON:
        # Trace once with dummy inputs (matching real decode-time shapes)
        example_am = torch.zeros(1, 128, dtype=torch.bool)
        example_pi = torch.tensor([[1024]], dtype=torch.int64)
        print("\nTracing on Neuron...")
        neuron_module = torch_neuronx.trace(module, (example_am, example_pi))
        print("Trace complete.\n")
    else:
        neuron_module = module

    positions = [0, 1, 500, 1023, 1024, 1448, 2045, 2600, 4000, 7999]
    all_ok = True
    for pos in positions:
        attention_mask = torch.zeros(1, 128, dtype=torch.bool)
        position_ids = torch.tensor([[pos]], dtype=torch.int64)

        mask_neuron = neuron_module(attention_mask, position_ids)
        mask_cpu = module(attention_mask, position_ids)
        mask_expected_ = expected_mask(pos, cache_len, window_size)

        # Compare Neuron vs CPU (bit-identical for boolean masks)
        neuron_v = mask_neuron[0, 0, 0].cpu()
        cpu_v = mask_cpu[0, 0, 0]

        neuron_matches_cpu = torch.equal(neuron_v, cpu_v)
        neuron_matches_expected = torch.equal(neuron_v, mask_expected_)
        n_valid = neuron_v.sum().item()
        n_exp = mask_expected_.sum().item()

        low = max(0, pos - window_size + 1)
        high = pos
        status = "OK" if neuron_matches_cpu and neuron_matches_expected else "FAIL"
        print(
            f"pos={pos:5d} | expected [{low}, {high-1}] ({n_exp} slots) | "
            f"neuron valid={n_valid} | neuron==cpu: {neuron_matches_cpu} | "
            f"neuron==expected: {neuron_matches_expected} | {status}"
        )
        if not (neuron_matches_cpu and neuron_matches_expected):
            all_ok = False

    print()
    print("=" * 80)
    if all_ok:
        print("PASS: All position cases match expected on-device")
    else:
        print("FAIL: some positions did not match")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    main()
