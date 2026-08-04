#!/usr/bin/env python3
"""Diff two codec dumps from generate_qwen25_omni_speech.py / test_gpu_baseline_bench.py.

Reports:
  - whether thinker text + thinker token ids are identical
  - first divergent codec step
  - common prefix length / total lengths

Usage::

    python examples/diff_codec.py trn2_codec.json gpu_codec.json
"""

import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    args = p.parse_args()

    with open(args.a) as f:
        a = json.load(f)
    with open(args.b) as f:
        b = json.load(f)

    print(f"  A ({a.get('platform','?')}): "
          f"greedy={a.get('greedy')} seed={a.get('seed')}")
    print(f"  B ({b.get('platform','?')}): "
          f"greedy={b.get('greedy')} seed={b.get('seed')}")

    # Thinker text
    same_text = a.get("thinker_text", "") == b.get("thinker_text", "")
    print(f"\n[thinker text] equal: {same_text}")
    if not same_text:
        print(f"  A: {a.get('thinker_text','')[:120]}")
        print(f"  B: {b.get('thinker_text','')[:120]}")

    # Thinker token ids
    ta = a.get("thinker_token_ids", [])
    tb = b.get("thinker_token_ids", [])
    common_t = 0
    for x, y in zip(ta, tb):
        if x != y:
            break
        common_t += 1
    print(f"[thinker tokens] len(A)={len(ta)} len(B)={len(tb)} "
          f"common_prefix={common_t}")
    if common_t < min(len(ta), len(tb)):
        i = common_t
        print(f"  first diff @ step {i}: A={ta[i]} B={tb[i]}")

    # Codec token ids
    ca = a.get("codec_token_ids", [])
    cb = b.get("codec_token_ids", [])
    common_c = 0
    for x, y in zip(ca, cb):
        if x != y:
            break
        common_c += 1
    print(f"\n[codec tokens] len(A)={len(ca)} len(B)={len(cb)} "
          f"common_prefix={common_c}")
    if common_c < min(len(ca), len(cb)):
        i = common_c
        a_window = ca[max(0, i - 3): i + 5]
        b_window = cb[max(0, i - 3): i + 5]
        print(f"  first diff @ step {i}: A={ca[i]} B={cb[i]}")
        print(f"  A window [{max(0, i - 3)}..{i + 5}]: {a_window}")
        print(f"  B window [{max(0, i - 3)}..{i + 5}]: {b_window}")
    elif len(ca) != len(cb):
        print(f"  one side ran longer: A ran {len(ca) - common_c} more, "
              f"B ran {len(cb) - common_c} more")
    else:
        print("  codec sequences are identical")

    return 0


if __name__ == "__main__":
    sys.exit(main())
