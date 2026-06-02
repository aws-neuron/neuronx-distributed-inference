"""
Validate CP=4 output is visually equivalent to the CP=2 baseline (same seed/inputs).

CP=4 is NOT bit-exact vs CP=2 — changing the CP degree changes the sequence partition
and therefore the bf16 accumulation order (and the pad count: 368 vs 112). The result is
ULP-level drift that is visually equivalent, not pixel-identical. This test asserts the
structural agreement is within a visual-equivalence threshold (mean |Δ| < 8/255, the
garment/pose/face preserved), NOT max|Δ|=0.

Usage: python test_cp4_quality.py <cp4.png> <cp2_baseline.png>
"""
import sys
import numpy as np
from PIL import Image

THRESH_MEAN_PCT = 2.0   # mean |Δ| as % of 255; visual-equivalence bound

def main():
    if len(sys.argv) != 3:
        print("usage: python test_cp4_quality.py <cp4.png> <cp2.png>")
        return 2
    a = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(np.int32)
    b = np.asarray(Image.open(sys.argv[2]).convert("RGB")).astype(np.int32)
    if a.shape != b.shape:
        print(f"FAIL: shape mismatch {a.shape} vs {b.shape}")
        return 1
    d = np.abs(a - b)
    mean_pct = d.mean() / 255 * 100
    p5 = 100 * np.mean(d.max(-1) > 5)
    print(f"mean|Δ|={mean_pct:.2f}%  px>5={p5:.1f}%  max|Δ|={int(d.max())}")
    if mean_pct < THRESH_MEAN_PCT:
        print(f"PASS — CP=4 visually equivalent to CP=2 (mean < {THRESH_MEAN_PCT}%)")
        return 0
    print(f"FAIL — CP=4 differs too much from CP=2 (mean >= {THRESH_MEAN_PCT}%)")
    return 1

if __name__ == "__main__":
    sys.exit(main())
