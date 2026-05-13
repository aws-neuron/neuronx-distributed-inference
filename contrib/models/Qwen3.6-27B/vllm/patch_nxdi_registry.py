#!/usr/bin/env python3
"""Register Qwen3.6 contrib model in the installed NxDI registry.

This patches the active Python environment, not the repository checkout. The
runtime still needs PYTHONPATH to include contrib/models/Qwen3.6-27B so that
`src.modeling_qwen35` can be imported by the vLLM process.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER_BEGIN = "# QWEN36_CONTRIB_VLLM_REGISTER_BEGIN"
MARKER_END = "# QWEN36_CONTRIB_VLLM_REGISTER_END"

REGISTRATION_BLOCK = f"""

{MARKER_BEGIN}
# Registered by contrib/models/Qwen3.6-27B/vllm/install_qwen36_vllm.sh.
# Requires PYTHONPATH to include the Qwen3.6-27B contrib directory at runtime.
try:
    from src.modeling_qwen35 import (
        NeuronQwen35ForCausalLM as _Qwen36ContribForCausalLM,
    )
except Exception:
    _Qwen36ContribForCausalLM = None

if _Qwen36ContribForCausalLM is not None:
    MODEL_TYPES.setdefault("qwen3_5", {{}})["causal-lm"] = _Qwen36ContribForCausalLM
    MODEL_TYPES.setdefault("qwen3_5_text", {{}})["causal-lm"] = _Qwen36ContribForCausalLM
{MARKER_END}
"""


def _constants_path() -> Path:
    import neuronx_distributed_inference.utils.constants as constants  # noqa: WPS433

    return Path(constants.__file__).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrib-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    contrib_root = Path(args.contrib_root).expanduser().resolve()
    if not (contrib_root / "src" / "modeling_qwen35.py").exists():
        raise FileNotFoundError(f"Qwen3.6 contrib root looks invalid: {contrib_root}")

    path = _constants_path()
    text = path.read_text()
    if MARKER_BEGIN in text:
        print(f"Registry already patched: {path}")
        return 0

    patched = text.rstrip() + REGISTRATION_BLOCK + "\n"
    print(f"Patch target: {path}")
    if args.dry_run:
        print("Dry run; no files written")
        return 0

    path.write_text(patched)
    print("Patched NxDI MODEL_TYPES with qwen3_5 and qwen3_5_text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
