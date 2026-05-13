#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTRIB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -gt 0 ]]; then
  VENV="$1"
else
  VENV=""
  for candidate in \
    /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference \
    /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16 \
    /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13 \
    /opt/aws_neuronx_venv_pytorch_inference_vllm_0_12 \
    /opt/aws_neuronx_venv_pytorch_inference_vllm_0_11
  do
    if [[ -x "${candidate}/bin/python" ]]; then
      VENV="${candidate}"
      break
    fi
  done
fi

if [[ -z "${VENV}" || ! -x "${VENV}/bin/python" ]]; then
  echo "ERROR: Could not find a vLLM/Neuron Python environment." >&2
  echo "Usage: $0 /path/to/venv" >&2
  exit 1
fi

PYTHON="${VENV}/bin/python"
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${CONTRIB_ROOT}:${PYTHONPATH:-}"

echo "vLLM/Neuron env : ${VENV}"
echo "Contrib root   : ${CONTRIB_ROOT}"

"${PYTHON}" "${SCRIPT_DIR}/patch_nxdi_registry.py" --contrib-root "${CONTRIB_ROOT}"

"${PYTHON}" - <<'PY'
import importlib.util
from neuronx_distributed_inference.utils.constants import MODEL_TYPES

if importlib.util.find_spec("vllm") is None:
    raise RuntimeError("vLLM is not installed in this environment")

if importlib.util.find_spec("vllm_neuron") is None:
    print(
        "WARNING: vllm_neuron package was not found. If this environment uses "
        "an AWS vLLM fork with built-in Neuron support this may be fine; "
        "otherwise install the Neuron vLLM plugin that matches this SDK.",
    )

for key in ("qwen3_5", "qwen3_5_text"):
    assert key in MODEL_TYPES, f"{key} missing from MODEL_TYPES"
    assert "causal-lm" in MODEL_TYPES[key], f"{key}/causal-lm missing"
print("Qwen3.6 vLLM registry verification OK")
PY

echo "Installation complete."
echo "Remember to set PYTHONPATH=${CONTRIB_ROOT} when starting vLLM."
