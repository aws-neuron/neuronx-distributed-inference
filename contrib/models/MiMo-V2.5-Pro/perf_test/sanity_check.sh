#!/bin/bash
# Quick sanity check against an already-running vLLM server.
#
# Posts a minimally-templated chat request to /v1/completions and prints the
# model's reply. We go through /v1/completions (not /v1/chat/completions)
# because Pro's default chat template prepends a ~240-token system prompt
# that by itself overflows the seq_len=256 compile-time bucket; building the
# im_start/im_end/assistant frame by hand keeps the prompt under ~30 tokens
# and fits cleanly.
#
# Usage:
#   bash sanity_check.sh                      # uses defaults
#   PORT=8001 bash sanity_check.sh            # custom port
#   PROMPT="..." bash sanity_check.sh         # custom user content

set -e

MODEL_PATH="${MIMO_V2_FLASH_PATH:-/opt/dlami/nvme/models/MiMo-V2.5-Pro-Neuron-FP8}"
PORT="${PORT:-8000}"
# "Introduce yourself" is the self-identification prompt that consistently
# lands in the model's MiMo-aware region. Swap PROMPT=... to probe others.
PROMPT="${PROMPT:-Hello! Please introduce yourself in one sentence.}"
MAX_TOKENS="${MAX_TOKENS:-80}"

echo "Sanity check: POST /v1/completions on port $PORT"
echo "  Model:      $MODEL_PATH"
echo "  Prompt:     $PROMPT"
echo "  Max tokens: $MAX_TOKENS"
echo ""

# Health check first — fail fast if server isn't up.
if ! curl -sf "http://localhost:$PORT/health" > /dev/null; then
    echo "ERROR: vLLM server is not responding on http://localhost:$PORT"
    echo "Start it with 'bash start_vllm_server.sh' (or bench_mimo_v2.sh)"
    echo "first and wait for 'Application startup complete.'"
    exit 1
fi

# NOTE: request-side `temperature` / `top_k` / `top_p` are ignored by
# vllm-neuron on this model: the on_device_sampling_config baked into the
# NEFF at compile time wins. Output is always stochastic; re-run to see
# variance.
#
# Build the chat framing in python so newlines and special tokens survive
# JSON encoding without shell escape pitfalls, then POST to /v1/completions.
python3 <<PYEOF
import json
import urllib.request
import sys

model = "$MODEL_PATH"
user = """$PROMPT"""
prompt = (
    "<|im_start|>user\n"
    + user
    + "<|im_end|>\n<|im_start|>assistant\n"
)
body = json.dumps({
    "model": model,
    "prompt": prompt,
    "max_tokens": int("$MAX_TOKENS"),
    "stream": False,
}).encode()
req = urllib.request.Request(
    "http://localhost:$PORT/v1/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
except urllib.error.HTTPError as e:
    print("HTTP error:", e.code, e.read().decode(errors="replace"))
    sys.exit(1)

if "error" in resp:
    print("Error from server:", json.dumps(resp["error"], indent=2))
    sys.exit(1)

text = resp["choices"][0]["text"]
print("Response:")
print(text)
print()
print("Usage:", resp.get("usage", {}))
PYEOF
