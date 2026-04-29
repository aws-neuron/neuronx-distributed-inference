#!/bin/bash
# Quick sanity check against an already-running vLLM server.
#
# Posts a chat request to /v1/completions and prints the reply.
#
# Pro's default chat template prepends a ~240-token system prompt that by
# itself overflows the seq_len=256 compile-time bucket, so we send an
# explicit short system message — apply_chat_template then uses ours
# instead of the default and the whole prompt fits in ~25 tokens.
#
# Usage:
#   bash sanity_check.sh                      # uses defaults
#   PORT=8001 bash sanity_check.sh            # custom port
#   PROMPT="..." bash sanity_check.sh         # custom user content
#   SYSTEM="..." bash sanity_check.sh         # custom system message

set -e

MODEL_PATH="${MIMO_V2_FLASH_PATH:-/opt/dlami/nvme/models/MiMo-V2.5-Pro-Neuron-FP8}"
PORT="${PORT:-8000}"
# Short system message (keeps total prompt ~25 tokens) — the checkpoint's
# default system prompt is ~240 tokens and would overflow seq_len=256.
SYSTEM="${SYSTEM:-You are MiMo, a helpful assistant developed by Xiaomi.}"
# "Introduce yourself" is the self-identification prompt that consistently
# lands in the model's MiMo-aware region. Swap PROMPT=... to probe others.
PROMPT="${PROMPT:-Hello! Please introduce yourself in one sentence.}"
MAX_TOKENS="${MAX_TOKENS:-80}"

echo "Sanity check: POST /v1/chat/completions on port $PORT"
echo "  Model:      $MODEL_PATH"
echo "  System:     $SYSTEM"
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
# variance, or restart the server with `do_sample=false` in
# start_vllm_server.sh to force deterministic greedy decoding.
python3 <<PYEOF
import json
import sys
import urllib.error
import urllib.request

model = "$MODEL_PATH"
system = """$SYSTEM"""
user = """$PROMPT"""
body = json.dumps({
    "model": model,
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ],
    "max_tokens": int("$MAX_TOKENS"),
    "stream": False,
}).encode()
req = urllib.request.Request(
    "http://localhost:$PORT/v1/chat/completions",
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

text = resp["choices"][0]["message"]["content"]
print("Response:")
print(text)
print()
print("Usage:", resp.get("usage", {}))
PYEOF
