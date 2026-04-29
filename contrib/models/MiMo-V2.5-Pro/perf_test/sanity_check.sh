#!/bin/bash
# Quick sanity check against an already-running vLLM server.
#
# Assumes vLLM is already listening on $PORT (default 8000) with MiMo-V2.5-Pro
# loaded. Sends a single chat completion and prints the model's reply.
#
# Usage:
#   bash sanity_check.sh                      # uses defaults
#   PORT=8001 bash sanity_check.sh            # custom port
#   PROMPT="..." bash sanity_check.sh         # custom prompt

set -e

MODEL_PATH="${MIMO_V2_FLASH_PATH:-/opt/dlami/nvme/models/MiMo-V2.5-Pro-Neuron-FP8}"
PORT="${PORT:-8000}"
# "Introduce yourself" is a high-signal self-identification prompt that the
# FP8 path answers coherently even under current MoE drift (see README
# Status). Swap PROMPT=... if you want to probe other prompts.
PROMPT="${PROMPT:-Hello! Please introduce yourself in one sentence.}"
MAX_TOKENS="${MAX_TOKENS:-64}"

echo "Sanity check: POST /v1/chat/completions on port $PORT"
echo "  Model:      $MODEL_PATH"
echo "  Prompt:     $PROMPT"
echo "  Max tokens: $MAX_TOKENS"
echo ""

# Health check first — fail fast if server isn't up.
if ! curl -sf "http://localhost:$PORT/health" > /dev/null; then
    echo "ERROR: vLLM server is not responding on http://localhost:$PORT"
    echo "Start it with 'bash bench_mimo_v2.sh' or your own launcher first."
    exit 1
fi

# NOTE: request-side `temperature` is ignored by vllm-neuron on this model:
# on-device sampling_config (set at compile time in start_vllm_server.sh as
# do_sample=true, T=0.6, top_k=20, top_p=0.95) is baked into the NEFF and
# request params don't override it. Output will be stochastic.
RESPONSE=$(curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(cat <<EOF
{
    "messages": [{"role": "user", "content": "$PROMPT"}],
    "model": "$MODEL_PATH",
    "max_tokens": $MAX_TOKENS,
    "stream": false
}
EOF
)")

echo "Response:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
echo ""

# Extract the model's reply for a human-friendly one-liner summary.
REPLY=$(echo "$RESPONSE" | python3 -c "
import json, sys
try:
    r = json.load(sys.stdin)
    print(r['choices'][0]['message']['content'].strip())
except Exception as e:
    print(f'(could not parse reply: {e})')
" 2>/dev/null)

echo "Model reply: $REPLY"
