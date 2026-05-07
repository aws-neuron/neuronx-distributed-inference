#!/bin/bash
# Quick sanity check against an already-running vLLM server.
#
# Assumes vLLM is already listening on $PORT (default 8000) with MiniMax-M2
# loaded. Sends a single chat completion and prints the model's reply.
#
# Usage:
#   bash sanity_check.sh                      # uses defaults
#   PORT=8001 bash sanity_check.sh            # custom port
#   PROMPT="..." bash sanity_check.sh         # custom prompt

set -e

MODEL_PATH="${MINIMAX_M2_PATH:-/opt/dlami/nvme/models/MiniMax-M2-BF16}"
PORT="${PORT:-8000}"
PROMPT="${PROMPT:-What is 1+1? Answer briefly.}"
MAX_TOKENS="${MAX_TOKENS:-64}"

echo "Sanity check: POST /v1/chat/completions on port $PORT"
echo "  Model:      $MODEL_PATH"
echo "  Prompt:     $PROMPT"
echo "  Max tokens: $MAX_TOKENS"
echo ""

if ! curl -sf "http://localhost:$PORT/health" > /dev/null; then
    echo "ERROR: vLLM server is not responding on http://localhost:$PORT"
    echo "Start it with 'bash bench_minimax_m2.sh' or your own launcher first."
    exit 1
fi

RESPONSE=$(curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(cat <<EOF
{
    "messages": [{"role": "user", "content": "$PROMPT"}],
    "model": "$MODEL_PATH",
    "max_tokens": $MAX_TOKENS,
    "temperature": 0.0,
    "stream": false
}
EOF
)")

echo "Response:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
echo ""

REPLY=$(echo "$RESPONSE" | python3 -c "
import json, sys
try:
    r = json.load(sys.stdin)
    print(r['choices'][0]['message']['content'].strip())
except Exception as e:
    print(f'(could not parse reply: {e})')
" 2>/dev/null)

echo "Model reply: $REPLY"
