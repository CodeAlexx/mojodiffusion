#!/usr/bin/env bash
# magic_prompt.sh — Ideogram-4 prompt generator.
# Expands a short natural-language idea into the structured Ideogram-4 caption JSON
# using a local GGUF LLM via an EPHEMERAL llama-server (spawn -> generate -> kill),
# so it never holds GPU VRAM while the image worker needs it.
#
# Usage: magic_prompt.sh <model_gguf> <aspect_ratio> <idea>
#   prints the model's caption (single-line JSON) on stdout; logs to stderr.
set -uo pipefail

MODEL="${1:?model gguf path required}"
ASPECT="${2:-1:1}"
IDEA="${3:?idea required}"

LLAMA="/home/alex/llama.cpp/build/bin/llama-server"
SYSPROMPT="/home/alex/ideogram4-ref/src/ideogram4/magic_prompt_system_prompts/v1.txt"
PORT="${MAGIC_PORT:-8090}"
NGL="${MAGIC_NGL:-99}"        # GPU layers; set 0 to force CPU
CTX="${MAGIC_CTX:-16384}"

[ -f "$MODEL" ] || { echo "magic_prompt: model not found: $MODEL" >&2; exit 2; }
[ -x "$LLAMA" ] || { echo "magic_prompt: llama-server not found: $LLAMA" >&2; exit 2; }
[ -f "$SYSPROMPT" ] || { echo "magic_prompt: system prompt not found: $SYSPROMPT" >&2; exit 2; }

# 1+2. spawn ephemeral llama-server and wait for readiness. The image worker can
#       transiently hold the GPU pool during init, so a GPU cudaMalloc may fail;
#       we retry once on GPU, then fall back to CPU (-ngl 0) so this never hard-fails.
SRV=""
cleanup() { [ -n "$SRV" ] && { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }; }
trap cleanup EXIT

try_serve() {  # $1 = ngl
  [ -n "$SRV" ] && { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
  "$LLAMA" -m "$MODEL" --host 127.0.0.1 --port "$PORT" -ngl "$1" -c "$CTX" \
    --jinja --no-webui >/tmp/magic_llama_$PORT.log 2>&1 &
  SRV=$!
  for _ in $(seq 1 120); do
    kill -0 "$SRV" 2>/dev/null || return 1          # process died (e.g. OOM)
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

if ! try_serve "$NGL"; then
  echo "magic_prompt: GPU load failed (retry on GPU)..." >&2
  sleep 3
  if ! try_serve "$NGL"; then
    echo "magic_prompt: GPU still failing, falling back to CPU..." >&2
    if ! try_serve 0; then
      echo "magic_prompt: llama-server not ready (GPU+CPU)" >&2
      tail -6 /tmp/magic_llama_$PORT.log >&2; exit 3
    fi
  fi
fi

# 3. build the chat request (jq encodes system prompt + idea safely). /no_think
#    disables Qwen3 reasoning per the v1.txt META (thinking_mode: disabled).
USER_MSG="Aspect ratio: ${ASPECT}. User idea: ${IDEA}. Use a normal opaque scene background that fills the frame; do NOT use a transparent background unless the idea explicitly asks for transparency. /no_think"
REQ=$(jq -n --rawfile sys "$SYSPROMPT" --arg user "$USER_MSG" '{
  messages: [ {role:"system", content:$sys}, {role:"user", content:$user} ],
  temperature: 0.7, top_p: 0.9, max_tokens: 1200, stream: false
}')

# 4. generate, extract the assistant content
RESP=$(curl -sf -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$REQ")
rc=$?
[ $rc -eq 0 ] || { echo "magic_prompt: chat request failed rc=$rc" >&2; exit 4; }

CONTENT=$(printf '%s' "$RESP" | jq -r '.choices[0].message.content // empty')
# strip any <think>...</think> the model may still emit, and code fences
CONTENT=$(printf '%s' "$CONTENT" | sed -E 's/<think>.*<\/think>//g; s/```json//g; s/```//g')
[ -n "$CONTENT" ] || { echo "magic_prompt: empty completion" >&2; printf '%s' "$RESP" | head -c 400 >&2; exit 5; }

printf '%s\n' "$CONTENT"
