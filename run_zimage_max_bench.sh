#!/usr/bin/env bash
# Z-Image-on-MAX latency bench, attempt 2 (deadlock fix).
# Changes vs attempt 1:
#  - --no-enable-overlap-scheduler  (overlap scheduler suspected in the futex deadlock)
#  - curl -m timeout per request    (can't hang forever)
#  - small probe first (512^2 / 8 steps) to prove generation works before scaling
# Compile is cached (.max_cache, 8 .mef) so ready should be ~2-3 min (load only).
set -uo pipefail
cd /home/alex/mojodiffusion

SERVE_LOG=zimage_max_serve.log
MODEL=Tongyi-MAI/Z-Image
export MAX_SERVE_API_TYPES='["responses"]'
# MAX's Z-Image attention dlopens cuDNN; it's not on the pixi env path. Point at
# the existing pip cuDNN-9 (same wheels flame-core's build.rs uses). NO install.
CUDNN_DIR=/home/alex/serenity/venv/lib/python3.12/site-packages/nvidia/cudnn/lib

nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader

echo "=== serve (warm cache, no overlap scheduler) ==="
: > "$SERVE_LOG"
pixi run bash -c "export LD_LIBRARY_PATH=$CUDNN_DIR:\$LD_LIBRARY_PATH; MAX_SERVE_API_TYPES='[\"responses\"]' max serve --model $MODEL --devices gpu --no-enable-overlap-scheduler" >"$SERVE_LOG" 2>&1 &
SERVE_PID=$!
echo "serve PID $SERVE_PID"
cleanup() { echo "=== teardown ==="; kill "$SERVE_PID" 2>/dev/null; pkill -9 -f 'max serve' 2>/dev/null; pkill -9 -f spawn_main 2>/dev/null; }
trap cleanup EXIT

echo "=== wait for ready (warm; up to 8 min) ==="
for i in $(seq 1 240); do
  if grep -qiE 'Server ready' "$SERVE_LOG"; then echo "READY after ${i}*2s"; break; fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then echo "!!! serve died:"; tail -25 "$SERVE_LOG"; exit 1; fi
  sleep 2
done
grep -qiE 'Server ready' "$SERVE_LOG" || { echo "!!! never ready"; tail -20 "$SERVE_LOG"; exit 1; }

req() { # $1=w $2=h $3=steps -> echoes time_total or TIMEOUT
  local w=$1 h=$2 s=$3
  local body='{"model":"'"$MODEL"'","input":"a red fox in a snowy forest","provider_options":{"image":{"height":'"$h"',"width":'"$w"',"steps":'"$s"'}}}'
  local t
  t=$(curl -s -m 240 -o /tmp/zi_resp.json -w '%{time_total}' -X POST http://localhost:8000/v1/responses -H 'Content-Type: application/json' -d "$body" 2>/dev/null)
  local rc=$?
  if [ $rc -eq 28 ]; then echo "TIMEOUT(240s)"; else echo "${t}s rc=$rc bytes=$(wc -c </tmp/zi_resp.json 2>/dev/null)"; fi
}

echo "=== PROBE: 512x512 / 8 steps (prove it generates; 240s cap) ==="
echo "  probe -> $(req 512 512 8)"
echo "  --- resp head ---"; head -c 250 /tmp/zi_resp.json 2>/dev/null; echo
# decode probe to confirm real image
jq -r '.output[0].content[0].image_data // empty' /tmp/zi_resp.json 2>/dev/null | base64 -d > zimage_probe.png 2>/dev/null
file zimage_probe.png 2>/dev/null

if [ -s zimage_probe.png ]; then
  echo "=== PROBE OK -> timed 1024x1024 / 20 steps x3 ==="
  for n in 1 2 3; do echo "  img $n -> $(req 1024 1024 20)"; done
  jq -r '.output[0].content[0].image_data // empty' /tmp/zi_resp.json 2>/dev/null | base64 -d > zimage_max_out.png 2>/dev/null
  file zimage_max_out.png 2>/dev/null
else
  echo "!!! PROBE produced no image — still deadlocked or errored. worker thread states:"
  VP=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1)
  ps -L -o state=,wchan:18 --no-headers -p "$VP" 2>/dev/null | awk '{print $1,$2}' | sort | uniq -c | sort -rn | head
fi

nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader
