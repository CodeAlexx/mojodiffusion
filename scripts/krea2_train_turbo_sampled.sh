#!/usr/bin/env bash
# krea2: train on RAW, sample on TURBO (the Krea-official pattern, 2026-07-02).
#
# Trains in segments with the normal raw-base trainer (inline sampler OFF),
# and between segments renders validation samples on the Krea-2-Turbo base
# (official recipe: 8 steps, cfg 1.0 single-forward) from the segment's
# checkpoint with creator guidance disabled (cfg 0.0) — using the trainer's
# proven full .state resume, so training
# numerics are IDENTICAL to an uninterrupted run (same moments, same steps).
# No Mojo changes; raw-resident training and turbo-resident sampling never
# share the 24GB card.
#
# Usage: krea2_train_turbo_sampled.sh <cache> <train_config> <total_steps> <segment> [<resume_ckpt> <start_step>]
#   train_config : normal krea2 train config (raw checkpoint, sample_every MUST be 0)
#   segment      : sample cadence, e.g. 500 (trains 500, turbo-renders, resumes)
set -euo pipefail

CACHE=$1; CONFIG=$2; TOTAL=$3; SEG=$4
RESUME=${5:-}; START=${6:-0}
ENV_LD=/home/alex/mojodiffusion/.pixi/envs/default/lib:/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib
BIN=/home/alex/mojodiffusion/output/bin/krea2_train
TURBO_CONFIG=/home/alex/mojodiffusion/serenitymojo/configs/krea2_turbo_eri2.json
WORKDIR=$(python3 -c "import json,sys; print(json.load(open('$CONFIG'))['workspace_dir'])")
PREFIX=$(python3 -c "import json,sys; print(json.load(open('$CONFIG'))['save_filename_prefix'])")
mkdir -p "$WORKDIR"

step=$START
# `resume` is what the NEXT segment resumes FROM. On a fresh start it is the
# caller's RESUME arg (may be empty); after each segment it becomes the FULL
# `.safetensors.state` sidecar (A/B + AdamW moments) so training numerics are
# IDENTICAL to an uninterrupted run — passing the weights-only `.safetensors`
# would WARM-resume (moments zeroed) and diverge.
resume=$RESUME
while [ "$step" -lt "$TOTAL" ]; do
  next=$(( step + SEG )); [ "$next" -gt "$TOTAL" ] && next=$TOTAL
  echo "=== TRAIN segment: steps $step -> $next (raw base) ==="
  MODULAR_DEVICE_CONTEXT_SYNC_MODE=true LD_LIBRARY_PATH=$ENV_LD \
    "$BIN" "$CACHE" "$next" "$CONFIG" ${resume:+"$resume"} ${resume:+"$step"}
  ckpt="$WORKDIR/${PREFIX}_${next}.safetensors"
  state="${ckpt}.state"
  if [ ! -f "$ckpt" ]; then
    echo "FATAL: expected checkpoint $ckpt not written (check save_every vs segment)"; exit 1
  fi
  if [ -f "$state" ]; then
    resume="$state"                       # FULL resume for the next segment
  else
    echo "WARN: $state missing — next segment WARM-resumes (AdamW moments zeroed)"
    resume="$ckpt"
  fi
  echo "=== SAMPLE on TURBO from ${PREFIX}_${next} (8 steps, cfg 0.0) ==="
  MODULAR_DEVICE_CONTEXT_SYNC_MODE=true LD_LIBRARY_PATH=$ENV_LD \
    "$BIN" "$CACHE" 1 "$TURBO_CONFIG" "$ckpt" 0 || echo "WARN: turbo sample render failed (training continues)"
  TS_DIR=$(python3 -c "import json; print(json.load(open('$TURBO_CONFIG'))['workspace_dir'])")
  if [ -d "$TS_DIR/samples" ]; then
    mkdir -p "$WORKDIR/turbo_samples/step_${next}"
    cp "$TS_DIR"/samples/step_0_*.png "$WORKDIR/turbo_samples/step_${next}/" 2>/dev/null || true
  fi
  step=$next
done
echo "=== DONE: $TOTAL steps trained on raw, turbo samples in $WORKDIR/turbo_samples/ ==="
