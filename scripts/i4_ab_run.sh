#!/usr/bin/env bash
# ideogram4 Rust-vs-Mojo A/B (MJ-1047/MJ-1051): byte-identical init noise +
# byte-identical llm_features conditioning through both stacks, then analyze.
# Run ONLY when the GPU is free (each render wants the whole 24GB card).
set -euo pipefail

AB=/home/alex/mojodiffusion/output/checks/i4_ab
IF=/home/alex/EriDiffusion/inference-flame
NOISE=$IF/output/ideogram4_latent.safetensors
EMB=$IF/output/ideogram4_embeddings.safetensors
CUDNN=$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib
mkdir -p "$AB"

echo "=== [1/3] Rust render (injected noise, V4_DEFAULT_20, 512px) ==="
(cd "$IF" && LD_LIBRARY_PATH=$CUDNN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  target/release/ideogram4_infer --size 512 --steps 20 \
  --noise-file "$NOISE" --latent-out "$AB/rust_final_latent.safetensors" \
  2>&1 | tee "$AB/rust_render.log")
cp "$IF/output/ideogram4_infer.png" "$AB/rust.png"

echo "=== [2/3] Mojo render, faithful 3/7 schedule (+ per-step dumps) ==="
cd /home/alex/mojodiffusion
LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib:/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib:$CUDNN \
MODULAR_DEVICE_CONTEXT_SYNC_MODE=true \
  output/bin/ideogram4_ab_harness "$NOISE" "$EMB" "$AB/mojo_sched" sched dump \
  2>&1 | tee "$AB/mojo_sched.log"

echo "=== [3/3] Mojo render, worker-style constant cfg 7.0 ==="
LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib:/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib:$CUDNN \
MODULAR_DEVICE_CONTEXT_SYNC_MODE=true \
  output/bin/ideogram4_ab_harness "$NOISE" "$EMB" "$AB/mojo_const7" const7 - \
  2>&1 | tee "$AB/mojo_const7.log"

echo "=== analyze ==="
python3 /home/alex/mojodiffusion/scripts/i4_ab_analyze.py "$AB"
