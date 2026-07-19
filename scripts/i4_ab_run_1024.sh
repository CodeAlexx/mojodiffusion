#!/usr/bin/env bash
set -euo pipefail
AB=/home/alex/mojodiffusion/output/checks/i4_ab_1024
IF=/home/alex/EriDiffusion/inference-flame
NOISE=/home/alex/mojodiffusion/output/checks/i4_noise_1024.safetensors
EMB=$IF/output/ideogram4_embeddings.safetensors
CUDNN=$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib
echo "=== [1/3] Rust 1024 ==="
(cd "$IF" && LD_LIBRARY_PATH=$CUDNN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  target/release/ideogram4_infer --size 1024 --steps 20 \
  --noise-file "$NOISE" --latent-out "$AB/rust_final_latent.safetensors")
cp "$IF/output/ideogram4_infer.png" "$AB/rust.png"
cd /home/alex/mojodiffusion
echo "=== [2/3] Mojo sched 1024 ==="
LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib:/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib:$CUDNN \
MODULAR_DEVICE_CONTEXT_SYNC_MODE=true \
  output/bin/ideogram4_ab_harness_1024 "$NOISE" "$EMB" "$AB/mojo_sched" sched -
echo "=== [3/3] Mojo const7 1024 ==="
LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib:/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib:$CUDNN \
MODULAR_DEVICE_CONTEXT_SYNC_MODE=true \
  output/bin/ideogram4_ab_harness_1024 "$NOISE" "$EMB" "$AB/mojo_const7" const7 -
echo "=== analyze ==="
python3 /home/alex/mojodiffusion/scripts/i4_ab_analyze.py "$AB"
echo I4_AB_1024_DONE
