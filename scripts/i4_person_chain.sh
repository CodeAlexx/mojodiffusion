#!/usr/bin/env bash
# person A/B: Mojo render (injected inputs) -> decode-only -> analyzer (MJ-1047)
set -uo pipefail
AB=/home/alex/mojodiffusion/output/checks/i4_ab_person
CUDNN=$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib
LDP=/home/alex/mojodiffusion/.pixi/envs/default/lib:/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib:$CUDNN
cd /home/alex/mojodiffusion
echo "[chain] mojo render start $(date +%T)" >> "$AB/chain.log"
LD_LIBRARY_PATH=$LDP MODULAR_DEVICE_CONTEXT_SYNC_MODE=true \
  output/bin/ideogram4_ab_harness_person \
  /home/alex/mojodiffusion/output/checks/i4_noise_1024.safetensors \
  /home/alex/EriDiffusion/inference-flame/output/ideogram4_embeddings.person.safetensors \
  "$AB/mojo_sched" sched - >> "$AB/chain.log" 2>&1
echo "[chain] render exit=$? $(date +%T)" >> "$AB/chain.log"
LD_LIBRARY_PATH=$LDP MODULAR_DEVICE_CONTEXT_SYNC_MODE=true \
  output/bin/ideogram4_ab_decode "$AB/mojo_sched_final_latent.safetensors" "$AB/mojo_sched" >> "$AB/chain.log" 2>&1
echo "[chain] decode exit=$? $(date +%T)" >> "$AB/chain.log"
python3 scripts/i4_ab_analyze.py "$AB" > "$AB/verdict.txt" 2>&1
echo "PERSON_CHAIN_DONE" >> "$AB/chain.log"
