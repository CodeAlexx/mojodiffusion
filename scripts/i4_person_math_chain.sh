#!/usr/bin/env bash
# MJ-1060 discriminator: person A/B with MATH SDPA (IDEOGRAM4_SDPA_FLASH=False build).
set -uo pipefail
AB=/home/alex/mojodiffusion/output/checks/i4_ab_person
CUDNN=$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib
LDP=/home/alex/mojodiffusion/.pixi/envs/default/lib:/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib:$CUDNN
cd /home/alex/mojodiffusion
echo "[math-chain] render start $(date +%T)" >> "$AB/chain.log"
LD_LIBRARY_PATH=$LDP MODULAR_DEVICE_CONTEXT_SYNC_MODE=true \
  output/bin/ideogram4_ab_harness_person_math \
  /home/alex/mojodiffusion/output/checks/i4_noise_1024.safetensors \
  /home/alex/EriDiffusion/inference-flame/output/ideogram4_embeddings.person.safetensors \
  "$AB/mojo_math" sched - >> "$AB/chain.log" 2>&1
echo "[math-chain] render exit=$? $(date +%T)" >> "$AB/chain.log"
LD_LIBRARY_PATH=$LDP MODULAR_DEVICE_CONTEXT_SYNC_MODE=true \
  output/bin/ideogram4_ab_decode "$AB/mojo_math_final_latent.safetensors" "$AB/mojo_math" >> "$AB/chain.log" 2>&1
echo "MATH_CHAIN_DONE" >> "$AB/chain.log"
