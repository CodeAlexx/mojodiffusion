#!/usr/bin/env bash
set -uo pipefail
AB=/home/alex/mojodiffusion/output/checks/i4_eri2lora
CUDNN=$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib
LDP=/home/alex/mojodiffusion/.pixi/envs/default/lib:/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib:$CUDNN
NOISE=/home/alex/mojodiffusion/output/checks/i4_noise_1024.safetensors
EMB=/home/alex/EriDiffusion/inference-flame/output/ideogram4_embeddings.eri2.safetensors
cd /home/alex/mojodiffusion
for CK in 1500 2000; do
  echo "[eri2lora] render ckpt $CK $(date +%T)" >> "$AB/chain.log"
  LD_LIBRARY_PATH=$LDP MODULAR_DEVICE_CONTEXT_SYNC_MODE=true \
    output/bin/ideogram4_ab_harness_eri2lora "$NOISE" "$EMB" "$AB/lora${CK}" sched - \
    "/home/alex/mojodiffusion/output/ideogram4_eri2_lora/lora_step_${CK}.safetensors" >> "$AB/chain.log" 2>&1
  LD_LIBRARY_PATH=$LDP MODULAR_DEVICE_CONTEXT_SYNC_MODE=true \
    output/bin/ideogram4_ab_decode "$AB/lora${CK}_final_latent.safetensors" "$AB/lora${CK}" >> "$AB/chain.log" 2>&1
done
echo "[eri2lora] base (no lora) reference $(date +%T)" >> "$AB/chain.log"
LD_LIBRARY_PATH=$LDP MODULAR_DEVICE_CONTEXT_SYNC_MODE=true \
  output/bin/ideogram4_ab_harness_eri2lora "$NOISE" "$EMB" "$AB/base" sched - - >> "$AB/chain.log" 2>&1
LD_LIBRARY_PATH=$LDP MODULAR_DEVICE_CONTEXT_SYNC_MODE=true \
  output/bin/ideogram4_ab_decode "$AB/base_final_latent.safetensors" "$AB/base" >> "$AB/chain.log" 2>&1
echo "ERI2LORA_CHAIN_DONE" >> "$AB/chain.log"
