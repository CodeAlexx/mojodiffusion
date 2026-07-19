#!/usr/bin/env bash
# LTX-2.3 musubi ORACLE reference run — IMAGE arm (eri2 identity, 512x512 F=1).
# Derived from ltx2_musubi_band_ref.sh (2026-07-16 musubi-parity audit).
# Community IMAGE recipe: lr 6e-5, rank/alpha 64, uniform_prob 0.30
# (flags verified against musubi argparse: --shifted_logit_uniform_prob
#  hv_train_network.py; default mode=stretched for --ltx_version 2.3).
# Cache: /home/alex/datasets/ltx2_eri2_512 (118 imgs, latents [128,1,16,16]
# = 256 tokens, TE mask all-ones — measured 2026-07-16).
# GPU-exclusive: do NOT run while a Mojo gate/smoke or the video ref holds the GPU.
set -euo pipefail
OUT=/home/alex/mojodiffusion/output/ltx2_musubi_image_ref
mkdir -p "$OUT"
cd /home/alex/musubi-tuner
exec .venv/bin/accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/ltx2_train_network.py \
  --mixed_precision bf16 \
  --dataset_config /home/alex/datasets/ltx2_eri2_512/dataset.toml \
  --ltx2_checkpoint /home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8-dequant-bf16.safetensors \
  --ltx_version 2.3 \
  --ltx2_mode video \
  --blocks_to_swap 36 \
  --sdpa \
  --gradient_checkpointing \
  --learning_rate 6e-5 \
  --optimizer_type adamw \
  --network_module networks.lora_ltx2 \
  --network_dim 64 \
  --network_alpha 64 \
  --timestep_sampling shifted_logit_normal \
  --shifted_logit_uniform_prob 0.3 \
  --max_train_steps 100 \
  --seed 42 \
  --output_dir "$OUT" \
  --output_name ltx2_musubi_image_ref \
  --log_with tensorboard --logging_dir "$OUT/logs" \
  2>&1 | tee "$OUT/train.log"
