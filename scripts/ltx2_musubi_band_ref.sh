#!/usr/bin/env bash
# LTX-2.3 musubi ORACLE reference run (band gate, MJ-1041 discipline).
# Trains musubi-tuner on the SAME musubi-native cache the Mojo trainer reads
# (/home/alex/datasets/ltx2_musubi_v3), 100 steps, seed 42, video mode.
# Produces: per-step loss band (median / frac>0.30 / max) + a reference LoRA.
# Cache exists -> no gemma needed (no sampling args).
# GPU-exclusive: do NOT run while a Mojo gate/smoke holds the GPU.
set -euo pipefail
OUT=/home/alex/mojodiffusion/output/ltx2_musubi_ref
mkdir -p "$OUT"
cd /home/alex/musubi-tuner
exec .venv/bin/accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/ltx2_train_network.py \
  --mixed_precision bf16 \
  --dataset_config /home/alex/datasets/ltx2_musubi_v3/dataset.toml \
  --ltx2_checkpoint /home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8-dequant-bf16.safetensors \
  --ltx_version 2.3 \
  --ltx2_mode video \
  --blocks_to_swap 36 \
  --sdpa \
  --gradient_checkpointing \
  --learning_rate 1e-4 \
  --optimizer_type adamw \
  --network_module networks.lora_ltx2 \
  --network_dim 32 \
  --network_alpha 32 \
  --timestep_sampling shifted_logit_normal \
  --max_train_steps 100 \
  --seed 42 \
  --output_dir "$OUT" \
  --output_name ltx2_musubi_ref \
  --log_with tensorboard --logging_dir "$OUT/logs" \
  2>&1 | tee "$OUT/train.log"
