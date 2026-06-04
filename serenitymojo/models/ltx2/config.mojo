# models/ltx2/config.mojo — LTX-2 (22B video DiT) per-variant config accessor.
#
# Binding user rule (2026-05-31): NO hardcoded arch/recipe. This helper READS
# serenitymojo/configs/ltx2.json, the single source of truth, verified against
# the real ComfyUI checkpoint header
#   /home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors
# block-0 transformer_blocks.0.* tensors (confirmed via safetensors header read):
#   attn1.to_q/k/v.weight  [4096,4096]     attn1.to_q.bias [4096]
#   attn1.q_norm.weight    [4096]          attn1.to_out.0.weight [4096,4096]
#   attn1.to_gate_logits.weight [32,4096]  (per-head gate, 32 heads)
#   ff.net.0.proj.weight   [16384,4096]    ff.net.2.weight [4096,16384]
#   scale_shift_table      [9,4096] F32    => inner_dim 4096, heads 32, Dh 128,
#                                             ff_hidden 16384, 48 layers.
#
# Recipe cited from EriDiffusion-v2 ltx2 trainer:
#   crates/eridiffusion-cli/src/bin/train_ltx2.rs:406-409
#     rank default 16, lora_alpha default 1.0, learning_rate default 3e-4.
#   crates/eridiffusion-core/src/models/ltx2.rs:74-89
#     NUM_LAYERS=48, HEADS=32, HEAD_DIM=128, INNER_DIM=4096, FFN_DIM=16384,
#     NORM_EPS=1e-6, rope_theta=10000.
#   crates/eridiffusion-core/src/models/ltx2.rs:357-358 reads
#     config.lora_rank / config.lora_alpha for the per-block LoRA slots.
#
# LTX-2 has ONE block kind (not double/single), so num_double=0, num_single=48.

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config


comptime LTX2_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/ltx2.json"


def ltx2() raises -> TrainConfig:
    return read_model_config(String(LTX2_CONFIG))
