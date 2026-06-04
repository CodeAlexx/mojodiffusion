# models/qwenimage/config.mojo — Qwen-Image MMDiT per-variant config accessor.
#
# Dims CONFIRMED from the real diffusers transformer/config.json
# (/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image-2512/.../transformer/
#  config.json): num_layers 60, attention_head_dim 128, num_attention_heads 24
#  => inner_dim 3072; in_channels 64, out_channels 16, joint_attention_dim 3584,
#  axes_dims_rope [16,56,56], patch_size 2. The full 12G transformer shards were
#  not present locally (only the index.json + text_encoder), so block tensor
#  shapes are taken from config.json + the inference port qwenimage_dit.mojo
#  (which read inference-flame/src/models/qwenimage_dit.rs line-by-line).
#
# Recipe cited from EriDiffusion-v2 crates/eridiffusion-core/src/models/qwenimage.rs:
#   - consts (qwenimage.rs:40-50): NUM_LAYERS=60, DIM=3072, NUM_HEADS=24,
#     HEAD_DIM=128, IN_CHANNELS=64, OUT_CHANNELS=16, JOINT_DIM=3584,
#     MLP_HIDDEN=12288, NORM_EPS=1e-6, ROPE_THETA=10000.0,
#     ROPE_AXES_DIMS=[16,56,56].
#   - LoRA targets (qwenimage.rs:70-82 QWENIMAGE_TARGETS): 12 targets per block
#     (img/txt × q/k/v/out + img/txt × ffn_up/ffn_down).
#   - rank/alpha default (flame-diffusion qwenimage-trainer config.rs:101-102):
#     default_lora_rank 16, alpha=rank (scale 1.0). NUM_LAYERS=60 => all-double.
#
# Qwen-Image is ALL double-stream (num_double=60, num_single=0): no single blocks.
# These map onto the shared TrainConfig (which has num_double/num_single slots).

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config


comptime QWENIMAGE_CONFIG = (
    "/home/alex/mojodiffusion/serenitymojo/configs/qwenimage.json"
)


def qwen_image() raises -> TrainConfig:
    return read_model_config(String(QWENIMAGE_CONFIG))
