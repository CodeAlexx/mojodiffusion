# models/anima/config.mojo — Anima (Cosmos-Predict2 MiniTrainDIT) per-variant config.
#
# Binding user rule (2026-05-31): NO hardcoded arch/recipe in code. These helpers
# READ the variant config file (serenitymojo/configs/anima.json), the single
# source of truth for arch + recipe + paths, verified against the checkpoint
# header (block-0 tensor shapes confirmed 2026-06-01):
#   net.x_embedder.proj.1.weight        [2048, 68]    -> in_channels 68, d_model 2048
#   net.blocks.0.self_attn.q_proj.weight[2048, 2048]  -> H*Dh = 2048 (16 heads * 128)
#   net.blocks.0.mlp.layer1.weight      [8192, 2048]  -> mlp_hidden 8192 (GELU, NOT SwiGLU)
#   net.blocks.0.mlp.layer2.weight      [2048, 8192]
#   net.final_layer.linear.weight       [64, 2048]    -> out_channels 64
#   net.blocks.0.cross_attn.k_proj.weight[2048, 1024] -> joint_attention_dim 1024
#
# NOTE on TrainConfig reuse: TrainConfig is FLUX/Klein-shaped (num_double /
# num_single / SwiGLU mlp_hidden). Anima maps onto it as:
#   num_double = 0, num_single = 28  (28 uniform MiniTrainDIT blocks)
#   mlp_hidden = 8192                 (PLAIN GELU hidden — TrainConfig's "fc1
#                                      stores 2*this" comment does NOT apply to
#                                      Anima; the Anima MLP is layer1/layer2, no
#                                      gated fusion)
#   timestep_dim = 2048               (sinusoidal embed width = model_channels)
# Anima-ONLY dims that TrainConfig has no field for (adaln_lora_dim=256, the
# 6-block LLM adapter dims, patch sizes) live as comptime constants in
# serenitymojo/models/dit/anima_contract.mojo (ANIMA_ADALN_LORA_DIM, etc.) and
# are asserted there — same pattern as Klein keeping H/Dh/N comptime.

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config


comptime ANIMA_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/anima.json"


def anima() raises -> TrainConfig:
    return read_model_config(String(ANIMA_CONFIG))
