# models/l2p/config.mojo — Z-Image L2P (pixel-space) per-model config accessor.
#
# Binding user rule (2026-05-31): NO hardcoded arch/recipe in a model file. This
# helper READS serenitymojo/configs/l2p.json (single source of truth). Mirrors
# models/zimage/config.mojo EXACTLY.
#
# Z-Image L2P REUSES the Z-Image-Turbo DiT body VERBATIM, so the per-block training
# backward surface == the Z-Image main block (models/zimage/block.mojo +
# models/zimage/lora_block.mojo). The L2P deltas (pixel-space patchify16 input proj
# + MicroDiffusionModel local-decoder head replacing FinalLayer/unpatchify) live
# OUTSIDE the transformer block and are NOT part of the per-block backward gate.
#
# Body dims (CONFIRMED from the real Z-Image base transformer safetensors header,
# .../Z-Image/.../transformer/diffusion_pytorch_model-00001-of-00002.safetensors):
#   inner_dim   = 3840 (D = 30*128)         attention.to_{q,k,v,out.0} = [3840,3840]
#   num_heads   = 30                        attention.norm_{q,k} = [128]
#   head_dim    = 128                       feed_forward.w1/w3 = [10240,3840]
#   mlp_hidden  = 10240 (SwiGLU per-gate)   feed_forward.w2 = [3840,10240]
#   num_single  = 30 main layers            adaLN_modulation.0 = [15360,256] (4 chunks)
#   timestep_dim= 1024                       norm_eps = 1e-5  final_norm_eps = 1e-6
#   rope_theta  = 256  rope_axes = [32,48,48] (sum 64 = head_dim/2), INTERLEAVED
# L2P-specific (vs base Z-Image): in/out_channels = 3 (pixel-space), patch_size = 16
#   (patchify16: 16*16*3=768 -> 3840), NO VAE, MicroDiffusionModel head, shift = 3.0.
#   Source: ZIMAGE_L2P_STARTING_PASS_2026-05-28.md lines 29-50;
#   EriDiffusion/inference-flame/src/models/l2p/{dit.rs,local_decoder.rs}.
#
# LoRA recipe (CONFIRMED from the real L2P LoRA safetensors header,
# /home/alex/samples/l2p_lora_box1jana_1000steps_bf16.safetensors): rank=16, on the
# FUSED qkv [16,11520] + out [16,3840] + feed_forward.w1 [16,10240], across
# diffusion_model.{layers, noise_refiner, context_refiner}. The per-block backward
# MATH (LoRA d_A/d_B via linear_backward composition) is identical whether QKV is
# fused or split, so the parity gate (parity/block_lora_parity.mojo) verifies it
# in the un-fused 7-slot form proven for base Z-Image.

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config


comptime L2P_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/l2p.json"


def l2p() raises -> TrainConfig:
    return read_model_config(String(L2P_CONFIG))
