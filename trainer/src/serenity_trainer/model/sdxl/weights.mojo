# model/sdxl/weights.mojo — REAL SDXL conv-UNet safetensors -> the bundled weight
# struct (SdxlRealWeights) the proven real-dims fwd/bwd consume, plus the per-ST
# shape/prefix helpers the driver uses to size the LoRA carrier.
#
# SDXL stores its UNet weights in the diffusers conv-UNet layout the proven
# builder expects (conv_in, time_embedding/add_embedding linear_1/2, down_blocks
# resnets+attentions+downsamplers, mid_block, up_blocks, conv_norm_out/conv_out).
# There is NO separate->fused row-stack to do (the genuine Chroma-specific
# compute), so this file simply RE-EXPORTS the proven serenitymojo builder
# (serenitymojo/models/sdxl/real_weights.mojo build_sdxl_real_weights:123) and
# the ST sizing helpers under the serenity_trainer namespace — 1:1, no
# transcription. This is the Ernie/Anima precedent (weights.mojo re-exports).
#
# The checkpoint is the real UNet-only BF16 single file
# (/home/alex/.serenity/models/checkpoints/sdxl_unet_bf16.safetensors); the
# resident fast path keeps base UNet weights frozen for LoRA training.

from serenitymojo.models.sdxl.real_weights import (
    build_sdxl_real_weights,
    sdxl_st_prefixes,
    sdxl_st_C,
    sdxl_st_Cff,
    sdxl_st_depth,
    load_emb_weights,
    load_st_weights,
)
