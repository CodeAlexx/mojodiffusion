# model/sdxl/sdxl_block.mojo — SDXL SpatialTransformer LoRA block training surface.
#
# SDXL's trainable LoRA lives on the 11 SpatialTransformer (ST) blocks of the
# conv-UNet. Each ST basic transformer block exposes SDXL_SLOTS=10 LoRA slots:
#   attn1 (self-attn)  to_q / to_k / to_v / to_out.0   (slots 0..3)
#   attn2 (cross-attn) to_q / to_k / to_v / to_out.0   (slots 4..7)
#   ff GEGLU           net.0.proj (in-proj) / net.2 (out-proj)  (slots 8..9)
# The per-ST fwd/bwd is gated in serenitymojo (serenitymojo/models/sdxl/
# sdxl_unet_stack_lora.mojo sdxl_st_lora_forward:523 / sdxl_st_lora_backward:577)
# and the per-block LoRA carrier in serenitymojo/models/sdxl/lora_block.mojo.
#
# This file RE-EXPORTS the proven serenitymojo block kinds under the
# serenity_trainer namespace (1:1, no math copied) — exactly the Ernie/Anima
# precedent (model/ernie/ernie_block.mojo re-exports the ERNIE block).

# LoRA-on-projection slot indices + per-block carrier/grad structs — lora_block.mojo
from serenitymojo.models.sdxl.lora_block import (
    SDXL_SLOTS,
    SLOT_A1_Q, SLOT_A1_K, SLOT_A1_V, SLOT_A1_O,
    SLOT_A2_Q, SLOT_A2_K, SLOT_A2_V, SLOT_A2_O,
    SLOT_FF_PROJ, SLOT_FF_OUT,
    SdxlLoraGrads,
    SdxlBlockLora,
    SdxlBlockLoraGrads,
)
