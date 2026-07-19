# model/sdxl/sdxl_unet_stack_lora.mojo — SDXL conv-UNet FULL STACK *WITH LoRA*:
# real-dims resident forward (saving acts) + full-depth backward (training) +
# per-adapter AdamW + LoRA carrier/save/resume.
#
# This RE-EXPORTS the proven serenitymojo SDXL stack
# (serenitymojo/models/sdxl/sdxl_real_train.mojo + sdxl_unet_stack_lora.mojo)
# under the serenity_trainer namespace — 1:1, no math copied. The re-exported
# functions are the EXACT ones the real-loop driver calls
# (serenitymojo/training/train_sdxl_real.mojo:524 forward / :548 backward /
# :555 adamw):
#
#   sdxl_real_forward[L]   (sdxl_real_train.mojo:271)
#       embed(t, ADM y) -> conv_in(4->320) -> encoder (in1..in8: ResBlocks +
#       11 ST LoRA blocks + 2 downsamples) -> middle (Res+ST+Res) -> decoder
#       (out0..out8: cat-skip + ResBlocks + ST LoRA + 2 upsamples) -> final
#       GroupNorm -> SiLU -> conv_out(320->4) = eps NHWC [1,L,L,4].
#       NOTE: SQUARE only (H0=L, H1=L//2, H2=L//4).
#   sdxl_real_backward[L]  (sdxl_real_train.mojo:432)
#       reverse walk; splits each decoder concat into (carry, skip) and adds the
#       skip slab into the matching encoder block's output grad; collects per-ST
#       LoRA d_A/d_B for every trained slot. d_A is EXACTLY 0 at step 0 because B
#       is zero-init (d_A = scale * B^T d_y x^T == 0).
#   sdxl_lora_adamw_step    (sdxl_unet_stack_lora.mojo:632)
#       per-adapter proven AdamW over every slot of one ST set.
#
# WHY a re-export (not a port-local stack like model/chroma/chroma_stack_lora):
#   Chroma needed a port-local stack ONLY to translate Flux's pre-fused offload
#   keys into Chroma's separate-projection diffusers keys (the row-stack). SDXL
#   has no such layout mismatch — its serenitymojo stack already streams the real
#   SDXL UNet keys and is the production driver's spine. Copying it would only
#   risk drift. (Mirror of the Ernie/Anima decision: import the gated math.)
#
# The SDXL per-ST LoRA backward MATH is independently proven torch-faithful:
# serenitymojo/models/sdxl/parity/lora_stack_parity.mojo passed 44/44
# (forward + every LoRA A/B grad, cos>=0.999, nonfinite 0) vs torch.autograd.

# bundled base weights + real-dims forward/backward + run-order index names.
from serenitymojo.models.sdxl.sdxl_real_train import (
    SdxlRealWeights,
    SdxlRealActs,
    SdxlRealFwd,
    SdxlRealGrads,
    sdxl_real_forward,
    sdxl_real_backward,
    N_ST, N_RES,
    ST_IN4, ST_IN5, ST_IN7, ST_IN8, ST_MID,
    ST_OUT0, ST_OUT1, ST_OUT2, ST_OUT3, ST_OUT4, ST_OUT5,
)

# per-ST LoRA carrier + per-ST forward/backward + AdamW + SerenityTrainer save/resume.
from serenitymojo.models.sdxl.sdxl_unet_stack_lora import (
    SdxlLoraSet,
    SdxlStLoraActs,
    SdxlStLoraFwd,
    SdxlStLoraGrads,
    build_sdxl_lora_set,
    sdxl_st_lora_forward,
    sdxl_st_lora_backward,
    sdxl_lora_adamw_step,
    save_sdxl_lora,
    save_sdxl_lora_with_text_encoder_flags,
    save_sdxl_lora_state,
)

# slot count for sizing the flat carrier (block*SDXL_SLOTS + slot).
from serenitymojo.models.sdxl.lora_block import SDXL_SLOTS

# the per-adapter LoRA struct the carrier holds + per-adapter AdamW + grad clip.
from serenitymojo.training.train_step import LoraAdapter, LoraGrads, _lora_adamw
