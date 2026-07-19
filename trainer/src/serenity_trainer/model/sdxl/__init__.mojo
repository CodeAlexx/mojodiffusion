# model/sdxl — Stable Diffusion XL conv-UNet (eps-prediction DDPM) LoRA training
# surface, BORROWED (import-style re-export) from serenitymojo/models/sdxl.
# Mirrors the Ernie/Anima port precedent (serenity_trainer.model.{ernie,anima}
# re-export the proven serenitymojo stack via import; serenity_trainer imports
# serenitymojo directly with -I).
#
# UNLIKE the Flux-family DiTs, SDXL is a conv-UNet eps-predictor (NOT flow-match):
# dual text encoders (CLIP-L 768 + CLIP-G 1280) concatenated to a [.,77,2048]
# context + a [.,1280] CLIP-G POOLED embedding, plus SDXL added conditioning
# (add_time_ids = original_size/crop/target_size -> sin_embed_256 -> 1536, and the
# pooled embed -> 1280) concatenated into the ADM y vector [.,2816]. The UNet
# predicts NOISE epsilon (prediction_type "epsilon"), scaled-linear beta DDPM.
#
# LIKE Ernie/Anima (and UNLIKE Chroma, which needed a port-local row-stack loader
# to translate Flux's pre-fused offload keys), SDXL's serenitymojo stack is
# ALREADY a complete, self-contained real-dims conv-UNet LoRA training vertical:
# the resident forward/backward and per-adapter AdamW
# (serenitymojo.models.sdxl.sdxl_real_train + sdxl_unet_stack_lora) are the EXACT
# functions the real-loop driver (serenitymojo/training/train_sdxl_real.mojo:524
# forward / :548 backward / :555 adamw) calls. So there is NO port-local
# orchestration to copy — re-exporting the proven stack under
# serenity_trainer.model.sdxl is the faithful 1:1 mirror, and avoids the
# transcription drift hand-rewriting gated UNet math would risk.
#
# The SDXL per-ST LoRA backward MATH is independently proven torch-faithful:
# serenitymojo/models/sdxl/parity/lora_stack_parity.mojo passed 44/44
# (forward + every LoRA A/B grad, cos>=0.999, nonfinite 0) vs torch.autograd.
#
# SCOPE NOTE (honest): the serenitymojo SDXL LoRA covers the 11 SpatialTransformer
# blocks (N_ST=11, SDXL_SLOTS=10: attn1 q/k/v/o + attn2 q/k/v/o + ff proj/out).
# The REFERENCE training dump (sdxl_train_ref_*) trains a FULL-UNet LoRA (794
# modules incl. conv_in/out, time/add embedding, resnet convs, down/upsamplers),
# and runs at the native rectangular bucket resolution (latent 168x96). The
# serenitymojo real-dims forward (sdxl_real_forward[L]) is SQUARE-only
# (H0=L, H1=L//2, H2=L//4). These two facts mean a native forward-cos /
# grad-norm gate vs the rectangular full-LoRA oracle is NOT achievable with the
# current production spine (see smoke headers + handoff report). The oracle-
# consuming numeric gate that DOES hold is the loss replay (predicted vs target).
#
# Package layout (mirrors model/ernie):
#   sdxl_block.mojo            — re-export lora_block slots + block kinds/structs
#   weights.mojo              — re-export SdxlRealWeights builder + ST helpers
#   sdxl_unet_stack_lora.mojo — re-export real-dims forward/backward (the driver
#                               spine) + per-adapter AdamW + LoRA carrier/save
#   config.mojo               — read serenitymojo/configs/sdxl.json
