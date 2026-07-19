# model/ernie/ernie_stack_lora.mojo — ERNIE-Image FULL DiT STACK *WITH LoRA*:
# resident-device forward (saving ckpt-inputs) + full-depth backward (training) +
# per-adapter AdamW + LoRA carrier/save/resume.
#
# This RE-EXPORTS the proven serenitymojo ERNIE stack
# (serenitymojo/models/ernie/ernie_stack_lora.mojo) under the serenity_trainer
# namespace — 1:1, no math copied. The re-exported functions are the EXACT ones
# the real-loop driver calls (serenitymojo/training/train_ernie_real.mojo:729
# forward / :748 backward / :758 adamw):
#
#   ernie_stack_lora_forward_resident_device   (ernie_stack_lora.mojo:756)
#       patch_embed(img)+text_proj(txt) -> concat(img,txt) -> 36 single LoRA
#       blocks (shared ModVecs `mv`) -> layer_norm -> modulate(f_scale,f_shift)
#       -> final_linear -> slice first N_IMG rows = [N_IMG, out_ch] velocity.
#   ernie_stack_lora_backward_resident_device  (ernie_stack_lora.mojo:844)
#       final-layer bwd -> reverse 36 blocks (recompute saved fwd per block) ->
#       collect LoRA d_A/d_B for all 36*7=252 adapters; shared-mod grads summed
#       (the shared-AdaLN source is FROZEN under LoRA scope, so they are not
#       optimised — only d_A/d_B feed AdamW). d_A is EXACTLY 0 at step 0 because
#       B is zero-init (d_A = scale * B^T d_y x^T == 0).
#   ernie_lora_adamw_step                       (ernie_stack_lora.mojo:914)
#       per-adapter proven _lora_adamw over every slot.
#
# WHY a re-export (not a port-local stack like model/chroma/chroma_stack_lora):
#   Chroma needed a port-local stack ONLY to translate Flux's pre-fused offload
#   keys into Chroma's separate-projection diffusers keys (the row-stack). ERNIE
#   has no such layout mismatch — its serenitymojo stack already streams the real
#   ERNIE diffusers keys and is the production driver's spine. Copying it would
#   only risk drift. (Mirror of the Chroma decision: import the gated math.)

from serenitymojo.models.ernie.ernie_stack_lora import (
    # LoRA carrier + device staging
    ErnieLoraSet,
    ErnieLoraDeviceSet,
    ErnieLoraGrads,
    build_ernie_lora_set,
    ernie_lora_set_to_device,
    ernie_lora_get,
    make_lora_adapter,
    # resident-device forward / predict / backward (the production spine)
    ernie_stack_lora_forward_resident_device,
    ernie_stack_lora_predict_resident_device,
    ernie_stack_lora_backward_resident_device,
    # optimizer
    ernie_lora_adamw_step,
    # SerenityTrainer raw-key save / resume
    ernie_lora_prefixes,
    save_ernie_lora,
    save_ernie_lora_state,
    load_ernie_lora_resume,
    load_ernie_lora_state,
)

# stack forward tape struct (saved ckpt inputs) lives in ernie_stack.mojo.
from serenitymojo.models.ernie.ernie_stack import ErnieStackForward
