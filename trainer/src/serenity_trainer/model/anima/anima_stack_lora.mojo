# model/anima/anima_stack_lora.mojo — Anima MiniTrainDIT FULL DiT STACK *WITH
# LoRA*: streamed forward (per-block weight swap, saving fwd tape) + full-depth
# streamed backward (training) + per-adapter AdamW + LoRA carrier/save/resume.
#
# This RE-EXPORTS the proven serenitymojo Anima stack
# (serenitymojo/models/anima/anima_stack_lora.mojo) under the serenity_trainer
# namespace — 1:1, no math copied. The re-exported functions are the EXACT ones
# the real-loop driver calls (serenitymojo/training/train_anima_real.mojo:660
# forward / :683 backward / :701 adamw):
#
#   anima_stack_lora_forward_streamed   (anima_stack_lora.mojo:359)
#       x_embedder(patches) -> 28 MiniTrainDIT LoRA blocks (each: AdaLN-LoRA
#       modulation from t_silu + base_adaln, self-attn(RoPE) + cross-attn(context)
#       + GELU MLP, streamed weights swapped in/out) -> final layer (silu->mod1->
#       mod2 + base_adaln half -> layer_norm -> scale/shift -> linear) =
#       [B*S_IMG, OUT_PATCH=64] velocity patches.
#   anima_stack_lora_backward_streamed  (anima_stack_lora.mojo:415)
#       final-layer bwd -> reverse 28 blocks (recompute saved fwd per block,
#       streamed) -> collect LoRA d_A/d_B for all 28*10=280 adapters. d_A is
#       EXACTLY 0 at step 0 because B is zero-init (d_A = scale * B^T d_y x^T == 0).
#   anima_lora_adamw_step                (anima_stack_lora.mojo:879)
#       per-adapter proven _lora_adamw over every slot.
#
# WHY a re-export (not a port-local stack like model/chroma/chroma_stack_lora):
#   Chroma needed a port-local stack ONLY to translate Flux's pre-fused offload
#   keys into Chroma's separate-projection diffusers keys (the row-stack). Anima
#   has no such layout mismatch — its serenitymojo stack already streams the real
#   Anima checkpoint keys and is the production driver's spine. Copying it would
#   only risk drift. (Mirror of the Ernie decision: import the gated math.)
#
# The Anima backward MATH is independently proven torch-faithful:
# serenitymojo/models/anima/parity/lora_stack_parity.mojo passed 64/64
# (forward + every LoRA A/B grad, cos>=0.999, nonfinite 0) vs torch.autograd.

from serenitymojo.models.anima.anima_stack_lora import (
    # LoRA carrier + adapter factory
    AnimaLoraSet,
    AnimaLoraDeviceSet,
    AnimaLoraGrads,
    build_anima_lora_set,
    anima_lora_set_to_device,
    anima_lora_get,
    anima_block_lora_for,
    make_lora_adapter,
    # streamed forward / backward (the production spine the driver calls)
    anima_stack_lora_forward_streamed,
    anima_stack_lora_backward_streamed,
    # resident-block forward / backward (synthetic-weight / parity path)
    anima_stack_lora_forward,
    anima_stack_lora_backward,
    # device-resident forward / predict / backward (reference trainer-parity fast path)
    anima_stack_lora_forward_device_resident,
    anima_stack_lora_predict_device_resident,
    anima_stack_lora_backward_device_resident,
    # optimizer
    anima_lora_adamw_step,
    # SerenityTrainer raw-key save / resume
    anima_lora_prefixes,
    anima_serenity_prefixes,
    save_anima_lora,
    save_anima_lora_serenity,
    save_anima_lora_state,
    load_anima_lora_resume,
    load_anima_lora_state,
)

# stack forward tape struct (saved fwd activations) lives in anima_stack.mojo.
from serenitymojo.models.anima.anima_stack import AnimaStackForward

# the per-adapter LoRA struct the carrier holds (re-exported for smoke override).
from serenitymojo.training.train_step import LoraAdapter
