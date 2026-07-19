# model/ernie/ernie_block.mojo — ERNIE single-stream DiT block training surface.
#
# ERNIE-Image is a SINGLE-STREAM (0 double + 36 single) DiT: D=4096, H=32, Dh=128,
# F=12288 (GELU-GATED MLP: gate_proj/up_proj/linear_fc2, NOT the 2x Flux GELU),
# shared-AdaLN modulation (one mod source for every block), q/k norm, 3-axis
# RoPE split [32,48,48]. Its per-block fwd/bwd is gated in serenitymojo
# (serenitymojo/models/ernie/block.mojo ernie_block_forward:242 /
# ernie_block_backward:324) and the LoRA-on-projection variants in
# serenitymojo/models/ernie/lora_block.mojo (7 slots: to_q/to_k/to_v/to_out.0 +
# mlp gate_proj/up_proj/linear_fc2 = ERNIE_SLOTS=7).
#
# This file RE-EXPORTS the proven serenitymojo block kinds under the
# serenity_trainer namespace (1:1, no math copied) — exactly the Chroma precedent
# (model/chroma/chroma_block.mojo re-exports the Flux block).

# per-block block kind (saved/forward/grads/modvecs) — block.mojo
from serenitymojo.models.ernie.block import (
    ErnieModVecs,
    ErnieBlockSaved,
    ErnieBlockForward,
    ErnieBlockGrads,
    ernie_block_forward,
    ernie_block_backward,
)

# LoRA-on-projection block kind + slot indices — lora_block.mojo
from serenitymojo.models.ernie.lora_block import (
    ERNIE_SLOTS, SLOT_Q, SLOT_K, SLOT_V, SLOT_O, SLOT_GATE, SLOT_UP, SLOT_DOWN,
    ErnieBlockLora,
    ErnieBlockLoraDevice,
    ErnieBlockLoraGrads,
    ernie_block_lora_forward,
    ernie_block_lora_forward_device_tensor,
    ernie_block_lora_backward,
    ernie_block_lora_backward_device_tensors,
)
