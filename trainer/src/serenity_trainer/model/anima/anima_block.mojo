# model/anima/anima_block.mojo — Anima MiniTrainDIT block training surface.
#
# Anima is a UNIFORM 28-block MiniTrainDIT: D=2048, H=16, Dh=128, F=8192 (PLAIN
# GELU layer1/layer2 MLP, NOT SwiGLU), AdaLN-LoRA modulation (per-block adaln from
# t_silu + base_adaln), q/k RMSNorm per head, 3D RoPE on the image tokens, and a
# SEPARATE cross-attention (attn2) to the frozen LLM-adapter context (JOINT=1024).
# Its per-block fwd/bwd is gated in serenitymojo (serenitymojo/models/anima/
# block.mojo) and the LoRA-on-projection variants in
# serenitymojo/models/anima/lora_block.mojo (10 slots: self_attn q/k/v/out +
# cross_attn q/k/v/out + mlp layer1/layer2 = ANIMA_SLOTS=10).
#
# This file RE-EXPORTS the proven serenitymojo block kinds under the
# serenity_trainer namespace (1:1, no math copied) — exactly the Ernie precedent
# (model/ernie/ernie_block.mojo re-exports the ERNIE block).

# per-block base block kind (saved/grads) — block.mojo
from serenitymojo.models.anima.block import (
    AnimaBlockSaved,
    AnimaBlockGrads,
)

# LoRA-on-projection block kind + slot indices — lora_block.mojo
from serenitymojo.models.anima.lora_block import (
    ANIMA_SLOTS,
    SLOT_SA_Q, SLOT_SA_K, SLOT_SA_V, SLOT_SA_O,
    SLOT_CA_Q, SLOT_CA_K, SLOT_CA_V, SLOT_CA_O,
    SLOT_MLP1, SLOT_MLP2,
    AnimaBlockLora,
    AnimaBlockLoraDevice,
    AnimaBlockLoraGrads,
    anima_block_lora_forward,
    anima_block_lora_backward,
    anima_block_lora_forward_device_tensor,
    anima_block_lora_backward_device_tensors,
)
