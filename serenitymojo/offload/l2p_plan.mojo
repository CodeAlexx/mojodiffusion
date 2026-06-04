# offload/l2p_plan.mojo — Block-offload plan for the Z-Image L2P DiT.
#
# L2P reuses the Z-Image-Turbo DiT body VERBATIM: 2 noise_refiner blocks
# (modulated), 2 context_refiner blocks (unmodulated), and 30 main layers.
# All live in a single-file safetensors checkpoint:
#   /home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors
#
# This file is SEPARATE from offload/plan.mojo (per user rule: do NOT edit
# plan.mojo or turbo_loader.mojo). The BlockPlan / OffloadConfig / BlockKind
# types are imported from plan.mojo; this file only adds L2P-specific builders.
#
# Key layout confirmed from the real checkpoint header (model-1k-merge.safetensors,
# 545 tensors, 2026-06-03 header scan):
#   noise_refiner.{0,1}.attention.to_q.weight  [3840,3840] BF16 (modulated)
#   context_refiner.{0,1}.attention.to_q.weight [3840,3840] BF16 (unmodulated)
#   layers.{0..29}.attention.to_q.weight        [3840,3840] BF16/F32 mixed
# There is NO all_final_layer prefix; the local_decoder head is separate.
# The training LoRA targets are:
#   noise_refiner.{i}.{attention,feed_forward}  (NR, trainable in L2P)
#   layers.{i}.{attention,feed_forward}         (MAIN, trainable)
# context_refiner has no adaLN so it is excluded from the OT LoRA filter (same
# as zimage base: ^(?=.*attention)(?!.*context_refiner).*,^(?=.*feed_forward)(?!.*context_refiner).*).
#
# Mojo 0.26.x+ / 1.0.0b1: def not fn; no fn.

from serenitymojo.offload.plan import BlockPlan, BlockKind, OffloadConfig


comptime L2P_NUM_NR = 2
comptime L2P_NUM_CR = 2
comptime L2P_MAIN_DEPTH = 30
comptime L2P_TOTAL_BLOCKS = L2P_NUM_NR + L2P_NUM_CR + L2P_MAIN_DEPTH


def build_l2p_block_plan() -> BlockPlan:
    """Build the full L2P block plan: NR + CR + main layers.

    The prefix scheme matches the real checkpoint key layout and the zimage
    training weight loader (`load_zimage_block_weights_prefixed_mixed`).
    Context-refiner blocks are included in the plan for completeness but are
    NOT trained (no adaLN => excluded from OT LoRA filter); their LoRA
    adapters are allocated but frozen (never touched by the main-only AdamW).
    """
    var plan = BlockPlan(String("l2p"))
    for i in range(L2P_NUM_NR):
        plan.append(
            String("noise_refiner.") + String(i),
            BlockKind.transformer(),
        )
    for i in range(L2P_NUM_CR):
        plan.append(
            String("context_refiner.") + String(i),
            BlockKind.transformer(),
        )
    for i in range(L2P_MAIN_DEPTH):
        plan.append(
            String("layers.") + String(i),
            BlockKind.transformer(),
        )
    return plan^
