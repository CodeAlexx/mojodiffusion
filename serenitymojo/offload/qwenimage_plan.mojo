# offload/qwenimage_plan.mojo — Qwen-Image block-swap offload plan.
#
# Standalone copy/adaptation of the build_qwenimage_block_plan function from
# plan.mojo (read-only there per memory-safety rule). This file ONLY imports
# from plan.mojo (for BlockPlan / BlockKind / OffloadConfig) and does NOT
# modify plan.mojo.
#
# Qwen-Image: 60 ALL-double-stream blocks, key prefix transformer_blocks.{i}.
# byte_count_hint per block (FP8): 32 tensors * ~21 MB avg = 679_662_592 bytes.
#
# Mojo 0.26.x+.

from serenitymojo.offload.plan import BlockPlan, BlockKind, OffloadConfig


def build_qwenimage_offload_plan() -> BlockPlan:
    """60 double-stream transformer blocks for Qwen-Image block-swap offload.

    Equivalent to plan.mojo::build_qwenimage_block_plan() but lives here so
    callers can import from this file without touching plan.mojo.

    tensor_count_hint = 32  (verified from checkpoint header: 32 tensors/block).
    byte_count_hint  = 679_662_592  (~648 MB BF16/FP8 per block, from plan.mojo).
    """
    var plan = BlockPlan(String("qwen_image"))
    for i in range(60):
        plan.append(
            String("transformer_blocks.") + String(i),
            BlockKind.double_stream(),
            32,
            679662592,
        )
    return plan^
