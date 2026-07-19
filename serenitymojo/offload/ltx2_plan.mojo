# offload/ltx2_plan.mojo — Block-offload plan for the LTX-2 video DiT.
#
# LTX-2 has ONE block kind (not double/single): 48 transformer_blocks.<i>.
# Key layout confirmed from the real ComfyUI checkpoint header:
#   model.diffusion_model.transformer_blocks.{i}.attn1.to_q.weight [4096,4096]
#   (ComfyUI prefix) OR transformer_blocks.{i}.* (diffusers prefix).
#
# This file is SEPARATE from offload/plan.mojo to avoid concurrent edit conflicts
# (another agent owns plan.mojo). The BlockPlan / OffloadConfig types are imported
# from plan.mojo; this file only adds the LTX-2–specific builder.
#
# Mojo 0.26.x+ / 1.0.0b1: def not fn; no fn.

from serenitymojo.offload.plan import BlockPlan, BlockKind, OffloadConfig


def build_ltx2_block_plan(num_layers: Int) -> BlockPlan:
    # LTX-2 (22B dev) stores all 48 video blocks under the full key prefix
    #   model.diffusion_model.transformer_blocks.{i}.*
    # (skeptic-confirmed against the real ltx-2.3-22b-dev header). The SEPARATE
    # audio branch (model.diffusion_model.audio_embeddings_connector.
    # transformer_1d_blocks.{i}.*) is NOT a video-LoRA target and is excluded by
    # using the transformer_blocks prefix only. The TurboPlannedLoader keys each
    # streamed Block by FULL tensor name, so the per-block weight extractor
    # (weights.mojo) prepends this same prefix when reading.
    var plan = BlockPlan(String("ltx2"))
    for i in range(num_layers):
        plan.append(
            String("model.diffusion_model.transformer_blocks.") + String(i),
            BlockKind.transformer(),
        )
    return plan^


def build_ltx2_48_block_plan() -> BlockPlan:
    # Default: 48 video transformer blocks (LTX-2.x standard depth).
    return build_ltx2_block_plan(48)
