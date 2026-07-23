# serenitymojo/offload/wan22_plan.mojo — Block-swap offload plan for Wan2.2-T2V.
#
# Wan2.2-T2V-14B (low-noise) checkpoint key layout (confirmed from
# wan2.2_t2v_low_noise_14b_fp16.safetensors header):
#   blocks.{i}.self_attn.{q,k,v,o}.{weight,bias}     [5120,5120] / [5120] F16
#   blocks.{i}.self_attn.norm_{q,k}.weight            [5120]       F16
#   blocks.{i}.cross_attn.{q,k,v,o}.{weight,bias}    [5120,5120] / [5120] F16
#   blocks.{i}.cross_attn.norm_{q,k}.weight           [5120]       F16
#   blocks.{i}.norm3.{weight,bias}                    [5120]       F16
#   blocks.{i}.ffn.0.{weight,bias}  [13824,5120]/[13824]          F16
#   blocks.{i}.ffn.2.{weight,bias}  [5120,13824]/[5120]           F16
#   blocks.{i}.modulation            [1,6,5120]                    F16
#   i in range(40)  -> 40 WanAttentionBlocks
#
# Non-block keys (resident in Wan22StackBase):
#   patch_embedding.{weight,bias}   [5120,16,1,2,2] / [5120]
#   text_embedding.{0,2}.{weight,bias}
#   time_embedding.{0,2}.{weight,bias}
#   time_projection.1.{weight,bias} [30720,5120] / [30720]
#   head.head.{weight,bias}         [64,5120] / [64]
#   head.modulation                 [1,2,5120]
#
# All 40 blocks use BlockKind.transformer() (they are single-stream
# WanAttentionBlocks, not the double/single Flux topology).
#
# DO NOT import from or modify turbo_loader.mojo / plan.mojo.

from serenitymojo.offload.plan import BlockPlan, BlockKind

comptime WAN22_14B_NUM_BLOCKS = 40


def build_wan22_block_plan(
    num_blocks: Int = WAN22_14B_NUM_BLOCKS, prefix: String = String(""),
) -> BlockPlan:
    """Build the block-swap offload plan for a Wan2.2-T2V checkpoint.

    Each entry maps to prefix "{prefix}blocks.{i}" in the safetensors file. The
    TurboPlannedLoader will stream block i's tensors when that prefix is
    awaited. Prefixes are WITHOUT trailing dots; the loader adds "." when
    building the full key.

    `prefix` is "" for bare Wan2.2 keys ("blocks.0.self_attn.q.weight") and
    "model.diffusion_model." for the ComfyUI Wan2.1 layout
    ("model.diffusion_model.blocks.0.self_attn.q.weight"). It MUST match the
    prefix detected by detect_wan22_prefix / passed to load_wan22_stack_base so
    the streamer and the resident base agree. The loader's _extract_block_prefix
    already groups keys by their first all-numeric segment, so a prefixed block
    key indexes under "model.diffusion_model.blocks.N." — which normalizes to
    exactly the plan prefix built here.

    num_blocks=40 for wan2.2_t2v_low_noise_14b (14B param model).
    """
    var plan = BlockPlan(String("wan22"))
    for i in range(num_blocks):
        # 27 tensors per block (8 linear weights + 8 biases + 2 qk-norms x2 +
        # norm3 w/b + ffn.0/2 w/b + modulation = 27). Byte count: rough estimate
        # based on 5120-dim F16: (8*5120*5120*2 + 8*5120*2 + 4*5120*2
        # + 2*5120*2 + 2*13824*5120*2 + 2*13824*2 + 1*6*5120*2) bytes.
        plan.append(
            prefix + "blocks." + String(i),
            BlockKind.transformer(),
            27,
            0,
        )
    return plan^


def build_wan22_14b_block_plan() -> BlockPlan:
    return build_wan22_block_plan(WAN22_14B_NUM_BLOCKS)


# ── Wan2.1 I2V-14B DUAL cross-attn block plan ─────────────────────────────────
# Same 40-block transformer topology as T2V, but each block additionally streams
# the image-branch cross-attn weights:
#   blocks.{i}.cross_attn.k_img.{weight,bias}   [5120,5120]/[5120]
#   blocks.{i}.cross_attn.v_img.{weight,bias}   [5120,5120]/[5120]
#   blocks.{i}.cross_attn.norm_k_img.weight     [5120]
# -> 27 base + 5 = 32 tensors/block. tensor_count_hint is advisory (the loader
# groups ALL "blocks.N.*" keys by the numeric segment, so the extra keys stream
# regardless); the corrected hint just keeps the plan honest.
def build_wan21_i2v_block_plan(
    num_blocks: Int = WAN22_14B_NUM_BLOCKS, prefix: String = String(""),
) -> BlockPlan:
    var plan = BlockPlan(String("wan21_i2v"))
    for i in range(num_blocks):
        plan.append(
            prefix + "blocks." + String(i),
            BlockKind.transformer(),
            32,
            0,
        )
    return plan^
