# LTX-2 HQ multi-LoRA stack coverage/apply gate.
#
# This is not a generated-video quality test. It is the fail-closed gate that
# proves the HQ render engine can load and apply the LoRA surfaces commonly
# needed for high-quality LTX2 runs:
#   - official 22B distilled LoRA
#   - camera static LoRA
#   - IC detailer LoRA
#   - local Musubi-trained Comfy LoRA
#
# The test uses the same LoraSet.apply_to_av_block path as inference. Header-only
# coverage is not enough: every block-0 key for each LoRA must map to a real
# LTX2AVBlockWeights linear and apply successfully.
#
# Run:
#   pixi run mojo run -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/pipeline/ltx2_hq_lora_stack_smoke.mojo

from std.gpu.host import DeviceContext

from serenitymojo.lora import LoraSet
from serenitymojo.models.dit.ltx2_dit import LTX2Config, LTX2AVBlockWeights


comptime CKPT_FP8 = (
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
)
comptime DISTILLED_LORA = (
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-lora-384.safetensors"
)
comptime CAMERA_STATIC_LORA = (
    "/home/alex/.serenity/models/loras/ltx-2-19b-lora-camera-control-static.safetensors"
)
comptime DETAILER_LORA = (
    "/home/alex/.serenity/models/loras/ltx-2-19b-ic-lora-detailer.safetensors"
)
comptime LOCAL_MUSUBI_LORA = (
    "/home/alex/musubi-tuner/output/ltx23_eri2/ltx23_eri2_lora.comfy.safetensors"
)
comptime BLOCK_IDX = 0
comptime MULT = Float32(1.0)


def _check_counts(
    name: String,
    pairs: Int,
    mappings: Int,
    expected_pairs: Int,
    block_count: Int,
    expected_block_count: Int,
    global_count: Int,
    expected_global_count: Int,
) raises:
    print("  ", name, "pairs:", pairs, "mappings:", mappings,
          "block0:", block_count, "global:", global_count)
    if pairs != expected_pairs:
        raise Error(
            name + String(" pair count mismatch: got ")
            + String(pairs) + String(" expected ") + String(expected_pairs)
        )
    if mappings != pairs:
        raise Error(
            name + String(" mapping count mismatch: got ")
            + String(mappings) + String(" pairs ") + String(pairs)
        )
    if block_count != expected_block_count:
        raise Error(
            name + String(" block-0 mapping mismatch: got ")
            + String(block_count) + String(" expected ")
            + String(expected_block_count)
        )
    if global_count != expected_global_count:
        raise Error(
            name + String(" global mapping mismatch: got ")
            + String(global_count) + String(" expected ")
            + String(expected_global_count)
        )


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    print("=== LTX-2 HQ multi-LoRA stack coverage/apply gate ===")
    print("checkpoint:", CKPT_FP8)

    print("[load] block-0 AV weights")
    var block = LTX2AVBlockWeights.load(String(CKPT_FP8), BLOCK_IDX, cfg, ctx).to_f32(ctx)

    print("[load/apply] official distilled LoRA")
    var distilled = LoraSet.load(String(DISTILLED_LORA))
    var distilled_pairs = distilled.num_lora_pairs_in_file()
    var distilled_mappings = distilled.num_mappings()
    var distilled_block = distilled.ltx2_block_mapping_count(BLOCK_IDX)
    var distilled_global = distilled.ltx2_global_mapping_count()
    _check_counts(
        String("distilled"),
        distilled_pairs,
        distilled_mappings,
        1660,
        distilled_block,
        34,
        distilled_global,
        28,
    )
    var distilled_applied = distilled.apply_to_av_block(BLOCK_IDX, block, MULT, ctx)
    if distilled_applied != distilled_block:
        raise Error("distilled LoRA block apply count mismatch")

    print("[load/apply] camera static LoRA")
    var camera = LoraSet.load(String(CAMERA_STATIC_LORA))
    var camera_pairs = camera.num_lora_pairs_in_file()
    var camera_mappings = camera.num_mappings()
    var camera_block = camera.ltx2_block_mapping_count(BLOCK_IDX)
    var camera_global = camera.ltx2_global_mapping_count()
    _check_counts(
        String("camera_static"),
        camera_pairs,
        camera_mappings,
        1248,
        camera_block,
        26,
        camera_global,
        0,
    )
    var camera_applied = camera.apply_to_av_block(BLOCK_IDX, block, MULT, ctx)
    if camera_applied != camera_block:
        raise Error("camera static LoRA block apply count mismatch")

    print("[load/apply] IC detailer LoRA")
    var detailer = LoraSet.load(String(DETAILER_LORA))
    var detailer_pairs = detailer.num_lora_pairs_in_file()
    var detailer_mappings = detailer.num_mappings()
    var detailer_block = detailer.ltx2_block_mapping_count(BLOCK_IDX)
    var detailer_global = detailer.ltx2_global_mapping_count()
    _check_counts(
        String("detailer"),
        detailer_pairs,
        detailer_mappings,
        480,
        detailer_block,
        10,
        detailer_global,
        0,
    )
    var detailer_applied = detailer.apply_to_av_block(BLOCK_IDX, block, MULT, ctx)
    if detailer_applied != detailer_block:
        raise Error("detailer LoRA block apply count mismatch")

    print("[load/apply] local Musubi-trained LoRA")
    var local = LoraSet.load(String(LOCAL_MUSUBI_LORA))
    var local_pairs = local.num_lora_pairs_in_file()
    var local_mappings = local.num_mappings()
    var local_block = local.ltx2_block_mapping_count(BLOCK_IDX)
    var local_global = local.ltx2_global_mapping_count()
    _check_counts(
        String("local_musubi"),
        local_pairs,
        local_mappings,
        576,
        local_block,
        12,
        local_global,
        0,
    )
    var local_applied = local.apply_to_av_block(BLOCK_IDX, block, MULT, ctx)
    if local_applied != local_block:
        raise Error("local Musubi LoRA block apply count mismatch")

    var total_block = (
        distilled_applied + camera_applied + detailer_applied + local_applied
    )
    print("  applied block-0 stack deltas:", total_block)
    if total_block != 82:
        raise Error("HQ LoRA stack block-0 total apply count mismatch")

    print("LTX-2 HQ multi-LoRA stack coverage/apply PASS")
