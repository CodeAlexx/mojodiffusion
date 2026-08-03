# serenitymojo/models/text_encoder/parity/minimax_h3_qwen3vl_vision_gate.mojo
#
# ███ PENDING-GPU ███  Qwen3-VL VISION TOWER forward parity gate.
#
# Written now so the oracle contract is fixed before the forward is
# implemented. Blocked on TWO things:
#   1. the weights — all 351 vision tensors live in
#      text_encoder/model-00014-of-00014.safetensors, which has NOT downloaded;
#   2. the forward — `minimax_h3_vision_forward_seam` still raises.
# It never reports a pass, so it cannot be mistaken for green in a sweep.
#
# The tower's GEOMETRY is already gated, host-side, with no GPU and no weights:
# `minimax_h3_qwen3vl_vision_probe.mojo` (25 checks) covers cu_seqlens, the 2-D
# rotary coordinates, the f32 inv_freq table, the bilinear position-embed
# interpolation, token counts, the deepstack index spaces and the manifest.
# What is left here is the weighted forward.
#
# ── ORACLE CONTRACT ──────────────────────────────────────────────────────────
# `scripts/minimax_h3_vision_oracle.py`, run on the GPU in BF16 against the real
# Ref2VA text_encoder, constructing transformers' OWN `Qwen3VLVisionModel` (the
# installed 4.57.6) and dumping one safetensors:
#
#   in.pixel_values      bf16  [num_patches, 1536]  flattened patches, the
#                                                   layout patch_embed consumes
#   in.grid_thw          int32 [num_refs, 3]        (t, h, w) patch grids
#   out.after_patch      f32   [num_patches, 1152]  patch_embed + pos_embed
#   out.block_00         f32   [num_patches, 1152]  after vision block 0
#   out.block_08         f32   [num_patches, 1152]  after block 8  (deepstack tap)
#   out.block_16         f32   [num_patches, 1152]  after block 16 (deepstack tap)
#   out.block_24         f32   [num_patches, 1152]  after block 24 (deepstack tap)
#   out.block_26         f32   [num_patches, 1152]  after the last block
#   out.deepstack        f32   [3, num_tokens, 5120] the three merged taps
#   out.embeds           f32   [num_tokens, 5120]   the final merger's output
#
# Dumping the per-block states is the point: it lands a failure on ONE block
# rather than on "the tower". Run it with at least TWO references of DIFFERENT
# grids, one of them a t>1 video, so the per-FRAME cu_seqlens segmentation and
# the per-reference position-embed interpolation are both exercised — a
# single-image oracle would pass a tower that ignored both.
#
# ── BARS ─────────────────────────────────────────────────────────────────────
#   out.after_patch   cos >= 0.9999. Patch embed is one linear plus an
#                     interpolated lookup; the lookup INDICES are already
#                     bit-gated, so any miss here is the linear or the dtype.
#   out.block_NN      cos >= 0.999 each, checked IN ORDER and reported at the
#                     first block that drops. bf16 attention accumulates, so a
#                     slow decay across 27 blocks is expected and a CLIFF is the
#                     signal.
#   out.deepstack     cos >= 0.999 per tap. Check all THREE separately — a port
#                     that used the pre-shuffle norm for these (the final
#                     merger's form) still produces the right SHAPE, and this is
#                     the check that catches it.
#   out.embeds        cos >= 0.999.
#
# ── THE CHECK THAT MATTERS MOST, AND IS EASIEST TO OMIT ──────────────────────
# The three deepstack tensors are consumed at LANGUAGE decoder layers 0, 1, 2 —
# not at vision blocks 8/16/24, which is only where they are TAPPED. Whatever
# consumes this gate's output must therefore also assert that
# `minimax_h3_qwen3vl_streamed` adds them at layers 0-2 at visual token
# positions only. That is an integration check, not a tower check, and it does
# not belong here — but it is the one that silently produces plausible-looking
# conditioning if it is wrong, so it is recorded here rather than lost.
#
# Run (once the shard AND the forward exist):
#   python3 scripts/minimax_h3_vision_oracle.py
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/models/text_encoder/parity/minimax_h3_qwen3vl_vision_gate.mojo \
#     -o output/checks/minimax_h3_qwen3vl_vision_gate \
#   && output/checks/minimax_h3_qwen3vl_vision_gate

from std.collections import List
from std.sys import argv

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors

from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_vision import (
    minimax_h3_vision_tensor_names,
)


comptime ORACLE = (
    "/home/alex/mojodiffusion/output/minimax_h3_ref2va/vision_ref.safetensors"
)
comptime TEXT_ENCODER_DIR = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/Ref2VA/text_encoder"
)


def _oracle_keys() -> List[String]:
    return [
        String("in.pixel_values"),
        String("in.grid_thw"),
        String("out.after_patch"),
        String("out.block_00"),
        String("out.block_08"),
        String("out.block_16"),
        String("out.block_24"),
        String("out.block_26"),
        String("out.deepstack"),
        String("out.embeds"),
    ]


def main() raises:
    var args = argv()
    var oracle_path = String(ORACLE)
    if len(args) >= 2:
        oracle_path = String(args[1])

    print("MiniMax-H3 Qwen3-VL VISION TOWER parity gate")
    print("")
    print("  ###################################################################")
    print("  # PENDING-GPU. Written, not runnable. Blocked on BOTH:            #")
    print("  #   1. weights — all 351 vision tensors are in text_encoder shard #")
    print("  #      14 of 14, which has not downloaded                         #")
    print("  #   2. minimax_h3_vision_forward_seam, which still raises         #")
    print("  # The tower's GEOMETRY is gated host-side, no GPU, 25 checks:     #")
    print("  #   .../parity/minimax_h3_qwen3vl_vision_probe.mojo               #")
    print("  ###################################################################")
    print("")

    # ── weight availability ──────────────────────────────────────────────────
    print("  checkpoint:", String(TEXT_ENCODER_DIR))
    var names = minimax_h3_vision_tensor_names()
    var have_weights = False
    var present = 0
    try:
        var shards = ShardedSafeTensors.open(String(TEXT_ENCODER_DIR))
        for i in range(len(names)):
            if shards.has_tensor(names[i]):
                present += 1
        print("    vision tensors present:", present, "of", len(names))
        have_weights = present == len(names)
    except e:
        print("    checkpoint INCOMPLETE:", e)
    if not have_weights:
        print("    -> the vision tower cannot be loaded yet")

    # ── oracle availability ──────────────────────────────────────────────────
    print("  expected oracle:", oracle_path)
    var have_oracle = True
    try:
        var st = SafeTensors.open(oracle_path)
        var have = st.names()
        var wanted = _oracle_keys()
        var missing = List[String]()
        for i in range(len(wanted)):
            if wanted[i] not in have:
                missing.append(wanted[i])
        if len(missing) > 0:
            print("    oracle INCOMPLETE, missing:")
            for i in range(len(missing)):
                print("      ", missing[i])
            have_oracle = False
        else:
            print("    oracle present:", len(have), "tensors")
    except:
        print("    oracle ABSENT")
        have_oracle = False

    print("")
    raise Error(
        String("minimax_h3_qwen3vl_vision_gate: PENDING-GPU — weights ")
        + ("present" if have_weights else "MISSING")
        + " ("
        + String(present)
        + "/351), oracle "
        + ("present" if have_oracle else "MISSING")
        + ", and minimax_h3_vision_forward_seam is not implemented. See this"
        " file's ORACLE CONTRACT header for the exact key set and the per-key"
        " bars, and remember the integration check it records: the three"
        " deepstack tensors are consumed at LANGUAGE layers 0/1/2, not at the"
        " vision blocks 8/16/24 where they are tapped."
    )
