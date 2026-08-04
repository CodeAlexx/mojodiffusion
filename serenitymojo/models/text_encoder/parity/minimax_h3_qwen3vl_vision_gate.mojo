# serenitymojo/models/text_encoder/parity/minimax_h3_qwen3vl_vision_gate.mojo
#
# ███ SUPERSEDED ███  Qwen3-VL VISION TOWER forward parity gate — this file's
# ORACLE CONTRACT (below) has been FULFILLED elsewhere; run those gates:
#
#   1. HOST forward, CPU/f32, derived bars (4x torch's own f32 error):
#        minimax_h3_qwen3vl_vision_cpu_gate.mojo
#        + scripts/minimax_h3_vision_oracle.py                — 10/10 green
#   2. DEVICE forward, GPU/bf16, matching dtype, at the REAL 2304-patch
#      keyframe geometry, per-stage bars derived from torch's own measured
#      bf16 noise floor:
#        models/minimax_h3_device/parity/minimax_h3_vision_tower_device_parity.mojo
#        + scripts/minimax_h3_vision_tower_device_oracle.py   — 12/12 green
#
# The two blockers this file was written against are both RESOLVED:
#   1. the weights — all 351 vision tensors landed in FL2VA's
#      text_encoder/model-00014-of-00014.safetensors (present, checked below);
#   2. the forward — the seam stub was replaced by the arbitration-kept
#      `minimax_h3_vision_forward` (host) in minimax_h3_qwen3vl_vision.mojo,
#      and by `minimax_h3_vision_forward_device` (GPU) in
#      models/minimax_h3_device/vision_tower_device.mojo.
# This file still never reports a pass, so it cannot be mistaken for green in
# a sweep — it raises SUPERSEDED, naming the real gates.
#
# The tower's GEOMETRY gate is unchanged and still green, host-side:
# `minimax_h3_qwen3vl_vision_probe.mojo` (25 checks) covers cu_seqlens, the 2-D
# rotary coordinates, the f32 inv_freq table, the bilinear position-embed
# interpolation, token counts, the deepstack index spaces and the manifest.
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
    print("  # SUPERSEDED. This file's oracle contract was fulfilled by:       #")
    print("  #   CPU/f32 host gate  (10/10):                                   #")
    print("  #     .../parity/minimax_h3_qwen3vl_vision_cpu_gate.mojo          #")
    print("  #   GPU/bf16 device gate at 2304 patches (12/12):                 #")
    print("  #     models/minimax_h3_device/parity/                            #")
    print("  #       minimax_h3_vision_tower_device_parity.mojo                #")
    print("  # Run THOSE. This file only checks artifact presence and raises.  #")
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
        String("minimax_h3_qwen3vl_vision_gate: SUPERSEDED — weights ")
        + ("present" if have_weights else "MISSING")
        + " ("
        + String(present)
        + "/351), oracle "
        + ("present" if have_oracle else "MISSING")
        + ". The weighted forward exists and is gated elsewhere: CPU/f32 by"
        " minimax_h3_qwen3vl_vision_cpu_gate.mojo (10/10) and GPU/bf16 by"
        " models/minimax_h3_device/parity/minimax_h3_vision_tower_device_parity"
        ".mojo (12/12, real 2304-patch keyframe geometry). Run those. The"
        " integration check this file records still stands: the three deepstack"
        " tensors are consumed at LANGUAGE layers 0/1/2, not at the vision"
        " blocks 8/16/24 where they are tapped."
    )
