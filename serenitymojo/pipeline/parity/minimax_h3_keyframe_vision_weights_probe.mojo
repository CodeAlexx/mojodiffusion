# serenitymojo/pipeline/parity/minimax_h3_keyframe_vision_weights_probe.mojo
#
# Resolves `models/text_encoder/minimax_h3_qwen3vl_vision.mojo`'s expected
# tensor list against the REAL FL2VA checkpoint, and cross-checks the shapes
# that make three of that module's five documented traps visible.
#
# Host only: safetensors HEADER reads via `tensor_info`, no tensor bytes, no
# DeviceContext, no page-cache warming of a 62 GiB checkpoint. Runs while the
# GPU is held.
#
# ── WHY THIS EXISTS, AND WHY IT IS NOT A SECOND TOWER ───────────────────────
# The tower module was committed (ff2c3e6) with its geometry gated at 25 host
# checks but its weights unverifiable — its own docstring says the names are
# "checked against the checkpoint index by the probe once
# text_encoder/model-00014-of-00014.safetensors lands", and at the time it had
# not. It has. This probe is that check, and nothing more: it ports no
# architecture, duplicates no forward, and touches no file that module owns.
#
# It matters for the keyframe path specifically. I2VA / FL2VA / L2VA presentations
# carry a `<Picture i>` vision block, so `pipeline/minimax_h3_i2va.mojo` is
# blocked on this tower running against these weights. Confirming the names and
# shapes resolve BEFORE anyone spends GPU time on a weighted forward is the
# cheapest possible ordering — a prefix mismatch (`model.visual.` vs `visual.`)
# or a fused-qkv layout surprise would otherwise surface halfway through a load.
#
# ── WHICH CHECKPOINT ────────────────────────────────────────────────────────
# FL2VA/text_encoder, NOT Ref2VA/text_encoder. They are separate copies of the
# same conditioner and only FL2VA is complete; for I2VA/FL2VA/L2VA the FL2VA
# copy is also the CORRECT one. A "0 of 351" reading against the still-arriving
# Ref2VA copy says nothing about this path.
#
# ── THE SHAPE CHECKS ARE THE TRAPS, RESTATED AS EVIDENCE ────────────────────
# Trap 1 (patch embed is a LINEAR): `patch_embed.proj.weight` is
# `[1152, 3, 2, 16, 16]` and `3*2*16*16 == 1536` — kernel equals stride equals
# the whole input extent, so a patch is one dot product. The shape proves it.
# Trap 3 (the two mergers normalize at DIFFERENT widths): `merger.norm.weight`
# is `[1152]` (before the merge reshape) while
# `deepstack_merger_list.i.norm.weight` is `[4608]` (after it). Two different
# widths in the same checkpoint is the trap made mechanical — swap them and the
# LOAD fails, rather than the numbers quietly drifting.
# Trap 5's rotary is not weight-visible (it is computed, not stored), so it
# stays the geometry probe's to gate; this file does not pretend to cover it.
#
# Run (no GPU, no weights read — headers only):
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/pipeline/parity/minimax_h3_keyframe_vision_weights_probe.mojo \
#     -o /tmp/h3_keyframe_vision_weights_probe -Xlinker -lm \
#   && /tmp/h3_keyframe_vision_weights_probe

from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_vision import (
    H3_VIS_DEPTH,
    H3_VIS_HIDDEN,
    H3_VIS_INTERMEDIATE,
    H3_VIS_MERGED_WIDTH,
    H3_VIS_NUM_DEEPSTACK,
    H3_VIS_NUM_POSITION_EMBEDDINGS,
    H3_VIS_OUT_HIDDEN,
    H3_VIS_PATCH_NUMEL,
    minimax_h3_vision_param_count,
    minimax_h3_vision_tensor_names,
)

comptime TEXT_ENCODER_DIR = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder"
)


struct Report(Copyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def ok(mut self, label: String, detail: String):
        self.checks += 1
        print("  ok  ", label, "—", detail)

    def fail(mut self, label: String, detail: String):
        self.checks += 1
        self.failures += 1
        print("  FAIL", label, "—", detail)

    def eq_int(mut self, label: String, got: Int, want: Int):
        if got == want:
            self.ok(label, String(got))
        else:
            self.fail(label, String("got ") + String(got) + ", want " + String(want))


def _shape_str(shape: List[Int]) -> String:
    var s = String("[")
    for i in range(len(shape)):
        if i > 0:
            s += ", "
        s += String(shape[i])
    return s + "]"


def _check_shape(
    mut report: Report,
    shards: ShardedSafeTensors,
    name: String,
    want: List[Int],
) raises:
    if not shards.has_tensor(name):
        report.fail(name, "absent")
        return
    var info = shards.tensor_info(name)
    var got = info.shape.copy()
    if len(got) != len(want):
        report.fail(name, String("rank ") + _shape_str(got) + ", want " + _shape_str(want))
        return
    for d in range(len(want)):
        if got[d] != want[d]:
            report.fail(name, _shape_str(got) + ", want " + _shape_str(want))
            return
    report.ok(name, _shape_str(got))


def main() raises:
    print("MiniMax-H3 keyframe unit — vision-tower WEIGHT RESOLUTION probe")
    print("  checkpoint:", String(TEXT_ENCODER_DIR))
    print("  (headers only — no tensor bytes are read)")
    print("")
    var report = Report()

    var shards = ShardedSafeTensors.open(String(TEXT_ENCODER_DIR))
    print("  opened", shards.num_shards(), "shard(s),", shards.num_tensors(), "tensors")
    print("")

    # ── [1] every name the tower expects resolves ───────────────────────────
    print("[1] minimax_h3_vision_tensor_names() vs the FL2VA index")
    var names = minimax_h3_vision_tensor_names()
    report.eq_int("names the tower declares", len(names), 351)
    var absent = 0
    var first_absent = String("")
    var not_bf16 = 0
    var first_wrong_dtype = String("")
    for i in range(len(names)):
        ref n = names[i]
        if not shards.has_tensor(n):
            absent += 1
            if first_absent == String(""):
                first_absent = n
            continue
        if shards.tensor_info(n).dtype != STDtype.BF16:
            not_bf16 += 1
            if first_wrong_dtype == String(""):
                first_wrong_dtype = n
    if absent == 0:
        report.ok(
            "all names present",
            String(len(names)) + " of " + String(len(names))
            + " resolve in the checkpoint index",
        )
    else:
        report.fail(
            "all names present",
            String(absent) + " absent, first: " + first_absent
            + " — a prefix or spelling mismatch, not a missing download",
        )
    if not_bf16 == 0:
        report.ok("all BF16", "no dtype surprises")
    else:
        report.fail("all BF16", String(not_bf16) + " differ, first: " + first_wrong_dtype)

    # ── [2] the shapes that make the traps mechanical ───────────────────────
    print("")
    print("[2] the shapes behind the module's documented traps")

    # Trap 1: the patch embed is a linear — kernel == stride == whole extent.
    var patch_shape: List[Int] = [
        H3_VIS_HIDDEN, 3, 2, 16, 16
    ]
    _check_shape(
        report, shards, String("model.visual.patch_embed.proj.weight"), patch_shape
    )
    var patch_numel = 3 * 2 * 16 * 16
    report.eq_int(
        "patch numel == H3_VIS_PATCH_NUMEL (one dot product per patch)",
        patch_numel,
        H3_VIS_PATCH_NUMEL,
    )

    # Fused QKV: one [3*hidden, hidden] tensor, not three.
    var qkv_shape: List[Int] = [3 * H3_VIS_HIDDEN, H3_VIS_HIDDEN]
    _check_shape(
        report, shards, String("model.visual.blocks.0.attn.qkv.weight"), qkv_shape
    )

    # Trap 3: the FINAL merger norms at 1152 (BEFORE the reshape); the DEEPSTACK
    # mergers norm at 4608 (AFTER it). Two widths, same checkpoint.
    var merger_norm: List[Int] = [H3_VIS_HIDDEN]
    _check_shape(report, shards, String("model.visual.merger.norm.weight"), merger_norm)
    var ds_norm: List[Int] = [H3_VIS_MERGED_WIDTH]
    _check_shape(
        report, shards,
        String("model.visual.deepstack_merger_list.0.norm.weight"), ds_norm
    )
    if H3_VIS_MERGED_WIDTH != H3_VIS_HIDDEN:
        report.ok(
            "the two mergers normalize at DIFFERENT widths",
            String(H3_VIS_HIDDEN) + " (final, pre-reshape) vs "
            + String(H3_VIS_MERGED_WIDTH) + " (deepstack, post-reshape)",
        )
    else:
        report.fail("the two merger widths differ", "they are equal")

    # Both merger families project to the language model's width.
    var out_shape: List[Int] = [H3_VIS_OUT_HIDDEN, H3_VIS_MERGED_WIDTH]
    _check_shape(
        report, shards, String("model.visual.merger.linear_fc2.weight"), out_shape
    )
    _check_shape(
        report, shards,
        String("model.visual.deepstack_merger_list.0.linear_fc2.weight"), out_shape
    )

    # The learned position table is the 48x48 grid the interpolation assumes.
    var pos_shape: List[Int] = [H3_VIS_NUM_POSITION_EMBEDDINGS, H3_VIS_HIDDEN]
    _check_shape(report, shards, String("model.visual.pos_embed.weight"), pos_shape)

    # Block MLP width, at the last block as well as the first — a per-block
    # config drift would otherwise hide behind block 0.
    var fc1_shape: List[Int] = [H3_VIS_INTERMEDIATE, H3_VIS_HIDDEN]
    _check_shape(
        report, shards,
        String("model.visual.blocks.") + String(H3_VIS_DEPTH - 1)
        + String(".mlp.linear_fc1.weight"),
        fc1_shape,
    )

    # ── [3] computed parameter count vs the checkpoint's own ────────────────
    print("")
    print("[3] parameter count — the module COMPUTES it from config; measure it")
    var measured = 0
    for i in range(len(names)):
        ref n = names[i]
        if not shards.has_tensor(n):
            continue
        var shape = shards.tensor_info(n).shape.copy()
        var numel = 1
        for d in range(len(shape)):
            numel *= shape[d]
        measured += numel
    var computed = minimax_h3_vision_param_count()
    if measured == computed:
        report.ok(
            "param count",
            String(measured) + " measured == computed (the config the module was"
            " ported from matches these weights)",
        )
    else:
        report.fail(
            "param count",
            String(measured) + " measured vs " + String(computed) + " computed"
            " — the module's config and this checkpoint disagree",
        )
    report.eq_int("deepstack taps", H3_VIS_NUM_DEEPSTACK, 3)

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("minimax_h3_keyframe_vision_weights probe FAILED")
