# serenitymojo/pipeline/parity/minimax_h3_keyframe_presentation_probe.mojo
#
# Gates `pipeline/minimax_h3_keyframe_presentation.mojo` against the REAL H3
# tokenizer and the REAL image processor. Host only — no GPU, no model weights.
#
# ── WHAT THIS ADDS OVER THE TWO GATES IT SITS ON TOP OF ────────────────────
# `minimax_h3_image_grid_parity.mojo` gates the grid and
# `minimax_h3_presentation_parity.mojo` gates the ids/tags GIVEN a token count.
# This gates the JOIN, on canvases a real request actually resolves to — which
# is the only place TEXT_TOKENS, the video-tagged text rows, and the pad
# positions are decided. Passing both halves separately does not imply this
# passes: the halves agree on a token count that neither of them computes from
# a canvas.
#
# BIT-EXACT throughout. Token ids are integers, tags are integers, positions are
# integers — nothing here rounds, so anything short of equality is a bug.
#
# THE CHECK THAT MATTERS MOST is the pad POSITIONS, not their count. A correct
# count with a wrong placement produces a perfectly-shaped conditioning in which
# the vision embeds land on the wrong rows — no shape assertion anywhere
# downstream would notice, and the render would just be subtly wrong.
#
# Run (no GPU):
#   CUDA_VISIBLE_DEVICES="" /home/alex/torchref/venv/bin/python \
#     scripts/minimax_h3_keyframe_presentation_oracle.py
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/pipeline/parity/minimax_h3_keyframe_presentation_probe.mojo \
#     -o /tmp/h3_keyframe_presentation_probe -Xlinker -lm \
#   && /tmp/h3_keyframe_presentation_probe \
#        output/minimax_h3_keyframe/keyframe_presentation_ref.json

from std.sys import argv
from std.collections import List

from json.parser import loads

from serenitymojo.models.minimax_h3.presentation import (
    MINIMAX_H3_IMAGE_PAD,
    MINIMAX_H3_VISION_END,
    MINIMAX_H3_VISION_START,
    minimax_h3_special_id,
)
from serenitymojo.pipeline.minimax_h3_keyframe_presentation import (
    minimax_h3_keyframe_presentation,
    minimax_h3_keyframe_tag_summary,
    minimax_h3_keyframe_vision_grid,
)
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer

comptime PROCESSOR_DIR = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/processor"
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


def _read_text(path: String) raises -> String:
    var text: String
    with open(path, "r") as f:
        text = f.read()
    return text^


def main() raises:
    var args = argv()
    if len(args) < 2:
        print(
            "usage: minimax_h3_keyframe_presentation_probe"
            " <keyframe_presentation_ref.json>"
        )
        return
    var ref_path = String(args[1])
    print("MiniMax-H3 keyframe unit — presentation COMPOSITION probe")
    print("  reference:", ref_path)
    print("")
    var report = Report()
    var doc = loads(_read_text(ref_path))

    var tokenizer = Qwen3Tokenizer(String(PROCESSOR_DIR) + "/tokenizer.json")

    # ── [1] the three special ids the vision block is built from ───────────
    print("[1] special token ids")
    report.eq_int(
        "<|image_pad|>",
        minimax_h3_special_id(tokenizer, String(MINIMAX_H3_IMAGE_PAD)),
        doc[String("image_pad_id")].as_int(),
    )
    report.eq_int(
        "<|vision_start|>",
        minimax_h3_special_id(tokenizer, String(MINIMAX_H3_VISION_START)),
        doc[String("vision_start_id")].as_int(),
    )
    report.eq_int(
        "<|vision_end|>",
        minimax_h3_special_id(tokenizer, String(MINIMAX_H3_VISION_END)),
        doc[String("vision_end_id")].as_int(),
    )

    # ── [2] every case: grid, budget, ids, tags, pad positions ─────────────
    print("")
    print("[2] canvas -> grid -> presentation, on real canvases")
    var cases = doc[String("cases")]
    var num_cases = cases.length()
    var grid_bad = 0
    var budget_bad = 0
    var ids_bad = 0
    var tags_bad = 0
    var pads_bad = 0
    var first_fail = String("")

    for c in range(num_cases):
        var cs = cases[c]
        var ch = cs[String("canvas_h")].as_int()
        var cw = cs[String("canvas_w")].as_int()
        var nkf = cs[String("num_keyframes")].as_int()
        var prompt = cs[String("prompt")].as_string()
        var label = (
            String(cw) + "x" + String(ch) + " kf=" + String(nkf)
            + " p=" + String(cs[String("prompt_index")].as_int())
        )

        var grid = minimax_h3_keyframe_vision_grid(ch, cw)
        if (
            grid.grid_h != cs[String("grid_h")].as_int()
            or grid.grid_w != cs[String("grid_w")].as_int()
            or grid.num_vision_tokens != cs[String("vision_tokens")].as_int()
        ):
            grid_bad += 1
            if first_fail == String(""):
                first_fail = (
                    label + ": grid " + String(grid.grid_h) + "x" + String(grid.grid_w)
                    + " (" + String(grid.num_vision_tokens) + " tok) vs ref "
                    + String(cs[String("grid_h")].as_int()) + "x"
                    + String(cs[String("grid_w")].as_int()) + " ("
                    + String(cs[String("vision_tokens")].as_int()) + ")"
                )
            continue

        var p = minimax_h3_keyframe_presentation(tokenizer, prompt, ch, cw, nkf)

        if p.num_text_tokens() != cs[String("num_text_tokens")].as_int():
            budget_bad += 1
            if first_fail == String(""):
                first_fail = (
                    label + ": TEXT_TOKENS " + String(p.num_text_tokens())
                    + " vs ref " + String(cs[String("num_text_tokens")].as_int())
                )
            continue

        var want_ids = cs[String("token_ids")]
        var want_tags = cs[String("token_tags")]
        var want_pads = cs[String("pad_positions")]

        var bad_here = 0
        for i in range(len(p.token_ids)):
            if p.token_ids[i] != want_ids[i].as_int():
                bad_here += 1
                if first_fail == String(""):
                    first_fail = (
                        label + ": id[" + String(i) + "]=" + String(p.token_ids[i])
                        + " vs " + String(want_ids[i].as_int())
                    )
                break
        if bad_here > 0:
            ids_bad += 1
            continue

        for i in range(len(p.token_tags)):
            if p.token_tags[i] != want_tags[i].as_int():
                bad_here += 1
                if first_fail == String(""):
                    first_fail = (
                        label + ": tag[" + String(i) + "]=" + String(p.token_tags[i])
                        + " vs " + String(want_tags[i].as_int())
                    )
                break
        if bad_here > 0:
            tags_bad += 1
            continue

        if len(p.pad_positions) != want_pads.length():
            pads_bad += 1
            if first_fail == String(""):
                first_fail = (
                    label + ": " + String(len(p.pad_positions)) + " pads vs "
                    + String(want_pads.length())
                )
            continue
        for i in range(len(p.pad_positions)):
            if p.pad_positions[i] != want_pads[i].as_int():
                bad_here += 1
                if first_fail == String(""):
                    first_fail = (
                        label + ": pad[" + String(i) + "]="
                        + String(p.pad_positions[i]) + " vs "
                        + String(want_pads[i].as_int())
                    )
                break
        if bad_here > 0:
            pads_bad += 1

    if grid_bad == 0:
        report.ok("conditioner grid", String(num_cases) + " canvases, exact")
    else:
        report.fail("conditioner grid", String(grid_bad) + " of " + String(num_cases))
    if budget_bad == 0:
        report.ok(
            "TEXT_TOKENS budget",
            String(num_cases) + " cases, exact (prompt + label + 2 + vision per keyframe)",
        )
    else:
        report.fail("TEXT_TOKENS budget", String(budget_bad) + " of " + String(num_cases))
    if ids_bad == 0:
        report.ok("token ids", String(num_cases) + " cases, bit-exact")
    else:
        report.fail("token ids", String(ids_bad) + " of " + String(num_cases))
    if tags_bad == 0:
        report.ok(
            "token tags (vision block tagged VIDEO inside the text run)",
            String(num_cases) + " cases, bit-exact",
        )
    else:
        report.fail("token tags", String(tags_bad) + " of " + String(num_cases))
    if pads_bad == 0:
        report.ok(
            "image_pad POSITIONS",
            String(num_cases) + " cases, exact placement (not merely the right count)",
        )
    else:
        report.fail("image_pad positions", String(pads_bad) + " of " + String(num_cases))
    if first_fail != String(""):
        print("      first divergence:", first_fail)

    # ── [3] the invariants the pipeline relies on ──────────────────────────
    print("")
    print("[3] invariants at the product canvas (1184x768)")
    var one = minimax_h3_keyframe_presentation(
        tokenizer, String("a short prompt"), 768, 1184, 1
    )
    var two = minimax_h3_keyframe_presentation(
        tokenizer, String("a short prompt"), 768, 1184, 2
    )
    # A second keyframe costs a label plus a whole vision block. If the budget
    # did not grow by at least the vision tokens, the second block was dropped.
    var delta = two.num_text_tokens() - one.num_text_tokens()
    if delta >= one.vision_tokens_each + 2:
        report.ok(
            "FL2VA costs a second full vision block",
            String(delta) + " extra rows for keyframe 2 (>= "
            + String(one.vision_tokens_each + 2) + ")",
        )
    else:
        report.fail(
            "FL2VA costs a second full vision block",
            String(delta) + " extra rows — the second block was dropped or shared",
        )

    var summary = minimax_h3_keyframe_tag_summary(one)
    if summary[0] + summary[1] == one.num_text_tokens():
        report.ok(
            "tags partition the run",
            String(summary[0]) + " text + " + String(summary[1]) + " video = "
            + String(one.num_text_tokens()),
        )
    else:
        report.fail("tags partition the run", "counts do not sum")
    if summary[1] >= one.vision_tokens_each:
        report.ok(
            "video-tagged rows cover the vision block",
            String(summary[1]) + " >= " + String(one.vision_tokens_each)
            + " (pads) + 2 (start/end)",
        )
    else:
        report.fail("video-tagged rows", String(summary[1]))

    # A zero-keyframe request must be rejected — that is the t2va path, and
    # silently returning a prompt-only presentation here would produce a layout
    # with condition rows and no vision block.
    var rejected = False
    try:
        var bad = minimax_h3_keyframe_presentation(
            tokenizer, String("x"), 768, 1184, 0
        )
        _ = bad.num_text_tokens()
    except:
        rejected = True
    if rejected:
        report.ok("zero keyframes", "rejected (that is the t2va path)")
    else:
        report.fail("zero keyframes", "accepted")

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("minimax_h3_keyframe_presentation probe FAILED")
