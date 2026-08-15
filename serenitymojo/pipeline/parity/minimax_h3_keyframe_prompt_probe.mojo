# serenitymojo/pipeline/parity/minimax_h3_keyframe_prompt_probe.mojo
#
# Gates the §2.1 alignment-instruction emitters and the base-prompt builder in
# `pipeline/minimax_h3_ref_prompt.mojo`. Host only — no GPU, no weights, no
# torch.
#
# The expected strings are EXTRACTED from the vendor's own
# VIDEO_PROMPT_WRITING_GUIDE_base_en.md by scripts/minimax_h3_keyframe_prompt_oracle.py
# (its ```text blocks, matched by their opening words), so this compares the
# emitters against the DOC rather than against a retyped copy of it. That is the
# only way a byte-exact claim means anything for a template whose whole content
# is prose: an em dash silently becoming a hyphen, or `<Picture 1>` losing its
# brackets in the FL2VA line where the doc does not bracket it, is exactly the
# kind of drift a hand-written expectation would reproduce instead of catch.
#
# Run:
#   /home/alex/torchref/venv/bin/python scripts/minimax_h3_keyframe_prompt_oracle.py
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/pipeline/parity/minimax_h3_keyframe_prompt_probe.mojo \
#     -o /tmp/h3_keyframe_prompt_probe -Xlinker -lm \
#   && /tmp/h3_keyframe_prompt_probe output/minimax_h3_keyframe/keyframe_prompt_ref.json

from std.sys import argv
from std.collections import List

from json.parser import loads

from serenitymojo.pipeline.minimax_h3_ref_prompt import (
    MINIMAX_H3_TASK_FL2VA,
    MINIMAX_H3_TASK_I2VA,
    MINIMAX_H3_TASK_L2VA,
    MINIMAX_H3_TASK_T2VA,
    MiniMaxH3BasePrompt,
    minimax_h3_alignment_instruction,
    minimax_h3_alignment_seconds,
    minimax_h3_format_two_decimals,
    minimax_h3_keyframe_prompt,
    minimax_h3_task_from_keyframes,
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

    def eq_str(mut self, label: String, got: String, want: String):
        if got == want:
            self.ok(label, String(got.byte_length()) + " bytes, exact")
        else:
            # Report the first differing byte: a template drift is usually one
            # character (an em dash, a bracket), which a diff of whole strings
            # buries.
            var gb = got.as_bytes()
            var wb = want.as_bytes()
            var n = got.byte_length()
            if want.byte_length() < n:
                n = want.byte_length()
            var at = -1
            for i in range(n):
                if gb[i] != wb[i]:
                    at = i
                    break
            if at < 0:
                at = n
            self.fail(
                label,
                String("differ at byte ") + String(at) + " (got "
                + String(got.byte_length()) + " bytes, want "
                + String(want.byte_length()) + ")\n        got : " + got
                + "\n        want: " + want,
            )


def _read_text(path: String) raises -> String:
    var text: String
    with open(path, "r") as f:
        text = f.read()
    return text^


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: minimax_h3_keyframe_prompt_probe <h3_prompt_ref.json>")
        return
    var ref_path = String(args[1])
    print("MiniMax-H3 keyframe unit — §2.1 alignment instruction probe")
    print("  reference (extracted from the vendor guide):", ref_path)
    print("")
    var report = Report()
    var doc = loads(_read_text(ref_path))

    # ── [1] the worked cases (§5), which carry concrete N and S.SS ───────────
    print("[1] the guide's own worked cases")
    report.eq_str(
        "I2VA (Case 2, guide:187)",
        minimax_h3_alignment_instruction(MINIMAX_H3_TASK_I2VA, 192, 1),
        doc[String("case_i2va")].as_string(),
    )
    report.eq_str(
        "FL2VA (Case 3, guide:201) — 8.00 s = 192 frames",
        minimax_h3_alignment_instruction(MINIMAX_H3_TASK_FL2VA, 192, 1),
        doc[String("case_fl2va")].as_string(),
    )
    report.eq_str(
        "L2VA (Case 4, guide:215) — 6.00 s = 144 frames",
        minimax_h3_alignment_instruction(MINIMAX_H3_TASK_L2VA, 144, 1),
        doc[String("case_l2va")].as_string(),
    )

    # ── [2] the raw §2.1 templates with N and S.SS substituted ───────────────
    print("")
    print("[2] the §2.1 templates at a final shot != 1 (N=3, S.SS=7.25)")
    report.eq_str(
        "I2VA template (guide:19)",
        minimax_h3_alignment_instruction(MINIMAX_H3_TASK_I2VA, 174, 3),
        doc[String("template_i2va")].as_string(),
    )
    report.eq_str(
        "FL2VA template (guide:25)",
        minimax_h3_alignment_instruction(MINIMAX_H3_TASK_FL2VA, 174, 3),
        doc[String("sub_fl2va")].as_string(),
    )
    report.eq_str(
        "L2VA template (guide:31)",
        minimax_h3_alignment_instruction(MINIMAX_H3_TASK_L2VA, 174, 3),
        doc[String("sub_l2va")].as_string(),
    )

    # ── [3] `S.SS` for every legal frame count, vs CPython's "{:.2f}" ────────
    print("")
    print("[3] duration formatting vs CPython, every legal aligned frame count")
    ref two_dec = doc[String("two_decimals")]
    var frames = List[Int]()
    var n = 1
    while n < 400:
        if n % 17 == 5:
            var duration = Float64(n) / 24.0
            if duration >= 5.0 and duration <= 15.0:
                frames.append(n)
        n += 1
    var mismatch = 0
    var detail = String("")
    for i in range(len(frames)):
        var f = frames[i]
        var got = minimax_h3_alignment_seconds(f)
        var want = two_dec[String(f)].as_string()
        if got != want:
            mismatch += 1
            if detail == String(""):
                detail = String(f) + " frames: got " + got + ", want " + want
    if mismatch == 0:
        report.ok(
            "S.SS over " + String(len(frames)) + " frame counts",
            String("exact (") + minimax_h3_alignment_seconds(frames[0]) + " .. "
            + minimax_h3_alignment_seconds(frames[len(frames) - 1]) + ")",
        )
    else:
        report.fail("S.SS", String(mismatch) + " differ; first " + detail)

    # A ratio whose hundredth lands exactly on a tie must round to EVEN, the way
    # Python formats the exact value.
    report.eq_str("tie -> even (1/8 = 0.125)", minimax_h3_format_two_decimals(1, 8), String("0.12"))
    report.eq_str("tie -> even (3/8 = 0.375)", minimax_h3_format_two_decimals(3, 8), String("0.38"))

    # ── [4] task selection and the assembled prompt ──────────────────────────
    print("")
    print("[4] task selection and the assembled prompt")
    if minimax_h3_task_from_keyframes(True, False) == MINIMAX_H3_TASK_I2VA:
        report.ok("task(image only)", "I2VA")
    else:
        report.fail("task(image only)", "not I2VA")
    if minimax_h3_task_from_keyframes(False, True) == MINIMAX_H3_TASK_L2VA:
        report.ok("task(last_image only)", "L2VA")
    else:
        report.fail("task(last_image only)", "not L2VA")
    if minimax_h3_task_from_keyframes(True, True) == MINIMAX_H3_TASK_FL2VA:
        report.ok("task(both)", "FL2VA")
    else:
        report.fail("task(both)", "not FL2VA")
    if minimax_h3_task_from_keyframes(False, False) == MINIMAX_H3_TASK_T2VA:
        report.ok("task(neither)", "T2VA")
    else:
        report.fail("task(neither)", "not T2VA")

    var body = MiniMaxH3BasePrompt(
        String("[Shot 1] Live-action, cinematic, a test."),
        String("Room tone."),
        String("N/A"),
    )
    var t2va_prompt = minimax_h3_keyframe_prompt(MINIMAX_H3_TASK_T2VA, body, 192, 1)
    report.eq_str(
        "T2VA prompt has no instruction line (guide:14)",
        t2va_prompt,
        String(
            "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a test.\n\n"
            "overall_soundscape: Room tone.\n\n"
            "non_diegetic_music: N/A"
        ),
    )
    var i2va_prompt = minimax_h3_keyframe_prompt(MINIMAX_H3_TASK_I2VA, body, 192, 1)
    var want_i2va = (
        doc[String("case_i2va")].as_string() + "\n\n"
        + "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a test.\n\n"
        + "overall_soundscape: Room tone.\n\n"
        + "non_diegetic_music: N/A"
    )
    report.eq_str(
        "I2VA prompt = instruction, blank line, fields (guide:34)",
        i2va_prompt,
        want_i2va,
    )

    # An empty core field must fail loud rather than emit a header with nothing
    # after it — the guide requires `N/A`, not omission (§4.6/§4.7).
    var rejected = False
    try:
        var bad = MiniMaxH3BasePrompt(
            String("[Shot 1] x"), String("y"), String("")
        )
        var s = bad.render()
        _ = s.byte_length()
    except:
        rejected = True
    if rejected:
        report.ok("empty non_diegetic_music", "rejected — the guide wants N/A")
    else:
        report.fail("empty non_diegetic_music", "silently emitted an empty field")

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("minimax_h3_keyframe_prompt probe FAILED")
