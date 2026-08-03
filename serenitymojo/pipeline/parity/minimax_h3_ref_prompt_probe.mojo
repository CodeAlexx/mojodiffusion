# serenitymojo/pipeline/parity/minimax_h3_ref_prompt_probe.mojo
#
# UNIT C probe — MiniMax-H3 ref2va structured prompt builder.
#
# Gates `serenitymojo/pipeline/minimax_h3_ref_prompt.mojo` against the VENDOR'S
# OWN BYTES: it reads the reference request script, pulls the `prompt` field out
# of its heredoc with the JSON parser, and byte-compares.
#
#   /home/alex/minimax_h3_ref/creator-MiniMax-H3/scripts/readme/
#     reproducible-768p-ref2va-request.sh   (`.prompt`, 3554 bytes)
#
# No pre-extraction step and no checked-in copy of the prompt: the script is
# parsed here, so the gate cannot drift from the vendor file. No DeviceContext,
# no GPU, no weights.
#
# ── WHAT IT PROVES ───────────────────────────────────────────────────────────
# [1] The six section headers, and their ORDER, are exactly the vendor's.
# [2] Round trip: split the vendor prompt into sections, feed those bodies back
#     through `MiniMaxH3Ref2VAPrompt.render()`, and the result is byte-identical
#     to the original 3554 bytes. That gates the assembly — headers, the single
#     newline between body lines, the blank line between sections, and the
#     absence of a trailing newline — against real vendor bytes rather than
#     against this port's own idea of the format.
# [3] The structured emitters reproduce specific vendor lines exactly:
#     a subject definition, three retention entries covering both the
#     with-qualifier and bare forms, and both dialogue tags.
# [4] The optional section really is optional: dropping `non_diegetic_music`
#     removes it and its separator, and nothing else.
#
# Run (no GPU, no weights):
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/pipeline/parity/minimax_h3_ref_prompt_probe.mojo \
#     -o <scratch>/minimax_h3_ref_prompt_probe \
#   && <scratch>/minimax_h3_ref_prompt_probe

from std.collections import List
from std.sys import argv

from json.parser import loads

from serenitymojo.pipeline.minimax_h3_ref_prompt import (
    MINIMAX_H3_RETENTION_FULLY_PRESERVED,
    MINIMAX_H3_RETENTION_PARTIALLY_COPY,
    MINIMAX_H3_RETENTION_REFERENCE,
    MiniMaxH3Ref2VAPrompt,
    MiniMaxH3RetentionEntry,
    MiniMaxH3Shot,
    MiniMaxH3SubjectDefinition,
    minimax_h3_dialogue,
    minimax_h3_ref2va_section_names,
    minimax_h3_summary_line,
)

comptime VENDOR_SCRIPT = (
    "/home/alex/minimax_h3_ref/creator-MiniMax-H3/scripts/readme/"
    "reproducible-768p-ref2va-request.sh"
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
            self.ok(label, String("exact (") + String(got.byte_length()) + " bytes)")
        else:
            self.fail(
                label,
                String("got '") + got + "' want '" + want + "'",
            )

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


def _split(s: String, sep: String) raises -> List[String]:
    """Split on a multi-byte separator."""
    var out = List[String]()
    var rest = s
    var seplen = sep.byte_length()
    while True:
        var at = rest.find(sep)
        if at < 0:
            out.append(rest)
            break
        out.append(String(rest[byte=0 : at]))
        rest = String(rest[byte = at + seplen :])
    return out^


def _vendor_prompt() raises -> String:
    """The `prompt` field of the vendor's ref2va request script.

    The script embeds its request as a `<<'JSON' ... JSON` heredoc; that slice
    is parsed with the JSON parser, so every escape (`\\n`, apostrophes) is
    unescaped exactly as curl would have sent it."""
    var script = _read_text(String(VENDOR_SCRIPT))
    var marker = String("<<'JSON'\n")
    var start = script.find(marker)
    if start < 0:
        raise Error(
            String("probe: no <<'JSON' heredoc in ") + String(VENDOR_SCRIPT)
        )
    start += marker.byte_length()
    var tail = String(script[byte=start:])
    var stop = tail.find(String("\nJSON\n"))
    if stop < 0:
        raise Error("probe: unterminated JSON heredoc in the vendor script")
    var request = loads(String(tail[byte=0 : stop]))
    if not request.contains(String("prompt")):
        raise Error("probe: the vendor request carries no 'prompt' field")
    return request[String("prompt")].as_string()


def main() raises:
    var args = argv()
    var script_path = String(VENDOR_SCRIPT)
    if len(args) >= 2:
        script_path = String(args[1])

    print("MiniMax-H3 ref2va UNIT C probe — structured prompt builder")
    print("  vendor script:", script_path)
    var report = Report()

    var vendor = _vendor_prompt()
    print("  vendor prompt:", vendor.byte_length(), "bytes")
    print("")

    # ── [1] section headers and their order ──────────────────────────────────
    print("[1] section headers and order")
    var sections = _split(vendor, String("\n\n"))
    var names = minimax_h3_ref2va_section_names()
    report.eq_int("section count", len(sections), 6)
    if len(sections) == 6:
        for i in range(6):
            var lines = _split(sections[i], String("\n"))
            report.eq_str(
                String("header[") + String(i) + "]",
                lines[0],
                names[i] + String(":"),
            )

    # ── [2] round trip through the builder, byte-for-byte ────────────────────
    print("")
    print("[2] round trip: vendor bodies -> builder -> vendor bytes")
    if len(sections) != 6:
        report.fail("round trip", "section split did not yield 6 sections")
    else:
        var bodies = List[List[String]]()
        for i in range(6):
            var lines = _split(sections[i], String("\n"))
            var body = List[String]()
            for j in range(1, len(lines)):
                body.append(lines[j])
            bodies.append(body^)

        report.eq_int("subject_definitions lines", len(bodies[0]), 4)
        report.eq_int("summary lines", len(bodies[1]), 1)
        report.eq_int("retention_analysis lines", len(bodies[2]), 4)
        report.eq_int("detailed_description lines", len(bodies[3]), 2)
        report.eq_int("overall_soundscape lines", len(bodies[4]), 1)
        report.eq_int("non_diegetic_music lines", len(bodies[5]), 1)

        var prompt = MiniMaxH3Ref2VAPrompt(
            bodies[0].copy(),
            bodies[1][0],
            bodies[2].copy(),
            bodies[3].copy(),
            bodies[4][0],
            bodies[5][0],
        )
        var rendered = prompt.render()
        if rendered == vendor:
            report.ok(
                "rendered == vendor prompt",
                String("byte-exact over ") + String(vendor.byte_length()) + " bytes",
            )
        else:
            # Locate the first differing byte, which is far more useful than
            # "they differ" on a 3.5 KB string.
            var a = rendered.as_bytes()
            var b = vendor.as_bytes()
            var limit = rendered.byte_length()
            if vendor.byte_length() < limit:
                limit = vendor.byte_length()
            var at = -1
            for i in range(limit):
                if a[i] != b[i]:
                    at = i
                    break
            report.fail(
                "rendered == vendor prompt",
                String("lengths ") + String(rendered.byte_length()) + " vs "
                + String(vendor.byte_length()) + ", first differing byte at "
                + String(at),
            )

        # ── [3] structured emitters vs specific vendor lines ─────────────────
        print("")
        print("[3] structured emitters reproduce vendor lines")
        var subject = MiniMaxH3SubjectDefinition(
            String("<Video 1>"),
            String("the source video for the editing task."),
        )
        report.eq_str("subject definition line", subject.render(), bodies[0][1])

        var audio_def = MiniMaxH3SubjectDefinition(
            String("<Audio 1>"),
            String(
                "the synchronized audio track of <Video 1>, providing the"
                " background music."
            ),
        )
        report.eq_str("audio definition line", audio_def.render(), bodies[0][2])

        # Bare form (no parenthesised qualifier).
        var bare = MiniMaxH3RetentionEntry(
            String("<Audio 2>"),
            String(""),
            String(MINIMAX_H3_RETENTION_REFERENCE),
            String(
                "the target audio references the male voice timbre from"
                " <Audio 2> to generate <Subject 1>'s spoken dialogue."
            ),
        )
        report.eq_str("retention entry (bare)", bare.render(), bodies[2][3])

        var copied = MiniMaxH3RetentionEntry(
            String("<Audio 1>"),
            String(""),
            String(MINIMAX_H3_RETENTION_PARTIALLY_COPY),
            String(
                "the atmospheric background music from <Audio 1> is reused in"
                " the target video, mixed beneath the newly added spoken"
                " dialogue."
            ),
        )
        report.eq_str("retention entry (partially_copy)", copied.render(), bodies[2][2])

        # Qualified form.
        var qualified = MiniMaxH3RetentionEntry(
            String("<Video 1>"),
            String("source video editing"),
            String(MINIMAX_H3_RETENTION_FULLY_PRESERVED),
            String(
                "the original camera framing, warm golden hour lighting, grassy"
                " hill setting, and background white lambs are maintained while"
                " the central character is edited."
            ),
        )
        report.eq_str("retention entry (qualified)", qualified.render(), bodies[2][1])

        # Dialogue tags appear inline in the shot prose.
        var line_one = minimax_h3_dialogue(
            String("English"), String("Follow the wind, live free.")
        )
        report.eq_str(
            "dialogue tag 1",
            line_one,
            String("<d>[English] Follow the wind, live free.</d>"),
        )
        var line_two = minimax_h3_dialogue(
            String("English"), String("Leave worries behind, enjoy the moment.")
        )
        report.eq_str(
            "dialogue tag 2",
            line_two,
            String("<d>[English] Leave worries behind, enjoy the moment.</d>"),
        )
        if bodies[3][1].find(line_one) >= 0 and bodies[3][1].find(line_two) >= 0:
            report.ok(
                "dialogue tags live in detailed_description",
                "both found inline in the shot line",
            )
        else:
            report.fail(
                "dialogue tags live in detailed_description",
                "not found in the shot line",
            )

        # The shot line itself is `[Shot 1] <prose>`.
        var shot_prefix = String("[Shot 1] ")
        if bodies[3][1].find(shot_prefix) == 0:
            var prose = String(bodies[3][1][byte = shot_prefix.byte_length() :])
            var shot = MiniMaxH3Shot(String("[Shot 1]"), prose)
            report.eq_str("shot line", shot.render(), bodies[3][1])
        else:
            report.fail("shot line", "vendor line does not open with '[Shot 1] '")

        # The summary's bracketed tag list.
        var summary = minimax_h3_summary_line(
            [
                String("video editing"),
                String("audio reference"),
                String("audio reuse"),
            ],
            String(
                "The target video is an edited version of <Video 1>."
            ),
        )
        var want_prefix = String(
            "[video editing + audio reference + audio reuse] The target video"
            " is an edited version of <Video 1>."
        )
        report.eq_str("summary tag list", summary, want_prefix)

        # ── [4] non_diegetic_music is optional ───────────────────────────────
        print("")
        print("[4] non_diegetic_music is the only optional section")
        var without = MiniMaxH3Ref2VAPrompt(
            bodies[0].copy(),
            bodies[1][0],
            bodies[2].copy(),
            bodies[3].copy(),
            bodies[4][0],
            String(""),
        )
        var short_render = without.render()
        var short_sections = _split(short_render, String("\n\n"))
        report.eq_int("sections without music", len(short_sections), 5)
        # Everything before it must be untouched: the shorter render is exactly
        # a prefix of the full one, cut at the separator.
        var full_render = MiniMaxH3Ref2VAPrompt(
            bodies[0].copy(),
            bodies[1][0],
            bodies[2].copy(),
            bodies[3].copy(),
            bodies[4][0],
            bodies[5][0],
        ).render()
        var cut = String(full_render[byte=0 : short_render.byte_length()])
        if cut == short_render:
            report.ok(
                "omission removes only the tail",
                String("prefix of ") + String(short_render.byte_length()) + " bytes intact",
            )
        else:
            report.fail(
                "omission removes only the tail",
                "the shorter render is not a prefix of the full one",
            )

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("minimax_h3_ref_prompt probe FAILED")
