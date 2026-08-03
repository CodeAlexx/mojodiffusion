# serenitymojo/models/minimax_h3/parity/minimax_h3_presentation_parity.mojo
#
# MiniMax-H3 prompt-presentation parity gate.
#
# Reference: diffusers PR huggingface/diffusers#14355 at head e1b518df, run by
# scripts/minimax_h3_presentation_oracle.py over the REAL Qwen3-VL tokenizer we
# fetched — so both sides read the same tokenizer.json and a mismatch is the
# presentation, not the vocabulary.
#
# The label strings are also tokenized in isolation, so a failure can be
# localized to the BPE rather than to the presentation that calls it.
#
# `ref_mixed` and `ref_reordered` are the same three references in a different
# request order: same length, different ids, because labels are numbered per
# modality in request order. A port that ignored order passes on length and
# fails here.
#
# Oracle: python3 scripts/minimax_h3_presentation_oracle.py
# Run:
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_presentation_parity.mojo \
#     -o output/checks/minimax_h3_presentation_parity \
#   && output/checks/minimax_h3_presentation_parity

from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.minimax_h3.presentation import (
    MINIMAX_H3_IMAGE_PAD,
    MINIMAX_H3_REF_AUDIO,
    MINIMAX_H3_REF_IMAGE,
    MINIMAX_H3_REF_VIDEO,
    MINIMAX_H3_VIDEO_PAD,
    MINIMAX_H3_VISION_END,
    MINIMAX_H3_VISION_START,
    MiniMaxH3PresentationReference,
    minimax_h3_format_seconds,
    minimax_h3_ref2va_presentation,
    minimax_h3_special_id,
    minimax_h3_t2va_presentation,
)

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_presentation/presentation_ref.safetensors"
comptime TOKENIZER = "/home/alex/.serenity/models/text_encoders/qwen3vl-32b-instruct-h3/tokenizer.json"

comptime PROMPT_PLAIN = "A red fox trotting through a snowy pine forest, snow crunching underfoot"
comptime PROMPT_PUNCT = "Close-up: the subject's face, lit by neon. 35mm, f/1.4 — shallow depth!"
comptime PROMPT_UNICODE = "夜の街を歩く女性、ネオンの光 — cinematic, 24fps"
comptime PROMPT_SHORT = "a cat"


def _load_i64(ref st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.I64:
        raise Error(String("_load_i64: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Int64]()
    var out = List[Int]()
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


struct Report(Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def exact_int(mut self, label: String, got: List[Int], want: List[Int]):
        self.checks += 1
        if len(got) != len(want):
            self.failures += 1
            print("  FAIL", label, "length", len(got), "!=", len(want))
            return
        var bad = 0
        var first = -1
        for i in range(len(got)):
            if got[i] != want[i]:
                bad += 1
                if first < 0:
                    first = i
        if bad == 0:
            print("  ok  ", label, "exact over", len(got), "values")
        else:
            self.failures += 1
            print(
                "  FAIL", label, bad, "of", len(got), "differ; first at", first,
                "got", got[first], "want", want[first],
            )

    def equal_str(mut self, label: String, got: String, want: String):
        self.checks += 1
        if got == want:
            print("  ok  ", label, "=", got)
        else:
            self.failures += 1
            print("  FAIL", label, "got", got, "want", want)


def _counts(values: List[Int]) -> List[Int]:
    return values.copy()


def _check_t2va(
    mut report: Report,
    ref st: SafeTensors,
    ref tokenizer: Qwen3Tokenizer,
    name: String,
    prompt: String,
    image_counts: List[Int],
) raises:
    var out = minimax_h3_t2va_presentation(tokenizer, prompt, image_counts)
    report.exact_int(name + ".ids", out.ids, _load_i64(st, name + ".ids"))
    report.exact_int(name + ".tags", out.tags, _load_i64(st, name + ".tags"))


def _image(count: Int) -> MiniMaxH3PresentationReference:
    return MiniMaxH3PresentationReference(
        MINIMAX_H3_REF_IMAGE, False, List[Float64](), count
    )


def _audio() -> MiniMaxH3PresentationReference:
    return MiniMaxH3PresentationReference(
        MINIMAX_H3_REF_AUDIO, True, List[Float64](), 0
    )


def _video(
    stamps: List[Float64], has_audio: Bool, count: Int
) -> MiniMaxH3PresentationReference:
    return MiniMaxH3PresentationReference(MINIMAX_H3_REF_VIDEO, has_audio, stamps.copy(), count)


def _check_ref2va(
    mut report: Report,
    ref st: SafeTensors,
    ref tokenizer: Qwen3Tokenizer,
    name: String,
    references: List[MiniMaxH3PresentationReference],
) raises:
    var out = minimax_h3_ref2va_presentation(tokenizer, String(PROMPT_PLAIN), references)
    report.exact_int(name + ".ids", out.ids, _load_i64(st, name + ".ids"))
    report.exact_int(name + ".tags", out.tags, _load_i64(st, name + ".tags"))


def main() raises:
    print("MiniMax-H3 presentation parity gate")
    print("  reference:", REF)
    print("  tokenizer:", TOKENIZER)
    var st = SafeTensors.open(String(REF))
    var tokenizer = Qwen3Tokenizer(String(TOKENIZER))
    var report = Report()

    print("[1] special token ids")
    var specials = List[Int]()
    specials.append(minimax_h3_special_id(tokenizer, String(MINIMAX_H3_VISION_START)))
    specials.append(minimax_h3_special_id(tokenizer, String(MINIMAX_H3_VISION_END)))
    specials.append(minimax_h3_special_id(tokenizer, String(MINIMAX_H3_IMAGE_PAD)))
    specials.append(minimax_h3_special_id(tokenizer, String(MINIMAX_H3_VIDEO_PAD)))
    report.exact_int("specials", specials, _load_i64(st, "specials"))

    print("[2] label strings tokenized in isolation")
    var labels = [
        String("<Picture 1>: "), String("<Picture 2>: "), String("<Picture 9>: "),
        String("<Audio 1>: "), String("<Audio 3>: "),
        String("<Video 1>: "), String("<Video 2>: "),
        String("<0.2 seconds>"), String("<0.8 seconds>"),
        String("<1.2 seconds>"), String("<10.5 seconds>"),
    ]
    for i in range(len(labels)):
        report.exact_int(
            String("label.") + String(i),
            tokenizer.encode(labels[i]),
            _load_i64(st, String("label.") + String(i)),
        )

    print("[3] Python-compatible '{:.1f}' on block timestamps")
    report.equal_str("format(0.0)", minimax_h3_format_seconds(0.0), String("0.0"))
    report.equal_str("format(0.25)", minimax_h3_format_seconds(0.25), String("0.2"))
    report.equal_str("format(0.5)", minimax_h3_format_seconds(0.5), String("0.5"))
    report.equal_str("format(0.75)", minimax_h3_format_seconds(0.75), String("0.8"))
    report.equal_str("format(1.0)", minimax_h3_format_seconds(1.0), String("1.0"))
    report.equal_str("format(1.25)", minimax_h3_format_seconds(1.25), String("1.2"))
    report.equal_str("format(1.75)", minimax_h3_format_seconds(1.75), String("1.8"))
    report.equal_str("format(2.25)", minimax_h3_format_seconds(2.25), String("2.2"))
    report.equal_str("format(10.5)", minimax_h3_format_seconds(10.5), String("10.5"))

    print("[4] t2va / fl2va presentations")
    var none = List[Int]()
    _check_t2va(report, st, tokenizer, String("t2va_plain"), String(PROMPT_PLAIN), none)
    _check_t2va(report, st, tokenizer, String("t2va_punct"), String(PROMPT_PUNCT), none)
    _check_t2va(report, st, tokenizer, String("t2va_unicode"), String(PROMPT_UNICODE), none)
    _check_t2va(report, st, tokenizer, String("t2va_short"), String(PROMPT_SHORT), none)
    _check_t2va(report, st, tokenizer, String("fl2va_one"), String(PROMPT_PLAIN), [12])
    _check_t2va(report, st, tokenizer, String("fl2va_two"), String(PROMPT_PLAIN), [12, 7])
    _check_t2va(report, st, tokenizer, String("fl2va_unicode"), String(PROMPT_UNICODE), [3])

    print("[5] ref2va presentations (request order is a contract)")
    var two_stamps = List[Float64]()
    two_stamps.append(0.25)
    two_stamps.append(1.25)
    var three_stamps = two_stamps.copy()
    three_stamps.append(2.25)
    var mixed_stamps = List[Float64]()
    mixed_stamps.append(0.25)
    mixed_stamps.append(0.75)

    _check_ref2va(report, st, tokenizer, String("ref_image"), [_image(9)])
    _check_ref2va(report, st, tokenizer, String("ref_audio"), [_audio()])
    _check_ref2va(
        report, st, tokenizer, String("ref_video_silent"), [_video(two_stamps, False, 6)]
    )
    _check_ref2va(
        report, st, tokenizer, String("ref_video_sound"), [_video(three_stamps, True, 6)]
    )
    _check_ref2va(
        report, st, tokenizer, String("ref_mixed"),
        [_image(9), _video(mixed_stamps, True, 4), _audio()],
    )
    _check_ref2va(
        report, st, tokenizer, String("ref_reordered"),
        [_audio(), _video(mixed_stamps, True, 4), _image(9)],
    )
    _check_ref2va(report, st, tokenizer, String("ref_two_images"), [_image(9), _image(5)])

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks, all exact")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks differ")
        raise Error("MiniMax-H3 presentation parity gate failed")
