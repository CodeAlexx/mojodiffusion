# pdf/tests/layout_test.mojo
# Tests for pdf/metrics.mojo and pdf/layout.mojo.
#
# Run:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . \
#       pdf/tests/layout_test.mojo

from pdf.metrics import char_width_1000, text_width
from pdf.layout import wrap_text, align_x


def _check(cond: Bool, name: String, mut passed: Int, mut failed: Int) raises:
    if cond:
        passed += 1
        print("PASS:", name)
    else:
        failed += 1
        print("FAIL:", name)


def _approx(a: Float64, b: Float64) -> Bool:
    var d = a - b
    if d < 0.0:
        d = -d
    return d < 1e-9


def main() raises:
    var passed = 0
    var failed = 0

    # ---- metrics: Helvetica canonical AFM widths ----
    _check(
        char_width_1000("Helvetica", ord("A")) == 667,
        "helvetica A == 667",
        passed,
        failed,
    )
    _check(
        char_width_1000("Helvetica", ord(" ")) == 278,
        "helvetica space == 278",
        passed,
        failed,
    )
    _check(
        char_width_1000("Helvetica", ord("i")) == 222,
        "helvetica i == 222",
        passed,
        failed,
    )
    _check(
        char_width_1000("Helvetica", ord("0")) == 556,
        "helvetica 0 == 556",
        passed,
        failed,
    )

    # Case-insensitivity of the font name.
    _check(
        char_width_1000("helvetica", ord("M")) == 833,
        "helvetica (lowercase name) M == 833",
        passed,
        failed,
    )

    # ---- metrics: text_width ----
    # Courier monospace: 5 glyphs * 600/1000 * 10 = 30.0
    _check(
        _approx(text_width("Courier", 10.0, "ABCDE"), 30.0),
        "courier text_width ABCDE@10 == 30.0",
        passed,
        failed,
    )
    # Helvetica AB at size 1000: 667 + 667 = 1334.0
    _check(
        _approx(text_width("Helvetica", 1000.0, "AB"), 1334.0),
        "helvetica text_width AB@1000 == 1334.0",
        passed,
        failed,
    )

    # ---- layout: wrap_text ----
    var sentence = String(
        "the quick brown fox jumps over the lazy dog near the river bank"
    )
    var font = String("Helvetica")
    var size = 12.0
    var max_w = 80.0  # small box, forces multiple lines

    var lines = wrap_text(sentence, font, size, max_w)
    _check(len(lines) > 1, "wrap yields more than one line", passed, failed)

    # Every line fits within max_width (single-word-too-long exception does not
    # apply here: all words are short).
    var all_fit = True
    for i in range(len(lines)):
        if text_width(font, size, lines[i]) > max_w:
            all_fit = False
    _check(all_fit, "every wrapped line <= max_width", passed, failed)

    # Concatenating the wrapped lines' words reproduces the original word
    # sequence in order.
    var rejoined = String("")
    for i in range(len(lines)):
        if rejoined.byte_length() > 0:
            rejoined += " "
        rejoined += lines[i]
    _check(
        rejoined == sentence,
        "wrapped words reproduce original sequence",
        passed,
        failed,
    )

    # Over-long single word goes alone on its own line (and may overflow).
    var longword = String("supercalifragilisticexpialidocious")
    var lines2 = wrap_text(longword, font, size, 10.0)
    _check(
        len(lines2) == 1 and lines2[0] == longword,
        "over-long word kept on its own line",
        passed,
        failed,
    )

    # ---- layout: align_x ----
    var line = String("Hello")
    var box_x = 100.0
    var box_w = 400.0
    var lx = align_x(line, font, size, box_x, box_w, 0)
    var cx = align_x(line, font, size, box_x, box_w, 1)
    var rx = align_x(line, font, size, box_x, box_w, 2)
    _check(_approx(lx, box_x), "align left == box_x", passed, failed)
    _check(cx > lx, "align center x > left x", passed, failed)
    _check(rx > cx, "align right x > center x", passed, failed)

    # ---- summary ----
    print("---")
    print("passed:", passed, " failed:", failed)
    if failed == 0:
        print("ALL LAYOUT TESTS PASSED")
    else:
        raise Error("layout tests FAILED")
