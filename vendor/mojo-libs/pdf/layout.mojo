# pdf/layout.mojo
# Text wrapping and horizontal alignment built on pdf/metrics.mojo.
#
# Coordinates are PDF points. Wrapping is greedy word-wrap on spaces; a single
# word that is wider than max_width is placed alone on its own line (it will
# overflow, which is the standard typesetting fallback).

from pdf.metrics import text_width


# Split a string into words on ASCII spaces, collapsing runs of spaces.
# Tabs/newlines are treated as ordinary non-space characters here; callers that
# need to honor them should pre-split. Returns words with no empty entries.
def _split_words(s: String) raises -> List[String]:
    var words = List[String]()
    var cur = String("")
    for cp_slice in s.codepoint_slices():
        if cp_slice == " ":
            if cur.byte_length() > 0:
                words.append(cur.copy())
                cur = String("")
        else:
            cur += cp_slice
    if cur.byte_length() > 0:
        words.append(cur.copy())
    return words^


# Greedy word wrap. Each returned line's text_width <= max_width, except a line
# holding a single over-long word (which cannot be split further).
def wrap_text(
    s: String, font_name: String, size: Float64, max_width: Float64
) raises -> List[String]:
    var lines = List[String]()
    var words = _split_words(s)

    if len(words) == 0:
        return lines^

    var cur = String("")
    for i in range(len(words)):
        var w = words[i].copy()
        if cur.byte_length() == 0:
            # Start a fresh line with this word regardless of its width.
            cur = w.copy()
        else:
            var candidate = cur + " " + w
            if text_width(font_name, size, candidate) <= max_width:
                cur = candidate.copy()
            else:
                # Flush current line, start new one with this word.
                lines.append(cur.copy())
                cur = w.copy()

    if cur.byte_length() > 0:
        lines.append(cur.copy())

    return lines^


# Return the x at which to start drawing `line` so it is aligned within the
# box [box_x, box_x + box_w].
#   align == 0 -> left, 1 -> center, 2 -> right
def align_x(
    line: String,
    font_name: String,
    size: Float64,
    box_x: Float64,
    box_w: Float64,
    align: Int,
) raises -> Float64:
    if align == 1:
        var w = text_width(font_name, size, line)
        return box_x + (box_w - w) / 2.0
    if align == 2:
        var w = text_width(font_name, size, line)
        return box_x + (box_w - w)
    # Left (default).
    return box_x
