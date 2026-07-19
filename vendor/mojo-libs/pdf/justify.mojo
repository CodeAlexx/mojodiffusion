# pdf/justify.mojo
# Full text justification for the standard-14 fonts using the PDF TJ operator.
#
# A "justified" line is rendered as a single TJ array:
#   [ (word1) adj1 (word2) adj2 ... (wordN) ] TJ
# where each adj is a number in THOUSANDTHS of an em. The PDF imaging model
# multiplies a TJ number by -size/1000 and applies it to the text position, so a
# NEGATIVE number moves the pen to the RIGHT (i.e. it ADDS horizontal space).
#
# To add `g` points of extra space after a word we emit adj = -g*1000/size.
# Distributing the slack (box_w - natural_width) evenly over the N-1 inter-word
# gaps fills the box exactly.
#
# Words are split on ASCII spaces; each word is escaped ( ( ) \ ) and transcoded
# UTF-8 -> WinAnsi exactly like pdf/text.mojo's show_text.

from pdf.text import (
    begin_text,
    end_text,
    set_font,
    text_pos,
    show_text,
    _put_str,
    _put_real,
    _to_winansi,
)
from pdf.metrics import text_width
from pdf.layout import wrap_text


def _split_on_spaces(s: String) raises -> List[String]:
    # Split on ASCII spaces, collapsing runs; no empty entries.
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


def _put_word_literal(mut b: List[UInt8], w: String) raises:
    # Emit "(...)" for a single word: transcode UTF-8 -> WinAnsi and escape
    # the three special literal-string characters '(' ')' '\'.
    b.append(UInt8(ord("(")))
    var lp = ord("(")
    var rp = ord(")")
    var bs = ord("\\")
    for cp in w.codepoints():
        var wc = _to_winansi(Int(cp))
        if wc == lp or wc == rp or wc == bs:
            b.append(UInt8(bs))
        b.append(UInt8(wc & 0xFF))
    b.append(UInt8(ord(")")))


def show_justified_line(
    mut b: List[UInt8],
    font_name: String,
    size: Float64,
    line: String,
    box_w: Float64,
) raises:
    # Render ONE line so its left edge is at the current text position and the
    # words are spread to fill exactly box_w points. Uses the TJ array operator.
    #
    # Single-word lines (no inter-word gaps to stretch) fall back to show_text.
    var words = _split_on_spaces(line)

    if len(words) <= 1:
        show_text(b, line)
        return

    # Natural rendered width = sum of word widths + (N-1) space advances, which
    # is exactly text_width of the joined line (spaces included).
    var natural = text_width(font_name, size, line)
    var slack = box_w - natural
    var gaps = len(words) - 1

    # Per-gap extra space in points (may be negative if the line is wider than
    # the box, e.g. an over-long single word that wrap_text could not split —
    # then this compresses, which is the standard fallback).
    var per_gap = slack / Float64(gaps)

    # TJ adjustment in thousandths of an em. To ADD `g` points of gap we emit
    # adj = -g * 1000 / size  (PDF subtracts adj*size/1000 from the position).
    var adj = -per_gap * 1000.0 / size

    b.append(UInt8(ord("[")))
    b.append(UInt8(ord(" ")))
    for i in range(len(words)):
        _put_word_literal(b, words[i])
        b.append(UInt8(ord(" ")))
        if i < len(words) - 1:
            # Emit a literal space glyph inside the same string would also work,
            # but the natural width already accounts for the space advance; so we
            # keep an explicit space character between words by appending it to
            # the rendered text. We render each word as its own (...) and add the
            # space as a separate (space) so its advance is preserved, then the
            # numeric adjustment adds the slack.
            _put_word_literal(b, String(" "))
            b.append(UInt8(ord(" ")))
            _put_real(b, adj)
            b.append(UInt8(ord(" ")))
    _put_str(b, "] TJ\n")


def render_justified_paragraph(
    mut b: List[UInt8],
    font_name: String,
    size: Float64,
    text: String,
    box_x: Float64,
    box_y: Float64,
    box_w: Float64,
    leading: Float64,
) raises:
    # Wrap the paragraph, then emit a full BT..ET. Every line EXCEPT the last is
    # full-justified; the last line is shown left-aligned. The first Td places
    # the text at (box_x, box_y); subsequent lines step down by `leading`.
    var lines = wrap_text(text, font_name, size, box_w)

    begin_text(b)
    set_font(b, "F1", size)

    if len(lines) == 0:
        end_text(b)
        return

    # Absolute placement for the first line.
    text_pos(b, box_x, box_y)

    for i in range(len(lines)):
        if i > 0:
            # Step to next line (relative move from the current line origin).
            text_pos(b, 0.0, -leading)
        if i < len(lines) - 1:
            show_justified_line(b, font_name, size, lines[i], box_w)
        else:
            show_text(b, lines[i])

    end_text(b)
