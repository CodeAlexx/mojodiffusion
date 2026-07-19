# pdf/extras.mojo
# Production-grade PDF extras: richer color (gray, CMYK), graphics state
# (dash, line cap/join), transparency (ExtGState + alpha), link annotations,
# and document outline (bookmarks).
#
# Two flavors of API, matching the rest of the library:
#   * CONTENT ops  -> append PDF content-stream operators into a mut List[UInt8].
#                     (color values are 0.0..1.0; operator lines end with '\n')
#   * OBJECT builders -> RETURN a List[UInt8] holding an indirect-object BODY
#                     (the integrator wraps it with "N 0 obj ... endobj").
#
# Coordinate system / units match content.mojo: origin bottom-left, points.

from pdf.objects import (
    append_str,
    append_int,
    append_real,
    pdf_literal_string,
)


# ---- COLOR ops (append into a content buffer) --------------------------------


def set_fill_gray(mut b: List[UInt8], g: Float64) raises:
    # Gray fill: "<g> g"
    append_real(b, g)
    append_str(b, " g\n")


def set_stroke_gray(mut b: List[UInt8], g: Float64) raises:
    # Gray stroke: "<g> G"
    append_real(b, g)
    append_str(b, " G\n")


def set_fill_cmyk(
    mut b: List[UInt8], c: Float64, m: Float64, y: Float64, k: Float64
) raises:
    # CMYK fill: "<c> <m> <y> <k> k"
    append_real(b, c)
    append_str(b, " ")
    append_real(b, m)
    append_str(b, " ")
    append_real(b, y)
    append_str(b, " ")
    append_real(b, k)
    append_str(b, " k\n")


def set_stroke_cmyk(
    mut b: List[UInt8], c: Float64, m: Float64, y: Float64, k: Float64
) raises:
    # CMYK stroke: "<c> <m> <y> <k> K"
    append_real(b, c)
    append_str(b, " ")
    append_real(b, m)
    append_str(b, " ")
    append_real(b, y)
    append_str(b, " ")
    append_real(b, k)
    append_str(b, " K\n")


# ---- GRAPHICS STATE ops ------------------------------------------------------


def set_dash(mut b: List[UInt8], on: Float64, off: Float64, phase: Float64) raises:
    # Dash pattern: "[<on> <off>] <phase> d"
    # on <= 0 means "solid" -> empty pattern array "[] 0 d".
    if on <= 0.0:
        append_str(b, "[] 0 d\n")
        return
    append_str(b, "[")
    append_real(b, on)
    append_str(b, " ")
    append_real(b, off)
    append_str(b, "] ")
    append_real(b, phase)
    append_str(b, " d\n")


def set_line_cap(mut b: List[UInt8], cap: Int) raises:
    # Line cap: "<cap> J"  (0 butt, 1 round, 2 square)
    append_int(b, cap)
    append_str(b, " J\n")


def set_line_join(mut b: List[UInt8], join: Int) raises:
    # Line join: "<join> j"  (0 miter, 1 round, 2 bevel)
    append_int(b, join)
    append_str(b, " j\n")


# ---- TRANSPARENCY (ExtGState) ------------------------------------------------


def extgstate_object(fill_alpha: Float64, stroke_alpha: Float64) raises -> List[UInt8]:
    # Returns an ExtGState object body:
    #   << /Type /ExtGState /ca <fill_alpha> /CA <stroke_alpha> >>
    var b = List[UInt8]()
    append_str(b, "<< /Type /ExtGState /ca ")
    append_real(b, fill_alpha)
    append_str(b, " /CA ")
    append_real(b, stroke_alpha)
    append_str(b, " >>")
    return b^


def apply_gstate(mut b: List[UInt8], res_name: String) raises:
    # Apply a named ExtGState from /Resources /ExtGState: "/GS1 gs"
    append_str(b, "/")
    append_str(b, res_name)
    append_str(b, " gs\n")


# ---- LINK ANNOTATION ---------------------------------------------------------


def link_uri_annotation(
    x0: Float64, y0: Float64, x1: Float64, y1: Float64, url: String
) raises -> List[UInt8]:
    # Returns a Link annotation object body that opens a URI when clicked.
    #   << /Type /Annot /Subtype /Link /Rect [x0 y0 x1 y1]
    #      /Border [0 0 0] /A << /S /URI /URI (url) >> >>
    var b = List[UInt8]()
    append_str(b, "<< /Type /Annot /Subtype /Link /Rect [")
    append_real(b, x0)
    append_str(b, " ")
    append_real(b, y0)
    append_str(b, " ")
    append_real(b, x1)
    append_str(b, " ")
    append_real(b, y1)
    append_str(b, "] /Border [0 0 0] /A << /S /URI /URI ")
    pdf_literal_string(b, url)
    append_str(b, " >> >>")
    return b^


# ---- DOCUMENT OUTLINE (bookmarks) --------------------------------------------


def outline_item_object(
    title: String,
    page_obj: Int,
    top_y: Float64,
    parent_obj: Int,
    prev_obj: Int,
    next_obj: Int,
) raises -> List[UInt8]:
    # Returns an outline item object body. Prev/Next are omitted when 0.
    #   << /Title (title) /Parent P 0 R
    #      [/Prev R 0 R] [/Next R 0 R]
    #      /Dest [ <page_obj> 0 R /XYZ 0 top_y 0 ] >>
    var b = List[UInt8]()
    append_str(b, "<< /Title ")
    pdf_literal_string(b, title)
    append_str(b, " /Parent ")
    append_int(b, parent_obj)
    append_str(b, " 0 R")
    if prev_obj != 0:
        append_str(b, " /Prev ")
        append_int(b, prev_obj)
        append_str(b, " 0 R")
    if next_obj != 0:
        append_str(b, " /Next ")
        append_int(b, next_obj)
        append_str(b, " 0 R")
    append_str(b, " /Dest [ ")
    append_int(b, page_obj)
    append_str(b, " 0 R /XYZ 0 ")
    append_real(b, top_y)
    append_str(b, " 0 ] >>")
    return b^


def outlines_root_object(first_obj: Int, last_obj: Int, count: Int) raises -> List[UInt8]:
    # Returns the outline root (catalog /Outlines target) body.
    #   << /Type /Outlines /First F 0 R /Last L 0 R /Count count >>
    var b = List[UInt8]()
    append_str(b, "<< /Type /Outlines /First ")
    append_int(b, first_obj)
    append_str(b, " 0 R /Last ")
    append_int(b, last_obj)
    append_str(b, " 0 R /Count ")
    append_int(b, count)
    append_str(b, " >>")
    return b^
