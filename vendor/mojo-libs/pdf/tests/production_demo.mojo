# pdf/tests/production_demo.mojo
# COMPREHENSIVE production demo exercising every PDF feature together into a
# single multi-page document at /tmp/pdf_pro.pdf.
#
# Features exercised:
#   1. Document metadata (set_info: title/author/subject/creator).
#   2. Embedded Unicode TTF font (DejaVuSans) via load_ttf+embed_font, rendered
#      with show_text_unicode using non-CP1252 glyphs.
#   3. Wrapped paragraph via metrics+layout (wrap_text + align_x), standard-14.
#   4. FlateDecode-compressed page content stream (add_page_ext flate=True).
#   5. Color: gray + CMYK + RGB fills/strokes (extras + content).
#   6. Transparency: ExtGState (extgstate_object) + apply_gstate, q..Q overlap.
#   7. Link annotation (link_uri_annotation) wired into page /Annots.
#   8. Outline/bookmark tree: 2 items + root, set_outlines.
#   9. Embedded image XObject (image_xobject_stream + draw_image).
#
# Object-number bookkeeping: PdfDoc.save() appends Catalog/Pages/Info as the
# LAST objects, so every object we add via add_object/add_stream_object/add_page*
# gets a stable 1-based number returned at add time. Outline items cross-
# reference each other by object number, so we precompute those numbers from the
# current object count before building their bodies.

from pdf.document import PdfDoc
from pdf.objects import append_str, append_int, append_real
from pdf.text import (
    begin_text,
    end_text,
    set_font,
    text_pos,
    show_text,
    standard_font_object,
)
from pdf.ttf import load_ttf
from pdf.font_embed import embed_font, show_text_unicode, add_codepoints
from pdf.metrics import text_width
from pdf.layout import wrap_text, align_x
from pdf.image import image_xobject_stream, draw_image
from pdf.content import Content
from pdf.extras import (
    set_fill_gray,
    set_stroke_gray,
    set_fill_cmyk,
    set_stroke_cmyk,
    set_dash,
    set_line_cap,
    set_line_join,
    extgstate_object,
    apply_gstate,
    link_uri_annotation,
    outline_item_object,
    outlines_root_object,
)


def _append_bytes(mut dst: List[UInt8], src: List[UInt8]):
    for i in range(len(src)):
        dst.append(src[i])


def _build_resources(
    std_font_num: Int,
    type0_font_num: Int,
    image_num: Int,
    extg_num: Int,
) raises -> List[UInt8]:
    # Build a /Resources dict referencing a standard font (/F1), an embedded
    # Type0 font (/F2), an image XObject (/Im1), and an ExtGState (/GS1).
    var r = List[UInt8]()
    append_str(r, "<< /Font << /F1 ")
    append_int(r, std_font_num)
    append_str(r, " 0 R /F2 ")
    append_int(r, type0_font_num)
    append_str(r, " 0 R >> /XObject << /Im1 ")
    append_int(r, image_num)
    append_str(r, " 0 R >> /ExtGState << /GS1 ")
    append_int(r, extg_num)
    append_str(r, " 0 R >> >>")
    return r^


def main() raises:
    var doc = PdfDoc()

    # ---- 1) Document metadata ----
    doc.set_info(
        String("MOJO-libs Production PDF"),
        String("Integrator-Agent-5"),
        String("Every feature in one document"),
        String("MOJO-libs pure-Mojo pdf"),
    )

    # ---- Standard-14 font object (Helvetica) ----
    var std_font_num = doc.add_object(standard_font_object("Helvetica"))

    # ---- 2) Embedded Unicode TTF font ----
    # Collect EVERY codepoint rendered with the embedded font across the WHOLE
    # document (both pages) BEFORE embedding, so ToUnicode + /W cover all glyphs.
    var dejavu = load_ttf(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    var unicode_line = String("Helló Ünïcödé — Mojo 2026 © ®")
    var unicode_line2 = String("Ünïcödé on page two — ©®")
    var cps = List[Int]()
    add_codepoints(cps, unicode_line)
    add_codepoints(cps, unicode_line2)
    var emb = embed_font(doc, dejavu, "DejaVuSans", cps)
    var type0_font_num = emb.obj_num

    # ---- 9) Embedded image XObject (a 32x32 RGB gradient) ----
    var iw = 32
    var ih = 32
    var rgb = List[UInt8]()
    for y in range(ih):
        for x in range(iw):
            rgb.append(UInt8((x * 255) // (iw - 1)))      # R ramps across
            rgb.append(UInt8((y * 255) // (ih - 1)))      # G ramps down
            rgb.append(UInt8(128))                          # B constant
    var image_num = doc.add_object(image_xobject_stream(iw, ih, rgb))

    # ---- 6) ExtGState for transparency ----
    var extg_num = doc.add_object(extgstate_object(0.45, 0.8))

    # ============ PAGE 1: text + unicode + wrapped paragraph + image ============
    var c1 = Content()

    # Title in standard-14 Helvetica.
    begin_text(c1.b)
    set_font(c1.b, "F1", 22.0)
    text_pos(c1.b, 72.0, 740.0)
    show_text(c1.b, "MOJO-libs PDF — Production Demo")
    end_text(c1.b)

    # 2) Unicode line in the embedded Type0 font (/F2).
    begin_text(c1.b)
    set_font(c1.b, "F2", 18.0)
    text_pos(c1.b, 72.0, 705.0)
    show_text_unicode(c1.b, dejavu, emb, unicode_line)
    end_text(c1.b)

    # 3) Wrapped + aligned paragraph using metrics + layout (standard-14).
    var para = String(
        "This paragraph is wrapped using pdf.layout.wrap_text driven by the"
        " standard-14 AFM metrics in pdf.metrics, then each line is horizontally"
        " positioned with align_x so that the body reads as a justified block of"
        " text inside a fixed measure. Word wrapping is greedy on spaces and each"
        " produced line fits within the box width measured in PDF points."
    )
    var box_x = 72.0
    var box_w = 468.0
    var fsize = 11.0
    var lines = wrap_text(para, "Helvetica", fsize, box_w)
    var ty = 670.0
    var leading = 15.0
    # Each line is left-aligned at box_x (align_x mode 0). Td is relative, so we
    # move to the first line absolutely, then step down one leading per line —
    # AND show_text every line (not just the first).
    begin_text(c1.b)
    set_font(c1.b, "F1", fsize)
    var line0_x = align_x(lines[0], "Helvetica", fsize, box_x, box_w, 0)
    text_pos(c1.b, line0_x, ty)
    show_text(c1.b, lines[0])
    for i in range(1, len(lines)):
        # All lines share the same left x, so the per-line x delta is 0.
        text_pos(c1.b, 0.0, -leading)
        show_text(c1.b, lines[i])
    end_text(c1.b)

    # 9) Draw the embedded image.
    draw_image(c1.b, "Im1", 72.0, 560.0, 96.0, 96.0)

    # 5) Color sampler: gray, CMYK, RGB rectangles.
    # Gray filled square.
    c1.save_state()
    set_fill_gray(c1.b, 0.6)
    c1.rect(200.0, 560.0, 60.0, 60.0)
    c1.fill()
    c1.restore_state()
    # CMYK filled square (cyan).
    c1.save_state()
    set_fill_cmyk(c1.b, 1.0, 0.0, 0.0, 0.0)
    c1.rect(280.0, 560.0, 60.0, 60.0)
    c1.fill()
    # CMYK stroke around it (magenta), dashed, round cap/join.
    set_stroke_cmyk(c1.b, 0.0, 1.0, 0.0, 0.0)
    set_dash(c1.b, 4.0, 3.0, 0.0)
    set_line_cap(c1.b, 1)
    set_line_join(c1.b, 1)
    c1.set_line_width(3.0)
    c1.rect(280.0, 560.0, 60.0, 60.0)
    c1.stroke()
    c1.restore_state()
    # RGB filled rectangle.
    c1.save_state()
    c1.set_fill_rgb(0.1, 0.5, 0.9)
    c1.rect(360.0, 560.0, 60.0, 60.0)
    c1.fill()
    c1.restore_state()

    # 6) Transparency: overlapping semi-transparent shapes inside q..Q + gs.
    c1.save_state()
    apply_gstate(c1.b, "GS1")
    c1.set_fill_rgb(1.0, 0.0, 0.0)
    c1.rect(200.0, 460.0, 90.0, 70.0)
    c1.fill()
    c1.set_fill_rgb(0.0, 0.0, 1.0)
    c1.rect(245.0, 460.0, 90.0, 70.0)
    c1.fill()
    c1.restore_state()

    # A clickable link area (visible label).
    begin_text(c1.b)
    set_font(c1.b, "F1", 12.0)
    text_pos(c1.b, 72.0, 430.0)
    show_text(c1.b, "Visit the MOJO-libs project ->")
    end_text(c1.b)

    # 7) Link annotation object + wire into page /Annots.
    var link_num = doc.add_object(
        link_uri_annotation(
            72.0, 425.0, 260.0, 445.0, "https://github.com/CodeAlexx/MOJO-libs"
        )
    )

    var res1 = _build_resources(std_font_num, type0_font_num, image_num, extg_num)
    var annots1 = List[Int]()
    annots1.append(link_num)
    # 4) FlateDecode-compressed page content stream.
    var page1_num = doc.add_page_ext(
        612.0, 792.0, c1.bytes(), res1, annots1, True
    )

    # ============ PAGE 2: second bookmark target ============
    var c2 = Content()
    begin_text(c2.b)
    set_font(c2.b, "F1", 20.0)
    text_pos(c2.b, 72.0, 740.0)
    show_text(c2.b, "Page Two: Bookmark Target")
    end_text(c2.b)
    begin_text(c2.b)
    set_font(c2.b, "F2", 14.0)
    text_pos(c2.b, 72.0, 710.0)
    show_text_unicode(c2.b, dejavu, emb, unicode_line2)
    end_text(c2.b)
    # A standard-14 line so pdftotext has plain content here too.
    begin_text(c2.b)
    set_font(c2.b, "F1", 12.0)
    text_pos(c2.b, 72.0, 680.0)
    show_text(c2.b, "This is the second page reached by the second bookmark.")
    end_text(c2.b)

    var res2 = _build_resources(std_font_num, type0_font_num, image_num, extg_num)
    var no_annots = List[Int]()
    var page2_num = doc.add_page_ext(
        612.0, 792.0, c2.bytes(), res2, no_annots, True
    )

    # ---- 8) Outline / bookmark tree ----
    # Precompute object numbers: items and root are added next, sequentially.
    var next_num = len(doc.objects) + 1
    var item1_num = next_num
    var item2_num = next_num + 1
    var root_num = next_num + 2

    # Item 1 -> page 1, no prev, next=item2, parent=root.
    var item1 = outline_item_object(
        String("Cover & Feature Showcase"),
        page1_num,
        760.0,
        root_num,
        0,
        item2_num,
    )
    var added_item1 = doc.add_object(item1)

    # Item 2 -> page 2, prev=item1, no next, parent=root.
    var item2 = outline_item_object(
        String("Page Two"),
        page2_num,
        760.0,
        root_num,
        item1_num,
        0,
    )
    var added_item2 = doc.add_object(item2)

    var root = outlines_root_object(item1_num, item2_num, 2)
    var added_root = doc.add_object(root)

    # Sanity: the precomputed numbers must equal the actually-assigned numbers.
    if added_item1 != item1_num or added_item2 != item2_num or added_root != root_num:
        raise Error("outline object-number precomputation mismatch")

    doc.set_outlines(root_num)

    doc.save("/tmp/pdf_pro.pdf")
    print("wrote /tmp/pdf_pro.pdf")
    print("pages:", len(doc.page_numbers))
    print("page1_obj:", page1_num, " page2_obj:", page2_num)
    print("type0_font_obj:", type0_font_num, " image_obj:", image_num)
    print("outline root:", root_num, " items:", item1_num, item2_num)
