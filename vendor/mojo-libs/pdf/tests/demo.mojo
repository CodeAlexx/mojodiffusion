# pdf/tests/demo.mojo
# Integration showcase: builds a 3-page PDF exercising every feature.
#   Page 1: vector graphics (filled rects + stroked paths)
#   Page 2: text (standard font, escaped parentheses)
#   Page 3: embedded RGB image
# Output: /tmp/pdf_showcase.pdf

from pdf.objects import append_str, append_int, append_real, pdf_name
from pdf.document import PdfDoc
from pdf.content import Content
from pdf.text import (
    begin_text,
    end_text,
    set_font,
    text_pos,
    show_text,
    standard_font_object,
)
from pdf.image import image_xobject_stream, draw_image


def build_page1_content() raises -> List[UInt8]:
    var c = Content()
    # Filled rectangle (red).
    c.set_fill_rgb(0.85, 0.1, 0.1)
    c.rect(72.0, 600.0, 200.0, 120.0)
    c.fill()
    # Filled rectangle (blue).
    c.set_fill_rgb(0.1, 0.2, 0.8)
    c.rect(320.0, 600.0, 200.0, 120.0)
    c.fill()
    # Stroked path (green diagonal lines).
    c.set_stroke_rgb(0.0, 0.6, 0.0)
    c.set_line_width(3.0)
    c.move_to(72.0, 500.0)
    c.line_to(540.0, 560.0)
    c.stroke()
    c.move_to(72.0, 540.0)
    c.line_to(540.0, 460.0)
    c.stroke()
    # Stroked closed triangle.
    c.set_stroke_rgb(0.5, 0.0, 0.5)
    c.set_line_width(2.0)
    c.move_to(200.0, 200.0)
    c.line_to(400.0, 200.0)
    c.line_to(300.0, 380.0)
    c.close_path()
    c.stroke()
    return c.bytes()


def build_page2_content() raises -> List[UInt8]:
    var b = List[UInt8]()
    begin_text(b)
    set_font(b, "F1", 24.0)
    text_pos(b, 72.0, 700.0)
    show_text(b, "Pure-Mojo PDF Writer")
    set_font(b, "F1", 14.0)
    text_pos(b, 0.0, -40.0)
    show_text(b, "Page 2: text rendering with the standard-14 fonts.")
    text_pos(b, 0.0, -24.0)
    show_text(b, "Hello (PDF) from Mojo!")
    text_pos(b, 0.0, -24.0)
    show_text(b, "Special chars: backslash \\ and (nested (parens)).")
    end_text(b)
    return b.copy()


def build_image_raster(w: Int, h: Int) raises -> List[UInt8]:
    # Build a w*h*3 RGB raster: horizontal red ramp, vertical green ramp,
    # constant blue. Produces a visible color gradient.
    var rgb = List[UInt8]()
    for y in range(h):
        for x in range(w):
            var r = (x * 255) // (w - 1)
            var g = (y * 255) // (h - 1)
            rgb.append(UInt8(r))
            rgb.append(UInt8(g))
            rgb.append(UInt8(64))
    return rgb.copy()


def main() raises:
    var doc = PdfDoc()

    # ---- Page 1: vector graphics --------------------------------------
    var page1 = build_page1_content()
    var res1 = List[UInt8]()
    append_str(res1, "<< >>")
    _ = doc.add_page(612.0, 792.0, page1, res1)

    # ---- Page 2: text -------------------------------------------------
    # Add the font object FIRST so we know its object number.
    var font_body = standard_font_object("Helvetica")
    var font_num = doc.add_object(font_body)

    var page2 = build_page2_content()
    var res2 = List[UInt8]()
    append_str(res2, "<< /Font << /F1 ")
    append_int(res2, font_num)
    append_str(res2, " 0 R >> >>")
    _ = doc.add_page(612.0, 792.0, page2, res2)

    # ---- Page 3: embedded image --------------------------------------
    var iw = 64
    var ih = 48
    var raster = build_image_raster(iw, ih)
    var img_body = image_xobject_stream(iw, ih, raster)
    var img_num = doc.add_object(img_body)

    var page3b = List[UInt8]()
    draw_image(page3b, "Im1", 178.0, 300.0, 256.0, 192.0)
    var res3 = List[UInt8]()
    append_str(res3, "<< /XObject << /Im1 ")
    append_int(res3, img_num)
    append_str(res3, " 0 R >> >>")
    _ = doc.add_page(612.0, 792.0, page3b, res3)

    doc.save("/tmp/pdf_showcase.pdf")
    print("wrote /tmp/pdf_showcase.pdf")
