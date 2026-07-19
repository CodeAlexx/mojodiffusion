# pdf/tests/adversarial.mojo
# Adversarial stress test for the pure-Mojo PDF writer.
# Produces /tmp/pdf_adv.pdf exercising edge cases that commonly break writers:
#   Page 0 : tricky text chars (backslash, parens, leading ')', non-ASCII)
#   Page 1 : EMPTY content + EMPTY resources
#   Pages 2..31 : 30 extra pages each with one identifying line  (xref/Count stress)
#   Page 32: large / extreme real coordinates
#   Page 33: a SECOND image of a different size (32x80)
# Output: /tmp/pdf_adv.pdf

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


def tricky_text_content() raises -> List[UInt8]:
    var b = List[UInt8]()
    begin_text(b)
    set_font(b, "F1", 14.0)
    text_pos(b, 72.0, 700.0)
    # backslash and both parens, literal ')' at start
    show_text(b, ")leading close paren and backslash \\ here")
    text_pos(b, 0.0, -20.0)
    show_text(b, "open ( and close ) and pair ()")
    text_pos(b, 0.0, -20.0)
    # non-ASCII: e-acute, em dash, percent
    show_text(b, "café — 50%% off")
    end_text(b)
    return b.copy()


def simple_line(line: String) raises -> List[UInt8]:
    var b = List[UInt8]()
    begin_text(b)
    set_font(b, "F1", 12.0)
    text_pos(b, 72.0, 700.0)
    show_text(b, line)
    end_text(b)
    return b.copy()


def big_reals_content() raises -> List[UInt8]:
    var c = Content()
    c.set_fill_rgb(0.2, 0.2, 0.2)
    # extreme reals exercising append_real / _push_real
    c.rect(12345.6789, -9999.0, 100.0, 50.0)
    c.fill()
    c.move_to(-9999.0, 12345.6789)
    c.line_to(12345.6789, -9999.0)
    c.stroke()
    return c.bytes()


def build_raster(w: Int, h: Int) raises -> List[UInt8]:
    var rgb = List[UInt8]()
    for y in range(h):
        for x in range(w):
            rgb.append(UInt8((x * 255) // (w - 1)))
            rgb.append(UInt8((y * 255) // (h - 1)))
            rgb.append(UInt8(128))
    return rgb.copy()


def main() raises:
    var doc = PdfDoc()

    # Shared font object first.
    var font_body = standard_font_object("Helvetica")
    var font_num = doc.add_object(font_body)

    var font_res = List[UInt8]()
    append_str(font_res, "<< /Font << /F1 ")
    append_int(font_res, font_num)
    append_str(font_res, " 0 R >> >>")

    # ---- Page 0: tricky chars
    var p0 = tricky_text_content()
    _ = doc.add_page(612.0, 792.0, p0, font_res)

    # ---- Page 1: EMPTY content + EMPTY resources
    var empty_content = List[UInt8]()
    var empty_res = List[UInt8]()
    _ = doc.add_page(612.0, 792.0, empty_content, empty_res)

    # ---- Pages 2..31: 30 pages
    for i in range(30):
        var line = String("This is page index ")
        line += String(i)
        var pc = simple_line(line)
        _ = doc.add_page(612.0, 792.0, pc, font_res)

    # ---- Page 32: large reals
    var pbig = big_reals_content()
    var pbig_res = List[UInt8]()
    append_str(pbig_res, "<< >>")
    _ = doc.add_page(612.0, 792.0, pbig, pbig_res)

    # ---- Page 33: second image, different size 32x80
    var iw = 32
    var ih = 80
    var raster = build_raster(iw, ih)
    var img_body = image_xobject_stream(iw, ih, raster)
    var img_num = doc.add_object(img_body)

    var pimg = List[UInt8]()
    draw_image(pimg, "Im1", 100.0, 200.0, 128.0, 320.0)
    var img_res = List[UInt8]()
    append_str(img_res, "<< /XObject << /Im1 ")
    append_int(img_res, img_num)
    append_str(img_res, " 0 R >> >>")
    _ = doc.add_page(612.0, 792.0, pimg, img_res)

    doc.save("/tmp/pdf_adv.pdf")
    print("wrote /tmp/pdf_adv.pdf")
