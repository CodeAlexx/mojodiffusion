# pdf/tests/compressed_test.mojo
# Exercises the ADDED enterprise features on PdfDoc:
#   1) save_compressed()  -> PDF 1.5 /ObjStm + cross-reference stream.
#   2) enable_xmp()       -> archival XMP /Metadata stream + document /ID.
#
# Builds one document with several pages + a standard font + an embedded RGB
# image + a document outline, saves it BOTH ways (plain save() and
# save_compressed()), and a third time with XMP enabled, then prints sizes so
# the externally-run pdfinfo / pdftotext / pypdf / exiftool checks can verify.
#
# Run from /home/alex/MOJO-libs:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . pdf/tests/compressed_test.mojo

from pdf.document import PdfDoc
from pdf.objects import append_str
from pdf.text import standard_font_object
from pdf.image import image_xobject_stream, draw_image
from pdf.extras import outline_item_object, outlines_root_object


def build_doc(mut doc: PdfDoc) raises:
    doc.set_info(
        String("Compressed & XMP Demo"),
        String("Mojo Author"),
        String("ObjStm + XMP test"),
        String("MojoPDF Creator"),
    )

    # Standard font object (Helvetica).
    var font_obj = standard_font_object(String("Helvetica"))
    var font_num = doc.add_object(font_obj)

    # Embedded RGB image: a small 8x8 checkerboard.
    var W = 8
    var H = 8
    var rgb = List[UInt8]()
    for y in range(H):
        for x in range(W):
            if ((x + y) % 2) == 0:
                rgb.append(UInt8(220)); rgb.append(UInt8(40)); rgb.append(UInt8(40))
            else:
                rgb.append(UInt8(40)); rgb.append(UInt8(40)); rgb.append(UInt8(220))
    var img_body = image_xobject_stream(W, H, rgb)
    var img_num = doc.add_object(img_body)

    # Resources shared by all pages: font /F1 + image /Im1.
    var resources = List[UInt8]()
    append_str(resources, "<< /Font << /F1 ")
    append_str(resources, String(font_num))
    append_str(resources, " 0 R >> /XObject << /Im1 ")
    append_str(resources, String(img_num))
    append_str(resources, " 0 R >> >>")

    # Three pages, each with text and the image.
    for pidx in range(1, 4):
        var c = List[UInt8]()
        append_str(c, "BT /F1 24 Tf 72 700 Td (Compressed Page ")
        append_str(c, String(pidx))
        append_str(c, ") Tj ET\n")
        draw_image(c, String("Im1"), 72.0, 500.0, 120.0, 120.0)
        var _p = doc.add_page(612.0, 792.0, c, resources)

    # A 2-item outline (bookmarks) pointing at the first two pages.
    # Object numbers are 1-based in append order; pre-compute them so the
    # items can reference their parent (the outline root) and each other.
    var first_num = len(doc.objects) + 1
    var second_num = first_num + 1
    var root_num = second_num + 1
    # outline_item_object(title, page_obj, top_y, parent_obj, prev_obj, next_obj)
    var first_item = outline_item_object(
        String("First Page"), doc.page_numbers[0], 720.0, root_num, 0, second_num
    )
    var _f = doc.add_object(first_item)
    var second_item = outline_item_object(
        String("Second Page"), doc.page_numbers[1], 720.0, root_num, first_num, 0
    )
    var _s = doc.add_object(second_item)
    var root = outlines_root_object(first_num, second_num, 2)
    var _r = doc.add_object(root)
    doc.set_outlines(root_num)


def file_size(path: String) raises -> Int:
    var sz = 0
    with open(path, "r") as f:
        var data = f.read_bytes()
        sz = len(data)
    return sz


def main() raises:
    # ---- Build + save plain and compressed (no XMP) ----
    var doc = PdfDoc()
    build_doc(doc)
    doc.save("/tmp/pdf_plain.pdf")
    print("wrote /tmp/pdf_plain.pdf")

    var doc2 = PdfDoc()
    build_doc(doc2)
    doc2.save_compressed("/tmp/pdf_compressed.pdf")
    print("wrote /tmp/pdf_compressed.pdf")

    var plain_sz = file_size("/tmp/pdf_plain.pdf")
    var comp_sz = file_size("/tmp/pdf_compressed.pdf")
    print("plain size     =", plain_sz)
    print("compressed size=", comp_sz)
    if comp_sz < plain_sz:
        print("OK: compressed is smaller")
    else:
        print("NOTE: compressed NOT smaller (tiny doc / overhead dominates)")

    # ---- Build + save with XMP enabled (compressed + classic) ----
    var doc3 = PdfDoc()
    build_doc(doc3)
    doc3.enable_xmp()
    doc3.save_compressed("/tmp/pdf_compressed_xmp.pdf")
    print("wrote /tmp/pdf_compressed_xmp.pdf (XMP + /ID)")

    var doc4 = PdfDoc()
    build_doc(doc4)
    doc4.enable_xmp()
    doc4.save_with_xmp("/tmp/pdf_xmp.pdf")
    print("wrote /tmp/pdf_xmp.pdf (classic xref + XMP + /ID)")

    print("DONE")
