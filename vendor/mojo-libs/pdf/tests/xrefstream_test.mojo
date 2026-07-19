# pdf/tests/xrefstream_test.mojo
# Builds a 2-page PDF saved with a PDF-1.5 cross-reference STREAM.
# Verify externally with pdfinfo / pdftotext / pypdf / qpdf --check.
#
# Run from /home/alex/MOJO-libs:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . pdf/tests/xrefstream_test.mojo

from pdf.document import PdfDoc
from pdf.objects import append_str
from pdf.text import standard_font_object


def main() raises:
    var doc = PdfDoc()
    doc.set_info(
        String("XRefStream Doc"),
        String("Mojo"),
        String("xref stream test"),
        String("MojoPDF"),
    )

    var font_obj = standard_font_object(String("Helvetica"))
    var font_num = doc.add_object(font_obj)

    var resources = List[UInt8]()
    append_str(resources, "<< /Font << /F1 ")
    append_str(resources, String(font_num))
    append_str(resources, " 0 R >> >>")

    # Page 1
    var c1 = List[UInt8]()
    append_str(c1, "BT /F1 24 Tf 72 700 Td (XRefStream Page One) Tj ET\n")
    var _p1 = doc.add_page(612.0, 792.0, c1, resources)

    # Page 2
    var c2 = List[UInt8]()
    append_str(c2, "BT /F1 24 Tf 72 700 Td (XRefStream Page Two) Tj ET\n")
    var _p2 = doc.add_page(612.0, 792.0, c2, resources)

    doc.save_xref_stream("/tmp/pdf_xrefstream.pdf")
    print("wrote /tmp/pdf_xrefstream.pdf")
