# pdf/tests/encrypt_test.mojo
# Builds a 1-page encrypted PDF (RC4 V=2 R=3 128-bit) and writes it.
# Verify externally with pypdf / pdftotext / qpdf.
#
# Run from /home/alex/MOJO-libs:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . pdf/tests/encrypt_test.mojo

from pdf.document import PdfDoc
from pdf.objects import append_str
from pdf.text import standard_font_object


def main() raises:
    var doc = PdfDoc()
    doc.set_info(
        String("Encrypted Doc Title"),
        String("Mojo"),
        String("Encryption test"),
        String("MojoPDF"),
    )

    # Font object (Helvetica), 1-based number.
    var font_obj = standard_font_object(String("Helvetica"))
    var font_num = doc.add_object(font_obj)

    # Content stream: visible text using the font resource /F1.
    var content = List[UInt8]()
    append_str(content, "BT /F1 24 Tf 72 700 Td (Encrypted Mojo PDF 2026) Tj ET\n")

    # Resources referencing the font object.
    var resources = List[UInt8]()
    append_str(resources, "<< /Font << /F1 ")
    append_str(resources, String(font_num))
    append_str(resources, " 0 R >> >>")

    var _page = doc.add_page(612.0, 792.0, content, resources)

    doc.save_encrypted("/tmp/pdf_enc.pdf", "userpw", "ownerpw")
    print("wrote /tmp/pdf_enc.pdf")
