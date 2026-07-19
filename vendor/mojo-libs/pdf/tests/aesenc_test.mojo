# pdf/tests/aesenc_test.mojo
# Builds two 1-page modern-AES encrypted PDFs and writes them:
#   /tmp/pdf_aes128.pdf  (AESV2, V=4 R=4, 128-bit)
#   /tmp/pdf_aes256.pdf  (AESV3, V=5 R=6, 256-bit)
# Verify EXTERNALLY (decisive) with pypdf decrypt+extract, pdfinfo, pdftotext.
#
# Run from /home/alex/MOJO-libs:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . pdf/tests/aesenc_test.mojo

from pdf.document import PdfDoc
from pdf.objects import append_str
from pdf.text import standard_font_object


def build(path: String, want256: Bool) raises:
    var doc = PdfDoc()
    doc.set_info(
        String("AES Doc Title"),
        String("Mojo"),
        String("AES encryption test"),
        String("MojoPDF"),
    )

    var font_obj = standard_font_object(String("Helvetica"))
    var font_num = doc.add_object(font_obj)

    var content = List[UInt8]()
    append_str(content, "BT /F1 24 Tf 72 700 Td (AES Encrypted Mojo 2026) Tj ET\n")

    var resources = List[UInt8]()
    append_str(resources, "<< /Font << /F1 ")
    append_str(resources, String(font_num))
    append_str(resources, " 0 R >> >>")

    var _page = doc.add_page(612.0, 792.0, content, resources)

    if want256:
        doc.save_encrypted_aes256(path, "userpw", "ownerpw")
    else:
        doc.save_encrypted_aes128(path, "userpw", "ownerpw")
    print("wrote", path)


def main() raises:
    build(String("/tmp/pdf_aes128.pdf"), False)
    build(String("/tmp/pdf_aes256.pdf"), True)
