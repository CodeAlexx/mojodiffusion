# pdf/tests/doc_test.mojo
# Smoke test: build a one-page PDF and write it to /tmp/doc_test.pdf.
# Verify with: pdfinfo /tmp/doc_test.pdf  (must report Pages: 1, 612 x 792).

from pdf.document import PdfDoc
from pdf.objects import append_str


def main() raises:
    var doc = PdfDoc()

    # Tiny content stream: a filled red rectangle.
    var content = List[UInt8]()
    append_str(content, "1 0 0 rg 100 100 200 150 re f\n")

    # Empty resources dict.
    var resources = List[UInt8]()
    append_str(resources, "<< >>")

    var _page = doc.add_page(612.0, 792.0, content, resources)

    doc.save("/tmp/doc_test.pdf")
    print("wrote /tmp/doc_test.pdf")
