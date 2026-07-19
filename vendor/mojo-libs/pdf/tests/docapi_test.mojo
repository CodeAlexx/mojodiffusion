# pdf/tests/docapi_test.mojo
# Exercises the production-grade document API:
#   - PdfDoc.set_info (emits /Info dict + trailer /Info ref)
#   - the Page facade (new_page, add_font, draw text into page.buf, finish flate)
#   - FlateDecode page content stream (must still be pdftotext-extractable)
#
# Verify after running:
#   pdfinfo  /tmp/docapi.pdf   -> Title/Author/Producer present
#   pdftotext /tmp/docapi.pdf  -> text comes back (proves Flate decodes)

from pdf.document import PdfDoc
from pdf.page import Page, new_page
from pdf.text import (
    begin_text,
    end_text,
    set_font,
    text_pos,
    show_text,
    standard_font_object,
)


def main() raises:
    var doc = PdfDoc()

    # Document metadata.
    doc.set_info(
        String("Production PDF API"),
        String("Builder-C"),
        String("Facade + metadata + flate"),
        String("MOJO-libs pdf"),
    )

    # Register a standard font object up front so we know its number.
    var font_body = standard_font_object("Helvetica")
    var font_num = doc.add_object(font_body)

    # Build a page via the facade.
    var page = new_page(612.0, 792.0)
    var f1 = page.add_font(font_num)  # -> "F1"

    begin_text(page.buf)
    set_font(page.buf, f1, 24.0)
    text_pos(page.buf, 72.0, 700.0)
    show_text(page.buf, "Hello from the Page facade!")
    set_font(page.buf, f1, 12.0)
    text_pos(page.buf, 0.0, -30.0)
    show_text(page.buf, "This page stream is FlateDecode compressed.")
    end_text(page.buf)

    # FlateDecode the content stream.
    _ = page.finish(doc, True)

    doc.save("/tmp/docapi.pdf")
    print("wrote /tmp/docapi.pdf")
