# pdf/tests/justify_test.mojo
# Builds a one-page PDF with a fully-justified paragraph and writes it to
# /tmp/pdf_justify.pdf for external verification (pdftotext / pypdf / TJ math).
#
# Run:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . \
#       pdf/tests/justify_test.mojo

from pdf.document import PdfDoc
from pdf.text import begin_text, end_text, set_font, text_pos, show_text
from pdf.objects import append_str, append_int
from pdf.justify import render_justified_paragraph, show_justified_line


def _build_resources(font_obj: Int) raises -> List[UInt8]:
    # "<< /Font << /F1 N 0 R >> >>"
    var r = List[UInt8]()
    append_str(r, "<< /Font << /F1 ")
    append_int(r, font_obj)
    append_str(r, " 0 R >> >>")
    return r.copy()


def _std_font_body(base: String) raises -> List[UInt8]:
    var b = List[UInt8]()
    append_str(b, "<< /Type /Font /Subtype /Type1 /BaseFont /")
    append_str(b, base)
    append_str(b, " /Encoding /WinAnsiEncoding >>")
    return b.copy()


def main() raises:
    var font_name = String("Helvetica")
    var size = 12.0
    var box_x = 72.0
    var box_y = 720.0
    var box_w = 400.0
    var leading = 16.0

    var para = String(
        "Mojo is a programming language that combines the usability of Python "
        "with the performance of systems languages, and this paragraph exists "
        "purely so that the text wrapper produces several lines which the "
        "justification routine can then stretch to fill the box exactly."
    )

    var content = List[UInt8]()
    render_justified_paragraph(
        content, font_name, size, para, box_x, box_y, box_w, leading
    )

    var doc = PdfDoc()
    var font_obj = doc.add_object(_std_font_body("Helvetica"))
    var res = _build_resources(font_obj)
    # Use UNCOMPRESSED content stream so the Python verifier can parse the TJ
    # arrays directly from the raw PDF bytes.
    _ = doc.add_page(612.0, 792.0, content, res)
    doc.save("/tmp/pdf_justify.pdf")

    print("WROTE /tmp/pdf_justify.pdf")
    print("box_x:", box_x, " box_w:", box_w, " size:", size)
