# pdf/tests/subset_test.mojo
# Subset DejaVuSans to a small Unicode string, embed in a 1-page PDF, and dump
# the subset bytes to /tmp/subset.ttf for external verification with fontTools /
# poppler (pdffonts, pdftotext) / pypdf.
#
# Run (from /home/alex/MOJO-libs):
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . \
#       pdf/tests/subset_test.mojo

from pdf.document import PdfDoc
from pdf.objects import append_str
from pdf.ttf import load_ttf
from pdf.subset import subset_ttf, embed_subset_font
from pdf.font_embed import collect_codepoints, show_text_unicode
from pdf.text import begin_text, end_text, set_font, text_pos


def _write_bytes(path: String, data: List[UInt8]) raises:
    var f = open(path, "w")
    f.write_bytes(Span(data))
    f.close()


def main() raises:
    var path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    var font = load_ttf(path)

    var text = String("Helló Ünïcödé 2026 ©®")
    var cps = collect_codepoints(text)

    # Subset bytes for fontTools / size inspection.
    var sub = subset_ttf(font, cps)
    _write_bytes("/tmp/subset.ttf", sub)

    var full = font.raw()
    print("full_ttf_size", len(full))
    print("subset_ttf_size", len(sub))

    # Build the PDF using the subset font.
    var doc = PdfDoc()
    var emb = embed_subset_font(doc, font, "DejaVuSubset", cps)

    var res = List[UInt8]()
    append_str(res, "<< /Font << /F1 ")
    append_str(res, String(emb.obj_num))
    append_str(res, " 0 R >> >>")

    var content = List[UInt8]()
    begin_text(content)
    set_font(content, "F1", 24.0)
    text_pos(content, 72.0, 700.0)
    show_text_unicode(content, font, emb, text)
    end_text(content)

    _ = doc.add_page(612.0, 792.0, content, res)
    doc.save("/tmp/pdf_subset.pdf")
    print("wrote /tmp/pdf_subset.pdf")
    print("wrote /tmp/subset.ttf")
