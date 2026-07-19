# pdf/tests/pdfa_test.mojo
# Build a 1-page PDF/A-2b document: set_info metadata + an EMBEDDED TrueType
# (DejaVuSans) Type0/Identity-H font rendering real text, an embedded sRGB ICC
# OutputIntent, an XMP /Metadata stream with the pdfaid schema, and a /ID.
# Saved via PdfDoc.save_pdfa("/tmp/pdf_pdfa.pdf").
#
# veraPDF is NOT installed in this environment, so full PDF/A conformance is
# NOT machine-validated here. The Python/poppler driver checks every structural
# marker that CAN be verified (OutputIntents/GTS_PDFA1, DestOutputProfile /N 3,
# pdfaid:part 2 / conformance B, /ID, embedded fonts only).

from pdf.document import PdfDoc
from pdf.objects import append_str
from pdf.ttf import load_ttf
from pdf.font_embed import embed_font, show_text_unicode, collect_codepoints
from pdf.text import begin_text, end_text, set_font, text_pos


def main() raises:
    var font = load_ttf("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

    var text = String("PDF/A-2b Mojo 2026 conformance test")

    var doc = PdfDoc()
    doc.set_info(
        String("Mojo PDF/A-2b Sample"),
        String("MojoPDF"),
        String("PDF/A-2b structural conformance"),
        String("MojoPDF test suite"),
    )

    var cps = collect_codepoints(text)
    var emb = embed_font(doc, font, "DejaVuSans", cps)

    # Page resources reference the embedded Type0 font as /F1.
    var res = List[UInt8]()
    append_str(res, "<< /Font << /F1 ")
    append_str(res, String(emb.obj_num))
    append_str(res, " 0 R >> >>")

    # Content stream: place and show the text in DeviceRGB black.
    var content = List[UInt8]()
    append_str(content, "0 0 0 rg\n")
    begin_text(content)
    set_font(content, "F1", 18.0)
    text_pos(content, 72.0, 700.0)
    show_text_unicode(content, font, emb, text)
    end_text(content)

    _ = doc.add_page(612.0, 792.0, content, res)
    doc.save_pdfa("/tmp/pdf_pdfa.pdf")
    print("wrote /tmp/pdf_pdfa.pdf")
