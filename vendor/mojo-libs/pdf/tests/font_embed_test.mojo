# pdf/tests/font_embed_test.mojo
# Build a PDF with an EMBEDDED TrueType (DejaVuSans) Type0/Identity-H font and
# real Unicode text. Verified externally with pdffonts (emb yes) + pdftotext.

from pdf.document import PdfDoc
from pdf.objects import append_str
from pdf.ttf import load_ttf
from pdf.font_embed import embed_font, show_text_unicode, collect_codepoints
from pdf.text import begin_text, end_text, set_font, text_pos


def main() raises:
    var font = load_ttf("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

    var text = String("Helló Ünïcödé Mojo 2026 © ®")

    var doc = PdfDoc()
    var cps = collect_codepoints(text)
    var emb = embed_font(doc, font, "DejaVuSans", cps)

    # Page resources reference the embedded Type0 font as /F1.
    var res = List[UInt8]()
    append_str(res, "<< /Font << /F1 ")
    append_str(res, String(emb.obj_num))
    append_str(res, " 0 R >> >>")

    # Content stream: place and show the Unicode text.
    var content = List[UInt8]()
    begin_text(content)
    set_font(content, "F1", 24.0)
    text_pos(content, 72.0, 700.0)
    show_text_unicode(content, font, emb, text)
    end_text(content)

    _ = doc.add_page(612.0, 792.0, content, res)
    doc.save("/tmp/font_embed.pdf")
    print("wrote /tmp/font_embed.pdf")
