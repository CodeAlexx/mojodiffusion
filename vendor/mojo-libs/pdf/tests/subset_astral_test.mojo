# pdf/tests/subset_astral_test.mojo
# Prove the cmap FORMAT-12 path in pdf/subset.mojo: subset DejaVuSans with a
# mix of BMP codepoints AND at least one ASTRAL codepoint (> U+FFFF), embed into
# a PDF, and render the astral codepoint with show_text_unicode.
#
# DejaVuSans has no native glyph for U+1F600, so we exercise the format-12
# encoder + ToUnicode plumbing by GID REUSE: we map the astral codepoint onto an
# EXISTING glyph id (the gid of 'A'). This proves the encoder emits a valid
# (3,10) format-12 subtable and that ToUnicode round-trips the astral codepoint,
# independent of native coverage.
#
# Run (from /home/alex/MOJO-libs):
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . \
#       pdf/tests/subset_astral_test.mojo
#
# External verification (run after this test produces the artifacts):
#   - fontTools: a cmap subtable with format==12 exists and its (3,10) table
#     maps U+1F600 -> the reused gid.
#   - pdftotext / pypdf: the astral codepoint round-trips via ToUnicode.

from pdf.document import PdfDoc
from pdf.objects import append_str
from pdf.ttf import load_ttf, ttf_from_bytes
from pdf.subset import subset_ttf_mapped
from pdf.font_embed import embed_font, show_text_unicode
from pdf.text import begin_text, end_text, set_font, text_pos


def _write_bytes(path: String, data: List[UInt8]) raises:
    var f = open(path, "w")
    f.write_bytes(Span(data))
    f.close()


def main() raises:
    var path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    var font = load_ttf(path)

    # Astral codepoint under test (GRINNING FACE). Will be mapped to 'A''s gid.
    var astral = 0x1F600

    # A few BMP codepoints (these have native glyphs).
    var bmp_A = ord("A")
    var bmp_B = ord("B")
    var bmp_z = ord("z")

    var gid_A = font.gid_for(bmp_A)
    var gid_B = font.gid_for(bmp_B)
    var gid_z = font.gid_for(bmp_z)
    print("native gid_for('A')", gid_A)
    print("native gid_for('B')", gid_B)
    print("native gid_for('z')", gid_z)
    print("native gid_for(U+1F600)", font.gid_for(astral))  # expect 0 (no glyph)

    # Explicit (codepoint -> gid) mapping. The astral codepoint REUSES the gid of
    # glyph 'z' (a real, existing glyph). We deliberately do NOT also map the BMP
    # codepoint 'z' into the embedded set, because embed_font de-duplicates by
    # gid: if two codepoints shared one gid, the second would be dropped from the
    # /W + ToUnicode tables. So 'z''s gid is dedicated to the astral codepoint
    # here — proving the astral cp gets its own ToUnicode entry via gid reuse.
    var cps = List[Int]()
    var gids = List[Int]()
    cps.append(bmp_A)
    gids.append(gid_A)
    cps.append(bmp_B)
    gids.append(gid_B)
    cps.append(astral)
    gids.append(gid_z)  # gid-reuse: astral -> glyph 'z''s real glyph id

    # Subset with the explicit mapping -> exercises the format-12 cmap path.
    var sub = subset_ttf_mapped(font, cps, gids)
    _write_bytes("/tmp/subset_astral.ttf", sub)
    print("subset_astral_ttf_size", len(sub))
    print("astral_gid_expected", gid_z)

    # Wrap the subset bytes; gid_for on the SUBSET must resolve the astral
    # codepoint via the new format-12 subtable.
    var sub_font = ttf_from_bytes(sub^)
    print("subset gid_for(U+1F600)", sub_font.gid_for(astral))

    # Build the PDF. embed_font resolves each codepoint via the subset cmap, so
    # the astral codepoint gets /W width + ToUnicode mapping via gid reuse.
    var doc = PdfDoc()
    var emb = embed_font(doc, sub_font, "DejaVuAstral", cps)

    var res = List[UInt8]()
    append_str(res, "<< /Font << /F1 ")
    append_str(res, String(emb.obj_num))
    append_str(res, " 0 R >> >>")

    var content = List[UInt8]()
    begin_text(content)
    set_font(content, "F1", 24.0)
    text_pos(content, 72.0, 700.0)
    # Render a string containing the astral codepoint (and BMP neighbors).
    # Only embedded codepoints (A, B, U+1F600) appear.
    var text = String("AB")
    text += chr(astral)
    show_text_unicode(content, sub_font, emb, text)
    end_text(content)

    _ = doc.add_page(612.0, 792.0, content, res)
    doc.save("/tmp/pdf_astral.pdf")
    print("wrote /tmp/subset_astral.ttf")
    print("wrote /tmp/pdf_astral.pdf")
