# pdf/font_embed.mojo
# Embed a TrueType font as a PDF Type0 composite font (Identity-H encoding)
# with CIDFontType2 descendant, FontFile2 (FlateDecode), and a ToUnicode CMap
# so text is both rendered AND extractable / copyable.
#
# Identity-H + /CIDToGIDMap /Identity means CID == GID, and the 2-byte codes we
# emit in content streams ARE glyph IDs. show_text_unicode() therefore writes
# each glyph as 4 hex digits inside a <...> hex string.

from pdf.document import PdfDoc
from pdf.objects import append_str, append_int
from pdf.ttf import TtfFont
from graphics.deflate import deflate


def _zlib_wrap(raw: List[UInt8]) raises -> List[UInt8]:
    # Wrap raw DEFLATE in a zlib (RFC 1950) container for PDF /FlateDecode:
    # 0x78 0x9C header + raw deflate body + big-endian Adler-32 of UNcompressed.
    var body = deflate(raw)
    var z = List[UInt8]()
    z.append(UInt8(0x78))
    z.append(UInt8(0x9C))
    for i in range(len(body)):
        z.append(body[i])
    var a: Int = 1
    var b2: Int = 0
    var MOD: Int = 65521
    for i in range(len(raw)):
        a = (a + Int(raw[i])) % MOD
        b2 = (b2 + a) % MOD
    var adler = (b2 << 16) | a
    z.append(UInt8((adler >> 24) & 0xFF))
    z.append(UInt8((adler >> 16) & 0xFF))
    z.append(UInt8((adler >> 8) & 0xFF))
    z.append(UInt8(adler & 0xFF))
    return z.copy()


def _hex4(mut b: List[UInt8], v: Int) raises:
    # Append v as 4 uppercase hex digits (big-endian 16-bit).
    comptime HEX = "0123456789ABCDEF"
    var hexb = HEX.as_bytes()
    b.append(hexb[(v >> 12) & 0xF])
    b.append(hexb[(v >> 8) & 0xF])
    b.append(hexb[(v >> 4) & 0xF])
    b.append(hexb[v & 0xF])


def collect_codepoints(s: String) raises -> List[Int]:
    # Unique codepoints used in s, in first-seen order.
    var seen = List[Int]()
    for cp in s.codepoints():
        var c = Int(cp)
        var found = False
        for i in range(len(seen)):
            if seen[i] == c:
                found = True
                break
        if not found:
            seen.append(c)
    return seen.copy()


def add_codepoints(mut acc: List[Int], s: String) raises:
    # Append the codepoints of s into acc (deduplicated). Use this to gather
    # EVERY string you will render with an embedded font BEFORE calling
    # embed_font, so the ToUnicode CMap and /W widths cover all shown glyphs.
    for cp in s.codepoints():
        var c = Int(cp)
        var found = False
        for i in range(len(acc)):
            if acc[i] == c:
                found = True
                break
        if not found:
            acc.append(c)


struct EmbeddedFont(Movable):
    # Handle returned by embed_font: the Type0 font object number plus the exact
    # set of codepoints that were embedded (i.e. have a /W width AND a ToUnicode
    # mapping). show_text_unicode validates against this set so a codepoint that
    # was never embedded can never be silently emitted as unextractable text.
    var obj_num: Int
    var cps: List[Int]

    def __init__(out self, obj_num: Int, var cps: List[Int]):
        self.obj_num = obj_num
        self.cps = cps^

    def has_cp(self, cp: Int) raises -> Bool:
        for i in range(len(self.cps)):
            if self.cps[i] == cp:
                return True
        return False


def embed_font(
    mut doc: PdfDoc,
    font: TtfFont,
    base_name: String,
    used_codepoints: List[Int],
) raises -> EmbeddedFont:
    # Adds 5 objects (FontFile2, FontDescriptor, CIDFontType2, ToUnicode, Type0)
    # and returns an EmbeddedFont handle: the Type0 font object number plus the
    # codepoints actually embedded. Pass EVERY codepoint you will render — any
    # codepoint not in used_codepoints (or absent from the font) is rejected by
    # show_text_unicode rather than silently emitted as unextractable text.

    # ---- 1) FontFile2 stream (raw TTF bytes, FlateDecode) ----
    var ttf_bytes = font.raw()
    var uncompressed_len = len(ttf_bytes)
    var compressed = _zlib_wrap(ttf_bytes)
    var ff_dict = List[UInt8]()
    append_str(ff_dict, "/Filter /FlateDecode /Length1 ")
    append_int(ff_dict, uncompressed_len)
    var fontfile_num = doc.add_stream_object(ff_dict, compressed)

    # ---- Build the list of used GIDs (deduplicated, paired with codepoint) ----
    # Each entry: gid and its codepoint (for /W and ToUnicode).
    var gids = List[Int]()
    var cps = List[Int]()
    for i in range(len(used_codepoints)):
        var cp = used_codepoints[i]
        var gid = font.gid_for(cp)
        if gid == 0:
            continue  # glyph not in font; skip (notdef)
        var dup = False
        for j in range(len(gids)):
            if gids[j] == gid:
                dup = True
                break
        if not dup:
            gids.append(gid)
            cps.append(cp)

    # ---- 2) FontDescriptor ----
    var fd = List[UInt8]()
    append_str(fd, "<< /Type /FontDescriptor /FontName /")
    append_str(fd, base_name)
    append_str(fd, " /Flags 4 /FontBBox [ -1000 -300 2000 1000 ]")
    append_str(fd, " /ItalicAngle 0 /Ascent 900 /Descent -236 /CapHeight 700")
    append_str(fd, " /StemV 80 /FontFile2 ")
    append_int(fd, fontfile_num)
    append_str(fd, " 0 R >>")
    var fontdesc_num = doc.add_object(fd)

    # ---- 3) CIDFontType2 (descendant font) ----
    var cf = List[UInt8]()
    append_str(cf, "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /")
    append_str(cf, base_name)
    append_str(
        cf,
        " /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity)"
        " /Supplement 0 >> /FontDescriptor ",
    )
    append_int(cf, fontdesc_num)
    append_str(cf, " 0 R /CIDToGIDMap /Identity /DW 1000 /W [ ")
    # /W: one entry per gid -> [ gid [advance] ]
    for i in range(len(gids)):
        append_int(cf, gids[i])
        append_str(cf, " [ ")
        append_int(cf, font.advance_1000(gids[i]))
        append_str(cf, " ] ")
    append_str(cf, "] >>")
    var cidfont_num = doc.add_object(cf)

    # ---- 4) ToUnicode CMap stream ----
    var cmap = List[UInt8]()
    append_str(
        cmap,
        "/CIDInit /ProcSet findresource begin\n"
        "12 dict begin\n"
        "begincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >>"
        " def\n"
        "/CMapName /Adobe-Identity-UCS def\n"
        "/CMapType 2 def\n"
        "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n",
    )
    append_int(cmap, len(gids))
    append_str(cmap, " beginbfchar\n")
    for i in range(len(gids)):
        cmap.append(UInt8(ord("<")))
        _hex4(cmap, gids[i])
        append_str(cmap, "> <")
        # UTF-16BE of the codepoint (BMP -> single unit; astral -> surrogate pair)
        var cp = cps[i]
        if cp <= 0xFFFF:
            _hex4(cmap, cp)
        else:
            var v = cp - 0x10000
            var hi = 0xD800 + (v >> 10)
            var lo = 0xDC00 + (v & 0x3FF)
            _hex4(cmap, hi)
            _hex4(cmap, lo)
        append_str(cmap, ">\n")
    append_str(cmap, "endbfchar\n")
    append_str(
        cmap,
        "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n",
    )
    var empty_dict = List[UInt8]()
    var tounicode_num = doc.add_stream_object(empty_dict, cmap)

    # ---- 5) Type0 font ----
    var t0 = List[UInt8]()
    append_str(t0, "<< /Type /Font /Subtype /Type0 /BaseFont /")
    append_str(t0, base_name)
    append_str(t0, " /Encoding /Identity-H /DescendantFonts [ ")
    append_int(t0, cidfont_num)
    append_str(t0, " 0 R ] /ToUnicode ")
    append_int(t0, tounicode_num)
    append_str(t0, " 0 R >>")
    var type0_num = doc.add_object(t0)

    # The embedded set is exactly the codepoints whose glyph was emitted into
    # /W and ToUnicode (gid != 0). cps parallels gids.
    return EmbeddedFont(type0_num, cps.copy())


def show_text_unicode(
    mut b: List[UInt8], font: TtfFont, emb: EmbeddedFont, s: String
) raises:
    # Emit "<HHHH HHHH ...> Tj\n": for each codepoint write its GID as 4 hex
    # digits. Identity-H consumes 2 bytes per code (= the GID). Every codepoint
    # MUST be in the embedded set (so it has a ToUnicode mapping + width) — a
    # missing one raises rather than producing unextractable / mis-measured text.
    b.append(UInt8(ord("<")))
    var first = True
    for cp in s.codepoints():
        var c = Int(cp)
        if not emb.has_cp(c):
            raise Error(
                "show_text_unicode: codepoint not embedded (U+"
                + _hex_str(c)
                + "); add it to the codepoint list passed to embed_font"
            )
        var gid = font.gid_for(c)
        if not first:
            b.append(UInt8(ord(" ")))
        first = False
        _hex4(b, gid)
    append_str(b, "> Tj\n")


def _hex_str(v: Int) raises -> String:
    # Uppercase hex of v (>=0), at least 4 digits, for error messages.
    var hexb = String("0123456789ABCDEF").as_bytes()
    var n = v
    var digits = List[Int]()
    if n == 0:
        digits.append(0)
    while n > 0:
        digits.append(n & 0xF)
        n = n >> 4
    while len(digits) < 4:
        digits.append(0)
    var out = String("")
    var k = len(digits)
    while k > 0:
        k = k - 1
        out += chr(Int(hexb[digits[k]]))
    return out^
