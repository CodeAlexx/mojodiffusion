# pdf/subset.mojo
# Pure-Mojo TrueType FONT SUBSETTING.
#
# Strategy (the ORIGINAL-GID-PRESERVING path — see "WHICH PATH" below):
#   We keep the original glyph numbering. numGlyphs is UNCHANGED, so loca keeps
#   its full length (numGlyphs+1 offsets). For every glyph that is NOT in the
#   keep-set we emit a ZERO-LENGTH glyph program (loca[i] == loca[i+1]); kept
#   glyphs copy their original outline bytes verbatim. Because gids are never
#   renumbered, glyf / loca / hmtx / maxp / head / hhea / cmap all keep agreeing
#   trivially. We then rebuild a fresh format-4 `cmap` that maps exactly the used
#   codepoints to their (unchanged) original gids.
#
#   Keep-set = {0 (.notdef)} ∪ {gid_for(cp) for cp in used} ∪ composite closure
#   (a composite glyph references component glyphs; those components must be kept
#   or the composite renders empty/broken).
#
# Tables rebuilt: glyf, loca, cmap, maxp (numGlyphs unchanged but re-emitted),
#   head (indexToLocFormat + checkSumAdjustment), hhea (copied verbatim; numHM
#   unchanged), hmtx (copied verbatim). All OTHER tables present in the source
#   (cvt fpgm prep post name OS/2 ...) are copied VERBATIM. Table directory is
#   rebuilt with correct offsets/lengths (4-byte aligned, zero-padded) and per
#   table checksums + head.checkSumAdjustment recomputed.
#
# WHICH PATH SHIPPED: full subset with original-gid preservation (the glyf table
#   shrinks to only the kept glyph programs; everything else stays consistent).
#   This is NOT the degenerate "keep-all-tables-byte-for-byte" fallback — glyf is
#   genuinely rebuilt and shrunk. It is, by design, the "keep original gids"
#   variant the spec calls out as simpler & valid.
#
# Endianness: ALL multi-byte TrueType integers are BIG-ENDIAN.

from pdf.document import PdfDoc
from pdf.ttf import TtfFont, ttf_from_bytes
from pdf.font_embed import embed_font, EmbeddedFont


# ---------------------------------------------------------------------------
# Big-endian read helpers (over a List[UInt8]).
# ---------------------------------------------------------------------------
def _ru16(d: List[UInt8], off: Int) raises -> Int:
    return (Int(d[off]) << 8) | Int(d[off + 1])


def _ri16(d: List[UInt8], off: Int) raises -> Int:
    var v = (Int(d[off]) << 8) | Int(d[off + 1])
    if v >= 0x8000:
        v = v - 0x10000
    return v


def _ru32(d: List[UInt8], off: Int) raises -> Int:
    return (
        (Int(d[off]) << 24)
        | (Int(d[off + 1]) << 16)
        | (Int(d[off + 2]) << 8)
        | Int(d[off + 3])
    )


# ---------------------------------------------------------------------------
# Big-endian write helpers (append onto a List[UInt8]).
# ---------------------------------------------------------------------------
def _wu16(mut b: List[UInt8], v: Int) raises:
    b.append(UInt8((v >> 8) & 0xFF))
    b.append(UInt8(v & 0xFF))


def _wu32(mut b: List[UInt8], v: Int) raises:
    b.append(UInt8((v >> 24) & 0xFF))
    b.append(UInt8((v >> 16) & 0xFF))
    b.append(UInt8((v >> 8) & 0xFF))
    b.append(UInt8(v & 0xFF))


def _set_u16(mut b: List[UInt8], off: Int, v: Int) raises:
    b[off] = UInt8((v >> 8) & 0xFF)
    b[off + 1] = UInt8(v & 0xFF)


def _set_u32(mut b: List[UInt8], off: Int, v: Int) raises:
    b[off] = UInt8((v >> 24) & 0xFF)
    b[off + 1] = UInt8((v >> 16) & 0xFF)
    b[off + 2] = UInt8((v >> 8) & 0xFF)
    b[off + 3] = UInt8(v & 0xFF)


# ---------------------------------------------------------------------------
# Table directory record (parsed from the source font).
# ---------------------------------------------------------------------------
struct _Table(Movable, Copyable):
    var tag: String       # 4-char ASCII tag, e.g. "glyf"
    var off: Int          # offset of table in source bytes
    var length: Int       # length of table in source bytes

    def __init__(out self, var tag: String, off: Int, length: Int):
        self.tag = tag^
        self.off = off
        self.length = length


def _tag_str(d: List[UInt8], rec: Int) raises -> String:
    var s = String("")
    for i in range(4):
        s += chr(Int(d[rec + i]))
    return s^


def _read_dir(d: List[UInt8]) raises -> List[_Table]:
    # Parse the sfnt table directory: numTables at byte 4, records at 12.
    var num = _ru16(d, 4)
    var out = List[_Table]()
    for i in range(num):
        var rec = 12 + i * 16
        var tag = _tag_str(d, rec)
        var off = _ru32(d, rec + 8)
        var ln = _ru32(d, rec + 12)
        out.append(_Table(tag^, off, ln))
    return out^


def _find(tables: List[_Table], tag: String) raises -> Int:
    # Index of table with tag, or -1.
    for i in range(len(tables)):
        if tables[i].tag == tag:
            return i
    return -1


# ---------------------------------------------------------------------------
# checksum: uint32 sum of 32-bit big-endian words over [off, off+len),
# zero-padding the final partial word, masked to 32 bits.
# ---------------------------------------------------------------------------
def _checksum(b: List[UInt8], off: Int, length: Int) raises -> Int:
    var sum = 0
    var i = 0
    while i < length:
        var w = 0
        for k in range(4):
            w = w << 8
            if i + k < length:
                w = w | Int(b[off + i + k])
        sum = (sum + w) & 0xFFFFFFFF
        i = i + 4
    return sum & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# loca reader: returns the glyph-data offset for glyph index `g` (0..numGlyphs).
# index_to_loc_format: 0 = short (offset/2 as u16), 1 = long (u32).
# ---------------------------------------------------------------------------
def _loca_at(d: List[UInt8], loca_off: Int, fmt: Int, g: Int) raises -> Int:
    if fmt == 0:
        return _ru16(d, loca_off + g * 2) * 2
    return _ru32(d, loca_off + g * 4)


# ---------------------------------------------------------------------------
# Composite-glyph closure: a glyph program whose numberOfContours (i16 at the
# start of the program) is < 0 is a composite. Its component records each carry
# a glyphIndex (u16) we must keep. Walk components, expanding MORE_COMPONENTS.
# Flags bit layout (per the OpenType spec):
#   bit0 ARG_1_AND_2_ARE_WORDS (args 4 bytes vs 2)
#   bit3 WE_HAVE_A_SCALE       (+2)
#   bit5 MORE_COMPONENTS
#   bit6 WE_HAVE_AN_X_AND_Y_SCALE (+4)
#   bit7 WE_HAVE_A_TWO_BY_TWO  (+8)
# ---------------------------------------------------------------------------
def _add_components(
    d: List[UInt8],
    glyf_off: Int,
    loca_off: Int,
    loc_fmt: Int,
    num_glyphs: Int,
    gid: Int,
    mut keep: List[Bool],
) raises:
    if gid < 0 or gid >= num_glyphs:
        return
    var g0 = _loca_at(d, loca_off, loc_fmt, gid)
    var g1 = _loca_at(d, loca_off, loc_fmt, gid + 1)
    if g1 <= g0:
        return  # empty glyph, nothing to recurse into
    var p = glyf_off + g0
    var ncont = _ri16(d, p)
    if ncont >= 0:
        return  # simple glyph, no components
    # Composite: skip the 10-byte glyph header (numberOfContours + bbox xMin..),
    # then walk component records.
    var q = p + 10
    while True:
        var flags = _ru16(d, q)
        var comp_gid = _ru16(d, q + 2)
        q = q + 4
        # arg1/arg2 size
        if (flags & 0x0001) != 0:
            q = q + 4
        else:
            q = q + 2
        # transform fields
        if (flags & 0x0008) != 0:
            q = q + 2
        elif (flags & 0x0040) != 0:
            q = q + 4
        elif (flags & 0x0080) != 0:
            q = q + 8
        if comp_gid >= 0 and comp_gid < num_glyphs:
            if not keep[comp_gid]:
                keep[comp_gid] = True
                # Recurse into the newly-kept component (it may itself be composite).
                _add_components(
                    d, glyf_off, loca_off, loc_fmt, num_glyphs, comp_gid, keep
                )
        if (flags & 0x0020) == 0:  # no MORE_COMPONENTS
            break


# ---------------------------------------------------------------------------
# Build a fresh format-4 cmap SUBTABLE BODY (the format-4 subtable bytes only,
# NOT the cmap table header) mapping each BMP (codepoint -> gid) pair. Astral
# codepoints (>0xFFFF) are dropped — format 4 cannot represent them. Returns the
# subtable bytes; the caller wraps these in a cmap table header.
# ---------------------------------------------------------------------------
def _build_cmap4_sub(cps: List[Int], gids: List[Int]) raises -> List[UInt8]:
    # Collect BMP (cp,gid) pairs, sorted ascending by cp.
    var pc = List[Int]()
    var pg = List[Int]()
    for i in range(len(cps)):
        if cps[i] <= 0xFFFF and cps[i] >= 0 and gids[i] != 0:
            pc.append(cps[i])
            pg.append(gids[i])
    # Insertion sort by codepoint (small N).
    var n = len(pc)
    for i in range(1, n):
        var cc = pc[i]
        var gg = pg[i]
        var j = i - 1
        while j >= 0 and pc[j] > cc:
            pc[j + 1] = pc[j]
            pg[j + 1] = pg[j]
            j = j - 1
        pc[j + 1] = cc
        pg[j + 1] = gg

    # Each consecutive codepoint with consecutive-or-arbitrary gid becomes one
    # segment. We use the simple, always-correct strategy: ONE segment per
    # contiguous run of codepoints, with idRangeOffset=0 and idDelta chosen so
    # (cp + idDelta) & 0xFFFF == gid. That only works when within a run the gids
    # are also contiguous with the same delta. To stay fully correct regardless
    # of gid layout, we emit ONE SEGMENT PER CODEPOINT (start==end), idDelta =
    # (gid - cp) & 0xFFFF, idRangeOffset = 0. Plus the mandatory terminating
    # 0xFFFF segment.
    var seg_cp_start = List[Int]()
    var seg_cp_end = List[Int]()
    var seg_delta = List[Int]()
    for i in range(n):
        seg_cp_start.append(pc[i])
        seg_cp_end.append(pc[i])
        seg_delta.append((pg[i] - pc[i]) & 0xFFFF)
    # Terminating segment: 0xFFFF -> 0xFFFF maps to glyph 0 (delta 1).
    seg_cp_start.append(0xFFFF)
    seg_cp_end.append(0xFFFF)
    seg_delta.append(1)

    var seg_count = len(seg_cp_start)
    var seg_x2 = seg_count * 2

    # ---- format-4 subtable ----
    var sub = List[UInt8]()
    _wu16(sub, 4)            # format
    _wu16(sub, 0)            # length placeholder (patched below)
    _wu16(sub, 0)            # language
    _wu16(sub, seg_x2)       # segCountX2
    # searchRange = 2 * 2^floor(log2(segCount)); entrySelector = log2; rangeShift
    var pow2 = 1
    var es = 0
    while pow2 * 2 <= seg_count:
        pow2 = pow2 * 2
        es = es + 1
    var search_range = pow2 * 2
    _wu16(sub, search_range)
    _wu16(sub, es)
    _wu16(sub, seg_x2 - search_range)
    # endCode[]
    for i in range(seg_count):
        _wu16(sub, seg_cp_end[i])
    _wu16(sub, 0)            # reservedPad
    # startCode[]
    for i in range(seg_count):
        _wu16(sub, seg_cp_start[i])
    # idDelta[]
    for i in range(seg_count):
        _wu16(sub, seg_delta[i])
    # idRangeOffset[] (all zero)
    for _i in range(seg_count):
        _wu16(sub, 0)
    # patch subtable length
    _set_u16(sub, 2, len(sub))
    return sub^


# ---------------------------------------------------------------------------
# Build a fresh format-12 cmap SUBTABLE BODY (the "Segmented coverage" subtable
# bytes only, NOT the cmap table header) mapping each (codepoint -> gid) pair.
# Format 12 uses u32 fields, so it can represent the full Unicode range
# (including astral codepoints > 0xFFFF). We emit ONE SequentialMapGroup per
# codepoint (start==end) — always correct regardless of gid layout. Returns the
# subtable bytes; the caller wraps these in a cmap table header.
#
# Subtable layout (big-endian):
#   format(u16=12) reserved(u16=0) length(u32) language(u32=0) nGroups(u32)
#   then nGroups SequentialMapGroup = startCharCode(u32) endCharCode(u32)
#                                     startGlyphID(u32)
# ---------------------------------------------------------------------------
def _build_cmap12_sub(cps: List[Int], gids: List[Int]) raises -> List[UInt8]:
    # Collect (cp,gid) pairs with a real gid, sorted ascending by cp.
    var pc = List[Int]()
    var pg = List[Int]()
    for i in range(len(cps)):
        if cps[i] >= 0 and gids[i] != 0:
            pc.append(cps[i])
            pg.append(gids[i])
    # Insertion sort by codepoint (small N).
    var n = len(pc)
    for i in range(1, n):
        var cc = pc[i]
        var gg = pg[i]
        var j = i - 1
        while j >= 0 and pc[j] > cc:
            pc[j + 1] = pc[j]
            pg[j + 1] = pg[j]
            j = j - 1
        pc[j + 1] = cc
        pg[j + 1] = gg

    var sub = List[UInt8]()
    _wu16(sub, 12)          # format = 12
    _wu16(sub, 0)           # reserved
    _wu32(sub, 0)           # length placeholder (patched below)
    _wu32(sub, 0)           # language = 0
    _wu32(sub, n)           # nGroups (one group per codepoint)
    for i in range(n):
        _wu32(sub, pc[i])   # startCharCode
        _wu32(sub, pc[i])   # endCharCode (== start; one codepoint per group)
        _wu32(sub, pg[i])   # startGlyphID (kept original gid)
    # patch subtable length (u32 at offset 4)
    _set_u32(sub, 4, len(sub))
    return sub^


# ---------------------------------------------------------------------------
# Build the WHOLE cmap table. If ALL used codepoints are <= 0xFFFF, emit ONLY a
# format-4 subtable under a (3,1) record (unchanged legacy behavior — no
# regression for BMP-only inputs). If ANY codepoint is > 0xFFFF (astral), emit a
# cmap with TWO encoding records: a (3,1) format-4 subtable covering the BMP
# subset (for max reader compatibility) AND a (3,10) format-12 subtable covering
# the full set incl. astral codepoints.
#
# cmap table header: version(u16=0) numTables(u16), then numTables encoding
# records each = platformID(u16) encodingID(u16) offset(u32 from cmap start).
# ---------------------------------------------------------------------------
def _build_cmap(cps: List[Int], gids: List[Int]) raises -> List[UInt8]:
    var has_astral = False
    for i in range(len(cps)):
        if cps[i] > 0xFFFF and gids[i] != 0:
            has_astral = True
            break

    var cmap = List[UInt8]()
    if not has_astral:
        # ---- BMP-only: single (3,1) format-4 subtable (legacy behavior) ----
        var sub4 = _build_cmap4_sub(cps, gids)
        _wu16(cmap, 0)      # version
        _wu16(cmap, 1)      # numTables
        _wu16(cmap, 3)      # platformID = 3 (Windows)
        _wu16(cmap, 1)      # encodingID = 1 (Unicode BMP)
        _wu32(cmap, 12)     # offset to subtable (after 4 + 1*8 byte header)
        for i in range(len(sub4)):
            cmap.append(sub4[i])
        return cmap^

    # ---- Astral present: two records, (3,1) format-4 + (3,10) format-12 ----
    var sub4 = _build_cmap4_sub(cps, gids)  # BMP subset only (astral dropped)
    var sub12 = _build_cmap12_sub(cps, gids)  # full set incl. astral
    # Header = version(2) + numTables(2) + 2 records * 8 = 4 + 16 = 20 bytes.
    var header_len = 4 + 2 * 8
    var off4 = header_len
    var off12 = off4 + len(sub4)
    _wu16(cmap, 0)          # version
    _wu16(cmap, 2)          # numTables
    # record 0: (3,1) -> format-4 subtable
    _wu16(cmap, 3)          # platformID = 3
    _wu16(cmap, 1)          # encodingID = 1 (Unicode BMP)
    _wu32(cmap, off4)
    # record 1: (3,10) -> format-12 subtable
    _wu16(cmap, 3)          # platformID = 3
    _wu16(cmap, 10)         # encodingID = 10 (Unicode full repertoire)
    _wu32(cmap, off12)
    for i in range(len(sub4)):
        cmap.append(sub4[i])
    for i in range(len(sub12)):
        cmap.append(sub12[i])
    return cmap^


# ---------------------------------------------------------------------------
# subset_ttf: produce a NEW valid TrueType file containing only the glyphs
# needed for `used_codepoints` (+ .notdef + composite closure).
# ---------------------------------------------------------------------------
def subset_ttf(font: TtfFont, used_codepoints: List[Int]) raises -> List[UInt8]:
    # Public entry point: resolve each codepoint -> gid via the font's cmap, then
    # build the subset. Codepoints whose gid resolves to 0 (not in font) are
    # dropped. This preserves the original behavior for every existing caller.
    var cps = List[Int]()
    var gids = List[Int]()
    for i in range(len(used_codepoints)):
        var cp = used_codepoints[i]
        var gid = font.gid_for(cp)
        if gid > 0:
            cps.append(cp)
            gids.append(gid)
    return subset_ttf_mapped(font, cps, gids)


# ---------------------------------------------------------------------------
# subset_ttf_mapped: like subset_ttf but the (codepoint -> gid) mapping is
# supplied EXPLICITLY (parallel lists). This lets a caller map a codepoint that
# the font's own cmap does NOT cover (e.g. an astral codepoint with no native
# glyph) onto an EXISTING glyph id — exercising the format-12 cmap encoder and
# the ToUnicode plumbing end-to-end via gid reuse. cps[i] is the Unicode
# codepoint; gids[i] is the original gid to map it to (must be > 0 and within
# numGlyphs). Pairs with gid == 0 are ignored.
# ---------------------------------------------------------------------------
def subset_ttf_mapped(
    font: TtfFont, cps: List[Int], gids: List[Int]
) raises -> List[UInt8]:
    var d = font.raw()
    var tables = _read_dir(d)

    var i_head = _find(tables, "head")
    var i_maxp = _find(tables, "maxp")
    var i_loca = _find(tables, "loca")
    var i_glyf = _find(tables, "glyf")
    var i_hhea = _find(tables, "hhea")
    var i_hmtx = _find(tables, "hmtx")
    var i_cmap = _find(tables, "cmap")

    if i_head < 0 or i_maxp < 0 or i_loca < 0 or i_glyf < 0:
        raise Error(
            "subset_ttf: font lacks glyf-based outlines (head/maxp/loca/glyf "
            "required); CFF/OTF not supported"
        )

    var head_off = tables[i_head].off
    var maxp_off = tables[i_maxp].off
    var loca_off = tables[i_loca].off
    var glyf_off = tables[i_glyf].off
    var glyf_len = tables[i_glyf].length

    var num_glyphs = _ru16(d, maxp_off + 4)
    var loc_fmt = _ru16(d, head_off + 50)  # indexToLocFormat

    # ---- 1) Determine kept gids ----
    var keep = List[Bool]()
    for _i in range(num_glyphs):
        keep.append(False)
    if num_glyphs > 0:
        keep[0] = True  # .notdef always kept

    # Use the explicit (codepoint -> gid) mapping supplied by the caller.
    var used_cps = List[Int]()
    var used_gids = List[Int]()
    for i in range(len(cps)):
        var cp = cps[i]
        var gid = gids[i]
        if gid > 0 and gid < num_glyphs:
            keep[gid] = True
            used_cps.append(cp)
            used_gids.append(gid)

    # ---- 2) Composite closure: keep component glyphs of kept composites ----
    # Iterate to a fixpoint (closure may add new composites).
    var changed = True
    while changed:
        changed = False
        for g in range(num_glyphs):
            if not keep[g]:
                continue
            var before = 0
            for k in range(num_glyphs):
                if keep[k]:
                    before = before + 1
            _add_components(
                d, glyf_off, loca_off, loc_fmt, num_glyphs, g, keep
            )
            var after = 0
            for k in range(num_glyphs):
                if keep[k]:
                    after = after + 1
            if after != before:
                changed = True

    # ---- 3) Build new glyf + loca ----
    # New loca uses LONG format (indexToLocFormat = 1) unconditionally — simplest
    # and always valid. New glyf concatenates kept glyph programs (4-byte
    # aligned per glyph); unused glyphs contribute zero length.
    var new_glyf = List[UInt8]()
    var new_loca_off = List[Int]()  # numGlyphs+1 offsets into new_glyf
    new_loca_off.append(0)
    for g in range(num_glyphs):
        if keep[g]:
            var o0 = _loca_at(d, loca_off, loc_fmt, g)
            var o1 = _loca_at(d, loca_off, loc_fmt, g + 1)
            var glen = o1 - o0
            if glen > 0 and o0 >= 0 and (glyf_off + o1) <= (glyf_off + glyf_len):
                for k in range(glen):
                    new_glyf.append(d[glyf_off + o0 + k])
                # pad to 4-byte boundary (long loca permits any, but keep tidy)
                while (len(new_glyf) % 4) != 0:
                    new_glyf.append(UInt8(0))
        new_loca_off.append(len(new_glyf))

    # Build new loca bytes (LONG format).
    var new_loca = List[UInt8]()
    for g in range(num_glyphs + 1):
        _wu32(new_loca, new_loca_off[g])

    # ---- 4) Assemble the list of output tables ----
    # Copy every source table verbatim EXCEPT head/glyf/loca (rebuilt) and cmap
    # (rebuilt). maxp/hhea/hmtx are copied verbatim (numGlyphs and numHMetrics
    # unchanged since we preserve gids).
    var new_cmap = _build_cmap(used_cps, used_gids)

    # Collect (tag, bytes) for each output table.
    var out_tags = List[String]()
    var out_bytes = List[List[UInt8]]()

    for ti in range(len(tables)):
        var tag = tables[ti].tag
        if tag == "glyf":
            out_tags.append(String("glyf"))
            out_bytes.append(new_glyf.copy())
        elif tag == "loca":
            out_tags.append(String("loca"))
            out_bytes.append(new_loca.copy())
        elif tag == "cmap":
            out_tags.append(String("cmap"))
            out_bytes.append(new_cmap.copy())
        elif tag == "head":
            # Copy head verbatim, then patch indexToLocFormat -> 1 (long) and
            # zero checkSumAdjustment (recomputed at the very end).
            var hb = List[UInt8]()
            for k in range(tables[ti].length):
                hb.append(d[tables[ti].off + k])
            _set_u32(hb, 8, 0)       # checkSumAdjustment = 0 (recompute later)
            _set_u16(hb, 50, 1)      # indexToLocFormat = 1 (long)
            out_tags.append(String("head"))
            out_bytes.append(hb.copy())
        else:
            # Copy verbatim (maxp, hhea, hmtx, cvt, fpgm, prep, post, name, OS/2,
            # GSUB, GPOS, ...). Some of these (GSUB/GPOS) reference glyph ids but
            # since gids are preserved they stay valid.
            var cb = List[UInt8]()
            for k in range(tables[ti].length):
                cb.append(d[tables[ti].off + k])
            out_tags.append(tag)
            out_bytes.append(cb.copy())

    var num_out = len(out_tags)

    # ---- 5) Emit sfnt: header + directory + table data (4-byte aligned) ----
    # First compute table offsets. Directory starts at 12, each record 16 bytes.
    var dir_size = num_out * 16
    var data_start = 12 + dir_size
    # offsets/lengths/padded-lengths per table, in directory order. The sfnt
    # spec requires the directory be sorted by tag ascending.
    # Sort indices by tag.
    var order = List[Int]()
    for i in range(num_out):
        order.append(i)
    for i in range(1, num_out):
        var key = order[i]
        var j = i - 1
        while j >= 0 and out_tags[order[j]] > out_tags[key]:
            order[j + 1] = order[j]
            j = j - 1
        order[j + 1] = key

    var offs = List[Int]()
    var lens = List[Int]()
    for _i in range(num_out):
        offs.append(0)
        lens.append(0)
    var cursor = data_start
    for oi in range(num_out):
        var ti = order[oi]
        offs[ti] = cursor
        lens[ti] = len(out_bytes[ti])
        var padded = lens[ti]
        while (padded % 4) != 0:
            padded = padded + 1
        cursor = cursor + padded

    # Build the full output buffer.
    var out = List[UInt8]()
    # sfnt header.
    _wu32(out, 0x00010000)  # version 1.0 (TrueType outlines)
    _wu16(out, num_out)     # numTables
    # searchRange / entrySelector / rangeShift over numTables.
    var p2 = 1
    var es = 0
    while p2 * 2 <= num_out:
        p2 = p2 * 2
        es = es + 1
    var search_range = p2 * 16
    _wu16(out, search_range)
    _wu16(out, es)
    _wu16(out, num_out * 16 - search_range)

    # Directory records (sorted by tag). checksum computed over table bytes
    # (with the head-checksum special case handled after assembly).
    for oi in range(num_out):
        var ti = order[oi]
        var tag = out_tags[ti]
        for c in range(4):
            out.append(UInt8(ord(tag[byte=c])))
        var cks = _checksum(out_bytes[ti], 0, len(out_bytes[ti]))
        _wu32(out, cks)
        _wu32(out, offs[ti])
        _wu32(out, lens[ti])

    # Table data (in directory order; padded to 4 bytes).
    for oi in range(num_out):
        var ti = order[oi]
        # pad output to this table's offset (should already match offs[ti])
        while len(out) < offs[ti]:
            out.append(UInt8(0))
        for k in range(len(out_bytes[ti])):
            out.append(out_bytes[ti][k])
        while (len(out) % 4) != 0:
            out.append(UInt8(0))

    # ---- 6) head.checkSumAdjustment ----
    # = 0xB1B0AFBA - (checksum of the ENTIRE file with head.checkSumAdjustment=0)
    # We already wrote head with checkSumAdjustment=0, so checksum the whole file.
    var i_head_out = -1
    for ti in range(num_out):
        if out_tags[ti] == "head":
            i_head_out = ti
            break
    if i_head_out >= 0:
        var file_cks = _checksum(out, 0, len(out))
        var adj = (0xB1B0AFBA - file_cks) & 0xFFFFFFFF
        # head's checkSumAdjustment lives at head_offset + 8.
        _set_u32(out, offs[i_head_out] + 8, adj)

    return out^


# ---------------------------------------------------------------------------
# Convenience: subset then embed.
# ---------------------------------------------------------------------------
def embed_subset_font(
    mut doc: PdfDoc,
    font: TtfFont,
    name: String,
    codepoints: List[Int],
) raises -> EmbeddedFont:
    # Subset the font to just `codepoints` (+ closure), wrap the subset bytes in a
    # fresh TtfFont (so gid_for / advance_1000 read the SUBSET cmap & hmtx), and
    # embed. Because gids are preserved, the embedded /W widths, ToUnicode map and
    # the content-stream GIDs all stay consistent with the original font too.
    var sub_bytes = subset_ttf(font, codepoints)
    var sub_font = ttf_from_bytes(sub_bytes^)
    return embed_font(doc, sub_font, name, codepoints)
