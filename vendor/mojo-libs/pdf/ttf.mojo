# pdf/ttf.mojo
# Minimal TrueType font parser over raw font bytes.
#
# Parses the table directory and the head / maxp / hhea / hmtx / cmap tables
# enough to support PDF Type0/CIDFontType2 embedding:
#   - units_per_em()           : font design grid size
#   - num_glyphs()             : glyph count (maxp)
#   - gid_for(codepoint)       : Unicode -> glyph id via cmap (format 4 / 12)
#   - advance_1000(gid)        : horizontal advance scaled to a 1000-unit em
#   - raw()                    : the original font bytes (for /FontFile2)
#
# Endianness: all multi-byte TrueType integers are BIG-ENDIAN.


def _u16(data: List[UInt8], off: Int) raises -> Int:
    # Unsigned 16-bit big-endian.
    return (Int(data[off]) << 8) | Int(data[off + 1])


def _i16(data: List[UInt8], off: Int) raises -> Int:
    # Signed 16-bit big-endian.
    var v = (Int(data[off]) << 8) | Int(data[off + 1])
    if v >= 0x8000:
        v = v - 0x10000
    return v


def _u32(data: List[UInt8], off: Int) raises -> Int:
    # Unsigned 32-bit big-endian.
    return (
        (Int(data[off]) << 24)
        | (Int(data[off + 1]) << 16)
        | (Int(data[off + 2]) << 8)
        | Int(data[off + 3])
    )


struct TtfFont(Movable):
    var _data: List[UInt8]
    var _units_per_em: Int
    var _num_glyphs: Int
    var _num_hmetrics: Int
    var _hmtx_off: Int

    # cmap format-4 segment arrays (parsed once at load).
    var _seg_count: Int
    var _end_code: List[Int]
    var _start_code: List[Int]
    var _id_delta: List[Int]
    var _id_range_offset: List[Int]
    # Absolute file offset of the idRangeOffset array start (needed because the
    # glyphIdArray is addressed relative to entries inside idRangeOffset).
    var _id_range_off_base: Int
    # cmap format-12 groups (optional): startChar, endChar, startGID triples.
    var _f12_start: List[Int]
    var _f12_end: List[Int]
    var _f12_gid: List[Int]

    def __init__(out self, var data: List[UInt8]) raises:
        self._data = data^
        self._units_per_em = 1000
        self._num_glyphs = 0
        self._num_hmetrics = 0
        self._hmtx_off = 0
        self._seg_count = 0
        self._end_code = List[Int]()
        self._start_code = List[Int]()
        self._id_delta = List[Int]()
        self._id_range_offset = List[Int]()
        self._id_range_off_base = 0
        self._f12_start = List[Int]()
        self._f12_end = List[Int]()
        self._f12_gid = List[Int]()
        self._parse()

    def _parse(mut self) raises:
        ref d = self._data
        # sfnt header: scalerType(4) numTables(2) searchRange(2) ...
        var num_tables = _u16(d, 4)
        var head_off = -1
        var maxp_off = -1
        var hhea_off = -1
        var hmtx_off = -1
        var cmap_off = -1

        # Table directory begins at byte 12; each record is 16 bytes:
        #   tag(4) checksum(4) offset(4) length(4)
        for i in range(num_tables):
            var rec = 12 + i * 16
            var t0 = Int(d[rec])
            var t1 = Int(d[rec + 1])
            var t2 = Int(d[rec + 2])
            var t3 = Int(d[rec + 3])
            var toff = _u32(d, rec + 8)
            # Compare tag against ASCII of known table names.
            if t0 == 0x68 and t1 == 0x65 and t2 == 0x61 and t3 == 0x64:
                head_off = toff  # "head"
            elif t0 == 0x6D and t1 == 0x61 and t2 == 0x78 and t3 == 0x70:
                maxp_off = toff  # "maxp"
            elif t0 == 0x68 and t1 == 0x68 and t2 == 0x65 and t3 == 0x61:
                hhea_off = toff  # "hhea"
            elif t0 == 0x68 and t1 == 0x6D and t2 == 0x74 and t3 == 0x78:
                hmtx_off = toff  # "hmtx"
            elif t0 == 0x63 and t1 == 0x6D and t2 == 0x61 and t3 == 0x70:
                cmap_off = toff  # "cmap"

        if head_off >= 0:
            self._units_per_em = _u16(d, head_off + 18)
        if maxp_off >= 0:
            self._num_glyphs = _u16(d, maxp_off + 4)
        if hhea_off >= 0:
            self._num_hmetrics = _u16(d, hhea_off + 34)
        self._hmtx_off = hmtx_off

        if cmap_off >= 0:
            self._parse_cmap(cmap_off)

    def _parse_cmap(mut self, cmap_off: Int) raises:
        ref d = self._data
        # cmap header: version(2) numTables(2), then records of
        #   platformID(2) encodingID(2) offset(4 from cmap start)
        var num = _u16(d, cmap_off + 2)
        var best = -1
        var best_score = -1
        for i in range(num):
            var rec = cmap_off + 4 + i * 8
            var plat = _u16(d, rec)
            var enc = _u16(d, rec + 2)
            var sub = _u32(d, rec + 4)
            var fmt = _u16(d, cmap_off + sub)
            # Score: prefer (3,1) BMP, then (3,10)/(0,*) format12, then (0,*).
            var score = -1
            if plat == 3 and enc == 1 and (fmt == 4 or fmt == 12):
                score = 90
            elif plat == 3 and enc == 10 and fmt == 12:
                score = 95
            elif plat == 0 and (fmt == 4 or fmt == 12):
                score = 80
            elif fmt == 4 or fmt == 12:
                score = 50
            if score > best_score:
                best_score = score
                best = cmap_off + sub
        if best < 0:
            return
        var fmt = _u16(d, best)
        if fmt == 4:
            self._parse_format4(best)
        elif fmt == 12:
            self._parse_format12(best)

    def _parse_format4(mut self, sub: Int) raises:
        ref d = self._data
        # format(2) length(2) language(2) segCountX2(2) searchRange(2)
        # entrySelector(2) rangeShift(2) endCode[segCount](2 each) reserved(2)
        # startCode[segCount] idDelta[segCount] idRangeOffset[segCount] ...
        var seg_x2 = _u16(d, sub + 6)
        var seg_count = seg_x2 // 2
        self._seg_count = seg_count
        var end_base = sub + 14
        var start_base = end_base + seg_x2 + 2  # +2 reservedPad
        var delta_base = start_base + seg_x2
        var range_base = delta_base + seg_x2
        self._id_range_off_base = range_base
        for i in range(seg_count):
            self._end_code.append(_u16(d, end_base + i * 2))
            self._start_code.append(_u16(d, start_base + i * 2))
            self._id_delta.append(_u16(d, delta_base + i * 2))
            self._id_range_offset.append(_u16(d, range_base + i * 2))

    def _parse_format12(mut self, sub: Int) raises:
        ref d = self._data
        # format(2) reserved(2) length(4) language(4) nGroups(4)
        # groups: startCharCode(4) endCharCode(4) startGlyphID(4)
        var ngroups = _u32(d, sub + 12)
        var gbase = sub + 16
        for i in range(ngroups):
            var g = gbase + i * 12
            self._f12_start.append(_u32(d, g))
            self._f12_end.append(_u32(d, g + 4))
            self._f12_gid.append(_u32(d, g + 8))

    def units_per_em(self) -> Int:
        return self._units_per_em

    def num_glyphs(self) -> Int:
        return self._num_glyphs

    def gid_for(self, cp: Int) raises -> Int:
        # Try format 12 groups first (covers full Unicode if present).
        for i in range(len(self._f12_start)):
            if cp >= self._f12_start[i] and cp <= self._f12_end[i]:
                return self._f12_gid[i] + (cp - self._f12_start[i])
        # Format 4 (BMP only).
        if self._seg_count == 0:
            return 0
        if cp > 0xFFFF:
            return 0
        ref d = self._data
        for i in range(self._seg_count):
            if cp <= self._end_code[i]:
                if cp < self._start_code[i]:
                    return 0
                var ro = self._id_range_offset[i]
                if ro == 0:
                    return (cp + self._id_delta[i]) & 0xFFFF
                # glyphIndexAddress = idRangeOffset[i] + 2*(cp - startCode[i])
                #   + addressof(idRangeOffset[i])
                var addr = (
                    self._id_range_off_base
                    + i * 2
                    + ro
                    + 2 * (cp - self._start_code[i])
                )
                var gid = _u16(d, addr)
                if gid == 0:
                    return 0
                return (gid + self._id_delta[i]) & 0xFFFF
        return 0

    def advance_1000(self, gid: Int) raises -> Int:
        # advanceWidth for gid from hmtx, scaled to a 1000-unit em.
        if self._hmtx_off <= 0 or self._num_hmetrics <= 0:
            return 0
        var idx = gid
        if idx >= self._num_hmetrics:
            idx = self._num_hmetrics - 1  # glyphs past the table reuse last AW
        var aw = _u16(self._data, self._hmtx_off + idx * 4)
        var upm = self._units_per_em
        if upm <= 0:
            return aw
        # Round-half-up scaling.
        return (aw * 1000 + upm // 2) // upm

    def raw(self) raises -> List[UInt8]:
        return self._data.copy()


def ttf_from_bytes(data: List[UInt8]) raises -> TtfFont:
    # Build a TtfFont directly from in-memory font bytes (e.g. a subset produced
    # by pdf/subset.mojo). Parses head/maxp/hhea/hmtx/cmap like load_ttf.
    return TtfFont(data.copy())


def load_ttf(path: String) raises -> TtfFont:
    var f = open(path, "r")
    var data = f.read_bytes()
    f.close()
    var bytes = List[UInt8]()
    for i in range(len(data)):
        bytes.append(data[i])
    return TtfFont(bytes^)
