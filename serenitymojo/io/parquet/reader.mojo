# serenitymojo/io/parquet/reader.mojo — Parquet file/footer reader (100% Mojo).
#
# Part 1 (metadata): locate the footer, decode FileMetaData via the Thrift
# compact protocol, and expose the per-column-chunk facts a value reader needs —
# name, physical type, codec, page offsets, sizes, and the column's max
# definition level (derived from the schema's repetition type).
#
# Part 2 (values): walk a column chunk's pages, Snappy-decompress, and decode
# PLAIN / dictionary BYTE_ARRAY values (V1 and V2 data pages).
#
# Scope: flat schemas (no nested/repeated groups), BYTE_ARRAY leaf columns,
# UNCOMPRESSED or SNAPPY codec. Everything these ML dataset shards use.
#
# Parquet Thrift structs referenced (field ids matter):
#   FileMetaData{1 version, 2 schema[], 3 num_rows, 4 row_groups[], 6 created_by}
#   SchemaElement{1 type, 3 repetition_type, 4 name, 5 num_children}
#   RowGroup{1 columns[], 2 total_byte_size, 3 num_rows}
#   ColumnChunk{2 file_offset, 3 meta_data}
#   ColumnMetaData{1 type, 2 encodings[], 3 path_in_schema[], 4 codec,
#                  5 num_values, 7 total_compressed_size, 9 data_page_offset,
#                  11 dictionary_page_offset}

from serenitymojo.io.parquet.thrift import (
    th_field_header, th_list_header, th_read_varint, th_read_zigzag,
    th_read_string, th_skip,
    CT_STOP, CT_STRUCT, CT_LIST, CT_BINARY, CT_BOOL_TRUE,
)
from serenitymojo.io.parquet.snappy import snappy_decompress
from serenitymojo.io.ffi import (
    BytePtr, O_RDONLY, sys_open, sys_pread, sys_close, file_size,
)

# ── Parquet physical type ids (SchemaElement.type / ColumnMetaData.type) ─────
comptime PT_BYTE_ARRAY: Int = 6

# ── repetition types (SchemaElement.repetition_type) ─────────────────────────
comptime REP_REQUIRED: Int = 0
comptime REP_OPTIONAL: Int = 1
comptime REP_REPEATED: Int = 2

# ── compression codecs (ColumnMetaData.codec) ────────────────────────────────
comptime CODEC_UNCOMPRESSED: Int = 0
comptime CODEC_SNAPPY: Int = 1

# ── page types (PageHeader.type) ─────────────────────────────────────────────
comptime PAGE_DATA_V1: Int = 0
comptime PAGE_DICTIONARY: Int = 2
comptime PAGE_DATA_V2: Int = 3

# ── encodings (DataPageHeader.encoding, dict page encoding) ──────────────────
comptime ENC_PLAIN: Int = 0
comptime ENC_PLAIN_DICTIONARY: Int = 2
comptime ENC_RLE: Int = 3
comptime ENC_RLE_DICTIONARY: Int = 8


@fieldwise_init
struct ColumnChunkMeta(Copyable, Movable):
    var name: String
    var ptype: Int          # physical type (PT_BYTE_ARRAY expected)
    var codec: Int
    var num_values: Int
    var total_compressed_size: Int
    var data_page_offset: Int
    var dict_page_offset: Int   # -1 when absent
    var max_def_level: Int      # 1 if column is OPTIONAL else 0

    def page_start(self) -> Int:
        # Pages begin at the dictionary page when present, else the first data page.
        if self.dict_page_offset >= 0:
            return self.dict_page_offset
        return self.data_page_offset


@fieldwise_init
struct ParquetMeta(Copyable, Movable):
    var num_rows: Int
    var columns: List[ColumnChunkMeta]

    def col_index(self, name: String) raises -> Int:
        for i in range(len(self.columns)):
            if self.columns[i].name == name:
                return i
        raise Error("parquet: no column named '" + name + "'")


@fieldwise_init
struct _SchemaLeaf(Copyable, Movable):
    var name: String
    var rep: Int


# ── file load + footer location ──────────────────────────────────────────────

def read_file_bytes(path: String) raises -> List[UInt8]:
    # File I/O routes through io/ffi (sys_open/sys_pread) — never the builtin
    # `open`, whose symbol collides with ffi's external_call["open"] when both
    # land in one compilation unit (see serenitymojo/MAP.md).
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        raise Error("parquet: cannot open '" + path + "'")
    var sz = file_size(fd)
    if sz < 0:
        _ = sys_close(fd)
        raise Error("parquet: cannot size '" + path + "'")
    var out = List[UInt8](capacity=sz if sz > 0 else 1)
    out.resize(sz, 0)
    if sz > 0:
        var bp = BytePtr(unsafe_from_address=Int(out.unsafe_ptr()))
        var done = 0
        while done < sz:
            var n = sys_pread(fd, bp + done, sz - done, done)
            if n <= 0:
                break
            done += n
        _ = sys_close(fd)
        if done != sz:
            raise Error("parquet: short read on '" + path + "'")
    else:
        _ = sys_close(fd)
    return out^


def _u32le(data: List[UInt8], off: Int) -> Int:
    return (
        Int(data[off])
        | (Int(data[off + 1]) << 8)
        | (Int(data[off + 2]) << 16)
        | (Int(data[off + 3]) << 24)
    )


def parse_metadata(data: List[UInt8]) raises -> ParquetMeta:
    var n = len(data)
    if n < 12:
        raise Error("parquet: file too small")
    # trailer: [FileMetaData][u32 footer_len][magic 'PAR1']
    if not (Int(data[n - 4]) == ord("P") and Int(data[n - 3]) == ord("A")
            and Int(data[n - 2]) == ord("R") and Int(data[n - 1]) == ord("1")):
        raise Error("parquet: bad trailing magic (not a Parquet file?)")
    var footer_len = _u32le(data, n - 8)
    var meta_start = n - 8 - footer_len
    if meta_start < 4:
        raise Error("parquet: footer length out of range")

    var pos = meta_start
    var last = 0
    var num_rows = 0
    var leaves = List[_SchemaLeaf]()
    var columns = List[ColumnChunkMeta]()

    # ── FileMetaData ─────────────────────────────────────────────────────────
    while True:
        var fh = th_field_header(data, pos, last)
        if fh.ctype == CT_STOP:
            break
        if fh.fid == 2 and fh.ctype == CT_LIST:
            _parse_schema(data, pos, leaves)
        elif fh.fid == 3:
            num_rows = th_read_zigzag(data, pos)  # i64
        elif fh.fid == 4 and fh.ctype == CT_LIST:
            _parse_row_groups(data, pos, columns)
        else:
            th_skip(data, pos, fh.ctype)

    # Assign each column its max definition level from the matching schema leaf.
    for i in range(len(columns)):
        var lvl = 0
        for j in range(len(leaves)):
            if leaves[j].name == columns[i].name:
                if leaves[j].rep == REP_OPTIONAL:
                    lvl = 1
                break
        columns[i].max_def_level = lvl

    return ParquetMeta(num_rows, columns^)


def _parse_schema(data: List[UInt8], mut pos: Int, mut leaves: List[_SchemaLeaf]) raises:
    var lh = th_list_header(data, pos)  # list<SchemaElement>
    for _ in range(lh.size):
        var name = String("")
        var rep = REP_REQUIRED
        var has_type = False
        var last = 0
        while True:
            var fh = th_field_header(data, pos, last)
            if fh.ctype == CT_STOP:
                break
            if fh.fid == 1:
                _ = th_read_zigzag(data, pos)  # type (i32) — presence ⇒ leaf
                has_type = True
            elif fh.fid == 3:
                rep = th_read_zigzag(data, pos)  # repetition_type
            elif fh.fid == 4:
                name = th_read_string(data, pos)
            else:
                th_skip(data, pos, fh.ctype)
        if has_type:  # root/group elements carry no type; only leaves do
            leaves.append(_SchemaLeaf(name, rep))


def _parse_row_groups(data: List[UInt8], mut pos: Int, mut columns: List[ColumnChunkMeta]) raises:
    var lh = th_list_header(data, pos)  # list<RowGroup>
    for _ in range(lh.size):
        var last = 0
        while True:
            var fh = th_field_header(data, pos, last)
            if fh.ctype == CT_STOP:
                break
            if fh.fid == 1 and fh.ctype == CT_LIST:
                _parse_columns(data, pos, columns)
            else:
                th_skip(data, pos, fh.ctype)


def _parse_columns(data: List[UInt8], mut pos: Int, mut columns: List[ColumnChunkMeta]) raises:
    var lh = th_list_header(data, pos)  # list<ColumnChunk>
    for _ in range(lh.size):
        var last = 0
        while True:
            var fh = th_field_header(data, pos, last)
            if fh.ctype == CT_STOP:
                break
            if fh.fid == 3 and fh.ctype == CT_STRUCT:
                columns.append(_parse_column_meta(data, pos))
            else:
                th_skip(data, pos, fh.ctype)


def _parse_column_meta(data: List[UInt8], mut pos: Int) raises -> ColumnChunkMeta:
    var name = String("")
    var ptype = 0
    var codec = CODEC_UNCOMPRESSED
    var num_values = 0
    var total_comp = 0
    var data_off = 0
    var dict_off = -1
    var last = 0
    while True:
        var fh = th_field_header(data, pos, last)
        if fh.ctype == CT_STOP:
            break
        if fh.fid == 1:
            ptype = th_read_zigzag(data, pos)
        elif fh.fid == 3 and fh.ctype == CT_LIST:
            # path_in_schema: list<string> — join with '.'
            var pl = th_list_header(data, pos)
            for k in range(pl.size):
                var seg = th_read_string(data, pos)
                if k == 0:
                    name = seg
                else:
                    name += "." + seg
        elif fh.fid == 4:
            codec = th_read_zigzag(data, pos)
        elif fh.fid == 5:
            num_values = th_read_zigzag(data, pos)
        elif fh.fid == 7:
            total_comp = th_read_zigzag(data, pos)
        elif fh.fid == 9:
            data_off = th_read_zigzag(data, pos)
        elif fh.fid == 11:
            dict_off = th_read_zigzag(data, pos)
        else:
            th_skip(data, pos, fh.ctype)
    return ColumnChunkMeta(
        name, ptype, codec, num_values, total_comp, data_off, dict_off, 0
    )


# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — page walking + BYTE_ARRAY value decode
# ═══════════════════════════════════════════════════════════════════════════

@fieldwise_init
struct PageHead(Copyable, Movable):
    var ptype: Int
    var uncompressed_size: Int
    var compressed_size: Int
    var num_values: Int
    var encoding: Int
    var def_len: Int        # DATA_PAGE_V2 only
    var rep_len: Int        # DATA_PAGE_V2 only
    var is_compressed: Int  # DATA_PAGE_V2 only (default 1)


def _bits(maxval: Int) -> Int:
    """Bit width needed to store levels/indices in 0..maxval (Parquet BitWidth)."""
    var b = 0
    var v = maxval
    while v > 0:
        b += 1
        v >>= 1
    return b


def _slice(data: List[UInt8], start: Int, n: Int) raises -> List[UInt8]:
    if start < 0 or start + n > len(data):
        raise Error("parquet: slice out of range")
    var out = List[UInt8](capacity=n)
    for i in range(n):
        out.append(data[start + i])
    return out^


def _decompress(data: List[UInt8], start: Int, comp_len: Int, uncomp_len: Int, codec: Int) raises -> List[UInt8]:
    if codec == CODEC_UNCOMPRESSED:
        return _slice(data, start, comp_len)
    elif codec == CODEC_SNAPPY:
        var blk = _slice(data, start, comp_len)
        var out = snappy_decompress(blk)
        if len(out) != uncomp_len:
            raise Error("parquet: decompressed size mismatch")
        return out^
    else:
        raise Error("parquet: unsupported codec " + String(codec))


def _u32le_buf(buf: List[UInt8], off: Int) -> Int:
    return (
        Int(buf[off]) | (Int(buf[off + 1]) << 8)
        | (Int(buf[off + 2]) << 16) | (Int(buf[off + 3]) << 24)
    )


def _plain_byte_arrays(buf: List[UInt8], mut off: Int, count: Int) raises -> List[List[UInt8]]:
    """Decode `count` PLAIN BYTE_ARRAYs: repeated [u32 LE length][bytes]."""
    var out = List[List[UInt8]]()
    for _ in range(count):
        if off + 4 > len(buf):
            raise Error("parquet: plain byte_array truncated length")
        var n = _u32le_buf(buf, off)
        off += 4
        if off + n > len(buf):
            raise Error("parquet: plain byte_array truncated value")
        var v = List[UInt8](capacity=n)
        for i in range(n):
            v.append(buf[off + i])
        off += n
        out.append(v^)
    return out^


def _rle_bitpack(buf: List[UInt8], mut off: Int, end: Int, bitwidth: Int, count: Int) raises -> List[Int]:
    """RLE/bit-packing hybrid decoder (Parquet levels + dictionary indices)."""
    var out = List[Int](capacity=count)
    if bitwidth == 0:
        for _ in range(count):
            out.append(0)
        return out^
    var mask = (1 << bitwidth) - 1
    while len(out) < count and off < end:
        var header = th_read_varint(buf, off)
        if (header & 1) == 0:
            # RLE run: repeat one value `run` times; value is ceil(bw/8) bytes LE.
            var run = header >> 1
            var nbytes = (bitwidth + 7) // 8
            var val = 0
            for i in range(nbytes):
                val |= Int(buf[off]) << (8 * i)
                off += 1
            for _ in range(run):
                if len(out) >= count:
                    break
                out.append(val)
        else:
            # bit-packed run: `groups` groups of 8 values, `bitwidth` bits each.
            var groups = header >> 1
            var nvals = groups * 8
            var bitbuf = 0
            var bitcnt = 0
            for _ in range(nvals):
                while bitcnt < bitwidth:
                    bitbuf |= Int(buf[off]) << bitcnt
                    off += 1
                    bitcnt += 8
                var v = bitbuf & mask
                bitbuf >>= bitwidth
                bitcnt -= bitwidth
                if len(out) < count:
                    out.append(v)
    return out^


def _parse_page_header(data: List[UInt8], mut pos: Int) raises -> PageHead:
    var ptype = 0
    var usz = 0
    var csz = 0
    var nvals = 0
    var enc = 0
    var deflen = 0
    var replen = 0
    var iscomp = 1
    var last = 0
    while True:
        var fh = th_field_header(data, pos, last)
        if fh.ctype == CT_STOP:
            break
        if fh.fid == 1:
            ptype = th_read_zigzag(data, pos)
        elif fh.fid == 2:
            usz = th_read_zigzag(data, pos)
        elif fh.fid == 3:
            csz = th_read_zigzag(data, pos)
        elif fh.fid == 5 and fh.ctype == CT_STRUCT:  # DataPageHeader (V1)
            var l2 = 0
            while True:
                var f2 = th_field_header(data, pos, l2)
                if f2.ctype == CT_STOP:
                    break
                if f2.fid == 1:
                    nvals = th_read_zigzag(data, pos)
                elif f2.fid == 2:
                    enc = th_read_zigzag(data, pos)
                else:
                    th_skip(data, pos, f2.ctype)
        elif fh.fid == 7 and fh.ctype == CT_STRUCT:  # DictionaryPageHeader
            var l2 = 0
            while True:
                var f2 = th_field_header(data, pos, l2)
                if f2.ctype == CT_STOP:
                    break
                if f2.fid == 1:
                    nvals = th_read_zigzag(data, pos)
                elif f2.fid == 2:
                    enc = th_read_zigzag(data, pos)
                else:
                    th_skip(data, pos, f2.ctype)
        elif fh.fid == 8 and fh.ctype == CT_STRUCT:  # DataPageHeaderV2
            var l2 = 0
            while True:
                var f2 = th_field_header(data, pos, l2)
                if f2.ctype == CT_STOP:
                    break
                if f2.fid == 1:
                    nvals = th_read_zigzag(data, pos)
                elif f2.fid == 4:
                    enc = th_read_zigzag(data, pos)
                elif f2.fid == 5:
                    deflen = th_read_zigzag(data, pos)
                elif f2.fid == 6:
                    replen = th_read_zigzag(data, pos)
                elif f2.fid == 7:
                    # is_compressed bool — value rides in the type nibble
                    iscomp = 1 if f2.ctype == CT_BOOL_TRUE else 0
                else:
                    th_skip(data, pos, f2.ctype)
        else:
            th_skip(data, pos, fh.ctype)
    return PageHead(ptype, usz, csz, nvals, enc, deflen, replen, iscomp)


def _decode_values_into(
    buf: List[UInt8], mut off: Int, end: Int, encoding: Int, count: Int,
    dict: List[List[UInt8]], have_dict: Bool, mut out: List[List[UInt8]],
) raises:
    if encoding == ENC_PLAIN:
        var vals = _plain_byte_arrays(buf, off, count)
        for i in range(len(vals)):
            out.append(vals[i].copy())
    elif encoding == ENC_RLE_DICTIONARY or encoding == ENC_PLAIN_DICTIONARY:
        if not have_dict:
            raise Error("parquet: dictionary-encoded page with no dictionary page")
        var bw = Int(buf[off])  # first byte of the data is the index bit width
        off += 1
        var idxs = _rle_bitpack(buf, off, end, bw, count)
        for i in range(count):
            var k = idxs[i]
            if k < 0 or k >= len(dict):
                raise Error("parquet: dictionary index out of range")
            out.append(dict[k].copy())
    else:
        raise Error("parquet: unsupported value encoding " + String(encoding))


def read_byte_array_column(data: List[UInt8], meta: ColumnChunkMeta) raises -> List[List[UInt8]]:
    """Decode every value of one BYTE_ARRAY column chunk (PLAIN or dictionary)."""
    if meta.ptype != PT_BYTE_ARRAY:
        raise Error("parquet: read_byte_array_column on non-BYTE_ARRAY column")
    var pos = meta.page_start()
    var span_end = pos + meta.total_compressed_size
    var dict = List[List[UInt8]]()
    var have_dict = False
    var values = List[List[UInt8]]()
    var dbits = _bits(meta.max_def_level)

    while len(values) < meta.num_values and pos < span_end:
        var ph = _parse_page_header(data, pos)
        var data_start = pos
        pos += ph.compressed_size  # step to the next page regardless of what we do

        if ph.ptype == PAGE_DICTIONARY:
            var buf = _decompress(data, data_start, ph.compressed_size, ph.uncompressed_size, meta.codec)
            var off = 0
            dict = _plain_byte_arrays(buf, off, ph.num_values)
            have_dict = True

        elif ph.ptype == PAGE_DATA_V1:
            var buf = _decompress(data, data_start, ph.compressed_size, ph.uncompressed_size, meta.codec)
            var off = 0
            var present = ph.num_values
            if meta.max_def_level > 0:
                # V1 def levels: [u32 LE byte-length][RLE hybrid]
                var dlen = _u32le_buf(buf, 0)
                off = 4
                var lvl_end = off + dlen
                var levels = _rle_bitpack(buf, off, lvl_end, dbits, ph.num_values)
                off = lvl_end
                present = _count_present(levels, meta.max_def_level)
            _decode_values_into(buf, off, len(buf), ph.encoding, present, dict, have_dict, values)

        elif ph.ptype == PAGE_DATA_V2:
            # Layout: [rep levels | def levels | values]; levels uncompressed.
            var present = ph.num_values
            if meta.max_def_level > 0:
                var doff = data_start + ph.rep_len
                var dend = doff + ph.def_len
                var levels = _rle_bitpack(data, doff, dend, dbits, ph.num_values)
                present = _count_present(levels, meta.max_def_level)
            var vstart = data_start + ph.rep_len + ph.def_len
            var vcomp = ph.compressed_size - ph.rep_len - ph.def_len
            var vuncomp = ph.uncompressed_size - ph.rep_len - ph.def_len
            var off = 0
            if ph.is_compressed == 1:
                var vbuf = _decompress(data, vstart, vcomp, vuncomp, meta.codec)
                _decode_values_into(vbuf, off, len(vbuf), ph.encoding, present, dict, have_dict, values)
            else:
                var vbuf = _slice(data, vstart, vcomp)
                _decode_values_into(vbuf, off, len(vbuf), ph.encoding, present, dict, have_dict, values)
        # else: INDEX_PAGE or unknown — already stepped past it.

    if len(values) != meta.num_values:
        raise Error(
            "parquet: decoded " + String(len(values)) + " values, expected "
            + String(meta.num_values) + " (nulls present?)"
        )
    return values^


def _count_present(levels: List[Int], max_def: Int) -> Int:
    var c = 0
    for i in range(len(levels)):
        if levels[i] >= max_def:
            c += 1
    return c


# ═══════════════════════════════════════════════════════════════════════════
# Nullable-aware read — values ALIGNED to rows + a present mask, so a NULL cell
# stays in position (SimpleTuner-style metadata shards have null captions that
# fall back to another column). `read_byte_array_column` above is unchanged (it
# raises on nulls); this parallel reader scatters decoded values by def-level.
# ═══════════════════════════════════════════════════════════════════════════

@fieldwise_init
struct AlignedColumn(Copyable, Movable):
    var values: List[List[UInt8]]   # length == num_values; null cells are empty
    var present: List[Bool]         # False where the cell was NULL


def _present_count(levels: List[Int], max_def: Int, n: Int) -> Int:
    if max_def == 0:
        return n  # non-nullable column has no def levels — all rows present
    return _count_present(levels, max_def)


def _scatter(levels: List[Int], max_def: Int, n: Int, pv: List[List[UInt8]],
             mut values: List[List[UInt8]], mut present: List[Bool]) raises:
    var ptr = 0
    for r in range(n):
        var is_present = (max_def == 0) or (levels[r] >= max_def)
        if is_present:
            values.append(pv[ptr].copy())
            present.append(True)
            ptr += 1
        else:
            values.append(List[UInt8]())
            present.append(False)


def read_byte_array_column_aligned(data: List[UInt8], meta: ColumnChunkMeta) raises -> AlignedColumn:
    """Like read_byte_array_column but returns num_values entries aligned to rows,
    with null cells as empty values (present=False). Used for metadata shards."""
    if meta.ptype != PT_BYTE_ARRAY:
        raise Error("parquet: read_byte_array_column_aligned on non-BYTE_ARRAY column")
    var pos = meta.page_start()
    var span_end = pos + meta.total_compressed_size
    var dict = List[List[UInt8]]()
    var have_dict = False
    var values = List[List[UInt8]]()
    var present = List[Bool]()
    var dbits = _bits(meta.max_def_level)

    while len(values) < meta.num_values and pos < span_end:
        var ph = _parse_page_header(data, pos)
        var data_start = pos
        pos += ph.compressed_size

        if ph.ptype == PAGE_DICTIONARY:
            var buf = _decompress(data, data_start, ph.compressed_size, ph.uncompressed_size, meta.codec)
            var off = 0
            dict = _plain_byte_arrays(buf, off, ph.num_values)
            have_dict = True

        elif ph.ptype == PAGE_DATA_V1:
            var buf = _decompress(data, data_start, ph.compressed_size, ph.uncompressed_size, meta.codec)
            var off = 0
            var levels = List[Int]()
            if meta.max_def_level > 0:
                var dlen = _u32le_buf(buf, 0)
                off = 4
                var lvl_end = off + dlen
                levels = _rle_bitpack(buf, off, lvl_end, dbits, ph.num_values)
                off = lvl_end
            var np = _present_count(levels, meta.max_def_level, ph.num_values)
            var pv = List[List[UInt8]]()
            _decode_values_into(buf, off, len(buf), ph.encoding, np, dict, have_dict, pv)
            _scatter(levels, meta.max_def_level, ph.num_values, pv, values, present)

        elif ph.ptype == PAGE_DATA_V2:
            var levels = List[Int]()
            if meta.max_def_level > 0:
                var doff = data_start + ph.rep_len
                var dend = doff + ph.def_len
                levels = _rle_bitpack(data, doff, dend, dbits, ph.num_values)
            var np = _present_count(levels, meta.max_def_level, ph.num_values)
            var vstart = data_start + ph.rep_len + ph.def_len
            var vcomp = ph.compressed_size - ph.rep_len - ph.def_len
            var vuncomp = ph.uncompressed_size - ph.rep_len - ph.def_len
            var pv = List[List[UInt8]]()
            var voff = 0
            if ph.is_compressed == 1:
                var vbuf = _decompress(data, vstart, vcomp, vuncomp, meta.codec)
                _decode_values_into(vbuf, voff, len(vbuf), ph.encoding, np, dict, have_dict, pv)
            else:
                var vbuf = _slice(data, vstart, vcomp)
                _decode_values_into(vbuf, voff, len(vbuf), ph.encoding, np, dict, have_dict, pv)
            _scatter(levels, meta.max_def_level, ph.num_values, pv, values, present)

    if len(values) != meta.num_values:
        raise Error("parquet: aligned decode row-count mismatch")
    return AlignedColumn(values^, present^)
