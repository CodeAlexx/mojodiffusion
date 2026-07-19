# serenitymojo/io/parquet/thrift.mojo — Thrift *compact protocol* reader (Mojo).
#
# Parquet stores its footer (FileMetaData) and every PageHeader as a Thrift
# compact-protocol struct. This module is the generic decoder for that wire
# format; the Parquet-specific struct navigation lives in reader.mojo.
#
# Compact protocol in one screen:
#   * varint            — unsigned LEB128, 7 bits/byte, low byte first.
#   * zigzag            — signed ints: value = (u >> 1) ^ -(u & 1), then varint.
#   * struct            — a run of fields terminated by a 0x00 STOP byte.
#   * field header byte — high nibble = field-id delta from the previous field
#                         (0 → the id follows as a zigzag varint); low nibble =
#                         compact type. Delta resets to id 0 at each struct.
#   * list/set header   — one byte: high nibble = size (0xF → varint follows),
#                         low nibble = element type; then `size` elements.
#   * map               — varint size (0 → done); else a key/value-type byte,
#                         then size (key,value) pairs.
#   * bool              — the *value* rides in the type nibble (1 true / 2 false).
#
# Everything a caller doesn't want is stepped over with `th_skip`, which keeps
# the cursor in sync no matter how the writer laid out optional fields.
#
# Reference: thrift/doc/specs/thrift-compact-protocol.md

# ── compact type ids (the low nibble of a field/element header) ──────────────
comptime CT_STOP: Int = 0
comptime CT_BOOL_TRUE: Int = 1
comptime CT_BOOL_FALSE: Int = 2
comptime CT_BYTE: Int = 3
comptime CT_I16: Int = 4
comptime CT_I32: Int = 5
comptime CT_I64: Int = 6
comptime CT_DOUBLE: Int = 7
comptime CT_BINARY: Int = 8
comptime CT_LIST: Int = 9
comptime CT_SET: Int = 10
comptime CT_MAP: Int = 11
comptime CT_STRUCT: Int = 12


@fieldwise_init
struct FieldHdr(Copyable, Movable):
    var ctype: Int  # compact type; CT_STOP marks end of struct
    var fid: Int    # absolute field id


@fieldwise_init
struct ListHdr(Copyable, Movable):
    var etype: Int  # element compact type
    var size: Int   # element count


def th_read_byte(data: List[UInt8], mut pos: Int) raises -> Int:
    if pos >= len(data):
        raise Error("thrift: eof reading byte")
    var b = Int(data[pos])
    pos += 1
    return b


def th_read_varint(data: List[UInt8], mut pos: Int) raises -> Int:
    var value = 0
    var shift = 0
    while True:
        if pos >= len(data):
            raise Error("thrift: truncated varint")
        var b = Int(data[pos])
        pos += 1
        value |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
    return value


def th_read_zigzag(data: List[UInt8], mut pos: Int) raises -> Int:
    var u = th_read_varint(data, pos)
    return (u >> 1) ^ (-(u & 1))


def th_field_header(data: List[UInt8], mut pos: Int, mut last_id: Int) raises -> FieldHdr:
    var b = th_read_byte(data, pos)
    var ctype = b & 0x0F
    if ctype == CT_STOP:
        return FieldHdr(CT_STOP, 0)
    var delta = (b >> 4) & 0x0F
    var fid: Int
    if delta == 0:
        fid = th_read_zigzag(data, pos)  # long-form field id
    else:
        fid = last_id + delta
    last_id = fid
    return FieldHdr(ctype, fid)


def th_list_header(data: List[UInt8], mut pos: Int) raises -> ListHdr:
    var b = th_read_byte(data, pos)
    var size = (b >> 4) & 0x0F
    var etype = b & 0x0F
    if size == 15:
        size = th_read_varint(data, pos)
    return ListHdr(etype, size)


def th_read_binary(data: List[UInt8], mut pos: Int) raises -> List[UInt8]:
    var n = th_read_varint(data, pos)
    if pos + n > len(data):
        raise Error("thrift: binary overruns buffer")
    var out = List[UInt8](capacity=n)
    for i in range(n):
        out.append(data[pos + i])
    pos += n
    return out^


def th_read_string(data: List[UInt8], mut pos: Int) raises -> String:
    var b = th_read_binary(data, pos)
    var s = String("")
    for i in range(len(b)):
        s += chr(Int(b[i]))
    return s


def th_skip(data: List[UInt8], mut pos: Int, ctype: Int) raises:
    """Advance the cursor past one value of the given compact type."""
    if ctype == CT_BOOL_TRUE or ctype == CT_BOOL_FALSE:
        return  # value lives in the type nibble — no payload
    elif ctype == CT_BYTE:
        pos += 1
    elif ctype == CT_I16 or ctype == CT_I32 or ctype == CT_I64:
        _ = th_read_varint(data, pos)  # zigzag varint — width doesn't matter to skip
    elif ctype == CT_DOUBLE:
        pos += 8
    elif ctype == CT_BINARY:
        var n = th_read_varint(data, pos)
        pos += n
    elif ctype == CT_LIST or ctype == CT_SET:
        var lh = th_list_header(data, pos)
        for _ in range(lh.size):
            th_skip(data, pos, lh.etype)
    elif ctype == CT_MAP:
        var sz = th_read_varint(data, pos)
        if sz != 0:
            var kv = th_read_byte(data, pos)
            var kt = (kv >> 4) & 0x0F
            var vt = kv & 0x0F
            for _ in range(sz):
                th_skip(data, pos, kt)
                th_skip(data, pos, vt)
    elif ctype == CT_STRUCT:
        var last = 0
        while True:
            var fh = th_field_header(data, pos, last)
            if fh.ctype == CT_STOP:
                break
            th_skip(data, pos, fh.ctype)
    else:
        raise Error("thrift: cannot skip unknown compact type " + String(ctype))
