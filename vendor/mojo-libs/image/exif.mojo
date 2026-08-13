# image/exif.mojo — EXIF (TIFF/IFD) metadata read + write.
#
# EXIF metadata lives in a TIFF stream: an 8-byte header (byte-order marker +
# magic 42 + offset to IFD0), then one or more IFDs. Each IFD is a count of
# 12-byte entries plus a trailing u32 "next IFD" offset. An entry is
#   tag(u16) type(u16) count(u32) value-or-offset(u32).
# When the typed payload fits in 4 bytes it lives inline in the value field;
# otherwise the value field is an offset (from the TIFF start) to the payload.
#
# We parse IFD0 and follow the ExifIFD pointer (tag 0x8769) into the EXIF
# sub-IFD, flattening both into one tag table. We decode the common scalar
# types: BYTE(1), ASCII(2), SHORT(3), LONG(4), RATIONAL(5), SRATIONAL(10).
#
# build_exif() writes a minimal little-endian TIFF/EXIF blob (Make, Model,
# Orientation, DateTime in IFD0) that round-trips through parse_exif and is
# readable by PIL's Image.Exif().load().
#
# Byte order: TIFF is byte-order-flagged — "II" = little-endian, "MM" = big-
# endian. parse_exif handles BOTH. build_exif emits little-endian.
#
# NOT parsed: GPS IFD (0x8825), Interoperability IFD (0xA005), MakerNote
# (0x927C is opaque), thumbnail IFD1, UNDEFINED(7)/FLOAT(11)/DOUBLE(12) types.

from std.memory import alloc

# ---- well-known tags ----
comptime TAG_MAKE: Int = 0x010F
comptime TAG_MODEL: Int = 0x0110
comptime TAG_ORIENTATION: Int = 0x0112
comptime TAG_DATETIME: Int = 0x0132
comptime TAG_EXIF_IFD: Int = 0x8769
comptime TAG_EXPOSURE_TIME: Int = 0x829A
comptime TAG_FNUMBER: Int = 0x829D
comptime TAG_ISO: Int = 0x8827
comptime TAG_FOCAL_LENGTH: Int = 0x920A
comptime TAG_PIXEL_X: Int = 0xA002
comptime TAG_PIXEL_Y: Int = 0xA003

# ---- TIFF field types ----
comptime TYPE_BYTE: Int = 1
comptime TYPE_ASCII: Int = 2
comptime TYPE_SHORT: Int = 3
comptime TYPE_LONG: Int = 4
comptime TYPE_RATIONAL: Int = 5
comptime TYPE_SBYTE: Int = 6
comptime TYPE_UNDEFINED: Int = 7
comptime TYPE_SSHORT: Int = 8
comptime TYPE_SLONG: Int = 9
comptime TYPE_SRATIONAL: Int = 10


def _type_size(t: Int) -> Int:
    """Bytes per component for a TIFF field type (0 = unknown/unsupported)."""
    if t == TYPE_BYTE or t == TYPE_ASCII or t == TYPE_SBYTE or t == TYPE_UNDEFINED:
        return 1
    if t == TYPE_SHORT or t == TYPE_SSHORT:
        return 2
    if t == TYPE_LONG or t == TYPE_SLONG:
        return 4
    if t == TYPE_RATIONAL or t == TYPE_SRATIONAL:
        return 8
    return 0


@fieldwise_init
struct Rational(Copyable, Movable):
    """A rational value: num/den. Used for RATIONAL/SRATIONAL fields."""
    var num: Int
    var den: Int

    def as_float(self) -> Float64:
        if self.den == 0:
            return 0.0
        return Float64(self.num) / Float64(self.den)


@fieldwise_init
struct ExifEntry(Copyable, Movable):
    """One decoded IFD entry, payload kept as raw bytes plus type/count so the
    accessors can re-interpret it on demand."""
    var tag: Int
    var type: Int
    var count: Int
    var data: List[UInt8]   # raw payload bytes (little-endian-normalized? no — kept in source order)
    var little: Bool        # byte order this payload was read in


struct ExifData(Movable):
    """Parsed EXIF: a flat list of entries from IFD0 + the EXIF sub-IFD."""
    var entries: List[ExifEntry]
    var little: Bool        # byte order of the source stream

    def __init__(out self):
        self.entries = List[ExifEntry]()
        self.little = True

    def __init__(out self, *, copy: Self):
        self.entries = copy.entries.copy()
        self.little = copy.little

    def __init__(out self, *, deinit move: Self):
        self.entries = move.entries^
        self.little = move.little

    # ---- low-level lookups ----
    def has(self, tag: Int) -> Bool:
        for ref e in self.entries:
            if e.tag == tag:
                return True
        return False

    def _find(self, tag: Int) raises -> Int:
        for i in range(len(self.entries)):
            if self.entries[i].tag == tag:
                return i
        raise Error("exif: tag not present: " + String(tag))

    # ---- typed accessors ----
    def get_int(self, tag: Int) raises -> Int:
        """First component of a BYTE/SHORT/LONG (or SBYTE/SSHORT/SLONG) field."""
        ref e = self.entries[self._find(tag)]
        return _read_int(e.data, 0, e.type, e.little)

    def get_ascii(self, tag: Int) raises -> String:
        """An ASCII field, NUL terminator stripped."""
        ref e = self.entries[self._find(tag)]
        var out = String("")
        for i in range(len(e.data)):
            var c = e.data[i]
            if c == 0:
                break
            out += chr(Int(c))
        return out

    def get_rational(self, tag: Int) raises -> Rational:
        """First component of a RATIONAL/SRATIONAL field."""
        ref e = self.entries[self._find(tag)]
        if e.type != TYPE_RATIONAL and e.type != TYPE_SRATIONAL:
            raise Error("exif: tag " + String(tag) + " is not rational")
        var num = _read_u32(e.data, 0, e.little)
        var den = _read_u32(e.data, 4, e.little)
        if e.type == TYPE_SRATIONAL:
            return Rational(_to_signed32(num), _to_signed32(den))
        return Rational(Int(num), Int(den))

    # ---- named helpers ----
    def make(self) raises -> String:
        return self.get_ascii(TAG_MAKE)

    def model(self) raises -> String:
        return self.get_ascii(TAG_MODEL)

    def orientation(self) raises -> Int:
        return self.get_int(TAG_ORIENTATION)

    def datetime(self) raises -> String:
        return self.get_ascii(TAG_DATETIME)

    def exposure_time(self) raises -> Rational:
        return self.get_rational(TAG_EXPOSURE_TIME)

    def fnumber(self) raises -> Rational:
        return self.get_rational(TAG_FNUMBER)

    def iso(self) raises -> Int:
        return self.get_int(TAG_ISO)

    def focal_length(self) raises -> Rational:
        return self.get_rational(TAG_FOCAL_LENGTH)

    def pixel_x_dimension(self) raises -> Int:
        return self.get_int(TAG_PIXEL_X)

    def pixel_y_dimension(self) raises -> Int:
        return self.get_int(TAG_PIXEL_Y)


# ---------------------------------------------------------------------------
# byte readers (byte-order aware)
# ---------------------------------------------------------------------------
def _read_u16(d: List[UInt8], off: Int, little: Bool) raises -> Int:
    if off + 2 > len(d):
        raise Error("exif: u16 read out of range")
    var a = Int(d[off])
    var b = Int(d[off + 1])
    if little:
        return a | (b << 8)
    return (a << 8) | b


def _read_u32(d: List[UInt8], off: Int, little: Bool) raises -> Int:
    if off + 4 > len(d):
        raise Error("exif: u32 read out of range")
    var a = Int(d[off])
    var b = Int(d[off + 1])
    var c = Int(d[off + 2])
    var e = Int(d[off + 3])
    if little:
        return a | (b << 8) | (c << 16) | (e << 24)
    return (a << 24) | (b << 16) | (c << 8) | e


def _to_signed32(v: Int) -> Int:
    if v >= 0x80000000:
        return v - 0x100000000
    return v


def _to_signed16(v: Int) -> Int:
    if v >= 0x8000:
        return v - 0x10000
    return v


def _read_int(d: List[UInt8], off: Int, type: Int, little: Bool) raises -> Int:
    """Read one integer component of the given TIFF type from a payload buffer."""
    if type == TYPE_BYTE or type == TYPE_UNDEFINED:
        return Int(d[off])
    if type == TYPE_SBYTE:
        var v = Int(d[off])
        return v - 256 if v >= 128 else v
    if type == TYPE_SHORT:
        return _read_u16(d, off, little)
    if type == TYPE_SSHORT:
        return _to_signed16(_read_u16(d, off, little))
    if type == TYPE_LONG:
        return _read_u32(d, off, little)
    if type == TYPE_SLONG:
        return _to_signed32(_read_u32(d, off, little))
    raise Error("exif: cannot read tag type " + String(type) + " as int")


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------
def _parse_ifd(
    blob: List[UInt8], ifd_off: Int, little: Bool, mut out: List[ExifEntry]
) raises -> Int:
    """Parse one IFD at ifd_off; append entries to `out`; return the EXIF
    sub-IFD offset if tag 0x8769 was present, else 0."""
    if ifd_off + 2 > len(blob):
        raise Error("exif: IFD offset out of range")
    var n = _read_u16(blob, ifd_off, little)
    var exif_sub = 0
    for i in range(n):
        var e_off = ifd_off + 2 + i * 12
        if e_off + 12 > len(blob):
            raise Error("exif: IFD entry out of range")
        var tag = _read_u16(blob, e_off, little)
        var type = _read_u16(blob, e_off + 2, little)
        var count = _read_u32(blob, e_off + 4, little)
        var tsize = _type_size(type)

        if tag == TAG_EXIF_IFD:
            # value field is the sub-IFD offset (LONG)
            exif_sub = _read_u32(blob, e_off + 8, little)
            continue

        if tsize == 0:
            continue  # unsupported type — skip rather than fail the whole blob

        var total = tsize * count
        var payload = List[UInt8]()
        if total <= 4:
            # inline in the value field (e_off+8 .. e_off+12)
            for k in range(total):
                payload.append(blob[e_off + 8 + k])
        else:
            var data_off = _read_u32(blob, e_off + 8, little)
            if data_off + total > len(blob):
                raise Error("exif: value offset out of range for tag " + String(tag))
            for k in range(total):
                payload.append(blob[data_off + k])
        out.append(ExifEntry(tag, type, count, payload^, little))
    return exif_sub


def parse_exif(blob: List[UInt8]) raises -> ExifData:
    """Parse a raw TIFF/EXIF stream (the bytes PIL exposes as img.info['exif']
    or Image.getexif().tobytes())."""
    if len(blob) < 8:
        raise Error("exif: blob too short for TIFF header")
    # Some sources (PIL's getexif().tobytes(), a JPEG APP1 segment) prepend a
    # 6-byte "Exif\0\0" marker before the TIFF stream. Skip it if present.
    var start = 0
    if (len(blob) >= 14 and blob[0] == UInt8(ord("E")) and blob[1] == UInt8(ord("x"))
            and blob[2] == UInt8(ord("i")) and blob[3] == UInt8(ord("f"))
            and blob[4] == UInt8(0) and blob[5] == UInt8(0)):
        start = 6
    # Re-base the blob to the TIFF start so all offsets resolve correctly.
    var tiff: List[UInt8]
    if start == 0:
        tiff = blob.copy()
    else:
        tiff = List[UInt8]()
        for i in range(start, len(blob)):
            tiff.append(blob[i])
    return _parse_tiff(tiff)


def _parse_tiff(blob: List[UInt8]) raises -> ExifData:
    var little: Bool
    if blob[0] == UInt8(ord("I")) and blob[1] == UInt8(ord("I")):
        little = True
    elif blob[0] == UInt8(ord("M")) and blob[1] == UInt8(ord("M")):
        little = False
    else:
        raise Error("exif: bad byte-order marker (not II/MM)")
    var magic = _read_u16(blob, 2, little)
    if magic != 42:
        raise Error("exif: bad TIFF magic (expected 42, got " + String(magic) + ")")
    var ifd0 = _read_u32(blob, 4, little)

    var result = ExifData()
    result.little = little
    var exif_sub = _parse_ifd(blob, ifd0, little, result.entries)
    if exif_sub != 0 and exif_sub + 2 <= len(blob):
        _ = _parse_ifd(blob, exif_sub, little, result.entries)
    return result^


# ---------------------------------------------------------------------------
# build (little-endian writer)
# ---------------------------------------------------------------------------
def _put_u16(mut buf: List[UInt8], v: Int):
    buf.append(UInt8(v & 0xFF))
    buf.append(UInt8((v >> 8) & 0xFF))


def _put_u32(mut buf: List[UInt8], v: Int):
    buf.append(UInt8(v & 0xFF))
    buf.append(UInt8((v >> 8) & 0xFF))
    buf.append(UInt8((v >> 16) & 0xFF))
    buf.append(UInt8((v >> 24) & 0xFF))


def _set_u32(mut buf: List[UInt8], off: Int, v: Int):
    buf[off] = UInt8(v & 0xFF)
    buf[off + 1] = UInt8((v >> 8) & 0xFF)
    buf[off + 2] = UInt8((v >> 16) & 0xFF)
    buf[off + 3] = UInt8((v >> 24) & 0xFF)


def build_exif(
    Make: String = "MojoCam",
    Model: String = "X1",
    Orientation: Int = 1,
    DateTime: String = "2026:06:10 12:00:00",
) raises -> List[UInt8]:
    """Write a minimal little-endian TIFF/EXIF blob with Make, Model,
    Orientation (SHORT) and DateTime (ASCII) in IFD0. Round-trips through
    parse_exif and is readable by PIL's Image.Exif().load()."""
    # ASCII fields must be NUL-terminated.
    var make_b = Make + String("\0")
    var model_b = Model + String("\0")
    var dt_b = DateTime + String("\0")

    var make_len = len(make_b)
    var model_len = len(model_b)
    var dt_len = len(dt_b)

    # Layout: header(8) + IFD. IFD = count(2) + N*12 + next(4).
    # Entries (sorted by tag, required by TIFF): Make(0x010F), Model(0x0110),
    # Orientation(0x0112), DateTime(0x0132). ASCII payloads >4 bytes follow the
    # IFD out-of-line; <=4-byte ASCII is stored inline in the value field.
    var n_entries = 4
    var ifd_off = 8
    var ifd_size = 2 + n_entries * 12 + 4
    var data_start = ifd_off + ifd_size

    # offsets for the out-of-line ASCII payloads, assigned in entry order so the
    # payload region matches the offsets we write. <=4-byte fields are inline
    # and consume no data-region space.
    var data_cursor = data_start
    var make_off = 0
    var make_inline = make_len <= 4
    if not make_inline:
        make_off = data_cursor
        data_cursor += make_len
    var dt_off = 0
    var dt_inline = dt_len <= 4
    if not dt_inline:
        dt_off = data_cursor
        data_cursor += dt_len

    var buf = List[UInt8]()
    # --- TIFF header (little-endian) ---
    buf.append(UInt8(ord("I")))
    buf.append(UInt8(ord("I")))
    _put_u16(buf, 42)
    _put_u32(buf, ifd_off)

    # --- IFD0 ---
    _put_u16(buf, n_entries)
    _write_ascii_entry(buf, TAG_MAKE, make_b, make_off)
    _write_ascii_entry(buf, TAG_MODEL, model_b, 0)  # Model is short -> inline
    # Orientation (SHORT, inline). count=1, value in low 2 bytes of the field.
    _put_u16(buf, TAG_ORIENTATION)
    _put_u16(buf, TYPE_SHORT)
    _put_u32(buf, 1)
    _put_u16(buf, Orientation)
    _put_u16(buf, 0)  # pad the value field to 4 bytes
    _write_ascii_entry(buf, TAG_DATETIME, dt_b, dt_off)
    # next-IFD offset = 0 (no IFD1)
    _put_u32(buf, 0)

    # --- out-of-line ASCII payloads (only those >4 bytes), in offset order ---
    if not make_inline:
        var make_bytes = make_b.as_bytes()
        for i in range(make_len):
            buf.append(make_bytes[i])
    if not dt_inline:
        var dt_bytes = dt_b.as_bytes()
        for i in range(dt_len):
            buf.append(dt_bytes[i])

    return buf^


def _write_ascii_entry(mut buf: List[UInt8], tag: Int, s: String, offset: Int):
    """Write a 12-byte ASCII IFD entry. If the payload is <=4 bytes it is stored
    inline in the value field; otherwise `offset` is written as the data offset."""
    var n = len(s)
    _put_u16(buf, tag)
    _put_u16(buf, TYPE_ASCII)
    _put_u32(buf, n)
    if n <= 4:
        var sb = s.as_bytes()
        for i in range(4):
            if i < n:
                buf.append(sb[i])
            else:
                buf.append(UInt8(0))
    else:
        _put_u32(buf, offset)
