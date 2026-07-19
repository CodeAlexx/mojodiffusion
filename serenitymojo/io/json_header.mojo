# json_header.mojo — hand-rolled minimal JSON parser for the FLAT safetensors
# header schema ONLY. No general JSON, NO std.python (PLAN.md forbids Python in
# the runtime path).
#
# Schema (exactly):
#   {"<name>":{"dtype":"<str>","shape":[<ints>],"data_offsets":[<int>,<int>]},
#    ..., "__metadata__":{...}}
#
# We parse the top-level object key-by-key. For each non-__metadata__ key, the
# value is an object with keys dtype/shape/data_offsets (any order, whitespace
# tolerated). The "__metadata__" key's value is an arbitrary object which we
# skip by balanced-brace scanning.
#
# Notes mirrored from the task spec:
#   * tensor names contain '.', '/', '-', digits — handled by generic string
#     scanning (we read everything between unescaped quotes).
#   * data_offsets / shape values are 64-bit; Mojo Int is 64-bit here.
#   * basic JSON string escapes (\" \\ \/ \n \t \r \b \f \uXXXX) are handled in
#     string scanning so tensor names with escapes don't break the parser.


@fieldwise_init
struct HeaderEntry(Copyable, Movable):
    """One parsed tensor entry from the header."""

    var name: String
    var dtype: String
    var shape: List[Int]
    var off_start: Int
    var off_end: Int


struct _Cursor:
    """Byte cursor over the header bytes."""

    var data: List[UInt8]
    var pos: Int
    var n: Int

    def __init__(out self, var data: List[UInt8]):
        self.n = len(data)
        self.data = data^
        self.pos = 0

    def peek(self) -> Int:
        """Current byte, or -1 at end."""
        if self.pos >= self.n:
            return -1
        return Int(self.data[self.pos])

    def at_end(self) -> Bool:
        return self.pos >= self.n

    def advance(mut self):
        self.pos += 1

    def skip_ws(mut self):
        while self.pos < self.n:
            var c = Int(self.data[self.pos])
            # space, tab, newline, carriage return
            if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D:
                self.pos += 1
            else:
                break

    def expect(mut self, ch: Int) raises:
        self.skip_ws()
        if self.peek() != ch:
            raise Error(
                String("JSON parse: expected '")
                + _chr(ch)
                + "' at byte "
                + String(self.pos)
            )
        self.advance()


def _chr(c: Int) raises -> String:
    """Single-byte Int -> String (ASCII)."""
    var b = List[UInt8]()
    b.append(UInt8(c))
    return String(from_utf8=b)


def _parse_string(mut cur: _Cursor) raises -> String:
    """Parse a JSON string starting at the opening quote. Handles escapes.
    Returns the decoded (UTF-8) string."""
    cur.skip_ws()
    if cur.peek() != 0x22:  # '"'
        raise Error(
            String("JSON parse: expected string at byte ") + String(cur.pos)
        )
    cur.advance()  # consume opening quote
    var out = List[UInt8]()
    while not cur.at_end():
        var c = Int(cur.data[cur.pos])
        cur.pos += 1
        if c == 0x22:  # closing '"'
            return String(from_utf8=out)
        if c == 0x5C:  # backslash escape
            if cur.at_end():
                raise Error("JSON parse: dangling escape")
            var e = Int(cur.data[cur.pos])
            cur.pos += 1
            if e == 0x22:  # \"
                out.append(0x22)
            elif e == 0x5C:  # \\
                out.append(0x5C)
            elif e == 0x2F:  # \/
                out.append(0x2F)
            elif e == 0x6E:  # \n
                out.append(0x0A)
            elif e == 0x74:  # \t
                out.append(0x09)
            elif e == 0x72:  # \r
                out.append(0x0D)
            elif e == 0x62:  # \b
                out.append(0x08)
            elif e == 0x66:  # \f
                out.append(0x0C)
            elif e == 0x75:  # \uXXXX
                # Decode 4 hex digits to a code point, emit as UTF-8.
                if cur.pos + 4 > cur.n:
                    raise Error("JSON parse: truncated \\u escape")
                var cp = 0
                for _i in range(4):
                    cp = cp * 16 + _hex_val(Int(cur.data[cur.pos]))
                    cur.pos += 1
                _emit_utf8(out, cp)
            else:
                raise Error(
                    String("JSON parse: bad escape \\")
                    + _chr(e)
                )
        else:
            out.append(UInt8(c))
    raise Error("JSON parse: unterminated string")


def _hex_val(c: Int) raises -> Int:
    if c >= 0x30 and c <= 0x39:  # 0-9
        return c - 0x30
    if c >= 0x41 and c <= 0x46:  # A-F
        return c - 0x41 + 10
    if c >= 0x61 and c <= 0x66:  # a-f
        return c - 0x61 + 10
    raise Error("JSON parse: bad hex digit")


def _emit_utf8(mut out: List[UInt8], cp: Int):
    """Encode a Unicode code point as UTF-8 bytes (BMP-only here; safetensors
    tensor names are effectively ASCII, this is defensive)."""
    if cp < 0x80:
        out.append(UInt8(cp))
    elif cp < 0x800:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))


def _parse_int(mut cur: _Cursor) raises -> Int:
    """Parse a non-negative JSON integer (offsets/shape dims are >= 0)."""
    cur.skip_ws()
    var start = cur.pos
    var val = 0
    var any = False
    while not cur.at_end():
        var c = Int(cur.data[cur.pos])
        if c >= 0x30 and c <= 0x39:
            val = val * 10 + (c - 0x30)
            cur.pos += 1
            any = True
        else:
            break
    if not any:
        raise Error(
            String("JSON parse: expected integer at byte ") + String(start)
        )
    return val


def _parse_int_array(mut cur: _Cursor) raises -> List[Int]:
    """Parse a JSON array of non-negative integers: [a, b, c] (possibly []))."""
    var out = List[Int]()
    cur.expect(0x5B)  # '['
    cur.skip_ws()
    if cur.peek() == 0x5D:  # ']' empty
        cur.advance()
        return out^
    while True:
        out.append(_parse_int(cur))
        cur.skip_ws()
        var c = cur.peek()
        if c == 0x2C:  # ','
            cur.advance()
            continue
        if c == 0x5D:  # ']'
            cur.advance()
            break
        raise Error(
            String("JSON parse: expected ',' or ']' in array at byte ")
            + String(cur.pos)
        )
    return out^


def _skip_value(mut cur: _Cursor) raises:
    """Skip an arbitrary JSON value (used for the __metadata__ object). Handles
    objects, arrays, strings, numbers, true/false/null via balanced scanning."""
    cur.skip_ws()
    var c = cur.peek()
    if c == 0x22:  # string
        _ = _parse_string(cur)
        return
    if c == 0x7B or c == 0x5B:  # object '{' or array '['
        var open_ch = c
        var close_ch = 0x7D if c == 0x7B else 0x5D
        var depth = 0
        while not cur.at_end():
            var ch = Int(cur.data[cur.pos])
            if ch == 0x22:  # nested string — consume properly (skip escapes)
                _ = _parse_string(cur)
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                cur.pos += 1
                if depth == 0:
                    return
                continue
            cur.pos += 1
        raise Error("JSON parse: unbalanced container in __metadata__")
    # number / true / false / null — read until a structural char.
    while not cur.at_end():
        var ch = Int(cur.data[cur.pos])
        if (
            ch == 0x2C
            or ch == 0x7D
            or ch == 0x5D
            or ch == 0x20
            or ch == 0x09
            or ch == 0x0A
            or ch == 0x0D
        ):
            break
        cur.pos += 1


def parse_header(var data: List[UInt8]) raises -> List[HeaderEntry]:
    """Parse a flat safetensors header. Returns one HeaderEntry per tensor,
    skipping the "__metadata__" key. Mirrors mmap.rs:204-235 semantics."""
    var cur = _Cursor(data^)
    var entries = List[HeaderEntry]()

    cur.expect(0x7B)  # top-level '{'
    cur.skip_ws()
    if cur.peek() == 0x7D:  # empty object '}'
        cur.advance()
        return entries^

    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)  # ':'

        if key == "__metadata__":
            _skip_value(cur)  # mmap.rs:207-209 — skip __metadata__
        else:
            # Value is an object {"dtype":..,"shape":..,"data_offsets":..}.
            var dtype = String("")
            var shape = List[Int]()
            var off_start = 0
            var off_end = 0
            var have_offsets = False

            cur.expect(0x7B)  # '{'
            cur.skip_ws()
            if cur.peek() != 0x7D:  # not immediately closed
                while True:
                    var field = _parse_string(cur)
                    cur.expect(0x3A)  # ':'
                    if field == "dtype":
                        dtype = _parse_string(cur)
                    elif field == "shape":
                        shape = _parse_int_array(cur)
                    elif field == "data_offsets":
                        var offs = _parse_int_array(cur)
                        if len(offs) >= 2:
                            off_start = offs[0]
                            off_end = offs[1]
                            have_offsets = True
                    else:
                        # tolerate unknown fields
                        _skip_value(cur)
                    cur.skip_ws()
                    var c = cur.peek()
                    if c == 0x2C:  # ','
                        cur.advance()
                        continue
                    if c == 0x7D:  # '}'
                        break
                    raise Error(
                        String(
                            "JSON parse: expected ',' or '}' in tensor object"
                            " at byte "
                        )
                        + String(cur.pos)
                    )
            cur.expect(0x7D)  # close tensor object '}'

            # mmap.rs defaults: dtype unwrap_or "F32"; offsets unwrap_or (0,0).
            if dtype.byte_length() == 0:
                dtype = String("F32")
            _ = have_offsets
            entries.append(
                HeaderEntry(
                    name=key,
                    dtype=dtype,
                    shape=shape^,
                    off_start=off_start,
                    off_end=off_end,
                )
            )

        # After a top-level value: ',' (more) or '}' (done).
        cur.skip_ws()
        var c = cur.peek()
        if c == 0x2C:  # ','
            cur.advance()
            continue
        if c == 0x7D:  # '}'
            cur.advance()
            break
        raise Error(
            String("JSON parse: expected ',' or '}' at top level at byte ")
            + String(cur.pos)
        )

    return entries^


# ── Unit test (task-required): hand-written header, 2 tensors + __metadata__ ──
def test_parse_sample() raises:
    from std.testing import assert_equal, assert_true
    var s = String(
        '{"__metadata__":{"format":"pt"},'
        '"model.layer.0/weight":{"dtype":"BF16","shape":[2,3],'
        '"data_offsets":[0,12]},'
        '"model.bias-1":{"shape":[4],"dtype":"F32",'
        '"data_offsets":[12,28]}}'
    )
    var bytes = List[UInt8]()
    for b in s.as_bytes():
        bytes.append(b)
    var entries = parse_header(bytes^)
    # __metadata__ skipped -> 2 entries.
    assert_equal(len(entries), 2)

    # entry 0
    assert_equal(entries[0].name, String("model.layer.0/weight"))
    assert_equal(entries[0].dtype, String("BF16"))
    assert_equal(len(entries[0].shape), 2)
    assert_equal(entries[0].shape[0], 2)
    assert_equal(entries[0].shape[1], 3)
    assert_equal(entries[0].off_start, 0)
    assert_equal(entries[0].off_end, 12)

    # entry 1 — fields in a different order (shape before dtype).
    assert_equal(entries[1].name, String("model.bias-1"))
    assert_equal(entries[1].dtype, String("F32"))
    assert_equal(len(entries[1].shape), 1)
    assert_equal(entries[1].shape[0], 4)
    assert_equal(entries[1].off_start, 12)
    assert_equal(entries[1].off_end, 28)
    print("test_parse_sample OK")


def main() raises:
    test_parse_sample()
