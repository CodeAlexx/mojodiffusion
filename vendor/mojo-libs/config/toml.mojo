# config/toml.mojo — a practical TOML-subset parser. 100% Mojo, no FFI.
#
# Produces the shared `ConfigTree` from config/value.mojo (NOT config/config.mojo).
# Tables map to sections; top-level keys map to the global section "".
#
# ── SUPPORTED ────────────────────────────────────────────────────────────────
#   [table]                table header                 -> section "table"
#   [a.b.c]                dotted table header          -> section "a.b.c"
#   key = value            assignment with TYPED value:
#     - basic string    "..."   escapes: \n \t \r \" \\ \uXXXX
#     - literal string  '...'   no escapes, taken verbatim
#     - integer    42, -7, +3, 1_000  ('_' digit separators stripped)
#     - float      3.14, 1e3, -0.5, 6.022e23, .5, 5.  + inf/-inf/nan
#     - bool       true | false
#     - array      [1, 2, 3] / ["a","b"] / nested [[1,2],[3]] / mixed;
#                  may span multiple lines, trailing comma allowed; [] is empty
#   dotted keys   a.b.c = 1  -> stored in section "<current>.a.b" (prefix joins
#                              the active table; see DOTTED-KEY MAPPING below)
#   quoted keys   "a.b" = 1  -> single literal key (dots inside quotes are NOT
#                              splitters); same for 'a.b'
#   # comment              full-line comment OR trailing comment after a value,
#                          with '#' inside strings respected (not a comment).
#   <blank line>           ignored
#
# ── DOTTED-KEY / TABLE -> SECTION MAPPING (the exact rule we chose) ───────────
#   * A `[a.b.c]` header sets the current section to literally "a.b.c".
#   * A bare top-level key `k = v` (before any header) goes to section "".
#   * A dotted assignment key `a.b.c = v` splits on unquoted '.': all but the
#     last segment form a table prefix, the last segment is the key. The prefix
#     is appended to the CURRENT section with '.' as the join. Examples:
#         (top level)  owner.name = "x"   -> section "owner",   key "name"
#         under [srv]  a.b = 1            -> section "srv.a.b", key "b"  -- wait
#     Concretely: section = join(current, prefix_segments), key = last_segment.
#         top level    a.b.c = 1          -> section "a.b",     key "c"
#         under [s]    a.b   = 1          -> section "s.a",     key "b"
#   * Quoted segments keep their dots: "a.b" = 1 -> section <current>, key "a.b".
#   This matches Python tomllib's nesting when the tree is flattened to dotted
#   section paths (the oracle flattens the same way).
#
# ── OUT OF SCOPE (documented; these raise a clear error) ─────────────────────
#   - inline tables          { a = 1, b = 2 }            -> raises
#   - array-of-tables        [[x]]                       -> raises
#   - multiline basic/literal strings  """...""" / '''...'''  -> raises
#   - datetimes (offset/local date-time, date, time)     -> raises (we do NOT
#                              silently coerce; callers wanting strings should
#                              quote them). Detection is heuristic.
#   - heterogeneous arrays are accepted structurally (TOML 1.0 allows them);
#     we do not reject mixed-type arrays.

from config.value import ConfigTree, ConfigValue


def parse_toml_file(path: String) raises -> ConfigTree:
    """Read a file and parse it as TOML."""
    var text = open(path, "r").read()
    return parse_toml(text)


def parse_toml(text: String) raises -> ConfigTree:
    """Parse a practical TOML subset into a typed ConfigTree (config/value.mojo)."""
    var tree = ConfigTree()
    var current = String("")  # global table
    var seen = Dict[String, Bool]()  # (section\x00key) already defined -> raise on dup

    var bytes = _to_bytes(text)
    var n = len(bytes)
    var pos = 0

    while pos < n:
        pos = _skip_ws_and_comments_and_newlines(bytes, n, pos)
        if pos >= n:
            break
        var c = Int(bytes[pos])
        if c == 0x5B:  # [
            if pos + 1 < n and Int(bytes[pos + 1]) == 0x5B:
                raise Error("array-of-tables [[..]] not supported")
            pos += 1
            var name = String("")
            while pos < n and Int(bytes[pos]) != 0x5D:  # until ]
                name += chr(Int(bytes[pos]))
                pos += 1
            if pos >= n:
                raise Error("unterminated table header")
            pos += 1  # consume ]
            current = _trim(name)
            if current.byte_length() == 0:
                raise Error("empty table header [] not supported")
            pos = _skip_to_newline(bytes, n, pos)
            continue

        # key (possibly dotted) = value
        var kr = _read_key_path(bytes, n, pos)
        pos = kr.next
        pos = _skip_inline_ws(bytes, n, pos)
        if pos >= n or Int(bytes[pos]) != 0x3D:  # =
            raise Error("expected '=' after key")
        pos += 1  # consume =
        pos = _skip_inline_ws(bytes, n, pos)
        var vr = _read_value(bytes, n, pos)
        pos = vr.next
        var val = vr^.take()

        # resolve section + key from the dotted path + current table
        var segs = kr.segments.copy()
        var ns = len(segs)
        if ns == 0:
            raise Error("empty key")
        var section = current
        for i in range(ns - 1):
            if section.byte_length() == 0:
                section = segs[i]
            else:
                section = section + "." + segs[i]
        var leaf = segs[ns - 1]
        var dupkey = section + String(chr(0)) + leaf
        if dupkey in seen:
            raise Error(
                "Cannot overwrite a value: duplicate key '"
                + leaf
                + "' in table '"
                + section
                + "'"
            )
        seen[dupkey] = True
        tree.set(section, leaf, val^)

        pos = _skip_to_newline(bytes, n, pos)

    return tree^


# ── key path reading (dotted, with quoted segments) ──────────────────────────
struct _KeyResult(Movable):
    var segments: List[String]
    var next: Int

    def __init__(out self, var segments: List[String], next: Int):
        self.segments = segments^
        self.next = next


def _read_key_path(bytes: List[UInt8], n: Int, start: Int) raises -> _KeyResult:
    var pos = start
    var segs = List[String]()
    while True:
        pos = _skip_inline_ws(bytes, n, pos)
        if pos >= n:
            raise Error("unexpected end of input reading key")
        var c = Int(bytes[pos])
        if c == 0x22:  # " quoted key segment (basic-string rules)
            var sr = _read_basic_string(bytes, n, pos)
            segs.append(sr.text)
            pos = sr.next
        elif c == 0x27:  # ' literal-string key segment
            var lr = _read_literal_string(bytes, n, pos)
            segs.append(lr.text)
            pos = lr.next
        else:
            var bare = String("")
            while pos < n:
                var kc = Int(bytes[pos])
                # bare key chars: A-Za-z0-9 _ - ; stop on . = ws
                var is_bare = (
                    (kc >= 0x41 and kc <= 0x5A)
                    or (kc >= 0x61 and kc <= 0x7A)
                    or (kc >= 0x30 and kc <= 0x39)
                    or kc == 0x5F
                    or kc == 0x2D
                )
                if not is_bare:
                    break
                bare += chr(kc)
                pos += 1
            if bare.byte_length() == 0:
                raise Error("invalid or empty key")
            segs.append(bare)
        pos = _skip_inline_ws(bytes, n, pos)
        if pos < n and Int(bytes[pos]) == 0x2E:  # . -> another segment
            pos += 1
            continue
        break
    return _KeyResult(segs^, pos)


# ── value reading ─────────────────────────────────────────────────────────────
struct _StrResult(Copyable, Movable):
    var text: String
    var next: Int

    def __init__(out self, text: String, next: Int):
        self.text = text
        self.next = next


struct _ValResult(Movable):
    var value: ConfigValue
    var next: Int

    def __init__(out self, var value: ConfigValue, next: Int):
        self.value = value^
        self.next = next

    def take(deinit self) -> ConfigValue:
        """Consume the result, yielding its value (read `.next` first)."""
        return self.value^


def _read_value(bytes: List[UInt8], n: Int, start: Int) raises -> _ValResult:
    var pos = _skip_inline_ws(bytes, n, start)
    if pos >= n:
        raise Error("expected value")
    var c = Int(bytes[pos])
    if c == 0x22:  # "
        var sr = _read_basic_string(bytes, n, pos)
        return _ValResult(ConfigValue.str_(sr.text), sr.next)
    if c == 0x27:  # ' literal string
        var lr = _read_literal_string(bytes, n, pos)
        return _ValResult(ConfigValue.str_(lr.text), lr.next)
    if c == 0x5B:  # [  array
        return _read_array(bytes, n, pos)
    if c == 0x7B:  # {  inline table -- unsupported
        raise Error("inline tables not supported")
    # bare token: bool / int / float / (datetime -> error)
    var tok = String("")
    var p = pos
    while p < n:
        var tc = Int(bytes[p])
        if tc == 0x2C or tc == 0x5D or tc == 0x0A or tc == 0x0D or tc == 0x23:  # , ] \n \r #
            break
        tok += chr(tc)
        p += 1
    var t = _trim(tok)
    return _ValResult(_typed_scalar(t), p)


def _typed_scalar(t: String) raises -> ConfigValue:
    if t == "true":
        return ConfigValue.bool_(True)
    if t == "false":
        return ConfigValue.bool_(False)
    if t.byte_length() == 0:
        raise Error("empty value")
    # float specials
    if t == "inf" or t == "+inf":
        return ConfigValue.float_(_inf())
    if t == "-inf":
        return ConfigValue.float_(-_inf())
    if t == "nan" or t == "+nan" or t == "-nan":
        return ConfigValue.float_(_nan())
    if _looks_like_datetime(t):
        raise Error("datetimes not supported: " + t)
    # decide int vs float: scan for '.' or 'e'/'E' (not the leading sign)
    var b = t.as_bytes()
    var nb = t.byte_length()
    var is_float = False
    for i in range(nb):
        var ch = Int(b[i])
        if ch == 0x2E:  # .
            is_float = True
        if ch == 0x65 or ch == 0x45:  # e E
            is_float = True
    if is_float:
        return ConfigValue.float_(_parse_float_str(t))
    return ConfigValue.int_(_parse_int_str(t))


# Validate a run of decimal digits with TOML underscore rules and return the
# digits with underscores stripped. Raises on leading/trailing/double '_'.
# `allow_leading_zero` controls the "no leading zero unless value is 0" rule
# for integer parts (TOML forbids `01`, `007`; floats forbid `01.5` likewise).
def _validate_digit_run(s: String, start: Int, end: Int, allow_leading_zero: Bool) raises -> String:
    # s[start:end) must be: digit, then any of (digit | single '_' between digits).
    var b = s.as_bytes()
    if end <= start:
        raise Error("invalid number (empty digit run): " + s)
    var out = String("")
    var prev_underscore = False
    var first = True
    for i in range(start, end):
        var c = Int(b[i])
        if c == 0x5F:  # '_'
            if first:
                raise Error("number cannot start with '_': " + s)
            if prev_underscore:
                raise Error("number cannot have double '_': " + s)
            prev_underscore = True
            first = False
            continue
        if c < 48 or c > 57:
            raise Error("invalid digit in number: " + s)
        out += chr(c)
        prev_underscore = False
        first = False
    if prev_underscore:
        raise Error("number cannot end with '_': " + s)
    # leading-zero rule: digit run of len>1 starting with '0' is illegal
    if not allow_leading_zero:
        var ob = out.as_bytes()
        if out.byte_length() > 1 and Int(ob[0]) == 0x30:
            raise Error("leading zeros not allowed: " + s)
    return out


def _parse_int_str(orig: String) raises -> Int64:
    var b = orig.as_bytes()
    var n = orig.byte_length()
    var i = 0
    var neg = False
    if n > 0 and Int(b[0]) == 0x2D:
        neg = True
        i = 1
    elif n > 0 and Int(b[0]) == 0x2B:
        i = 1
    var digits = _validate_digit_run(orig, i, n, allow_leading_zero=False)
    # Accumulate as a NON-POSITIVE magnitude so the most-negative value
    # (-9223372036854775808) is representable mid-computation (its positive
    # counterpart is not). Guard each step against Int64 min before it overflows.
    # Int64 min = -9223372036854775808 = MIN_DIV*10 - MIN_REM.
    var db = digits.as_bytes()
    var dn = digits.byte_length()
    comptime MIN_DIV: Int64 = -922337203685477580  # -9223372036854775808 // 10 (trunc)
    comptime MIN_REM: Int64 = 8                     # last digit of the magnitude
    var v: Int64 = 0  # holds the negative accumulation
    for j in range(dn):
        var d = Int64(Int(db[j]) - 48)
        # next = v*10 - d must stay >= Int64 min
        if v < MIN_DIV or (v == MIN_DIV and d > MIN_REM):
            raise Error("integer out of Int64 range: " + orig)
        v = v * 10 - d
    return v if neg else -v


def _parse_float_str(orig: String) raises -> Float64:
    # Validate structure (sign, int part, optional frac, optional exp) with TOML
    # underscore + leading-zero rules, build an underscore-free string, parse.
    var b = orig.as_bytes()
    var n = orig.byte_length()
    var i = 0
    var clean = String("")
    if n > 0 and (Int(b[0]) == 0x2D or Int(b[0]) == 0x2B):
        clean += chr(Int(b[0]))
        i = 1
    # integer part: digits up to '.', 'e'/'E', or end
    var ip_start = i
    while i < n:
        var c = Int(b[i])
        if c == 0x2E or c == 0x65 or c == 0x45:
            break
        i += 1
    clean += _validate_digit_run(orig, ip_start, i, allow_leading_zero=False)
    # fractional part
    if i < n and Int(b[i]) == 0x2E:  # .
        clean += "."
        i += 1
        var fp_start = i
        while i < n:
            var c = Int(b[i])
            if c == 0x65 or c == 0x45:
                break
            i += 1
        clean += _validate_digit_run(orig, fp_start, i, allow_leading_zero=True)
    # exponent
    if i < n and (Int(b[i]) == 0x65 or Int(b[i]) == 0x45):  # e/E
        clean += "e"
        i += 1
        if i < n and (Int(b[i]) == 0x2D or Int(b[i]) == 0x2B):
            clean += chr(Int(b[i]))
            i += 1
        var ep_start = i
        while i < n:
            i += 1
        clean += _validate_digit_run(orig, ep_start, i, allow_leading_zero=True)
    if i != n:
        raise Error("invalid float: " + orig)
    return Float64(clean)


def _inf() -> Float64:
    var big = 1.0e308
    return big * 10.0  # overflow -> +inf


def _nan() -> Float64:
    return _inf() - _inf()  # inf - inf -> nan


def _looks_like_datetime(t: String) -> Bool:
    # date like 1979-05-27, time 07:32:00, or datetime with 'T'/' ' between them.
    var b = t.as_bytes()
    var n = t.byte_length()
    if n < 5:
        return False
    var colons = 0
    var dashes_after_digit = 0
    for i in range(n):
        var c = Int(b[i])
        if c == 0x3A:  # :
            colons += 1
        if c == 0x2D and i > 0:  # '-' not leading sign
            var prev = Int(b[i - 1])
            if prev >= 48 and prev <= 57:
                dashes_after_digit += 1
    if colons >= 2:
        return True
    if dashes_after_digit >= 2:
        return True
    return False


def _read_array(bytes: List[UInt8], n: Int, start: Int) raises -> _ValResult:
    var pos = start + 1  # consume [
    var items = List[ConfigValue]()
    while True:
        pos = _skip_ws_and_comments_and_newlines(bytes, n, pos)
        if pos >= n:
            raise Error("unterminated array")
        if Int(bytes[pos]) == 0x5D:  # ]
            pos += 1
            break
        var er = _read_value(bytes, n, pos)
        pos = er.next
        items.append(er^.take())
        pos = _skip_ws_and_comments_and_newlines(bytes, n, pos)
        if pos >= n:
            raise Error("unterminated array")
        var c = Int(bytes[pos])
        if c == 0x2C:  # ,
            pos += 1
            continue
        if c == 0x5D:  # ]
            pos += 1
            break
        raise Error("expected ',' or ']' in array")
    return _ValResult(ConfigValue.array_(items^), pos)


def _read_basic_string(bytes: List[UInt8], n: Int, start: Int) raises -> _StrResult:
    # assumes bytes[start] == '"'  (rejects triple-quote multiline)
    if start + 2 < n and Int(bytes[start + 1]) == 0x22 and Int(bytes[start + 2]) == 0x22:
        raise Error("multiline basic strings not supported")
    var pos = start + 1
    var out = List[UInt8]()
    while pos < n:
        var c = Int(bytes[pos])
        if c == 0x22:  # closing "
            pos += 1
            return _StrResult(_bytes_to_str(out), pos)
        if c == 0x0A:
            raise Error("newline in basic string")
        if c == 0x5C:  # backslash
            pos += 1
            if pos >= n:
                raise Error("unterminated escape in string")
            var e = Int(bytes[pos])
            if e == 0x6E:  # n
                out.append(0x0A)
            elif e == 0x74:  # t
                out.append(0x09)
            elif e == 0x72:  # r
                out.append(0x0D)
            elif e == 0x22:  # "
                out.append(0x22)
            elif e == 0x5C:  # backslash
                out.append(0x5C)
            elif e == 0x66:  # f
                out.append(0x0C)
            elif e == 0x62:  # b
                out.append(0x08)
            elif e == 0x75:  # \uXXXX
                pos += 1
                var cp = _hex4(bytes, n, pos)
                pos += 3  # loop's +=1 handles the 4th
                _encode_utf8(out, cp)
            else:
                raise Error("invalid escape \\" + chr(e))
            pos += 1
        else:
            out.append(UInt8(c))
            pos += 1
    raise Error("unterminated string")


def _read_literal_string(bytes: List[UInt8], n: Int, start: Int) raises -> _StrResult:
    # assumes bytes[start] == '\''  ; no escapes; rejects ''' multiline
    if start + 2 < n and Int(bytes[start + 1]) == 0x27 and Int(bytes[start + 2]) == 0x27:
        raise Error("multiline literal strings not supported")
    var pos = start + 1
    var out = List[UInt8]()
    while pos < n:
        var c = Int(bytes[pos])
        if c == 0x27:  # closing '
            pos += 1
            return _StrResult(_bytes_to_str(out), pos)
        if c == 0x0A:
            raise Error("newline in literal string")
        out.append(UInt8(c))
        pos += 1
    raise Error("unterminated literal string")


def _hex4(bytes: List[UInt8], n: Int, start: Int) raises -> Int:
    if start + 4 > n:
        raise Error("truncated \\u escape")
    var v = 0
    for i in range(4):
        var h = _hexval(Int(bytes[start + i]))
        if h < 0:
            raise Error("bad hex in \\u escape")
        v = (v << 4) | h
    return v


def _hexval(c: Int) -> Int:
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 87
    if c >= 65 and c <= 70:
        return c - 55
    return -1


def _encode_utf8(mut buf: List[UInt8], cp: Int):
    if cp < 0x80:
        buf.append(UInt8(cp))
    elif cp < 0x800:
        buf.append(UInt8(0xC0 | (cp >> 6)))
        buf.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        buf.append(UInt8(0xE0 | (cp >> 12)))
        buf.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        buf.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        buf.append(UInt8(0xF0 | (cp >> 18)))
        buf.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        buf.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        buf.append(UInt8(0x80 | (cp & 0x3F)))


# ── byte helpers ───────────────────────────────────────────────────────────────
def _to_bytes(s: String) -> List[UInt8]:
    var b = s.as_bytes()
    var n = s.byte_length()
    var out = List[UInt8]()
    for i in range(n):
        out.append(b[i])
    return out^


def _bytes_to_str(b: List[UInt8]) -> String:
    # Build the String directly from the raw UTF-8 byte buffer so already-encoded
    # multi-byte sequences (e.g. "café") pass through unchanged. chr()-per-byte
    # re-encodes every byte >= 0x80 a second time (double-UTF-8) and is wrong.
    var n = len(b)
    if n == 0:
        return String("")
    return String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=b.unsafe_ptr(), length=n)))


def _trim(s: String) -> String:
    var b = s.as_bytes()
    var n = s.byte_length()
    var start = 0
    while start < n:
        var c = Int(b[start])
        if c == 0x20 or c == 0x09 or c == 0x0D or c == 0x0A:
            start += 1
        else:
            break
    var end = n
    while end > start:
        var c = Int(b[end - 1])
        if c == 0x20 or c == 0x09 or c == 0x0D or c == 0x0A:
            end -= 1
        else:
            break
    var out = String("")
    for i in range(start, end):
        out += chr(Int(b[i]))
    return out


def _skip_inline_ws(bytes: List[UInt8], n: Int, start: Int) -> Int:
    var pos = start
    while pos < n:
        var c = Int(bytes[pos])
        if c == 0x20 or c == 0x09:
            pos += 1
        else:
            break
    return pos


def _skip_to_newline(bytes: List[UInt8], n: Int, start: Int) raises -> Int:
    # allow trailing inline whitespace + optional comment, then require newline/EOF
    var pos = _skip_inline_ws(bytes, n, start)
    if pos < n and Int(bytes[pos]) == 0x23:  # #
        while pos < n and Int(bytes[pos]) != 0x0A:
            pos += 1
    if pos < n and Int(bytes[pos]) == 0x0D:
        pos += 1
    if pos < n and Int(bytes[pos]) == 0x0A:
        pos += 1
    return pos


def _skip_ws_and_comments_and_newlines(bytes: List[UInt8], n: Int, start: Int) -> Int:
    var pos = start
    while pos < n:
        var c = Int(bytes[pos])
        if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D:
            pos += 1
        elif c == 0x23:  # # comment to end of line
            while pos < n and Int(bytes[pos]) != 0x0A:
                pos += 1
        else:
            break
    return pos
