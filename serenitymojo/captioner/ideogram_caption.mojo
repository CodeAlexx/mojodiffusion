# ideogram_caption.mojo — pure-Mojo 1:1 port of ai-toolkit's
# toolkit/ideogram_caption.py (module A: the deterministic caption formatter).
#
# Reference (line-by-line truth): /home/alex/ai-toolkit/toolkit/ideogram_caption.py
# Spec:                           /home/alex/EriTrainer/IDEOGRAM_CAPTIONER_PORT_SPEC.md
#
# This is HOST string/JSON logic only — NO GPU, NO LLM, NO torch. It normalizes /
# migrates / minifies an *already-structured* Ideogram-4 caption (dict or JSON
# string) into the compact model-ready string, and passes plain prose through
# byte-for-byte unchanged.
#
# WHY a hand-rolled ordered JSON type (not MOJO-libs json):
#   Key order is LOAD-BEARING — the compact output must be byte-identical to
#   Python's json.dumps(..., separators=(",",":"), ensure_ascii=False). MOJO-libs
#   json/value.mojo stores objects as Dict[String, JSONValue], whose iteration
#   order is hash-based, NOT insertion order, so it cannot reproduce the documented
#   strict key orders. We therefore carry objects as an ORDERED List of (key,value)
#   pairs (JObject below), parse into that, and serialize compactly from it.
#
# Mojo notes:
#   - `comptime` (not `alias`); raises on fallible defs; file I/O via io/ffi
#     (sys_open/sys_pread), never builtin `open`.
#   - Recursive JSON value: JValue holds List[JValue] (array) and List[JPair]
#     (object). Both box their elements on the heap so recursion is fine.

from std.memory import UnsafePointer, alloc


# ───────────────────────────── constants (verbatim) ────────────────────────────
# MAX_IMAGE_PALETTE = 16  # style_description.color_palette
# MAX_ELEMENT_PALETTE = 5 # per-element color_palette
comptime MAX_IMAGE_PALETTE = 16
comptime MAX_ELEMENT_PALETTE = 5


# Canonical medium tokens (official set). MEDIUM_OPTIONS in the Python source.
def _medium_options() -> List[String]:
    return [
        String("photograph"),
        String("illustration"),
        String("3d_render"),
        String("painting"),
        String("graphic_design"),
    ]


# _MEDIUM_ALIASES table, verbatim (key -> canonical token). Returns the canonical
# token for `key` if present, else empty string sentinel "" meaning "not found".
def _medium_alias_lookup(key: String) -> String:
    if key == "photograph":
        return String("photograph")
    if key == "photo":
        return String("photograph")
    if key == "illustration":
        return String("illustration")
    if key == "3d render":
        return String("3d_render")
    if key == "3d_render":
        return String("3d_render")
    if key == "3d-render":
        return String("3d_render")
    if key == "3drender":
        return String("3d_render")
    if key == "render":
        return String("3d_render")
    if key == "3d":
        return String("3d_render")
    if key == "painting":
        return String("painting")
    if key == "graphic design":
        return String("graphic_design")
    if key == "graphic_design":
        return String("graphic_design")
    if key == "graphic-design":
        return String("graphic_design")
    if key == "graphic":
        return String("graphic_design")
    return String("")  # sentinel: not in the alias table


# ═══════════════════════════ ordered JSON value type ═══════════════════════════
# Tag values for JValue.kind.
comptime J_NULL = 0
comptime J_BOOL = 1
comptime J_INT = 2
comptime J_FLOAT = 3
comptime J_STR = 4
comptime J_ARR = 5
comptime J_OBJ = 6


@fieldwise_init
struct JPair(Copyable, Movable):
    """One (key, value) entry in an ordered JSON object."""

    var key: String
    var value: JValue


struct JValue(Copyable, Movable):
    """Tagged JSON value. Objects keep INSERTION ORDER via a List[JPair] (not a
    hash map), which is the whole point — key order is load-bearing for output."""

    var kind: Int
    var b: Bool
    # Numbers: ints (J_INT) keep their textual form so re-emit is byte-exact;
    # floats (J_FLOAT) likewise keep their original token. JSON numbers in the
    # caption schema (bbox) are integers; we still round-trip the raw token.
    var num_text: String
    var s: String
    var arr: List[JValue]
    var obj: List[JPair]

    def __init__(out self):
        self.kind = J_NULL
        self.b = False
        self.num_text = String("")
        self.s = String("")
        self.arr = List[JValue]()
        self.obj = List[JPair]()

    # ── constructors ──────────────────────────────────────────────────────────
    @staticmethod
    def null() -> JValue:
        return JValue()

    @staticmethod
    def from_bool(v: Bool) -> JValue:
        var x = JValue()
        x.kind = J_BOOL
        x.b = v
        return x^

    @staticmethod
    def from_int_text(t: String) -> JValue:
        var x = JValue()
        x.kind = J_INT
        x.num_text = t
        return x^

    @staticmethod
    def from_float_text(t: String) raises -> JValue:
        # Python parses a JSON float to a Python `float` and json.dumps
        # re-serializes it via repr(float) — so the ORIGINAL token is NOT echoed:
        # 1e2→100.0, 1.5e3→1500.0, 1E2→100.0, 2e+2→200.0, 3.0→3.0, -0.0→-0.0.
        # We reproduce that by parsing the token to Float64 (atof handles the full
        # exponent grammar) and reformatting with String(Float64), which Mojo
        # produces with the SAME shortest-round-trip algorithm as Python repr —
        # verified byte-identical across 0–1000 bbox decimals and the exponent
        # boundaries (1e+16, 1e-05, 1e+20, 10000000000.0, -0.0).
        # Residual (documented, out of caption domain): the two shortest-float
        # algorithms could in principle disagree on a pathological 17-significant-
        # digit double; captions carry only plain bbox/extra decimals, never such
        # values, so this path is exact for every real input.
        var x = JValue()
        x.kind = J_FLOAT
        x.num_text = String(Float64(atof(t)))
        return x^

    @staticmethod
    def from_string(v: String) -> JValue:
        var x = JValue()
        x.kind = J_STR
        x.s = v
        return x^

    @staticmethod
    def new_array() -> JValue:
        var x = JValue()
        x.kind = J_ARR
        return x^

    @staticmethod
    def new_object() -> JValue:
        var x = JValue()
        x.kind = J_OBJ
        return x^

    # ── type tests ───────────────────────────────────────────────────────────
    def is_null(self) -> Bool:
        return self.kind == J_NULL

    def is_string(self) -> Bool:
        return self.kind == J_STR

    def is_array(self) -> Bool:
        return self.kind == J_ARR

    def is_object(self) -> Bool:
        return self.kind == J_OBJ

    # ── object helpers (ordered) ───────────────────────────────────────────────
    def has_key(self, key: String) -> Bool:
        if self.kind != J_OBJ:
            return False
        for i in range(len(self.obj)):
            if self.obj[i].key == key:
                return True
        return False

    def get(self, key: String) -> JValue:
        """First value for `key`, or null if absent / not an object."""
        if self.kind == J_OBJ:
            for i in range(len(self.obj)):
                if self.obj[i].key == key:
                    return self.obj[i].value.copy()
        return JValue()

    def set(mut self, key: String, var v: JValue):
        """Append a (key, value) pair preserving insertion order. Used when
        BUILDING normalized output, where every key is unique by construction."""
        self.obj.append(JPair(key, v^))

    def od_set(mut self, key: String, var v: JValue):
        """OrderedDict.__setitem__ semantics for the PARSER: a duplicate key keeps
        the LAST value but the FIRST-SEEN position (single entry). This mirrors
        Python's json.loads(object_pairs_hook=OrderedDict): `OrderedDict([(k,v1),
        (k,v2)])` → one key `k` mapping to `v2`, positioned where it first
        appeared. Replace in place if present; append if new."""
        for i in range(len(self.obj)):
            if self.obj[i].key == key:
                self.obj[i].value = v^
                return
        self.obj.append(JPair(key, v^))

    def append(mut self, var v: JValue):
        self.arr.append(v^)


# ═════════════════════════════════ JSON parser ════════════════════════════════
# Minimal RFC-8259 parser into the ordered JValue tree. Tolerant of the subset
# Python's json.loads accepts for our inputs; anything malformed -> raise (the
# caller catches and treats it as "not a structured caption", matching the
# Python try/except json.loads).

struct _Parser:
    var data: List[UInt8]
    var pos: Int
    var n: Int

    def __init__(out self, text: String):
        self.data = List[UInt8]()
        var b = text.as_bytes()
        for i in range(len(b)):
            self.data.append(b[i])
        self.pos = 0
        self.n = len(self.data)

    def _peek(self) -> Int:
        if self.pos < self.n:
            return Int(self.data[self.pos])
        return -1

    def _skip_ws(mut self):
        # JSON whitespace: space, tab, newline, carriage return.
        while self.pos < self.n:
            var c = Int(self.data[self.pos])
            if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D:
                self.pos += 1
            else:
                break

    def parse_value(mut self) raises -> JValue:
        self._skip_ws()
        var c = self._peek()
        if c == -1:
            raise Error("json: unexpected end of input")
        if c == 0x7B:  # {
            return self._parse_object()
        if c == 0x5B:  # [
            return self._parse_array()
        if c == 0x22:  # "
            return JValue.from_string(self._parse_string())
        if c == 0x74 or c == 0x66:  # t(rue) / f(alse)
            return self._parse_bool()
        if c == 0x6E:  # n(ull)
            return self._parse_null()
        # number: digit or '-'
        if c == 0x2D or (c >= 0x30 and c <= 0x39):
            return self._parse_number()
        # DELIBERATE DIVERGENCE (documented, FRAGILE not BLOCKER): Python's
        # json.loads accepts the non-standard literals NaN / Infinity / -Infinity
        # by default and would MINIFY such a caption. We do NOT special-case them:
        # they are not valid RFC-8259 JSON, so we raise here → digest_caption_string
        # returns the input UNCHANGED (passthrough). That is a defensible, strictly-
        # conformant choice and these literals never occur in real Ideogram
        # captions (bbox/extra values are finite decimals). This is the one
        # accepted byte-divergence from Python and is therefore excluded from the
        # adversarial pass-required gate by design (not by code special-casing).
        raise Error("json: unexpected character")

    def _parse_object(mut self) raises -> JValue:
        var out = JValue.new_object()
        self.pos += 1  # consume '{'
        self._skip_ws()
        if self._peek() == 0x7D:  # } empty object
            self.pos += 1
            return out^
        while True:
            self._skip_ws()
            if self._peek() != 0x22:
                raise Error("json: expected string key")
            var key = self._parse_string()
            self._skip_ws()
            if self._peek() != 0x3A:  # :
                raise Error("json: expected ':'")
            self.pos += 1
            var val = self.parse_value()
            out.od_set(key, val^)  # OrderedDict semantics: dup key → last-wins, first position
            self._skip_ws()
            var c = self._peek()
            if c == 0x2C:  # ,
                self.pos += 1
                continue
            if c == 0x7D:  # }
                self.pos += 1
                break
            raise Error("json: expected ',' or '}'")
        return out^

    def _parse_array(mut self) raises -> JValue:
        var out = JValue.new_array()
        self.pos += 1  # consume '['
        self._skip_ws()
        if self._peek() == 0x5D:  # ] empty array
            self.pos += 1
            return out^
        while True:
            var val = self.parse_value()
            out.append(val^)
            self._skip_ws()
            var c = self._peek()
            if c == 0x2C:  # ,
                self.pos += 1
                continue
            if c == 0x5D:  # ]
                self.pos += 1
                break
            raise Error("json: expected ',' or ']'")
        return out^

    def _parse_string(mut self) raises -> String:
        # pos is at the opening quote. Returns the DECODED string value.
        self.pos += 1  # consume opening "
        var out = List[UInt8]()
        while self.pos < self.n:
            var c = Int(self.data[self.pos])
            if c == 0x22:  # closing "
                self.pos += 1
                return _bytes_to_string(out)
            if c == 0x5C:  # backslash escape
                self.pos += 1
                if self.pos >= self.n:
                    raise Error("json: bad escape")
                var e = Int(self.data[self.pos])
                if e == 0x22:  # \"
                    out.append(0x22)
                elif e == 0x5C:  # \\
                    out.append(0x5C)
                elif e == 0x2F:  # \/
                    out.append(0x2F)
                elif e == 0x62:  # \b
                    out.append(0x08)
                elif e == 0x66:  # \f
                    out.append(0x0C)
                elif e == 0x6E:  # \n
                    out.append(0x0A)
                elif e == 0x72:  # \r
                    out.append(0x0D)
                elif e == 0x74:  # \t
                    out.append(0x09)
                elif e == 0x75:  # \uXXXX
                    var cp = self._parse_hex4()
                    # Handle UTF-16 surrogate pair for codepoints > 0xFFFF.
                    if cp >= 0xD800 and cp <= 0xDBFF:
                        # high surrogate; expect a following \uXXXX low surrogate
                        if (
                            self.pos + 2 < self.n
                            and Int(self.data[self.pos + 1]) == 0x5C
                            and Int(self.data[self.pos + 2]) == 0x75
                        ):
                            self.pos += 2  # consume "\u"
                            var lo = self._parse_hex4()
                            cp = (
                                0x10000
                                + ((cp - 0xD800) << 10)
                                + (lo - 0xDC00)
                            )
                    _append_utf8(out, cp)
                else:
                    raise Error("json: bad escape char")
                self.pos += 1
            else:
                # STRICT JSON: an unescaped control char (<0x20) inside a string
                # literal is a parse error (Python json.loads rejects it by
                # default → digest passes the input through unchanged). A raw TAB
                # or newline byte here must NOT be accepted-then-re-escaped.
                if c < 0x20:
                    raise Error("json: raw control char in string")
                # raw byte (UTF-8 passthrough for ≥0x20, incl. multibyte UTF-8)
                out.append(UInt8(c))
                self.pos += 1
        raise Error("json: unterminated string")

    def _parse_hex4(mut self) raises -> Int:
        # pos is at 'u'; read the 4 following hex digits, advance pos to last one.
        var v = 0
        for _k in range(4):
            self.pos += 1
            if self.pos >= self.n:
                raise Error("json: bad \\u escape")
            var d = Int(self.data[self.pos])
            var hv: Int
            if d >= 0x30 and d <= 0x39:
                hv = d - 0x30
            elif d >= 0x41 and d <= 0x46:
                hv = d - 0x41 + 10
            elif d >= 0x61 and d <= 0x66:
                hv = d - 0x61 + 10
            else:
                raise Error("json: bad hex digit")
            v = v * 16 + hv
        return v

    def _parse_number(mut self) raises -> JValue:
        # STRICT JSON number grammar (RFC 8259), matching Python json.loads, so an
        # input Python would reject (and pass through) does NOT get parsed+minified:
        #   number = [ "-" ] int [ frac ] [ exp ]
        #   int    = "0" | ( digit1-9 *digit )      ; NO leading zeros (rejects 01)
        #   frac   = "." 1*digit                     ; ≥1 digit after '.' (rejects 5.)
        #   exp    = ("e"|"E") ["+"|"-"] 1*digit
        # No leading '+'. A '.' with no following digit, a leading-zero int, or a
        # bare exponent are parse errors → the whole digest passes the input through.
        var start = self.pos
        var is_float = False
        if self._peek() == 0x2D:  # optional leading '-'
            self.pos += 1
        # int: "0" or digit1-9 *digit
        var d0 = self._peek()
        if d0 < 0x30 or d0 > 0x39:
            raise Error("json: number missing integer digit")
        if d0 == 0x30:  # leading zero → must be a lone '0' (no more digits)
            self.pos += 1
        else:
            while self.pos < self.n:
                var c = Int(self.data[self.pos])
                if c >= 0x30 and c <= 0x39:
                    self.pos += 1
                else:
                    break
        # frac: "." 1*digit
        if self.pos < self.n and Int(self.data[self.pos]) == 0x2E:  # '.'
            is_float = True
            self.pos += 1
            var fd = self._peek()
            if fd < 0x30 or fd > 0x39:
                raise Error("json: number trailing dot")  # rejects "5."
            while self.pos < self.n:
                var c = Int(self.data[self.pos])
                if c >= 0x30 and c <= 0x39:
                    self.pos += 1
                else:
                    break
        # exp: ("e"|"E") ["+"|"-"] 1*digit
        if self.pos < self.n and (
            Int(self.data[self.pos]) == 0x65 or Int(self.data[self.pos]) == 0x45
        ):
            is_float = True
            self.pos += 1
            if self.pos < self.n and (
                Int(self.data[self.pos]) == 0x2B or Int(self.data[self.pos]) == 0x2D
            ):
                self.pos += 1
            var ed = self._peek()
            if ed < 0x30 or ed > 0x39:
                raise Error("json: number bad exponent")
            while self.pos < self.n:
                var c = Int(self.data[self.pos])
                if c >= 0x30 and c <= 0x39:
                    self.pos += 1
                else:
                    break
        var tok = _slice_to_string(Span(self.data), start, self.pos)
        if is_float:
            return JValue.from_float_text(tok)
        return JValue.from_int_text(tok)

    def _parse_bool(mut self) raises -> JValue:
        if self.pos + 4 <= self.n and self._match("true", 4):
            self.pos += 4
            return JValue.from_bool(True)
        if self.pos + 5 <= self.n and self._match("false", 5):
            self.pos += 5
            return JValue.from_bool(False)
        raise Error("json: bad literal")

    def _parse_null(mut self) raises -> JValue:
        if self.pos + 4 <= self.n and self._match("null", 4):
            self.pos += 4
            return JValue.null()
        raise Error("json: bad literal")

    def _match(self, lit: String, length: Int) -> Bool:
        var lb = lit.as_bytes()
        for i in range(length):
            if Int(self.data[self.pos + i]) != Int(lb[i]):
                return False
        return True


def _bytes_to_string(b: List[UInt8]) -> String:
    # Build a String from a UInt8 list (UTF-8 bytes) by EXPLICIT LENGTH — never
    # NUL-terminated. `String(unsafe_from_utf8_ptr=...)` would stop at the first
    # 0x00, silently truncating any value that contains an embedded NUL; Python
    # keeps such bytes. `unsafe_from_utf8` consumes the whole List by length (no
    # validation, no raise — so this stays callable from non-raising helpers), so
    # a 0x00 byte is preserved verbatim (verified: "a\x00b" → byte_length 3,
    # bytes 97 0 98).
    return String(unsafe_from_utf8=b.copy())


def _slice_to_string[o: ImmutOrigin](b: Span[UInt8, o], start: Int, end: Int) -> String:
    var out = List[UInt8]()
    for i in range(start, end):
        out.append(b[i])
    return _bytes_to_string(out)


def _append_utf8(mut out: List[UInt8], cp: Int):
    # Encode a Unicode scalar value as UTF-8 bytes appended to `out`.
    if cp < 0x80:
        out.append(UInt8(cp))
    elif cp < 0x800:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))


def parse_json(text: String) raises -> JValue:
    """Parse `text` into the ordered JValue tree. Raises on malformed input
    (mirrors Python json.loads raising; caller treats a raise as 'not a caption').
    Trailing non-whitespace after the value is an error, matching json.loads."""
    var p = _Parser(text)
    var v = p.parse_value()
    p._skip_ws()
    if p.pos != p.n:
        raise Error("json: trailing data")
    return v^


# ═══════════════════════════ compact serializer ═══════════════════════════════
# Matches json.dumps(data, ensure_ascii=False, separators=(",",":")):
#   - no spaces between tokens
#   - ensure_ascii=False → raw UTF-8 passthrough (no \uXXXX for non-ASCII)
#   - string escaping per Python: \" \\ and control chars < 0x20 → \b \t \n \f \r
#     or \u00XX; everything else (incl. forward slash and non-ASCII) is literal.

def _hex_nibble(v: Int) -> UInt8:
    # 0..15 → ASCII '0'..'9','a'..'f' (Python json uses lowercase \u escapes).
    if v < 10:
        return UInt8(0x30 + v)
    return UInt8(0x61 + (v - 10))


def _escape_string_into(mut out: List[UInt8], s: String):
    out.append(0x22)  # opening "
    var b = s.as_bytes()
    for i in range(len(b)):
        var c = Int(b[i])
        if c == 0x22:  # "
            out.append(0x5C)
            out.append(0x22)
        elif c == 0x5C:  # backslash
            out.append(0x5C)
            out.append(0x5C)
        elif c == 0x08:  # \b
            out.append(0x5C)
            out.append(0x62)
        elif c == 0x09:  # \t
            out.append(0x5C)
            out.append(0x74)
        elif c == 0x0A:  # \n
            out.append(0x5C)
            out.append(0x6E)
        elif c == 0x0C:  # \f
            out.append(0x5C)
            out.append(0x66)
        elif c == 0x0D:  # \r
            out.append(0x5C)
            out.append(0x72)
        elif c < 0x20:  # other control char → \u00XX
            out.append(0x5C)
            out.append(0x75)
            out.append(0x30)
            out.append(0x30)
            out.append(_hex_nibble((c >> 4) & 0xF))
            out.append(_hex_nibble(c & 0xF))
        else:
            out.append(UInt8(c))  # literal byte (ASCII or UTF-8 continuation)
    out.append(0x22)  # closing "


def _serialize_into(mut out: List[UInt8], v: JValue):
    if v.kind == J_NULL:
        _append_ascii(out, "null")
    elif v.kind == J_BOOL:
        if v.b:
            _append_ascii(out, "true")
        else:
            _append_ascii(out, "false")
    elif v.kind == J_INT or v.kind == J_FLOAT:
        _append_ascii(out, v.num_text)
    elif v.kind == J_STR:
        _escape_string_into(out, v.s)
    elif v.kind == J_ARR:
        out.append(0x5B)  # [
        for i in range(len(v.arr)):
            if i > 0:
                out.append(0x2C)  # ,
            _serialize_into(out, v.arr[i])
        out.append(0x5D)  # ]
    elif v.kind == J_OBJ:
        out.append(0x7B)  # {
        for i in range(len(v.obj)):
            if i > 0:
                out.append(0x2C)  # ,
            _escape_string_into(out, v.obj[i].key)
            out.append(0x3A)  # :
            _serialize_into(out, v.obj[i].value)
        out.append(0x7D)  # }


def _append_ascii(mut out: List[UInt8], s: String):
    var b = s.as_bytes()
    for i in range(len(b)):
        out.append(b[i])


def to_model_string(data: JValue) -> String:
    """Serialize a caption value to the compact, model-ready string the renderer
    wants. Equivalent to json.dumps(data, ensure_ascii=False, separators=(",",":"))."""
    var out = List[UInt8]()
    _serialize_into(out, data)
    return _bytes_to_string(out)


# ═════════════════════════════ string helpers ═════════════════════════════════

def _strip(s: String) -> String:
    """Python str.strip(): trim leading/trailing ASCII whitespace
    (space, \\t, \\n, \\r, \\f, \\v)."""
    var b = s.as_bytes()
    var n = len(b)
    var start = 0
    while start < n and _is_py_space(Int(b[start])):
        start += 1
    var end = n
    while end > start and _is_py_space(Int(b[end - 1])):
        end -= 1
    return _slice_to_string(b, start, end)


def _is_py_space(c: Int) -> Bool:
    # Python str.strip() default whitespace: space, \t, \n, \r, \f, \v.
    return (
        c == 0x20
        or c == 0x09
        or c == 0x0A
        or c == 0x0D
        or c == 0x0C
        or c == 0x0B
    )


def _rstrip_char(s: String, ch: Int) -> String:
    """Python str.rstrip(c): remove trailing occurrences of byte `ch`."""
    var b = s.as_bytes()
    var end = len(b)
    while end > 0 and Int(b[end - 1]) == ch:
        end -= 1
    return _slice_to_string(b, 0, end)


def _to_lower(s: String) -> String:
    """ASCII lowercase (Python str.lower for ASCII; medium tokens are ASCII)."""
    var b = s.as_bytes()
    var out = List[UInt8]()
    for i in range(len(b)):
        var c = Int(b[i])
        if c >= 0x41 and c <= 0x5A:
            out.append(UInt8(c + 32))
        else:
            out.append(UInt8(c))
    return _bytes_to_string(out)


def _to_upper(s: String) -> String:
    """ASCII uppercase (used for hex normalization)."""
    var b = s.as_bytes()
    var out = List[UInt8]()
    for i in range(len(b)):
        var c = Int(b[i])
        if c >= 0x61 and c <= 0x7A:
            out.append(UInt8(c - 32))
        else:
            out.append(UInt8(c))
    return _bytes_to_string(out)


def _starts_with(s: String, prefix: String) -> Bool:
    var sb = s.as_bytes()
    var pb = prefix.as_bytes()
    if len(pb) > len(sb):
        return False
    for i in range(len(pb)):
        if sb[i] != pb[i]:
            return False
    return True


def _is_hex_digit(c: Int) -> Bool:
    return (
        (c >= 0x30 and c <= 0x39)
        or (c >= 0x41 and c <= 0x46)
        or (c >= 0x61 and c <= 0x66)
    )


# ═══════════════════════════ ported public functions ══════════════════════════

def canon_medium(medium: String) -> String:
    """Canonicalize a medium string to an official token when recognized,
    otherwise return it stripped (custom mediums are allowed, preserved as-is).

    Python:
      key = medium.strip().rstrip(".").strip().lower()
      if key in _MEDIUM_ALIASES: return _MEDIUM_ALIASES[key]
      return medium.strip()
    (The Python `if not isinstance(medium, str)` guard is handled by callers; in
    Mojo this overload takes a String, and the JValue-aware path checks type.)"""
    var key = _to_lower(_strip(_rstrip_char(_strip(medium), 0x2E)))  # '.' = 0x2E
    var aliased = _medium_alias_lookup(key)
    if aliased.byte_length() > 0:
        return aliased
    return _strip(medium)


def is_photo_medium(medium: String) -> Bool:
    """True for the photograph branch (uses `photo`), False for art_style branch."""
    return canon_medium(medium) == "photograph"


# normalize_hex returns "None" in Python. We model that with a (Bool, String)
# pair: ok=False means None. A thin wrapper returns the sentinel string for the
# probe's string-comparison convenience.
struct HexResult(Copyable, Movable):
    var ok: Bool
    var value: String

    def __init__(out self, ok: Bool, value: String):
        self.ok = ok
        self.value = value


def normalize_hex(color: String) -> HexResult:
    """Return an UPPERCASE #RRGGBB string, expanding #RGB -> #RRGGBB.
    ok=False (Python None) if invalid.

    Python:
      s = color.strip()
      if _HEX6_RE.match(s): return "#" + s[1:].upper()
      if _HEX3_RE.match(s): return "#" + "".join(ch*2 for ch in s[1:]).upper()
      return None
    The regexes are anchored ^...$ — match the WHOLE string."""
    var s = _strip(color)
    var b = s.as_bytes()
    var n = len(b)
    # #RRGGBB : leading '#', exactly 6 hex digits, nothing else.
    if n == 7 and Int(b[0]) == 0x23:
        var all_hex = True
        for i in range(1, 7):
            if not _is_hex_digit(Int(b[i])):
                all_hex = False
                break
        if all_hex:
            return HexResult(True, String("#") + _to_upper(_slice_to_string(b, 1, 7)))
    # #RGB : leading '#', exactly 3 hex digits → expand each to a pair.
    if n == 4 and Int(b[0]) == 0x23:
        var all_hex = True
        for i in range(1, 4):
            if not _is_hex_digit(Int(b[i])):
                all_hex = False
                break
        if all_hex:
            var expanded = List[UInt8]()
            for i in range(1, 4):
                expanded.append(b[i])
                expanded.append(b[i])
            return HexResult(True, String("#") + _to_upper(_bytes_to_string(expanded)))
    return HexResult(False, String(""))


def sanitize_palette(values: List[String], max_len: Int) -> List[String]:
    """Keep unique, valid, UPPERCASE hex colors in order, capped to max_len.
    Returns the cleaned list (possibly empty → caller drops the key).

    Python dedupes by NORMALIZED hex (first wins, order preserved); stops once
    len(out) >= max_len."""
    var out = List[String]()
    var seen = List[String]()
    for ci in range(len(values)):
        var h = normalize_hex(values[ci])
        if not h.ok:
            continue
        var dup = False
        for si in range(len(seen)):
            if seen[si] == h.value:
                dup = True
                break
        if dup:
            continue
        seen.append(h.value)
        out.append(h.value)
        if len(out) >= max_len:
            break
    return out^


def _palette_strings(v: JValue) -> List[String]:
    """Extract the raw string members of a JValue array (color_palette). Non-array
    or non-string members are skipped (normalize_hex would reject non-strings as
    None anyway, matching Python's isinstance checks)."""
    var out = List[String]()
    if v.is_array():
        for i in range(len(v.arr)):
            if v.arr[i].is_string():
                out.append(v.arr[i].s)
            else:
                # A non-string palette entry → normalize_hex(None/num) = None →
                # skipped. Push an empty placeholder that normalize_hex rejects.
                out.append(String(""))
    return out^


def normalize_style(style: JValue) -> JValue:
    """Reorder/clean style_description into the correct branch (photo vs art_style)
    with the strict key order, canonical medium, and uppercase palette. Accepts the
    old shape (always `photo`) and migrates it based on the medium."""
    if not style.is_object():
        return style.copy()

    var raw_medium = style.get("medium")
    # medium = canon_medium(raw_medium) if raw_medium is not None else None
    # In Python raw_medium could be a non-str; canon_medium returns it unchanged.
    # Our schema mediums are strings; treat present+string as the medium.
    var medium = String("")
    var medium_is_none = True
    if style.has_key("medium") and not raw_medium.is_null():
        medium_is_none = False
        if raw_medium.is_string():
            medium = canon_medium(raw_medium.s)
        # else: non-string medium kept as-is conceptually; not in our fixtures.

    var has_photo = _truthy(style.get("photo"))
    var has_art = _truthy(style.get("art_style"))

    # Decide the branch.
    var photo_branch: Bool
    if (not medium_is_none) and _in_medium_options(medium):
        photo_branch = medium == "photograph"
    elif has_art and not has_photo:
        photo_branch = False
    else:
        photo_branch = True

    var photo_val = style.get("photo") if has_photo else JValue.null()
    var has_photo_val = has_photo
    var art_val = style.get("art_style") if has_art else JValue.null()
    var has_art_val = has_art

    var out = JValue.new_object()
    if style.has_key("aesthetics"):
        out.set("aesthetics", style.get("aesthetics"))
    if style.has_key("lighting"):
        out.set("lighting", style.get("lighting"))

    if photo_branch:
        # aesthetics, lighting, photo, medium, color_palette
        # val = photo_val if photo_val is not None else art_val
        if has_photo_val:
            out.set("photo", photo_val^)
        elif has_art_val:
            out.set("photo", art_val^)
        if not medium_is_none:
            out.set("medium", _medium_jvalue(medium, raw_medium))
    else:
        # aesthetics, lighting, medium, art_style, color_palette
        if not medium_is_none:
            out.set("medium", _medium_jvalue(medium, raw_medium))
        # val = art_val if art_val is not None else photo_val
        if has_art_val:
            out.set("art_style", art_val^)
        elif has_photo_val:
            out.set("art_style", photo_val^)

    var pal = sanitize_palette(
        _palette_strings(style.get("color_palette")), MAX_IMAGE_PALETTE
    )
    if len(pal) > 0:
        out.set("color_palette", _palette_jvalue(pal))

    # Preserve any unexpected extra keys at the end rather than dropping them.
    for i in range(len(style.obj)):
        var k = style.obj[i].key
        if not (
            k == "aesthetics"
            or k == "lighting"
            or k == "photo"
            or k == "art_style"
            or k == "medium"
            or k == "color_palette"
        ):
            out.set(k, style.obj[i].value.copy())
    return out^


def _medium_jvalue(canon: String, raw: JValue) -> JValue:
    """The medium value to emit. canon_medium produced `canon` from a string
    medium; emit it as a JSON string. (If the source medium was a non-string,
    Python would emit it unchanged — out of scope for the caption schema.)"""
    if raw.is_string():
        return JValue.from_string(canon)
    return raw.copy()


def _palette_jvalue(pal: List[String]) -> JValue:
    var arr = JValue.new_array()
    for i in range(len(pal)):
        arr.append(JValue.from_string(pal[i]))
    return arr^


def _truthy(v: JValue) -> Bool:
    """Python bool(x) for the value types we see: None/missing → False; empty
    string → False; empty array/object → False; non-empty → True; bool → itself.
    style.get returns null for a missing key (matches `bool(style.get(...))`)."""
    if v.is_null():
        return False
    if v.kind == J_BOOL:
        return v.b
    if v.is_string():
        return v.s.byte_length() > 0
    if v.is_array():
        return len(v.arr) > 0
    if v.is_object():
        return len(v.obj) > 0
    if v.kind == J_INT or v.kind == J_FLOAT:
        # bool(0) is False; numbers in style.get(photo/art_style) are atypical.
        return not (v.num_text == "0" or v.num_text == "0.0")
    return True


def _in_medium_options(m: String) -> Bool:
    var opts = _medium_options()
    for i in range(len(opts)):
        if opts[i] == m:
            return True
    return False


def normalize_element(el: JValue) -> JValue:
    """Reorder an element's keys to the strict schema order and uppercase its
    palette. obj: type, bbox, desc, color_palette. text: type, bbox, text, desc,
    color_palette. bbox is kept verbatim."""
    if not el.is_object():
        return el.copy()
    # etype = el.get("type", "obj")
    var etype_v = el.get("type")
    var etype = el.get("type") if el.has_key("type") else JValue.from_string("obj")
    var etype_str = etype_v.s if (el.has_key("type") and etype_v.is_string()) else String("obj")

    var out = JValue.new_object()
    out.set("type", etype^)
    # if el.get("bbox") is not None: out["bbox"] = el["bbox"]
    if el.has_key("bbox") and not el.get("bbox").is_null():
        out.set("bbox", el.get("bbox"))
    if etype_str == "text":
        if el.has_key("text"):
            out.set("text", el.get("text"))
        if el.has_key("desc"):
            out.set("desc", el.get("desc"))
    else:
        if el.has_key("desc"):
            out.set("desc", el.get("desc"))
    var pal = sanitize_palette(
        _palette_strings(el.get("color_palette")), MAX_ELEMENT_PALETTE
    )
    if len(pal) > 0:
        out.set("color_palette", _palette_jvalue(pal))
    # Preserve extras at the end: for k,v in el.items(): if k not in out and
    # k != "color_palette": out[k] = v
    for i in range(len(el.obj)):
        var k = el.obj[i].key
        if (not out.has_key(k)) and k != "color_palette":
            out.set(k, el.obj[i].value.copy())
    return out^


def normalize_caption_dict(data: JValue) -> JValue:
    """Drop aspect_ratio; enforce top-level key order; normalize style and every
    element. Returns a new object."""
    if not data.is_object():
        return data.copy()
    # data.pop("aspect_ratio", None) — handled by skipping it everywhere below.

    var out = JValue.new_object()
    if data.has_key("high_level_description"):
        out.set("high_level_description", data.get("high_level_description"))
    if data.has_key("style_description"):
        out.set("style_description", normalize_style(data.get("style_description")))

    var decon = data.get("compositional_deconstruction")
    if data.has_key("compositional_deconstruction") and decon.is_object():
        var nd = JValue.new_object()
        if decon.has_key("background"):
            nd.set("background", decon.get("background"))
        var els = decon.get("elements")
        if decon.has_key("elements") and els.is_array():
            var arr = JValue.new_array()
            for i in range(len(els.arr)):
                arr.append(normalize_element(els.arr[i]))
            nd.set("elements", arr^)
        for i in range(len(decon.obj)):
            var k = decon.obj[i].key
            if not (k == "background" or k == "elements"):
                nd.set(k, decon.obj[i].value.copy())
        out.set("compositional_deconstruction", nd^)
    elif data.has_key("compositional_deconstruction") and not decon.is_null():
        out.set("compositional_deconstruction", decon^)

    # for k,v in data.items(): append extras (not the 4 handled keys, incl.
    # aspect_ratio which is dropped).
    for i in range(len(data.obj)):
        var k = data.obj[i].key
        if not (
            k == "high_level_description"
            or k == "style_description"
            or k == "compositional_deconstruction"
            or k == "aspect_ratio"
        ):
            out.set(k, data.obj[i].value.copy())
    return out^


# ─────────────────── bbox text rewrite (no JSON parse) ─────────────────────────
# Hand-written scanner for `"bbox"\s*:\s*[\s*n\s*,\s*n\s*,\s*n\s*,\s*n\s*]`
# matching Python's _BBOX_TEXT_RE. Numbers are signed ints or decimals
# (-?\d+(?:\.\d+)?). On a match: swap [x1,y1,x2,y2] → [y1,x1,y2,x2], clamp each to
# 0–1000, sort each axis pair. Only bbox arrays are touched; everything else is
# copied byte-for-byte.

def _clamp_1000(v: String) -> Int:
    """Python: max(0, min(1000, round(float(v)))). v is a numeric token
    (-?\\d+(?:\\.\\d+)?). round() is banker's rounding in Python 3."""
    var f = _parse_float(v)
    var r = _py_round(f)
    if r < 0:
        return 0
    if r > 1000:
        return 1000
    return r


def _parse_float(v: String) -> Float64:
    # Parse a simple decimal token "-?digits(.digits)?" into Float64.
    var b = v.as_bytes()
    var i = 0
    var n = len(b)
    var neg = False
    if i < n and Int(b[i]) == 0x2D:  # '-'
        neg = True
        i += 1
    var intpart: Float64 = 0.0
    while i < n and Int(b[i]) >= 0x30 and Int(b[i]) <= 0x39:
        intpart = intpart * 10.0 + Float64(Int(b[i]) - 0x30)
        i += 1
    var frac: Float64 = 0.0
    var scale: Float64 = 1.0
    if i < n and Int(b[i]) == 0x2E:  # '.'
        i += 1
        while i < n and Int(b[i]) >= 0x30 and Int(b[i]) <= 0x39:
            scale = scale * 10.0
            frac = frac + Float64(Int(b[i]) - 0x30) / scale
            i += 1
    var val = intpart + frac
    if neg:
        val = -val
    return val


def _py_round(x: Float64) -> Int:
    """Python 3 round() to nearest int with banker's rounding (round-half-to-even).
    bbox values in fixtures are integers, so the .5 path is essentially unused, but
    we replicate the rule for faithfulness."""
    var floor_x = _floor(x)
    var diff = x - Float64(floor_x)
    if diff < 0.5:
        return floor_x
    if diff > 0.5:
        return floor_x + 1
    # exactly .5 → round to even
    if floor_x % 2 == 0:
        return floor_x
    return floor_x + 1


def _floor(x: Float64) -> Int:
    var t = Int(x)
    if Float64(t) > x:  # truncation went up (negative non-integer)
        return t - 1
    return t


@fieldwise_init
struct _Pair2(Copyable, Movable):
    var lo: Int
    var hi: Int


def _sort2(a: Int, b: Int) -> _Pair2:
    if a <= b:
        return _Pair2(a, b)
    return _Pair2(b, a)


def swap_bbox_xy_in_text(text: String) -> String:
    """Swap every [x1,y1,x2,y2] bbox to the stored [y1,x1,y2,x2] order directly in
    the raw text — clamping each value to 0-1000 and ordering each axis pair. Never
    parses the surrounding JSON. Only `"bbox":[n,n,n,n]` arrays are touched;
    everything else is byte-for-byte. Returns the rewritten text."""
    var b = text.as_bytes()
    var n = len(b)
    var out = List[UInt8]()
    var i = 0
    while i < n:
        var m = _try_match_bbox(b, n, i)
        if m.matched:
            # Emit the swapped/clamped/sorted replacement.
            var cx = _sort2(_clamp_1000(m.x1), _clamp_1000(m.x2))
            var cy = _sort2(_clamp_1000(m.y1), _clamp_1000(m.y2))
            # f'"bbox":[{cy1},{cx1},{cy2},{cx2}]'
            _append_ascii(out, String('"bbox":['))
            _append_ascii(out, String(cy.lo))
            out.append(0x2C)
            _append_ascii(out, String(cx.lo))
            out.append(0x2C)
            _append_ascii(out, String(cy.hi))
            out.append(0x2C)
            _append_ascii(out, String(cx.hi))
            out.append(0x5D)  # ]
            i = m.end
        else:
            out.append(b[i])
            i += 1
    return _bytes_to_string(out)


struct _BBoxMatch(Copyable, Movable):
    var matched: Bool
    var end: Int  # index just past the matched ']'
    var x1: String
    var y1: String
    var x2: String
    var y2: String

    def __init__(out self):
        self.matched = False
        self.end = 0
        self.x1 = String("")
        self.y1 = String("")
        self.x2 = String("")
        self.y2 = String("")


def _try_match_bbox[o: ImmutOrigin](b: Span[UInt8, o], n: Int, start: Int) -> _BBoxMatch:
    """Try to match _BBOX_TEXT_RE anchored at `start`. The regex is:
      "bbox"\\s*:\\s*\\[\\s* num \\s*,\\s* num \\s*,\\s* num \\s*,\\s* num \\s*\\]
    where num = -?\\d+(?:\\.\\d+)?. Returns matched=False if no match here."""
    var res = _BBoxMatch()
    # literal "bbox"
    var lit = String('"bbox"').as_bytes()
    if start + len(lit) > n:
        return res^
    for k in range(len(lit)):
        if b[start + k] != lit[k]:
            return res^
    var p = start + len(lit)
    p = _skip_re_ws(b, n, p)
    if p >= n or Int(b[p]) != 0x3A:  # ':'
        return res^
    p += 1
    p = _skip_re_ws(b, n, p)
    if p >= n or Int(b[p]) != 0x5B:  # '['
        return res^
    p += 1
    # number 1
    p = _skip_re_ws(b, n, p)
    var r1 = _scan_number(b, n, p)
    if not r1.ok:
        return res^
    res.x1 = r1.tok
    p = r1.end
    p = _skip_re_ws(b, n, p)
    if p >= n or Int(b[p]) != 0x2C:  # ','
        return res^
    p += 1
    p = _skip_re_ws(b, n, p)
    var r2 = _scan_number(b, n, p)
    if not r2.ok:
        return res^
    res.y1 = r2.tok
    p = r2.end
    p = _skip_re_ws(b, n, p)
    if p >= n or Int(b[p]) != 0x2C:
        return res^
    p += 1
    p = _skip_re_ws(b, n, p)
    var r3 = _scan_number(b, n, p)
    if not r3.ok:
        return res^
    res.x2 = r3.tok
    p = r3.end
    p = _skip_re_ws(b, n, p)
    if p >= n or Int(b[p]) != 0x2C:
        return res^
    p += 1
    p = _skip_re_ws(b, n, p)
    var r4 = _scan_number(b, n, p)
    if not r4.ok:
        return res^
    res.y2 = r4.tok
    p = r4.end
    p = _skip_re_ws(b, n, p)
    if p >= n or Int(b[p]) != 0x5D:  # ']'
        return res^
    p += 1
    res.matched = True
    res.end = p
    return res^


def _skip_re_ws[o: ImmutOrigin](b: Span[UInt8, o], n: Int, p0: Int) -> Int:
    # Python regex \s matches [ \t\n\r\f\v].
    var p = p0
    while p < n:
        var c = Int(b[p])
        if (
            c == 0x20
            or c == 0x09
            or c == 0x0A
            or c == 0x0D
            or c == 0x0C
            or c == 0x0B
        ):
            p += 1
        else:
            break
    return p


struct _NumScan(Copyable, Movable):
    var ok: Bool
    var end: Int
    var tok: String

    def __init__(out self):
        self.ok = False
        self.end = 0
        self.tok = String("")


def _scan_number[o: ImmutOrigin](b: Span[UInt8, o], n: Int, start: Int) -> _NumScan:
    """Match -?\\d+(?:\\.\\d+)? anchored at start. Requires ≥1 integer digit."""
    var res = _NumScan()
    var p = start
    if p < n and Int(b[p]) == 0x2D:  # '-'
        p += 1
    var digit_start = p
    while p < n and Int(b[p]) >= 0x30 and Int(b[p]) <= 0x39:
        p += 1
    if p == digit_start:
        return res^  # need at least one digit
    # optional .\d+
    if p < n and Int(b[p]) == 0x2E:  # '.'
        var after_dot = p + 1
        var q = after_dot
        while q < n and Int(b[q]) >= 0x30 and Int(b[q]) <= 0x39:
            q += 1
        if q > after_dot:  # at least one fractional digit
            p = q
        # else: a bare '.' is not part of the number (regex needs \d after '.')
    res.ok = True
    res.end = p
    res.tok = _slice_to_string(b, start, p)
    return res^


# ───────────────────────── top-level entry points ─────────────────────────────

def is_ideogram_caption_str(text: String) -> Bool:
    """True if text parses as a JSON object with a compositional_deconstruction
    block (a dict). Mirrors the Python try/except json.loads path."""
    var t = _strip(text)
    if not _starts_with(t, "{"):
        return False
    try:
        var d = parse_json(t)
        if not d.is_object():
            return False
        var decon = d.get("compositional_deconstruction")
        return d.has_key("compositional_deconstruction") and decon.is_object()
    except:
        return False


def digest_caption_string(text: String) -> String:
    """Parse, normalize (migrating old format), and return the compact model-ready
    string. Returns the INPUT UNCHANGED if it is not an Ideogram structured caption
    (plain-text captions pass straight through — byte-for-byte the original)."""
    var t = _strip(text)
    if not _starts_with(t, "{"):
        return text  # prose passthrough: original string, unchanged
    try:
        var data = parse_json(t)
        if not data.is_object():
            return text
        var decon = data.get("compositional_deconstruction")
        if not (data.has_key("compositional_deconstruction") and decon.is_object()):
            return text
        return to_model_string(normalize_caption_dict(data))
    except:
        return text  # json.loads failed → not a caption → unchanged
