# json.stream — streaming / SAX (pull) JSON parser. 100% Mojo, no FFI.
#
# Walks a JSON document WITHOUT building a tree, emitting one event at a time so
# arbitrarily large documents are processed in BOUNDED memory (the cost is the
# parse stack + the path stack + the single scalar payload of the current event,
# not the whole tree). The caller DRIVES the parser:
#
#     var sp = StreamParser(text)
#     while True:
#         var ev = sp.next_event()
#         if ev.kind == EV_EOF: break
#         ... inspect ev.kind / ev.str_val / ev.int_val / ev.float_val ...
#
# Mojo can't store closures as values easily, so this is a PULL model rather than
# a callback/push sink: next_event() returns the next StreamEvent (a small struct
# with a kind tag plus the scalar payload). EV_EOF is the end-of-input sentinel.
#
# Byte/number/string scanning mirrors json.tape: strings keep a raw [offset,len)
# slice and are unescaped lazily; numbers reuse the same exact-fast-path float
# parser and Int64-overflow-to-float promotion. No List[Node] tape is built — the
# only growing structures are the parse-state stack and the JSON-Pointer path
# stack, both O(nesting depth), capped at MAX_DEPTH.

from std.memory import UnsafePointer, alloc

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]

# ── event kinds ──────────────────────────────────────────────────────────────
comptime EV_EOF = 0           # no more events (end-of-input sentinel)
comptime EV_START_OBJECT = 1  # {
comptime EV_END_OBJECT = 2    # }
comptime EV_START_ARRAY = 3   # [
comptime EV_END_ARRAY = 4     # ]
comptime EV_KEY = 5           # object member name (str_val holds the key)
comptime EV_STRING = 6        # string scalar (str_val)
comptime EV_INT = 7           # integer scalar (int_val)
comptime EV_FLOAT = 8         # float scalar (float_val)
comptime EV_BOOL = 9          # boolean scalar (bool_val)
comptime EV_NULL = 10         # null scalar

comptime MAX_DEPTH = 512  # nesting limit — guards against stack-overflow DoS

# parse-state container kinds (internal stack)
comptime _ST_ARR = 0
comptime _ST_OBJ = 1


fn event_name(kind: Int) -> String:
    if kind == EV_EOF:
        return String("eof")
    if kind == EV_START_OBJECT:
        return String("start_object")
    if kind == EV_END_OBJECT:
        return String("end_object")
    if kind == EV_START_ARRAY:
        return String("start_array")
    if kind == EV_END_ARRAY:
        return String("end_array")
    if kind == EV_KEY:
        return String("key")
    if kind == EV_STRING:
        return String("string")
    if kind == EV_INT:
        return String("int")
    if kind == EV_FLOAT:
        return String("float")
    if kind == EV_BOOL:
        return String("bool")
    if kind == EV_NULL:
        return String("null")
    return String("?")


@fieldwise_init
struct StreamEvent(Copyable, Movable):
    """One pull-parser event: a kind tag plus whichever scalar payload applies.

    For EV_KEY / EV_STRING the text is in `str_val`; for EV_INT it's `int_val`;
    for EV_FLOAT it's `float_val`; for EV_BOOL it's `bool_val`. Structural events
    (start/end object/array, null, eof) carry no payload."""

    var kind: Int
    var str_val: String
    var int_val: Int
    var float_val: Float64
    var bool_val: Bool

    @staticmethod
    fn make(kind: Int) -> StreamEvent:
        return StreamEvent(kind, String(""), 0, 0.0, False)

    @staticmethod
    fn make_str(kind: Int, var s: String) -> StreamEvent:
        return StreamEvent(kind, s^, 0, 0.0, False)

    @staticmethod
    fn make_int(v: Int) -> StreamEvent:
        return StreamEvent(EV_INT, String(""), v, 0.0, False)

    @staticmethod
    fn make_float(v: Float64) -> StreamEvent:
        return StreamEvent(EV_FLOAT, String(""), 0, v, False)

    @staticmethod
    fn make_bool(v: Bool) -> StreamEvent:
        return StreamEvent(EV_BOOL, String(""), 0, 0.0, v)


# ── number / unescape helpers (mirrors json.tape; kept local so this file owns
#    its own copy and does not edit tape.mojo) ─────────────────────────────────
fn _pow10(e: Int) -> Float64:
    var r = 1.0
    for _ in range(e):
        r *= 10.0
    return r


fn _hexval(c: Int) -> Int:
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 87
    if c >= 65 and c <= 70:
        return c - 55
    return -1


fn _parse_f64_naive(p: BytePtr, off: Int, ln: Int) -> Float64:
    var i = off
    var end = off + ln
    var neg = False
    if i < end and Int(p[i]) == 0x2D:
        neg = True
        i += 1
    elif i < end and Int(p[i]) == 0x2B:
        i += 1
    var mant = 0.0
    while i < end:
        var c = Int(p[i])
        if c >= 48 and c <= 57:
            mant = mant * 10.0 + Float64(c - 48)
            i += 1
        else:
            break
    if i < end and Int(p[i]) == 0x2E:
        i += 1
        var scale = 0.1
        while i < end:
            var c = Int(p[i])
            if c >= 48 and c <= 57:
                mant += Float64(c - 48) * scale
                scale *= 0.1
                i += 1
            else:
                break
    var exp = 0
    var eneg = False
    if i < end and (Int(p[i]) == 0x65 or Int(p[i]) == 0x45):
        i += 1
        if i < end and Int(p[i]) == 0x2D:
            eneg = True
            i += 1
        elif i < end and Int(p[i]) == 0x2B:
            i += 1
        while i < end:
            var c = Int(p[i])
            if c >= 48 and c <= 57:
                exp = exp * 10 + (c - 48)
                i += 1
            else:
                break
    var result = mant
    if exp != 0:
        var factor = 1.0
        for _ in range(exp):
            factor *= 10.0
        result = result / factor if eneg else result * factor
    return -result if neg else result


fn _parse_f64(p: BytePtr, off: Int, ln: Int) -> Float64:
    """Exact fast path (Clinger) with naive fallback — same algorithm as
    json.tape._parse_f64."""
    var i = off
    var end = off + ln
    var neg = False
    if i < end and Int(p[i]) == 0x2D:
        neg = True
        i += 1
    elif i < end and Int(p[i]) == 0x2B:
        i += 1
    var mant = 0
    var nsig = 0
    var frac = 0
    var seen_dot = False
    var inexact = False
    while i < end:
        var c = Int(p[i])
        if c >= 48 and c <= 57:
            if nsig < 18:
                mant = mant * 10 + (c - 48)
                nsig += 1
                if seen_dot:
                    frac += 1
            else:
                inexact = True
                break
            i += 1
        elif c == 0x2E and not seen_dot:
            seen_dot = True
            i += 1
        else:
            break
    var eexp = 0
    var eneg = False
    if i < end and (Int(p[i]) == 0x65 or Int(p[i]) == 0x45):
        i += 1
        if i < end and Int(p[i]) == 0x2D:
            eneg = True
            i += 1
        elif i < end and Int(p[i]) == 0x2B:
            i += 1
        while i < end:
            var c = Int(p[i])
            if c >= 48 and c <= 57:
                eexp = eexp * 10 + (c - 48)
                i += 1
            else:
                break
    var exp = (-eexp if eneg else eexp) - frac
    if (not inexact) and mant < (1 << 53) and exp >= -22 and exp <= 22:
        var m = Float64(mant)
        var r = m * _pow10(exp) if exp >= 0 else m / _pow10(-exp)
        return -r if neg else r
    return _parse_f64_naive(p, off, ln)


fn _b2s(b: List[UInt8]) -> String:
    var n = len(b)
    if n == 0:
        return String("")
    var out_buf = alloc[UInt8](n)
    for i in range(n):
        out_buf[i] = b[i]
    var s = String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=BytePtr(unsafe_from_address=Int(out_buf)), length=n)))
    out_buf.free()
    return s


fn _unescape(p: BytePtr, off: Int, ln: Int) -> String:
    var has_esc = False
    for i in range(ln):
        if Int(p[off + i]) == 0x5C:
            has_esc = True
            break
    if not has_esc:
        return String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=p + off, length=ln)))
    var buf = List[UInt8]()
    var i = 0
    while i < ln:
        var c = Int(p[off + i])
        if c == 0x5C and i + 1 < ln:
            i += 1
            var e = Int(p[off + i])
            if e == 0x6E:
                buf.append(0x0A)
            elif e == 0x74:
                buf.append(0x09)
            elif e == 0x72:
                buf.append(0x0D)
            elif e == 0x62:
                buf.append(0x08)
            elif e == 0x66:
                buf.append(0x0C)
            elif e == 0x2F:
                buf.append(0x2F)
            elif e == 0x22:
                buf.append(0x22)
            elif e == 0x5C:
                buf.append(0x5C)
            elif e == 0x75:
                var cp = 0
                for _ in range(4):
                    i += 1
                    cp = (cp << 4) | _hexval(Int(p[off + i]))
                if cp >= 0xD800 and cp <= 0xDBFF and i + 6 < ln:
                    i += 2
                    var lo = 0
                    for _ in range(4):
                        i += 1
                        lo = (lo << 4) | _hexval(Int(p[off + i]))
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
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
            i += 1
        else:
            buf.append(UInt8(c))
            i += 1
    return _b2s(buf)


# ── the pull parser ──────────────────────────────────────────────────────────
struct StreamParser(Movable):
    """Pull (SAX) parser. Drive it with repeated `next_event()` calls until the
    returned event's kind is EV_EOF. Memory used is bounded by nesting depth — no
    document-sized tree is ever materialized.

    Internal state machine:
      - `_stack` holds the open containers (_ST_ARR / _ST_OBJ), one entry per
        open `[` / `{`. Its length IS the current nesting depth.
      - `_seen` parallels `_stack`: whether the current container has already had
        at least one element (so we know to expect a ',' before the next).
      - `_expect_key` is set inside an object when the next thing must be a member
        name (a key string) rather than a value.
      - `_path_*` stacks mirror the container nesting for current_path()."""

    var src: String
    var n: Int
    var pos: Int
    var done: Bool
    var _top_done: Bool     # the single top-level value has been fully consumed

    var _stack: List[Int]   # container kinds, len == depth
    var _seen: List[Bool]   # has current container produced an element yet?
    var _expect_key: Bool   # inside an object, awaiting a member name?

    # path tracking: one segment string per open container; for arrays the
    # segment is the running index, for objects the last key seen.
    var _path_seg: List[String]
    var _path_is_arr: List[Bool]
    var _path_idx: List[Int]

    def __init__(out self, var src: String):
        self.src = src^
        self.n = self.src.byte_length()
        self.pos = 0
        self.done = False
        self._top_done = False
        self._stack = List[Int]()
        self._seen = List[Bool]()
        self._expect_key = False
        self._path_seg = List[String]()
        self._path_is_arr = List[Bool]()
        self._path_idx = List[Int]()

    fn _p(self) -> BytePtr:
        # derived live (never cached) so it survives a move of self.src, mirroring
        # json.tape._p()'s reasoning about SSO + ASAP destruction.
        return BytePtr(unsafe_from_address=Int(self.src.unsafe_ptr()))

    def _ws(mut self):
        var p = self._p()
        while self.pos < self.n:
            var c = Int(p[self.pos])
            if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D:
                self.pos += 1
            else:
                break

    def _peek(self) -> Int:
        if self.pos >= self.n:
            return -1
        return Int(self._p()[self.pos])

    def _scan_string_slice(mut self) raises -> String:
        # current byte is the opening quote; returns the unescaped contents and
        # advances past the closing quote.
        var p = self._p()
        var start = self.pos + 1
        self.pos += 1
        while self.pos < self.n:
            var c = Int(p[self.pos])
            if c == 0x5C:
                self.pos += 2
                continue
            if c == 0x22:
                break
            self.pos += 1
        if self.pos >= self.n:
            raise Error("unterminated string at " + String(start))
        var ln = self.pos - start
        self.pos += 1  # closing quote
        return _unescape(p, start, ln)

    def _scan_number(mut self) raises -> StreamEvent:
        var p = self._p()
        var start = self.pos
        var is_float = False
        if Int(p[self.pos]) == 0x2D:
            self.pos += 1
        while self.pos < self.n:
            var c = Int(p[self.pos])
            if c >= 48 and c <= 57:
                self.pos += 1
            elif c == 0x2E or c == 0x65 or c == 0x45:
                is_float = True
                self.pos += 1
            elif c == 0x2B or c == 0x2D:
                self.pos += 1
            else:
                break
        if is_float:
            return StreamEvent.make_float(_parse_f64(p, start, self.pos - start))
        # integer with Int64-overflow -> float promotion (same as tape)
        var neg = Int(p[start]) == 0x2D
        var i = start + 1 if neg else start
        var v = 0
        var overflow = False
        while i < self.pos:
            var nv = v * 10 + (Int(p[i]) - 48)
            if nv < v:
                overflow = True
                break
            v = nv
            i += 1
        if overflow:
            return StreamEvent.make_float(_parse_f64(p, start, self.pos - start))
        return StreamEvent.make_int(-v if neg else v)

    def _scan_literal(mut self, word: String, var ev: StreamEvent) raises -> StreamEvent:
        var p = self._p()
        var wb = word.as_bytes()
        var wn = word.byte_length()
        if self.pos + wn > self.n:
            raise Error("bad literal at " + String(self.pos))
        for i in range(wn):
            if Int(p[self.pos + i]) != Int(wb[i]):
                raise Error("invalid literal at " + String(self.pos))
        self.pos += wn
        return ev^

    def _push(mut self, kind: Int) raises:
        if len(self._stack) + 1 > MAX_DEPTH:
            raise Error("max nesting depth exceeded")
        self._stack.append(kind)
        self._seen.append(False)
        if kind == _ST_ARR:
            self._path_is_arr.append(True)
            self._path_idx.append(0)
            self._path_seg.append(String("0"))
        else:
            self._path_is_arr.append(False)
            self._path_idx.append(0)
            self._path_seg.append(String(""))

    def _pop(mut self):
        _ = self._stack.pop()
        _ = self._seen.pop()
        _ = self._path_seg.pop()
        _ = self._path_is_arr.pop()
        _ = self._path_idx.pop()

    def _mark_seen(mut self):
        if len(self._seen) > 0:
            self._seen[len(self._seen) - 1] = True

    def current_path(self) -> String:
        """JSON-Pointer-ish path to the CURRENT event's value: '/' separated
        segments, object keys verbatim, array elements by index. Root is ''.

        Reported after the event has been produced, so for a scalar/leaf it names
        that leaf; for start_object/start_array it names the container just
        opened; for a key it names the member about to receive a value."""
        if len(self._path_seg) == 0:
            return String("")
        var out = String("")
        for i in range(len(self._path_seg)):
            out += "/"
            out += self._path_seg[i]
        return out

    def _scan_value_event(mut self) raises -> StreamEvent:
        """Reads ONE value token at the current position (already non-ws). For a
        container it emits the start event and pushes state; the matching end is
        emitted later by next_event when the closing bracket is hit."""
        var c = self._peek()
        if c == 0x7B:  # {
            self.pos += 1
            self._push(_ST_OBJ)
            self._expect_key = True
            return StreamEvent.make(EV_START_OBJECT)
        if c == 0x5B:  # [
            self.pos += 1
            self._push(_ST_ARR)
            self._expect_key = False
            return StreamEvent.make(EV_START_ARRAY)
        if c == 0x22:  # "
            return StreamEvent.make_str(EV_STRING, self._scan_string_slice())
        if c == 0x74:  # true
            return self._scan_literal(String("true"), StreamEvent.make_bool(True))
        if c == 0x66:  # false
            return self._scan_literal(String("false"), StreamEvent.make_bool(False))
        if c == 0x6E:  # null
            return self._scan_literal(String("null"), StreamEvent.make(EV_NULL))
        if c == 0x2D or (c >= 48 and c <= 57):
            return self._scan_number()
        raise Error("unexpected byte at " + String(self.pos))

    def next_event(mut self) raises -> StreamEvent:
        """Advance and return the next event. Returns an EV_EOF event once the
        whole (single top-level) document has been consumed."""
        if self.done:
            return StreamEvent.make(EV_EOF)

        self._ws()

        # top-level: either the first/only value, or we've finished it.
        if len(self._stack) == 0:
            if self.pos >= self.n:
                self.done = True
                return StreamEvent.make(EV_EOF)
            # only one top-level value is allowed; if we already emitted it,
            # _stack would be 0 AND we'd have advanced — detect trailing data.
            if self._top_done:
                self._ws()
                if self.pos != self.n:
                    raise Error("trailing data at byte " + String(self.pos))
                self.done = True
                return StreamEvent.make(EV_EOF)
            var ev = self._scan_value_event()
            # a bare scalar at the root completes the document immediately
            if ev.kind != EV_START_OBJECT and ev.kind != EV_START_ARRAY:
                self._top_done = True
            return ev^

        # inside a container.
        var cur = self._stack[len(self._stack) - 1]

        if cur == _ST_OBJ and self._expect_key:
            var c = self._peek()
            if c == 0x7D:  # } empty or after trailing handled below
                self.pos += 1
                self._pop()
                self._after_container_close()
                return StreamEvent.make(EV_END_OBJECT)
            if self._seen[len(self._seen) - 1]:
                # need a ',' before the next key
                if c != 0x2C:
                    raise Error("expected ',' or '}' at " + String(self.pos))
                self.pos += 1
                self._ws()
                c = self._peek()
            if c != 0x22:
                raise Error("expected string key at " + String(self.pos))
            var key = self._scan_string_slice()
            # record key as the current path segment for this object
            self._path_seg[len(self._path_seg) - 1] = key
            self._ws()
            if self._peek() != 0x3A:
                raise Error("expected ':' at " + String(self.pos))
            self.pos += 1
            self._expect_key = False  # next call yields the value
            self._mark_seen()
            return StreamEvent.make_str(EV_KEY, key^)

        # expecting a value (array element, or object member value)
        var c = self._peek()

        if cur == _ST_ARR:
            if c == 0x5D:  # ]
                self.pos += 1
                self._pop()
                self._after_container_close()
                return StreamEvent.make(EV_END_ARRAY)
            if self._seen[len(self._seen) - 1]:
                if c != 0x2C:
                    raise Error("expected ',' or ']' at " + String(self.pos))
                self.pos += 1
                self._ws()
                # advance array index for the path
                self._path_idx[len(self._path_idx) - 1] += 1
                self._path_seg[len(self._path_seg) - 1] = String(self._path_idx[len(self._path_idx) - 1])
                c = self._peek()
            self._mark_seen()
            return self._scan_value_event()

        # cur == _ST_OBJ, value position (after a key:)
        var v = self._scan_value_event()
        # after an object value, the next thing is either ',' (more) or '}'.
        self._expect_key = True
        return v^

    def _after_container_close(mut self):
        # after closing a container, if it was nested its parent has now "seen"
        # an element; the object/array logic above already marked it seen when
        # the value was scanned, so nothing extra is needed except resetting the
        # object key-expectation for the parent.
        if len(self._stack) == 0:
            self._top_done = True
            return
        var parent = self._stack[len(self._stack) - 1]
        if parent == _ST_OBJ:
            self._expect_key = True
        else:
            self._expect_key = False


def stream(var text: String) raises -> StreamParser:
    """Construct a pull parser over `text`."""
    return StreamParser(text^)
