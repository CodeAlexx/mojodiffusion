# json.tape — high-performance flat-tape JSON parser + accessor API. 100% Mojo.
#
# Instead of a tree of heap-allocated nodes (json.value, ~19 MB/s), the document
# is parsed into ONE contiguous List[Node] of small, trivially-copyable nodes
# (simdjson/yyjson "tape" design). Strings are NOT copied — a string node stores
# the (offset, length) of its bytes in the source, materialized lazily on access.
# Measured ~800-860 MB/s on an 8 MB document (~0.6x yyjson 0.12, ~43x the tree).
#
# Each node carries `skip` = the index one past this node's entire subtree, so a
# value of any size occupies [idx, skip); array indexing and object field lookup
# walk siblings by jumping to `skip` — no recursion, no allocation.
#
# Access is a path-based API ON the document — `doc.get_str("user.name")`,
# `doc.get_int("items.0.id")`, `doc.length(path)`, `doc.has(path)`. Accessors
# return OWNED values (nothing escapes the doc), so there's no dangling cursor;
# just keep the JSONDoc alive while you query it.

from std.memory import UnsafePointer, alloc

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]
comptime NodePtr = UnsafePointer[Node, MutExternalOrigin]

comptime T_NULL = 0
comptime T_BOOL = 1
comptime T_INT = 2
comptime T_FLOAT = 3
comptime T_STR = 4
comptime T_ARR = 5
comptime T_OBJ = 6

comptime MAX_DEPTH = 512  # nesting limit — guards against stack-overflow DoS


@fieldwise_init
struct Node(ImplicitlyCopyable, Movable):
    var tag: Int32
    var a: Int  # int value | bool(0/1) | string/float byte-offset into source
    var b: Int  # string/float byte-length | array/object element count
    var skip: Int  # index one past this node's whole subtree


fn _pow10(e: Int) -> Float64:
    """10^e for 0 <= e <= 22, computed exactly. Each 10^k (k<=22) is exactly
    representable in Float64 (= 5^k * 2^k, and 5^22 < 2^52), and multiplying two
    exact f64 whose exact product is representable rounds to that product — so
    repeated *10 stays exact through 10^22."""
    var r = 1.0
    for _ in range(e):
        r *= 10.0
    return r


fn _parse_f64_naive(p: BytePtr, off: Int, ln: Int) -> Float64:
    """Fallback: mantissa*10^exp accumulated in Float64. Last-ULP differences are
    possible (used only outside the exact fast path — very long mantissas / huge
    exponents). For bit-exact round-trip of such extremes, a big-integer parser
    would be needed; web payloads hit the exact fast path below."""
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
    """Non-raising, allocation-free float parse over source bytes.

    Exact fast path (Clinger): collect the significant digits into an integer
    mantissa and a base-10 exponent; when the mantissa is exactly representable
    (< 2^53) and the exponent is in [-22, 22], the result is `mant * 10^exp` (or
    `mant / 10^-exp`) — both operands exact, so the single hardware multiply/
    divide is CORRECTLY ROUNDED. This covers essentially all real-world JSON
    numbers (<=15 significant digits, modest exponent). Anything outside that
    (very long mantissa, |exp| > 22) falls back to _parse_f64_naive."""
    var i = off
    var end = off + ln
    var neg = False
    if i < end and Int(p[i]) == 0x2D:
        neg = True
        i += 1
    elif i < end and Int(p[i]) == 0x2B:
        i += 1

    var mant = 0           # integer mantissa of the significant digits
    var nsig = 0           # significant digits accumulated (cap to avoid overflow)
    var frac = 0           # digits accumulated after the decimal point
    var seen_dot = False
    var inexact = False    # more digits than we can hold exactly -> use fallback
    while i < end:
        var c = Int(p[i])
        if c >= 48 and c <= 57:
            if nsig < 18:  # 18 digits stay within Int64
                mant = mant * 10 + (c - 48)
                nsig += 1
                if seen_dot:
                    frac += 1
            else:
                inexact = True  # extra significant digits dropped -> not exact
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
    # exact fast path: exact mantissa, exact 10^|exp|, one rounded op
    if (not inexact) and mant < (1 << 53) and exp >= -22 and exp <= 22:
        var m = Float64(mant)
        var r = m * _pow10(exp) if exp >= 0 else m / _pow10(-exp)
        return -r if neg else r
    return _parse_f64_naive(p, off, ln)


fn _hexval(c: Int) -> Int:
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 87
    if c >= 65 and c <= 70:
        return c - 55
    return -1


struct JSONDoc(Movable):
    """Owns the parsed tape and the source buffer."""

    var nodes: List[Node]
    var src: String
    var n: Int
    var pos: Int
    var depth: Int

    def __init__(out self, var src: String) raises:
        self.src = src^
        self.n = self.src.byte_length()
        self.nodes = List[Node]()
        self.nodes.reserve(self.n // 8 + 8)  # pre-size: the big perf win
        self.pos = 0
        self.depth = 0
        self._value()
        self._ws()
        if self.pos != self.n:
            raise Error("trailing data at byte " + String(self.pos))

    fn _p(self) -> BytePtr:
        """Source byte pointer, derived from `self.src` on each call rather than
        cached. A cached pointer would dangle after JSONDoc is moved (parse()
        returns by move) if String stored the bytes inline (SSO); deriving here
        always tracks the current buffer. src is never mutated, so callers can
        safely hoist this into a local for a hot loop."""
        return BytePtr(unsafe_from_address=Int(self.src.unsafe_ptr()))

    def _ws(mut self):
        var p = self._p()
        while self.pos < self.n:
            var c = Int(p[self.pos])
            if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D:
                self.pos += 1
            else:
                break

    def _value(mut self) raises:
        self._ws()
        if self.pos >= self.n:
            raise Error("unexpected end of input")
        var c = Int(self._p()[self.pos])
        if c == 0x7B:
            self._obj()
        elif c == 0x5B:
            self._arr()
        elif c == 0x22:
            self._string()
        elif c == 0x74:  # true
            self._lit(4, T_BOOL, 1)
        elif c == 0x66:  # false
            self._lit(5, T_BOOL, 0)
        elif c == 0x6E:  # null
            self._lit(4, T_NULL, 0)
        elif c == 0x2D or (c >= 48 and c <= 57):
            self._number()
        else:
            raise Error("unexpected byte at " + String(self.pos))

    def _lit(mut self, length: Int, tag: Int32, val: Int) raises:
        if self.pos + length > self.n:
            raise Error("bad literal at " + String(self.pos))
        self.pos += length
        self.nodes.append(Node(tag, val, 0, len(self.nodes) + 1))

    def _string(mut self):
        # records raw [offset,len) between the quotes; unescaped lazily on read
        var p = self._p()
        var start = self.pos + 1
        self.pos += 1
        while self.pos < self.n:
            var c = Int(p[self.pos])
            if c == 0x5C:  # backslash: skip escaped byte
                self.pos += 2
                continue
            if c == 0x22:
                break
            self.pos += 1
        var ln = self.pos - start
        self.pos += 1  # closing quote
        self.nodes.append(Node(T_STR, start, ln, len(self.nodes) + 1))

    def _number(mut self):
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
        var nxt = len(self.nodes) + 1
        if is_float:
            self.nodes.append(Node(T_FLOAT, start, self.pos - start, nxt))
        else:
            var neg = Int(p[start]) == 0x2D
            var i = start + 1 if neg else start
            var v = 0
            var overflow = False
            while i < self.pos:
                var nv = v * 10 + (Int(p[i]) - 48)
                if nv < v:  # Int wraps on overflow -> magnitude exceeds Int64
                    overflow = True
                    break
                v = nv
                i += 1
            if overflow:
                # too big for Int — keep it as a float (JS-like), parsed on read.
                # Avoids the silent wrap to a garbage value.
                self.nodes.append(Node(T_FLOAT, start, self.pos - start, nxt))
            else:
                self.nodes.append(Node(T_INT, -v if neg else v, 0, nxt))

    def _arr(mut self) raises:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise Error("max nesting depth exceeded")
        var ci = len(self.nodes)
        self.nodes.append(Node(T_ARR, 0, 0, 0))
        self.pos += 1
        self._ws()
        var cnt = 0
        if self.pos < self.n and Int(self._p()[self.pos]) == 0x5D:
            self.pos += 1
        else:
            while True:
                self._value()
                cnt += 1
                self._ws()
                if self.pos >= self.n:
                    raise Error("unterminated array")
                var c = Int(self._p()[self.pos])
                self.pos += 1
                if c == 0x5D:
                    break
                if c != 0x2C:
                    raise Error("expected ',' or ']' at " + String(self.pos - 1))
        self.nodes[ci].b = cnt
        self.nodes[ci].skip = len(self.nodes)
        self.depth -= 1

    def _obj(mut self) raises:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise Error("max nesting depth exceeded")
        var ci = len(self.nodes)
        self.nodes.append(Node(T_OBJ, 0, 0, 0))
        self.pos += 1
        self._ws()
        var cnt = 0
        if self.pos < self.n and Int(self._p()[self.pos]) == 0x7D:
            self.pos += 1
        else:
            while True:
                self._ws()
                if self.pos >= self.n or Int(self._p()[self.pos]) != 0x22:
                    raise Error("expected string key at " + String(self.pos))
                self._string()  # key node
                self._ws()
                if self.pos >= self.n or Int(self._p()[self.pos]) != 0x3A:
                    raise Error("expected ':' at " + String(self.pos))
                self.pos += 1
                self._value()  # value node
                cnt += 1
                self._ws()
                if self.pos >= self.n:
                    raise Error("unterminated object")
                var c = Int(self._p()[self.pos])
                self.pos += 1
                if c == 0x7D:
                    break
                if c != 0x2C:
                    raise Error("expected ',' or '}' at " + String(self.pos - 1))
        self.nodes[ci].b = cnt
        self.nodes[ci].skip = len(self.nodes)
        self.depth -= 1

    # ── accessor API ─────────────────────────────────────────────────────────
    # Path-based and value-returning, so nothing escapes the document: `doc` is
    # the live receiver of every call and results are owned. Paths are
    # dot-separated; a numeric segment indexes an array. "" / "$" is the root.
    #   doc.get_str("user.name");  doc.get_int("items.0.id");  doc.length("tags")

    fn _np(self) -> NodePtr:
        return NodePtr(unsafe_from_address=Int(self.nodes.unsafe_ptr()))

    fn _atoi(self, s: String) -> Int:
        var v = 0
        var b = s.as_bytes()
        if s.byte_length() == 0:
            return -1
        for i in range(s.byte_length()):
            var c = Int(b[i])
            if c < 48 or c > 57:
                return -1
            v = v * 10 + (c - 48)
        return v

    fn _obj_field(self, oi: Int, key: String) -> Int:
        var np = self._np()
        var p = self._p()
        var nd = np[oi]
        var kb = key.as_bytes()
        var kn = key.byte_length()
        var i = oi + 1
        for _ in range(nd.b):
            var kno = np[i]
            var vidx = kno.skip
            if kno.b == kn:
                var eq = True
                for j in range(kn):
                    if Int(p[kno.a + j]) != Int(kb[j]):
                        eq = False
                        break
                if eq:
                    return vidx
            i = np[vidx].skip
        return -1

    fn _arr_elem(self, ai: Int, i: Int) -> Int:
        var np = self._np()
        var nd = np[ai]
        if i < 0 or i >= nd.b:
            return -1
        var idx = ai + 1
        for _ in range(i):
            idx = np[idx].skip
        return idx

    fn find(self, path: String) -> Int:
        """Resolve a dot path to a node index, or -1 if absent."""
        if path == "" or path == "$":
            return 0
        var np = self._np()
        var segs = path.split(".")
        var idx = 0
        for si in range(len(segs)):
            if idx < 0:
                return -1
            var tag = Int(np[idx].tag)
            var seg = String(segs[si])
            if tag == T_OBJ:
                idx = self._obj_field(idx, seg)
            elif tag == T_ARR:
                idx = self._arr_elem(idx, self._atoi(seg))
            else:
                return -1
        return idx

    fn has(self, path: String) -> Bool:
        return self.find(path) >= 0

    fn kind(self, path: String) -> Int:
        var idx = self.find(path)
        if idx < 0:
            return T_NULL
        return Int(self._np()[idx].tag)

    fn get_int(self, path: String) -> Int:
        var idx = self.find(path)
        if idx < 0:
            return 0
        var nd = self._np()[idx]
        if Int(nd.tag) == T_INT:
            return nd.a
        if Int(nd.tag) == T_FLOAT:
            return Int(_parse_f64(self._p(), nd.a, nd.b))
        return 0

    fn get_float(self, path: String) -> Float64:
        var idx = self.find(path)
        if idx < 0:
            return 0.0
        var nd = self._np()[idx]
        if Int(nd.tag) == T_FLOAT:
            return _parse_f64(self._p(), nd.a, nd.b)
        if Int(nd.tag) == T_INT:
            return Float64(nd.a)
        return 0.0

    fn get_bool(self, path: String) -> Bool:
        var idx = self.find(path)
        if idx < 0:
            return False
        var nd = self._np()[idx]
        return Int(nd.tag) == T_BOOL and nd.a == 1

    fn get_str(self, path: String) -> String:
        var idx = self.find(path)
        if idx < 0:
            return String("")
        var nd = self._np()[idx]
        if Int(nd.tag) != T_STR:
            return String("")
        return _unescape(self._p(), nd.a, nd.b)

    fn length(self, path: String) -> Int:
        var idx = self.find(path)
        if idx < 0:
            return 0
        var nd = self._np()[idx]
        if Int(nd.tag) == T_ARR or Int(nd.tag) == T_OBJ:
            return nd.b
        return 0


fn _unescape(p: BytePtr, off: Int, ln: Int) -> String:
    # fast path: no backslash -> the slice IS the string
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
            elif e == 0x75:  # \uXXXX
                var cp = 0
                for _ in range(4):
                    i += 1
                    cp = (cp << 4) | _hexval(Int(p[off + i]))
                if cp >= 0xD800 and cp <= 0xDBFF and i + 6 < ln:
                    # surrogate pair \uXXXX
                    i += 2  # skip \u
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


fn _b2s(b: List[UInt8]) -> String:
    # Copy bytes into a fresh buffer via a loop that keeps `b` alive across the
    # String construction. Reading b.unsafe_ptr() directly into the String lets
    # ASAP-destruction free `b` first (its last use), yielding freed memory.
    var n = len(b)
    if n == 0:
        return String("")
    var out_buf = alloc[UInt8](n)
    for i in range(n):
        out_buf[i] = b[i]
    var s = String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=BytePtr(unsafe_from_address=Int(out_buf)), length=n)))
    out_buf.free()
    return s


def parse(var text: String) raises -> JSONDoc:
    """Parse `text` into a tape document. Keep the returned doc alive while using
    refs from doc.root()."""
    return JSONDoc(text^)
