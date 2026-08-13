# http.request — an HTTP/1.1 request and a parser for it.
#
# parse_request takes the raw bytes off the wire (as a String) and pulls out the
# method, target, version, headers and body. Header names are lower-cased so
# lookups are case-insensitive, matching httpserver.h's hs_request_header.
# Path params (from the router) and query-string values are exposed on Request.

from std.memory import UnsafePointer, alloc

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]


def byte_substr(s: String, start: Int, end: Int) -> String:
    """Byte substring s[start:end). Mojo String has no slice operator, so we view
    the backing bytes and copy out the range. Used across the http layer."""
    var n = end - start
    if n <= 0:
        return String("")
    var sp = s.as_bytes()
    var base = BytePtr(unsafe_from_address=Int(sp.unsafe_ptr()) + start)
    return String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=base, length=n)))


fn _hexv(c: Int) -> Int:
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 87
    if c >= 65 and c <= 70:
        return c - 55
    return -1


def pct_decode(s: String, plus_as_space: Bool) -> String:
    """Percent-decode (%XX -> byte). In query strings `+` is a space; in paths it
    is literal (pass plus_as_space=False)."""
    var n = s.byte_length()
    if n == 0:
        return String("")
    var sp = s.as_bytes()
    var out = alloc[UInt8](n)  # decoded length <= input length
    var w = 0
    var i = 0
    while i < n:
        var c = Int(sp[i])
        if c == 37 and i + 2 < n:  # '%'
            var hi = _hexv(Int(sp[i + 1]))
            var lo = _hexv(Int(sp[i + 2]))
            if hi >= 0 and lo >= 0:
                out[w] = UInt8(hi * 16 + lo)
                w += 1
                i += 3
                continue
        if plus_as_space and c == 43:  # '+'
            out[w] = 32
            w += 1
            i += 1
            continue
        out[w] = UInt8(c)
        w += 1
        i += 1
    var res = String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=BytePtr(unsafe_from_address=Int(out)), length=w)))
    out.free()
    return res


fn _lower_b(c: Int) -> Int:
    """ASCII lower-case of one byte (leaves non-letters, incl. UTF-8 bytes, as-is)."""
    if c >= 65 and c <= 90:
        return c + 32
    return c


fn _header_limit(raw: String) -> Int:
    """Byte offset of the blank-line terminator (start of "\\r\\n\\r\\n"), or the
    full length if not present yet. Bounds header scans so we never read into a
    (possibly binary) body."""
    var sp = raw.as_bytes()
    var n = raw.byte_length()
    var i = 0
    while i + 3 < n:
        if (Int(sp[i]) == 13 and Int(sp[i + 1]) == 10
                and Int(sp[i + 2]) == 13 and Int(sp[i + 3]) == 10):
            return i
        i += 1
    return n


fn _ci_find(raw: String, needle: String, start: Int, limit: Int) -> Int:
    """Case-insensitive byte search for an ASCII `needle` (must be lowercase) in
    raw[start:limit]. Pure byte comparison — never decodes UTF-8, so it is safe
    over a buffer whose body holds arbitrary binary bytes (unlike String.lower /
    String.find, which assert on non-codepoint boundaries)."""
    var sp = raw.as_bytes()
    var m = needle.byte_length()
    if m == 0:
        return start
    var np = needle.as_bytes()
    var i = start
    while i + m <= limit:
        var ok = True
        for j in range(m):
            if _lower_b(Int(sp[i + j])) != Int(np[j]):
                ok = False
                break
        if ok:
            return i
        i += 1
    return -1


def content_length(raw: String) -> Int:
    """Value of the Content-Length header, or 0 if absent/unparseable. Scans only
    the header region with a byte-level case-insensitive match, so a binary body
    can't corrupt the lookup (and there's no whole-buffer lower-case allocation)."""
    var sp = raw.as_bytes()
    var limit = _header_limit(raw)
    var idx = _ci_find(raw, "content-length:", 0, limit)
    if idx < 0:
        return 0
    var i = idx + 15  # len("content-length:")
    while i < limit and (Int(sp[i]) == 32 or Int(sp[i]) == 9):  # skip SP / TAB
        i += 1
    var v = 0
    while i < limit:
        var c = Int(sp[i])
        if c < 48 or c > 57:  # non-digit
            break
        v = v * 10 + (c - 48)
        i += 1
    return v


def has_chunked(raw: String) -> Bool:
    """Whether the request uses Transfer-Encoding: chunked (header-region scan)."""
    var limit = _header_limit(raw)
    var te = _ci_find(raw, "transfer-encoding:", 0, limit)
    if te < 0:
        return False
    return _ci_find(raw, "chunked", te, limit) >= 0


def keep_alive_for(raw: String) -> Bool:
    """Whether to keep the connection alive, from the request line's HTTP version
    and the Connection header. Byte-level + non-raising (binary-body safe), so it
    can be used off the raw buffer without parsing — e.g. in an async coroutine
    body where a raising call would mis-lower. HTTP/1.1 defaults to keep-alive,
    HTTP/1.0 to close; an explicit Connection header overrides."""
    var limit = _header_limit(raw)
    var ci = _ci_find(raw, "connection:", 0, limit)
    if ci >= 0:
        if _ci_find(raw, "close", ci, limit) >= 0:
            return False
        if _ci_find(raw, "keep-alive", ci, limit) >= 0:
            return True
    return _ci_find(raw, "http/1.1", 0, limit) >= 0


fn _list_to_str(b: List[UInt8]) -> String:
    var n = len(b)
    if n == 0:
        return String("")
    var buf = alloc[UInt8](n)
    for i in range(n):
        buf[i] = b[i]
    var s = String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=BytePtr(unsafe_from_address=Int(buf)), length=n)))
    buf.free()
    return s


def chunked_end(raw: String, start: Int) -> Int:
    """Index one past a complete chunked body that begins at `start`, or -1 if it
    isn't fully present yet."""
    var sp = raw.as_bytes()
    var n = raw.byte_length()
    var i = start
    while True:
        var k = i
        var crlf = -1
        while k + 1 < n:
            if Int(sp[k]) == 13 and Int(sp[k + 1]) == 10:
                crlf = k
                break
            k += 1
        if crlf < 0:
            return -1
        var size = 0
        var j = i
        while j < crlf:
            var d = _hexv(Int(sp[j]))
            if d < 0:
                break  # chunk extension (';') or end of hex
            size = size * 16 + d
            j += 1
        var data_start = crlf + 2
        if size == 0:
            return data_start + 2 if data_start + 2 <= n else -1  # trailing CRLF
        var data_end = data_start + size
        if data_end + 2 > n:
            return -1
        i = data_end + 2


def chunked_decode(raw: String, start: Int) -> String:
    """Concatenate the data of a (complete) chunked body beginning at `start`."""
    var sp = raw.as_bytes()
    var n = raw.byte_length()
    var out = List[UInt8]()
    var i = start
    while True:
        var k = i
        var crlf = -1
        while k + 1 < n:
            if Int(sp[k]) == 13 and Int(sp[k + 1]) == 10:
                crlf = k
                break
            k += 1
        if crlf < 0:
            break
        var size = 0
        var j = i
        while j < crlf:
            var d = _hexv(Int(sp[j]))
            if d < 0:
                break
            size = size * 16 + d
            j += 1
        var data_start = crlf + 2
        if size == 0:
            break
        for x in range(data_start, data_start + size):
            out.append(sp[x])
        i = data_start + size + 2
    return _list_to_str(out)


def is_request_complete(raw: String) -> Bool:
    """True once the accumulated bytes hold a full request: headers terminated by
    a blank line, plus a complete body (Content-Length bytes, or a chunked body
    ending in the zero-size chunk)."""
    var sep = raw.find("\r\n\r\n")
    if sep < 0:
        return False  # headers not finished
    if has_chunked(raw):
        return chunked_end(raw, sep + 4) >= 0
    var cl = content_length(raw)
    if cl == 0:
        return True
    return raw.byte_length() >= sep + 4 + cl


def request_consumed_len(raw: String) -> Int:
    """How many bytes the first complete request occupies (headers + body), or -1
    if `raw` does not yet hold a complete request. Lets the keep-alive loop slice
    one request off the buffer and leave the rest (pipelining)."""
    var sep = raw.find("\r\n\r\n")
    if sep < 0:
        return -1
    var header_end = sep + 4
    if has_chunked(raw):
        return chunked_end(raw, header_end)  # -1 if incomplete
    var cl = content_length(raw)
    if raw.byte_length() < header_end + cl:
        return -1
    return header_end + cl


@fieldwise_init
struct Request(Copyable, Movable):
    """A parsed HTTP request."""

    var method: String
    var target: String
    var version: String
    var headers: Dict[String, String]
    var body: String
    var params: Dict[String, String]  # path params filled in by the router

    def header(self, key: String) raises -> String:
        """Case-insensitive header lookup; "" if absent."""
        var k = key.lower()
        if k in self.headers:
            return self.headers[k]
        return String("")

    def param(self, key: String) raises -> String:
        """Path parameter captured by the route pattern (e.g. {id}); "" if none."""
        if key in self.params:
            return self.params[key]
        return String("")

    def path(self) -> String:
        """The target with any query string stripped, percent-decoded
        (e.g. /search?q=x -> /search; /a%2Fb -> /a/b)."""
        var q = self.target.find("?")
        var raw = self.target if q < 0 else byte_substr(self.target, 0, q)
        return pct_decode(raw, False)

    def query(self, key: String) -> String:
        """Value of a query-string key, percent-decoded (`+` -> space); "" if absent."""
        var q = self.target.find("?")
        if q < 0:
            return String("")
        var qs = byte_substr(self.target, q + 1, self.target.byte_length())
        var pairs = qs.split("&")
        for i in range(len(pairs)):
            var pair = String(pairs[i])
            var eq = pair.find("=")
            if eq < 0:
                continue
            if byte_substr(pair, 0, eq) == key:
                return pct_decode(byte_substr(pair, eq + 1, pair.byte_length()), True)
        return String("")


def parse_request(raw: String) raises -> Request:
    """Parse one HTTP/1.1 request. Splits head from body on the blank line, then
    the request line and headers. Malformed input raises. `params` starts empty;
    the router fills it after matching."""
    if raw.byte_length() == 0:
        raise Error("empty request")

    var sep = raw.find("\r\n\r\n")
    var head = raw if sep < 0 else byte_substr(raw, 0, sep)
    var body = String("")
    if sep >= 0:
        if has_chunked(raw):
            body = chunked_decode(raw, sep + 4)  # de-chunk into the real body
        else:
            body = byte_substr(raw, sep + 4, raw.byte_length())

    var lines = head.split("\r\n")
    if len(lines) == 0:
        raise Error("no request line")

    var line0 = String(lines[0])
    var parts = line0.split(" ")
    if len(parts) < 3:
        raise Error("malformed request line: " + line0)
    var method = String(parts[0])
    var target = String(parts[1])
    var version = String(parts[2])

    var headers = Dict[String, String]()
    for i in range(1, len(lines)):
        var line = String(lines[i])
        var c = line.find(":")
        if c <= 0:
            continue  # skip blank / malformed header lines
        var key = String(byte_substr(line, 0, c).strip()).lower()
        var val = String(byte_substr(line, c + 1, line.byte_length()).strip())
        headers[key] = val

    return Request(
        method, target, version, headers^, body, Dict[String, String]()
    )
