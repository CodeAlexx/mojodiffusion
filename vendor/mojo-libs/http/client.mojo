# http.client — a real blocking HTTP/1.1 client (pure Mojo).
#
# Capabilities:
#   * DNS resolution (getaddrinfo) with a dotted-quad literal fast path
#   * URL parsing (http://host[:port][/path])
#   * Content-Length AND Transfer-Encoding: chunked response framing
#   * gzip / deflate response decompression (via graphics.inflate)
#   * recv() timeouts (SO_RCVTIMEO)
#   * 3xx redirect following (capped, with a Location)
#   * keep-alive connection reuse (Client struct pools one live socket)
#
# TLS / https:// IS supported via net.tls (OpenSSL client side). Build/link with
#   -Xlinker -lssl -Xlinker -lcrypto
# https connections do a real handshake (SNI + optional CA verification) and the
# request/response flow exactly as cleartext, through a unified `Conn`.
#
# Built on net.tcp / net.socket / net.tls. The original minimal
# `get(ip, port, path, host)` API is preserved (see ClientResponse + the 4-arg
# `get` overload at the bottom).

from std.ffi import external_call
from std.memory import alloc, UnsafePointer
from net.tcp import tcp_connect, _parse_ipv4
from net.socket import Socket
from net.syscalls import BytePtr, sys_socket, sys_connect, errno_str, AF_INET, SOCK_STREAM
from net.tls import (
    make_client_ctx,
    client_connect,
    conn_new as tls_conn_new,
    conn_read as tls_read,
    conn_send_bytes as tls_send,
    conn_free as tls_free,
    free_ctx as tls_free_ctx,
)
from graphics.inflate import inflate, zlib_inflate


# ─────────────────────────────────────────────────────────────────────────────
# DNS
# ─────────────────────────────────────────────────────────────────────────────

def _is_dotted_quad(host: String) -> Bool:
    """Cheap check: all chars are digits or dots, and there are exactly 3 dots."""
    var sp = host.as_bytes()
    var n = host.byte_length()
    if n == 0:
        return False
    var dots = 0
    for i in range(n):
        var c = Int(sp[i])
        if c == 46:  # '.'
            dots += 1
        elif c < 48 or c > 57:
            return False
    return dots == 3


def resolve_host(host: String) raises -> String:
    """Resolve `host` to a dotted-quad IPv4 string. A dotted-quad literal is
    returned as-is; a name is resolved via getaddrinfo(3).

    getaddrinfo(node, service, *hints, **res): we pass hints=NULL, service=NULL.
    `res` is filled with a pointer to a `struct addrinfo`. On Linux x86-64:
        struct addrinfo {            // sizes / offsets
            int    ai_flags;         // 0
            int    ai_family;        // 4
            int    ai_socktype;      // 8
            int    ai_protocol;      // 12
            socklen_t ai_addrlen;    // 16
            struct sockaddr *ai_addr;// 24  (8-byte pointer)
            char  *ai_canonname;     // 32
            struct addrinfo *ai_next;// 40
        };
    ai_addr points at a `struct sockaddr_in` whose bytes 4..7 are the IPv4 addr
    in network byte order. We pull those four bytes back out."""
    if _is_dotted_quad(host):
        # Validate it parses, then return it unchanged.
        _ = _parse_ipv4(host)
        return host

    # NUL-terminate the host name for the C call.
    var hn = host.byte_length()
    var hbuf = alloc[UInt8](hn + 1)
    var hs = host.as_bytes()
    for i in range(hn):
        hbuf[i] = hs[i]
    hbuf[hn] = 0

    # res: a single pointer slot the kernel fills with the addrinfo* head.
    var res = alloc[Int](1)
    res[0] = 0
    var rc = external_call["getaddrinfo", Int32](
        BytePtr(unsafe_from_address=Int(hbuf)),       # node
        BytePtr(unsafe_from_address=Int(0)),          # service = NULL
        BytePtr(unsafe_from_address=Int(0)),          # hints = NULL
        BytePtr(unsafe_from_address=Int(res)),        # &res
    )
    hbuf.free()
    if rc != 0 or res[0] == 0:
        res.free()
        raise Error("resolve_host: getaddrinfo failed for '" + host + "' (rc=" + String(Int(rc)) + ")")

    var ai = res[0]  # addrinfo*
    # ai_addr is the 8-byte pointer at offset 24.
    var ai_addr_pp = UnsafePointer[Int, MutExternalOrigin](unsafe_from_address=ai + 24)
    var sockaddr = ai_addr_pp[]  # struct sockaddr_in*
    var sa = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=sockaddr)
    # sockaddr_in: family(2) port(2) addr(4 @ offset 4)
    var b0 = Int(sa[4])
    var b1 = Int(sa[5])
    var b2 = Int(sa[6])
    var b3 = Int(sa[7])
    var ip = String(b0) + "." + String(b1) + "." + String(b2) + "." + String(b3)

    _ = external_call["freeaddrinfo", Int32](BytePtr(unsafe_from_address=Int(ai)))
    res.free()
    return ip


# ─────────────────────────────────────────────────────────────────────────────
# URL parsing
# ─────────────────────────────────────────────────────────────────────────────

@fieldwise_init
struct ParsedUrl(Copyable, Movable):
    var scheme: String   # "http" or "https"
    var host: String
    var port: Int
    var path: String     # includes leading '/' and any query string


def parse_url(url: String) raises -> ParsedUrl:
    """Minimal URL split: scheme://host[:port][/path]. Inlined on purpose to
    avoid a build-time dependency on http/url.mojo."""
    var scheme = String("http")
    var rest = url
    var si = url.find("://")
    if si >= 0:
        scheme = _byte_slice(url, 0, si)
        rest = _byte_slice(url, si + 3, url.byte_length())
    # rest = host[:port][/path]
    var slash = rest.find("/")
    var authority = rest if slash < 0 else _byte_slice(rest, 0, slash)
    var path = String("/") if slash < 0 else _byte_slice(rest, slash, rest.byte_length())
    if path.byte_length() == 0:
        path = String("/")
    # authority = host[:port]
    var host = authority
    var port = 80 if scheme == "http" else 443
    var colon = authority.find(":")
    if colon >= 0:
        host = _byte_slice(authority, 0, colon)
        var ps = _byte_slice(authority, colon + 1, authority.byte_length())
        port = _atoi(ps)
    return ParsedUrl(scheme, host, port, path)


# ─────────────────────────────────────────────────────────────────────────────
# small byte helpers (binary-safe; no String.lower / find on bodies)
# ─────────────────────────────────────────────────────────────────────────────

def _byte_slice(s: String, start: Int, end: Int) -> String:
    var n = end - start
    if n <= 0:
        return String("")
    var sp = s.as_bytes()
    var base = BytePtr(unsafe_from_address=Int(sp.unsafe_ptr()) + start)
    return String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=base, length=n)))


def _atoi(s: String) -> Int:
    var v = 0
    var sb = s.as_bytes()
    for i in range(s.byte_length()):
        var c = Int(sb[i])
        if c < 48 or c > 57:
            break
        v = v * 10 + (c - 48)
    return v


def _lower(c: Int) -> Int:
    if c >= 65 and c <= 90:
        return c + 32
    return c


def _hexval(c: Int) -> Int:
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 87
    if c >= 65 and c <= 70:
        return c - 55
    return -1


def _bytes_to_str(b: List[UInt8], start: Int, end: Int) -> String:
    """Materialize b[start:end) as a (binary-safe) String."""
    var n = end - start
    if n <= 0:
        return String("")
    var buf = alloc[UInt8](n)
    for i in range(n):
        buf[i] = b[start + i]
    var s = String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=BytePtr(unsafe_from_address=Int(buf)), length=n)))
    buf.free()
    return s


def _find_crlfcrlf(b: List[UInt8]) -> Int:
    """Index of the start of the first \\r\\n\\r\\n, or -1."""
    var n = len(b)
    var i = 0
    while i + 3 < n:
        if Int(b[i]) == 13 and Int(b[i + 1]) == 10 and Int(b[i + 2]) == 13 and Int(b[i + 3]) == 10:
            return i
        i += 1
    return -1


# ─────────────────────────────────────────────────────────────────────────────
# Public types
# ─────────────────────────────────────────────────────────────────────────────

@fieldwise_init
struct Header(Copyable, Movable):
    var name: String   # lower-cased
    var value: String


struct HttpResponse(Movable):
    var status: Int
    var headers: List[Header]
    var body: List[UInt8]

    def __init__(out self, status: Int, var headers: List[Header], var body: List[UInt8]):
        self.status = status
        self.headers = headers^
        self.body = body^

    def header(self, name: String) -> String:
        """Case-insensitive header lookup; "" if absent."""
        var key = String("")
        var nb = name.as_bytes()
        for i in range(name.byte_length()):
            key += chr(_lower(Int(nb[i])))
        for i in range(len(self.headers)):
            if self.headers[i].name == key:
                return self.headers[i].value
        return String("")

    def text(self) -> String:
        """Body as a String (assumes UTF-8/text)."""
        return _bytes_to_str(self.body, 0, len(self.body))


@fieldwise_init
struct ClientOptions(Copyable, Movable):
    var timeout_ms: Int
    var max_redirects: Int
    var decompress: Bool
    var verify_tls: Bool   # for https: verify the server certificate chain + host

    def __init__(out self):
        self.timeout_ms = 10000
        self.max_redirects = 5
        self.decompress = True
        self.verify_tls = True


# ─────────────────────────────────────────────────────────────────────────────
# Conn — a unified read/write channel over either a plain Socket or a TLS SSL*.
# Owns the socket; for TLS also owns the SSL* and its SSL_CTX*, freed on drop.
# ─────────────────────────────────────────────────────────────────────────────

struct Conn(Movable):
    var sock: Socket
    var tls: Bool
    var ssl: Int     # SSL* address (0 when plain)
    var ctx: Int     # SSL_CTX* address (0 when plain)

    def __init__(out self, var sock: Socket):
        self.sock = sock^
        self.tls = False
        self.ssl = 0
        self.ctx = 0

    @staticmethod
    def open_plain(ip: String, port: Int) raises -> Conn:
        return Conn(tcp_connect(ip, port))

    @staticmethod
    def open_tls(ip: String, port: Int, host: String, verify: Bool) raises -> Conn:
        var sock = tcp_connect(ip, port)
        var ctx = make_client_ctx(verify)
        var ssl = tls_conn_new(ctx, sock.fd)
        try:
            client_connect(ssl, host, verify)
        except e:
            tls_free(ssl)
            tls_free_ctx(ctx)
            raise e^
        var c = Conn(sock^)
        c.tls = True
        c.ssl = ssl
        c.ctx = ctx
        return c^

    def set_recv_timeout(self, ms: Int) raises:
        self.sock.set_recv_timeout(ms)

    def read_into(mut self, bp: BytePtr, cap: Int) -> Int:
        """>0 = bytes read; <=0 = closed / would-block / error (caller stops)."""
        if self.tls:
            return tls_read(self.ssl, bp, cap)
        return self.sock.recv_into(bp, cap)

    def send(mut self, p: BytePtr, n: Int) -> Int:
        if self.tls:
            return tls_send(self.ssl, p, n)
        return self.sock.send_bytes(p, n)

    def __del__(deinit self):
        # Free the SSL* + CTX before the Socket field's own __del__ closes the fd.
        if self.tls:
            tls_free(self.ssl)
            tls_free_ctx(self.ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Reading a full response off a socket
# ─────────────────────────────────────────────────────────────────────────────

def _recv_all(mut conn: Conn, want_close: Bool) raises -> List[UInt8]:
    """Read a complete HTTP response into a byte buffer.

    Strategy: pull bytes until we have the header terminator, then decide framing
    (Content-Length vs chunked vs read-until-close) and keep reading exactly until
    the message is complete. Returns the FULL buffer (headers + raw body)."""
    var acc = List[UInt8]()
    var bufcap = 65536
    var buf = alloc[UInt8](bufcap)
    var bp = BytePtr(unsafe_from_address=Int(buf))

    var header_end = -1
    var content_length = -1
    var is_chunked = False
    var conn_close = want_close

    while True:
        # Do we know how to decide completion yet?
        if header_end < 0:
            header_end = _find_crlfcrlf(acc)
            if header_end >= 0:
                # parse framing headers (only the header region, byte-level)
                var hdr = _parse_header_framing(acc, header_end)
                content_length = hdr[0]
                is_chunked = hdr[1] == 1
                if hdr[2] == 1:
                    conn_close = True
                elif hdr[2] == 0:
                    conn_close = False

        if header_end >= 0:
            var body_start = header_end + 4
            if is_chunked:
                if _chunked_complete(acc, body_start):
                    break
            elif content_length >= 0:
                if len(acc) >= body_start + content_length:
                    break
            elif not conn_close:
                # No length, no chunked, keep-alive: per HTTP/1.1 there is no body
                # framing → treat as empty body (e.g. 204/304 or HEAD).
                break
            # else: read-until-close → fall through and keep recv'ing

        var k = conn.read_into(bp, bufcap)
        if k <= 0:
            # peer closed (or timeout). For read-until-close this is the end.
            if k < 0 and header_end < 0:
                buf.free()
                raise Error("recv failed before any response: " + errno_str())
            break
        for i in range(k):
            acc.append(buf[i])

    buf.free()
    if header_end < 0:
        raise Error("incomplete response: no header terminator")
    return acc^


def _parse_header_framing(b: List[UInt8], header_end: Int) -> List[Int]:
    """Scan the header region b[0:header_end] for Content-Length / chunked /
    Connection. Returns [content_length(-1=absent), is_chunked(0/1),
    conn(-1 unknown,0 keep-alive,1 close)]."""
    var content_length = -1
    var is_chunked = 0
    var conn = -1

    # iterate header lines after the status line
    var i = 0
    # skip status line
    while i + 1 < header_end:
        if Int(b[i]) == 13 and Int(b[i + 1]) == 10:
            i += 2
            break
        i += 1

    while i < header_end:
        # find end of this line
        var j = i
        while j < header_end and not (Int(b[j]) == 13 and Int(b[j + 1]) == 10):
            j += 1
        # line is b[i:j]  (j lands on the line's terminating CR, or at header_end
        # for the final header line whose CRLF coincides with the blank-line CRLF)
        var colon = -1
        var c = i
        while c < j:
            if Int(b[c]) == 58:  # ':'
                colon = c
                break
            c += 1
        if colon > i:
            # name (lower) and value
            var name = String("")
            for x in range(i, colon):
                name += chr(_lower(Int(b[x])))
            # value, skip leading spaces/tabs
            var vstart = colon + 1
            while vstart < j and (Int(b[vstart]) == 32 or Int(b[vstart]) == 9):
                vstart += 1
            if name == "content-length":
                var v = 0
                for x in range(vstart, j):
                    var d = Int(b[x])
                    if d < 48 or d > 57:
                        break
                    v = v * 10 + (d - 48)
                content_length = v
            elif name == "transfer-encoding":
                # contains "chunked"?
                if _region_contains_lower(b, vstart, j, "chunked"):
                    is_chunked = 1
            elif name == "connection":
                if _region_contains_lower(b, vstart, j, "close"):
                    conn = 1
                elif _region_contains_lower(b, vstart, j, "keep-alive"):
                    conn = 0
        i = j + 2  # skip CRLF
    return [content_length, is_chunked, conn]


def _region_contains_lower(b: List[UInt8], start: Int, end: Int, needle: String) -> Bool:
    """Case-insensitive substring search of `needle` (lowercase) in b[start:end)."""
    var m = needle.byte_length()
    if m == 0:
        return True
    var np = needle.as_bytes()
    var i = start
    while i + m <= end:
        var ok = True
        for k in range(m):
            if _lower(Int(b[i + k])) != Int(np[k]):
                ok = False
                break
        if ok:
            return True
        i += 1
    return False


def _chunked_complete(b: List[UInt8], start: Int) -> Bool:
    """True once a complete chunked body (ending in the 0-size chunk + CRLF) is
    present from `start`."""
    var n = len(b)
    var i = start
    while True:
        # read hex size line
        var k = i
        var crlf = -1
        while k + 1 < n:
            if Int(b[k]) == 13 and Int(b[k + 1]) == 10:
                crlf = k
                break
            k += 1
        if crlf < 0:
            return False
        var size = 0
        var j = i
        while j < crlf:
            var d = _hexval(Int(b[j]))
            if d < 0:
                break  # chunk extension ';' or end of hex
            size = size * 16 + d
            j += 1
        var data_start = crlf + 2
        if size == 0:
            # need trailing CRLF (final)
            return data_start + 2 <= n
        var data_end = data_start + size
        if data_end + 2 > n:
            return False
        i = data_end + 2


def _chunked_decode(b: List[UInt8], start: Int) -> List[UInt8]:
    """Concatenate the data of a complete chunked body beginning at `start`."""
    var n = len(b)
    var out = List[UInt8]()
    var i = start
    while True:
        var k = i
        var crlf = -1
        while k + 1 < n:
            if Int(b[k]) == 13 and Int(b[k + 1]) == 10:
                crlf = k
                break
            k += 1
        if crlf < 0:
            break
        var size = 0
        var j = i
        while j < crlf:
            var d = _hexval(Int(b[j]))
            if d < 0:
                break
            size = size * 16 + d
            j += 1
        var data_start = crlf + 2
        if size == 0:
            break
        for x in range(data_start, data_start + size):
            out.append(b[x])
        i = data_start + size + 2
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# gzip / deflate decompression
# ─────────────────────────────────────────────────────────────────────────────

def _gunzip(b: List[UInt8]) raises -> List[UInt8]:
    """Strip the 10-byte gzip header (+ optional extra/name/comment/HCRC fields)
    and the 8-byte trailer, then raw-inflate. RFC 1952."""
    var n = len(b)
    if n < 18:
        raise Error("gunzip: stream too short")
    if Int(b[0]) != 0x1F or Int(b[1]) != 0x8B:
        raise Error("gunzip: bad magic")
    if Int(b[2]) != 8:
        raise Error("gunzip: not deflate (CM != 8)")
    var flg = Int(b[3])
    var pos = 10
    # FEXTRA
    if (flg & 0x04) != 0:
        var xlen = Int(b[pos]) | (Int(b[pos + 1]) << 8)
        pos += 2 + xlen
    # FNAME (NUL-terminated)
    if (flg & 0x08) != 0:
        while pos < n and Int(b[pos]) != 0:
            pos += 1
        pos += 1
    # FCOMMENT (NUL-terminated)
    if (flg & 0x10) != 0:
        while pos < n and Int(b[pos]) != 0:
            pos += 1
        pos += 1
    # FHCRC
    if (flg & 0x02) != 0:
        pos += 2
    # deflate body = b[pos : n-8]
    var body = List[UInt8]()
    for i in range(pos, n - 8):
        body.append(b[i])
    return inflate(body)


def _inflate_deflate_encoding(b: List[UInt8]) raises -> List[UInt8]:
    """HTTP `Content-Encoding: deflate`. Servers send either a zlib-wrapped
    stream (correct per RFC 7230) or, due to a well-known ambiguity, raw deflate.
    Try zlib first, fall back to raw."""
    try:
        return zlib_inflate(b)
    except:
        return inflate(b)


# ─────────────────────────────────────────────────────────────────────────────
# Core request logic
# ─────────────────────────────────────────────────────────────────────────────

def _build_request(method: String, host: String, port: Int, path: String,
                   extra_headers: List[Header], body: List[UInt8],
                   decompress: Bool, omit_port: Bool = False) -> String:
    var hostval = host
    if not omit_port:
        hostval = host + ":" + String(port)
    var req = method + " " + path + " HTTP/1.1\r\n"
    req += "Host: " + hostval + "\r\n"
    if decompress:
        req += "Accept-Encoding: gzip, deflate\r\n"
    else:
        req += "Accept-Encoding: identity\r\n"
    var has_ua = False
    for i in range(len(extra_headers)):
        if extra_headers[i].name == "user-agent":
            has_ua = True
        req += extra_headers[i].name + ": " + extra_headers[i].value + "\r\n"
    if not has_ua:
        req += "User-Agent: mojo-http-client/1.0\r\n"
    if len(body) > 0:
        req += "Content-Length: " + String(len(body)) + "\r\n"
    req += "Connection: keep-alive\r\n\r\n"
    return req


def _send_all(mut conn: Conn, data: String) raises:
    var n = data.byte_length()
    if n == 0:
        return
    var sbuf = alloc[UInt8](n)
    var rs = data.as_bytes()
    for i in range(n):
        sbuf[i] = rs[i]
    var sp = BytePtr(unsafe_from_address=Int(sbuf))
    var total = 0
    while total < n:
        var w = conn.send(sp + total, n - total)
        if w <= 0:
            sbuf.free()
            raise Error("send failed: " + errno_str())
        total += w
    sbuf.free()


def _send_body(mut conn: Conn, body: List[UInt8]) raises:
    var n = len(body)
    if n == 0:
        return
    var sbuf = alloc[UInt8](n)
    for i in range(n):
        sbuf[i] = body[i]
    var sp = BytePtr(unsafe_from_address=Int(sbuf))
    var total = 0
    while total < n:
        var w = conn.send(sp + total, n - total)
        if w <= 0:
            sbuf.free()
            raise Error("send body failed: " + errno_str())
        total += w
    sbuf.free()


def _status_of(raw: List[UInt8], header_end: Int) -> Int:
    """Parse the numeric status code out of the status line."""
    # status line: HTTP/1.1 <code> <reason>
    var i = 0
    # skip to first space
    while i < header_end and Int(raw[i]) != 32:
        i += 1
    i += 1
    var status = 0
    while i < header_end:
        var c = Int(raw[i])
        if c < 48 or c > 57:
            break
        status = status * 10 + (c - 48)
        i += 1
    return status


def _collect_headers(raw: List[UInt8], header_end: Int) -> List[Header]:
    var headers = List[Header]()
    var i = 0
    # skip status line
    while i + 1 < header_end:
        if Int(raw[i]) == 13 and Int(raw[i + 1]) == 10:
            i += 2
            break
        i += 1
    while i < header_end:
        var j = i
        while j < header_end and not (Int(raw[j]) == 13 and Int(raw[j + 1]) == 10):
            j += 1
        var colon = -1
        var c = i
        while c < j:
            if Int(raw[c]) == 58:
                colon = c
                break
            c += 1
        if colon > i:
            var name = String("")
            for x in range(i, colon):
                name += chr(_lower(Int(raw[x])))
            var vstart = colon + 1
            while vstart < j and (Int(raw[vstart]) == 32 or Int(raw[vstart]) == 9):
                vstart += 1
            var value = _bytes_to_str(raw, vstart, j)
            headers.append(Header(name, value))
        i = j + 2
    return headers^


def _location_header(headers: List[Header]) -> String:
    for i in range(len(headers)):
        if headers[i].name == "location":
            return headers[i].value
    return String("")


def _content_encoding(headers: List[Header]) -> String:
    for i in range(len(headers)):
        if headers[i].name == "content-encoding":
            var v = headers[i].value
            var out = String("")
            var vb = v.as_bytes()
            for k in range(v.byte_length()):
                out += chr(_lower(Int(vb[k])))
            return out
    return String("")


def _resolve_redirect(base: ParsedUrl, location: String) raises -> String:
    """Build an absolute URL for a Location that may be absolute or relative."""
    if location.find("://") >= 0:
        return location
    if location.byte_length() > 0 and Int(location.as_bytes()[0]) == 47:
        var hostport = base.host
        if base.port != 80:
            hostport = base.host + ":" + String(base.port)
        return base.scheme + "://" + hostport + location
    # relative to current dir
    var dir = base.path
    var slash = -1
    var pb = dir.as_bytes()
    for i in range(dir.byte_length()):
        if Int(pb[i]) == 47:
            slash = i
    var prefix = _byte_slice(dir, 0, slash + 1) if slash >= 0 else String("/")
    var hostport2 = base.host
    if base.port != 80:
        hostport2 = base.host + ":" + String(base.port)
    return base.scheme + "://" + hostport2 + prefix + location


def request(method: String, url: String, headers: List[Header],
            body: List[UInt8], opts: ClientOptions) raises -> HttpResponse:
    """Issue an HTTP request, following redirects up to opts.max_redirects.
    Each redirect opens a fresh connection (simple + correct). For pooled reuse
    against one host, use the Client struct."""
    var cur_url = url
    var redirects_left = opts.max_redirects
    while True:
        var p = parse_url(cur_url)
        var ip = resolve_host(p.host)
        var omit_port = (p.scheme == "http" and p.port == 80) or (p.scheme == "https" and p.port == 443)

        var conn: Conn
        if p.scheme == "https":
            conn = Conn.open_tls(ip, p.port, p.host, opts.verify_tls)
        else:
            conn = Conn.open_plain(ip, p.port)
        conn.set_recv_timeout(opts.timeout_ms)

        var req = _build_request(method, p.host, p.port, p.path, headers, body, opts.decompress, omit_port)
        _send_all(conn, req)
        _send_body(conn, body)

        var raw = _recv_all(conn, False)
        var header_end = _find_crlfcrlf(raw)
        var status = _status_of(raw, header_end)
        var resp_headers = _collect_headers(raw, header_end)

        # redirect?
        if (status == 301 or status == 302 or status == 303 or status == 307 or status == 308) and redirects_left > 0:
            var loc = _location_header(resp_headers)
            if loc.byte_length() > 0:
                redirects_left -= 1
                cur_url = _resolve_redirect(p, loc)
                continue

        # extract + decode body
        var body_start = header_end + 4
        var ish = False
        for i in range(len(resp_headers)):
            if resp_headers[i].name == "transfer-encoding" and _str_has_lower(resp_headers[i].value, "chunked"):
                ish = True
        var raw_body: List[UInt8]
        if ish:
            raw_body = _chunked_decode(raw, body_start)
        else:
            raw_body = _slice_list(raw, body_start, len(raw))

        var final_body = raw_body^
        if opts.decompress:
            var enc = _content_encoding(resp_headers)
            if enc == "gzip" or enc == "x-gzip":
                final_body = _gunzip(final_body)
            elif enc == "deflate":
                final_body = _inflate_deflate_encoding(final_body)

        return HttpResponse(status, resp_headers^, final_body^)


def _str_has_lower(s: String, needle: String) -> Bool:
    var sb = s.as_bytes()
    var n = s.byte_length()
    var nb = needle.as_bytes()
    var m = needle.byte_length()
    if m == 0:
        return True
    var i = 0
    while i + m <= n:
        var ok = True
        for k in range(m):
            if _lower(Int(sb[i + k])) != Int(nb[k]):
                ok = False
                break
        if ok:
            return True
        i += 1
    return False


def _slice_list(b: List[UInt8], start: Int, end: Int) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(start, end):
        out.append(b[i])
    return out^


def get(url: String, opts: ClientOptions) raises -> HttpResponse:
    """GET a URL (http://...). Follows redirects, decodes chunked + gzip/deflate."""
    return request(String("GET"), url, List[Header](), List[UInt8](), opts)


def get(url: String) raises -> HttpResponse:
    """GET with default options."""
    return request(String("GET"), url, List[Header](), List[UInt8](), ClientOptions())


# ─────────────────────────────────────────────────────────────────────────────
# Keep-alive pooled client (one persistent connection to a single host:port)
# ─────────────────────────────────────────────────────────────────────────────

struct Client(Movable):
    """Holds one live keep-alive connection to host:port (http or https) and
    reuses it across requests. Reconnects transparently if the peer dropped it."""

    var host: String
    var port: Int
    var ip: String
    var timeout_ms: Int
    var scheme: String      # "http" or "https"
    var verify: Bool        # https: verify the server cert chain + hostname
    var conn: Conn
    var connected: Bool

    def __init__(
        out self,
        host: String,
        port: Int = 80,
        timeout_ms: Int = 10000,
        scheme: String = String("http"),
        verify: Bool = True,
    ) raises:
        self.host = host
        self.port = port
        self.ip = resolve_host(host)
        self.timeout_ms = timeout_ms
        self.scheme = scheme
        self.verify = verify
        self.conn = Conn(Socket(Int32(-1)))
        self.connected = False

    def _connect(mut self) raises:
        if self.scheme == "https":
            self.conn = Conn.open_tls(self.ip, self.port, self.host, self.verify)
        else:
            self.conn = Conn.open_plain(self.ip, self.port)
        self.conn.set_recv_timeout(self.timeout_ms)
        self.connected = True

    def get(mut self, path: String, decompress: Bool = True) raises -> HttpResponse:
        return self.request(String("GET"), path, List[Header](), List[UInt8](), decompress)

    def request(mut self, method: String, path: String, headers: List[Header],
                body: List[UInt8], decompress: Bool = True) raises -> HttpResponse:
        """Send on the pooled connection; if the server closed it, reconnect once."""
        var omit_port = (self.scheme == "http" and self.port == 80) or (self.scheme == "https" and self.port == 443)
        var attempt = 0
        while attempt < 2:
            attempt += 1
            if not self.connected:
                self._connect()
            var req = _build_request(method, self.host, self.port, path, headers, body, decompress, omit_port)
            # send; if the write fails the connection is dead → retry
            var ok = True
            try:
                _send_all(self.conn, req)
                _send_body(self.conn, body)
            except:
                ok = False
            if not ok:
                self.connected = False
                self.conn = Conn(Socket(Int32(-1)))
                continue
            var raw: List[UInt8]
            try:
                raw = _recv_all(self.conn, False)
            except:
                self.connected = False
                self.conn = Conn(Socket(Int32(-1)))
                continue
            var header_end = _find_crlfcrlf(raw)
            if header_end < 0:
                self.connected = False
                continue
            var status = _status_of(raw, header_end)
            var resp_headers = _collect_headers(raw, header_end)
            # if server said Connection: close, drop the socket after this response
            var hdr = _parse_header_framing(raw, header_end)
            if hdr[2] == 1:
                self.connected = False
                self.conn = Conn(Socket(Int32(-1)))

            var body_start = header_end + 4
            var raw_body: List[UInt8]
            var ish = False
            for i in range(len(resp_headers)):
                if resp_headers[i].name == "transfer-encoding" and _str_has_lower(resp_headers[i].value, "chunked"):
                    ish = True
            if ish:
                raw_body = _chunked_decode(raw, body_start)
            else:
                raw_body = _slice_list(raw, body_start, len(raw))

            var final_body = raw_body^
            if decompress:
                var enc = _content_encoding(resp_headers)
                if enc == "gzip" or enc == "x-gzip":
                    final_body = _gunzip(final_body)
                elif enc == "deflate":
                    final_body = _inflate_deflate_encoding(final_body)
            return HttpResponse(status, resp_headers^, final_body^)
        raise Error("request failed: connection could not be (re)established")


# ─────────────────────────────────────────────────────────────────────────────
# Legacy API (preserved) — minimal blocking GET by ip/port/path
# ─────────────────────────────────────────────────────────────────────────────

@fieldwise_init
struct ClientResponse(Copyable, Movable):
    var status: Int
    var body: String


def get(ip: String, port: Int, path: String, host: String) raises -> ClientResponse:
    """Legacy minimal GET: ip:port/path, identity encoding, Connection: close.
    Preserved for examples/client_main.mojo and any existing callers."""
    var conn = Conn(tcp_connect(ip, port))
    var hostval = host if host != "" else ip
    var req = String("GET ") + path + " HTTP/1.1\r\nHost: " + hostval
    req += "\r\nConnection: close\r\nAccept-Encoding: identity\r\n\r\n"
    _send_all(conn, req)

    var raw = _recv_all(conn, True)
    var header_end = _find_crlfcrlf(raw)
    if header_end < 0:
        return ClientResponse(0, String(""))
    var status = _status_of(raw, header_end)
    var headers = _collect_headers(raw, header_end)
    var body_start = header_end + 4
    var raw_body: List[UInt8]
    var ish = False
    for i in range(len(headers)):
        if headers[i].name == "transfer-encoding" and _str_has_lower(headers[i].value, "chunked"):
            ish = True
    if ish:
        raw_body = _chunked_decode(raw, body_start)
    else:
        raw_body = _slice_list(raw, body_start, len(raw))
    return ClientResponse(status, _bytes_to_str(raw_body, 0, len(raw_body)))
