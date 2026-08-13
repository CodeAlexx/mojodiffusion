# http.url — URL parsing, query-string handling and percent-encoding (RFC 3986-ish).
#
# parse_url splits a URL into its components (scheme://[userinfo@]host[:port]
# [/path][?query][#fragment]). percent_encode / percent_decode implement the
# %XX escaping; decode_query_component additionally maps `+` -> space, matching
# how query components are encoded by HTML forms (application/x-www-form-urlencoded).
# parse_query / build_query work over QueryParam structs (no tuple returns).
#
# Conventions follow http.request: `def ... raises`, byte-level scanning via
# as_bytes(), and String built from a StringSlice over an alloc'd buffer.

from std.memory import UnsafePointer, alloc

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]


def _substr(s: String, start: Int, end: Int) raises -> String:
    """Byte substring s[start:end). Mirrors request.byte_substr but kept local so
    url.mojo owns all its helpers."""
    var n = end - start
    if n <= 0:
        return String("")
    var sp = s.as_bytes()
    var base = BytePtr(unsafe_from_address=Int(sp.unsafe_ptr()) + start)
    return String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=base, length=n)))


def _hexv(c: Int) -> Int:
    """Value of one ASCII hex nibble, or -1 if not a hex digit."""
    if c >= 48 and c <= 57:  # 0-9
        return c - 48
    if c >= 97 and c <= 102:  # a-f
        return c - 87
    if c >= 65 and c <= 70:  # A-F
        return c - 55
    return -1


def _is_unreserved(c: Int) -> Bool:
    """RFC 3986 unreserved set: A-Z a-z 0-9 and -._~ (never percent-encoded)."""
    if c >= 48 and c <= 57:  # 0-9
        return True
    if c >= 65 and c <= 90:  # A-Z
        return True
    if c >= 97 and c <= 122:  # a-z
        return True
    if c == 45 or c == 46 or c == 95 or c == 126:  # - . _ ~
        return True
    return False


# --------------------------------------------------------------------------
# Percent-encoding
# --------------------------------------------------------------------------

def percent_encode(s: String, safe: String = "") raises -> String:
    """Percent-encode `s`: every byte that is not unreserved (A-Za-z0-9-._~) and
    not listed in `safe` becomes %XX (upper-case hex). Operates byte-wise, so
    multi-byte UTF-8 (e.g. café) is encoded one byte at a time per RFC 3986."""
    var n = s.byte_length()
    if n == 0:
        return String("")
    var sp = s.as_bytes()
    var sb = safe.as_bytes()
    var sn = safe.byte_length()
    # worst case each byte -> 3 chars
    var out = alloc[UInt8](n * 3)
    var w = 0
    comptime HEX = "0123456789ABCDEF"
    var hx = HEX.as_bytes()
    for i in range(n):
        var c = Int(sp[i])
        var keep = _is_unreserved(c)
        if not keep:
            for j in range(sn):
                if Int(sb[j]) == c:
                    keep = True
                    break
        if keep:
            out[w] = UInt8(c)
            w += 1
        else:
            out[w] = 37  # '%'
            out[w + 1] = hx[(c >> 4) & 0xF]
            out[w + 2] = hx[c & 0xF]
            w += 3
    var res = String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=BytePtr(unsafe_from_address=Int(out)), length=w)))
    out.free()
    return res


def _pct_decode(s: String, plus_as_space: Bool) raises -> String:
    """Shared %XX decoder. `plus_as_space` turns '+' into a space (query variant)."""
    var n = s.byte_length()
    if n == 0:
        return String("")
    var sp = s.as_bytes()
    var out = alloc[UInt8](n)  # decoded length <= input length
    var w = 0
    var i = 0
    while i < n:
        var c = Int(sp[i])
        if c == 37 and i + 2 < n:  # '%' with two following bytes
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


def percent_decode(s: String) raises -> String:
    """Decode %XX escapes back to raw bytes. A literal '+' stays a '+' (use
    decode_query_component for the form-encoded `+` -> space behaviour)."""
    return _pct_decode(s, False)


def decode_query_component(s: String) raises -> String:
    """Decode a query component: %XX escapes plus `+` -> space (form encoding)."""
    return _pct_decode(s, True)


# --------------------------------------------------------------------------
# Query strings
# --------------------------------------------------------------------------

@fieldwise_init
struct QueryParam(Copyable, Movable):
    """One key=value pair from a query string (values are decoded)."""

    var key: String
    var value: String


def parse_query(q: String) raises -> List[QueryParam]:
    """Parse a query string (no leading '?') into key/value pairs. Splits on '&',
    then on the first '='; both key and value are decoded with `+` -> space. A
    bare key (no '=') yields an empty value; empty segments are skipped."""
    var out = List[QueryParam]()
    if q.byte_length() == 0:
        return out^
    var pairs = q.split("&")
    for i in range(len(pairs)):
        var pair = String(pairs[i])
        if pair.byte_length() == 0:
            continue
        var eq = pair.find("=")
        if eq < 0:
            out.append(QueryParam(decode_query_component(pair), String("")))
        else:
            var k = decode_query_component(_substr(pair, 0, eq))
            var v = decode_query_component(_substr(pair, eq + 1, pair.byte_length()))
            out.append(QueryParam(k, v))
    return out^


def build_query(params: List[QueryParam]) raises -> String:
    """Build a query string from params: percent-encode each key and value and
    join the `key=value` pairs with '&'. Keys/values are encoded with the default
    unreserved set (spaces become %20, not '+')."""
    var out = String("")
    for i in range(len(params)):
        if i > 0:
            out += "&"
        out += percent_encode(params[i].key)
        out += "="
        out += percent_encode(params[i].value)
    return out^


def query_get(params: List[QueryParam], key: String) raises -> String:
    """First value whose key matches `key`, or "" if none."""
    for i in range(len(params)):
        if params[i].key == key:
            return params[i].value
    return String("")


# --------------------------------------------------------------------------
# URL parsing
# --------------------------------------------------------------------------

@fieldwise_init
struct Url(Copyable, Movable):
    """A parsed URL. Missing components are "" (or port -1 / default)."""

    var scheme: String
    var host: String
    var port: Int
    var path: String
    var query: String
    var fragment: String
    var userinfo: String


def _default_port(scheme: String) raises -> Int:
    """Default port for a scheme: 80 http, 443 https, -1 otherwise."""
    var s = scheme.lower()
    if s == "http":
        return 80
    if s == "https":
        return 443
    return -1


def parse_url(s: String) raises -> Url:
    """Parse an absolute URL of the form
        scheme://[userinfo@]host[:port][/path][?query][#fragment]
    Components that are absent come back empty; an omitted port defaults to 80
    (http) / 443 (https) / -1 (unknown scheme). The scheme + "://" authority form
    is required to populate host/port; a bare path (no scheme) parses into `path`
    with everything else empty."""
    var scheme = String("")
    var userinfo = String("")
    var host = String("")
    var port = -1
    var query = String("")
    var fragment = String("")

    var rest = s

    # scheme:  up to "://"
    var scheme_sep = rest.find("://")
    if scheme_sep > 0:
        scheme = _substr(rest, 0, scheme_sep)
        rest = _substr(rest, scheme_sep + 3, rest.byte_length())

        # authority is everything up to the first '/', '?' or '#'
        var auth_end = rest.byte_length()
        var slash = rest.find("/")
        var qm = rest.find("?")
        var hsh = rest.find("#")
        if slash >= 0 and slash < auth_end:
            auth_end = slash
        if qm >= 0 and qm < auth_end:
            auth_end = qm
        if hsh >= 0 and hsh < auth_end:
            auth_end = hsh

        var authority = _substr(rest, 0, auth_end)
        rest = _substr(rest, auth_end, rest.byte_length())  # path/query/frag

        # userinfo @ host[:port]
        var at = authority.find("@")
        var hostport = authority
        if at >= 0:
            userinfo = _substr(authority, 0, at)
            hostport = _substr(authority, at + 1, authority.byte_length())

        # host[:port] — last ':' separates port (keeps IPv6 untouched here; see notes)
        var colon = hostport.find(":")
        if colon >= 0:
            host = _substr(hostport, 0, colon)
            var port_str = _substr(hostport, colon + 1, hostport.byte_length())
            if port_str.byte_length() > 0:
                var pv = 0
                var ok = True
                var pb = port_str.as_bytes()
                for i in range(port_str.byte_length()):
                    var c = Int(pb[i])
                    if c < 48 or c > 57:
                        ok = False
                        break
                    pv = pv * 10 + (c - 48)
                if ok:
                    port = pv
                else:
                    port = _default_port(scheme)
            else:
                port = _default_port(scheme)
        else:
            host = hostport
            port = _default_port(scheme)

    # rest now holds [path][?query][#fragment] (or the whole string if no scheme)

    # fragment first (it ends the URL)
    var hsh2 = rest.find("#")
    if hsh2 >= 0:
        fragment = _substr(rest, hsh2 + 1, rest.byte_length())
        rest = _substr(rest, 0, hsh2)

    # query next
    var qm2 = rest.find("?")
    if qm2 >= 0:
        query = _substr(rest, qm2 + 1, rest.byte_length())
        rest = _substr(rest, 0, qm2)

    # whatever is left is the path
    var path = rest

    return Url(scheme, host, port, path, query, fragment, userinfo)
