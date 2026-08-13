# http.cookies — request/response cookie parsing + serialization, plus a small
# CookieJar, plus application/x-www-form-urlencoded parse/build.
#
# Request cookies arrive in one `Cookie:` header as `name=val; name2=val2`.
# Responses set cookies one-per-line via `Set-Cookie: name=val; Path=/; ...`.
# Cookie *values* are ASCII text (RFC 6265), so these parsers work on String —
# unlike multipart, which is binary and lives in multipart.mojo.

from std.memory import UnsafePointer, alloc

comptime CkBytePtr = UnsafePointer[UInt8, MutExternalOrigin]


def _sub(s: String, start: Int, end: Int) -> String:
    """Byte substring s[start:end). Mirrors request.byte_substr (no slice op)."""
    var n = end - start
    if n <= 0:
        return String("")
    var sp = s.as_bytes()
    var base = CkBytePtr(unsafe_from_address=Int(sp.unsafe_ptr()) + start)
    return String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=base, length=n)))


fn _ck_hexv(c: Int) -> Int:
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 87
    if c >= 65 and c <= 70:
        return c - 55
    return -1


def _pct_decode(s: String, plus_as_space: Bool) -> String:
    """Percent-decode %XX -> byte; with plus_as_space, `+` -> space (form data)."""
    var n = s.byte_length()
    if n == 0:
        return String("")
    var sp = s.as_bytes()
    var out = alloc[UInt8](n)
    var w = 0
    var i = 0
    while i < n:
        var c = Int(sp[i])
        if c == 37 and i + 2 < n:  # '%'
            var hi = _ck_hexv(Int(sp[i + 1]))
            var lo = _ck_hexv(Int(sp[i + 2]))
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
    var res = String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=CkBytePtr(unsafe_from_address=Int(out)), length=w)))
    out.free()
    return res


fn _is_unreserved(c: Int) -> Bool:
    """RFC 3986 unreserved: ALPHA / DIGIT / - _ . ~ . Everything else %-encoded."""
    if c >= 48 and c <= 57:
        return True
    if c >= 65 and c <= 90:
        return True
    if c >= 97 and c <= 122:
        return True
    return c == 45 or c == 95 or c == 46 or c == 126  # - _ . ~


fn _hex_digit(v: Int) -> UInt8:
    if v < 10:
        return UInt8(48 + v)
    return UInt8(55 + v)  # 'A'..'F'


def _pct_encode(s: String) -> String:
    """Percent-encode for form bodies: space -> '+', unreserved kept, rest %XX."""
    var n = s.byte_length()
    if n == 0:
        return String("")
    var sp = s.as_bytes()
    var out = alloc[UInt8](n * 3)
    var w = 0
    for i in range(n):
        var c = Int(sp[i])
        if c == 32:  # space -> '+'
            out[w] = 43
            w += 1
        elif _is_unreserved(c):
            out[w] = UInt8(c)
            w += 1
        else:
            out[w] = 37  # '%'
            out[w + 1] = _hex_digit((c >> 4) & 0xF)
            out[w + 2] = _hex_digit(c & 0xF)
            w += 3
    var res = String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=CkBytePtr(unsafe_from_address=Int(out)), length=w)))
    out.free()
    return res


def _strip(s: String) -> String:
    """Trim ASCII spaces/tabs from both ends (String.strip is fine for ASCII)."""
    return String(s.strip())


def _ascii_lower(s: String) -> String:
    return String(s.lower())


# ---------------------------------------------------------------------------
# Request cookies:  Cookie: a=1; b=2; c=hello
# ---------------------------------------------------------------------------

@fieldwise_init
struct Cookie(Copyable, Movable):
    """One request cookie: a name and a value."""

    var name: String
    var value: String


def parse_cookie_header(s: String) raises -> List[Cookie]:
    """Parse a request `Cookie:` header value (`a=1; b=2`) into name/value pairs.
    Splits on ';', then on the first '='. Values may legitimately contain '=',
    so only the first '=' is the separator. Pairs with no '=' are skipped."""
    var out = List[Cookie]()
    var parts = s.split(";")
    for i in range(len(parts)):
        var pair = _strip(String(parts[i]))
        if pair.byte_length() == 0:
            continue
        var eq = pair.find("=")
        if eq < 0:
            continue
        var name = _strip(_sub(pair, 0, eq))
        var value = _strip(_sub(pair, eq + 1, pair.byte_length()))
        # A cookie value may be wrapped in double quotes (RFC 6265 quoted form).
        var vb = value.as_bytes()
        if value.byte_length() >= 2 and Int(vb[0]) == 34 and Int(vb[value.byte_length() - 1]) == 34:
            value = _sub(value, 1, value.byte_length() - 1)
        out.append(Cookie(name, value))
    return out^


def serialize_cookie(name: String, value: String) -> String:
    """Render one `name=value` pair as it appears in a request `Cookie:` header."""
    return name + "=" + value


# ---------------------------------------------------------------------------
# Response cookies:  Set-Cookie: name=val; Path=/; HttpOnly; Max-Age=3600; ...
# ---------------------------------------------------------------------------

struct SetCookie(Copyable, Movable):
    """A parsed/buildable `Set-Cookie` directive. Unset attributes are empty
    strings; max_age = -1 means "absent" (0 is a valid Max-Age = delete now)."""

    var name: String
    var value: String
    var path: String
    var domain: String
    var max_age: Int
    var secure: Bool
    var http_only: Bool
    var same_site: String
    var expires: String

    def __init__(out self, name: String = String(""), value: String = String("")):
        self.name = name
        self.value = value
        self.path = String("")
        self.domain = String("")
        self.max_age = -1
        self.secure = False
        self.http_only = False
        self.same_site = String("")
        self.expires = String("")


def parse_set_cookie(s: String) raises -> SetCookie:
    """Parse a `Set-Cookie` header value. The first `;`-segment is the cookie's
    own name=value; the rest are attributes (Path, Domain, Max-Age, Expires,
    SameSite) and flags (Secure, HttpOnly). Attribute names are case-insensitive."""
    var sc = SetCookie()
    var parts = s.split(";")
    if len(parts) == 0:
        return sc^
    # first segment: name=value
    var first = _strip(String(parts[0]))
    var eq = first.find("=")
    if eq >= 0:
        sc.name = _strip(_sub(first, 0, eq))
        sc.value = _strip(_sub(first, eq + 1, first.byte_length()))
    else:
        sc.name = first
    for i in range(1, len(parts)):
        var attr = _strip(String(parts[i]))
        if attr.byte_length() == 0:
            continue
        var ae = attr.find("=")
        var key: String
        var val = String("")
        if ae >= 0:
            key = _ascii_lower(_strip(_sub(attr, 0, ae)))
            val = _strip(_sub(attr, ae + 1, attr.byte_length()))
        else:
            key = _ascii_lower(attr)
        if key == "path":
            sc.path = val
        elif key == "domain":
            sc.domain = val
        elif key == "max-age":
            var v = 0
            var neg = False
            var vb = val.as_bytes()
            var start = 0
            if val.byte_length() > 0 and Int(vb[0]) == 45:  # '-'
                neg = True
                start = 1
            var ok = val.byte_length() > start
            for k in range(start, val.byte_length()):
                var c = Int(vb[k])
                if c < 48 or c > 57:
                    ok = False
                    break
                v = v * 10 + (c - 48)
            if ok:
                sc.max_age = -v if neg else v
        elif key == "expires":
            sc.expires = val
        elif key == "samesite":
            sc.same_site = val
        elif key == "secure":
            sc.secure = True
        elif key == "httponly":
            sc.http_only = True
    return sc^


def serialize_set_cookie(c: SetCookie) -> String:
    """Render a SetCookie back into a `Set-Cookie` header value, attributes in a
    stable order. Inverse of parse_set_cookie for the fields we model."""
    var out = c.name + "=" + c.value
    if c.path.byte_length() > 0:
        out += "; Path=" + c.path
    if c.domain.byte_length() > 0:
        out += "; Domain=" + c.domain
    if c.max_age >= 0:
        out += "; Max-Age=" + String(c.max_age)
    if c.expires.byte_length() > 0:
        out += "; Expires=" + c.expires
    if c.same_site.byte_length() > 0:
        out += "; SameSite=" + c.same_site
    if c.secure:
        out += "; Secure"
    if c.http_only:
        out += "; HttpOnly"
    return out


# ---------------------------------------------------------------------------
# CookieJar — store cookies by name, emit a request Cookie: line.
# ---------------------------------------------------------------------------

struct CookieJar(Copyable, Movable):
    """A minimal cookie store keyed by name. `set`/`get` manage values; `header`
    emits a single `Cookie:` request line from everything stored."""

    var _store: Dict[String, String]

    def __init__(out self):
        self._store = Dict[String, String]()

    def set(mut self, name: String, value: String):
        self._store[name] = value

    def get(self, name: String) raises -> String:
        """Stored value for `name`, or "" if not present."""
        if name in self._store:
            return self._store[name]
        return String("")

    def has(self, name: String) raises -> Bool:
        return name in self._store

    def header(self) raises -> String:
        """A request `Cookie:` line body (`a=1; b=2`) from all stored cookies."""
        var out = String("")
        var first = True
        for entry in self._store.items():
            if not first:
                out += "; "
            out += entry.key + "=" + entry.value
            first = False
        return out


# ---------------------------------------------------------------------------
# application/x-www-form-urlencoded
# ---------------------------------------------------------------------------

@fieldwise_init
struct FormField(Copyable, Movable):
    """One decoded form field: a name and a value (both already %-decoded)."""

    var name: String
    var value: String


def parse_form_urlencoded(body: String) raises -> List[FormField]:
    """Parse `a=1&b=two+words&c=%2F` into decoded name/value pairs. '+' becomes a
    space and %XX is decoded in both name and value. A key with no '=' yields an
    empty value; empty segments (from `&&`) are skipped."""
    var out = List[FormField]()
    if body.byte_length() == 0:
        return out^
    var pairs = body.split("&")
    for i in range(len(pairs)):
        var pair = String(pairs[i])
        if pair.byte_length() == 0:
            continue
        var eq = pair.find("=")
        if eq < 0:
            out.append(FormField(_pct_decode(pair, True), String("")))
        else:
            var name = _pct_decode(_sub(pair, 0, eq), True)
            var value = _pct_decode(_sub(pair, eq + 1, pair.byte_length()), True)
            out.append(FormField(name, value))
    return out^


def build_form_urlencoded(fields: List[FormField]) raises -> String:
    """Encode fields into `a=1&b=two+words`. Names and values are percent-encoded
    (space -> '+'), the inverse of parse_form_urlencoded."""
    var out = String("")
    for i in range(len(fields)):
        if i > 0:
            out += "&"
        out += _pct_encode(fields[i].name) + "=" + _pct_encode(fields[i].value)
    return out
