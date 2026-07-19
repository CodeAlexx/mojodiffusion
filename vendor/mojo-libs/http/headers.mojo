# http.headers — content negotiation, conditional requests, range requests,
# and HTTP-date helpers (RFC 7231 / RFC 7232 / RFC 7233).
#
# This module is self-contained: it only needs the stdlib. It mirrors the
# conventions of http.request / http.url:
#   * `def ... raises` for anything that can fail (parsing user input)
#   * byte-level scanning via String.as_bytes() (never lower-cases whole buffers)
#   * String built from a StringSlice over an alloc'd buffer
#   * structs instead of tuple returns
#
# What's here:
#   Content negotiation: parse_accept -> List[MediaRange] (sorted by q desc),
#       select_media_type / select_encoding (best offered honoring '*').
#   Conditional requests: etag_strong / etag_weak (FNV-1a over the body),
#       if_none_match_satisfied (handles '*', comma lists, weak compare),
#       if_modified_since_satisfied (parses an IMF-fixdate).
#   Range requests: parse_range -> List[ByteRange] (bytes=a-b / a- / -n),
#       content_range_header.
#   HTTP-date: format_http_date / parse_http_date (IMF-fixdate, integer date
#       math; days-since-epoch). See parse_http_date for supported formats.

from std.memory import UnsafePointer, alloc
from std.builtin.type_aliases import MutExternalOrigin

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------

def _substr(s: String, start: Int, end: Int) raises -> String:
    """Byte substring s[start:end). Local copy of request.byte_substr so
    headers.mojo owns all its helpers."""
    var n = end - start
    if n <= 0:
        return String("")
    var sp = s.as_bytes()
    var base = BytePtr(unsafe_from_address=Int(sp.unsafe_ptr()) + start)
    return String(StringSlice(ptr=base, length=n))


def _lower_b(c: Int) -> Int:
    """ASCII lower-case of one byte."""
    if c >= 65 and c <= 90:
        return c + 32
    return c


def _ascii_lower(s: String) raises -> String:
    """Lower-case an ASCII string byte-wise (tokens here are ASCII: media types,
    encodings, etags). Safe on the bounded, ASCII inputs used in this module."""
    var n = s.byte_length()
    if n == 0:
        return String("")
    var sp = s.as_bytes()
    var out = alloc[UInt8](n)
    for i in range(n):
        out[i] = UInt8(_lower_b(Int(sp[i])))
    var res = String(StringSlice(ptr=BytePtr(unsafe_from_address=Int(out)), length=n))
    out.free()
    return res


def _parse_int(s: String) -> Int:
    """Parse a non-negative decimal integer; -1 if empty or non-digit found."""
    var n = s.byte_length()
    if n == 0:
        return -1
    var sp = s.as_bytes()
    var v = 0
    for i in range(n):
        var c = Int(sp[i])
        if c < 48 or c > 57:
            return -1
        v = v * 10 + (c - 48)
    return v


# --------------------------------------------------------------------------
# Content negotiation (RFC 7231 §5.3)
# --------------------------------------------------------------------------

@fieldwise_init
struct MediaRange(Copyable, Movable, ImplicitlyCopyable):
    """One entry of an Accept header, e.g. `application/json;q=0.9`.
    type/subtype are lower-cased; `*` is preserved as a wildcard. `q` is the
    quality factor (default 1.0). `params` keeps any non-q parameters joined
    back as `;k=v` for completeness (rarely used by callers)."""

    var type: String
    var subtype: String
    var q: Float64
    var params: String


def _parse_one_media_range(token: String) raises -> MediaRange:
    """Parse a single Accept entry `type/subtype;q=..;other=..`."""
    var parts = token.split(";")
    var first = String(String(parts[0]).strip())
    var slash = first.find("/")
    var typ: String
    var sub: String
    if slash >= 0:
        typ = _ascii_lower(String(_substr(first, 0, slash).strip()))
        sub = _ascii_lower(String(_substr(first, slash + 1, first.byte_length()).strip()))
    else:
        typ = _ascii_lower(first)
        sub = String("")
    var q = 1.0
    var extra = String("")
    for i in range(1, len(parts)):
        var p = String(String(parts[i]).strip())
        if p.byte_length() == 0:
            continue
        var eq = p.find("=")
        var k = p if eq < 0 else String(_substr(p, 0, eq).strip())
        var v = String("") if eq < 0 else String(_substr(p, eq + 1, p.byte_length()).strip())
        if _ascii_lower(k) == "q":
            try:
                q = Float64(v)
            except:
                q = 1.0
        else:
            extra += ";"
            extra += p
    return MediaRange(typ, sub, q, extra)


def parse_accept(s: String) raises -> List[MediaRange]:
    """Parse an Accept (or any q-valued) header into MediaRanges, sorted by q
    descending (stable: equal-q entries keep input order). An empty string
    yields an empty list."""
    var out = List[MediaRange]()
    var trimmed = String(s.strip())
    if trimmed.byte_length() == 0:
        return out^
    var toks = trimmed.split(",")
    for i in range(len(toks)):
        var tk = String(String(toks[i]).strip())
        if tk.byte_length() == 0:
            continue
        out.append(_parse_one_media_range(tk))
    # stable insertion sort by q desc (lists here are short)
    var n = len(out)
    for i in range(1, n):
        var cur = out[i].copy()
        var j = i - 1
        while j >= 0 and out[j].q < cur.q:
            out[j + 1] = out[j].copy()
            j -= 1
        out[j + 1] = cur.copy()
    return out^


def _media_matches(mr: MediaRange, typ: String, sub: String) -> Bool:
    """Does range `mr` match concrete media type `typ/sub` (lower-cased)?
    `*` in the range's type or subtype is a wildcard."""
    if mr.type != "*" and mr.type != typ:
        return False
    if mr.subtype != "*" and mr.subtype != "" and mr.subtype != sub:
        return False
    return True


def select_media_type(accept: String, offered: List[String]) raises -> String:
    """Pick the best media type from `offered` given an Accept header. Returns
    the chosen *offered* string (original casing), or "" if nothing is
    acceptable (every match would have q=0). An empty/absent Accept means the
    client accepts anything → the first offered is returned.

    Specificity: among ranges that match an offer, the most specific match wins
    (exact type+subtype > type/* > */*); ties break on q, then offer order."""
    if len(offered) == 0:
        return String("")
    var ranges = parse_accept(accept)
    if len(ranges) == 0:
        return offered[0].copy()  # no preference stated

    var best = String("")
    var best_q = 0.0
    var best_spec = -1
    for oi in range(len(offered)):
        var off = offered[oi]
        var slash = off.find("/")
        var otyp = _ascii_lower(off) if slash < 0 else _ascii_lower(_substr(off, 0, slash))
        var osub = String("") if slash < 0 else _ascii_lower(_substr(off, slash + 1, off.byte_length()))
        # find the matching range with highest specificity, then q
        var matched_q = -1.0
        var matched_spec = -1
        for ri in range(len(ranges)):
            var mr = ranges[ri]
            if not _media_matches(mr, otyp, osub):
                continue
            var spec = 0
            if mr.type != "*":
                spec += 1
            if mr.subtype != "*" and mr.subtype != "":
                spec += 1
            if spec > matched_spec or (spec == matched_spec and mr.q > matched_q):
                matched_spec = spec
                matched_q = mr.q
        if matched_q <= 0.0:
            continue  # not acceptable (no match, or q=0)
        if matched_q > best_q or (matched_q == best_q and matched_spec > best_spec):
            best_q = matched_q
            best_spec = matched_spec
            best = off.copy()
    return best


def select_encoding(accept_encoding: String, offered: List[String]) raises -> String:
    """Best content-coding from `offered` given an Accept-Encoding header. Same
    rules as select_media_type but tokens are single coding names (gzip, br,
    identity, *). Empty header → first offered. "" if none acceptable."""
    if len(offered) == 0:
        return String("")
    var ranges = parse_accept(accept_encoding)
    if len(ranges) == 0:
        return offered[0].copy()
    var best = String("")
    var best_q = 0.0
    var best_spec = -1
    for oi in range(len(offered)):
        var off = _ascii_lower(offered[oi])
        var matched_q = -1.0
        var matched_spec = -1
        for ri in range(len(ranges)):
            var mr = ranges[ri]
            # encodings are stored in `type` (no slash); subtype is ""
            var name = mr.type
            if name != "*" and name != off:
                continue
            var spec = 1 if name != "*" else 0
            if spec > matched_spec or (spec == matched_spec and mr.q > matched_q):
                matched_spec = spec
                matched_q = mr.q
        if matched_q <= 0.0:
            continue
        if matched_q > best_q or (matched_q == best_q and matched_spec > best_spec):
            best_q = matched_q
            best_spec = matched_spec
            best = offered[oi].copy()  # original casing
    return best


# --------------------------------------------------------------------------
# Conditional requests (RFC 7232)
# --------------------------------------------------------------------------

def _fnv1a64(body: List[UInt8]) -> UInt64:
    """FNV-1a 64-bit hash. Chosen for ETags because it is tiny, dependency-free,
    deterministic and well-distributed for entity-tag use (this is NOT a
    cryptographic hash — ETags only need stability + low collision, per
    RFC 7232 §2.3 which leaves the generation method to the server)."""
    var h = UInt64(14695981039346656037)  # FNV offset basis
    var prime = UInt64(1099511628211)     # FNV prime
    for i in range(len(body)):
        h = h ^ UInt64(Int(body[i]))
        h = h * prime
    return h


def _hex16(v: UInt64) -> String:
    """16-char lower-case hex of a UInt64."""
    var digits = "0123456789abcdef"
    var db = digits.as_bytes()
    var out = alloc[UInt8](16)
    var x = v
    for i in range(16):
        var nib = Int(x & 0xF)
        out[15 - i] = db[nib]
        x = x >> 4
    var res = String(StringSlice(ptr=BytePtr(unsafe_from_address=Int(out)), length=16))
    out.free()
    return res


def etag_strong(body: List[UInt8]) -> String:
    """A strong ETag: a quoted 16-hex FNV-1a hash of the body, e.g.
    `"a1b2c3d4e5f6a7b8"`. Same bytes → same tag; different bytes → (almost
    surely) different. Strong means byte-for-byte equality is implied."""
    return String('"') + _hex16(_fnv1a64(body)) + String('"')


def etag_weak(body: List[UInt8]) -> String:
    """A weak ETag: the strong tag prefixed with `W/`, e.g.
    `W/"a1b2c3d4e5f6a7b8"`. Weak means semantic (not byte-exact) equivalence."""
    return String('W/"') + _hex16(_fnv1a64(body)) + String('"')


def _strip_weak(tag: String) raises -> String:
    """Drop a leading `W/` so weak and strong forms of the same opaque-tag
    compare equal (used for the weak comparison If-None-Match needs)."""
    var t = String(tag.strip())
    if t.byte_length() >= 2:
        var sp = t.as_bytes()
        if Int(sp[0]) == 87 and Int(sp[1]) == 47:  # 'W' '/'
            return _substr(t, 2, t.byte_length())
    return t


def if_none_match_satisfied(if_none_match: String, etag: String) raises -> Bool:
    """RFC 7232 §3.2: returns True when the precondition is *met* meaning the
    handler should send 304 Not Modified (the client already has a current
    representation). I.e. True if the resource's `etag` matches the
    If-None-Match header.

    Handles `*` (matches any existing representation), a comma-separated list of
    tags, and uses the weak comparison function (W/ prefix ignored)."""
    var inm = String(if_none_match.strip())
    if inm.byte_length() == 0:
        return False
    if inm == "*":
        return True
    var want = _strip_weak(etag)
    var toks = inm.split(",")
    for i in range(len(toks)):
        var t = _strip_weak(String(String(toks[i]).strip()))
        if t == want:
            return True
    return False


def _two(s: String, start: Int) raises -> Int:
    """Parse exactly two ASCII digits at s[start]; -1 on non-digit."""
    var sp = s.as_bytes()
    if start + 1 >= s.byte_length():
        return -1
    var a = Int(sp[start]) - 48
    var b = Int(sp[start + 1]) - 48
    if a < 0 or a > 9 or b < 0 or b > 9:
        return -1
    return a * 10 + b


def if_modified_since_satisfied(ims: String, last_modified_epoch: Int) raises -> Bool:
    """RFC 7232 §3.3: returns True when the resource has NOT been modified since
    the If-Modified-Since date — i.e. the handler should send 304. That is True
    when last_modified_epoch <= the parsed IMS date. A malformed/empty date
    returns False (treat as "modified", send full response)."""
    var t = String(ims.strip())
    if t.byte_length() == 0:
        return False
    try:
        var ims_epoch = parse_http_date(t)
        return last_modified_epoch <= ims_epoch
    except:
        return False


# --------------------------------------------------------------------------
# Range requests (RFC 7233)
# --------------------------------------------------------------------------

@fieldwise_init
struct ByteRange(Copyable, Movable):
    """An inclusive byte range [start, end] resolved against a known total
    length. Both ends are concrete (no open ends after parse_range)."""

    var start: Int
    var end: Int


def parse_range(s: String, total: Int) raises -> List[ByteRange]:
    """Parse a Range header value against `total` bytes. Supports the byte-range
    forms:  `bytes=0-499` (explicit), `bytes=500-` (from offset to end),
    `bytes=-500` (last 500 bytes). Multiple comma-separated ranges are allowed.

    Each returned ByteRange is clamped to [0, total-1] inclusive. A range whose
    start is past the end of the resource is *unsatisfiable* and is skipped. If
    the header is malformed or no range is satisfiable, an empty list is
    returned (the caller should then answer 416, or ignore Range and send 200).
    """
    var out = List[ByteRange]()
    var t = String(s.strip())
    if t.byte_length() == 0 or total <= 0:
        return out^
    # require the "bytes=" unit prefix
    var eq = t.find("=")
    if eq < 0:
        return out^
    var unit = _ascii_lower(String(_substr(t, 0, eq).strip()))
    if unit != "bytes":
        return out^
    var spec = _substr(t, eq + 1, t.byte_length())
    var parts = spec.split(",")
    for i in range(len(parts)):
        var part = String(String(parts[i]).strip())
        if part.byte_length() == 0:
            continue
        var dash = part.find("-")
        if dash < 0:
            continue  # malformed
        var first_str = String(_substr(part, 0, dash).strip())
        var last_str = String(_substr(part, dash + 1, part.byte_length()).strip())
        var start: Int
        var end: Int
        if first_str.byte_length() == 0:
            # suffix range: -N -> last N bytes
            var nlast = _parse_int(last_str)
            if nlast <= 0:
                continue
            if nlast > total:
                nlast = total
            start = total - nlast
            end = total - 1
        else:
            var fs = _parse_int(first_str)
            if fs < 0:
                continue
            if fs >= total:
                continue  # start past EOF -> unsatisfiable, skip
            start = fs
            if last_str.byte_length() == 0:
                end = total - 1  # open-ended -> to last byte
            else:
                var ls = _parse_int(last_str)
                if ls < 0 or ls < fs:
                    continue  # malformed / inverted
                end = ls
                if end > total - 1:
                    end = total - 1
        out.append(ByteRange(start, end))
    return out^


def content_range_header(start: Int, end: Int, total: Int) -> String:
    """The Content-Range header value for a 206 response, e.g.
    `bytes 0-499/1234`. (The full header line is `Content-Range: ` + this.)"""
    return (
        String("bytes ")
        + String(start)
        + String("-")
        + String(end)
        + String("/")
        + String(total)
    )


# --------------------------------------------------------------------------
# HTTP-date (RFC 7231 §7.1.1.1)  — integer date math, no libc
# --------------------------------------------------------------------------
#
# We support the IMF-fixdate format on both format and parse:
#     Sun, 06 Nov 1994 08:49:37 GMT
# (3-letter weekday, 2-digit day, 3-letter month, 4-digit year, HH:MM:SS GMT).
# parse_http_date is lenient about the weekday and surrounding whitespace but
# requires the IMF-fixdate field layout. The two obsolete formats (RFC 850
# `Sunday, 06-Nov-94 ...` and asctime `Sun Nov  6 ...`) are NOT parsed — see
# the module docs.

comptime _MONTHS = "JanFebMarAprMayJunJulAugSepOctNovDec"
comptime _WDAYS = "MonTueWedThuFriSatSun"


def _is_leap(y: Int) -> Bool:
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


def _days_in_month(y: Int, m: Int) -> Int:
    """Days in month m (1-12) of year y."""
    if m == 2:
        return 29 if _is_leap(y) else 28
    if m == 4 or m == 6 or m == 9 or m == 11:
        return 30
    return 31


def _days_from_civil(y: Int, m: Int, d: Int) -> Int:
    """Days from 1970-01-01 to y-m-d (proleptic Gregorian). Counts forward/back
    year by year then month by month — simple and exact for the range of dates
    HTTP cares about."""
    var days = 0
    if y >= 1970:
        for yy in range(1970, y):
            days += 366 if _is_leap(yy) else 365
    else:
        for yy in range(y, 1970):
            days -= 366 if _is_leap(yy) else 365
    for mm in range(1, m):
        days += _days_in_month(y, mm)
    days += d - 1
    return days


def _civil_from_days(z: Int) -> SIMD[DType.int64, 4]:
    """Inverse of _days_from_civil: days-since-epoch -> (year, month, day) packed
    into a SIMD (avoids a tuple return). Lanes: [0]=year [1]=month [2]=day."""
    # Walk years from the epoch (linear; fine for HTTP date ranges).
    var y = 1970
    var days = z
    if days >= 0:
        while True:
            var yl = 366 if _is_leap(y) else 365
            if days < yl:
                break
            days -= yl
            y += 1
    else:
        while days < 0:
            y -= 1
            days += 366 if _is_leap(y) else 365
    var m = 1
    while True:
        var dim = _days_in_month(y, m)
        if days < dim:
            break
        days -= dim
        m += 1
    var d = days + 1
    return SIMD[DType.int64, 4](Int64(y), Int64(m), Int64(d), 0)


def _two_digit(v: Int) -> String:
    if v < 10:
        return String("0") + String(v)
    return String(v)


def _weekday_index(days_since_epoch: Int) -> Int:
    """Weekday for a day number where 1970-01-01 was a Thursday. Returns 0=Mon
    .. 6=Sun (matching _WDAYS order)."""
    # 1970-01-01 = Thursday = index 3 in Mon..Sun
    var idx = (days_since_epoch + 3) % 7
    if idx < 0:
        idx += 7
    return idx


def format_http_date(epoch: Int) raises -> String:
    """Format a Unix epoch (seconds, UTC) as an IMF-fixdate, e.g.
    `Sun, 06 Nov 1994 08:49:37 GMT`. Pure integer math (no libc)."""
    var secs_per_day = 86400
    var days = epoch // secs_per_day
    var rem = epoch % secs_per_day
    if rem < 0:
        rem += secs_per_day
        days -= 1
    var hh = rem // 3600
    var mm = (rem % 3600) // 60
    var ss = rem % 60

    var ymd = _civil_from_days(days)
    var year = Int(ymd[0])
    var month = Int(ymd[1])
    var day = Int(ymd[2])

    var wi = _weekday_index(days)
    var wday = _substr(String(_WDAYS), 3 * wi, 3 * wi + 3)
    var mon = _substr(String(_MONTHS), 3 * (month - 1), 3 * (month - 1) + 3)

    var out = String(wday)
    out += ", "
    out += _two_digit(day)
    out += " "
    out += String(mon)
    out += " "
    out += String(year)
    out += " "
    out += _two_digit(hh)
    out += ":"
    out += _two_digit(mm)
    out += ":"
    out += _two_digit(ss)
    out += " GMT"
    return out


def _month_index(mon: String) raises -> Int:
    """0-based month index for a 3-letter English month name; -1 if unknown."""
    for i in range(12):
        if _substr(String(_MONTHS), 3 * i, 3 * i + 3) == mon:
            return i
    return -1


def parse_http_date(s: String) raises -> Int:
    """Parse an IMF-fixdate (`Sun, 06 Nov 1994 08:49:37 GMT`) to a Unix epoch
    (seconds, UTC). Supported format: IMF-fixdate only. The leading weekday and
    its comma are tolerated/skipped; fields are read positionally from the day
    onward. Raises on a malformed date.

    NOTE: the two obsolete HTTP-date forms (RFC 850 and asctime) are not
    supported — modern servers emit IMF-fixdate exclusively."""
    var t = String(s.strip())
    # Drop the weekday + comma if present: find the first space after a comma,
    # else just split on spaces.
    var work = t
    var comma = t.find(",")
    if comma >= 0:
        work = String(_substr(t, comma + 1, t.byte_length()).strip())
    var f = work.split(" ")
    # Expect: DD Mon YYYY HH:MM:SS [GMT]
    if len(f) < 4:
        raise Error("bad http-date: " + s)
    var day = _parse_int(String(String(f[0]).strip()))
    var mon_idx = _month_index(String(String(f[1]).strip()))
    var year = _parse_int(String(String(f[2]).strip()))
    if day < 0 or mon_idx < 0 or year < 0:
        raise Error("bad http-date fields: " + s)
    var time_str = String(String(f[3]).strip())
    if time_str.byte_length() < 8:
        raise Error("bad http-date time: " + s)
    var hh = _two(time_str, 0)
    var mm = _two(time_str, 3)
    var ss = _two(time_str, 6)
    if hh < 0 or mm < 0 or ss < 0:
        raise Error("bad http-date time fields: " + s)
    var days = _days_from_civil(year, mon_idx + 1, day)
    return days * 86400 + hh * 3600 + mm * 60 + ss
