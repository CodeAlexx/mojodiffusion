# http.multipart — multipart/form-data parser (RFC 7578). BINARY-SAFE.
#
# A multipart body is a sequence of parts separated by `--<boundary>` delimiter
# lines, terminated by `--<boundary>--`. Each part has its own headers (notably
# Content-Disposition with name= and optional filename=, and Content-Type),
# a blank line, then the raw bytes of the part up to the next boundary.
#
# Parts can be file uploads, so the body is arbitrary bytes (it may contain
# 0x00 and even byte sequences that LOOK like the boundary without the leading
# CRLF "--"). Therefore everything here scans raw List[UInt8] / Span[UInt8, _] and
# NEVER lower-cases or String-decodes the body as a whole. Only the per-part
# header block (ASCII) is turned into a String for parsing.

from std.memory import UnsafePointer, alloc
from std.builtin.type_aliases import MutExternalOrigin

comptime MpBytePtr = UnsafePointer[UInt8, MutExternalOrigin]


@fieldwise_init
struct Part(Copyable, Movable):
    """One parsed multipart part. `filename`/`content_type` are "" when absent.
    `data` holds the part's raw bytes (binary-safe)."""

    var name: String
    var filename: String
    var content_type: String
    var data: List[UInt8]


def boundary_from_content_type(ct: String) -> String:
    """Extract the boundary from a Content-Type like
    `multipart/form-data; boundary=----abc`. Returns "" if none. The boundary
    value may be double-quoted (`boundary="..."`)."""
    var idx = ct.find("boundary=")
    if idx < 0:
        return String("")
    var start = idx + 9  # len("boundary=")
    var sp = ct.as_bytes()
    var n = ct.byte_length()
    # quoted form: boundary="..."
    if start < n and Int(sp[start]) == 34:  # '"'
        var j = start + 1
        while j < n and Int(sp[j]) != 34:
            j += 1
        return _bytes_to_str(ct, start + 1, j)
    # unquoted: up to ';' or end (also trim trailing whitespace)
    var k = start
    while k < n and Int(sp[k]) != 59:  # ';'
        k += 1
    var end = k
    while end > start and (Int(sp[end - 1]) == 32 or Int(sp[end - 1]) == 9):
        end -= 1
    return _bytes_to_str(ct, start, end)


def _bytes_to_str(s: String, start: Int, end: Int) -> String:
    var n = end - start
    if n <= 0:
        return String("")
    var sp = s.as_bytes()
    var base = MpBytePtr(unsafe_from_address=Int(sp.unsafe_ptr()) + start)
    return String(StringSlice(ptr=base, length=n))


fn _span_str(sp: Span[UInt8, _], start: Int, end: Int) -> String:
    """Make a String from raw bytes sp[start:end). Only used for the ASCII header
    block of a part — never for body bytes."""
    var n = end - start
    if n <= 0:
        return String("")
    var base = MpBytePtr(unsafe_from_address=Int(sp.unsafe_ptr()) + start)
    return String(StringSlice(ptr=base, length=n))


fn _find_seq(hay: Span[UInt8, _], needle: Span[UInt8, _], start: Int) -> Int:
    """First index >= start where `needle` occurs in `hay`, or -1. Pure byte
    compare — safe over binary data."""
    var hn = len(hay)
    var nn = len(needle)
    if nn == 0:
        return start
    var i = start
    while i + nn <= hn:
        var ok = True
        for j in range(nn):
            if hay[i + j] != needle[j]:
                ok = False
                break
        if ok:
            return i
        i += 1
    return -1


fn _find_crlfcrlf(hay: Span[UInt8, _], start: Int, limit: Int) -> Int:
    """Index of the "\\r\\n\\r\\n" header/body separator within [start, limit), or
    -1. Used to split a part's header block from its data."""
    var i = start
    while i + 3 < limit:
        if (Int(hay[i]) == 13 and Int(hay[i + 1]) == 10
                and Int(hay[i + 2]) == 13 and Int(hay[i + 3]) == 10):
            return i
        i += 1
    return -1


def _strip(s: String) -> String:
    return String(s.strip())


def _ascii_lower(s: String) -> String:
    return String(s.lower())


def _cd_attr(disposition: String, key: String) raises -> String:
    """Pull a quoted attribute out of a Content-Disposition value, e.g.
    name="x" -> x. Returns "" if the key isn't present. Looks for `key="` and
    reads to the closing quote (filenames/names with ';' inside quotes survive)."""
    var needle = key + "=\""
    var idx = disposition.find(needle)
    if idx < 0:
        return String("")
    var start = idx + needle.byte_length()
    var sp = disposition.as_bytes()
    var n = disposition.byte_length()
    var j = start
    while j < n and Int(sp[j]) != 34:  # '"'
        j += 1
    return _bytes_to_str(disposition, start, j)


def _parse_part(body_sp: Span[UInt8, _], pstart: Int, pend: Int) raises -> Part:
    """Parse one part occupying body[pstart:pend) (between two boundary lines,
    excluding the leading CRLF after the opening delimiter and the trailing CRLF
    before the next delimiter). Splits headers from data on the blank line."""
    var hdr_end = _find_crlfcrlf(body_sp, pstart, pend)
    var name = String("")
    var filename = String("")
    var content_type = String("")
    var data_start: Int
    if hdr_end < 0:
        # No header block found — treat whole region as data (degenerate).
        data_start = pstart
    else:
        var head = _span_str(body_sp, pstart, hdr_end)  # ASCII header block
        var lines = head.split("\r\n")
        for i in range(len(lines)):
            var line = String(lines[i])
            var c = line.find(":")
            if c <= 0:
                continue
            var hkey = _ascii_lower(_strip(_bytes_to_str(line, 0, c)))
            var hval = _strip(_bytes_to_str(line, c + 1, line.byte_length()))
            if hkey == "content-disposition":
                name = _cd_attr(hval, String("name"))
                filename = _cd_attr(hval, String("filename"))
            elif hkey == "content-type":
                content_type = hval
        data_start = hdr_end + 4  # skip "\r\n\r\n"
    var data = List[UInt8]()
    for x in range(data_start, pend):
        data.append(body_sp[x])
    return Part(name, filename, content_type, data^)


def parse_multipart(body: List[UInt8], boundary: String) raises -> List[Part]:
    """Parse a multipart/form-data body. Splits on the `--<boundary>` delimiter,
    handles the final `--<boundary>--` terminator, and returns one Part per
    section. Binary-safe: part data (including 0x00 and boundary-like-but-not
    byte runs) is preserved exactly."""
    var parts = List[Part]()
    var body_sp = Span(body)
    var total = len(body)

    # The on-wire delimiter is CRLF + "--" + boundary, except the very first one
    # which has no leading CRLF. We search for "--" + boundary and then handle
    # the CRLF that precedes the *content-bearing* boundaries ourselves.
    var dash_boundary = String("--") + boundary
    var db = dash_boundary.as_bytes()
    var db_list = List[UInt8]()
    for i in range(dash_boundary.byte_length()):
        db_list.append(db[i])
    var db_sp = Span(db_list)
    var dblen = len(db_list)

    # The needle for *subsequent* delimiters is CRLF + "--" + boundary. RFC 7578
    # requires every part to be preceded by CRLF then the dash-boundary, so a
    # delimiter is only real when that leading CRLF is present. Searching for the
    # CRLF-prefixed form means a "--boundary" run that appears *inside* part data
    # (not preceded by CRLF) is NOT mistaken for a delimiter — this is what makes
    # the parser binary-safe.
    var crlf_db_list = List[UInt8]()
    crlf_db_list.append(13)  # \r
    crlf_db_list.append(10)  # \n
    for i in range(dblen):
        crlf_db_list.append(db_list[i])
    var crlf_db_sp = Span(crlf_db_list)

    # Find the first delimiter (no leading CRLF required for the opener).
    var pos = _find_seq(body_sp, db_sp, 0)
    if pos < 0:
        return parts^  # no boundary at all -> no parts

    while pos >= 0:
        var after = pos + dblen
        # Terminator?  "--boundary--"
        if after + 1 < total and Int(body_sp[after]) == 45 and Int(body_sp[after + 1]) == 45:
            break
        # Otherwise the delimiter is followed by CRLF, then the part headers.
        var content_start = after
        if content_start + 1 < total and Int(body_sp[content_start]) == 13 and Int(body_sp[content_start + 1]) == 10:
            content_start += 2
        # Find the next delimiter via its mandatory CRLF prefix.
        var nxt = _find_seq(body_sp, crlf_db_sp, content_start)
        if nxt < 0:
            # Malformed (no closing boundary) — take the rest as this part.
            parts.append(_parse_part(body_sp, content_start, total))
            break
        # nxt points at the CRLF; the part data ends there (CRLF is framing).
        parts.append(_parse_part(body_sp, content_start, nxt))
        pos = nxt + 2  # advance past the CRLF to land on the "--boundary"

    return parts^


def build_multipart_part(
    mut body: List[UInt8],
    boundary: String,
    name: String,
    filename: String,
    content_type: String,
    data: List[UInt8],
) raises:
    """Append one part (a `--boundary` delimiter line + headers + data) to the
    `body` buffer being built. Used by tests/clients to construct multipart
    bodies; binary `data` is copied verbatim. Caller writes the final
    `--boundary--` terminator (see build_multipart_end)."""
    _append_str(body, String("--") + boundary + "\r\n")
    var cd = String("Content-Disposition: form-data; name=\"") + name + "\""
    if filename.byte_length() > 0:
        cd += "; filename=\"" + filename + "\""
    _append_str(body, cd + "\r\n")
    if content_type.byte_length() > 0:
        _append_str(body, String("Content-Type: ") + content_type + "\r\n")
    _append_str(body, String("\r\n"))
    for i in range(len(data)):
        body.append(data[i])
    _append_str(body, String("\r\n"))


def build_multipart_end(mut body: List[UInt8], boundary: String) raises:
    """Write the closing `--boundary--` terminator to finish a multipart body."""
    _append_str(body, String("--") + boundary + "--\r\n")


fn _append_str(mut body: List[UInt8], s: String):
    var sp = s.as_bytes()
    for i in range(s.byte_length()):
        body.append(sp[i])
