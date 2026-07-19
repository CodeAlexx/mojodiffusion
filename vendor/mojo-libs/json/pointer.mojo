# json.pointer — JSON Pointer (RFC 6901). 100% Mojo, no FFI.
#
# A JSON Pointer is a string of "/"-separated reference tokens that identify a
# value within a JSON document. The empty string "" references the whole
# document. Within a token, "~1" decodes to "/" and "~0" decodes to "~"
# (the order matters: ~1 first, then ~0). Array indices are decimal numbers;
# "-" refers to the (nonexistent) element after the last array element.

from json.value import JSONValue


def _unescape_token(tok: String) -> String:
    # RFC 6901 §4: transform "~1" -> "/" then "~0" -> "~".
    # We scan once, byte-by-byte, honoring that order.
    var sb = tok.as_bytes()
    var n = tok.byte_length()
    var out = String("")
    var i = 0
    while i < n:
        var c = Int(sb[i])
        if c == 0x7E and i + 1 < n:  # '~'
            var nxt = Int(sb[i + 1])
            if nxt == 0x31:  # '1'
                out += "/"
                i += 2
                continue
            elif nxt == 0x30:  # '0'
                out += "~"
                i += 2
                continue
        out += chr(c)
        i += 1
    return out


def split_pointer(ptr: String) raises -> List[String]:
    """Split a JSON Pointer into its unescaped reference tokens.

    "" -> [] (whole document); "/a/b/0" -> ["a","b","0"]; a leading "/" is
    required when the pointer is non-empty.
    """
    var toks = List[String]()
    if ptr.byte_length() == 0:
        return toks^
    var sb = ptr.as_bytes()
    var n = ptr.byte_length()
    if Int(sb[0]) != 0x2F:  # '/'
        raise Error("invalid JSON Pointer (must be empty or start with '/'): " + ptr)
    # Split on '/' starting after the leading slash.
    var cur = String("")
    var i = 1
    while i < n:
        var c = Int(sb[i])
        if c == 0x2F:  # '/'
            toks.append(_unescape_token(cur))
            cur = String("")
        else:
            cur += chr(c)
        i += 1
    toks.append(_unescape_token(cur))
    return toks^


def _parse_index(tok: String) raises -> Int:
    # Strict array index: a non-negative decimal integer with no leading zeros
    # (except "0" itself). "-" is handled by callers, not here.
    var sb = tok.as_bytes()
    var n = tok.byte_length()
    if n == 0:
        raise Error("empty array index token")
    if n > 1 and Int(sb[0]) == 0x30:  # leading zero
        raise Error("array index has leading zero: " + tok)
    var v = 0
    for i in range(n):
        var c = Int(sb[i])
        if c < 0x30 or c > 0x39:
            raise Error("non-numeric array index: " + tok)
        v = v * 10 + (c - 0x30)
    return v


def resolve(doc: JSONValue, ptr: String) raises -> JSONValue:
    """Resolve a JSON Pointer against a document; raise if the target is missing."""
    var toks = split_pointer(ptr)
    var cur = doc.copy()
    for i in range(len(toks)):
        var tok = toks[i]
        if cur.is_object():
            if not cur.contains(tok):
                raise Error("JSON Pointer: object has no key '" + tok + "'")
            cur = cur[tok]
        elif cur.is_array():
            if tok == "-":
                raise Error("JSON Pointer: '-' does not reference an existing element")
            var idx = _parse_index(tok)
            if idx < 0 or idx >= cur.length():
                raise Error("JSON Pointer: array index out of range: " + tok)
            cur = cur[idx]
        else:
            raise Error("JSON Pointer: cannot descend into scalar at token '" + tok + "'")
    return cur^
