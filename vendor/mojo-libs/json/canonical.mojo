# json.canonical — canonical / minified / sorted JSON serialization. 100% Mojo, no FFI.
#
# Provides three entry points:
#   minify(v)         -> compact JSON text (no insignificant whitespace)
#   dumps_sorted(v)   -> compact JSON with object keys sorted at every level
#   canonicalize(text)-> loads(text) then dumps_sorted (parse + canonical re-emit)
#
# Key ordering follows RFC 8785 (JCS): object members are sorted lexicographically
# by the UTF-16 code units of their (unescaped) key strings. For all characters in
# the Basic Multilingual Plane (code points < 0x10000) UTF-16 code-unit order is
# identical to Unicode code-point order, which is in turn identical to UTF-8 byte
# order. So we sort by UTF-8 bytes — this is exactly correct for the BMP and matches
# Python's `json.dumps(..., sort_keys=True)` for those keys.
#
# ASTRAL-PLANE CAVEAT: for supplementary-plane characters (code points >= 0x10000),
# UTF-16 represents them as a surrogate pair whose lead unit is in 0xD800..0xDBFF.
# In UTF-16 order such a character sorts BEFORE characters in 0xE000..0xFFFF, but in
# UTF-8/code-point order it sorts AFTER them. We sort by UTF-8 bytes, which therefore
# diverges from strict RFC 8785 ordering only when keys mix astral-plane characters
# with BMP characters in 0xE000..0xFFFF. This matches Python's `sort_keys=True`
# (which also sorts by code point), and is documented here as an intentional limit.
#
# STRING ESCAPING: only the JSON-required characters are escaped — the control
# characters U+0000..U+001F (using the short escapes \b \f \n \r \t where defined,
# otherwise lowercase \u00XX), plus " and \. Forward slash '/' is NOT escaped and
# non-ASCII is emitted as raw UTF-8 bytes. This matches the existing dumps() in
# serialize.mojo and JCS.
#
# NUMBERS: integers are emitted as their decimal form. Floats are emitted via Mojo's
# String(Float64), which gives a shortest-ish round-trippable representation but is
# NOT the full ECMAScript Number-to-string algorithm that RFC 8785 mandates. For
# integer-valued and typical decimal data this matches; exotic float edge cases may
# differ in exponent/format from a strict JCS implementation. This is the same
# float path the existing serialize.mojo uses, and the limitation is called out
# honestly rather than hidden.

from std.memory import UnsafePointer, alloc
from std.builtin.type_aliases import MutExternalOrigin
from json.value import (
    JSONValue, JSON_NULL, JSON_BOOL, JSON_INT, JSON_FLOAT, JSON_STR, JSON_ARR, JSON_OBJ,
)
from json.parser import loads

comptime CanonBytePtr = UnsafePointer[UInt8, MutExternalOrigin]
comptime CANON_HEX = "0123456789abcdef"


def _c_append_str(mut buf: List[UInt8], s: String):
    var sb = s.as_bytes()
    for i in range(s.byte_length()):
        buf.append(sb[i])


def _c_write_escaped(mut buf: List[UInt8], s: String):
    var hx = String(CANON_HEX)
    var hb = hx.as_bytes()
    var sb = s.as_bytes()
    buf.append(0x22)  # opening "
    for i in range(s.byte_length()):
        var c = Int(sb[i])
        if c == 0x22:  # "
            buf.append(0x5C)
            buf.append(0x22)
        elif c == 0x5C:  # backslash
            buf.append(0x5C)
            buf.append(0x5C)
        elif c == 0x08:
            buf.append(0x5C)
            buf.append(0x62)  # \b
        elif c == 0x0C:
            buf.append(0x5C)
            buf.append(0x66)  # \f
        elif c == 0x0A:
            buf.append(0x5C)
            buf.append(0x6E)  # \n
        elif c == 0x0D:
            buf.append(0x5C)
            buf.append(0x72)  # \r
        elif c == 0x09:
            buf.append(0x5C)
            buf.append(0x74)  # \t
        elif c < 0x20:
            buf.append(0x5C)
            buf.append(0x75)  # \u
            buf.append(0x30)
            buf.append(0x30)  # 00
            buf.append(hb[(c >> 4) & 0xF])
            buf.append(hb[c & 0xF])
        else:
            buf.append(UInt8(c))  # UTF-8 bytes pass through unescaped
    buf.append(0x22)  # closing "


# Lexicographic comparison of two UTF-8 strings by their raw bytes.
# Returns True if a < b. For BMP code points this equals UTF-16 code-unit order.
def _key_less(a: String, b: String) -> Bool:
    var ab = a.as_bytes()
    var bb = b.as_bytes()
    var an = a.byte_length()
    var bn = b.byte_length()
    var m = an if an < bn else bn
    for i in range(m):
        var x = Int(ab[i])
        var y = Int(bb[i])
        if x != y:
            return x < y
    return an < bn


# Simple in-place insertion sort over a List[String]. Used instead of relying on a
# std sort for List[String] (availability varies); insertion sort is stable and
# trivially correct for the small key-count case typical of JSON objects.
def _sort_keys(mut ks: List[String]):
    var n = len(ks)
    for i in range(1, n):
        var j = i
        while j > 0 and _key_less(ks[j], ks[j - 1]):
            var tmp = ks[j].copy()
            ks[j] = ks[j - 1].copy()
            ks[j - 1] = tmp.copy()
            j -= 1


def _c_write(v: JSONValue, mut buf: List[UInt8]) raises:
    if v.kind == JSON_NULL:
        _c_append_str(buf, "null")
    elif v.kind == JSON_BOOL:
        _c_append_str(buf, "true" if v.as_bool() else "false")
    elif v.kind == JSON_INT:
        _c_append_str(buf, String(v.as_int()))
    elif v.kind == JSON_FLOAT:
        _c_append_str(buf, String(v.as_float()))
    elif v.kind == JSON_STR:
        _c_write_escaped(buf, v.as_string())
    elif v.kind == JSON_ARR:
        var n = v.length()
        if n == 0:
            _c_append_str(buf, "[]")
            return
        buf.append(0x5B)  # [
        for i in range(n):
            if i > 0:
                buf.append(0x2C)  # ,
            _c_write(v[i], buf)
        buf.append(0x5D)  # ]
    elif v.kind == JSON_OBJ:
        var ks = v.keys()
        if len(ks) == 0:
            _c_append_str(buf, "{}")
            return
        _sort_keys(ks)
        buf.append(0x7B)  # {
        for i in range(len(ks)):
            if i > 0:
                buf.append(0x2C)  # ,
            _c_write_escaped(buf, ks[i])
            buf.append(0x3A)  # :
            _c_write(v[ks[i]], buf)  # recursion sorts nested objects too
        buf.append(0x7D)  # }


# Compact writer that does NOT sort keys (preserves insertion order). Used by
# minify(). Shares escaping/number rules with the sorted writer.
def _c_write_unsorted(v: JSONValue, mut buf: List[UInt8]) raises:
    if v.kind == JSON_NULL:
        _c_append_str(buf, "null")
    elif v.kind == JSON_BOOL:
        _c_append_str(buf, "true" if v.as_bool() else "false")
    elif v.kind == JSON_INT:
        _c_append_str(buf, String(v.as_int()))
    elif v.kind == JSON_FLOAT:
        _c_append_str(buf, String(v.as_float()))
    elif v.kind == JSON_STR:
        _c_write_escaped(buf, v.as_string())
    elif v.kind == JSON_ARR:
        var n = v.length()
        if n == 0:
            _c_append_str(buf, "[]")
            return
        buf.append(0x5B)
        for i in range(n):
            if i > 0:
                buf.append(0x2C)
            _c_write_unsorted(v[i], buf)
        buf.append(0x5D)
    elif v.kind == JSON_OBJ:
        var ks = v.keys()
        if len(ks) == 0:
            _c_append_str(buf, "{}")
            return
        buf.append(0x7B)
        for i in range(len(ks)):
            if i > 0:
                buf.append(0x2C)
            _c_write_escaped(buf, ks[i])
            buf.append(0x3A)
            _c_write_unsorted(v[ks[i]], buf)
        buf.append(0x7D)


def _c_b2s(b: List[UInt8]) -> String:
    var n = len(b)
    if n == 0:
        return String("")
    var buf = alloc[UInt8](n)
    for i in range(n):
        buf[i] = b[i]
    var s = String(StringSlice(ptr=CanonBytePtr(unsafe_from_address=Int(buf)), length=n))
    buf.free()
    return s


def minify(v: JSONValue) raises -> String:
    """Compact serialization with no insignificant whitespace, insertion key order.

    Functionally equivalent to serialize.dumps(); reimplemented here so the
    canonical module is self-contained and shares one escaping path.
    """
    var buf = List[UInt8]()
    _c_write_unsorted(v, buf)
    return _c_b2s(buf)


def dumps_sorted(v: JSONValue) raises -> String:
    """Compact serialization with object keys sorted lexicographically (UTF-16
    code-unit / RFC 8785 JCS order) recursively at every level."""
    var buf = List[UInt8]()
    _c_write(v, buf)
    return _c_b2s(buf)


def canonicalize(text: String) raises -> String:
    """Parse `text` then re-emit with dumps_sorted, so two semantically-equal
    documents differing only in key order or whitespace produce byte-identical
    output."""
    var v = loads(text)
    return dumps_sorted(v)
