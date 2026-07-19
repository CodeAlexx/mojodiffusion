# json.serialize — JSONValue -> JSON text. 100% Mojo, no FFI.
#
# Builds output into a byte buffer (not String concatenation) so serializing a
# large tree is linear, not quadratic. Strings are escaped per RFC 8259; control
# characters below 0x20 become \u00XX. dumps() is compact; dumps_pretty() indents.

from std.memory import UnsafePointer, alloc
from std.builtin.type_aliases import MutExternalOrigin
from json.value import (
    JSONValue, JSON_NULL, JSON_BOOL, JSON_INT, JSON_FLOAT, JSON_STR, JSON_ARR, JSON_OBJ,
)

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]
comptime HEX = "0123456789abcdef"


def _append_str(mut buf: List[UInt8], s: String):
    var sb = s.as_bytes()
    for i in range(s.byte_length()):
        buf.append(sb[i])


def _write_escaped(mut buf: List[UInt8], s: String):
    var hx = String(HEX)
    var hb = hx.as_bytes()
    var sb = s.as_bytes()
    buf.append(0x22)  # "
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
            buf.append(UInt8(c))  # UTF-8 bytes pass through
    buf.append(0x22)  # "


def _write(v: JSONValue, mut buf: List[UInt8], indent: Int, depth: Int) raises:
    var pretty = indent > 0
    if v.kind == JSON_NULL:
        _append_str(buf, "null")
    elif v.kind == JSON_BOOL:
        _append_str(buf, "true" if v.as_bool() else "false")
    elif v.kind == JSON_INT:
        _append_str(buf, String(v.as_int()))
    elif v.kind == JSON_FLOAT:
        _append_str(buf, String(v.as_float()))
    elif v.kind == JSON_STR:
        _write_escaped(buf, v.as_string())
    elif v.kind == JSON_ARR:
        var n = v.length()
        if n == 0:
            _append_str(buf, "[]")
            return
        buf.append(0x5B)  # [
        for i in range(n):
            if i > 0:
                buf.append(0x2C)  # ,
            if pretty:
                _newline_indent(buf, indent, depth + 1)
            _write(v[i], buf, indent, depth + 1)
        if pretty:
            _newline_indent(buf, indent, depth)
        buf.append(0x5D)  # ]
    elif v.kind == JSON_OBJ:
        var ks = v.keys()
        if len(ks) == 0:
            _append_str(buf, "{}")
            return
        buf.append(0x7B)  # {
        for i in range(len(ks)):
            if i > 0:
                buf.append(0x2C)
            if pretty:
                _newline_indent(buf, indent, depth + 1)
            _write_escaped(buf, ks[i])
            buf.append(0x3A)  # :
            if pretty:
                buf.append(0x20)
            _write(v[ks[i]], buf, indent, depth + 1)
        if pretty:
            _newline_indent(buf, indent, depth)
        buf.append(0x7D)  # }


def _newline_indent(mut buf: List[UInt8], indent: Int, depth: Int):
    buf.append(0x0A)  # \n
    for _ in range(indent * depth):
        buf.append(0x20)  # space


def _b2s(b: List[UInt8]) -> String:
    var n = len(b)
    if n == 0:
        return String("")
    var buf = alloc[UInt8](n)
    for i in range(n):
        buf[i] = b[i]
    var s = String(StringSlice(ptr=BytePtr(unsafe_from_address=Int(buf)), length=n))
    buf.free()
    return s


def dumps(v: JSONValue) raises -> String:
    """Serialize to compact JSON text."""
    var buf = List[UInt8]()
    _write(v, buf, 0, 0)
    return _b2s(buf)


def dumps_pretty(v: JSONValue, indent: Int = 2) raises -> String:
    """Serialize to indented (pretty) JSON text."""
    var buf = List[UInt8]()
    _write(v, buf, indent, 0)
    return _b2s(buf)
