# serenitymojo.serve.product_manifest -- small helpers for backend result sidecars.

from std.memory import alloc

from serenitymojo.io.ffi import (
    BytePtr, sys_open, sys_close, sys_pwrite, O_WRONLY, O_CREAT, O_TRUNC,
)


def _append_ascii_bytes(mut out: List[UInt8], text: String):
    var bs = text.as_bytes()
    for i in range(text.byte_length()):
        out.append(bs[i])


def _hex_digit(n: UInt8) -> UInt8:
    if n < 10:
        return n + 0x30
    return (n - 10) + 0x61


def _append_control_escape(mut out: List[UInt8], ch: UInt8):
    _append_ascii_bytes(out, String("\\u00"))
    out.append(_hex_digit(ch >> 4))
    out.append(_hex_digit(ch & 0x0F))


def json_escape(s: String) raises -> String:
    var out = List[UInt8]()
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        var ch = bs[i]
        if ch == 0x22:
            _append_ascii_bytes(out, String("\\\""))
        elif ch == 0x5C:
            _append_ascii_bytes(out, String("\\\\"))
        elif ch == 0x0A:
            _append_ascii_bytes(out, String("\\n"))
        elif ch == 0x0D:
            _append_ascii_bytes(out, String("\\r"))
        elif ch == 0x09:
            _append_ascii_bytes(out, String("\\t"))
        elif ch < 0x20:
            _append_control_escape(out, ch)
        else:
            out.append(ch)
    return String(from_utf8=out)


def json_bool(v: Bool) -> String:
    return String("true") if v else String("false")


def peak_vram_mib(total_vram: Int, min_free: Int) -> Float64:
    return Float64(total_vram - min_free) / 1048576.0


def write_text_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("product_manifest: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("product_manifest: short write to ") + path)
