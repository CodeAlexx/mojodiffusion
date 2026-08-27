# Focused host-only safetensors metadata foundation gate.
# Proves exact header escaping, tensor bytes, reader exposure, and malformed /
# duplicate metadata rejection. No DeviceContext is constructed.

from std.collections import Dict, List
from std.memory import alloc

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import (
    BytePtr,
    O_CREAT,
    O_RDONLY,
    O_TRUNC,
    O_WRONLY,
    sys_close,
    sys_open,
    sys_pread,
    sys_pwrite,
    sys_remove,
)
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.safetensors_writer import (
    HostTensorDesc,
    save_safetensors_host_with_metadata,
)


comptime GOOD = "/tmp/serenity_safetensors_metadata_roundtrip.safetensors"
comptime BAD_DUP_KEY = "/tmp/serenity_safetensors_metadata_dup_key.safetensors"
comptime BAD_DUP_TOP = "/tmp/serenity_safetensors_metadata_dup_top.safetensors"
comptime BAD_VALUE = "/tmp/serenity_safetensors_metadata_bad_value.safetensors"
comptime BAD_CONTROL = "/tmp/serenity_safetensors_metadata_bad_control.safetensors"


def _require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


def _write_all(fd: Int, data: List[UInt8], offset: Int) raises:
    if len(data) == 0:
        return
    var wrote = sys_pwrite(
        fd,
        BytePtr(unsafe_from_address=Int(data.unsafe_ptr())),
        len(data),
        offset,
    )
    if wrote != len(data):
        raise Error("metadata fixture short write")


def _string_bytes(value: String) -> List[UInt8]:
    var out = List[UInt8]()
    for byte in value.as_bytes():
        out.append(byte)
    return out^


def _write_raw(path: String, header: String) raises:
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd < 0:
        raise Error(String("cannot create malformed fixture: ") + path)
    var len_bytes = List[UInt8]()
    var size = header.byte_length()
    var work = size
    for _ in range(8):
        len_bytes.append(UInt8(work & 0xFF))
        work = work >> 8
    var header_bytes = _string_bytes(header)
    var payload: List[UInt8] = [UInt8(0), UInt8(0), UInt8(0), UInt8(0)]
    try:
        _write_all(fd, len_bytes, 0)
        _write_all(fd, header_bytes, 8)
        _write_all(fd, payload, 8 + size)
    except e:
        _ = sys_close(fd)
        raise e^
    _ = sys_close(fd)


def _read_header(path: String) raises -> String:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error("cannot open round-trip artifact")
    var lenbuf = alloc[UInt8](8)
    if sys_pread(fd, lenbuf, 8, 0) != 8:
        lenbuf.free()
        _ = sys_close(fd)
        raise Error("cannot read round-trip header length")
    var size = 0
    for index in range(8):
        size |= Int(lenbuf[index]) << (8 * index)
    lenbuf.free()
    var buf = alloc[UInt8](size)
    if sys_pread(fd, buf, size, 8) != size:
        buf.free()
        _ = sys_close(fd)
        raise Error("cannot read round-trip header")
    _ = sys_close(fd)
    var value = String(
        StringSlice(unsafe_from_utf8=Span(unsafe_ptr=buf, length=size))
    )
    buf.free()
    return value^


def _expect_open_rejected(path: String, label: String) raises:
    var rejected = False
    try:
        _ = SafeTensors.open(path)
    except:
        rejected = True
    _require(rejected, label + String(" was accepted"))


def _cleanup():
    for path in [
        String(GOOD), String(BAD_DUP_KEY), String(BAD_DUP_TOP),
        String(BAD_VALUE), String(BAD_CONTROL),
    ]:
        _ = sys_remove(path)
        _ = sys_remove(path + String(".tmp"))


def main() raises:
    _cleanup()
    var tensor_bytes: List[UInt8] = [
        UInt8(0x00), UInt8(0x01), UInt8(0x22), UInt8(0x5C),
        UInt8(0x7F), UInt8(0x80), UInt8(0xFE), UInt8(0xFF),
    ]
    var metadata = Dict[String, String]()
    metadata[String("schema")] = String("serenity.minimax_h3.cache.latent.v2")
    metadata[String("task")] = String("t2va")
    metadata[String("empty")] = String("")
    metadata[String("caption")] = String(
        "  vrtlEri2 said \"hello\" \\path\nsecond\tline  \r\n"
    )
    metadata[String("x\"\\\n")] = String("key escapes")

    save_safetensors_host_with_metadata(
        [String("latents_1x1x1_float32")],
        [HostTensorDesc(STDtype.F32, [2], tensor_bytes.copy())],
        metadata,
        String(GOOD),
    )

    # Deterministic writer order and exact RFC-8259 escapes in header bytes.
    var expected_header = String(
        '{"__metadata__":{"caption":"  vrtlEri2 said \\"hello\\" '
        '\\\\path\\nsecond\\tline  \\r\\n","empty":"",'
        '"schema":"serenity.minimax_h3.cache.latent.v2","task":"t2va",'
        '"x\\"\\\\\\n":"key escapes"},'
        '"latents_1x1x1_float32":{"dtype":"F32","shape":[2],'
        '"data_offsets":[0,8]}}'
    )
    _require(_read_header(String(GOOD)) == expected_header, String("exact JSON header mismatch"))

    var st = SafeTensors.open(String(GOOD))
    _require(st.count() == 1, String("tensor count mismatch"))
    var got = st.tensor_bytes(String("latents_1x1x1_float32"))
    _require(len(got) == len(tensor_bytes), String("tensor byte count mismatch"))
    for index in range(len(tensor_bytes)):
        _require(got[index] == tensor_bytes[index], String("tensor byte mismatch"))
    _require(st.metadata_value(String("schema")) == metadata[String("schema")], String("schema metadata mismatch"))
    _require(st.metadata_value(String("task")) == String("t2va"), String("task metadata mismatch"))
    _require(st.metadata_value(String("empty")) == String(""), String("empty metadata mismatch"))
    _require(st.metadata_value(String("caption")) == metadata[String("caption")], String("caption metadata mismatch"))
    _require(st.metadata_value(String("x\"\\\n")) == String("key escapes"), String("escaped-key metadata mismatch"))
    var metadata_copy = st.metadata()
    _require(len(metadata_copy) == 5, String("metadata map size mismatch"))
    _require(not st.has_metadata(String("missing")), String("missing metadata reported present"))

    var missing_rejected = False
    try:
        _ = st.metadata_value(String("missing"))
    except:
        missing_rejected = True
    _require(missing_rejected, String("missing metadata lookup did not reject"))

    _write_raw(
        String(BAD_DUP_KEY),
        String('{"__metadata__":{"k":"one","k":"two"},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'),
    )
    _write_raw(
        String(BAD_DUP_TOP),
        String('{"__metadata__":{"k":"one"},"__metadata__":{"q":"two"},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'),
    )
    _write_raw(
        String(BAD_VALUE),
        String('{"__metadata__":{"k":7},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'),
    )
    _write_raw(
        String(BAD_CONTROL),
        String('{"__metadata__":{"bad\nkey":"value"},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'),
    )
    _expect_open_rejected(String(BAD_DUP_KEY), String("duplicate metadata key"))
    _expect_open_rejected(String(BAD_DUP_TOP), String("duplicate __metadata__"))
    _expect_open_rejected(String(BAD_VALUE), String("non-string metadata value"))
    _expect_open_rejected(String(BAD_CONTROL), String("unescaped metadata control byte"))

    _cleanup()
    print("PASS safetensors metadata host round-trip gate")
    print("  tensor bytes: exact")
    print("  metadata strings/JSON escaping: exact")
    print("  malformed/duplicate metadata: rejected")
    print("  DeviceContext: NOT CREATED")
