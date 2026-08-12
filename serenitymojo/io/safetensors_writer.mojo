# safetensors_writer.mojo — SafeTensors single-file WRITER. Pure-Mojo inverse of
# io/safetensors.mojo (the reader). Linux x86-64. Training-path artifact: lets
# trained weights / LoRA adapters be SAVED (the inference port only reads).
#
# Byte-exact format (must match the reader in io/safetensors.mojo line-for-line):
#   1. 8 bytes: header_len as little-endian u64 (reader mmap.rs:175-178).
#   2. header_len bytes: JSON header object, schema EXACTLY:
#        {"<name>":{"dtype":"<STR>","shape":[<ints>],"data_offsets":[<s>,<e>]},
#         ...}
#      data_offsets are byte offsets RELATIVE to the start of the data segment
#      (i.e. relative to byte 8+header_len in the file). start/end give the
#      half-open byte range [s, e) of the tensor's bytes (reader: size = e - s).
#   3. concatenated tensor bytes, in the SAME ORDER the entries appear in the
#      header, each tensor's bytes placed at its declared data_offsets[0].
#
# The reader (parse_header in io/json_header.mojo) tolerates whitespace, field
# order, and an optional "__metadata__" key. We emit the canonical compact form
# the `safetensors` Python library produces (no spaces, keys in insertion order,
# data_offsets contiguous and increasing) so external tools open it too.
#
# dtype/byte-size come straight from STDtype.name()/byte_size() (io/dtype.mojo).
# We support every STDtype the writer is handed; F32 and BF16 are the storage
# dtypes the training port uses, and are covered by the round-trip smoke.
#
# Raw-byte access: we copy each Tensor's device buffer (`Tensor.buf`, a
# DeviceBuffer[DType.uint8] of exactly numel*byte_size bytes) straight to a host
# buffer with the SAME enqueue_copy pattern Tensor.to_host uses, then pwrite the
# raw bytes. This is byte-exact for BF16/F16/F32 (no F32 round-trip), which is
# what makes the reload bit-identical to the original storage.
#
# Mojo 1.0.0b1: `def` not `fn`; Tensor is move-only so collections hold
# ArcPointer[Tensor]; file I/O via io/ffi.mojo (sys_open/sys_pwrite/sys_close);
# uint8 buffers throughout.

from std.memory import alloc, UnsafePointer, ArcPointer
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from .dtype import STDtype
from .ffi import (
    BytePtr,
    sys_open,
    sys_close,
    sys_pwrite,
    sys_rename,
    sys_remove,
    O_WRONLY,
    O_CREAT,
    O_TRUNC,
)

# The Tensor type lives at serenitymojo/tensor.mojo (package root). Import it by
# absolute package path; io/ is a subpackage so a relative `..tensor` import is
# brittle across Mojo versions — the absolute form is what the rest of the tree
# uses (see tensor.mojo importing serenitymojo.io.dtype).
from serenitymojo.tensor import Tensor


def _append_str(mut buf: List[UInt8], s: String):
    """Append a String's UTF-8 bytes to a byte buffer."""
    var src = s.as_bytes()
    for i in range(len(src)):
        buf.append(src[i])


def _append_json_string(mut buf: List[UInt8], s: String) raises:
    """Append a JSON-escaped, double-quoted string. Tensor names contain '.',
    '/', '-', digits — none need escaping — but '"' and '\\' (and control
    chars) do, so we escape them to stay valid JSON the reader's _parse_string
    can decode back exactly."""
    buf.append(0x22)  # opening '"'
    var src = s.as_bytes()
    for i in range(len(src)):
        var c = Int(src[i])
        if c == 0x22:  # '"'
            buf.append(0x5C)
            buf.append(0x22)
        elif c == 0x5C:  # '\'
            buf.append(0x5C)
            buf.append(0x5C)
        elif c == 0x0A:  # newline
            buf.append(0x5C)
            buf.append(0x6E)
        elif c == 0x0D:  # carriage return
            buf.append(0x5C)
            buf.append(0x72)
        elif c == 0x09:  # tab
            buf.append(0x5C)
            buf.append(0x74)
        elif c == 0x08:  # backspace
            buf.append(0x5C)
            buf.append(0x62)
        elif c == 0x0C:  # form feed
            buf.append(0x5C)
            buf.append(0x66)
        elif c < 0x20:
            # Other control chars -> \u00XX
            _append_str(buf, String("\\u00"))
            var hi = (c >> 4) & 0xF
            var lo = c & 0xF
            buf.append(_hex_digit(hi))
            buf.append(_hex_digit(lo))
        else:
            buf.append(UInt8(c))
    buf.append(0x22)  # closing '"'


def _hex_digit(v: Int) -> UInt8:
    """Lowercase hex digit byte for 0..15."""
    if v < 10:
        return UInt8(0x30 + v)  # '0'..'9'
    return UInt8(0x61 + (v - 10))  # 'a'..'f'


def _tensor_offsets(
    tensors: List[ArcPointer[Tensor]]
) -> List[Int]:
    """Contiguous starting byte offset of each tensor within the data segment.
    Returns a list of length len(tensors)+1: entry i is tensors[i]'s start
    offset, and the final entry is the total data-segment size. Contiguous
    layout (offsets[0]=0, offsets[i+1]=offsets[i]+nbytes(i)) is exactly what the
    canonical safetensors writer emits and what the reader's data_offsets
    parsing expects (size = data_offsets[1] - data_offsets[0])."""
    var offsets = List[Int]()
    var running = 0
    for i in range(len(tensors)):
        offsets.append(running)
        running += tensors[i][].nbytes()
    offsets.append(running)  # sentinel = total size / last tensor's end
    return offsets^


def _build_header(
    names: List[String],
    tensors: List[ArcPointer[Tensor]],
    offsets: List[Int],
) raises -> List[UInt8]:
    """Serialize the JSON header bytes (compact, insertion order) for the given
    names/tensors and their precomputed contiguous `offsets` (see
    _tensor_offsets — length len(tensors)+1). The emitted form matches the
    `safetensors` Python writer: no spaces, keys in insertion order, data_offsets
    [start,end) contiguous and increasing."""
    var hdr = List[UInt8]()
    hdr.append(0x7B)  # '{'
    for i in range(len(names)):
        if i > 0:
            hdr.append(0x2C)  # ','
        _append_json_string(hdr, names[i])
        hdr.append(0x3A)  # ':'
        hdr.append(0x7B)  # '{' open tensor object
        # "dtype":"<NAME>"
        _append_str(hdr, String('"dtype":'))
        var dt = tensors[i][].dtype()
        _append_json_string(hdr, dt.name())
        hdr.append(0x2C)  # ','
        # "shape":[...]
        _append_str(hdr, String('"shape":['))
        var shp = tensors[i][].shape()
        for d in range(len(shp)):
            if d > 0:
                hdr.append(0x2C)
            _append_str(hdr, String(shp[d]))
        hdr.append(0x5D)  # ']'
        hdr.append(0x2C)  # ','
        # "data_offsets":[start,end]
        _append_str(hdr, String('"data_offsets":['))
        _append_str(hdr, String(offsets[i]))
        hdr.append(0x2C)
        _append_str(hdr, String(offsets[i + 1]))
        hdr.append(0x5D)  # ']'
        hdr.append(0x7D)  # '}' close tensor object
    hdr.append(0x7D)  # '}'
    return hdr^


def _write_all(fd: Int, buf: BytePtr, count: Int, offset: Int) raises:
    """pwrite exactly `count` bytes from `buf` at `offset`, looping over short
    writes (pwrite(2) may write fewer than requested). Mirrors the reader's
    _pread_exact discipline in reverse."""
    var done = 0
    while done < count:
        var n = sys_pwrite(fd, buf + done, count - done, offset + done)
        if n < 0:
            raise Error("pwrite failed (I/O error)")
        if n == 0:
            raise Error("pwrite wrote 0 bytes (disk full?)")
        done += n


# ─────────────────────────────────────────────────────────────────────────────
# HOST-DIRECT write path: same file format, ZERO device involvement.
#
# The device path below stages every tensor host→device→host before pwrite —
# fine for weights already resident on the GPU, but a pure waste (and a real
# OOM exposure) for state that lives in host lists: the LTX2 trainer's 384-
# adapter F32 state is ~1.4 GB of transient DEVICE allocations on a card
# already peaking at ~22.5/24 GiB (measured 2026-07-16). Host-resident bytes
# must never touch VRAM on their way to disk.
# ─────────────────────────────────────────────────────────────────────────────
@fieldwise_init
struct HostTensorDesc(Copyable, Movable):
    """One named tensor's host-side payload for save_safetensors_host: storage
    dtype tag, logical shape, and the raw little-endian storage bytes
    (len(bytes) MUST equal numel(shape) * dtype.byte_size())."""
    var dtype: STDtype
    var shape: List[Int]
    var bytes: List[UInt8]


@fieldwise_init
struct HostBufferTensorDesc(Copyable, Movable):
    """Pinned-host sibling of HostTensorDesc for multi-gigabyte runtime caches.

    Ownership stays in the ArcPointer for the complete write, so payload bytes
    can be pwritten directly without materializing a second List[UInt8] copy."""
    var dtype: STDtype
    var shape: List[Int]
    var buffer: ArcPointer[HostBuffer[DType.uint8]]
    var nbytes: Int


def _tensor_offsets_host(descs: List[HostTensorDesc]) -> List[Int]:
    """Host-desc twin of _tensor_offsets: contiguous data-segment offsets, with
    the total size as the final sentinel entry."""
    var offsets = List[Int]()
    var running = 0
    for i in range(len(descs)):
        offsets.append(running)
        running += len(descs[i].bytes)
    offsets.append(running)
    return offsets^


def _build_header_host(
    names: List[String],
    descs: List[HostTensorDesc],
    offsets: List[Int],
) raises -> List[UInt8]:
    """Host-desc twin of _build_header: identical compact JSON, dtype/shape read
    from the descriptors instead of device Tensors."""
    var hdr = List[UInt8]()
    hdr.append(0x7B)  # '{'
    for i in range(len(names)):
        if i > 0:
            hdr.append(0x2C)  # ','
        _append_json_string(hdr, names[i])
        hdr.append(0x3A)  # ':'
        hdr.append(0x7B)  # '{'
        _append_str(hdr, String('"dtype":'))
        _append_json_string(hdr, descs[i].dtype.name())
        hdr.append(0x2C)
        _append_str(hdr, String('"shape":['))
        for d in range(len(descs[i].shape)):
            if d > 0:
                hdr.append(0x2C)
            _append_str(hdr, String(descs[i].shape[d]))
        hdr.append(0x5D)  # ']'
        hdr.append(0x2C)
        _append_str(hdr, String('"data_offsets":['))
        _append_str(hdr, String(offsets[i]))
        hdr.append(0x2C)
        _append_str(hdr, String(offsets[i + 1]))
        hdr.append(0x5D)  # ']'
        hdr.append(0x7D)  # '}'
    hdr.append(0x7D)  # '}'
    return hdr^


def save_safetensors_host(
    names: List[String],
    descs: List[HostTensorDesc],
    path: String,
) raises:
    """Write host-resident tensors to `path` as a single-file safetensors,
    byte-identical in format to save_safetensors, WITHOUT any DeviceContext:
    the payload bytes are pwritten straight from the host lists. Same atomic
    tmp-then-rename crash-safety as the device path."""
    if len(names) != len(descs):
        raise Error(
            String("save_safetensors_host: len(names)=") + String(len(names))
            + " != len(descs)=" + String(len(descs))
        )
    if len(names) == 0:
        raise Error("save_safetensors_host: refusing to write an empty file")
    for i in range(len(descs)):
        var numel = 1
        for d in range(len(descs[i].shape)):
            numel *= descs[i].shape[d]
        var expect = numel * descs[i].dtype.byte_size()
        if len(descs[i].bytes) != expect:
            raise Error(
                String("save_safetensors_host: '") + names[i] + "' byte size "
                + String(len(descs[i].bytes)) + " != numel*byte_size "
                + String(expect)
            )

    var offsets = _tensor_offsets_host(descs)
    var hdr = _build_header_host(names, descs, offsets)
    var header_len = len(hdr)

    var tmp_path = path + String(".tmp")
    var flags = O_WRONLY | O_CREAT | O_TRUNC
    var fd = sys_open(tmp_path, flags, Int32(0o644))
    if fd < 0:
        raise Error(String("save_safetensors_host: failed to open for write: ") + tmp_path)

    # 1) 8-byte LE header length at offset 0.
    var lenbuf = alloc[UInt8](8)
    var hl = header_len
    for i in range(8):
        lenbuf[i] = UInt8(hl & 0xFF)
        hl = hl >> 8
    try:
        _write_all(fd, BytePtr(unsafe_from_address=Int(lenbuf)), 8, 0)
    except e:
        lenbuf.free()
        _ = sys_close(fd)
        _ = sys_remove(tmp_path)
        raise Error(String("save_safetensors_host: writing header length: ") + String(e))
    lenbuf.free()

    # 2) header bytes at offset 8.
    var hbuf = alloc[UInt8](header_len)
    for i in range(header_len):
        hbuf[i] = hdr[i]
    try:
        _write_all(fd, BytePtr(unsafe_from_address=Int(hbuf)), header_len, 8)
    except e:
        hbuf.free()
        _ = sys_close(fd)
        _ = sys_remove(tmp_path)
        raise Error(String("save_safetensors_host: writing header bytes: ") + String(e))
    hbuf.free()

    # 3) payload bytes straight from each desc's host list (no device copy).
    var data_offset = 8 + header_len
    for i in range(len(descs)):
        var nbytes = len(descs[i].bytes)
        try:
            _write_all(
                fd,
                BytePtr(unsafe_from_address=Int(descs[i].bytes.unsafe_ptr())),
                nbytes,
                data_offset + offsets[i],
            )
        except e:
            _ = sys_close(fd)
            _ = sys_remove(tmp_path)
            raise Error(
                String("save_safetensors_host: writing tensor '") + names[i]
                + "': " + String(e)
            )

    _ = sys_close(fd)

    if sys_rename(tmp_path, path) != 0:
        _ = sys_remove(tmp_path)
        raise Error(
            String("save_safetensors_host: atomic rename ") + tmp_path + " -> "
            + path + " failed (different filesystem? out of space?)"
        )


def save_safetensors_host_buffers(
    names: List[String],
    descs: List[HostBufferTensorDesc],
    path: String,
) raises:
    """Atomic safetensors write directly from owned pinned-host buffers."""
    if len(names) != len(descs):
        raise Error(
            String("save_safetensors_host_buffers: len(names)=")
            + String(len(names)) + " != len(descs)=" + String(len(descs))
        )
    if len(names) == 0:
        raise Error("save_safetensors_host_buffers: refusing to write an empty file")

    var offsets = List[Int]()
    var header_descs = List[HostTensorDesc]()
    var running = 0
    for i in range(len(descs)):
        var numel = 1
        for d in range(len(descs[i].shape)):
            numel *= descs[i].shape[d]
        var expect = numel * descs[i].dtype.byte_size()
        if descs[i].nbytes != expect:
            raise Error(
                String("save_safetensors_host_buffers: '") + names[i]
                + String("' byte size ") + String(descs[i].nbytes)
                + String(" != numel*byte_size ") + String(expect)
            )
        offsets.append(running)
        running += descs[i].nbytes
        # Header construction reads only dtype/shape; keep the byte list empty.
        header_descs.append(HostTensorDesc(
            descs[i].dtype, descs[i].shape.copy(), List[UInt8]()
        ))
    offsets.append(running)
    var hdr = _build_header_host(names, header_descs, offsets)
    var header_len = len(hdr)

    var tmp_path = path + String(".tmp")
    var fd = sys_open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(
            String("save_safetensors_host_buffers: failed to open for write: ")
            + tmp_path
        )

    var lenbuf = alloc[UInt8](8)
    var hl = header_len
    for i in range(8):
        lenbuf[i] = UInt8(hl & 0xFF)
        hl = hl >> 8
    try:
        _write_all(fd, BytePtr(unsafe_from_address=Int(lenbuf)), 8, 0)
    except e:
        lenbuf.free()
        _ = sys_close(fd)
        _ = sys_remove(tmp_path)
        raise e^
    lenbuf.free()

    var hbuf = alloc[UInt8](header_len)
    for i in range(header_len):
        hbuf[i] = hdr[i]
    try:
        _write_all(fd, BytePtr(unsafe_from_address=Int(hbuf)), header_len, 8)
    except e:
        hbuf.free()
        _ = sys_close(fd)
        _ = sys_remove(tmp_path)
        raise e^
    hbuf.free()

    var data_offset = 8 + header_len
    for i in range(len(descs)):
        try:
            _write_all(
                fd,
                BytePtr(unsafe_from_address=Int(descs[i].buffer[].unsafe_ptr())),
                descs[i].nbytes,
                data_offset + offsets[i],
            )
        except e:
            _ = sys_close(fd)
            _ = sys_remove(tmp_path)
            raise Error(
                String("save_safetensors_host_buffers: writing tensor '")
                + names[i] + String("': ") + String(e)
            )
    _ = sys_close(fd)
    if sys_rename(tmp_path, path) != 0:
        _ = sys_remove(tmp_path)
        raise Error(
            String("save_safetensors_host_buffers: atomic rename ")
            + tmp_path + String(" -> ") + path + String(" failed")
        )


def save_safetensors(
    names: List[String],
    tensors: List[ArcPointer[Tensor]],
    path: String,
    ctx: DeviceContext,
) raises:
    """Write `tensors` (named by `names`, same order) to `path` as a single-file
    safetensors. The inverse of SafeTensors.open in io/safetensors.mojo.

    Layout: [8-byte LE header_len][JSON header][concatenated tensor bytes].
    Each tensor's device buffer is copied D2H raw (byte-exact, no F32 cast) and
    pwritten at 8 + header_len + data_offsets[0]. Supports any STDtype the
    Tensor carries; F32 and BF16 are exercised by the round-trip smoke.
    """
    if len(names) != len(tensors):
        raise Error(
            String("save_safetensors: len(names)=")
            + String(len(names))
            + " != len(tensors)="
            + String(len(tensors))
        )
    if len(names) == 0:
        raise Error("save_safetensors: refusing to write an empty file")

    var offsets = _tensor_offsets(tensors)
    var hdr = _build_header(names, tensors, offsets)
    var header_len = len(hdr)

    # ATOMIC WRITE: write the whole file to "<path>.tmp" then rename(2) it onto the
    # final path. rename is atomic on the same filesystem, so a crash mid-write
    # leaves the PREVIOUS good checkpoint at `path` untouched (never a truncated,
    # corrupt one) — the tmp is the only casualty. Every saver in the fleet routes
    # through here, so this makes ALL checkpoint writes crash-safe. On any write
    # error we close and unlink the tmp so no partial file is left behind.
    var tmp_path = path + String(".tmp")
    # open(tmp_path, O_WRONLY|O_CREAT|O_TRUNC, 0644).
    var flags = O_WRONLY | O_CREAT | O_TRUNC
    var fd = sys_open(tmp_path, flags, Int32(0o644))
    if fd < 0:
        raise Error(String("save_safetensors: failed to open for write: ") + tmp_path)

    # 1) 8-byte little-endian header length at offset 0 (reader mmap.rs:175-178).
    var lenbuf = alloc[UInt8](8)
    var hl = header_len
    for i in range(8):
        lenbuf[i] = UInt8(hl & 0xFF)
        hl = hl >> 8
    try:
        _write_all(fd, BytePtr(unsafe_from_address=Int(lenbuf)), 8, 0)
    except e:
        lenbuf.free()
        _ = sys_close(fd)
        _ = sys_remove(tmp_path)
        raise Error(String("save_safetensors: writing header length: ") + String(e))
    lenbuf.free()

    # 2) header bytes at offset 8.
    var hbuf = alloc[UInt8](header_len)
    for i in range(header_len):
        hbuf[i] = hdr[i]
    try:
        _write_all(fd, BytePtr(unsafe_from_address=Int(hbuf)), header_len, 8)
    except e:
        hbuf.free()
        _ = sys_close(fd)
        _ = sys_remove(tmp_path)
        raise Error(String("save_safetensors: writing header bytes: ") + String(e))
    hbuf.free()

    # 3) tensor bytes. data_offset = 8 + header_len (reader mmap.rs:190); each
    #    tensor goes at data_offset + offsets[i]. Copy each device buffer D2H raw
    #    (same enqueue pattern as Tensor.to_host) and pwrite the raw bytes — this
    #    preserves the exact storage dtype byte-for-byte (BF16 stays BF16).
    var data_offset = 8 + header_len
    for i in range(len(tensors)):
        var t = tensors[i]
        var nbytes = offsets[i + 1] - offsets[i]
        if nbytes != t[].nbytes():
            _ = sys_close(fd)
            _ = sys_remove(tmp_path)
            raise Error("save_safetensors: tensor size changed mid-write")
        # D2H raw copy.
        var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
        ctx.enqueue_copy(dst_buf=host, src_buf=t[].buf)
        ctx.synchronize()
        var hp = host.unsafe_ptr()
        try:
            _write_all(
                fd,
                BytePtr(unsafe_from_address=Int(hp)),
                nbytes,
                data_offset + offsets[i],
            )
        except e:
            _ = sys_close(fd)
            _ = sys_remove(tmp_path)
            raise Error(
                String("save_safetensors: writing tensor '")
                + names[i]
                + "': "
                + String(e)
            )

    _ = sys_close(fd)

    # Durable file is now complete at tmp_path — atomically publish it onto the
    # final path. Only after this rename does `path` reflect the new checkpoint;
    # until then the previous checkpoint at `path` (if any) stays intact.
    if sys_rename(tmp_path, path) != 0:
        _ = sys_remove(tmp_path)
        raise Error(
            String("save_safetensors: atomic rename ") + tmp_path + " -> " + path
            + " failed (different filesystem? out of space?)"
        )
