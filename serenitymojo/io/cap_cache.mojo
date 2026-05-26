# cap_cache.mojo - raw-bytes Tensor <-> disk cache for the Klein 9B caption
# split. Shared by klein9b_encode_smoke.mojo (writer) and
# klein9b_pipeline_multistep_smoke.mojo (reader) so the on-disk format has ONE
# definition and the two processes cannot drift.
#
# Why a separate encode process at all: the 24 GB card cannot hold Qwen3-8B
# (~16 GB) AND Klein 9B at once. In-process scope-freeing of the encoder is not
# a hard guarantee. Writing the caption embeddings to disk and letting the
# encode process EXIT is the hardest possible separation: process death frees
# every byte of encoder GPU memory before the DiT process ever starts.
#
# Bit-identical reload: we serialize the tensor's RAW device bytes (whatever the
# compute dtype is — BF16 for Klein), never an F32 upcast. So load == save byte
# for byte, and the DiT consumes exactly the dtype it consumed pre-split.
#
# File layout (little-endian, x86-64):
#   [0]    Int64  magic   = 0x4B4C4E43_41505631  ("KLNCAPV1")
#   [8]    Int64  dtype_tag (STDtype.tag)
#   [16]   Int64  rank
#   [24]   Int64  dims[0]
#   ...           dims[rank-1]
#   [24+8r] raw element bytes (numel * dtype.byte_size())
#
# All file I/O routes through io/ffi (sys_open/sys_pwrite/sys_pread) — never the
# stdlib builtin `open` or `external_call["write"]` (symbol-collision hazard
# documented in io/ffi.mojo).

from std.gpu.host import DeviceContext
from std.memory import alloc

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import (
    BytePtr,
    sys_open,
    sys_pwrite,
    sys_pread,
    sys_close,
    O_WRONLY,
    O_CREAT,
    O_TRUNC,
    O_RDONLY,
)


comptime _MAGIC = Int64(0x4B4C4E4341505631)
comptime _HDR_FIXED = 3  # magic, dtype_tag, rank


def _write_i64(fd: Int, value: Int64, offset: Int) raises:
    var tmp = alloc[Int64](1)
    tmp[0] = value
    var p = BytePtr(unsafe_from_address=Int(tmp))
    var w = sys_pwrite(fd, p, 8, offset)
    tmp.free()
    if w != 8:
        raise Error("cap_cache: short header write")


def _read_i64(fd: Int, offset: Int) raises -> Int64:
    var tmp = alloc[Int64](1)
    var p = BytePtr(unsafe_from_address=Int(tmp))
    var r = sys_pread(fd, p, 8, offset)
    var v = tmp[0]
    tmp.free()
    if r != 8:
        raise Error("cap_cache: short header read")
    return v


def save_tensor_bin(t: Tensor, path: String, ctx: DeviceContext) raises:
    """Serialize `t`'s raw device bytes + header to `path` via io/ffi.

    Writes the on-disk format documented at the top of this file. The element
    bytes are the device buffer verbatim (no dtype cast)."""
    var shape = t.shape()
    var rank = len(shape)
    var nbytes = t.nbytes()

    # Device -> host byte staging (raw uint8, no cast).
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=host, src_buf=t.buf)
    ctx.synchronize()

    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd < 0:
        raise Error(String("cap_cache.save: open failed for ") + path)

    var off = 0
    _write_i64(fd, _MAGIC, off)
    off += 8
    _write_i64(fd, Int64(t.dtype().tag), off)
    off += 8
    _write_i64(fd, Int64(rank), off)
    off += 8
    for i in range(rank):
        _write_i64(fd, Int64(shape[i]), off)
        off += 8

    var bp = BytePtr(unsafe_from_address=Int(host.unsafe_ptr()))
    var w = sys_pwrite(fd, bp, nbytes, off)
    if w != nbytes:
        _ = sys_close(fd)
        raise Error(
            String("cap_cache.save: short body write, wrote ")
            + String(w)
            + " of "
            + String(nbytes)
        )
    _ = sys_close(fd)


def load_tensor_bin(path: String, ctx: DeviceContext) raises -> Tensor:
    """Read a tensor written by `save_tensor_bin` back onto the device.

    Reconstructs shape + dtype from the header, reads the raw element bytes via
    `sys_pread`, and H2D-copies them so the result is bit-identical to the saved
    tensor. NEVER touches Qwen3Encoder — the denoise process has zero encoder
    code or weights."""
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        raise Error(String("cap_cache.load: open failed for ") + path)

    var magic = _read_i64(fd, 0)
    if magic != _MAGIC:
        _ = sys_close(fd)
        raise Error("cap_cache.load: bad magic (file not a cap cache)")
    var dtype_tag = Int(_read_i64(fd, 8))
    var rank = Int(_read_i64(fd, 16))
    if rank <= 0 or rank > 8:
        _ = sys_close(fd)
        raise Error("cap_cache.load: implausible rank")

    var off = 24
    var shape = List[Int]()
    var numel = 1
    for _ in range(rank):
        var d = Int(_read_i64(fd, off))
        off += 8
        shape.append(d)
        numel *= d

    var dtype = STDtype(dtype_tag)
    var nbytes = numel * dtype.byte_size()

    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var bp = BytePtr(unsafe_from_address=Int(host.unsafe_ptr()))
    var r = sys_pread(fd, bp, nbytes, off)
    _ = sys_close(fd)
    if r != nbytes:
        raise Error(
            String("cap_cache.load: short body read, read ")
            + String(r)
            + " of "
            + String(nbytes)
        )

    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, shape^, dtype)
