# cap_cache_header_smoke.mojo
#
# Header-only Klein cap-cache gate. This writes sparse files in /tmp, validates
# accepted BF16 [512,12288] and [1,512,12288] headers, and proves bad dtype/shape
# fail before any DeviceContext is created.
#
# Run:
#   pixi run mojo run -I . serenitymojo/io/cap_cache_header_smoke.mojo

from std.memory import alloc

from serenitymojo.io.cap_cache import validate_klein_cap_cache_header
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import (
    BytePtr,
    sys_open,
    sys_pwrite,
    sys_close,
    O_WRONLY,
    O_CREAT,
    O_TRUNC,
)


comptime _MAGIC = Int64(0x4B4C4E4341505631)
comptime _KLEIN_JOINT_DIM = 12288


def _write_i64(fd: Int, value: Int64, offset: Int) raises:
    var tmp = alloc[Int64](1)
    tmp[0] = value
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(tmp)), 8, offset)
    tmp.free()
    if wrote != 8:
        raise Error("cap cache header smoke: short i64 write")


def _sparse_write_last_byte(fd: Int, offset: Int) raises:
    var tmp = alloc[UInt8](1)
    tmp[0] = 0
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(tmp)), 1, offset)
    tmp.free()
    if wrote != 1:
        raise Error("cap cache header smoke: short sparse body write")


def _write_cap_header(
    path: String, dtype_tag: Int, rank: Int, d0: Int, d1: Int, d2: Int
) raises:
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("cap cache header smoke: cannot create ") + path)

    var off = 0
    _write_i64(fd, _MAGIC, off)
    off += 8
    _write_i64(fd, Int64(dtype_tag), off)
    off += 8
    _write_i64(fd, Int64(rank), off)
    off += 8
    _write_i64(fd, Int64(d0), off)
    off += 8
    _write_i64(fd, Int64(d1), off)
    off += 8
    var numel = d0 * d1
    if rank == 3:
        _write_i64(fd, Int64(d2), off)
        off += 8
        numel *= d2

    var expected_size = off + numel * STDtype.BF16.byte_size()
    _sparse_write_last_byte(fd, expected_size - 1)
    _ = sys_close(fd)


def _expect_raises(path: String, label: String) raises:
    var raised = False
    try:
        validate_klein_cap_cache_header(path, _KLEIN_JOINT_DIM)
    except e:
        raised = True
        print("  rejected as expected:", label, String(e))
    if not raised:
        raise Error(String("cap cache header smoke: expected rejection for ") + label)


def main() raises:
    print("=== Klein cap-cache header smoke ===")

    var valid_3d = String("/tmp/klein_cap_valid_3d.bin")
    _write_cap_header(valid_3d, STDtype.BF16.tag, 3, 1, 512, _KLEIN_JOINT_DIM)
    validate_klein_cap_cache_header(valid_3d, _KLEIN_JOINT_DIM)
    print("  accepted valid BF16 [1,512,12288]")

    var valid_2d = String("/tmp/klein_cap_valid_2d.bin")
    _write_cap_header(valid_2d, STDtype.BF16.tag, 2, 512, _KLEIN_JOINT_DIM, 0)
    validate_klein_cap_cache_header(valid_2d, _KLEIN_JOINT_DIM)
    print("  accepted valid BF16 [512,12288]")

    var bad_dtype = String("/tmp/klein_cap_bad_dtype.bin")
    _write_cap_header(bad_dtype, STDtype.F32.tag, 3, 1, 512, _KLEIN_JOINT_DIM)
    _expect_raises(bad_dtype, String("F32 dtype boundary"))

    var bad_shape = String("/tmp/klein_cap_bad_shape.bin")
    _write_cap_header(bad_shape, STDtype.BF16.tag, 3, 1, 512, 7680)
    _expect_raises(bad_shape, String("4B text width for 9B sampler"))

    print("cap_cache_header_smoke PASS")
