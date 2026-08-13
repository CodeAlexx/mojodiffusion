"""Exact-byte smoke gate for GPU `[1,3,F,H,W]` to RGB24 conversion."""

from max.gpu.host import DeviceContext
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.image.png import save_rgb24_video, ValueRange
from serenitymojo.io.ffi import (
    BytePtr, O_RDONLY, file_size, sys_close, sys_open, sys_pread, sys_remove,
)


def main() raises:
    var ctx = DeviceContext()
    var values = List[Float32]()
    # R plane, frame 0 then frame 1.
    values.extend([Float32(-1.0), Float32(0.0)])
    values.extend([Float32(1.0), Float32(-1.0)])
    # G plane.
    values.extend([Float32(0.0), Float32(1.0)])
    values.extend([Float32(-1.0), Float32(0.0)])
    # B plane.
    values.extend([Float32(1.0), Float32(-1.0)])
    values.extend([Float32(0.0), Float32(1.0)])

    var video = Tensor.from_host(
        values, [1, 3, 2, 1, 2], STDtype.F32, ctx
    )
    var path = String("/tmp/serenitymojo_rgb24_video_smoke.rgb")
    save_rgb24_video(video, path, ctx, ValueRange.SIGNED)

    var expected = [
        UInt8(0), UInt8(128), UInt8(255),
        UInt8(128), UInt8(255), UInt8(0),
        UInt8(255), UInt8(0), UInt8(128),
        UInt8(0), UInt8(128), UInt8(255),
    ]
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error("rgb24_video_smoke: cannot open output")
    if file_size(fd) != len(expected):
        _ = sys_close(fd)
        raise Error("rgb24_video_smoke: wrong file size")
    var buf = alloc[UInt8](len(expected))
    var got = sys_pread(
        fd, BytePtr(unsafe_from_address=Int(buf)), len(expected), 0
    )
    _ = sys_close(fd)
    if got != len(expected):
        buf.free()
        raise Error("rgb24_video_smoke: short read")
    for i in range(len(expected)):
        if buf[i] != expected[i]:
            buf.free()
            raise Error(
                String("rgb24_video_smoke: byte mismatch at ") + String(i)
            )
    buf.free()
    _ = sys_remove(path)
    print("RGB24_VIDEO_SMOKE: PASS")
