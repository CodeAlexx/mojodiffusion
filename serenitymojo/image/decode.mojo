# image/decode.mojo — Mojo-owned image decode for trainer staging.
#
# This module is intentionally small and production-path oriented:
#   * no Python
#   * no Rust/EriDiffusion cache dependency
#   * decode happens inside our Mojo tooling, with local system image libraries
#     reached through FFI only for the compressed image formats.
#
# Supported input today:
#   * PNG via libpng's simplified png_image API
#   * JPEG via libturbojpeg
#
# Output is host RGB8, row-major HWC. Trainer stagers then crop/resize and
# normalize to model tensors themselves.

from std.collections import List
from std.ffi import external_call
from std.memory import UnsafePointer, alloc

from serenitymojo.io.ffi import (
    BytePtr,
    O_RDONLY,
    file_size,
    sys_close,
    sys_open,
    sys_pread,
)


@fieldwise_init
struct DecodedImage(Movable):
    var width: Int
    var height: Int
    var rgb: List[UInt8]


def _u32_at(p: UnsafePointer[UInt8, _], offset: Int) -> UInt32:
    return (p + offset).bitcast[UInt32]()[0]


def _png_message(image: UnsafePointer[UInt8, _]) -> String:
    var out = List[UInt8]()
    # png_image.message is char[64] at offset 36 in libpng 1.6.
    for i in range(64):
        var b = image[36 + i]
        if b == 0:
            break
        out.append(b)
    if len(out) == 0:
        return String("unknown libpng error")
    return String(unsafe_from_utf8=out)


def decode_png(path: String) raises -> DecodedImage:
    # libpng 1.6 png_image layout:
    #   opaque pointer @0, version @8, width @12, height @16, format @20, ...
    # The struct is 104 bytes on x86-64; allocate a little extra and zero it.
    comptime PNG_IMAGE_BYTES = 128
    comptime PNG_IMAGE_VERSION = UInt32(1)
    comptime PNG_FORMAT_RGBA = UInt32(3)

    var image = alloc[UInt8](PNG_IMAGE_BYTES)
    for i in range(PNG_IMAGE_BYTES):
        image[i] = 0
    var version_p = (image + 8).bitcast[UInt32]()
    version_p[0] = PNG_IMAGE_VERSION

    var path_len = path.byte_length()
    var cpath = alloc[UInt8](path_len + 1)
    var path_bytes = path.as_bytes()
    for i in range(path_len):
        cpath[i] = path_bytes[i]
    cpath[path_len] = 0
    var ok = external_call["png_image_begin_read_from_file", Int32](
        BytePtr(unsafe_from_address=Int(image)),
        BytePtr(unsafe_from_address=Int(cpath)),
    )
    cpath.free()
    if ok == 0:
        var msg = _png_message(image)
        image.free()
        raise Error(String("PNG decode header failed for ") + path + String(": ") + msg)

    var width = Int(_u32_at(image, 12))
    var height = Int(_u32_at(image, 16))
    if width <= 0 or height <= 0:
        _ = external_call["png_image_free", Int32](BytePtr(unsafe_from_address=Int(image)))
        image.free()
        raise Error(String("PNG decode invalid dimensions for ") + path)

    var format_p = (image + 20).bitcast[UInt32]()
    format_p[0] = PNG_FORMAT_RGBA
    var rgba_n = width * height * 4
    var rgba = alloc[UInt8](rgba_n)
    var nullp = BytePtr(unsafe_from_address=0)
    ok = external_call["png_image_finish_read", Int32](
        BytePtr(unsafe_from_address=Int(image)),
        nullp,
        BytePtr(unsafe_from_address=Int(rgba)),
        Int32(width * 4),
        nullp,
    )
    if ok == 0:
        var msg2 = _png_message(image)
        rgba.free()
        _ = external_call["png_image_free", Int32](BytePtr(unsafe_from_address=Int(image)))
        image.free()
        raise Error(String("PNG decode pixels failed for ") + path + String(": ") + msg2)

    _ = external_call["png_image_free", Int32](BytePtr(unsafe_from_address=Int(image)))
    image.free()

    var rgb = List[UInt8](capacity=width * height * 3)
    for i in range(width * height):
        # Source screenshots are expected to be opaque. For correctness with
        # alpha, composite over white before converting to RGB.
        var r = UInt32(rgba[i * 4 + 0])
        var g = UInt32(rgba[i * 4 + 1])
        var b = UInt32(rgba[i * 4 + 2])
        var a = UInt32(rgba[i * 4 + 3])
        if a != UInt32(255):
            r = (r * a + UInt32(255) * (UInt32(255) - a) + UInt32(127)) / UInt32(255)
            g = (g * a + UInt32(255) * (UInt32(255) - a) + UInt32(127)) / UInt32(255)
            b = (b * a + UInt32(255) * (UInt32(255) - a) + UInt32(127)) / UInt32(255)
        rgb.append(UInt8(r))
        rgb.append(UInt8(g))
        rgb.append(UInt8(b))
    rgba.free()
    return DecodedImage(width, height, rgb^)


def decode_jpeg(path: String) raises -> DecodedImage:
    comptime TJPF_RGB = Int32(0)
    comptime TJFLAG_NONE = Int32(0)

    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        raise Error(String("JPEG open failed: ") + path)
    var jpeg_size = file_size(fd)
    if jpeg_size <= 0:
        _ = sys_close(fd)
        raise Error(String("JPEG empty file: ") + path)
    var jpeg = alloc[UInt8](jpeg_size)
    var done = 0
    while done < jpeg_size:
        var n = sys_pread(
            fd,
            BytePtr(unsafe_from_address=Int(jpeg + done)),
            jpeg_size - done,
            done,
        )
        if n < 0:
            jpeg.free()
            _ = sys_close(fd)
            raise Error(String("JPEG read failed: ") + path)
        if n == 0:
            jpeg.free()
            _ = sys_close(fd)
            raise Error(String("JPEG short read: ") + path)
        done += n
    _ = sys_close(fd)

    var handle = external_call["tjInitDecompress", BytePtr]()
    if Int(handle) == 0:
        jpeg.free()
        raise Error("tjInitDecompress failed")

    var wp = alloc[Int32](1)
    var hp = alloc[Int32](1)
    var sp = alloc[Int32](1)
    var cp = alloc[Int32](1)
    wp[0] = 0
    hp[0] = 0
    sp[0] = 0
    cp[0] = 0

    var rc = external_call["tjDecompressHeader3", Int32](
        handle,
        BytePtr(unsafe_from_address=Int(jpeg)),
        jpeg_size,
        BytePtr(unsafe_from_address=Int(wp)),
        BytePtr(unsafe_from_address=Int(hp)),
        BytePtr(unsafe_from_address=Int(sp)),
        BytePtr(unsafe_from_address=Int(cp)),
    )
    if rc != 0:
        _ = external_call["tjDestroy", Int32](handle)
        wp.free()
        hp.free()
        sp.free()
        cp.free()
        jpeg.free()
        raise Error(String("JPEG header decode failed: ") + path)

    var width = Int(wp[0])
    var height = Int(hp[0])
    wp.free()
    hp.free()
    sp.free()
    cp.free()
    if width <= 0 or height <= 0:
        _ = external_call["tjDestroy", Int32](handle)
        jpeg.free()
        raise Error(String("JPEG invalid dimensions: ") + path)

    var rgb_bytes = width * height * 3
    var dst = alloc[UInt8](rgb_bytes)
    rc = external_call["tjDecompress2", Int32](
        handle,
        BytePtr(unsafe_from_address=Int(jpeg)),
        jpeg_size,
        BytePtr(unsafe_from_address=Int(dst)),
        Int32(width),
        Int32(width * 3),
        Int32(height),
        TJPF_RGB,
        TJFLAG_NONE,
    )
    _ = external_call["tjDestroy", Int32](handle)
    jpeg.free()
    if rc != 0:
        dst.free()
        raise Error(String("JPEG pixel decode failed: ") + path)

    var rgb = List[UInt8](capacity=rgb_bytes)
    for i in range(rgb_bytes):
        rgb.append(dst[i])
    dst.free()
    return DecodedImage(width, height, rgb^)


def decode_image(path: String) raises -> DecodedImage:
    if path.endswith(".png") or path.endswith(".PNG"):
        return decode_png(path)
    if path.endswith(".jpg") or path.endswith(".jpeg") or path.endswith(".JPG") or path.endswith(".JPEG"):
        return decode_jpeg(path)
    raise Error(String("unsupported image extension: ") + path)
