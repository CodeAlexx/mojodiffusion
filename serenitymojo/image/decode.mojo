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
from std.ffi import OwnedDLHandle as DLHandle
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


# Mojo 1.0 rejects a COMPILE-TIME zero address ("Pointer is non-nullable; use
# Optional[Pointer]"), but Optional cannot cross an FFI boundary -- the C ABI
# here genuinely requires a NULL argument. Computing the address at runtime
# yields the same null pointer without tripping the comptime constraint.
@no_inline
def _null_addr() -> Int:
    var z = 0
    return z

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


def decode_png(path: String, drop_alpha: Bool = False) raises -> DecodedImage:
    # libpng 1.6 png_image layout:
    #   opaque pointer @0, version @8, width @12, height @16, format @20, ...
    # The struct is 104 bytes on x86-64; allocate a little extra and zero it.
    comptime PNG_IMAGE_BYTES = 128
    comptime PNG_IMAGE_VERSION = UInt32(1)
    comptime PNG_FORMAT_RGBA = UInt32(3)

    # Load libpng at runtime via a DLHandle and call through function pointers
    # (no named external symbol, so JIT `mojo run` resolves and libpng need not be
    # linked). png16 lives in the pixi env lib dir (on the loader path).
    var lib = DLHandle("libpng16.so.16")

    var image = alloc[UInt8](PNG_IMAGE_BYTES)
    for i in range(PNG_IMAGE_BYTES):
        image[i] = 0
    var version_p = (image + 8).bitcast[UInt32]()
    version_p[0] = PNG_IMAGE_VERSION
    var image_p = BytePtr(unsafe_from_address=Int(image))

    var path_len = path.byte_length()
    var cpath = alloc[UInt8](path_len + 1)
    var path_bytes = path.as_bytes()
    for i in range(path_len):
        cpath[i] = path_bytes[i]
    cpath[path_len] = 0
    var ok = lib.call["png_image_begin_read_from_file", Int32](image_p, BytePtr(unsafe_from_address=Int(cpath)))
    cpath.free()
    if ok == 0:
        var msg = _png_message(image)
        image.free()
        _ = lib^
        raise Error(String("PNG decode header failed for ") + path + String(": ") + msg)

    var width = Int(_u32_at(image, 12))
    var height = Int(_u32_at(image, 16))
    if width <= 0 or height <= 0:
        lib.call["png_image_free", NoneType](image_p)
        image.free()
        _ = lib^
        raise Error(String("PNG decode invalid dimensions for ") + path)

    var format_p = (image + 20).bitcast[UInt32]()
    format_p[0] = PNG_FORMAT_RGBA
    var rgba_n = width * height * 4
    var rgba = alloc[UInt8](rgba_n)
    var nullp = BytePtr(unsafe_from_address=_null_addr())
    ok = lib.call["png_image_finish_read", Int32](
        image_p,
        nullp,
        BytePtr(unsafe_from_address=Int(rgba)),
        Int32(width * 4),
        nullp,
    )
    if ok == 0:
        var msg2 = _png_message(image)
        rgba.free()
        lib.call["png_image_free", NoneType](image_p)
        image.free()
        _ = lib^
        raise Error(String("PNG decode pixels failed for ") + path + String(": ") + msg2)

    lib.call["png_image_free", NoneType](image_p)
    image.free()
    _ = lib^

    var rgb = List[UInt8](capacity=width * height * 3)
    for i in range(width * height):
        # Source screenshots are expected to be opaque. For correctness with
        # alpha, composite over white before converting to RGB.
        var r = UInt32(rgba[i * 4 + 0])
        var g = UInt32(rgba[i * 4 + 1])
        var b = UInt32(rgba[i * 4 + 2])
        var a = UInt32(rgba[i * 4 + 3])
        if (not drop_alpha) and a != UInt32(255):
            r = (r * a + UInt32(255) * (UInt32(255) - a) + UInt32(127)) / UInt32(255)
            g = (g * a + UInt32(255) * (UInt32(255) - a) + UInt32(127)) / UInt32(255)
            b = (b * a + UInt32(255) * (UInt32(255) - a) + UInt32(127)) / UInt32(255)
        rgb.append(UInt8(r))
        rgb.append(UInt8(g))
        rgb.append(UInt8(b))
    rgba.free()
    return DecodedImage(width, height, rgb^)


def decode_jpeg(path: String, drop_alpha: Bool = False) raises -> DecodedImage:
    # JPEG has no alpha channel; drop_alpha is accepted for a uniform signature.
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

    # Load libturbojpeg at runtime via a DLHandle + function pointers (see
    # decode_png rationale). libturbojpeg.so.0 lives in the pixi env lib dir.
    var lib = DLHandle("libturbojpeg.so.0")

    var handle = lib.call["tjInitDecompress", BytePtr]()
    if Int(handle) == 0:
        jpeg.free()
        _ = lib^
        raise Error("tjInitDecompress failed")

    var jpeg_p = BytePtr(unsafe_from_address=Int(jpeg))
    var wp = alloc[Int32](1)
    var hp = alloc[Int32](1)
    var sp = alloc[Int32](1)
    var cp = alloc[Int32](1)
    wp[0] = 0
    hp[0] = 0
    sp[0] = 0
    cp[0] = 0

    var rc = lib.call["tjDecompressHeader3", Int32](
        handle,
        jpeg_p,
        jpeg_size,
        BytePtr(unsafe_from_address=Int(wp)),
        BytePtr(unsafe_from_address=Int(hp)),
        BytePtr(unsafe_from_address=Int(sp)),
        BytePtr(unsafe_from_address=Int(cp)),
    )
    if rc != 0:
        _ = lib.call["tjDestroy", Int32](handle)
        wp.free()
        hp.free()
        sp.free()
        cp.free()
        jpeg.free()
        _ = lib^
        raise Error(String("JPEG header decode failed: ") + path)

    var width = Int(wp[0])
    var height = Int(hp[0])
    wp.free()
    hp.free()
    sp.free()
    cp.free()
    if width <= 0 or height <= 0:
        _ = lib.call["tjDestroy", Int32](handle)
        jpeg.free()
        _ = lib^
        raise Error(String("JPEG invalid dimensions: ") + path)

    var rgb_bytes = width * height * 3
    var dst = alloc[UInt8](rgb_bytes)
    rc = lib.call["tjDecompress2", Int32](
        handle,
        jpeg_p,
        jpeg_size,
        BytePtr(unsafe_from_address=Int(dst)),
        Int32(width),
        Int32(width * 3),
        Int32(height),
        TJPF_RGB,
        TJFLAG_NONE,
    )
    _ = lib.call["tjDestroy", Int32](handle)
    jpeg.free()
    _ = lib^
    if rc != 0:
        dst.free()
        raise Error(String("JPEG pixel decode failed: ") + path)

    var rgb = List[UInt8](capacity=rgb_bytes)
    for i in range(rgb_bytes):
        rgb.append(dst[i])
    dst.free()
    return DecodedImage(width, height, rgb^)


# WEBP via system libwebp (libwebp.so.7). libwebp is NOT a pixi dependency and is
# NOT linked at build time; we load it at runtime with a DLHandle and call through
# function pointers (no named external symbol, so JIT `mojo run` resolves fine and
# libwebp stays an OPTIONAL runtime dependency). If libwebp.so.7 is absent the
# DLHandle raises here — callers with .webp inputs must pre-convert to PNG/JPEG.
# Output is host RGB8 HWC; any alpha is composited over white (mirroring decode_png).
def decode_webp(path: String, drop_alpha: Bool = False) raises -> DecodedImage:
    # drop_alpha=True matches PIL Image.convert("RGB") (takes the raw RGB channels,
    # ignoring alpha). Default False composites over white (legacy screenshot path).
    comptime PtrI32 = UnsafePointer[Int32, MutExternalOrigin]

    var lib = DLHandle("libwebp.so.7")
    # int WebPGetInfo(const uint8_t* data, size_t size, int* w, int* h)
    # uint8_t* WebPDecodeRGBA(const uint8_t* data, size_t size, int* w, int* h)
    # void WebPFree(void* ptr)

    # Slurp the whole file (mirrors decode_jpeg's read loop).
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        raise Error(String("WEBP open failed: ") + path)
    var dsize = file_size(fd)
    if dsize <= 0:
        _ = sys_close(fd)
        raise Error(String("WEBP empty file: ") + path)
    var data = alloc[UInt8](dsize)
    var done = 0
    while done < dsize:
        var n = sys_pread(
            fd,
            BytePtr(unsafe_from_address=Int(data + done)),
            dsize - done,
            done,
        )
        if n <= 0:
            data.free()
            _ = sys_close(fd)
            raise Error(String("WEBP read failed: ") + path)
        done += n
    _ = sys_close(fd)

    var data_p = BytePtr(unsafe_from_address=Int(data))
    var wp = alloc[Int32](1)
    var hp = alloc[Int32](1)
    wp[0] = 0
    hp[0] = 0
    var info = lib.call["WebPGetInfo", Int32](data_p, dsize, wp, hp)
    if info == 0:
        wp.free()
        hp.free()
        data.free()
        _ = lib^
        raise Error(String("WEBP header decode failed: ") + path)
    var width = Int(wp[0])
    var height = Int(hp[0])

    # WebPDecodeRGBA returns a libwebp-malloc'd RGBA buffer (freed via WebPFree).
    var rgba_ptr = lib.call["WebPDecodeRGBA", BytePtr](data_p, dsize, wp, hp)
    data.free()
    wp.free()
    hp.free()
    if Int(rgba_ptr) == 0:
        _ = lib^
        raise Error(String("WEBP pixel decode failed: ") + path)

    var rgb = List[UInt8](capacity=width * height * 3)
    for i in range(width * height):
        var r = UInt32(rgba_ptr[i * 4 + 0])
        var g = UInt32(rgba_ptr[i * 4 + 1])
        var b = UInt32(rgba_ptr[i * 4 + 2])
        var a = UInt32(rgba_ptr[i * 4 + 3])
        if (not drop_alpha) and a != UInt32(255):
            r = (r * a + UInt32(255) * (UInt32(255) - a) + UInt32(127)) / UInt32(255)
            g = (g * a + UInt32(255) * (UInt32(255) - a) + UInt32(127)) / UInt32(255)
            b = (b * a + UInt32(255) * (UInt32(255) - a) + UInt32(127)) / UInt32(255)
        rgb.append(UInt8(r))
        rgb.append(UInt8(g))
        rgb.append(UInt8(b))
    lib.call["WebPFree", NoneType](rgba_ptr)
    # Hold the DLHandle alive past the last FFI use (OwnedDLHandle dlcloses ASAP).
    _ = lib^
    return DecodedImage(width, height, rgb^)


def decode_image(path: String, drop_alpha: Bool = False) raises -> DecodedImage:
    # drop_alpha=True matches PIL Image.convert("RGB") (ignore alpha); default
    # False composites over white (legacy screenshot path).
    # Select by file signature first. Real model/reference bundles sometimes
    # contain JPEG payloads with a .png filename; extension-only dispatch makes
    # those valid inputs fail before preprocessing.
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        raise Error(String("image open failed: ") + path)
    var signature = alloc[UInt8](12)
    var read_count = sys_pread(
        fd, BytePtr(unsafe_from_address=Int(signature)), 12, 0,
    )
    _ = sys_close(fd)
    if (
        read_count >= 8
        and signature[0] == UInt8(0x89)
        and signature[1] == UInt8(0x50)
        and signature[2] == UInt8(0x4E)
        and signature[3] == UInt8(0x47)
        and signature[4] == UInt8(0x0D)
        and signature[5] == UInt8(0x0A)
        and signature[6] == UInt8(0x1A)
        and signature[7] == UInt8(0x0A)
    ):
        signature.free()
        return decode_png(path, drop_alpha)
    if (
        read_count >= 3
        and signature[0] == UInt8(0xFF)
        and signature[1] == UInt8(0xD8)
        and signature[2] == UInt8(0xFF)
    ):
        signature.free()
        return decode_jpeg(path, drop_alpha)
    if (
        read_count >= 12
        and signature[0] == UInt8(ord("R"))
        and signature[1] == UInt8(ord("I"))
        and signature[2] == UInt8(ord("F"))
        and signature[3] == UInt8(ord("F"))
        and signature[8] == UInt8(ord("W"))
        and signature[9] == UInt8(ord("E"))
        and signature[10] == UInt8(ord("B"))
        and signature[11] == UInt8(ord("P"))
    ):
        signature.free()
        return decode_webp(path, drop_alpha)
    signature.free()
    if path.endswith(".png") or path.endswith(".PNG"):
        return decode_png(path, drop_alpha)
    if path.endswith(".jpg") or path.endswith(".jpeg") or path.endswith(".JPG") or path.endswith(".JPEG"):
        return decode_jpeg(path, drop_alpha)
    if path.endswith(".webp") or path.endswith(".WEBP"):
        return decode_webp(path, drop_alpha)
    raise Error(String("unsupported image extension: ") + path)
