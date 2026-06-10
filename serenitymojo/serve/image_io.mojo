# serenitymojo.serve.image_io — init-image decode helpers (plan P7, img2img).
#
# Pure-Mojo decode of a user-supplied init image via MOJO-libs image
# (png / jpeg / webp), shared by:
#   * serve/zimage_backend.mojo — decode + resize + [-1,1] NCHW for the
#     Z-Image VAE encoder (the real img2img path), and
#   * the SerenityUI params column — decode + resize + RGBA8 for the
#     init-image thumbnail texture.
#
# Format detection is MAGIC-BYTE based (extension only as a tiebreaker is
# unnecessary): PNG \x89PNG, JPEG \xFF\xD8, WebP RIFF....WEBP.

from std.io.file import open

from image.buffer import Image
from image.jpeg import decode_jpeg_bytes
from image.png import decode_png_bytes
from image.webp import decode_webp_bytes


def _read_file_bytes(path: String) raises -> List[UInt8]:
    var f = open(path, String("r"))
    var d = f.read_bytes()
    f.close()
    var data = List[UInt8](capacity=len(d))
    for i in range(len(d)):
        data.append(d[i])
    return data^


def decode_image_any(path: String) raises -> Image:
    """Decode a PNG / JPEG / WebP file (magic-byte sniffed). Raises with a
    clear message for missing files and unsupported formats."""
    var data: List[UInt8]
    try:
        data = _read_file_bytes(path)
    except:
        raise Error(String("init image not readable: ") + path)
    if len(data) < 12:
        raise Error(String("init image too small / empty: ") + path)
    if data[0] == 0x89 and data[1] == 0x50 and data[2] == 0x4E and data[3] == 0x47:
        return decode_png_bytes(data)
    if data[0] == 0xFF and data[1] == 0xD8:
        return decode_jpeg_bytes(data)
    if (
        data[0] == 0x52 and data[1] == 0x49 and data[2] == 0x46 and data[3] == 0x46
        and data[8] == 0x57 and data[9] == 0x45 and data[10] == 0x42 and data[11] == 0x50
    ):
        return decode_webp_bytes(data)
    raise Error(
        String("init image format not supported (need png/jpeg/webp): ") + path
    )


def image_rgb_at(img: Image, x: Int, y: Int, mut rgb: List[Int]) raises:
    """8-bit RGB triple at (x,y) for gray(1) / RGB(3) / RGBA(4) images,
    written into rgb[0..2] (caller provides a 3-slot list)."""
    if img.channels == 1:
        var g = Int(img.get(x, y, 0))
        rgb[0] = g
        rgb[1] = g
        rgb[2] = g
        return
    rgb[0] = Int(img.get(x, y, 0))
    rgb[1] = Int(img.get(x, y, 1))
    rgb[2] = Int(img.get(x, y, 2))


def image_to_rgba_bytes(img: Image) raises -> List[UInt8]:
    """Row-major RGBA8 bytes (alpha forced opaque) — the Backend
    make_texture_rgba upload format."""
    var out = List[UInt8](capacity=img.width * img.height * 4)
    var px: List[Int] = [0, 0, 0]
    for y in range(img.height):
        for x in range(img.width):
            image_rgb_at(img, x, y, px)
            out.append(UInt8(px[0]))
            out.append(UInt8(px[1]))
            out.append(UInt8(px[2]))
            out.append(UInt8(255))
    return out^


def image_to_signed_nchw(img: Image) raises -> List[Float32]:
    """[1,3,H,W] flat host floats in [-1,1] (v = px/127.5 - 1) — the VAE
    encoder input convention (staged images are [-1,1] RGB; see
    pipeline/zimage_prepare.mojo stage contract)."""
    var plane = img.width * img.height
    var out = List[Float32](capacity=3 * plane)
    for _ in range(3 * plane):
        out.append(Float32(0.0))
    var px: List[Int] = [0, 0, 0]
    for y in range(img.height):
        for x in range(img.width):
            image_rgb_at(img, x, y, px)
            var off = y * img.width + x
            out[0 * plane + off] = Float32(px[0]) / 127.5 - 1.0
            out[1 * plane + off] = Float32(px[1]) / 127.5 - 1.0
            out[2 * plane + off] = Float32(px[2]) / 127.5 - 1.0
    return out^
