# screen.framebuffer — Linux framebuffer capture (/dev/fb0), pure Mojo.
#
# Geometry comes from sysfs (no ioctl needed): /sys/class/graphics/fb0/
#   virtual_size   -> "W,H"
#   bits_per_pixel -> e.g. "32"
#   stride         -> bytes per scanline (may exceed W*bpp/8)
# Pixels come straight from /dev/fb0. 32-bpp is assumed BGRX, 24-bpp BGR — the
# usual fbdev/efifb layout on little-endian x86 (sysfs does not expose channel
# offsets; use ioctl FBIOGET_VSCREENINFO if you need exotic layouts).
#
# NOTE: /dev/fb0 is typically root:video 0660 — reading it requires membership
# in the `video` group (or root). grab_fb() fails loud otherwise.

from image.buffer import Image
from image.png import encode_png


def _read_text(path: String) raises -> String:
    var f = open(path, "r")
    var s = f.read()
    f.close()
    return s


def _read_bytes(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var s = f.read()
    f.close()
    var bs = s.as_bytes()
    var out = List[UInt8]()
    for i in range(len(bs)):
        out.append(bs[i])
    return out^


def _atoi(s: String) -> Int:
    var bs = s.as_bytes()
    var v = 0
    for i in range(len(bs)):
        var c = Int(bs[i])
        if c < ord("0") or c > ord("9"):
            break
        v = v * 10 + (c - ord("0"))
    return v


@fieldwise_init
struct FbInfo(ImplicitlyCopyable, Movable):
    var width: Int
    var height: Int
    var bpp: Int
    var stride: Int


def fb_info(sysfs: String) raises -> FbInfo:
    """Read framebuffer geometry from a sysfs dir (e.g. /sys/class/graphics/fb0)."""
    var vs = _read_text(String(sysfs + "/virtual_size"))  # "W,H\n"
    var comma = vs.split(",")
    if len(comma) < 2:
        raise Error("fb: bad virtual_size: " + vs)
    var w = _atoi(String(comma[0]))
    var h = _atoi(String(comma[1]))
    var bpp = _atoi(_read_text(String(sysfs + "/bits_per_pixel")))
    var stride = _atoi(_read_text(String(sysfs + "/stride")))
    if stride <= 0:
        stride = w * (bpp // 8)
    return FbInfo(w, h, bpp, stride)


def fb_decode(raw: List[UInt8], info: FbInfo) raises -> Image:
    """Decode raw framebuffer bytes (BGRX/BGR) into an RGB Image."""
    if info.width <= 0 or info.height <= 0:
        raise Error("fb: bad geometry")
    var bytespp = info.bpp // 8
    if bytespp < 3:
        raise Error("fb: unsupported bpp " + String(info.bpp))
    var img = Image.new(info.width, info.height, 3)
    for y in range(info.height):
        var row = y * info.stride
        for x in range(info.width):
            var p = row + x * bytespp
            if p + 2 >= len(raw):
                continue
            var b = raw[p]
            var g = raw[p + 1]
            var r = raw[p + 2]
            var o = (y * info.width + x) * 3
            img.data[o] = r
            img.data[o + 1] = g
            img.data[o + 2] = b
    return img^


def grab_fb(device: String = "/dev/fb0", sysfs: String = "/sys/class/graphics/fb0") raises -> Image:
    """Capture the framebuffer into an RGB Image. Needs read access to `device`."""
    var info = fb_info(sysfs)
    var raw = _read_bytes(device)
    return fb_decode(raw, info)


def capture_fb_png(path: String, device: String = "/dev/fb0", sysfs: String = "/sys/class/graphics/fb0") raises:
    var img = grab_fb(device, sysfs)
    encode_png(img, path)
