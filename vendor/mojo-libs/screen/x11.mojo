# screen.x11 — X11 screen capture via the screen_shim.so C floor.
#
# Xlib's Display* is opaque and crashes when passed back across Mojo FFI, so
# the Xlib calls live in cshim/screen_shim.c. This module loads that .so, asks
# for an RGB24 buffer, and hands back an `image.Image` (or writes a PNG).
#
# Build the shim first:
#   gcc -shared -fPIC -O2 screen/cshim/screen_shim.c -o screen/cshim/screen_shim.so -lX11
# Run from the repo root (so the default relative .so path resolves), or set
# SCREEN_SHIM_PATH to an absolute path.

from std.ffi import OwnedDLHandle as DLHandle
from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin
from std.os import getenv

from image.buffer import Image
from image.png import encode_png

comptime PtrI32 = UnsafePointer[Int32, MutExternalOrigin]
comptime PtrU8 = UnsafePointer[UInt8, MutExternalOrigin]
comptime DEFAULT_SHIM = "screen/cshim/screen_shim.so"


@fieldwise_init
struct Size(ImplicitlyCopyable, Movable):
    var w: Int
    var h: Int


def shim_path() -> String:
    var p = getenv("SCREEN_SHIM_PATH")
    if p.byte_length() > 0:
        return p
    return String(DEFAULT_SHIM)


def x11_size() raises -> Size:
    """Size of the default-screen root window. Raises if no display."""
    var lib = DLHandle(shim_path())
    var f = lib.get_function[
        def(PtrI32, PtrI32) thin abi("C") -> Int32
    ]("screen_x11_size")
    var wh = alloc[Int32](2)
    wh[0] = 0
    wh[1] = 0
    var rc = f(wh, wh + 1)
    var w = Int(wh[0])
    var h = Int(wh[1])
    wh.free()
    # Keep the handle alive past the FFI call: OwnedDLHandle is destroyed at its
    # last use (ASAP), which dlclose's the .so out from under the function
    # pointer and segfaults the call. `_ = lib^` defers that to here.
    _ = lib^
    if rc != 0:
        raise Error("screen.x11: XOpenDisplay failed (no DISPLAY?)")
    return Size(w, h)


def grab_x11(x: Int, y: Int, w: Int, h: Int) raises -> Image:
    """Capture rect (x,y,w,h) of the root window into an RGB Image.

    w<=0 or h<=0 captures the whole screen. The rect is clamped to the screen.
    """
    var lib = DLHandle(shim_path())
    var fgrab = lib.get_function[
        def(Int32, Int32, Int32, Int32, PtrI32, PtrI32) thin abi("C") -> PtrU8
    ]("screen_x11_grab")
    var ffree = lib.get_function[def(PtrU8) thin abi("C") -> None]("screen_free")

    var wh = alloc[Int32](2)
    wh[0] = 0
    wh[1] = 0
    var buf = fgrab(Int32(x), Int32(y), Int32(w), Int32(h), wh, wh + 1)
    var ow = Int(wh[0])
    var oh = Int(wh[1])
    wh.free()

    # The shim sets ow/oh only on success and returns NULL on failure — gate on
    # the dimensions so we never dereference a NULL buffer.
    if ow <= 0 or oh <= 0:
        raise Error("screen.x11: grab failed (no display or empty rect)")

    var img = Image.new(ow, oh, 3)
    var n = ow * oh * 3
    for i in range(n):
        img.data[i] = buf[i]
    ffree(buf)
    # Hold the handle alive through both FFI calls above (grab + free); see
    # x11_size for the ASAP-destruction rationale.
    _ = lib^
    return img^


def capture_x11_png(path: String, x: Int, y: Int, w: Int, h: Int) raises:
    """Grab the screen (or a rect) and write it to a PNG file."""
    var img = grab_x11(x, y, w, h)
    encode_png(img, path)
