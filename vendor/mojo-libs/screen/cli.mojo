# screen.cli — runnable entrypoint for screen snapshots.
#
#   mojo run -I . screen/cli.mojo size
#   mojo run -I . screen/cli.mojo x11  [-o out.png] [--rect X Y W H]
#   mojo run -I . screen/cli.mojo fb   [-o out.png]
#   mojo run -I . screen/cli.mojo vcsa [N] [-o out.txt]
#
# Build the X11 shim first and run from the repo root (default .so path is
# relative), or set SCREEN_SHIM_PATH=/abs/path/screen_shim.so:
#   gcc -shared -fPIC -O2 screen/cshim/screen_shim.c -o screen/cshim/screen_shim.so -lX11

from std.sys import argv

from screen.x11 import x11_size, capture_x11_png
from screen.framebuffer import capture_fb_png
from screen.vcsa import grab_vcsa, render_text


def _atoi(s: String) -> Int:
    var bs = s.as_bytes()
    var i = 0
    var neg = False
    if len(bs) > 0 and Int(bs[0]) == ord("-"):
        neg = True
        i = 1
    var v = 0
    while i < len(bs):
        var c = Int(bs[i])
        if c < ord("0") or c > ord("9"):
            break
        v = v * 10 + (c - ord("0"))
        i += 1
    return -v if neg else v


def _usage():
    print("usage:")
    print("  screen size")
    print("  screen x11  [-o out.png] [--rect X Y W H]")
    print("  screen fb   [-o out.png]")
    print("  screen vcsa [N] [-o out.txt]")


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(len(raw)):
        args.append(String(raw[i]))

    if len(args) < 2:
        _usage()
        return

    var cmd = args[1]

    if cmd == "size":
        var s = x11_size()
        print(String(s.w) + "x" + String(s.h))
        return

    if cmd == "x11":
        var out = String("screen.png")
        var x = 0
        var y = 0
        var w = 0
        var h = 0
        var i = 2
        while i < len(args):
            var a = args[i]
            if (a == "-o" or a == "--out") and i + 1 < len(args):
                i += 1
                out = args[i]
            elif a == "--rect" and i + 4 < len(args):
                x = _atoi(args[i + 1])
                y = _atoi(args[i + 2])
                w = _atoi(args[i + 3])
                h = _atoi(args[i + 4])
                i += 4
            i += 1
        capture_x11_png(out, x, y, w, h)
        print("wrote " + out)
        return

    if cmd == "fb":
        var out = String("screen.png")
        var i = 2
        while i < len(args):
            if (args[i] == "-o" or args[i] == "--out") and i + 1 < len(args):
                i += 1
                out = args[i]
            i += 1
        capture_fb_png(out)
        print("wrote " + out)
        return

    if cmd == "vcsa":
        var n = 0
        var out = String("")
        var i = 2
        while i < len(args):
            var a = args[i]
            if (a == "-o" or a == "--out") and i + 1 < len(args):
                i += 1
                out = args[i]
            elif _atoi(a) > 0 or a == "0":
                n = _atoi(a)
            i += 1
        var ts = grab_vcsa(n)
        var text = render_text(ts)
        if out.byte_length() == 0:
            print(text)
        else:
            with open(out, "w") as f:
                f.write(text)
            print("wrote " + out + " (" + String(ts.rows) + "x" + String(ts.cols) + ")")
        return

    _usage()
