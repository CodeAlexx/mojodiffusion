# screen — Mojo screen snapshots

Versatile screen capture for Mojo, with three sources:

| backend | captures | output | status |
|---|---|---|---|
| **x11** | the X11 desktop (root window or a rect) | RGB → PNG | ✅ verified live (xwininfo + PIL) |
| **fb** | the Linux framebuffer `/dev/fb0` | RGB → PNG | decode unit-tested; live read needs `video` group |
| **vcsa** | a virtual-console text grid `/dev/vcsaN` | text | decode unit-tested; live read needs `tty` group |

PNG output reuses the repo's `image` library; nothing else is third-party.

## X11 (pixels) — the primary path

Xlib's `Display*` is opaque and segfaults when passed back across Mojo FFI, so
the handful of Xlib calls live in a tiny C floor (`cshim/screen_shim.c`) — the
repo's "C for the gaps, Mojo for the rest" pattern. Build it once:

```sh
gcc -shared -fPIC -O2 screen/cshim/screen_shim.c -o screen/cshim/screen_shim.so -lX11
```

Then, from the repo root (so the default relative `.so` path resolves — or set
`SCREEN_SHIM_PATH=/abs/path/screen_shim.so`):

```sh
mojo build -I . screen/cli.mojo -o screen/screen

./screen/screen size                                 # -> 4096x2160
./screen/screen x11 -o shot.png                      # full screen
./screen/screen x11 --rect 100 100 800 600 -o w.png  # a region
```

Library API:

```mojo
from screen import x11_size, grab_x11, capture_x11_png

var s = x11_size()                       # Size{w, h}
var img = grab_x11(0, 0, 0, 0)           # image.Image (RGB); 0,0,0,0 = full screen
capture_x11_png("shot.png", 0, 0, 0, 0)  # grab + write PNG
```

## Framebuffer (pixels, no X needed)

```sh
./screen/screen fb -o console.png        # reads /dev/fb0 (+ sysfs geometry)
```

Geometry is read from `/sys/class/graphics/fb0/{virtual_size,bits_per_pixel,stride}`.
32-bpp is decoded as BGRX, 24-bpp as BGR (the usual little-endian fbdev layout).
`/dev/fb0` is `root:video 0660`, so reading it needs the `video` group (or root);
otherwise `grab_fb()` fails loud. Under X the framebuffer may not reflect the live
desktop (X renders via the GPU) — `fb` is for console/headless capture.

```mojo
from screen import grab_fb, fb_decode, fb_info, FbInfo
var img = grab_fb()                      # /dev/fb0 -> Image
var img2 = fb_decode(raw_bytes, FbInfo(w, h, 32, stride))  # decode a buffer
```

## Console text (vcsa)

Snapshots the **text** on a Linux virtual console (the tty cell grid) — the text
complement to the framebuffer.

```sh
./screen/screen vcsa        # active console -> stdout
./screen/screen vcsa 1 -o tty1.txt
```

`/dev/vcsaN` format: `[rows, cols, cursor_col, cursor_row]` then `rows*cols`
cells of `[char, attr]`; the char byte is kept, attributes dropped, NUL → space.
It is `root:tty 0660` (needs the `tty` group) and only exists for the real Linux
console — not for X terminals, tmux, or SSH sessions.

```mojo
from screen import grab_vcsa, vcsa_decode, render_text
var ts = grab_vcsa(0)                    # TextScreen{rows, cols, cursor, lines}
print(render_text(ts))
```

## Tests

```sh
mojo run -I . screen/tests/screen_test.mojo
```

4 unit tests for the framebuffer (BGRX→RGB, stride padding) and vcsa (cell grid →
text, short-buffer error) decode paths. The X11 path is verified live (a captured
PNG matches `xwininfo` dimensions and decodes in PIL as correct-color desktop).

## FFI gotcha (1.0.0b1)

`OwnedDLHandle` is destroyed at its last use (ASAP), which `dlclose`s the `.so`
out from under any function pointers obtained from it — so the *second* FFI call
segfaults. Every function here ends with `_ = lib^` to hold the handle alive past
all its calls.
