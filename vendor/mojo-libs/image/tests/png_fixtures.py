#!/usr/bin/env python3
# Generate PNG fixtures with Pillow and dump expected pixels as flat text files.
import os
from PIL import Image

OUT = "/tmp/png_fix"
os.makedirs(OUT, exist_ok=True)


def dump(name, img, mode):
    """Save PIL pixels (converted to `mode`) as a flat int-per-line file."""
    conv = img.convert(mode)
    data = list(conv.getdata())
    flat = []
    for px in data:
        if isinstance(px, int):
            flat.append(px)
        else:
            flat.extend(px)
    with open(os.path.join(OUT, name + ".pix"), "w") as f:
        f.write("%d %d %d\n" % (conv.width, conv.height, len(flat) // (conv.width * conv.height)))
        for v in flat:
            f.write("%d\n" % v)


# 1. 8-bit RGB gradient ------------------------------------------------------
w, h = 37, 23
rgb = Image.new("RGB", (w, h))
for y in range(h):
    for x in range(w):
        rgb.putpixel((x, y), ((x * 7) % 256, (y * 11) % 256, (x * y) % 256))
rgb.save(os.path.join(OUT, "rgb.png"))
dump("rgb", rgb, "RGB")

# 2. RGBA with varied alpha --------------------------------------------------
rgba = Image.new("RGBA", (w, h))
for y in range(h):
    for x in range(w):
        rgba.putpixel((x, y), ((x * 5) % 256, (y * 9) % 256, (x + y) % 256, (x * 13) % 256))
rgba.save(os.path.join(OUT, "rgba.png"))
dump("rgba", rgba, "RGBA")

# 3. 8-bit grayscale ---------------------------------------------------------
gray = Image.new("L", (w, h))
for y in range(h):
    for x in range(w):
        gray.putpixel((x, y), (x * 3 + y * 5) % 256)
gray.save(os.path.join(OUT, "gray.png"))
dump("gray", gray, "L")

# 4. Palette (P-mode) image with transparency -------------------------------
pal = Image.new("P", (w, h))
palette = []
for i in range(256):
    palette.extend([(i * 2) % 256, (i * 3) % 256, (i * 5) % 256])
pal.putpalette(palette)
for y in range(h):
    for x in range(w):
        pal.putpixel((x, y), (x + y * 2) % 64)
# add a tRNS so some indices are transparent
trns = bytes([255] * 256)
trns = bytearray(trns)
for i in range(0, 64, 7):
    trns[i] = i * 2 % 256
pal.save(os.path.join(OUT, "palette.png"), transparency=bytes(trns))
# our decoder emits RGBA for palette+tRNS -> compare against RGBA. Reload from
# the saved file so the tRNS-driven alpha is actually applied by convert().
pal_reloaded = Image.open(os.path.join(OUT, "palette.png"))
dump("palette", pal_reloaded, "RGBA")

# 5. Interlaced PNG (Adam7) — RGB content, written by hand (PIL save() ignores
#    the interlace kwarg in this Pillow build, so emit a real Adam7 stream). ---
import zlib, struct


def _png_chunk(ctype, data):
    out = struct.pack(">I", len(data)) + ctype + data
    crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
    return out + struct.pack(">I", crc)


inter = Image.new("RGB", (w, h))
for y in range(h):
    for x in range(w):
        inter.putpixel((x, y), ((x * 6) % 256, (y * 4) % 256, ((x + y) * 3) % 256))

# Adam7 passes
ox = [0, 4, 0, 2, 0, 1, 0]
oy = [0, 0, 4, 0, 2, 0, 1]
sx = [8, 8, 4, 4, 2, 2, 1]
sy = [8, 8, 8, 4, 4, 2, 2]
raw = bytearray()
px = inter.load()
for p in range(7):
    if ox[p] >= w or oy[p] >= h:
        continue
    pw = (w - 1 - ox[p]) // sx[p] + 1
    ph = (h - 1 - oy[p]) // sy[p] + 1
    for yy in range(ph):
        raw.append(0)  # filter None
        gy = oy[p] + yy * sy[p]
        for xx in range(pw):
            gx = ox[p] + xx * sx[p]
            r, g, b = px[gx, gy]
            raw.extend([r, g, b])
ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 1)  # color type 2, interlace 1
sig = bytes([137, 80, 78, 71, 13, 10, 26, 10])
out = sig + _png_chunk(b"IHDR", ihdr)
out += _png_chunk(b"IDAT", zlib.compress(bytes(raw)))
out += _png_chunk(b"IEND", b"")
with open(os.path.join(OUT, "interlaced.png"), "wb") as f:
    f.write(out)
dump("interlaced", inter, "RGB")

# 5b. sub-byte grayscale fixtures (depth 1/2/4) ------------------------------
def _gray_lowdepth(name, depth):
    maxv = (1 << depth) - 1
    img = Image.new("L", (w, h))
    for y in range(h):
        for x in range(w):
            img.putpixel((x, y), ((x + y) % (maxv + 1)) * (255 // maxv))
    # pack MSB-first
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter None
        acc = 0
        nbits = 0
        for x in range(w):
            v = ((x + y) % (maxv + 1))
            acc = (acc << depth) | v
            nbits += depth
            while nbits >= 8:
                nbits -= 8
                raw.append((acc >> nbits) & 0xFF)
        if nbits > 0:
            raw.append((acc << (8 - nbits)) & 0xFF)
    ihdr = struct.pack(">IIBBBBB", w, h, depth, 0, 0, 0, 0)
    out = sig + _png_chunk(b"IHDR", ihdr)
    out += _png_chunk(b"IDAT", zlib.compress(bytes(raw)))
    out += _png_chunk(b"IEND", b"")
    with open(os.path.join(OUT, name + ".png"), "wb") as f:
        f.write(out)
    # expected: scaled to 8-bit (value * 255/maxv == value * (255//maxv) since exact)
    g8 = Image.new("L", (w, h))
    scale = 255 // maxv
    for y in range(h):
        for x in range(w):
            g8.putpixel((x, y), ((x + y) % (maxv + 1)) * scale)
    dump(name, g8, "L")


_gray_lowdepth("gray1", 1)
_gray_lowdepth("gray2", 2)
_gray_lowdepth("gray4", 4)

# 6. 16-bit (I;16) PNG — compare against 8-bit downscale (high byte) ---------
i16 = Image.new("I;16", (w, h))
for y in range(h):
    for x in range(w):
        i16.putpixel((x, y), ((x * 1500 + y * 700) % 65536))
i16.save(os.path.join(OUT, "i16.png"))
# Our decoder downscales 16->8 by taking the HIGH byte (value >> 8).
# Build the expected high-byte 8-bit grayscale.
g8 = Image.new("L", (w, h))
for y in range(h):
    for x in range(w):
        v = i16.getpixel((x, y))
        g8.putpixel((x, y), (v >> 8) & 0xFF)
dump("i16", g8, "L")

print("FIXTURES_OK")
for fn in sorted(os.listdir(OUT)):
    print(" ", fn)
