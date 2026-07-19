#!/usr/bin/env python3
# Generate lossless WebP fixtures + expected RGBA pixel dumps using PIL (oracle).
import os
from PIL import Image
import random

BASE = "/tmp/webp_fix"
os.makedirs(BASE, exist_ok=True)
random.seed(1234)

def dump(im, name):
    p = os.path.join(BASE, name + ".webp")
    im.save(p, "WEBP", lossless=True, method=6)
    # Read back via PIL as the ORACLE (lossless -> must equal source for our compare)
    rt = Image.open(p).convert("RGBA")
    px = list(rt.getdata())  # list of (r,g,b,a)
    w, h = rt.size
    with open(os.path.join(BASE, name + ".pix"), "w") as f:
        f.write("%d %d 4\n" % (w, h))
        out = []
        for (r, g, b, a) in px:
            out.append("%d %d %d %d" % (r, g, b, a))
        f.write(" ".join(out))
    print("fixture %s: %dx%d  (%d bytes webp)" % (name, w, h, os.path.getsize(p)))

# 1) photo-like gradient + noise RGBA (exercises predictor/color transforms)
W, H = 48, 32
im = Image.new("RGBA", (W, H))
pix = im.load()
for y in range(H):
    for x in range(W):
        r = (x * 255 // (W - 1) + random.randint(-8, 8)) & 0xFF
        g = (y * 255 // (H - 1) + random.randint(-8, 8)) & 0xFF
        b = ((x + y) * 255 // (W + H - 2)) & 0xFF
        a = 255 - ((x * y) % 64)
        pix[x, y] = (r, g, b, a)
dump(im, "gradient_rgba")

# 2) flat-color RGB (exercises color-indexing / subtract-green)
im2 = Image.new("RGB", (40, 24), (37, 142, 211))
dump(im2, "flat_rgb")

# 3) small image (8x8 RGBA with a few distinct colors -> palette)
im3 = Image.new("RGBA", (8, 8))
p3 = im3.load()
cols = [(255,0,0,255),(0,255,0,255),(0,0,255,128),(255,255,0,255)]
for y in range(8):
    for x in range(8):
        p3[x, y] = cols[(x + y) % 4]
dump(im3, "small_rgba")

# 4) a few-color RGB photo region (palette + transforms mix)
im4 = Image.new("RGB", (32, 32))
p4 = im4.load()
for y in range(32):
    for x in range(32):
        v = ((x // 4) + (y // 4)) % 3
        p4[x, y] = [(10, 20, 30), (200, 50, 80), (40, 220, 130)][v]
dump(im4, "blocks_rgb")

print("DONE fixtures in", BASE)
