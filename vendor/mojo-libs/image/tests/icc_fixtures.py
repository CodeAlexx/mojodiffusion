#!/usr/bin/env python3
# Generate ICC fixtures with PIL/ImageCms (littleCMS).
#
#  /tmp/icc_fix/srgb.icc        -- raw sRGB profile bytes (ImageCms)
#  /tmp/icc_fix/tagged.png      -- a non-trivial RGB image tagged with sRGB
#  /tmp/icc_fix/tagged.pix      -- expected pixels of tagged.png (W H C + samples)
#  /tmp/icc_fix/expected.txt    -- parsed header fields / matrix / whitepoint
#                                  so the Mojo test can cross-check exact numbers.
import os, struct
from PIL import Image, ImageCms

OUT = "/tmp/icc_fix"
os.makedirs(OUT, exist_ok=True)

srgb = ImageCms.createProfile("sRGB")
icc = ImageCms.ImageCmsProfile(srgb).tobytes()
open(os.path.join(OUT, "srgb.icc"), "wb").write(icc)

# non-trivial RGB fixture
w, h = 16, 12
img = Image.new("RGB", (w, h))
for y in range(h):
    for x in range(w):
        img.putpixel((x, y), ((x * 17) % 256, (y * 21 + 3) % 256, (x * y + 7) % 256))
img.save(os.path.join(OUT, "tagged.png"), icc_profile=icc)

# expected pixel dump
data = list(img.getdata())
flat = []
for px in data:
    flat.extend(px)
with open(os.path.join(OUT, "tagged.pix"), "w") as f:
    f.write("%d %d %d\n" % (w, h, 3))
    for v in flat:
        f.write("%d\n" % v)

# parse header fields with struct to write a ground-truth file
b = icc
def s15f16(o):
    return struct.unpack(">i", b[o:o+4])[0] / 65536.0
size = struct.unpack(">I", b[0:4])[0]
dclass = b[12:16].decode("latin1")
cspace = b[16:20].decode("latin1")
pcs = b[20:24].decode("latin1")
acsp = b[36:40].decode("latin1")
cnt = struct.unpack(">I", b[128:132])[0]
tags = {}
for i in range(cnt):
    off = 132 + i * 12
    sig = b[off:off+4].decode("latin1")
    o = struct.unpack(">I", b[off+4:off+8])[0]
    s = struct.unpack(">I", b[off+8:off+12])[0]
    tags[sig] = (o, s)

def xyz(sig):
    o = tags[sig][0]
    return (s15f16(o+8), s15f16(o+12), s15f16(o+16))

with open(os.path.join(OUT, "expected.txt"), "w") as f:
    f.write("size %d\n" % size)
    f.write("device_class %s\n" % dclass)
    f.write("color_space %s\n" % repr(cspace))
    f.write("pcs %s\n" % repr(pcs))
    f.write("acsp %s\n" % acsp)
    f.write("tag_count %d\n" % cnt)
    wp = xyz("wtpt")
    f.write("wtpt %.6f %.6f %.6f\n" % wp)
    for t in ("rXYZ", "gXYZ", "bXYZ"):
        v = xyz(t)
        f.write("%s %.6f %.6f %.6f\n" % (t, v[0], v[1], v[2]))
    # rTRC type
    o = tags["rTRC"][0]
    f.write("rTRC_type %s\n" % b[o:o+4].decode("latin1"))

print("wrote fixtures to", OUT)
print("  srgb.icc size:", len(icc))
print("  tagged.png:", w, "x", h)
for line in open(os.path.join(OUT, "expected.txt")):
    print("  " + line.rstrip())
