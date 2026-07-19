#!/usr/bin/env python3
# Generates a PIL reference for the depth_test.mojo PIL cross-check.
#
# Builds a 16-bit grayscale PNG (mode "I;16") with a known 8-pixel ramp,
# reads back the raw 16-bit values, and also computes PIL's own L (8-bit)
# downconversion. Writes both as plain-text lines that the Mojo test reads:
#
#   line 1: N
#   line 2: space-separated raw u16 values (what we set16 in Mojo)
#   line 3: space-separated (raw >> 8) high-byte bytes (the to_u8 oracle)
#   line 4: space-separated PIL convert("L") bytes (informational only)
#
# NOTE: PIL's I;16 -> "L" conversion does NOT do a plain >>8 high-byte
# truncation; it applies its own scaling (which saturates here). The
# documented contract for our to_u8 is high-byte (>>8), so the Mojo test
# compares against line 3, and we print PIL's convert("L") for reference.
#
# Run via the project's pixi python; see depth_test.mojo header for the path.

import os
import numpy as np
from PIL import Image as PImage

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "depth_ref.txt")

# A spread-out 16-bit ramp (1xN grayscale). Include 0, 0xFFFF, and mid values.
vals = [0x0000, 0x0100, 0x1234, 0x8000, 0xABCD, 0xFFFF, 0x00FF, 0x7F80]
N = len(vals)

arr = np.array([vals], dtype=np.uint16)  # shape (1, N)
im16 = PImage.fromarray(arr, mode="I;16")

# round-trip through a real PNG file to prove it is genuine 16-bit on disk
png_path = os.path.join(HERE, "depth_ref.png")
im16.save(png_path)
reloaded = PImage.open(png_path)
raw = np.asarray(reloaded)            # (1, N) uint16
raw_vals = [int(v) for v in raw.reshape(-1)]

# PIL's own 16 -> 8 bit luminance conversion (its high-byte / >>8 path).
l8 = np.asarray(reloaded.convert("L"))  # (1, N) uint8
l8_vals = [int(v) for v in l8.reshape(-1)]

hi_vals = [v >> 8 for v in raw_vals]

with open(OUT, "w") as f:
    f.write(str(N) + "\n")
    f.write(" ".join(str(v) for v in raw_vals) + "\n")
    f.write(" ".join(str(v) for v in hi_vals) + "\n")
    f.write(" ".join(str(v) for v in l8_vals) + "\n")

print("wrote", OUT)
print("N      =", N)
print("raw u16=", raw_vals)
print("PIL L8 =", l8_vals)
print("hi>>8  =", [v >> 8 for v in raw_vals])
