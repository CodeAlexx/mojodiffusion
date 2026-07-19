#!/usr/bin/env python3
# image/tests/cmyk_fixtures.py — PIL (Pillow 10.2) oracle for the Mojo CMYK module.
#
# Roles:
#   PASS 1 (run BEFORE the Mojo test): build a deterministic RGB fixture, convert
#     to CMYK with PIL (rgb.convert("CMYK")), and dump:
#       /tmp/mojo_cmyk/orig_rgb.txt(+.bin)   : the original RGB bytes (HxWx3)
#       /tmp/mojo_cmyk/pil_cmyk.txt(+.bin)   : PIL's CMYK channel bytes (HxWx4)
#       /tmp/mojo_cmyk/pil_cmyk_rgb.txt(+.bin): PIL's CMYK.convert("RGB") (HxWx3)
#     PIL stores CMYK as STANDARD ink-amount samples (cyan->(255,0,0,0)), which is
#     exactly the convention image/cmyk.mojo's cmyk_to_rgb() expects.
#
#   PASS 2 (run AFTER the Mojo test): load the Mojo outputs and compare:
#       mojo_cmyk_to_rgb (from PIL's CMYK)  vs  orig_rgb        -> PSNR  (bar 30)
#       mojo_cmyk_to_rgb (from PIL's CMYK)  vs  PIL_cmyk.convert("RGB")  -> PSNR
#
# Run: python3 image/tests/cmyk_fixtures.py prep     # PASS 1
#      <run the mojo test>
#      python3 image/tests/cmyk_fixtures.py check    # PASS 2
import os, sys, math
import numpy as np
from PIL import Image

D = "/tmp/mojo_cmyk"
W, H = 48, 36


def make_rgb(w, h):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            a[y, x, 0] = (x * 7 + y * 3) & 0xFF
            a[y, x, 1] = (x * 3 + y * 11 + 40) & 0xFF
            a[y, x, 2] = ((x ^ y) * 5 + 17) & 0xFF
    return a


def dump(arr, name):
    h, w, c = arr.shape
    with open(os.path.join(D, name + ".txt"), "w") as f:
        f.write(f"{w} {h} {c}\n")
    arr.astype(np.uint8).tobytes()  # ensure contiguous
    np.ascontiguousarray(arr, dtype=np.uint8).tofile(os.path.join(D, name + ".txt.bin"))


def load(name):
    with open(os.path.join(D, name + ".txt")) as f:
        w, h, c = map(int, f.readline().split())
    raw = np.fromfile(os.path.join(D, name + ".txt.bin"), dtype=np.uint8)
    return raw.reshape(h, w, c)


def psnr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    if a.shape != b.shape:
        return -1.0
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return math.inf
    return 10.0 * math.log10((255.0 ** 2) / mse)


def maxdiff(a, b):
    if a.shape != b.shape:
        return None
    return int(np.max(np.abs(a.astype(int) - b.astype(int))))


def prep():
    os.makedirs(D, exist_ok=True)
    rgb = make_rgb(W, H)
    pil_rgb = Image.fromarray(rgb, "RGB")
    pil_cmyk = pil_rgb.convert("CMYK")
    cmyk = np.asarray(pil_cmyk).reshape(H, W, 4)
    pil_cmyk_rgb = np.asarray(pil_cmyk.convert("RGB")).reshape(H, W, 3)
    dump(rgb, "orig_rgb")
    dump(cmyk, "pil_cmyk")
    dump(pil_cmyk_rgb, "pil_cmyk_rgb")
    print("== cmyk PASS 1 (prep) ==")
    print(f"  wrote orig_rgb {rgb.shape}, pil_cmyk {cmyk.shape}, pil_cmyk_rgb {pil_cmyk_rgb.shape} to {D}")
    print(f"  sample CMYK[0,0] = {tuple(int(v) for v in cmyk[0,0])}")


def check():
    print("== cmyk PASS 2 (PIL parity check) ==")
    orig = load("orig_rgb")
    pil_cmyk_rgb = load("pil_cmyk_rgb")
    mojo = load("mojo_cmyk_to_rgb")  # mojo cmyk_to_rgb(PIL's CMYK)

    failed = 0

    p1 = psnr(mojo, orig)
    d1 = maxdiff(mojo, orig)
    ok1 = p1 >= 30.0
    failed += 0 if ok1 else 1
    print(f"  [{'PASS' if ok1 else 'FAIL'}] mojo_cmyk_to_rgb vs ORIGINAL rgb   "
          f"max|diff|={d1}  PSNR={p1:.2f} dB (bar 30)")

    p2 = psnr(mojo, pil_cmyk_rgb)
    d2 = maxdiff(mojo, pil_cmyk_rgb)
    ok2 = p2 >= 40.0
    failed += 0 if ok2 else 1
    print(f"  [{'PASS' if ok2 else 'FAIL'}] mojo_cmyk_to_rgb vs PIL CMYK.convert(RGB)  "
          f"max|diff|={d2}  PSNR={p2:.2f} dB (bar 40)")

    print(f"passed: {2 - failed} failed: {failed}")
    if failed == 0:
        print("ALL CMYK PIL-PARITY CHECKS PASSED")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "prep"
    if mode == "prep":
        prep()
    elif mode == "check":
        check()
    else:
        print("usage: cmyk_fixtures.py prep|check")
        sys.exit(2)
