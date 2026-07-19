#!/usr/bin/env python3
# image/tests/studio_ops_oracle.py — PIL parity oracle for studio_ops.mojo.
#
# Reads the fixture + op outputs dumped by studio_ops_test.mojo into /tmp/mojo_studio/,
# applies the PIL-equivalent op to the SAME fixture bytes, and compares:
#   - resize_lanczos (up + down) : max|diff| + PSNR vs Image.resize(..., LANCZOS)
#   - unsharp_mask               : PSNR vs ImageFilter.UnsharpMask + sharpen check
#
# Run AFTER the Mojo test:
#   python3 image/tests/studio_ops_oracle.py
import os, math, sys
import numpy as np
from PIL import Image, ImageFilter

D = "/tmp/mojo_studio"


def load(name):
    with open(os.path.join(D, name + ".txt")) as fp:
        w, h, c = map(int, fp.readline().split())
    raw = np.fromfile(os.path.join(D, name + ".txt.bin"), dtype=np.uint8)
    return raw.reshape(h, w, c)


def to_pil(arr):
    c = arr.shape[2]
    if c == 1:
        return Image.fromarray(arr[:, :, 0], "L")
    if c == 3:
        return Image.fromarray(arr, "RGB")
    return Image.fromarray(arr, "RGBA")


def from_pil(im):
    a = np.asarray(im)
    if a.ndim == 2:
        a = a[:, :, None]
    return a


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


passed = 0
failed = 0
results = []


def record(name, ok, detail):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
    results.append((name, ok, detail))


rgb = load("fixture_rgb")
pil_rgb = to_pil(rgb)

# ---- lanczos upscale ----
mo_up = load("lanczos_up")
pl_up = from_pil(pil_rgb.resize((96, 72), Image.LANCZOS))
up_psnr = psnr(mo_up, pl_up)
up_md = maxdiff(mo_up, pl_up)
record("lanczos_up", up_psnr >= 45.0, f"PSNR={up_psnr:.2f} dB  max|diff|={up_md}  (bar 45 dB)")

# ---- lanczos downscale ----
mo_dn = load("lanczos_down")
pl_dn = from_pil(pil_rgb.resize((24, 18), Image.LANCZOS))
dn_psnr = psnr(mo_dn, pl_dn)
dn_md = maxdiff(mo_dn, pl_dn)
record("lanczos_down", dn_psnr >= 45.0, f"PSNR={dn_psnr:.2f} dB  max|diff|={dn_md}  (bar 45 dB)")

# ---- unsharp mask ----
mo_um = load("unsharp")
pl_um = from_pil(pil_rgb.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)))
um_psnr = psnr(mo_um, pl_um)
um_md = maxdiff(mo_um, pl_um)
record("unsharp_mask", um_psnr >= 35.0, f"PSNR={um_psnr:.2f} dB  max|diff|={um_md}  (bar 35 dB, gaussian approx)")

# ---- unsharp actually sharpens (variance + edge energy up vs original) ----
def variance(a):
    a = a[:, :, :3].astype(np.float64)
    return float(a.var())

v_o = variance(rgb)
v_s = variance(mo_um)
record("unsharp sharpens (variance up)", v_s > v_o, f"var {v_o:.1f} -> {v_s:.1f}")

# bit-exact note for lanczos
up_exact = up_md is not None and up_md <= 1
dn_exact = dn_md is not None and dn_md <= 1

print("== PIL (Pillow) parity — studio ops ==")
namew = max(len(n) for n, _, _ in results)
for name, ok, detail in results:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name.ljust(namew)}  {detail}")
print(f"lanczos bit-exact vs PIL (<=1 LSB): up={up_exact} down={dn_exact}")
print(f"passed: {passed} failed: {failed}")
if failed == 0:
    print("ALL STUDIO OPS TESTS PASSED")
sys.exit(0 if failed == 0 else 1)
