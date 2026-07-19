#!/usr/bin/env python3
# image/tests/ops_oracle.py — PIL (Pillow 10.2) parity oracle for the Mojo CPU ops.
#
# Reads the fixtures and op outputs dumped by ops_test.mojo into /tmp/mojo_ops/,
# applies the PIL-equivalent op to the SAME fixture bytes, and compares:
#   - EXACT  : pixel-identical (allow <=1 LSB for luma rounding where noted)
#   - PSNR   : peak signal-to-noise ratio in dB (bar 40 dB; investigate <35)
#   - STRUCT : already covered in the Mojo self-checks; a couple re-checked here.
#
# Run AFTER the Mojo test:
#   python3 image/tests/ops_oracle.py
import os, math, sys
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

D = "/tmp/mojo_ops"

def load(name):
    """Load a Mojo-dumped image -> numpy HxWxC uint8 (C may be 1/3/4)."""
    with open(os.path.join(D, name + ".txt")) as f:
        w, h, c = map(int, f.readline().split())
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

def exact_diff(a, b):
    if a.shape != b.shape:
        return None
    return int(np.max(np.abs(a.astype(int) - b.astype(int))))

passed = 0
failed = 0
results = []

def record(name, ok, detail):
    global passed, failed
    if ok: passed += 1
    else: failed += 1
    results.append((name, ok, detail))

# ---- fixtures (PIL input == exact bytes Mojo used) ----
rgb = load("fixture_rgb")
rgba = load("fixture_rgba")
pil_rgb = to_pil(rgb)
pil_rgba = to_pil(rgba)

# ---------- EXACT-match ops ----------
def exact(name, mojo_name, pil_img, tol=0):
    mo = load(mojo_name)
    pl = from_pil(pil_img)
    d = exact_diff(mo, pl)
    ok = d is not None and d <= tol
    record(name, ok, f"max|diff|={d} (tol {tol})")

exact("flip_h",   "flip_h",   pil_rgb.transpose(Image.FLIP_LEFT_RIGHT))
exact("flip_v",   "flip_v",   pil_rgb.transpose(Image.FLIP_TOP_BOTTOM))
exact("rotate90", "rotate90", pil_rgb.transpose(Image.ROTATE_90))
exact("rotate180","rotate180",pil_rgb.transpose(Image.ROTATE_180))
exact("rotate270","rotate270",pil_rgb.transpose(Image.ROTATE_270))
exact("crop",     "crop",     pil_rgb.crop((5, 4, 5+20, 4+16)))
exact("transpose","transpose",pil_rgb.transpose(Image.TRANSPOSE))
exact("invert",   "invert",   ImageOps.invert(pil_rgb))
# grayscale: PIL convert("L") -> our to_gray1ch (1ch). Allow <=1 LSB.
exact("to_gray1ch","to_gray1ch", pil_rgb.convert("L"), tol=1)
# nearest resize (PIL NEAREST). Half-pixel convention can differ at edges; allow exact attempt
exact("rotate90_dims", "rotate90", pil_rgb.transpose(Image.ROTATE_90))  # redundant dims-safe
exact("rgba_flip_h", "rgba_flip_h", pil_rgba.transpose(Image.FLIP_LEFT_RIGHT))

# grayscale 3ch: compare against PIL L broadcast to RGB
mo_gs = load("grayscale")
pil_L = np.asarray(pil_rgb.convert("L"))
pil_gs3 = np.stack([pil_L, pil_L, pil_L], axis=2)
d = exact_diff(mo_gs, pil_gs3)
record("grayscale(3ch)", d is not None and d <= 1, f"max|diff|={d} (tol 1)")

# ---------- PSNR ops ----------
def psnr_check(name, mojo_name, pil_img, bar=40.0):
    mo = load(mojo_name)
    pl = from_pil(pil_img)
    val = psnr(mo, pl)
    ok = val >= bar
    record(name, ok, f"PSNR={val:.2f} dB (bar {bar})")

psnr_check("resize_bilinear",      "resize_bilinear",      pil_rgb.resize((96, 72), Image.BILINEAR))
psnr_check("resize_bilinear_down", "resize_bilinear_down", pil_rgb.resize((24, 18), Image.BILINEAR))
psnr_check("resize_bicubic",       "resize_bicubic",       pil_rgb.resize((96, 72), Image.BICUBIC))
psnr_check("gaussian_blur",        "gaussian_blur",        pil_rgb.filter(ImageFilter.GaussianBlur(2.0)))
psnr_check("box_blur",             "box_blur",             pil_rgb.filter(ImageFilter.BoxBlur(2)))

# resize_nearest: PIL NEAREST (may differ at boundaries due to rounding; report PSNR)
psnr_check("resize_nearest",       "resize_nearest",       pil_rgb.resize((96, 72), Image.NEAREST), bar=40.0)

# ---------- structural / convention checks ----------
def struct(name, ok, detail):
    record(name, ok, detail)

th = load("threshold")
struct("threshold 0/255 only", set(np.unique(th)).issubset({0, 255}),
       f"unique={sorted(set(np.unique(th)))[:4]}...")

sob = load("sobel")
# sobel of our fixture should have real edges (nonzero) but be a valid 1ch image
struct("sobel is 1ch & has signal", sob.shape[2] == 1 and sob.max() > 0,
       f"shape={sob.shape}, max={int(sob.max())}")

br = load("brightness")
struct("brightness(+50) raises mean", br[:,:,:3].mean() > rgb[:,:,:3].mean(),
       f"mean {rgb[:,:,:3].mean():.1f} -> {br[:,:,:3].mean():.1f}")

# contrast: compare to PIL ImageEnhance.Contrast? PIL uses mean-gray midpoint, not 128,
# so we only structurally check our own definition is sane (var increases for factor>1).
ct = load("contrast")
struct("contrast(1.5) increases variance",
       ct[:,:,:3].astype(float).var() > rgb[:,:,:3].astype(float).var(),
       f"var {rgb[:,:,:3].astype(float).var():.0f} -> {ct[:,:,:3].astype(float).var():.0f}")

# gamma vs PIL: PIL has no direct gamma; structural monotonic check via LUT match.
gm = load("gamma")
lut = np.array([min(255, int(math.floor(255.0*((i/255.0)**2.2)+0.5))) for i in range(256)], dtype=np.uint8)
gm_ref = lut[rgb[:,:,:3]]
gm_ref = np.concatenate([gm_ref, rgb[:,:,3:]], axis=2) if rgb.shape[2] == 4 else gm_ref
d = exact_diff(gm, gm_ref)
struct("gamma matches LUT", d is not None and d == 0, f"max|diff|={d}")

# rotate30: same-size, corners filled with black -> check dims + corner is near fill
rot = load("rotate30")
corner_dark = int(rot[0,0,:3].max())
struct("rotate(30) same-size, corner=fill",
       rot.shape[:2] == rgb.shape[:2] and corner_dark <= 5,
       f"dims={rot.shape[:2]}, corner_max={corner_dark}")

# ---- report ----
print("== PIL (Pillow 10.2) parity ==")
namew = max(len(n) for n,_,_ in results)
for name, ok, detail in results:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name.ljust(namew)}  {detail}")
print(f"passed: {passed} failed: {failed}")
if failed == 0:
    print("ALL OPS TESTS PASSED")
sys.exit(0 if failed == 0 else 1)
