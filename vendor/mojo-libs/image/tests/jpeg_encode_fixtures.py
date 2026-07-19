#!/usr/bin/env python3
# jpeg_encode_fixtures.py — PIL oracle driver for the Mojo JPEG ENCODER test.
#
# Usage:
#   python3 jpeg_encode_fixtures.py gen      # STAGE 1: synthesize originals
#   <run the mojo test>                       # STAGE 2: it encodes to .jpg
#   python3 jpeg_encode_fixtures.py verify   # STAGE 3: PIL opens our .jpg,
#                                             #          checks dims/mode, PSNRs
#
# Originals are dumped as "<name>.orig.pix" (W H C then ints). The Mojo test
# rebuilds these, encodes "<name>.jpg", and dumps its own decode as
# "<name>.ourdec.pix". Verify then opens each .jpg with PIL (the real-world
# proof), dumps PIL's decode as "<name>.pildec.pix", and reports PSNRs.

import os
import sys
import math
from PIL import Image

BASE = "/tmp/jpg_enc"

# name -> (quality_used_by_mojo, expected_PIL_mode)
CASES = {
    "rgb":     (90, "RGB"),
    "gray":    (90, "L"),
    "rgb_q50": (50, "RGB"),
    "rgb_q95": (95, "RGB"),
}


def dump_pix(path, im):
    w, h = im.size
    px = list(im.getdata())
    if im.mode == "L":
        c = 1
        flat = px
    else:
        c = 3
        flat = []
        for t in px:
            flat.extend(t[:3])
    with open(path, "w") as f:
        f.write(f"{w} {h} {c}\n")
        # write row-major; getdata is already row-major
        out = []
        for v in flat:
            out.append(str(int(v)))
        f.write(" ".join(out))
        f.write("\n")


def load_pix(path):
    with open(path) as f:
        toks = f.read().split()
    w, h, c = int(toks[0]), int(toks[1]), int(toks[2])
    vals = [int(t) for t in toks[3:3 + w * h * c]]
    return w, h, c, vals


def make_rgb(w=64, h=48):
    # Gradient + moderate checker + a band-limited ripple. Real AC content
    # (so the quality sweep separates clearly) but no single-pixel impulses
    # (which are a degenerate worst case even for PIL's own encoder), so q90
    # 4:4:4 lands comfortably above the 34 dB fidelity bar.
    im = Image.new("RGB", (w, h))
    px = im.load()
    for y in range(h):
        for x in range(w):
            r = (x * 255) // (w - 1)
            g = (y * 255) // (h - 1)
            b = ((x + y) * 255) // (w + h - 2)
            if ((x // 6) + (y // 6)) % 2 == 0:
                r = min(255, r + 35)
                g = max(0, g - 25)
            r = min(255, max(0, r + int(20 * math.sin(x * 0.6))))
            b = min(255, max(0, b + int(15 * math.cos(y * 0.5))))
            px[x, y] = (r, g, b)
    return im


def make_gray(w=64, h=48):
    im = Image.new("L", (w, h))
    px = im.load()
    for y in range(h):
        for x in range(w):
            v = ((x * 255) // (w - 1) + (y * 255) // (h - 1)) // 2
            if ((x // 6) + (y // 6)) % 2 == 0:
                v = min(255, v + 30)
            v = min(255, max(0, v + int(18 * math.sin(x * 0.5))))
            px[x, y] = v
    return im


def psnr(a, b):
    if len(a) != len(b):
        return -1.0
    sse = 0.0
    for x, y in zip(a, b):
        d = float(x) - float(y)
        sse += d * d
    mse = sse / len(a)
    if mse <= 0.0:
        return 999.0
    return 10.0 * math.log10(255.0 * 255.0 / mse)


def gen():
    os.makedirs(BASE, exist_ok=True)
    rgb = make_rgb()
    gray = make_gray()
    # same RGB image is reused for the quality sweep
    for name in CASES:
        im = gray if name == "gray" else rgb
        dump_pix(f"{BASE}/{name}.orig.pix", im)
    print(f"[gen] wrote {len(CASES)} original fixtures to {BASE}")


def verify():
    passed = 0
    failed = 0
    psnrs = {}
    for name, (q, mode) in CASES.items():
        jpg = f"{BASE}/{name}.jpg"
        if not os.path.exists(jpg):
            print(f"  FAIL {name}: {jpg} not produced by the Mojo encoder")
            failed += 1
            continue

        # --- 1. PIL MUST OPEN our JPEG ---
        try:
            im = Image.open(jpg)
            im.load()
        except Exception as e:
            print(f"  FAIL {name}: PIL could not open our JPEG: {e}")
            failed += 1
            continue

        ow, oh, oc, ovals = load_pix(f"{BASE}/{name}.orig.pix")
        if im.size != (ow, oh):
            print(f"  FAIL {name}: PIL dims {im.size} != original ({ow},{oh})")
            failed += 1
            continue
        if im.mode != mode:
            print(f"  FAIL {name}: PIL mode {im.mode} != expected {mode}")
            failed += 1
            continue
        print(f"  pass {name}: PIL opened our JPEG  dims={im.size} mode={im.mode}")
        passed += 1

        # PIL decode of OUR jpeg -> flat ints; dump for the record
        dump_pix(f"{BASE}/{name}.pildec.pix", im)
        _, _, _, pilvals = load_pix(f"{BASE}/{name}.pildec.pix")

        # --- 2. encode-fidelity PSNR: PIL-decode-of-our-jpeg vs original ---
        ps_fid = psnr(pilvals, ovals)
        psnrs[name] = ps_fid
        print(f"       PSNR(PIL-decode-our-jpeg vs original) = {ps_fid:.2f} dB")

        # --- 3. bitstream-agreement PSNR: our decode vs PIL decode (same file) ---
        ourpath = f"{BASE}/{name}.ourdec.pix"
        if os.path.exists(ourpath):
            _, _, _, ourvals = load_pix(ourpath)
            ps_agree = psnr(ourvals, pilvals)
            print(f"       PSNR(our-decode vs PIL-decode, same file) = {ps_agree:.2f} dB")
            if name in ("rgb", "gray"):
                if ps_agree >= 45.0:
                    passed += 1
                    print(f"       pass {name}: decoders agree (>=45 dB)")
                else:
                    failed += 1
                    print(f"       FAIL {name}: decoders disagree (<45 dB)")
        else:
            print(f"       (no {ourpath}; run the Mojo test first)")

    # --- bars on the q90 fidelity cases (>= 34 dB) ---
    for name in ("rgb", "gray"):
        if name in psnrs:
            if psnrs[name] >= 34.0:
                passed += 1
                print(f"  pass {name}: q90 fidelity PSNR {psnrs[name]:.2f} dB >= 34 dB")
            else:
                failed += 1
                print(f"  FAIL {name}: q90 fidelity PSNR {psnrs[name]:.2f} dB < 34 dB")

    # --- quality sweep: q95 must beat q50, both must open (checked above) ---
    if "rgb_q50" in psnrs and "rgb_q95" in psnrs:
        if psnrs["rgb_q95"] > psnrs["rgb_q50"]:
            passed += 1
            print(f"  pass quality-sweep: q95 {psnrs['rgb_q95']:.2f} dB > "
                  f"q50 {psnrs['rgb_q50']:.2f} dB")
        else:
            failed += 1
            print(f"  FAIL quality-sweep: q95 {psnrs['rgb_q95']:.2f} dB <= "
                  f"q50 {psnrs['rgb_q50']:.2f} dB")

    print(f"passed: {passed} failed: {failed}")
    if failed == 0:
        print("ALL JPEG ENCODE TESTS PASSED")
    else:
        print("JPEG ENCODE TESTS HAD FAILURES")
        sys.exit(1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gen"
    if mode == "gen":
        gen()
    elif mode == "verify":
        verify()
    else:
        print("usage: jpeg_encode_fixtures.py [gen|verify]")
        sys.exit(2)
