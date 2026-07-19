#!/usr/bin/env python3
# Independent PIL oracle for the svg render test. Run render_test.mojo first
# (it writes /tmp/svg_*.png), then this cross-checks coverage/bbox/color against
# PIL's own rasterization of the same geometry.
from PIL import Image, ImageDraw


def coverage(path):
    im = Image.open(path).convert("RGBA")
    px = im.load(); W, H = im.size
    n = 0; minx = W; miny = H; maxx = -1; maxy = -1
    for y in range(H):
        for x in range(W):
            if px[x, y][3] > 0:
                n += 1
                minx = min(minx, x); miny = min(miny, y)
                maxx = max(maxx, x); maxy = max(maxy, y)
    return n, (minx, miny, maxx, maxy)


def main():
    fails = 0

    # rect: 10x10 @ (7,7) in a 24 viewBox -> x4 -> (28,28)-(68,68) = 40x40 = 1600
    n, bb = coverage("/tmp/svg_rect.png")
    ok = n == 1600 and bb == (28, 28, 67, 67)
    print("rect  ", n, bb, "PASS" if ok else "FAIL"); fails += 0 if ok else 1

    # circle r=10 center 12 -> device r40 center48; compare to a PIL filled circle
    ref = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    ImageDraw.Draw(ref).ellipse([8, 8, 88, 88], fill=(0, 0, 0, 255))
    rn = sum(1 for y in range(96) for x in range(96) if ref.load()[x, y][3] > 0)
    n, bb = coverage("/tmp/svg_circle.png")
    err = abs(n - rn) / rn * 100
    ok = err < 5 and bb[0] <= 9 and bb[2] >= 86
    print("circle", n, bb, f"(PIL {rn}, {err:.1f}%)", "PASS" if ok else "FAIL"); fails += 0 if ok else 1

    # currentColor: 8x8 @ translate(2,2) -> (8,8)-(40,40)=1024; tint 200,30,30
    n, bb = coverage("/tmp/svg_currentcolor.png")
    mid = Image.open("/tmp/svg_currentcolor.png").convert("RGBA").load()[20, 20]
    ok = n == 1024 and bb == (8, 8, 39, 39) and mid == (200, 30, 30, 255)
    print("ccolor", n, bb, "mid", mid, "PASS" if ok else "FAIL"); fails += 0 if ok else 1

    # stroked check mark: nonempty
    n, bb = coverage("/tmp/svg_check.png")
    ok = n > 200
    print("check ", n, bb, "PASS" if ok else "FAIL"); fails += 0 if ok else 1

    print("\noracle:", "ALL PASS" if fails == 0 else f"{fails} FAILED")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
