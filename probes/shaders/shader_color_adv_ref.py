#!/usr/bin/env python3
# Numpy reference for group "color_adv": ColorBalance, CAS, USM, Lut3D.
# Builds the IDENTICAL 128x128 RGBA f32 test image and applies the SAME algorithms,
# writing flattened uint8 RGBA (space separated) to /tmp/shader_<name>_ref.txt.
#
# Citations match the Mojo file:
#   ColorBalance: /tmp/Med/plugin/nodes/filters/ColorBalance/ColorBalance_shader.h:36-129
#   CAS:          /tmp/Med/plugin/nodes/filters/CAS/CAS_shader.h:24-109
#   USM:          standard unsharp mask (gaussian sigma + amount + threshold)
#   Lut3D:        trilinear over identity LUT (LUTSIZE=17) => pass-through

import numpy as np

W, H = 128, 128
LUTSIZE = 17
KR = 7
KW = 2 * KR + 1


def build_test_image():
    img = np.zeros((H, W, 4), dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)  # X[y,x], Y[y,x]
    diag = (X + Y) / np.float32(W + H - 2)
    r = diag.copy()
    g = X / np.float32(W - 1)
    b = Y / np.float32(H - 1)
    # bright rect [20..50)x[20..50)
    rect = (X >= 20) & (X < 50) & (Y >= 20) & (Y < 50)
    r[rect] = 0.95
    g[rect] = 0.9
    b[rect] = 0.1
    # bright circle center (90,90) radius 18
    ddx = X - 90.0
    ddy = Y - 90.0
    circ = (ddx * ddx + ddy * ddy) < float(18 * 18)
    r[circ] = 0.1
    g[circ] = 0.85
    b[circ] = 0.95
    img[:, :, 0] = np.clip(r, 0.0, 1.0)
    img[:, :, 1] = np.clip(g, 0.0, 1.0)
    img[:, :, 2] = np.clip(b, 0.0, 1.0)
    img[:, :, 3] = 1.0
    return img


def f32_to_u8(a):
    return np.clip(np.floor(a * 255.0 + 0.5), 0, 255).astype(np.uint8)


def write_txt(path, img):
    u8 = f32_to_u8(img)
    flat = u8.reshape(-1)
    with open(path, "w") as f:
        f.write(" ".join(str(int(v)) for v in flat))


# ---------------------------------------------------------------------------
# ColorBalance (ColorBalance_shader.h:93-118), preserve_lightness = 0
# ---------------------------------------------------------------------------
def cb_get_component(v, l, s_in, m_in, h_in):
    a = np.float32(4.0)
    b = np.float32(0.333)
    scale = np.float32(0.7)
    s = s_in * np.clip((b - l) * a + 0.5, 0.0, 1.0) * scale
    m = (
        m_in
        * np.clip((l - b) * a + 0.5, 0.0, 1.0)
        * np.clip((1.0 - l - b) * a + 0.5, 0.0, 1.0)
        * scale
    )
    h = h_in * np.clip((l + b - 1.0) * a + 0.5, 0.0, 1.0) * scale
    return np.clip(v + s + m + h, 0.0, 1.0)


def colorbalance(img, params):
    rs, gs, bs, rm, gm, bm, rh, gh, bh = params
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    a = img[:, :, 3]
    l = np.maximum.reduce([r, g, b]) + np.minimum.reduce([r, g, b])  # CB_shader.h:111
    vr = cb_get_component(r, l, rs, rm, rh)
    vg = cb_get_component(g, l, gs, gm, gh)
    vb = cb_get_component(b, l, bs, bm, bh)
    out = np.zeros_like(img)
    out[:, :, 0] = np.clip(vr, 0.0, 1.0)
    out[:, :, 1] = np.clip(vg, 0.0, 1.0)
    out[:, :, 2] = np.clip(vb, 0.0, 1.0)
    out[:, :, 3] = a
    return out


# ---------------------------------------------------------------------------
# CAS (CAS_shader.h:28-97)
# ---------------------------------------------------------------------------
def cas(img, strength):
    out = np.zeros_like(img)
    # clamped neighbor shifts
    def shifted(arr, dx, dy):
        # arr[y,x] -> arr at clamped (x+dx, y+dy)
        xi = np.clip(np.arange(W) + dx, 0, W - 1)
        yi = np.clip(np.arange(H) + dy, 0, H - 1)
        return arr[np.ix_(yi, xi)]

    for ch in range(3):
        e = img[:, :, ch]
        a = shifted(e, -1, -1)
        b = shifted(e, 0, -1)
        c = shifted(e, 1, -1)
        d = shifted(e, -1, 0)
        f = shifted(e, 1, 0)
        g = shifted(e, -1, 1)
        h = shifted(e, 0, 1)
        i = shifted(e, 1, 1)

        mn = np.minimum.reduce([d, e, f, b, h])
        mn2 = np.minimum.reduce([mn, a, c, g, i])
        mn = mn + mn2
        mx = np.maximum.reduce([d, e, f, b, h])
        mx2 = np.maximum.reduce([mx, a, c, g, i])
        mx = mx + mx2

        amp = np.sqrt(np.clip(np.minimum(mn, np.float32(2.0) - mx) / mx, 0.0, 1.0))
        weight = amp / np.float32(strength)
        out[:, :, ch] = ((b + d + f + h) * weight + e) / (1.0 + 4.0 * weight)
    out[:, :, 3] = img[:, :, 3]
    return out


# ---------------------------------------------------------------------------
# USM (standard): separable gaussian blur (radius KR) -> orig + amount*(orig-blur),
# gated per channel by |diff| >= threshold.
# ---------------------------------------------------------------------------
def gaussian_kernel(sigma):
    d = np.arange(KW, dtype=np.float32) - KR
    w = np.exp(-(d * d) / (2.0 * sigma * sigma)).astype(np.float32)
    return (w / w.sum()).astype(np.float32)


def blur_separable(img, kern):
    out = img.copy()
    # horizontal
    tmp = np.zeros_like(img)
    for t in range(KW):
        dx = t - KR
        xi = np.clip(np.arange(W) + dx, 0, W - 1)
        tmp += kern[t] * img[:, xi, :]
    # vertical
    res = np.zeros_like(img)
    for t in range(KW):
        dy = t - KR
        yi = np.clip(np.arange(H) + dy, 0, H - 1)
        res += kern[t] * tmp[yi, :, :]
    return res


def usm(img, sigma, amount, threshold):
    kern = gaussian_kernel(sigma)
    blur = blur_separable(img, kern)
    out = img.copy()
    for ch in range(3):
        orig = img[:, :, ch]
        bl = blur[:, :, ch]
        diff = orig - bl
        sharp = orig + amount * diff
        gate = np.abs(diff) >= threshold
        res = np.where(gate, sharp, orig)
        out[:, :, ch] = np.clip(res, 0.0, 1.0)
    out[:, :, 3] = img[:, :, 3]
    return out


# ---------------------------------------------------------------------------
# Lut3D: trilinear over identity LUT (LUTSIZE^3, B fastest) => pass-through.
# Replicates the exact trilinear arithmetic (not just identity shortcut).
# ---------------------------------------------------------------------------
def build_identity_lut():
    sc = np.float32(LUTSIZE - 1)
    lut = np.zeros((LUTSIZE, LUTSIZE, LUTSIZE, 4), dtype=np.float32)
    ri = np.arange(LUTSIZE, dtype=np.float32)
    R, G, B = np.meshgrid(ri, ri, ri, indexing="ij")
    lut[:, :, :, 0] = R / sc
    lut[:, :, :, 1] = G / sc
    lut[:, :, :, 2] = B / sc
    lut[:, :, :, 3] = 0.0
    return lut


def lut3d(img, lut):
    sc = np.float32(LUTSIZE - 1)
    r = np.clip(img[:, :, 0], 0.0, 1.0)
    g = np.clip(img[:, :, 1], 0.0, 1.0)
    b = np.clip(img[:, :, 2], 0.0, 1.0)
    a = img[:, :, 3]
    fr = r * sc
    fg = g * sc
    fb = b * sc
    r0 = np.clip(fr.astype(np.int32), 0, LUTSIZE - 2)
    g0 = np.clip(fg.astype(np.int32), 0, LUTSIZE - 2)
    b0 = np.clip(fb.astype(np.int32), 0, LUTSIZE - 2)
    dr = (fr - r0).astype(np.float32)
    dg = (fg - g0).astype(np.float32)
    db = (fb - b0).astype(np.float32)
    out = np.zeros_like(img)
    for ch in range(3):
        def L(ro, go, bo):
            return lut[r0 + ro, g0 + go, b0 + bo, ch]
        c000 = L(0, 0, 0); c001 = L(0, 0, 1)
        c010 = L(0, 1, 0); c011 = L(0, 1, 1)
        c100 = L(1, 0, 0); c101 = L(1, 0, 1)
        c110 = L(1, 1, 0); c111 = L(1, 1, 1)
        c00 = c000 * (1 - db) + c001 * db
        c01 = c010 * (1 - db) + c011 * db
        c10 = c100 * (1 - db) + c101 * db
        c11 = c110 * (1 - db) + c111 * db
        c0 = c00 * (1 - dg) + c01 * dg
        c1 = c10 * (1 - dg) + c11 * dg
        out[:, :, ch] = c0 * (1 - dr) + c1 * dr
    out[:, :, 3] = a
    return out


def main():
    img = build_test_image()
    write_txt("/tmp/shader_color_adv_in.txt", img)

    cb_params = (0.3, -0.2, 0.1, -0.1, 0.25, -0.15, 0.2, -0.1, 0.3)
    write_txt("/tmp/shader_colorbalance_ref.txt", colorbalance(img, cb_params))
    write_txt("/tmp/shader_cas_ref.txt", cas(img, 0.8))
    write_txt("/tmp/shader_usm_ref.txt", usm(img, 3.0, 1.5, 0.0))
    lut = build_identity_lut()
    write_txt("/tmp/shader_lut3d_ref.txt", lut3d(img, lut))
    print("color_adv ref: wrote colorbalance/cas/usm/lut3d ref dumps + input")


if __name__ == "__main__":
    main()
