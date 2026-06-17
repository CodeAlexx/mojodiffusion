#!/usr/bin/env python3
# numpy reference for group "denoise" (SmartDenoise, HQDN3D, Guided, Deband).
# Builds the IDENTICAL 128x128 RGBA test image as shader_denoise.mojo and applies
# the SAME algorithms on CPU, writing concatenated flattened uint8 RGBA (in the
# order SmartDenoise, HQDN3D, Guided, Deband) to /tmp/shader_denoise_ref.txt.
#
# Algorithm sources mirrored from /tmp/Med (see shader_denoise.mojo header for line refs):
#   SmartDenoise: SmartDenoise_shader.h:43-90    sigma=1.2 ksigma=2.0 threshold=0.2
#   HQDN3D:       HQDN3D_shader.h:28-99 + coef gen HQDN3D_vulkan.cpp:87-99
#   Guided:       intended self-guided color guided filter; coef solve
#                 Guided_shader.h:99-117 + Matting :175-177; box Box_vulkan.cpp:16-39
#                 (r=4, mean of r x r window, clamp-edge, anchor=r//2); eps=1e-4
#   Deband:       DeBand_shader.h:27-97 (blur=false) + pos gen DeBand_vulkan.cpp:5-10,54-70
#                 (range=16, direction=2*pi, threshold=0.01)

import math
import numpy as np

W, H = 128, 128


# ---------------------------------------------------------------------------
# Test image (identical to build_test_image in shader_denoise.mojo)
# ---------------------------------------------------------------------------
def build_test_image():
    img = np.zeros((H, W, 4), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            diag = np.float32((x + y) / (W + H - 2))
            r = diag
            g = np.float32(x / (W - 1))
            b = np.float32(y / (H - 1))
            if 20 <= x < 50 and 20 <= y < 50:
                r, g, b = np.float32(0.95), np.float32(0.9), np.float32(0.1)
            ddx = x - 90
            ddy = y - 90
            if ddx * ddx + ddy * ddy < 18 * 18:
                r, g, b = np.float32(0.1), np.float32(0.85), np.float32(0.95)
            img[y, x, 0] = min(max(r, 0.0), 1.0)
            img[y, x, 1] = min(max(g, 0.0), 1.0)
            img[y, x, 2] = min(max(b, 0.0), 1.0)
            img[y, x, 3] = 1.0
    return img


def clampi(v, lo, hi):
    return max(lo, min(v, hi))


def ld(img, x, y, ch):
    # load_rgba clamps x,y to edge (imvk_mat_shader.h:579-580)
    return img[clampi(y, 0, H - 1), clampi(x, 0, W - 1), ch]


# ---------------------------------------------------------------------------
# 1) SmartDenoise (SmartDenoise_shader.h:43-90)
# ---------------------------------------------------------------------------
INV_SQRT_OF_2PI = np.float32(0.39894228040143267793994605993439)
INV_PI = np.float32(0.31830988618379067153776752674503)
invGamma = np.float32(0.454545454545455)


def smartdenoise(img, sigma=1.2, ksigma=2.0, threshold=0.2):
    f = np.float32
    sigma = f(sigma); ksigma = f(ksigma); threshold = f(threshold)
    out = np.zeros((H, W, 4), dtype=np.float32)
    radius = f(round(float(ksigma * sigma)))  # round() (:48)
    radQ = f(radius * radius)
    invSigmaQx2 = f(0.5) / (sigma * sigma)
    invSigmaQx2PI = INV_PI * invSigmaQx2
    invThresholdSqx2 = f(0.5) / (threshold * threshold)
    invThresholdSqrt2PI = INV_SQRT_OF_2PI / threshold
    for gy in range(H):
        for gx in range(W):
            cr = f(ld(img, gx, gy, 0)) ** invGamma
            cg = f(ld(img, gx, gy, 1)) ** invGamma
            cb = f(ld(img, gx, gy, 2)) ** invGamma
            zBuff = f(0.0); aR = f(0.0); aG = f(0.0); aB = f(0.0)
            dx = -radius
            while dx <= radius:
                pt = f(math.sqrt(float(radQ - dx * dx)))
                dy = -pt
                while dy <= pt:
                    bf = f(math.exp(-float((dx * dx + dy * dy) * invSigmaQx2))) * invSigmaQx2PI
                    ptx = f(gx) + dx
                    pty = f(gy) + dy
                    ix = int(ptx)   # GLSL int() truncates toward zero
                    iy = int(pty)
                    wr = f(ld(img, ix, iy, 0))
                    wg = f(ld(img, ix, iy, 1))
                    wb = f(ld(img, ix, iy, 2))
                    dCr = wr ** invGamma - cr
                    dCg = wg ** invGamma - cg
                    dCb = wb ** invGamma - cb
                    dotdC = dCr * dCr + dCg * dCg + dCb * dCb
                    df = f(math.exp(-float(dotdC * invThresholdSqx2))) * invThresholdSqrt2PI * bf
                    zBuff += df
                    aR += df * wr; aG += df * wg; aB += df * wb
                    dy = f(dy + f(1.0))
                dx = f(dx + f(1.0))
            out[gy, gx, 0] = aR / zBuff
            out[gy, gx, 1] = aG / zBuff
            out[gy, gx, 2] = aB / zBuff
            out[gy, gx, 3] = f(1.0)
    return out


# ---------------------------------------------------------------------------
# 2) HQDN3D (HQDN3D_shader.h:28-99 + coef gen HQDN3D_vulkan.cpp:87-99)
# ---------------------------------------------------------------------------
LUT_LEN = 512 * 16


def build_hqdn3d_lut():
    # strengths (HQDN3D_vulkan.cpp:9-11,46-53)
    P1, P2, P3 = 4.0, 3.0, 6.0
    s_luma_sp = P1
    s_chroma_sp = P2 * s_luma_sp / P1
    s_luma_tmp = P3 * s_luma_sp / P1
    s_chroma_tmp = s_luma_tmp * s_chroma_sp / s_luma_sp
    strengths = [s_luma_sp, s_luma_tmp, s_chroma_sp, s_chroma_tmp]  # classes 0,1,2,3
    lut = np.zeros((4, LUT_LEN), dtype=np.int64)
    for cls in range(4):
        dist25 = min(strengths[cls], 252.0)
        gamma = math.log(0.25) / math.log(1.0 - dist25 / 255.0 - 0.00001)
        for i in range(-256 * 16, 256 * 16):
            ff = ((i * 32) + 16 - 1) / 512.0
            simil = max(0.0, 1.0 - abs(ff) / 255.0)
            Cv = (simil ** gamma) * 256.0 * ff
            # round-half-up to match Mojo Int(Cv+0.5) / Int(Cv-0.5)
            ci = int(Cv + 0.5) if Cv >= 0 else int(Cv - 0.5)
            lut[cls, 256 * 16 + i] = ci
    return lut


def rgb_to_yuv(r, g, b):
    y = 0.262700 * r + 0.678000 * g + 0.059300 * b
    u = 0.5 + (-0.139630 * r + -0.360370 * g + 0.500000 * b)
    v = 0.5 + (0.500000 * r + -0.459786 * g + -0.040214 * b)
    f = np.float32
    return (min(max(f(y), 0.0), 1.0), min(max(f(u), 0.0), 1.0), min(max(f(v), 0.0), 1.0))


def yuv_to_rgb(y, u, v):
    yy = y - 0.0; uu = u - 0.5; vv = v - 0.5
    r = 1.0 * yy + 0.0 * uu + 1.474600 * vv
    g = 1.0 * yy + -0.164553 * uu + -0.571353 * vv
    b = 1.0 * yy + 1.881400 * uu + 0.0 * vv
    f = np.float32
    return (min(max(f(r), 0.0), 1.0), min(max(f(g), 0.0), 1.0), min(max(f(b), 0.0), 1.0))


def trunc_div16(num):
    # GLSL '/' truncates toward zero
    return int(num / 16) if num >= 0 else -int((-num) // 16)


def lowpass(lut, prev, cur, cls):
    num = prev - cur
    dt = trunc_div16(num)
    idx = dt + 256 * 16
    idx = clampi(idx, 0, LUT_LEN - 1)   # matches Mojo guard; shader OOB-reads near-white
    return cur + int(lut[cls, idx])


def hqdn3d(img):
    lut = build_hqdn3d_lut()
    out = np.zeros((H, W, 4), dtype=np.float32)
    LUMA_SPATIAL, LUMA_TMP, CHROMA_SPATIAL, CHROMA_TMP = 0, 1, 2, 3
    for y in range(H):
        for x in range(W):
            r0 = float(ld(img, x, y, 0)); g0 = float(ld(img, x, y, 1))
            b0 = float(ld(img, x, y, 2)); a0 = float(ld(img, x, y, 3))
            yuv0 = rgb_to_yuv(r0, g0, b0)
            # frame buffers zero (single frame); writes don't affect this pixel's output
            # Y
            pa = lowpass(lut, 0, int(yuv0[0] * np.float32(65535.0)) + 128, LUMA_SPATIAL)
            pa = lowpass(lut, 0, pa, LUMA_SPATIAL)
            tmpv = lowpass(lut, 0, pa, LUMA_TMP)
            outY = np.float32(tmpv) / np.float32(65535.0)
            # U
            pa = lowpass(lut, 0, int(yuv0[1] * np.float32(65535.0)) + 128, CHROMA_SPATIAL)
            pa = lowpass(lut, 0, pa, CHROMA_SPATIAL)
            tmpu = lowpass(lut, 0, pa, CHROMA_TMP)
            outU = np.float32(tmpu) / np.float32(65535.0)
            # V
            pa = lowpass(lut, 0, int(yuv0[2] * np.float32(65535.0)) + 128, CHROMA_SPATIAL)
            pa = lowpass(lut, 0, pa, CHROMA_SPATIAL)
            tmpvv = lowpass(lut, 0, pa, CHROMA_TMP)
            outV = np.float32(tmpvv) / np.float32(65535.0)
            rgb = yuv_to_rgb(outY, outU, outV)
            out[y, x, 0] = rgb[0]; out[y, x, 1] = rgb[1]
            out[y, x, 2] = rgb[2]; out[y, x, 3] = np.float32(a0)
    return out


# ---------------------------------------------------------------------------
# 3) Guided filter (intended self-guided color guided filter)
# ---------------------------------------------------------------------------
GR = 4
GEPS = np.float32(1.0e-4)


def box_blur(field):
    # separable: H pass then V pass; per-pass weight 1/GR; anchor GR//2; clamp-edge.
    f = np.float32
    anchor = GR // 2
    wv = f(1.0) / f(GR)
    Hh, Ww, Ch = field.shape
    tmp = np.zeros_like(field)
    for y in range(Hh):
        for x in range(Ww):
            for ch in range(Ch):
                acc = f(0.0)
                for k in range(GR):
                    sx = clampi(x - anchor + k, 0, Ww - 1)
                    acc += wv * f(field[y, sx, ch])
                tmp[y, x, ch] = acc
    out = np.zeros_like(field)
    for y in range(Hh):
        for x in range(Ww):
            for ch in range(Ch):
                acc = f(0.0)
                for k in range(GR):
                    sy = clampi(y - anchor + k, 0, Hh - 1)
                    acc += wv * f(tmp[sy, x, ch])
                out[y, x, ch] = acc
    return out


def guided(img):
    f = np.float32
    # moments: mean_I (rgb), mean_II (rgb)
    mI = np.zeros((H, W, 4), dtype=np.float32)
    mII = np.zeros((H, W, 4), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            for ch in range(3):
                v = f(ld(img, x, y, ch))
                mI[y, x, ch] = v
                mII[y, x, ch] = v * v
            mI[y, x, 3] = f(ld(img, x, y, 3))
            mII[y, x, 3] = f(ld(img, x, y, 3))
    bmI = box_blur(mI)
    bmII = box_blur(mII)
    out = np.zeros((H, W, 4), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            for ch in range(3):
                mi = f(bmI[y, x, ch])
                mii = f(bmII[y, x, ch])
                varI = mii - mi * mi
                a = varI / (varI + GEPS)
                b = mi * (f(1.0) - a)
                I = f(ld(img, x, y, ch))
                out[y, x, ch] = min(max(a * I + b, 0.0), 1.0)
            out[y, x, 3] = f(ld(img, x, y, 3))
    return out


# ---------------------------------------------------------------------------
# 4) Deband (DeBand_shader.h:27-97 blur=false; pos gen DeBand_vulkan.cpp:5-10,54-70)
# ---------------------------------------------------------------------------
def build_deband_pos():
    f = np.float32
    direction = f(2.0) * f(math.pi)   # direction(=2)*PI
    rng = f(16.0)
    xpos = np.zeros((H, W), dtype=np.int64)
    ypos = np.zeros((H, W), dtype=np.int64)
    for y in range(H):
        for x in range(W):
            arg = f(x) * f(12.9898) + f(y) * f(78.233)
            r = f(math.sin(float(arg))) * f(43758.545)
            fr = r - f(math.floor(float(r)))
            dirv = fr * direction
            dist = int(fr * rng)               # truncate toward zero
            xv = int(f(math.cos(float(dirv))) * f(dist))
            yv = int(f(math.sin(float(dirv))) * f(dist))
            xpos[y, x] = xv
            ypos[y, x] = yv
    return xpos, ypos


def deband(img, threshold=0.01):
    f = np.float32
    threshold = f(threshold)
    xpos, ypos = build_deband_pos()
    out = np.zeros((H, W, 4), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            xp = int(xpos[y, x]); yp = int(ypos[y, x])
            x_r = clampi(x + xp, 0, W - 1)
            y_p = clampi(y + yp, 0, H - 1)
            y_m = clampi(y - yp, 0, H - 1)
            x_l = clampi(x - xp, 0, W - 1)
            for ch in range(3):
                ref0 = f(ld(img, x_r, y_p, ch))
                ref1 = f(ld(img, x_r, y_m, ch))
                ref2 = f(ld(img, x_l, y_m, ch))
                ref3 = f(ld(img, x_l, y_p, ch))
                src0 = f(ld(img, x, y, ch))
                avg = (ref0 + ref1 + ref2 + ref3) / f(4.0)
                res = src0
                if (abs(src0 - ref0) < threshold and abs(src0 - ref1) < threshold
                        and abs(src0 - ref2) < threshold and abs(src0 - ref3) < threshold):
                    res = avg
                out[y, x, ch] = res
            out[y, x, 3] = f(ld(img, x, y, 3))
    return out


# ---------------------------------------------------------------------------
def to_u8_tokens(img):
    # Match Mojo f32_to_u8: arithmetic in float32, then truncate (floor for >=0), clamp.
    a = img.astype(np.float32)
    scaled = a * np.float32(255.0) + np.float32(0.5)
    u8 = np.clip(np.floor(scaled.astype(np.float64)), 0, 255).astype(np.int64)
    return u8.reshape(-1)


def main():
    img = build_test_image()
    sd = smartdenoise(img)
    hq = hqdn3d(img)
    gd = guided(img)
    db = deband(img)
    toks = np.concatenate([to_u8_tokens(sd), to_u8_tokens(hq),
                           to_u8_tokens(gd), to_u8_tokens(db)])
    with open("/tmp/shader_denoise_ref.txt", "w") as fo:
        fo.write(" ".join(str(int(t)) for t in toks))
    print("denoise ref: wrote /tmp/shader_denoise_ref.txt (%d tokens)" % len(toks))


if __name__ == "__main__":
    main()
