#!/usr/bin/env python3
# Numpy CPU reference for group "effects": WaterRipple, Soul, Sway, Star, Lighting, Jitter.
# Builds the IDENTICAL 128x128 RGBA f32 test image as shader_effects.mojo, applies the SAME
# algorithm (ported from the GLSL shaders), writes flattened uint8 RGBA to
# /tmp/shader_effects_ref.txt (concatenation of all six effect outputs, same order as the Mojo file).
#
# All math in float32 to mirror the GPU f32 compute. GLSL helpers (fract/mod/floor/smoothstep)
# reproduced exactly. GLSL int(f) truncates toward zero (sample coords >= 0 after uv clamps, so
# trunc == floor). load_rgba clamps integer sample coords to [0,w-1]/[0,h-1].
import numpy as np

W = 128
H = 128
f32 = np.float32

PI = f32(3.1415)        # Star
PI_L = f32(3.1415926)   # Lighting


def build_test_image():
    img = np.zeros((H, W, 4), dtype=f32)
    for y in range(H):
        for x in range(W):
            diag = f32(f32(x + y) / f32(W + H - 2))
            r = diag
            g = f32(f32(x) / f32(W - 1))
            b = f32(f32(y) / f32(H - 1))
            if 20 <= x < 50 and 20 <= y < 50:
                r = f32(0.95); g = f32(0.9); b = f32(0.1)
            ddx = f32(x - 90); ddy = f32(y - 90)
            if f32(ddx * ddx + ddy * ddy) < f32(18 * 18):
                r = f32(0.1); g = f32(0.85); b = f32(0.95)
            img[y, x, 0] = min(max(r, f32(0.0)), f32(1.0))
            img[y, x, 1] = min(max(g, f32(0.0)), f32(1.0))
            img[y, x, 2] = min(max(b, f32(0.0)), f32(1.0))
            img[y, x, 3] = f32(1.0)
    return img


# ---- GLSL helpers (scalar, float32) ----
def fclamp(x, lo, hi):
    return f32(min(max(f32(x), f32(lo)), f32(hi)))


def gfloor(x):
    x = f32(x)
    i = f32(int(x))  # toward zero
    if i > x:
        i = f32(i - f32(1.0))
    return i


def gfract(x):
    x = f32(x)
    return f32(x - gfloor(x))


def gmod(x, y):
    x = f32(x); y = f32(y)
    return f32(x - y * gfloor(f32(x / y)))


def gsmoothstep(e0, e1, x):
    e0 = f32(e0); e1 = f32(e1); x = f32(x)
    t = fclamp(f32(f32(x - e0) / f32(e1 - e0)), 0.0, 1.0)
    return f32(t * t * f32(f32(3.0) - f32(2.0) * t))


def trunc_coord(fv):
    return int(f32(fv))  # toward zero; fv>=0 here


def sample_clamped(img, x, y, ch):
    cx = 0 if x < 0 else (W - 1 if x > W - 1 else x)
    cy = 0 if y < 0 else (H - 1 if y > H - 1 else y)
    return f32(img[cy, cx, ch])


# ===========================================================================
# 1) WaterRipple (WaterRipple_shader.h:27-40)
# ===========================================================================
def waterripple(img, freq, amount):
    freq = f32(freq); amount = f32(amount)
    out = np.zeros_like(img)
    for y in range(H):
        for x in range(W):
            uvx = f32(f32(x) / f32(W - 1))
            uvy = f32(f32(y) / f32(H - 1))
            dirx = f32(uvx - f32(0.5))
            diry = f32(uvy - f32(0.5))
            ln = f32(np.sqrt(f32(dirx * dirx + diry * diry), dtype=f32))
            s = f32(np.sin(f32(ln * freq), dtype=f32))
            if s < 0.0:
                s = f32(-s)
            fac = f32(f32(amount * s) / ln)
            nux = f32(uvx + dirx * fac)
            nuy = f32(uvy + diry * fac)
            sx = trunc_coord(f32(nux * f32(W - 1)))
            sy = trunc_coord(f32(nuy * f32(H - 1)))
            for ch in range(4):
                out[y, x, ch] = sample_clamped(img, sx, sy, ch)
    return out


# ===========================================================================
# 2) Soul (Soul_shader.h:30-48)
# ===========================================================================
def soul(img, progress_in, count, max_scale, max_alpha, shrink):
    progress_in = f32(progress_in); max_scale = f32(max_scale); max_alpha = f32(max_alpha)
    out = np.zeros_like(img)
    duration = f32(f32(1.0) / f32(count))
    for y in range(H):
        for x in range(W):
            uvx = f32(f32(x) / f32(W - 1))
            uvy = f32(f32(y) / f32(H - 1))
            progress = f32(gmod(progress_in, duration) / duration)
            if shrink == 1 and progress > 0.5:
                progress = f32(f32(1.0) - progress)
            alpha = f32(max_alpha * f32(f32(1.0) - progress))
            scale = f32(f32(1.0) + f32(max_scale - f32(1.0)) * progress)
            weakX = f32(f32(0.5) + f32(uvx - f32(0.5)) / scale)
            weakY = f32(f32(0.5) + f32(uvy - f32(0.5)) / scale)
            weakX = fclamp(weakX, 0.0, 1.0)
            weakY = fclamp(weakY, 0.0, 1.0)
            wsx = trunc_coord(f32(weakX * f32(W - 1)))
            wsy = trunc_coord(f32(weakY * f32(H - 1)))
            msx = trunc_coord(f32(uvx * f32(W - 1)))
            msy = trunc_coord(f32(uvy * f32(H - 1)))
            for ch in range(4):
                a = sample_clamped(img, wsx, wsy, ch)
                b = sample_clamped(img, msx, msy, ch)
                out[y, x, ch] = f32(a * f32(f32(1.0) - alpha) + b * alpha)
    return out


# ===========================================================================
# 3) Sway (Sway_shader.h:29-40). shader 'speed' arg = time*m_speed
# ===========================================================================
def sway(img, speed, strength, density, horizontal):
    speed = f32(speed); strength = f32(strength); density = f32(density)
    out = np.zeros_like(img)
    for y in range(H):
        for x in range(W):
            uvx = f32(f32(x) / f32(W - 1))
            uvy = f32(f32(y) / f32(H - 1))
            waveu = f32(f32(np.sin(f32(f32(uvy + speed) * density), dtype=f32)) * strength / f32(1000.0))
            if horizontal == 1:
                uvx = f32(uvx + waveu)
            else:
                uvy = f32(uvy + waveu)
            uvx = fclamp(uvx, 0.0, 1.0)
            uvy = fclamp(uvy, 0.0, 1.0)
            sx = trunc_coord(f32(uvx * f32(W - 1)))
            sy = trunc_coord(f32(uvy * f32(H - 1)))
            for ch in range(4):
                out[y, x, ch] = sample_clamped(img, sx, sy, ch)
    return out


# ===========================================================================
# 4) Star (Star_shader.h:31-131)
# ===========================================================================
def star_pseudo_random(px_in, py_in):
    px = gfract(f32(px_in * f32(123.45)))
    py = gfract(f32(py_in * f32(345.67)))
    d = f32(f32(px * px) + f32(py * py) + f32(45.32) * f32(px + py))
    px = f32(px + d)
    py = f32(py + d)
    return gfract(f32(px * py))


def star_core(uvx, uvy, flaresize, rotAngle):
    uvx = f32(uvx); uvy = f32(uvy); flaresize = f32(flaresize); rotAngle = f32(rotAngle)
    d = f32(np.sqrt(f32(uvx * uvx + uvy * uvy), dtype=f32))
    starcore = f32(f32(0.05) / d)
    ang = f32(f32(-2.0) * PI * rotAngle)
    c = f32(np.cos(ang, dtype=f32)); s = f32(np.sin(ang, dtype=f32))
    nx = f32(c * uvx - s * uvy); ny = f32(s * uvx + c * uvy)
    uvx = nx; uvy = ny
    flareMax = f32(1.0)
    f1 = f32(uvx * uvy * f32(3000.0))
    if f1 < 0.0:
        f1 = f32(-f1)
    starflares = f32(flareMax - f1)
    if starflares < 0.0:
        starflares = f32(0.0)
    starcore = f32(starcore + starflares * flaresize)
    ang2 = f32(PI * f32(0.25))
    c2 = f32(np.cos(ang2, dtype=f32)); s2 = f32(np.sin(ang2, dtype=f32))
    nx2 = f32(c2 * uvx - s2 * uvy); ny2 = f32(s2 * uvx + c2 * uvy)
    uvx = nx2; uvy = ny2
    f2 = f32(uvx * uvy * f32(3000.0))
    if f2 < 0.0:
        f2 = f32(-f2)
    starflares2 = f32(flareMax - f2)
    if starflares2 < 0.0:
        starflares2 = f32(0.0)
    starcore = f32(starcore + starflares2 * f32(0.3) * flaresize)
    starcore = f32(starcore * gsmoothstep(1.0, 0.05, d))
    return starcore


def star_field_layer(uvx, uvy, rotAngle, progress, pr, pg, pb, layers):
    gvx = f32(gfract(uvx) - f32(0.5))
    gvy = f32(gfract(uvy) - f32(0.5))
    idx = gfloor(uvx)
    idy = gfloor(uvy)
    deltaTimeTwinkle = f32(progress)
    cr = f32(0.0); cg = f32(0.0); cb = f32(0.0)
    for yy in range(-1, 2):
        for xx in range(-1, 2):
            offx = f32(xx); offy = f32(yy)
            randomN = star_pseudo_random(f32(idx + offx), f32(idy + offy))
            randoX = f32(randomN - f32(0.5))
            randoY = f32(gfract(f32(randomN * f32(45.0))) - f32(0.5))
            rpx = f32(gvx - offx - randoX)
            rpy = f32(gvy - offy - randoY)
            size = gfract(f32(randomN * f32(1356.33)))
            flareSwitch = gsmoothstep(0.9, 1.0, size)
            star = star_core(rpx, rpy, flareSwitch, rotAngle)
            randomStarColorSeed = f32(gfract(f32(randomN * f32(2150.0))) * f32(f32(3.0) * PI) * deltaTimeTwinkle)
            colr = f32(np.sin(f32(pr * randomStarColorSeed), dtype=f32))
            colg = f32(np.sin(f32(pg * randomStarColorSeed), dtype=f32))
            colb = f32(np.sin(f32(pb * randomStarColorSeed), dtype=f32))
            m = f32(f32(0.4) * f32(np.sin(deltaTimeTwinkle, dtype=f32)))
            colr = f32(colr * m + f32(0.6))
            colg = f32(colg * m + f32(0.6))
            colb = f32(colb * m + f32(0.6))
            colr = f32(colr * pr)
            colg = f32(colg * pg)
            colb = f32(colb * f32(pb + size))
            dimByDensity = f32(f32(15.0) / layers)
            cr = f32(cr + star * size * colr * dimByDensity)
            cg = f32(cg + star * size * colg * dimByDensity)
            cb = f32(cb + star * size * colb * dimByDensity)
    return cr, cg, cb


def star(img, progress, speed, layers, red, green, blue):
    progress = f32(progress); speed = f32(speed); layers = f32(layers)
    red = f32(red); green = f32(green); blue = f32(blue)
    out = np.zeros_like(img)
    MIN_DIVIDE = f32(64.0)
    MAX_DIVIDE = f32(0.01)
    deltaTime = f32(progress * speed * f32(0.01))
    rotAngle = f32(progress * speed * f32(0.09))
    stepi = f32(f32(1.0) / layers)
    for y in range(H):
        for x in range(W):
            tuvx = f32(f32(x) / f32(W - 1))
            tuvy = f32(f32(y) / f32(H - 1))
            col_r = f32(0.0); col_g = f32(0.0); col_b = f32(0.0)
            i = f32(0.0)
            while i < f32(1.0):
                layerDepth = gfract(f32(i + deltaTime))
                layerScale = f32(MIN_DIVIDE + f32(MAX_DIVIDE - MIN_DIVIDE) * layerDepth)
                layerFader = f32(layerDepth * gsmoothstep(0.1, 1.1, layerDepth))
                layerOffset = f32(i * f32(f32(3430.00) + gfract(i)))
                ang = f32(rotAngle * i * f32(-10.0))
                c = f32(np.cos(ang, dtype=f32)); s = f32(np.sin(ang, dtype=f32))
                ntx = f32(c * tuvx - s * tuvy); nty = f32(s * tuvx + c * tuvy)
                tuvx = ntx; tuvy = nty
                sfx = f32(tuvx * layerScale + layerOffset)
                sfy = f32(tuvy * layerScale + layerOffset)
                lr, lg, lb = star_field_layer(sfx, sfy, rotAngle, progress, red, green, blue, layers)
                col_r = f32(col_r + lr * layerFader)
                col_g = f32(col_g + lg * layerFader)
                col_b = f32(col_b + lb * layerFader)
                i = f32(i + stepi)
            out[y, x, 0] = f32(img[y, x, 0] + col_r)
            out[y, x, 1] = f32(img[y, x, 1] + col_g)
            out[y, x, 2] = f32(img[y, x, 2] + col_b)
            out[y, x, 3] = img[y, x, 3]
    return out


# ===========================================================================
# 5) Lighting (Lighting_shader.h:29-167)
# ===========================================================================
def light_rgb2hsv(r, g, b):
    Kx = f32(0.0); Ky = f32(-1.0 / 3.0); Kz = f32(2.0 / 3.0); Kw = f32(-1.0)
    if g >= b:
        px, py, pz, pw = f32(g), f32(b), Kx, Ky
    else:
        px, py, pz, pw = f32(b), f32(g), Kw, Kz
    if r >= px:
        qx, qy, qz, qw = f32(r), py, pz, px
    else:
        qx, qy, qz, qw = px, py, pw, f32(r)
    d = f32(qx - min(qw, qy))
    e = f32(1.0e-10)
    hh = f32(qz + f32(f32(qw - qy) / f32(f32(6.0) * d + e)))
    if hh < 0.0:
        hh = f32(-hh)
    return hh, f32(d / f32(qx + e)), qx


def light_hsv2rgb(h, s, v):
    Kx = f32(1.0); Ky = f32(2.0 / 3.0); Kz = f32(1.0 / 3.0); Kw = f32(3.0)
    px = f32(gfract(f32(h + Kx)) * f32(6.0) - Kw)
    py = f32(gfract(f32(h + Ky)) * f32(6.0) - Kw)
    pz = f32(gfract(f32(h + Kz)) * f32(6.0) - Kw)
    if px < 0.0:
        px = f32(-px)
    if py < 0.0:
        py = f32(-py)
    if pz < 0.0:
        pz = f32(-pz)
    cr = fclamp(f32(px - Kx), 0.0, 1.0)
    cg = fclamp(f32(py - Kx), 0.0, 1.0)
    cb = fclamp(f32(pz - Kx), 0.0, 1.0)
    rr = f32(v * f32(Kx * f32(f32(1.0) - s) + cr * s))
    gg = f32(v * f32(Kx * f32(f32(1.0) - s) + cg * s))
    bb = f32(v * f32(Kx * f32(f32(1.0) - s) + cb * s))
    return rr, gg, bb


def light_hue2rgb(hue):
    a0 = f32(hue * f32(6.0) - f32(3.0))
    a1 = f32(hue * f32(6.0) - f32(2.0))
    a2 = f32(hue * f32(6.0) - f32(4.0))
    if a0 < 0.0:
        a0 = f32(-a0)
    if a1 < 0.0:
        a1 = f32(-a1)
    if a2 < 0.0:
        a2 = f32(-a2)
    rr = f32(a0 * f32(1.0) + f32(-1.0))
    gg = f32(a1 * f32(-1.0) + f32(2.0))
    bb = f32(a2 * f32(-1.0) + f32(2.0))
    return fclamp(rr, 0.0, 1.0), fclamp(gg, 0.0, 1.0), fclamp(bb, 0.0, 1.0)


def light_rgb2hcv(r, g, b):
    EPS = f32(1.0e-10)
    if g < b:
        px, py, pz, pw = f32(b), f32(g), f32(-1.0), f32(2.0 / 3.0)
    else:
        px, py, pz, pw = f32(g), f32(b), f32(0.0), f32(-1.0 / 3.0)
    if r < px:
        qx, qy, qz, qw = px, py, pw, f32(r)
    else:
        qx, qy, qz, qw = f32(r), py, pz, px
    cc = f32(qx - min(qw, qy))
    hnum = f32(f32(f32(qw - qy) / f32(f32(6.0) * cc + EPS)) + qz)
    if hnum < 0.0:
        hnum = f32(-hnum)
    return hnum, cc, qx


def light_rgb2hsl(r, g, b):
    EPS = f32(1.0e-10)
    hcv_h, hcv_c, hcv_v = light_rgb2hcv(r, g, b)
    z = f32(hcv_v - hcv_c * f32(0.5))
    az = f32(f32(2.0) * z - f32(1.0))
    if az < 0.0:
        az = f32(-az)
    den = f32(f32(1.0) - az + EPS)
    s = f32(hcv_c / den)
    return hcv_h, s, z


def light_hsl2rgb(h, s, z):
    rr, gg, bb = light_hue2rgb(h)
    az = f32(f32(2.0) * z - f32(1.0))
    if az < 0.0:
        az = f32(-az)
    c = f32(f32(f32(1.0) - az) * s)
    return (f32(f32(rr - f32(0.5)) * c + z),
            f32(f32(gg - f32(0.5)) * c + z),
            f32(f32(bb - f32(0.5)) * c + z))


def lighting(img, progress_in, count, saturation, light):
    progress_in = f32(progress_in); saturation = f32(saturation); light = f32(light)
    out = np.zeros_like(img)
    duration = f32(f32(1.0) / f32(count))
    for y in range(H):
        for x in range(W):
            progress = f32(gmod(progress_in, duration) / duration)
            amp = f32(np.sin(f32(progress * f32(PI_L / duration)), dtype=f32))
            if amp < 0.0:
                amp = f32(-amp)
            hue = f32(amp * f32(360.0) / f32(360.0))
            value = f32(saturation / f32(10.0))
            r = f32(img[y, x, 0]); g = f32(img[y, x, 1]); b = f32(img[y, x, 2]); a = f32(img[y, x, 3])
            fh, fs, fv = light_rgb2hsv(r, g, b)
            fh = f32(fh + hue)
            fv = f32(fv + value)
            fv = f32(max(min(fv, f32(1.0)), f32(0.0)))
            mr, mg, mb = light_hsv2rgb(fh, fs, fv)
            lh, ls, lz = light_rgb2hsl(mr, mg, mb)
            ls = f32(ls + saturation * f32(0.5))
            ls = f32(max(min(ls, f32(1.0)), f32(0.0)))
            rr, rg, rb = light_hsl2rgb(lh, ls, lz)
            out[y, x, 0] = f32(rr * f32(f32(1.0) - light) + f32(1.0) * light)
            out[y, x, 1] = f32(rg * f32(f32(1.0) - light) + f32(1.0) * light)
            out[y, x, 2] = f32(rb * f32(f32(1.0) - light) + f32(1.0) * light)
            out[y, x, 3] = a
    return out


# ===========================================================================
# 6) Jitter (Jitter_shader.h:29-53)
# ===========================================================================
def jitter(img, progress_in, count, max_scale, offset, shrink):
    progress_in = f32(progress_in); max_scale = f32(max_scale); offset = f32(offset)
    out = np.zeros_like(img)
    duration = f32(f32(1.0) / f32(count))
    for y in range(H):
        for x in range(W):
            uvx = f32(f32(x) / f32(W - 1))
            uvy = f32(f32(y) / f32(H - 1))
            progress = f32(gmod(progress_in, duration) / duration)
            if shrink == 1 and progress > 0.5:
                progress = f32(f32(1.0) - progress)
            offc = f32(offset * progress)
            scale = f32(f32(1.0) + f32(max_scale - f32(1.0)) * progress)
            scx = fclamp(f32(f32(0.5) + f32(uvx - f32(0.5)) / scale), 0.0, 1.0)
            scy = fclamp(f32(f32(0.5) + f32(uvy - f32(0.5)) / scale), 0.0, 1.0)
            rx = fclamp(f32(scx + offc), 0.0, 1.0)
            ry = fclamp(f32(scy + offc), 0.0, 1.0)
            bx = fclamp(f32(scx - offc), 0.0, 1.0)
            by = fclamp(f32(scy - offc), 0.0, 1.0)
            rsx = trunc_coord(f32(rx * f32(W - 1)))
            rsy = trunc_coord(f32(ry * f32(H - 1)))
            bsx = trunc_coord(f32(bx * f32(W - 1)))
            bsy = trunc_coord(f32(by * f32(H - 1)))
            msx = trunc_coord(f32(scx * f32(W - 1)))
            msy = trunc_coord(f32(scy * f32(H - 1)))
            out[y, x, 0] = sample_clamped(img, rsx, rsy, 0)
            out[y, x, 1] = sample_clamped(img, msx, msy, 1)
            out[y, x, 2] = sample_clamped(img, bsx, bsy, 2)
            out[y, x, 3] = sample_clamped(img, msx, msy, 3)
    return out


def to_u8_tokens(img):
    toks = []
    for y in range(H):
        for x in range(W):
            for ch in range(4):
                v = f32(img[y, x, ch])
                # Non-finite (Star's 0.05/d at d==0, identical on GPU): saturate to 255.
                if not np.isfinite(v):
                    toks.append("255")
                    continue
                t = f32(v * f32(255.0) + f32(0.5))
                if t < 0.0:
                    t = f32(0.0)
                if t > 255.0:
                    t = f32(255.0)
                toks.append(str(int(t)))
    return toks


def main():
    img = build_test_image()

    # input dump
    with open("/tmp/shader_effects_in.txt", "w") as fh:
        fh.write(" ".join(to_u8_tokens(img)))

    toks = []
    toks += to_u8_tokens(waterripple(img, 24.0, 0.03))
    toks += to_u8_tokens(soul(img, 0.37, 1, 1.8, 0.4, 0))
    toks += to_u8_tokens(sway(img, 7.4, 20.0, 20.0, 1))
    toks += to_u8_tokens(star(img, 0.37, 10.0, 2.0, 1.0, 0.1, 0.9))
    toks += to_u8_tokens(lighting(img, 0.37, 1, 0.3, 0.3))
    toks += to_u8_tokens(jitter(img, 0.37, 1, 1.1, 0.02, 0))

    with open("/tmp/shader_effects_ref.txt", "w") as fh:
        fh.write(" ".join(toks))
    print("effects ref: wrote /tmp/shader_effects_ref.txt + input")


if __name__ == "__main__":
    main()
