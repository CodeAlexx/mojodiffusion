# Group "color_adv": ColorBalance, CAS, USM, Lut3D
# Mojo GPU port of MediaEditor GLSL filters + standard numpy-defined filters.
# Build/run (from /home/alex/mojodiffusion): pixi run mojo run -I . shader_color_adv.mojo
#
# Source citations:
#   ColorBalance: /tmp/Med/plugin/nodes/filters/ColorBalance/ColorBalance_shader.h:36-129
#   CAS:          /tmp/Med/plugin/nodes/filters/CAS/CAS_shader.h:24-110
#   USM:          /tmp/Med/plugin/nodes/filters/USM/ImMatUSMNode.cpp:53 (filter(sigma,amount,threshold)); standard unsharp mask
#   Lut3D:        /tmp/Med/plugin/nodes/filters/Lut3D/ImMatLut3DNode.cpp (trilinear over rgbvec LUT, B-fastest); identity LUT => pass-through

from std.math import ceildiv, sqrt, exp, fmod
from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

comptime dtype = DType.float32
comptime W = 128
comptime H = 128
comptime C = 4
comptime img_layout = Layout.row_major(H, W, C)
comptime N = H * W * C

# Gaussian blur intermediate (RGBA), used by USM
comptime BLOCK = 16

# Identity LUT for Lut3D test: lutsize x lutsize x lutsize, 4 channels (rgba), B-fastest.
comptime LUTSIZE = 17
comptime lut_layout = Layout.row_major(LUTSIZE * LUTSIZE * LUTSIZE * 4)


# ---------------------------------------------------------------------------
# Helpers (device, f32) — clamp / min / max
# ---------------------------------------------------------------------------
fn fclamp(x: Float32, lo: Float32, hi: Float32) -> Float32:
    var v = x
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return v


fn fmin3(a: Float32, b: Float32, c: Float32) -> Float32:
    return min(min(a, b), c)


fn fmax3(a: Float32, b: Float32, c: Float32) -> Float32:
    return max(max(a, b), c)


# ===========================================================================
# 1) ColorBalance  (ColorBalance_shader.h:36-129)
# ===========================================================================
# hfun  (ColorBalance_shader.h:38-44)
fn cb_hfun(n: Float32, h: Float32, s: Float32, l: Float32) -> Float32:
    var a = s * min(l, Float32(1.0) - l)
    var k = fmod(n + h / Float32(30.0), Float32(12.0))
    return fclamp(
        l - a * max(min(k - Float32(3.0), min(Float32(9.0) - k, Float32(1.0))), Float32(-1.0)),
        Float32(0.0),
        Float32(1.0),
    )


# get_component  (ColorBalance_shader.h:93-106)
fn cb_get_component(
    vin: Float32, l: Float32, s_in: Float32, m_in: Float32, h_in: Float32
) -> Float32:
    var a = Float32(4.0)
    var b = Float32(0.333)
    var scale = Float32(0.7)
    var s = s_in
    var m = m_in
    var h = h_in
    s *= fclamp((b - l) * a + Float32(0.5), 0.0, 1.0) * scale
    m *= (
        fclamp((l - b) * a + Float32(0.5), 0.0, 1.0)
        * fclamp((Float32(1.0) - l - b) * a + Float32(0.5), 0.0, 1.0)
        * scale
    )
    h *= fclamp((l + b - Float32(1.0)) * a + Float32(0.5), 0.0, 1.0) * scale
    var v = vin
    v += s
    v += m
    v += h
    return fclamp(v, 0.0, 1.0)


# balance() with preserve_lightness == 0  (ColorBalance_shader.h:108-118)
# preservel path (preserve_lightness==1) is implemented in numpy ref but the
# GPU test runs with preserve_lightness=0 for a clean exact match.
def colorbalance_kernel(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    out: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    rs: Float32, gs: Float32, bs: Float32,
    rm: Float32, gm: Float32, bm: Float32,
    rh: Float32, gh: Float32, bh: Float32,
):
    var gx = global_idx.x
    var gy = global_idx.y
    if gx >= UInt(W) or gy >= UInt(H):
        return
    var r = rebind[Scalar[dtype]](inp[gy, gx, 0])
    var g = rebind[Scalar[dtype]](inp[gy, gx, 1])
    var b = rebind[Scalar[dtype]](inp[gy, gx, 2])
    var a = rebind[Scalar[dtype]](inp[gy, gx, 3])
    # l = max(rgb) + min(rgb)   (ColorBalance_shader.h:111)
    var l = fmax3(r, g, b) + fmin3(r, g, b)
    var vr = cb_get_component(r, l, rs, rm, rh)
    var vg = cb_get_component(g, l, gs, gm, gh)
    var vb = cb_get_component(b, l, bs, bm, bh)
    out[gy, gx, 0] = fclamp(vr, 0.0, 1.0)
    out[gy, gx, 1] = fclamp(vg, 0.0, 1.0)
    out[gy, gx, 2] = fclamp(vb, 0.0, 1.0)
    out[gy, gx, 3] = a


# ===========================================================================
# 2) CAS  (CAS_shader.h:24-109)
# ===========================================================================
fn cas_load(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin], x: Int, y: Int, ch: Int
) -> Float32:
    # caller clamps x,y to [0,W-1]/[0,H-1]
    return rebind[Scalar[dtype]](inp[y, x, ch])


def cas_kernel(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    out: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    strength: Float32,
):
    var gx = global_idx.x
    var gy = global_idx.y
    if gx >= UInt(W) or gy >= UInt(H):
        return
    var x = Int(gx)
    var y = Int(gy)
    # neighbor coords with clamp (CAS_shader.h:31-34)
    var y0 = max(y - 1, 0)
    var y1 = min(y + 1, H - 1)
    var x0 = max(x - 1, 0)
    var x1 = min(x + 1, W - 1)
    var cur_a = rebind[Scalar[dtype]](inp[y, x, 3])

    # per channel CAS (CAS_shader.h:49-93)
    for ch in range(3):
        var av = cas_load(inp, x0, y0, ch)
        var bv = cas_load(inp, x,  y0, ch)
        var cv = cas_load(inp, x1, y0, ch)
        var dv = cas_load(inp, x0, y,  ch)
        var ev = cas_load(inp, x,  y,  ch)
        var fv = cas_load(inp, x1, y,  ch)
        var gv = cas_load(inp, x0, y1, ch)
        var hv = cas_load(inp, x,  y1, ch)
        var iv = cas_load(inp, x1, y1, ch)

        var mn = fmin3(fmin3(dv, ev, fv), bv, hv)
        var mn2 = fmin3(fmin3(mn, av, cv), gv, iv)
        mn = mn + mn2

        var mx = fmax3(fmax3(dv, ev, fv), bv, hv)
        var mx2 = fmax3(fmax3(mx, av, cv), gv, iv)
        mx = mx + mx2

        var amp = sqrt(fclamp(min(mn, Float32(2.0) - mx) / mx, 0.0, 1.0))
        var weight = amp / strength
        var res = ((bv + dv + fv + hv) * weight + ev) / (Float32(1.0) + Float32(4.0) * weight)
        out[y, x, ch] = res
    out[y, x, 3] = cur_a


# ===========================================================================
# 3) USM  (standard unsharp mask; ImMatUSMNode.cpp:53)
#    blur = gaussian(sigma); result = orig + amount*(orig-blur) gated by threshold
# ===========================================================================
# We precompute a separable gaussian on GPU into a blur buffer, then combine.
comptime KR = 7  # gaussian radius (sigma=3 => radius ~ 3*sigma capped to 7 for 15-tap; matches numpy ref)
comptime KW = 2 * KR + 1

# Pass 1: horizontal blur (reads inp, writes tmp). Weights passed as a flat buffer.
comptime kern_layout = Layout.row_major(KW)


def usm_blur_h_kernel(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    tmp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    kern: LayoutTensor[dtype, kern_layout, MutAnyOrigin],
):
    var gx = global_idx.x
    var gy = global_idx.y
    if gx >= UInt(W) or gy >= UInt(H):
        return
    var x = Int(gx)
    var y = Int(gy)
    for ch in range(4):
        var acc = Float32(0.0)
        for t in range(KW):
            var dx = t - KR
            var sx = x + dx
            if sx < 0:
                sx = 0
            if sx > W - 1:
                sx = W - 1
            var wv = rebind[Scalar[dtype]](kern[t])
            acc += wv * rebind[Scalar[dtype]](inp[y, sx, ch])
        tmp[y, x, ch] = acc


# Pass 2: vertical blur (reads tmp, writes blur)
def usm_blur_v_kernel(
    tmp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    blur: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    kern: LayoutTensor[dtype, kern_layout, MutAnyOrigin],
):
    var gx = global_idx.x
    var gy = global_idx.y
    if gx >= UInt(W) or gy >= UInt(H):
        return
    var x = Int(gx)
    var y = Int(gy)
    for ch in range(4):
        var acc = Float32(0.0)
        for t in range(KW):
            var dy = t - KR
            var sy = y + dy
            if sy < 0:
                sy = 0
            if sy > H - 1:
                sy = H - 1
            var wv = rebind[Scalar[dtype]](kern[t])
            acc += wv * rebind[Scalar[dtype]](tmp[sy, x, ch])
        blur[y, x, ch] = acc


# Pass 3: combine. result = orig + amount*(orig-blur) when |orig-blur| (luma) > threshold contribution.
# Standard USM threshold gates the high-pass per channel: if |diff| < threshold, keep original.
def usm_combine_kernel(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    blur: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    out: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    amount: Float32,
    threshold: Float32,
):
    var gx = global_idx.x
    var gy = global_idx.y
    if gx >= UInt(W) or gy >= UInt(H):
        return
    var x = Int(gx)
    var y = Int(gy)
    for ch in range(3):
        var orig = rebind[Scalar[dtype]](inp[y, x, ch])
        var bl = rebind[Scalar[dtype]](blur[y, x, ch])
        var diff = orig - bl
        var ad = diff
        if ad < 0.0:
            ad = -ad
        var res = orig
        if ad >= threshold:
            res = orig + amount * diff
        out[y, x, ch] = fclamp(res, 0.0, 1.0)
    out[y, x, 3] = rebind[Scalar[dtype]](inp[y, x, 3])


# ===========================================================================
# 4) Lut3D  (trilinear over identity LUT => pass-through; ImMatLut3DNode.cpp)
#    LUT indexed [r][g][b], B fastest:  idx = ((ri*LUTSIZE)+gi)*LUTSIZE+bi
# ===========================================================================
def lut3d_kernel(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    lut: LayoutTensor[dtype, lut_layout, MutAnyOrigin],
    out: LayoutTensor[dtype, img_layout, MutAnyOrigin],
):
    var gx = global_idx.x
    var gy = global_idx.y
    if gx >= UInt(W) or gy >= UInt(H):
        return
    var x = Int(gx)
    var y = Int(gy)
    var r = fclamp(rebind[Scalar[dtype]](inp[y, x, 0]), 0.0, 1.0)
    var g = fclamp(rebind[Scalar[dtype]](inp[y, x, 1]), 0.0, 1.0)
    var b = fclamp(rebind[Scalar[dtype]](inp[y, x, 2]), 0.0, 1.0)
    var a = rebind[Scalar[dtype]](inp[y, x, 3])

    var scale = Float32(LUTSIZE - 1)
    var fr = r * scale
    var fg = g * scale
    var fb = b * scale
    var r0 = Int(fr)
    var g0 = Int(fg)
    var b0 = Int(fb)
    if r0 > LUTSIZE - 2:
        r0 = LUTSIZE - 2
    if g0 > LUTSIZE - 2:
        g0 = LUTSIZE - 2
    if b0 > LUTSIZE - 2:
        b0 = LUTSIZE - 2
    if r0 < 0:
        r0 = 0
    if g0 < 0:
        g0 = 0
    if b0 < 0:
        b0 = 0
    var dr = fr - Float32(r0)
    var dg = fg - Float32(g0)
    var db = fb - Float32(b0)

    # trilinear over the 8 corners, per output channel
    for ch in range(3):
        var c000 = lut_get(lut, r0,     g0,     b0,     ch)
        var c001 = lut_get(lut, r0,     g0,     b0 + 1, ch)
        var c010 = lut_get(lut, r0,     g0 + 1, b0,     ch)
        var c011 = lut_get(lut, r0,     g0 + 1, b0 + 1, ch)
        var c100 = lut_get(lut, r0 + 1, g0,     b0,     ch)
        var c101 = lut_get(lut, r0 + 1, g0,     b0 + 1, ch)
        var c110 = lut_get(lut, r0 + 1, g0 + 1, b0,     ch)
        var c111 = lut_get(lut, r0 + 1, g0 + 1, b0 + 1, ch)
        var c00 = c000 * (1.0 - db) + c001 * db
        var c01 = c010 * (1.0 - db) + c011 * db
        var c10 = c100 * (1.0 - db) + c101 * db
        var c11 = c110 * (1.0 - db) + c111 * db
        var c0 = c00 * (1.0 - dg) + c01 * dg
        var c1 = c10 * (1.0 - dg) + c11 * dg
        var cc = c0 * (1.0 - dr) + c1 * dr
        out[y, x, ch] = cc
    out[y, x, 3] = a


fn lut_get(
    lut: LayoutTensor[dtype, lut_layout, MutAnyOrigin],
    ri: Int, gi: Int, bi: Int, ch: Int
) -> Float32:
    var idx = (((ri * LUTSIZE) + gi) * LUTSIZE + bi) * 4 + ch
    return rebind[Scalar[dtype]](lut[idx])


# ===========================================================================
# Test image + driver
# ===========================================================================
fn build_test_image(host: LayoutTensor[dtype, img_layout, MutAnyOrigin]):
    # diagonal gradient + two bright regions (deterministic)
    for y in range(H):
        for x in range(W):
            var diag = Float32(x + y) / Float32(W + H - 2)
            var r = diag
            var g = Float32(x) / Float32(W - 1)
            var b = Float32(y) / Float32(H - 1)
            # bright rect [20..50)x[20..50)
            if x >= 20 and x < 50 and y >= 20 and y < 50:
                r = 0.95
                g = 0.9
                b = 0.1
            # bright circle center (90,90) radius 18
            var ddx = Float32(x - 90)
            var ddy = Float32(y - 90)
            if ddx * ddx + ddy * ddy < Float32(18 * 18):
                r = 0.1
                g = 0.85
                b = 0.95
            host[y, x, 0] = fclamp(r, 0.0, 1.0)
            host[y, x, 1] = fclamp(g, 0.0, 1.0)
            host[y, x, 2] = fclamp(b, 0.0, 1.0)
            host[y, x, 3] = 1.0


fn f32_to_u8(v: Float32) -> Int:
    var x = v * 255.0 + 0.5
    if x < 0.0:
        x = 0.0
    if x > 255.0:
        x = 255.0
    return Int(x)


# build a gaussian kernel for sigma into a host buffer (normalized)
fn gaussian_weights(sigma: Float32, host: LayoutTensor[dtype, kern_layout, MutAnyOrigin]):
    var s = 0.0
    var two_s2 = Float32(2.0) * sigma * sigma
    for t in range(KW):
        var d = Float32(t - KR)
        var w = exp(-(d * d) / two_s2)
        host[t] = w
        s += w
    for t in range(KW):
        host[t] = rebind[Scalar[dtype]](host[t]) / Float32(s)


def write_image_txt(path: String, host: LayoutTensor[dtype, img_layout, MutAnyOrigin]) raises:
    var s = String("")
    for y in range(H):
        for x in range(W):
            for ch in range(4):
                if len(s) > 0:
                    s += " "
                s += String(f32_to_u8(rebind[Scalar[dtype]](host[y, x, ch])))
    with open(path, "w") as f:
        f.write(s)


def main() raises:
    comptime if not has_accelerator():
        print("No GPU found")
        return
    var ctx = DeviceContext()

    # ---- input image ----
    var in_buf = ctx.enqueue_create_buffer[dtype](N)
    with in_buf.map_to_host() as hin:
        var t = LayoutTensor[dtype, img_layout](hin)
        build_test_image(t)
    var in_t = LayoutTensor[dtype, img_layout](in_buf)

    # write input dump (shared across all sub-filters)
    with in_buf.map_to_host() as hin:
        var t = LayoutTensor[dtype, img_layout](hin)
        write_image_txt(String("/tmp/shader_color_adv_in.txt"), t)

    var gdim = (W, H)
    var bdim = (BLOCK, BLOCK)

    # ====================== ColorBalance ======================
    var cb_buf = ctx.enqueue_create_buffer[dtype](N)
    var cb_t = LayoutTensor[dtype, img_layout](cb_buf)
    # fixed params (preserve_lightness = 0)
    ctx.enqueue_function[colorbalance_kernel, colorbalance_kernel](
        in_t, cb_t,
        Float32(0.3), Float32(-0.2), Float32(0.1),   # shadows r,g,b
        Float32(-0.1), Float32(0.25), Float32(-0.15),# midtones r,g,b
        Float32(0.2), Float32(-0.1), Float32(0.3),   # highlights r,g,b
        grid_dim=gdim, block_dim=bdim,
    )
    ctx.synchronize()
    with cb_buf.map_to_host() as h:
        var t = LayoutTensor[dtype, img_layout](h)
        write_image_txt(String("/tmp/shader_colorbalance_mojo.txt"), t)

    # ====================== CAS ======================
    var cas_buf = ctx.enqueue_create_buffer[dtype](N)
    var cas_t = LayoutTensor[dtype, img_layout](cas_buf)
    ctx.enqueue_function[cas_kernel, cas_kernel](
        in_t, cas_t, Float32(0.8),  # strength (CAS slider style)
        grid_dim=gdim, block_dim=bdim,
    )
    ctx.synchronize()
    with cas_buf.map_to_host() as h:
        var t = LayoutTensor[dtype, img_layout](h)
        write_image_txt(String("/tmp/shader_cas_mojo.txt"), t)

    # ====================== USM ======================
    var sigma = Float32(3.0)
    var amount = Float32(1.5)
    var threshold = Float32(0.0)  # 0 => sharpen everywhere (clean exact match)
    var kbuf = ctx.enqueue_create_buffer[dtype](KW)
    with kbuf.map_to_host() as hk:
        var kt = LayoutTensor[dtype, kern_layout](hk)
        gaussian_weights(sigma, kt)
    var kt_dev = LayoutTensor[dtype, kern_layout](kbuf)
    var tmp_buf = ctx.enqueue_create_buffer[dtype](N)
    var tmp_t = LayoutTensor[dtype, img_layout](tmp_buf)
    var blur_buf = ctx.enqueue_create_buffer[dtype](N)
    var blur_t = LayoutTensor[dtype, img_layout](blur_buf)
    var usm_buf = ctx.enqueue_create_buffer[dtype](N)
    var usm_t = LayoutTensor[dtype, img_layout](usm_buf)
    ctx.enqueue_function[usm_blur_h_kernel, usm_blur_h_kernel](
        in_t, tmp_t, kt_dev, grid_dim=gdim, block_dim=bdim)
    ctx.enqueue_function[usm_blur_v_kernel, usm_blur_v_kernel](
        tmp_t, blur_t, kt_dev, grid_dim=gdim, block_dim=bdim)
    ctx.enqueue_function[usm_combine_kernel, usm_combine_kernel](
        in_t, blur_t, usm_t, amount, threshold, grid_dim=gdim, block_dim=bdim)
    ctx.synchronize()
    with usm_buf.map_to_host() as h:
        var t = LayoutTensor[dtype, img_layout](h)
        write_image_txt(String("/tmp/shader_usm_mojo.txt"), t)

    # ====================== Lut3D (identity) ======================
    var lut_buf = ctx.enqueue_create_buffer[dtype](LUTSIZE * LUTSIZE * LUTSIZE * 4)
    with lut_buf.map_to_host() as hl:
        var lt = LayoutTensor[dtype, lut_layout](hl)
        var sc = Float32(LUTSIZE - 1)
        for ri in range(LUTSIZE):
            for gi in range(LUTSIZE):
                for bi in range(LUTSIZE):
                    var base = (((ri * LUTSIZE) + gi) * LUTSIZE + bi) * 4
                    lt[base + 0] = Float32(ri) / sc
                    lt[base + 1] = Float32(gi) / sc
                    lt[base + 2] = Float32(bi) / sc
                    lt[base + 3] = 0.0
    var lut_t = LayoutTensor[dtype, lut_layout](lut_buf)
    var l3d_buf = ctx.enqueue_create_buffer[dtype](N)
    var l3d_t = LayoutTensor[dtype, img_layout](l3d_buf)
    ctx.enqueue_function[lut3d_kernel, lut3d_kernel](
        in_t, lut_t, l3d_t, grid_dim=gdim, block_dim=bdim)
    ctx.synchronize()
    with l3d_buf.map_to_host() as h:
        var t = LayoutTensor[dtype, img_layout](h)
        write_image_txt(String("/tmp/shader_lut3d_mojo.txt"), t)

    print("color_adv: wrote colorbalance/cas/usm/lut3d mojo dumps + input")
