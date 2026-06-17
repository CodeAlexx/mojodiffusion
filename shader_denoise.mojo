# Group "denoise": SmartDenoise, HQDN3D, Guided, Deband
# Mojo GPU port of MediaEditor GLSL denoise filters.
# Build/run (from /home/alex/mojodiffusion): pixi run mojo run -I . shader_denoise.mojo
#
# Source citations (file:line of each ported formula):
#   SmartDenoise: /tmp/Med/plugin/nodes/filters/SmartDenoise/SmartDenoise_shader.h:43-90
#                 defaults sigma=1.2 ksigma=2.0 threshold=0.2 (ImMatSmartDenoiseNode.cpp:105,113,121)
#   HQDN3D:       /tmp/Med/plugin/nodes/filters/HQDN3D/HQDN3D_shader.h:28-99 (kernel)
#                 LUT + lowpass + YUV matrices; coef gen HQDN3D_vulkan.cpp:87-103;
#                 strengths HQDN3D_vulkan.cpp:9-11,46-53; load_rgba clamp imvk_mat_shader.h:579-580
#   Guided:       /tmp/Med/plugin/nodes/filters/Guided/Guided_shader.h:73-118 (coefficient solve +
#                 3x3 inverse) + Matting :164-179; box-blur Filter2DS_shader.h:107-114,199-205 +
#                 Box_vulkan.cpp:16-39 (kvulve=2/(xSize+ySize), anchor=size/2, clamp-edge);
#                 5-pass pipeline Guided_vulkan.cpp:66-164; defaults r=4 (ImMatGuidedNode.cpp:105),
#                 eps=1e-4 (ImMatGuidedNode.cpp:178).  NOTE: the shipped ToMatting shader has a
#                 typo (writes outTex2 twice, leaves outTex3 uninitialized — Guided_shader.h:31-33),
#                 so the GPU's exact bytes are NOT reproducible. We implement the INTENDED
#                 self-guided color guided filter (each RGB channel guided by itself) using the
#                 well-defined coefficient solve in Guided_shader.h:99-117 and verify the algorithm
#                 (not the shader UB) at PSNR>=40 vs our own f32 numpy reference.
#   Deband:       /tmp/Med/plugin/nodes/filters/Deband/DeBand_shader.h:27-97 (deband, blur=false path)
#                 xpos/ypos gen DeBand_vulkan.cpp:5-10,54-70 (frand, dir=r*direction, dist=int(r*range),
#                 xpos=int(cos(dir)*dist), ypos=int(sin(dir)*dist)); defaults range=16, direction=2
#                 (passed as direction*PI = 2*PI, ImMatDebandNode.cpp:55,115,214), threshold=0.01
#                 (ImMatDebandNode.cpp:212), blur=false (ImMatDebandNode.cpp:215).
#
# Output: writes the 4 filter outputs (concatenated, in the order SmartDenoise, HQDN3D, Guided,
# Deband) as flattened uint8 RGBA to /tmp/shader_denoise_mojo.txt, and the shared input image to
# /tmp/shader_denoise_in.txt.  The numpy reference (shader_denoise_ref.py) builds the identical
# image + identical algorithms and writes /tmp/shader_denoise_ref.txt for PSNR comparison.

from std.math import ceildiv, sqrt, exp, pow, sin, cos, log
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
comptime BLOCK = 16

# HQDN3D LUT: 512*16 int16 per channel-class, 4 classes. Stored as f32 buffer.
comptime LUT_LEN = 512 * 16
comptime lut_layout = Layout.row_major(LUT_LEN * 4)  # 4 coef classes packed

# Deband xpos/ypos LUT (one int per pixel), stored as f32 buffer
comptime pos_layout = Layout.row_major(H * W)


# ---------------------------------------------------------------------------
# Helpers (device, f32)
# ---------------------------------------------------------------------------
fn fclamp(x: Float32, lo: Float32, hi: Float32) -> Float32:
    var v = x
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return v


fn iclamp(x: Int, lo: Int, hi: Int) -> Int:
    var v = x
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return v


fn ld(inp: LayoutTensor[dtype, img_layout, MutAnyOrigin], x: Int, y: Int, ch: Int) -> Float32:
    # load_rgba clamps x,y to edge (imvk_mat_shader.h:579-580)
    var xx = iclamp(x, 0, W - 1)
    var yy = iclamp(y, 0, H - 1)
    return rebind[Scalar[dtype]](inp[yy, xx, ch])


# ===========================================================================
# 1) SmartDenoise (SmartDenoise_shader.h:43-90)
# ===========================================================================
comptime INV_SQRT_OF_2PI = Float32(0.39894228040143267793994605993439)
comptime INV_PI = Float32(0.31830988618379067153776752674503)
comptime invGamma = Float32(0.454545454545455)  # 1/2.2


def smartdenoise_kernel(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    out: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    sigma: Float32, ksigma: Float32, threshold: Float32,
):
    var gxu = global_idx.x
    var gyu = global_idx.y
    if gxu >= UInt(W) or gyu >= UInt(H):
        return
    var gx = Int(gxu)
    var gy = Int(gyu)

    # round(ksigma*sigma)  (SmartDenoise_shader.h:48)
    var radius_f = Float32(Int(ksigma * sigma + Float32(0.5)))  # round to nearest
    var radQ = radius_f * radius_f

    var invSigmaQx2 = Float32(0.5) / (sigma * sigma)            # :51
    var invSigmaQx2PI = INV_PI * invSigmaQx2                    # :52
    var invThresholdSqx2 = Float32(0.5) / (threshold * threshold)  # :54
    var invThresholdSqrt2PI = INV_SQRT_OF_2PI / threshold       # :55

    # center pixel, gamma-decoded  (:57-58)
    var cr = pow(ld(inp, gx, gy, 0), invGamma)
    var cg = pow(ld(inp, gx, gy, 1), invGamma)
    var cb = pow(ld(inp, gx, gy, 2), invGamma)

    var zBuff = Float32(0.0)
    var aR = Float32(0.0)
    var aG = Float32(0.0)
    var aB = Float32(0.0)

    # circular float-stepped loop, EXACT port (:64-78)
    var dx = -radius_f
    while dx <= radius_f:
        var pt = sqrt(radQ - dx * dx)        # :66
        var dy = -pt
        while dy <= pt:
            var bf = exp(-(dx * dx + dy * dy) * invSigmaQx2) * invSigmaQx2PI  # :69
            # point = uv + d (vec2 float); GLSL int(point.x) truncates toward zero (:71-72)
            var ptx = Float32(gx) + dx
            var pty = Float32(gy) + dy
            var ix = Int(ptx)   # GLSL int() truncates toward zero
            var iy = Int(pty)
            var wr = ld(inp, ix, iy, 0)
            var wg = ld(inp, ix, iy, 1)
            var wb = ld(inp, ix, iy, 2)
            var dCr = pow(wr, invGamma) - cr
            var dCg = pow(wg, invGamma) - cg
            var dCb = pow(wb, invGamma) - cb
            var dotdC = dCr * dCr + dCg * dCg + dCb * dCb
            var df = exp(-dotdC * invThresholdSqx2) * invThresholdSqrt2PI * bf  # :74
            zBuff += df
            aR += df * wr
            aG += df * wg
            aB += df * wb
            dy += Float32(1.0)
        dx += Float32(1.0)

    out[gy, gx, 0] = aR / zBuff
    out[gy, gx, 1] = aG / zBuff
    out[gy, gx, 2] = aB / zBuff
    out[gy, gx, 3] = Float32(1.0)


# ===========================================================================
# 2) HQDN3D (HQDN3D_shader.h:28-99)
#    Per-pixel, single frame, frame_spatial/frame_temporal start at 0.
# ===========================================================================
# YUV matrices (HQDN3D_shader.h:28-37). matrix is column-major in GLSL (mat[col][row]);
# yuv = offset + M_r2y * rgb  ->  yuv_i = sum_c M[c][i]*rgb_c  (column-vectors).
fn rgb_to_yuv(r: Float32, g: Float32, b: Float32) -> (Float32, Float32, Float32):
    # matrix_mat_r2y columns: c0=(.2627,-.13963,.5), c1=(.678,-.36037,-.459786), c2=(.0593,.5,-.040214)
    var y = Float32(0.262700) * r + Float32(0.678000) * g + Float32(0.059300) * b
    var u = Float32(0.5) + (Float32(-0.139630) * r + Float32(-0.360370) * g + Float32(0.500000) * b)
    var v = Float32(0.5) + (Float32(0.500000) * r + Float32(-0.459786) * g + Float32(-0.040214) * b)
    return (fclamp(y, 0.0, 1.0), fclamp(u, 0.0, 1.0), fclamp(v, 0.0, 1.0))


fn yuv_to_rgb(y: Float32, u: Float32, v: Float32) -> (Float32, Float32, Float32):
    # matrix_mat_y2r columns: c0=(1,1,1), c1=(0,-.164553,1.8814), c2=(1.4746,-.571353,0)
    var yy = y - Float32(0.0)
    var uu = u - Float32(0.5)
    var vv = v - Float32(0.5)
    var r = Float32(1.0) * yy + Float32(0.0) * uu + Float32(1.474600) * vv
    var g = Float32(1.0) * yy + Float32(-0.164553) * uu + Float32(-0.571353) * vv
    var b = Float32(1.0) * yy + Float32(1.881400) * uu + Float32(0.0) * vv
    return (fclamp(r, 0.0, 1.0), fclamp(g, 0.0, 1.0), fclamp(b, 0.0, 1.0))


# lowpass(prev, cur, class): d=(prev-cur)/16; return cur + LUT[class][d + 256*16] (HQDN3D_shader.h:50-57)
fn hq_lowpass(
    lut: LayoutTensor[dtype, lut_layout, MutAnyOrigin],
    prev: Int, cur: Int, cls: Int
) -> Int:
    # GLSL '/' truncates toward zero; Mojo // floors. Emulate trunc:
    var num = prev - cur
    var dt = num // 16
    if num < 0 and (num % 16) != 0:
        dt = dt + 1   # convert floor->trunc for negatives
    var idx = dt + 256 * 16
    idx = iclamp(idx, 0, LUT_LEN - 1)
    var coef = Int(rebind[Scalar[dtype]](lut[cls * LUT_LEN + idx]))
    return cur + coef


def hqdn3d_kernel(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    out: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    lut: LayoutTensor[dtype, lut_layout, MutAnyOrigin],
):
    var gxu = global_idx.x
    var gyu = global_idx.y
    if gxu >= UInt(W) or gyu >= UInt(H):
        return
    var x = Int(gxu)
    var y = Int(gyu)

    var r0 = ld(inp, x, y, 0)
    var g0c = ld(inp, x, y, 1)
    var b0c = ld(inp, x, y, 2)
    var a0 = ld(inp, x, y, 3)
    var yuv0 = rgb_to_yuv(r0, g0c, b0c)

    # yuv1 = rgb_to_yuv(load(x+1,y)) if x<w-1 else yuv0  (:68)
    var yuv1 = yuv0
    if x < W - 1:
        var r1 = ld(inp, x + 1, y, 0)
        var g1 = ld(inp, x + 1, y, 1)
        var b1 = ld(inp, x + 1, y, 2)
        yuv1 = rgb_to_yuv(r1, g1, b1)
    _ = yuv1  # yuv1 only feeds the temporal-buffer write, which doesn't alter THIS pixel's output

    # frame buffers are zero for a single frame. spatial=0, temporal=0.
    var LUMA_SPATIAL = 0
    var LUMA_TMP = 1
    var CHROMA_SPATIAL = 2
    var CHROMA_TMP = 3

    # --- Y --- (:70-79)
    var pixel_ant = hq_lowpass(lut, 0, Int(yuv0[0] * Float32(65535.0)) + 128, LUMA_SPATIAL)
    pixel_ant = hq_lowpass(lut, 0, pixel_ant, LUMA_SPATIAL)  # frame_spatial was 0
    var tmpv = hq_lowpass(lut, 0, pixel_ant, LUMA_TMP)       # frame_temporal was 0
    # (frame_temporal write is irrelevant to this pixel's output)
    var outY = Float32(tmpv) / Float32(65535.0)

    # --- U --- (:81-88)
    pixel_ant = hq_lowpass(lut, 0, Int(yuv0[1] * Float32(65535.0)) + 128, CHROMA_SPATIAL)
    pixel_ant = hq_lowpass(lut, 0, pixel_ant, CHROMA_SPATIAL)
    var tmpu = hq_lowpass(lut, 0, pixel_ant, CHROMA_TMP)
    var outU = Float32(tmpu) / Float32(65535.0)

    # --- V --- (:90-97)
    pixel_ant = hq_lowpass(lut, 0, Int(yuv0[2] * Float32(65535.0)) + 128, CHROMA_SPATIAL)
    pixel_ant = hq_lowpass(lut, 0, pixel_ant, CHROMA_SPATIAL)
    var tmpvv = hq_lowpass(lut, 0, pixel_ant, CHROMA_TMP)
    var outV = Float32(tmpvv) / Float32(65535.0)

    var rgb = yuv_to_rgb(outY, outU, outV)
    out[y, x, 0] = rgb[0]
    out[y, x, 1] = rgb[1]
    out[y, x, 2] = rgb[2]
    out[y, x, 3] = a0


# ===========================================================================
# 3) Guided filter (intended self-guided color guided filter)
#    box-blur radius r, eps. Uses the coefficient solve from Guided_shader.h:99-117
#    and the matting recombination Guided_shader.h:175-177.
#    Pipeline implemented as separate kernels mirroring the 5-pass vulkan flow.
# ===========================================================================
comptime GR = 4     # box radius (range default = 4, ImMatGuidedNode.cpp:105)
comptime GEPS = Float32(1.0e-4)

# Box mean with the shipped Box_vulkan semantics:
#  separable, square size = GR; per-pass weight kvulve = 2/(GR+GR) = 1/GR;
#  anchor = GR/2; window offset k-anchor for k in [0,GR); clamp-to-edge.
#  Two passes => total weight (1/GR)^2 over GR x GR window => true mean of GR x GR window.
# We compute the equivalent separable two-pass result exactly.

# Pass A: build the 6 moment fields from input.
#   t_I   = (r,g,b)            (guide = input)
#   t_p   = (r,g,b)            (input = same; self-guided per channel)
#   t_Ip  = (r*r, g*g, b*b)    (I*p, per channel self)
#   t_II  = (r*r, g*g, b*b)    (I*I = I*p here)  -> we only need I, Ip(=II) per channel
# For a per-channel self-guided filter: mean_I, mean_II, mean_Ip(=mean_II),
#   var = mean_II - mean_I^2 ; cov = mean_Ip - mean_I*mean_p = var ;
#   a = var/(var+eps) ; b = mean_p - a*mean_I = mean_I*(1-a) ; q = a*I + b.
# So we need two box-blurred fields: mean_I (per ch) and mean_II (per ch).

# moments kernel: out0 = I (rgb), out1 = I*I (rgb)
def guided_moments_kernel(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    mI: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    mII: LayoutTensor[dtype, img_layout, MutAnyOrigin],
):
    var gxu = global_idx.x
    var gyu = global_idx.y
    if gxu >= UInt(W) or gyu >= UInt(H):
        return
    var x = Int(gxu)
    var y = Int(gyu)
    for ch in range(3):
        var v = ld(inp, x, y, ch)
        mI[y, x, ch] = v
        mII[y, x, ch] = v * v
    mI[y, x, 3] = ld(inp, x, y, 3)
    mII[y, x, 3] = ld(inp, x, y, 3)


# Box horizontal pass: weight 1/GR, anchor GR/2, clamp-edge. (Filter2DS_shader.h:200-205)
def box_h_kernel(
    src: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    dst: LayoutTensor[dtype, img_layout, MutAnyOrigin],
):
    var gxu = global_idx.x
    var gyu = global_idx.y
    if gxu >= UInt(W) or gyu >= UInt(H):
        return
    var x = Int(gxu)
    var y = Int(gyu)
    var anchor = GR // 2
    var wv = Float32(1.0) / Float32(GR)
    for ch in range(4):
        var acc = Float32(0.0)
        for k in range(GR):
            var sx = x - anchor + k
            acc += wv * ld(src, sx, y, ch)
        dst[y, x, ch] = acc


# Box vertical pass: weight 1/GR, anchor GR/2, clamp-edge. (Filter2DS_shader.h:108-114)
def box_v_kernel(
    src: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    dst: LayoutTensor[dtype, img_layout, MutAnyOrigin],
):
    var gxu = global_idx.x
    var gyu = global_idx.y
    if gxu >= UInt(W) or gyu >= UInt(H):
        return
    var x = Int(gxu)
    var y = Int(gyu)
    var anchor = GR // 2
    var wv = Float32(1.0) / Float32(GR)
    for ch in range(4):
        var acc = Float32(0.0)
        for k in range(GR):
            var sy = y - anchor + k
            acc += wv * ld(src, x, sy, ch)
        dst[y, x, ch] = acc


# Coefficient solve + matting recombination (Guided_shader.h:105-117, 175-177)
#   per channel: var = mean_II - mean_I^2 ; a = var/(var+eps) ; b = mean_I*(1-a)
#   q = clamp(a*I + b, 0, 1)
def guided_combine_kernel(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    bmI: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    bmII: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    out: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    eps: Float32,
):
    var gxu = global_idx.x
    var gyu = global_idx.y
    if gxu >= UInt(W) or gyu >= UInt(H):
        return
    var x = Int(gxu)
    var y = Int(gyu)
    for ch in range(3):
        var mi = rebind[Scalar[dtype]](bmI[y, x, ch])
        var mii = rebind[Scalar[dtype]](bmII[y, x, ch])
        var varI = mii - mi * mi
        var a = varI / (varI + eps)
        var b = mi * (Float32(1.0) - a)
        var I = ld(inp, x, y, ch)
        out[y, x, ch] = fclamp(a * I + b, 0.0, 1.0)
    out[y, x, 3] = ld(inp, x, y, 3)


# ===========================================================================
# 4) Deband (DeBand_shader.h:27-97, blur=false path)
# ===========================================================================
def deband_kernel(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    out: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    xpos: LayoutTensor[dtype, pos_layout, MutAnyOrigin],
    ypos: LayoutTensor[dtype, pos_layout, MutAnyOrigin],
    threshold: Float32,
):
    var gxu = global_idx.x
    var gyu = global_idx.y
    if gxu >= UInt(W) or gyu >= UInt(H):
        return
    var x = Int(gxu)
    var y = Int(gyu)
    var xp = Int(rebind[Scalar[dtype]](xpos[y * W + x]))
    var yp = Int(rebind[Scalar[dtype]](ypos[y * W + x]))

    # CLIP(a,0,w-1) = max(0,min(a,w-1))  (DeBand_shader.h:27,35-38)
    var x_r = iclamp(x + xp, 0, W - 1)
    var y_p = iclamp(y + yp, 0, H - 1)
    var y_m = iclamp(y - yp, 0, H - 1)
    var x_l = iclamp(x - xp, 0, W - 1)

    for ch in range(3):
        var ref0 = ld(inp, x_r, y_p, ch)
        var ref1 = ld(inp, x_r, y_m, ch)
        var ref2 = ld(inp, x_l, y_m, ch)
        var ref3 = ld(inp, x_l, y_p, ch)
        var src0 = ld(inp, x, y, ch)
        var avg = (ref0 + ref1 + ref2 + ref3) / Float32(4.0)
        # blur=false branch (:49-52)
        var d0 = abs(src0 - ref0)
        var d1 = abs(src0 - ref1)
        var d2 = abs(src0 - ref2)
        var d3 = abs(src0 - ref3)
        var res = src0
        if d0 < threshold and d1 < threshold and d2 < threshold and d3 < threshold:
            res = avg
        out[y, x, ch] = res
    out[y, x, 3] = ld(inp, x, y, 3)


# ===========================================================================
# Host-side LUT / pos generation (mirror the C reference math)
# ===========================================================================

# HQDN3D coef gen (HQDN3D_vulkan.cpp:87-99), computed in Float64 to mirror C double.
fn build_hqdn3d_lut(lut: LayoutTensor[dtype, lut_layout, MutAnyOrigin]):
    # strengths (HQDN3D_vulkan.cpp:9-11,46-53): P1=4,P2=3,P3=6
    var P1 = Float64(4.0)
    var P2 = Float64(3.0)
    var P3 = Float64(6.0)
    var s_luma_sp = P1
    var s_chroma_sp = P2 * s_luma_sp / P1
    var s_luma_tmp = P3 * s_luma_sp / P1
    var s_chroma_tmp = s_luma_tmp * s_chroma_sp / s_luma_sp
    var strengths = List[Float64]()
    strengths.append(s_luma_sp)     # class 0 LUMA_SPATIAL
    strengths.append(s_luma_tmp)    # class 1 LUMA_TMP
    strengths.append(s_chroma_sp)   # class 2 CHROMA_SPATIAL
    strengths.append(s_chroma_tmp)  # class 3 CHROMA_TMP
    for cls in range(4):
        var dist25 = strengths[cls]
        var d = dist25
        if d > Float64(252.0):
            d = Float64(252.0)
        var gamma = log(Float64(0.25)) / log(Float64(1.0) - d / Float64(255.0) - Float64(0.00001))
        for i in range(-256 * 16, 256 * 16):
            var f = Float64((i * 32) + 16 - 1) / Float64(512.0)  # (i<<5)+16-1
            var af = f
            if af < 0:
                af = -af
            var simil = Float64(1.0) - af / Float64(255.0)
            if simil < 0:
                simil = Float64(0.0)
            var Cv = pow(simil, gamma) * Float64(256.0) * f
            # lrint: round to nearest, ties to even. Use round-half-away which matches for our values.
            var ci = Int(Cv + Float64(0.5)) if Cv >= 0 else Int(Cv - Float64(0.5))
            lut[cls * LUT_LEN + (256 * 16 + i)] = Float32(ci)


# Deband frand + pos gen (DeBand_vulkan.cpp:5-10,54-70). direction=2*PI, range=16.
fn build_deband_pos(
    xpos: LayoutTensor[dtype, pos_layout, MutAnyOrigin],
    ypos: LayoutTensor[dtype, pos_layout, MutAnyOrigin],
):
    var direction = Float32(2.0) * Float32(3.14159265358979323846)
    var rng = Float32(16.0)
    for y in range(H):
        for x in range(W):
            # frand: r = sin(x*12.9898 + y*78.233)*43758.545 ; return r - floor(r)
            var arg = Float32(x) * Float32(12.9898) + Float32(y) * Float32(78.233)
            var r = sin(arg) * Float32(43758.545)
            var fr = r - Float32(Int(r) if r >= 0 else Int(r) - 1)  # floor
            # dir = r*direction (direction>0); dist = int(r*range) (range>0)
            var dirv = fr * direction
            var dist = Int(fr * rng)        # truncate toward zero (range>=0 so fr*rng>=0)
            var xv = Int(cos(dirv) * Float32(dist))   # int() truncates
            var yv = Int(sin(dirv) * Float32(dist))
            xpos[y * W + x] = Float32(xv)
            ypos[y * W + x] = Float32(yv)


# ===========================================================================
# Test image + driver
# ===========================================================================
fn build_test_image(host: LayoutTensor[dtype, img_layout, MutAnyOrigin]):
    # diagonal gradient + two bright regions (identical to numpy ref)
    for y in range(H):
        for x in range(W):
            var diag = Float32(x + y) / Float32(W + H - 2)
            var r = diag
            var g = Float32(x) / Float32(W - 1)
            var b = Float32(y) / Float32(H - 1)
            if x >= 20 and x < 50 and y >= 20 and y < 50:
                r = 0.95
                g = 0.9
                b = 0.1
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


def append_image_txt(mut s: String, host: LayoutTensor[dtype, img_layout, MutAnyOrigin]):
    for y in range(H):
        for x in range(W):
            for ch in range(4):
                if len(s) > 0:
                    s += " "
                s += String(f32_to_u8(rebind[Scalar[dtype]](host[y, x, ch])))


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

    # write input dump
    with in_buf.map_to_host() as hin:
        var t = LayoutTensor[dtype, img_layout](hin)
        var s = String("")
        append_image_txt(s, t)
        with open(String("/tmp/shader_denoise_in.txt"), "w") as f:
            f.write(s)

    var gdim = (W, H)
    var bdim = (BLOCK, BLOCK)

    var out_s = String("")

    # ====================== SmartDenoise ======================
    var sd_buf = ctx.enqueue_create_buffer[dtype](N)
    var sd_t = LayoutTensor[dtype, img_layout](sd_buf)
    ctx.enqueue_function[smartdenoise_kernel, smartdenoise_kernel](
        in_t, sd_t, Float32(1.2), Float32(2.0), Float32(0.2),
        grid_dim=gdim, block_dim=bdim,
    )
    ctx.synchronize()
    with sd_buf.map_to_host() as h:
        var t = LayoutTensor[dtype, img_layout](h)
        append_image_txt(out_s, t)

    # ====================== HQDN3D ======================
    var lut_buf = ctx.enqueue_create_buffer[dtype](LUT_LEN * 4)
    with lut_buf.map_to_host() as hl:
        var lt = LayoutTensor[dtype, lut_layout](hl)
        build_hqdn3d_lut(lt)
    var lut_t = LayoutTensor[dtype, lut_layout](lut_buf)
    var hq_buf = ctx.enqueue_create_buffer[dtype](N)
    var hq_t = LayoutTensor[dtype, img_layout](hq_buf)
    ctx.enqueue_function[hqdn3d_kernel, hqdn3d_kernel](
        in_t, hq_t, lut_t, grid_dim=gdim, block_dim=bdim,
    )
    ctx.synchronize()
    with hq_buf.map_to_host() as h:
        var t = LayoutTensor[dtype, img_layout](h)
        append_image_txt(out_s, t)

    # ====================== Guided ======================
    var mI_buf = ctx.enqueue_create_buffer[dtype](N)
    var mII_buf = ctx.enqueue_create_buffer[dtype](N)
    var mI_t = LayoutTensor[dtype, img_layout](mI_buf)
    var mII_t = LayoutTensor[dtype, img_layout](mII_buf)
    ctx.enqueue_function[guided_moments_kernel, guided_moments_kernel](
        in_t, mI_t, mII_t, grid_dim=gdim, block_dim=bdim)
    # box-blur each moment field: H then V (separable)
    var tmp1_buf = ctx.enqueue_create_buffer[dtype](N)
    var tmp1_t = LayoutTensor[dtype, img_layout](tmp1_buf)
    var bmI_buf = ctx.enqueue_create_buffer[dtype](N)
    var bmI_t = LayoutTensor[dtype, img_layout](bmI_buf)
    var tmp2_buf = ctx.enqueue_create_buffer[dtype](N)
    var tmp2_t = LayoutTensor[dtype, img_layout](tmp2_buf)
    var bmII_buf = ctx.enqueue_create_buffer[dtype](N)
    var bmII_t = LayoutTensor[dtype, img_layout](bmII_buf)
    ctx.enqueue_function[box_h_kernel, box_h_kernel](mI_t, tmp1_t, grid_dim=gdim, block_dim=bdim)
    ctx.enqueue_function[box_v_kernel, box_v_kernel](tmp1_t, bmI_t, grid_dim=gdim, block_dim=bdim)
    ctx.enqueue_function[box_h_kernel, box_h_kernel](mII_t, tmp2_t, grid_dim=gdim, block_dim=bdim)
    ctx.enqueue_function[box_v_kernel, box_v_kernel](tmp2_t, bmII_t, grid_dim=gdim, block_dim=bdim)
    var gd_buf = ctx.enqueue_create_buffer[dtype](N)
    var gd_t = LayoutTensor[dtype, img_layout](gd_buf)
    ctx.enqueue_function[guided_combine_kernel, guided_combine_kernel](
        in_t, bmI_t, bmII_t, gd_t, GEPS, grid_dim=gdim, block_dim=bdim)
    ctx.synchronize()
    with gd_buf.map_to_host() as h:
        var t = LayoutTensor[dtype, img_layout](h)
        append_image_txt(out_s, t)

    # ====================== Deband ======================
    var xpos_buf = ctx.enqueue_create_buffer[dtype](H * W)
    var ypos_buf = ctx.enqueue_create_buffer[dtype](H * W)
    with xpos_buf.map_to_host() as hx:
        var xt = LayoutTensor[dtype, pos_layout](hx)
        with ypos_buf.map_to_host() as hy:
            var yt = LayoutTensor[dtype, pos_layout](hy)
            build_deband_pos(xt, yt)
    var xpos_t = LayoutTensor[dtype, pos_layout](xpos_buf)
    var ypos_t = LayoutTensor[dtype, pos_layout](ypos_buf)
    var db_buf = ctx.enqueue_create_buffer[dtype](N)
    var db_t = LayoutTensor[dtype, img_layout](db_buf)
    ctx.enqueue_function[deband_kernel, deband_kernel](
        in_t, db_t, xpos_t, ypos_t, Float32(0.01),
        grid_dim=gdim, block_dim=bdim,
    )
    ctx.synchronize()
    with db_buf.map_to_host() as h:
        var t = LayoutTensor[dtype, img_layout](h)
        append_image_txt(out_s, t)

    with open(String("/tmp/shader_denoise_mojo.txt"), "w") as f:
        f.write(out_s)

    print("denoise: wrote smartdenoise/hqdn3d/guided/deband mojo dump + input")
