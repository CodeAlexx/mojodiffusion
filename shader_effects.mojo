# Group "effects": WaterRipple, Soul, Sway, Star, Lighting, Jitter
# Mojo GPU port of MediaEditor GLSL time-parameterized per-pixel effects.
# Build/run (from /home/alex/mojodiffusion): pixi run mojo run -I . shader_effects.mojo
#
# Source citations (shader file:line per formula):
#   WaterRipple: /tmp/Med/plugin/nodes/effects/WaterRipple/WaterRipple_shader.h:27-40
#                (center .5,.5; dir=uv-center; len=|dir|; uv_new=uv+dir*amount*abs(sin(len*freq))/len;
#                 sample int(uv_new.x*(w-1)),int(uv_new.y*(h-1)) with clamp-to-edge)
#   Soul:        /tmp/Med/plugin/nodes/effects/Soul/Soul_shader.h:30-48
#                (duration=1/count; progress=mod(p.progress,duration)/duration; shrink fold;
#                 alpha=max_alpha*(1-progress); scale=1+(max_scale-1)*progress;
#                 weak=0.5+(uv-0.5)/scale clamp01; rgba=mix(weak_sample, unscaled_sample, alpha))
#   Sway:        /tmp/Med/plugin/nodes/effects/Sway/Sway_shader.h:29-40
#                (waveu=sin((uv.y+speed_arg)*density)*strength/1000; horizontal? +x : +y; clamp uv01;
#                 NOTE: shader 'speed' arg = time*m_speed per ImMatSwayEffectlNode.cpp:56)
#   Star:        /tmp/Med/plugin/nodes/effects/Star/Star_shader.h:31-131
#                (additive procedural starfield; per-layer Rotate/scale/offset; 3x3 cell pseudo-random;
#                 Star() flare core; col added to source rgb)
#   Lighting:    /tmp/Med/plugin/nodes/effects/Lighting/Lighting_shader.h:29-167
#                (duration=1/count; progress; amplitude=abs(sin(progress*PI/duration)); hue=amplitude;
#                 rgb2hsv add hue+value, hsv2rgb, RGBtoHSL add saturation, HSLtoRGB, mix white by light)
#   Jitter:      /tmp/Med/plugin/nodes/effects/Jitter/Jitter_shader.h:29-53
#                (duration=1/count; progress; shrink fold; offsetCoords=offset*progress;
#                 scale=1+(max_scale-1)*progress; ScaleTextureCoords clamp01;
#                 mask_r at +offset, mask_b at -offset, mask center; out=(mask_r.r, mask.g, mask_b.b, mask.a))
#
# Edge handling: GLSL load_rgba clamps the integer sample coords to [0,w-1]/[0,h-1]
#   (imvk_mat_shader.h:577-593). We replicate exactly. GLSL int(f) truncates toward zero;
#   all sample coordinates here are >= 0 after the uv-clamps, so Int(Float32) (trunc) == floor.
# All compute in f32 (image filter convention).

from std.math import ceildiv, sin, cos, sqrt
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
comptime PI = Float32(3.1415)          # Star uses 3.1415 (Star_shader.h:33)
comptime PI_L = Float32(3.1415926)     # Lighting uses 3.1415926 (Lighting_shader.h:29)


# ---------------------------------------------------------------------------
# Device helpers (f32). GLSL fract/mod/clamp/smoothstep/length equivalents.
# ---------------------------------------------------------------------------
fn fclamp(x: Float32, lo: Float32, hi: Float32) -> Float32:
    var v = x
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return v


fn glsl_fract(x: Float32) -> Float32:
    # GLSL fract(x) = x - floor(x)
    var f = x - _floorf(x)
    return f


fn _floorf(x: Float32) -> Float32:
    var i = Float32(Int(x))
    if i > x:
        i = i - Float32(1.0)
    return i


fn glsl_mod(x: Float32, y: Float32) -> Float32:
    # GLSL mod(x,y) = x - y*floor(x/y)
    return x - y * _floorf(x / y)


fn glsl_smoothstep(edge0: Float32, edge1: Float32, x: Float32) -> Float32:
    var t = fclamp((x - edge0) / (edge1 - edge0), Float32(0.0), Float32(1.0))
    return t * t * (Float32(3.0) - Float32(2.0) * t)


fn sample_clamped(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin], x: Int, y: Int, ch: Int
) -> Float32:
    # replicate load_rgba clamp-to-edge on integer coords (imvk_mat_shader.h:579-580)
    var cx = x
    var cy = y
    if cx < 0:
        cx = 0
    if cx > W - 1:
        cx = W - 1
    if cy < 0:
        cy = 0
    if cy > H - 1:
        cy = H - 1
    return rebind[Scalar[dtype]](inp[cy, cx, ch])


fn trunc_coord(f: Float32) -> Int:
    # GLSL int() truncates toward zero. f>=0 here so Int() (toward zero) is correct.
    return Int(f)


# ===========================================================================
# 1) WaterRipple  (WaterRipple_shader.h:27-40)
# ===========================================================================
def waterripple_kernel(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    out: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    freq: Float32, amount: Float32,
):
    var gx = global_idx.x
    var gy = global_idx.y
    if gx >= UInt(W) or gy >= UInt(H):
        return
    var x = Int(gx)
    var y = Int(gy)
    var uvx = Float32(x) / Float32(W - 1)
    var uvy = Float32(y) / Float32(H - 1)
    var dirx = uvx - Float32(0.5)
    var diry = uvy - Float32(0.5)
    var ln = sqrt(dirx * dirx + diry * diry)
    # uv_new = uv + dir * amount * abs(sin(len*freq)) / len  (WaterRipple_shader.h:37)
    var s = sin(ln * freq)
    if s < 0.0:
        s = -s
    var fac = amount * s / ln
    var nux = uvx + dirx * fac
    var nuy = uvy + diry * fac
    var sx = trunc_coord(nux * Float32(W - 1))
    var sy = trunc_coord(nuy * Float32(H - 1))
    for ch in range(4):
        out[y, x, ch] = sample_clamped(inp, sx, sy, ch)


# ===========================================================================
# 2) Soul  (Soul_shader.h:30-48)
# ===========================================================================
def soul_kernel(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    out: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    progress_in: Float32, count: Int, max_scale: Float32, max_alpha: Float32, shrink: Int,
):
    var gx = global_idx.x
    var gy = global_idx.y
    if gx >= UInt(W) or gy >= UInt(H):
        return
    var x = Int(gx)
    var y = Int(gy)
    var uvx = Float32(x) / Float32(W - 1)
    var uvy = Float32(y) / Float32(H - 1)
    var duration = Float32(1.0) / Float32(count)
    var progress = glsl_mod(progress_in, duration) / duration
    if shrink == 1 and progress > 0.5:
        progress = Float32(1.0) - progress
    var alpha = max_alpha * (Float32(1.0) - progress)
    var scale = Float32(1.0) + (max_scale - Float32(1.0)) * progress
    var weakX = Float32(0.5) + (uvx - Float32(0.5)) / scale
    var weakY = Float32(0.5) + (uvy - Float32(0.5)) / scale
    weakX = fclamp(weakX, Float32(0.0), Float32(1.0))
    weakY = fclamp(weakY, Float32(0.0), Float32(1.0))
    var wsx = trunc_coord(weakX * Float32(W - 1))
    var wsy = trunc_coord(weakY * Float32(H - 1))
    var msx = trunc_coord(uvx * Float32(W - 1))
    var msy = trunc_coord(uvy * Float32(H - 1))
    # rgba = mix(rgba, mask, alpha) = rgba*(1-alpha) + mask*alpha  (Soul_shader.h:47)
    for ch in range(4):
        var a = sample_clamped(inp, wsx, wsy, ch)
        var b = sample_clamped(inp, msx, msy, ch)
        out[y, x, ch] = a * (Float32(1.0) - alpha) + b * alpha


# ===========================================================================
# 3) Sway  (Sway_shader.h:29-40). shader 'speed' arg = time*m_speed (cpp:56)
# ===========================================================================
def sway_kernel(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    out: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    speed: Float32, strength: Float32, density: Float32, horizontal: Int,
):
    var gx = global_idx.x
    var gy = global_idx.y
    if gx >= UInt(W) or gy >= UInt(H):
        return
    var x = Int(gx)
    var y = Int(gy)
    var uvx = Float32(x) / Float32(W - 1)
    var uvy = Float32(y) / Float32(H - 1)
    # waveu = sin((uv.y + speed) * density) * strength / 1000  (Sway_shader.h:36)
    var waveu = sin((uvy + speed) * density) * strength / Float32(1000.0)
    if horizontal == 1:
        uvx = uvx + waveu
    else:
        uvy = uvy + waveu
    uvx = fclamp(uvx, Float32(0.0), Float32(1.0))
    uvy = fclamp(uvy, Float32(0.0), Float32(1.0))
    var sx = trunc_coord(uvx * Float32(W - 1))
    var sy = trunc_coord(uvy * Float32(H - 1))
    for ch in range(4):
        out[y, x, ch] = sample_clamped(inp, sx, sy, ch)


# ===========================================================================
# 4) Star  (Star_shader.h:31-131) — additive procedural starfield.
# ===========================================================================
fn star_pseudo_random(px_in: Float32, py_in: Float32) -> Float32:
    # PseudoRandomizer (Star_shader.h:63-68)
    var px = glsl_fract(px_in * Float32(123.45))
    var py = glsl_fract(py_in * Float32(345.67))
    var d = px * px + py * py + Float32(45.32) * (px + py)  # dot(point, point+45.32)
    # GLSL: point += dot(point, point + 45.32);  adds scalar to each comp
    px = px + d
    py = py + d
    return glsl_fract(px * py)


fn star_core(uvx_in: Float32, uvy_in: Float32, flaresize: Float32, rotAngle: Float32) -> Float32:
    # Star() (Star_shader.h:44-61)
    var uvx = uvx_in
    var uvy = uvy_in
    var d = sqrt(uvx * uvx + uvy * uvy)
    var starcore = Float32(0.05) / d
    # uv *= Rotate(-2*PI*rotAngle): mat2(c,-s,s,c) applied as
    #   v' = (c*x - s*y, s*x + c*y)
    var ang = Float32(-2.0) * PI * rotAngle
    var c = cos(ang)
    var s = sin(ang)
    var nx = c * uvx - s * uvy
    var ny = s * uvx + c * uvy
    uvx = nx
    uvy = ny
    var flareMax = Float32(1.0)
    var f1 = uvx * uvy * Float32(3000.0)
    if f1 < 0.0:
        f1 = -f1
    var starflares = flareMax - f1
    if starflares < 0.0:
        starflares = 0.0
    starcore = starcore + starflares * flaresize
    # uv *= Rotate(PI*0.25)
    var ang2 = PI * Float32(0.25)
    var c2 = cos(ang2)
    var s2 = sin(ang2)
    var nx2 = c2 * uvx - s2 * uvy
    var ny2 = s2 * uvx + c2 * uvy
    uvx = nx2
    uvy = ny2
    var f2 = uvx * uvy * Float32(3000.0)
    if f2 < 0.0:
        f2 = -f2
    var starflares2 = flareMax - f2
    if starflares2 < 0.0:
        starflares2 = 0.0
    starcore = starcore + starflares2 * Float32(0.3) * flaresize
    starcore = starcore * glsl_smoothstep(Float32(1.0), Float32(0.05), d)
    return starcore


fn star_field_layer(
    uvx: Float32, uvy: Float32, rotAngle: Float32, progress: Float32,
    pr: Float32, pg: Float32, pb: Float32, layers: Float32,
) -> (Float32, Float32, Float32):
    # StarFieldLayer (Star_shader.h:71-105). Returns this layer's (col.r, col.g, col.b).
    var gvx = glsl_fract(uvx) - Float32(0.5)
    var gvy = glsl_fract(uvy) - Float32(0.5)
    var idx = _floorf(uvx)
    var idy = _floorf(uvy)
    var deltaTimeTwinkle = progress
    var cr = Float32(0.0)
    var cg = Float32(0.0)
    var cb = Float32(0.0)
    for yy in range(-1, 2):
        for xx in range(-1, 2):
            var offx = Float32(xx)
            var offy = Float32(yy)
            var randomN = star_pseudo_random(idx + offx, idy + offy)
            var randoX = randomN - Float32(0.5)
            var randoY = glsl_fract(randomN * Float32(45.0)) - Float32(0.5)
            var rpx = gvx - offx - randoX
            var rpy = gvy - offy - randoY
            var size = glsl_fract(randomN * Float32(1356.33))
            var flareSwitch = glsl_smoothstep(Float32(0.9), Float32(1.0), size)
            var star = star_core(rpx, rpy, flareSwitch, rotAngle)
            var randomStarColorSeed = glsl_fract(randomN * Float32(2150.0)) * (Float32(3.0) * PI) * deltaTimeTwinkle
            # color = sin(vec3(red,green,blue) * seed)
            var colr = sin(pr * randomStarColorSeed)
            var colg = sin(pg * randomStarColorSeed)
            var colb = sin(pb * randomStarColorSeed)
            # color = color * (0.4*sin(deltaTimeTwinkle)) + 0.6
            var m = Float32(0.4) * sin(deltaTimeTwinkle)
            colr = colr * m + Float32(0.6)
            colg = colg * m + Float32(0.6)
            colb = colb * m + Float32(0.6)
            # color = color * vec3(red, green, blue + size)
            colr = colr * pr
            colg = colg * pg
            colb = colb * (pb + size)
            var dimByDensity = Float32(15.0) / layers
            cr = cr + star * size * colr * dimByDensity
            cg = cg + star * size * colg * dimByDensity
            cb = cb + star * size * colb * dimByDensity
    return (cr, cg, cb)


# Star kernel: applies layerFader to each layer's contribution (Star_shader.h:127),
# i.e. col += StarFieldLayer(...) * layerFader, by accumulating the raw layer into
# locals (lr,lg,lb) then scaling by layerFader before adding to the running total.
def star_kernel2(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    out: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    progress: Float32, speed: Float32, layers: Float32,
    red: Float32, green: Float32, blue: Float32,
):
    var gx = global_idx.x
    var gy = global_idx.y
    if gx >= UInt(W) or gy >= UInt(H):
        return
    var x = Int(gx)
    var y = Int(gy)
    var tuvx = Float32(x) / Float32(W - 1)
    var tuvy = Float32(y) / Float32(H - 1)
    var deltaTime = progress * speed * Float32(0.01)
    var col_r = Float32(0.0)
    var col_g = Float32(0.0)
    var col_b = Float32(0.0)
    var rotAngle = progress * speed * Float32(0.09)
    var MIN_DIVIDE = Float32(64.0)
    var MAX_DIVIDE = Float32(0.01)
    var i = Float32(0.0)
    var stepi = Float32(1.0) / layers
    while i < Float32(1.0):
        var layerDepth = glsl_fract(i + deltaTime)
        var layerScale = MIN_DIVIDE + (MAX_DIVIDE - MIN_DIVIDE) * layerDepth
        var layerFader = layerDepth * glsl_smoothstep(Float32(0.1), Float32(1.1), layerDepth)
        var layerOffset = i * (Float32(3430.00) + glsl_fract(i))
        var ang = rotAngle * i * Float32(-10.0)
        var c = cos(ang)
        var s = sin(ang)
        var ntx = c * tuvx - s * tuvy
        var nty = s * tuvx + c * tuvy
        tuvx = ntx
        tuvy = nty
        var sfx = tuvx * layerScale + layerOffset
        var sfy = tuvy * layerScale + layerOffset
        var lcol = star_field_layer(
            sfx, sfy, rotAngle, progress, red, green, blue, layers,
        )
        var lr = lcol[0]
        var lg = lcol[1]
        var lb = lcol[2]
        col_r = col_r + lr * layerFader
        col_g = col_g + lg * layerFader
        col_b = col_b + lb * layerFader
        i = i + stepi
    var r = rebind[Scalar[dtype]](inp[y, x, 0])
    var g = rebind[Scalar[dtype]](inp[y, x, 1])
    var b = rebind[Scalar[dtype]](inp[y, x, 2])
    var a = rebind[Scalar[dtype]](inp[y, x, 3])
    out[y, x, 0] = r + col_r
    out[y, x, 1] = g + col_g
    out[y, x, 2] = b + col_b
    out[y, x, 3] = a


# ===========================================================================
# 5) Lighting  (Lighting_shader.h:29-167)
# ===========================================================================
fn light_rgb2hsv(r: Float32, g: Float32, b: Float32) -> (Float32, Float32, Float32):
    # rgb2hsv (Lighting_shader.h:41-50). Returns (h, s, v).
    # K = (0, -1/3, 2/3, -1)
    var Kx = Float32(0.0)
    var Ky = Float32(-1.0 / 3.0)
    var Kz = Float32(2.0 / 3.0)
    var Kw = Float32(-1.0)
    # p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g))
    # step(b,g)=1 if g>=b else 0
    var px = Float32(0.0)
    var py = Float32(0.0)
    var pz = Float32(0.0)
    var pw = Float32(0.0)
    if g >= b:
        # use vec4(c.gb, K.xy) = (g, b, Kx, Ky)
        px = g
        py = b
        pz = Kx
        pw = Ky
    else:
        # vec4(c.bg, K.wz) = (b, g, Kw, Kz)
        px = b
        py = g
        pz = Kw
        pw = Kz
    # q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r))
    # step(p.x, r)=1 if r>=p.x else 0
    var qx = Float32(0.0)
    var qy = Float32(0.0)
    var qz = Float32(0.0)
    var qw = Float32(0.0)
    if r >= px:
        # vec4(c.r, p.yzx) = (r, p.y, p.z, p.x)
        qx = r
        qy = py
        qz = pz
        qw = px
    else:
        # vec4(p.xyw, c.r) = (p.x, p.y, p.w, r)
        qx = px
        qy = py
        qz = pw
        qw = r
    var d = qx - min(qw, qy)
    var e = Float32(1.0e-10)
    var hh = (qz + (qw - qy) / (Float32(6.0) * d + e))
    if hh < 0.0:
        hh = -hh
    return (hh, d / (qx + e), qx)


fn light_hsv2rgb(h: Float32, s: Float32, v: Float32) -> (Float32, Float32, Float32):
    # hsv2rgb (Lighting_shader.h:35-40). Returns (r, g, b).
    # K = (1, 2/3, 1/3, 3); p = abs(fract(c.xxx + K.xyz)*6 - K.www)
    var Kx = Float32(1.0)
    var Ky = Float32(2.0 / 3.0)
    var Kz = Float32(1.0 / 3.0)
    var Kw = Float32(3.0)
    var px = (glsl_fract(h + Kx) * Float32(6.0)) - Kw
    var py = (glsl_fract(h + Ky) * Float32(6.0)) - Kw
    var pz = (glsl_fract(h + Kz) * Float32(6.0)) - Kw
    if px < 0.0:
        px = -px
    if py < 0.0:
        py = -py
    if pz < 0.0:
        pz = -pz
    # return c.z * mix(K.xxx, clamp(p - K.xxx, 0, 1), c.y)
    # K.xxx = (1,1,1)
    var cr = fclamp(px - Kx, Float32(0.0), Float32(1.0))
    var cg = fclamp(py - Kx, Float32(0.0), Float32(1.0))
    var cb = fclamp(pz - Kx, Float32(0.0), Float32(1.0))
    # mix(1, c, s) = 1*(1-s) + c*s
    var rr = v * (Kx * (Float32(1.0) - s) + cr * s)
    var gg = v * (Kx * (Float32(1.0) - s) + cg * s)
    var bb = v * (Kx * (Float32(1.0) - s) + cb * s)
    return (rr, gg, bb)


fn light_hue2rgb(hue: Float32) -> (Float32, Float32, Float32):
    # HUEtoRGB (Lighting_shader.h:61-67). Returns (r, g, b).
    # rgb = abs(hue*6 - vec3(3,2,4)) * vec3(1,-1,-1) + vec3(-1,2,2)
    var a0 = hue * Float32(6.0) - Float32(3.0)
    var a1 = hue * Float32(6.0) - Float32(2.0)
    var a2 = hue * Float32(6.0) - Float32(4.0)
    if a0 < 0.0:
        a0 = -a0
    if a1 < 0.0:
        a1 = -a1
    if a2 < 0.0:
        a2 = -a2
    var rr = a0 * Float32(1.0) + Float32(-1.0)
    var gg = a1 * Float32(-1.0) + Float32(2.0)
    var bb = a2 * Float32(-1.0) + Float32(2.0)
    return (
        fclamp(rr, Float32(0.0), Float32(1.0)),
        fclamp(gg, Float32(0.0), Float32(1.0)),
        fclamp(bb, Float32(0.0), Float32(1.0)),
    )


fn light_rgb2hcv(r: Float32, g: Float32, b: Float32) -> (Float32, Float32, Float32):
    # RGBtoHCV (Lighting_shader.h:51-59). Returns (h, c, v).
    var EPSILON = Float32(1.0e-10)
    var px = Float32(0.0)
    var py = Float32(0.0)
    var pz = Float32(0.0)
    var pw = Float32(0.0)
    if g < b:
        # vec4(rgb.bg, -1, 2/3) = (b, g, -1, 2/3)
        px = b
        py = g
        pz = Float32(-1.0)
        pw = Float32(2.0 / 3.0)
    else:
        # vec4(rgb.gb, 0, -1/3) = (g, b, 0, -1/3)
        px = g
        py = b
        pz = Float32(0.0)
        pw = Float32(-1.0 / 3.0)
    var qx = Float32(0.0)
    var qy = Float32(0.0)
    var qz = Float32(0.0)
    var qw = Float32(0.0)
    if r < px:
        # vec4(p.xyw, rgb.r) = (p.x, p.y, p.w, r)
        qx = px
        qy = py
        qz = pw
        qw = r
    else:
        # vec4(rgb.r, p.yzx) = (r, p.y, p.z, p.x)
        qx = r
        qy = py
        qz = pz
        qw = px
    var cc = qx - min(qw, qy)
    var hnum = (qw - qy) / (Float32(6.0) * cc + EPSILON) + qz
    if hnum < 0.0:
        hnum = -hnum
    return (hnum, cc, qx)


fn light_rgb2hsl(r: Float32, g: Float32, b: Float32) -> (Float32, Float32, Float32):
    # RGBtoHSL (Lighting_shader.h:88-95). Returns (h, s, z).
    var EPSILON = Float32(1.0e-10)
    var hcv = light_rgb2hcv(r, g, b)
    var hcv_h = hcv[0]
    var hcv_c = hcv[1]
    var hcv_v = hcv[2]
    var z = hcv_v - hcv_c * Float32(0.5)
    # s = hcv.y / (1 - abs(z*2 - 1) + EPSILON)
    var az = Float32(2.0) * z - Float32(1.0)
    if az < 0.0:
        az = -az
    var den = Float32(1.0) - az + EPSILON
    var s = hcv_c / den
    return (hcv_h, s, z)


fn light_hsl2rgb(h: Float32, s: Float32, z: Float32) -> (Float32, Float32, Float32):
    # HSLtoRGB (Lighting_shader.h:81-87). Returns (r, g, b).
    var rgb = light_hue2rgb(h)
    var rr = rgb[0]
    var gg = rgb[1]
    var bb = rgb[2]
    var az = Float32(2.0) * z - Float32(1.0)
    if az < 0.0:
        az = -az
    var c = (Float32(1.0) - az) * s
    return (
        (rr - Float32(0.5)) * c + z,
        (gg - Float32(0.5)) * c + z,
        (bb - Float32(0.5)) * c + z,
    )


def lighting_kernel(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    out: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    progress_in: Float32, count: Int, saturation: Float32, light: Float32,
):
    var gx = global_idx.x
    var gy = global_idx.y
    if gx >= UInt(W) or gy >= UInt(H):
        return
    var x = Int(gx)
    var y = Int(gy)
    var duration = Float32(1.0) / Float32(count)
    var progress = glsl_mod(progress_in, duration) / duration
    var amp = sin(progress * (PI_L / duration))
    if amp < 0.0:
        amp = -amp
    var hue = amp * Float32(360.0) / Float32(360.0)
    var value = saturation / Float32(10.0)
    # vHSV = (hue, saturation, value)
    var r = rebind[Scalar[dtype]](inp[y, x, 0])
    var g = rebind[Scalar[dtype]](inp[y, x, 1])
    var b = rebind[Scalar[dtype]](inp[y, x, 2])
    var a = rebind[Scalar[dtype]](inp[y, x, 3])
    var hsv = light_rgb2hsv(r, g, b)
    var fh = hsv[0] + hue
    var fs = hsv[1]
    var fv = hsv[2] + value
    fv = max(min(fv, Float32(1.0)), Float32(0.0))
    var mrgb = light_hsv2rgb(fh, fs, fv)
    var mr = mrgb[0]
    var mg = mrgb[1]
    var mb = mrgb[2]
    var hsl = light_rgb2hsl(mr, mg, mb)
    var lh = hsl[0]
    var ls = hsl[1] + saturation * Float32(0.5)
    var lz = hsl[2]
    ls = max(min(ls, Float32(1.0)), Float32(0.0))
    var rgbret = light_hsl2rgb(lh, ls, lz)
    var rr = rgbret[0]
    var rg = rgbret[1]
    var rb = rgbret[2]
    # result = vec4(fragRetRGB, a)*(1-light) + white*light
    out[y, x, 0] = rr * (Float32(1.0) - light) + Float32(1.0) * light
    out[y, x, 1] = rg * (Float32(1.0) - light) + Float32(1.0) * light
    out[y, x, 2] = rb * (Float32(1.0) - light) + Float32(1.0) * light
    out[y, x, 3] = a


# ===========================================================================
# 6) Jitter  (Jitter_shader.h:29-53)
# ===========================================================================
def jitter_kernel(
    inp: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    out: LayoutTensor[dtype, img_layout, MutAnyOrigin],
    progress_in: Float32, count: Int, max_scale: Float32, offset: Float32, shrink: Int,
):
    var gx = global_idx.x
    var gy = global_idx.y
    if gx >= UInt(W) or gy >= UInt(H):
        return
    var x = Int(gx)
    var y = Int(gy)
    var uvx = Float32(x) / Float32(W - 1)
    var uvy = Float32(y) / Float32(H - 1)
    var duration = Float32(1.0) / Float32(count)
    var progress = glsl_mod(progress_in, duration) / duration
    if shrink == 1 and progress > 0.5:
        progress = Float32(1.0) - progress
    var offc = offset * progress
    var scale = Float32(1.0) + (max_scale - Float32(1.0)) * progress
    var scx = fclamp(Float32(0.5) + (uvx - Float32(0.5)) / scale, Float32(0.0), Float32(1.0))
    var scy = fclamp(Float32(0.5) + (uvy - Float32(0.5)) / scale, Float32(0.0), Float32(1.0))
    var rx = fclamp(scx + offc, Float32(0.0), Float32(1.0))
    var ry = fclamp(scy + offc, Float32(0.0), Float32(1.0))
    var bx = fclamp(scx - offc, Float32(0.0), Float32(1.0))
    var by = fclamp(scy - offc, Float32(0.0), Float32(1.0))
    var rsx = trunc_coord(rx * Float32(W - 1))
    var rsy = trunc_coord(ry * Float32(H - 1))
    var bsx = trunc_coord(bx * Float32(W - 1))
    var bsy = trunc_coord(by * Float32(H - 1))
    var msx = trunc_coord(scx * Float32(W - 1))
    var msy = trunc_coord(scy * Float32(H - 1))
    # out = (mask_r.r, mask.g, mask_b.b, mask.a)  (Jitter_shader.h:52)
    out[y, x, 0] = sample_clamped(inp, rsx, rsy, 0)
    out[y, x, 1] = sample_clamped(inp, msx, msy, 1)
    out[y, x, 2] = sample_clamped(inp, bsx, bsy, 2)
    out[y, x, 3] = sample_clamped(inp, msx, msy, 3)


# ===========================================================================
# Test image + driver
# ===========================================================================
fn build_test_image(host: LayoutTensor[dtype, img_layout, MutAnyOrigin]):
    # diagonal gradient + bright rect + bright circle (deterministic)
    for y in range(H):
        for x in range(W):
            var diag = Float32(x + y) / Float32(W + H - 2)
            var r = diag
            var g = Float32(x) / Float32(W - 1)
            var b = Float32(y) / Float32(H - 1)
            if x >= 20 and x < 50 and y >= 20 and y < 50:
                r = Float32(0.95)
                g = Float32(0.9)
                b = Float32(0.1)
            var ddx = Float32(x - 90)
            var ddy = Float32(y - 90)
            if ddx * ddx + ddy * ddy < Float32(18 * 18):
                r = Float32(0.1)
                g = Float32(0.85)
                b = Float32(0.95)
            host[y, x, 0] = fclamp(r, Float32(0.0), Float32(1.0))
            host[y, x, 1] = fclamp(g, Float32(0.0), Float32(1.0))
            host[y, x, 2] = fclamp(b, Float32(0.0), Float32(1.0))
            host[y, x, 3] = Float32(1.0)


fn f32_to_u8(v: Float32) -> Int:
    # Non-finite guard (Star's 0.05/d at d==0). NaN: v != v. +inf: v > 1e30.
    # Saturate to 255 to match the numpy reference's identical rule.
    if v != v:
        return 255
    if v > Float32(1.0e30):
        return 255
    if v < Float32(-1.0e30):
        return 0
    var x = v * Float32(255.0) + Float32(0.5)
    if x < 0.0:
        x = Float32(0.0)
    if x > 255.0:
        x = Float32(255.0)
    return Int(x)


def image_txt_chunk(host: LayoutTensor[dtype, img_layout, MutAnyOrigin]) -> String:
    # space-separated uint8 tokens for one image (leading space; caller concatenates)
    var s = String("")
    for y in range(H):
        for x in range(W):
            for ch in range(4):
                s += " "
                s += String(f32_to_u8(rebind[Scalar[dtype]](host[y, x, ch])))
    return s


def main() raises:
    comptime if not has_accelerator():
        print("No GPU found")
        return
    var ctx = DeviceContext()

    var in_buf = ctx.enqueue_create_buffer[dtype](N)
    with in_buf.map_to_host() as hin:
        var t = LayoutTensor[dtype, img_layout](hin)
        build_test_image(t)
    var in_t = LayoutTensor[dtype, img_layout](in_buf)

    # input dump
    var sin_str = String("")
    with in_buf.map_to_host() as hin:
        var t = LayoutTensor[dtype, img_layout](hin)
        sin_str = image_txt_chunk(t)
    with open(String("/tmp/shader_effects_in.txt"), "w") as f:
        f.write(sin_str.strip())

    var gdim = (W, H)
    var bdim = (BLOCK, BLOCK)
    # accumulate combined output across all six effects (PSNR covers the whole group)
    var out_str = String("")

    # ---- WaterRipple (freq=24, amount=0.03) ----
    var wr_buf = ctx.enqueue_create_buffer[dtype](N)
    var wr_t = LayoutTensor[dtype, img_layout](wr_buf)
    ctx.enqueue_function[waterripple_kernel, waterripple_kernel](
        in_t, wr_t, Float32(24.0), Float32(0.03), grid_dim=gdim, block_dim=bdim)
    ctx.synchronize()
    with wr_buf.map_to_host() as h:
        var t = LayoutTensor[dtype, img_layout](h)
        out_str += image_txt_chunk(t)

    # ---- Soul (progress=0.37, count=1, max_scale=1.8, max_alpha=0.4, shrink=0) ----
    var soul_buf = ctx.enqueue_create_buffer[dtype](N)
    var soul_t = LayoutTensor[dtype, img_layout](soul_buf)
    ctx.enqueue_function[soul_kernel, soul_kernel](
        in_t, soul_t, Float32(0.37), 1, Float32(1.8), Float32(0.4), 0,
        grid_dim=gdim, block_dim=bdim)
    ctx.synchronize()
    with soul_buf.map_to_host() as h:
        var t = LayoutTensor[dtype, img_layout](h)
        out_str += image_txt_chunk(t)

    # ---- Sway (speed_arg=time*m_speed=0.37*20=7.4, strength=20, density=20, horizontal=1) ----
    var sway_buf = ctx.enqueue_create_buffer[dtype](N)
    var sway_t = LayoutTensor[dtype, img_layout](sway_buf)
    ctx.enqueue_function[sway_kernel, sway_kernel](
        in_t, sway_t, Float32(7.4), Float32(20.0), Float32(20.0), 1,
        grid_dim=gdim, block_dim=bdim)
    ctx.synchronize()
    with sway_buf.map_to_host() as h:
        var t = LayoutTensor[dtype, img_layout](h)
        out_str += image_txt_chunk(t)

    # ---- Star (progress=0.37, speed=10, layers=2, color=(1.0,0.1,0.9)) ----
    var star_buf = ctx.enqueue_create_buffer[dtype](N)
    var star_t = LayoutTensor[dtype, img_layout](star_buf)
    ctx.enqueue_function[star_kernel2, star_kernel2](
        in_t, star_t, Float32(0.37), Float32(10.0), Float32(2.0),
        Float32(1.0), Float32(0.1), Float32(0.9),
        grid_dim=gdim, block_dim=bdim)
    ctx.synchronize()
    with star_buf.map_to_host() as h:
        var t = LayoutTensor[dtype, img_layout](h)
        out_str += image_txt_chunk(t)

    # ---- Lighting (progress=0.37, count=1, saturation=0.3, light=0.3) ----
    var lit_buf = ctx.enqueue_create_buffer[dtype](N)
    var lit_t = LayoutTensor[dtype, img_layout](lit_buf)
    ctx.enqueue_function[lighting_kernel, lighting_kernel](
        in_t, lit_t, Float32(0.37), 1, Float32(0.3), Float32(0.3),
        grid_dim=gdim, block_dim=bdim)
    ctx.synchronize()
    with lit_buf.map_to_host() as h:
        var t = LayoutTensor[dtype, img_layout](h)
        out_str += image_txt_chunk(t)

    # ---- Jitter (progress=0.37, count=1, max_scale=1.1, offset=0.02, shrink=0) ----
    var jit_buf = ctx.enqueue_create_buffer[dtype](N)
    var jit_t = LayoutTensor[dtype, img_layout](jit_buf)
    ctx.enqueue_function[jitter_kernel, jitter_kernel](
        in_t, jit_t, Float32(0.37), 1, Float32(1.1), Float32(0.02), 0,
        grid_dim=gdim, block_dim=bdim)
    ctx.synchronize()
    with jit_buf.map_to_host() as h:
        var t = LayoutTensor[dtype, img_layout](h)
        out_str += image_txt_chunk(t)

    with open(String("/tmp/shader_effects_mojo.txt"), "w") as f:
        f.write(out_str.strip())

    print("effects: wrote waterripple/soul/sway/star/lighting/jitter combined mojo dump + input")
