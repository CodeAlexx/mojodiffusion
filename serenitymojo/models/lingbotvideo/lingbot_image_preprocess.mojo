# models/lingbotvideo/lingbot_image_preprocess.mojo
#
# Pure-Mojo replacement for the Python/PIL condition-image prep used by the
# LingBot-Video i2v path (parity/prep_dance_i2v.py `preprocess`). Given an image
# path and target (H, W) it returns a device Tensor of shape [1, 3, 1, H, W],
# f32 in [0, 1], produced by:
#   1. decode -> host RGB8 HWC  (serenitymojo.image.decode; PNG/JPEG/WEBP)
#   2. COVER resize: scale = max(W/iw, H/ih), new = (round(iw*s), round(ih*s))
#   3. CENTER-CROP to (W, H)
#   4. /255, arrange to CHW with a T=1 axis -> [1, 3, 1, H, W], upload to device
#
# Resampler note: serenitymojo/ops/resample.mojo is a torchaudio 1-D sinc
# (audio) resampler and is NOT applicable to a 2-D image bicubic. To match PIL's
# `Image.BICUBIC` (which is separable, antialiased on downscale, edge-clamped and
# renormalized), we implement that exact algorithm here on the host in f64, then
# upload the final tensor. This reproduces PIL's precompute_coeffs / ResampleHV.

from std.gpu.host import DeviceContext
from std.math import ceil

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.image.decode import decode_image, DecodedImage


# PIL bicubic kernel, a = -0.5 (Catmull-Rom), support = 2.0.
def _cubic(xin: Float64) -> Float64:
    var x = xin if xin >= 0.0 else -xin
    comptime a = Float64(-0.5)
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return 0.0


@fieldwise_init
struct _Coeffs(Movable):
    var kmin: List[Int]      # out_size: first source index per output pixel
    var kcount: List[Int]    # out_size: number of taps per output pixel
    var w: List[Float64]     # out_size * ksize: normalized weights (row-major)
    var ksize: Int


# Faithful port of PIL's precompute_coeffs (Resample.c). support=2.0 (bicubic).
def _precompute(in_size: Int, out_size: Int) raises -> _Coeffs:
    comptime support0 = Float64(2.0)
    var scale = Float64(in_size) / Float64(out_size)
    var filterscale = scale if scale >= 1.0 else Float64(1.0)
    var support = support0 * filterscale
    var ksize = Int(ceil(support)) * 2 + 1

    var kmin = List[Int]()
    var kcount = List[Int]()
    var w = List[Float64]()
    kmin.resize(out_size, 0)
    kcount.resize(out_size, 0)
    w.resize(out_size * ksize, Float64(0.0))

    var ss = Float64(1.0) / filterscale
    for xx in range(out_size):
        var center = (Float64(xx) + 0.5) * scale
        var xmin = Int(center - support + 0.5)
        if xmin < 0:
            xmin = 0
        var xmax = Int(center + support + 0.5)
        if xmax > in_size:
            xmax = in_size
        var n = xmax - xmin
        var ww = Float64(0.0)
        for x in range(n):
            var wv = _cubic((Float64(xmin + x) - center + 0.5) * ss)
            w[xx * ksize + x] = wv
            ww += wv
        if ww != 0.0:
            for x in range(n):
                w[xx * ksize + x] = w[xx * ksize + x] / ww
        kmin[xx] = xmin
        kcount[xx] = n
    return _Coeffs(kmin^, kcount^, w^, ksize)


# Horizontal pass: [ih, iw, 3] -> [ih, ow, 3]. src/dst are f64, HWC.
def _resample_h(
    src: List[Float64], ih: Int, iw: Int, ow: Int, c: _Coeffs
) raises -> List[Float64]:
    var dst = List[Float64]()
    dst.resize(ih * ow * 3, Float64(0.0))
    for y in range(ih):
        var row = y * iw * 3
        for xx in range(ow):
            var xmin = c.kmin[xx]
            var n = c.kcount[xx]
            var base = xx * c.ksize
            var a0 = Float64(0.0)
            var a1 = Float64(0.0)
            var a2 = Float64(0.0)
            for k in range(n):
                var wv = c.w[base + k]
                var sidx = row + (xmin + k) * 3
                a0 += wv * src[sidx + 0]
                a1 += wv * src[sidx + 1]
                a2 += wv * src[sidx + 2]
            var didx = (y * ow + xx) * 3
            dst[didx + 0] = a0
            dst[didx + 1] = a1
            dst[didx + 2] = a2
    return dst^


# Vertical pass: [ih, ow, 3] -> [oh, ow, 3].
def _resample_v(
    src: List[Float64], ih: Int, ow: Int, oh: Int, c: _Coeffs
) raises -> List[Float64]:
    var dst = List[Float64]()
    dst.resize(oh * ow * 3, Float64(0.0))
    for yy in range(oh):
        var ymin = c.kmin[yy]
        var n = c.kcount[yy]
        var base = yy * c.ksize
        for x in range(ow):
            var a0 = Float64(0.0)
            var a1 = Float64(0.0)
            var a2 = Float64(0.0)
            for k in range(n):
                var wv = c.w[base + k]
                var sidx = ((ymin + k) * ow + x) * 3
                a0 += wv * src[sidx + 0]
                a1 += wv * src[sidx + 1]
                a2 += wv * src[sidx + 2]
            var didx = (yy * ow + x) * 3
            dst[didx + 0] = a0
            dst[didx + 1] = a1
            dst[didx + 2] = a2
    return dst^


# Cover-resize + center-crop an RGB8 HWC image to (H, W), returning host f64 HWC
# pixels in [0, 255] cropped to exactly H*W*3. Matches prep_dance_i2v.preprocess.
def _cover_resize_crop(
    img: DecodedImage, H: Int, W: Int
) raises -> List[Float64]:
    var iw = img.width
    var ih = img.height
    # COVER scale + PIL round-half-away rounding (round(x) == floor(x+0.5) for >0)
    var sw = Float64(W) / Float64(iw)
    var sh = Float64(H) / Float64(ih)
    var s = sw if sw > sh else sh
    var rw = Int(Float64(iw) * s + 0.5)
    var rh = Int(Float64(ih) * s + 0.5)
    if rw < W:
        rw = W
    if rh < H:
        rh = H

    # uint8 HWC -> f64 HWC
    var srcf = List[Float64]()
    srcf.resize(ih * iw * 3, Float64(0.0))
    for i in range(ih * iw * 3):
        srcf[i] = Float64(img.rgb[i])

    var ch = _precompute(iw, rw)
    var cv = _precompute(ih, rh)
    var h_out = _resample_h(srcf, ih, iw, rw, ch)
    var v_out = _resample_v(h_out, ih, rw, rh, cv)

    # center-crop to (W, H)
    var left = (rw - W) // 2
    var top = (rh - H) // 2
    var crop = List[Float64]()
    crop.resize(H * W * 3, Float64(0.0))
    for h in range(H):
        for wcol in range(W):
            var sidx = ((top + h) * rw + (left + wcol)) * 3
            var didx = (h * W + wcol) * 3
            crop[didx + 0] = v_out[sidx + 0]
            crop[didx + 1] = v_out[sidx + 1]
            crop[didx + 2] = v_out[sidx + 2]
    return crop^


# Public entry: load an image and return the i2v condition tensor
# [1, 3, 1, H, W] f32 in [0, 1] on `ctx`. PNG/JPEG always; WEBP when libwebp.so.7
# is present (else pre-convert to PNG/JPEG).
def load_condition_image(
    path: String, H: Int, W: Int, ctx: DeviceContext, drop_alpha: Bool = False
) raises -> Tensor:
    # drop_alpha=True matches PIL convert("RGB") (drops alpha) instead of the
    # default composite-over-white — use it when the i2v reference used PIL on a
    # transparent asset (same fix as the ti2v vision-preprocess path).
    var img = decode_image(path, drop_alpha=drop_alpha)
    var crop = _cover_resize_crop(img, H, W)  # H*W*3 f64 HWC in [0,255]
    # -> CHW with T=1, /255, clamped to [0,1]. Flat order for [1,3,1,H,W] is
    #    (c*H + h)*W + w, value = crop[(h*W+w)*3 + c] / 255.
    var vals = List[Float32]()
    vals.resize(3 * H * W, Float32(0.0))
    for c in range(3):
        for h in range(H):
            for wcol in range(W):
                var v = crop[(h * W + wcol) * 3 + c] / 255.0
                if v < 0.0:
                    v = 0.0
                elif v > 1.0:
                    v = 1.0
                vals[(c * H + h) * W + wcol] = Float32(v)
    return Tensor.from_host(vals, [1, 3, 1, H, W], STDtype.F32, ctx)


# Public: bicubic-resize a whole video clip. Mirrors creator resize_video_tensor
# (lingbot_video/utils.py 199-206: torch bicubic F.interpolate + clamp[0,1]).
# `clip` is NCTHW [1,3,T,ih,iw] F32 in [0,1]; returns NCTHW [1,3,T,oh,ow] F32 in
# [0,1]. Each frame is resized independently with the SAME PIL-exact separable
# bicubic used above (_precompute + _resample_h then _resample_v, f64 HWC in
# [0,255]), then /255 and clamped. Coeffs are precomputed once (all frames share
# the same in/out sizes).
def resize_video_bicubic(
    clip: Tensor, oh: Int, ow: Int, ctx: DeviceContext
) raises -> Tensor:
    var cs = clip.shape()
    if len(cs) != 5 or cs[0] != 1 or cs[1] != 3:
        raise Error("resize_video_bicubic expects clip [1,3,T,ih,iw]")
    var T = cs[2]
    var ih = cs[3]
    var iw = cs[4]
    var host = clip.to_host(ctx)  # NCTHW flat: ((c*T + t)*ih + h)*iw + w

    var ch = _precompute(iw, ow)  # horizontal: iw -> ow
    var cv = _precompute(ih, oh)  # vertical:   ih -> oh

    var out = List[Float32]()
    out.resize(3 * T * oh * ow, Float32(0.0))

    var srcf = List[Float64]()
    srcf.resize(ih * iw * 3, Float64(0.0))

    for t in range(T):
        # NCTHW frame -> HWC f64 in [0,255]  (values are [0,1] -> x255)
        for c in range(3):
            var cbase = ((c * T + t) * ih) * iw
            for h in range(ih):
                var rbase = cbase + h * iw
                for wcol in range(iw):
                    srcf[(h * iw + wcol) * 3 + c] = \
                        Float64(host[rbase + wcol]) * 255.0
        var h_out = _resample_h(srcf, ih, iw, ow, ch)   # [ih,ow,3]
        var v_out = _resample_v(h_out, ih, ow, oh, cv)   # [oh,ow,3]
        # HWC f64 [0,255] -> NCTHW F32 [0,1] clamped
        for c in range(3):
            var obase = ((c * T + t) * oh) * ow
            for h in range(oh):
                var orow = obase + h * ow
                for wcol in range(ow):
                    var v = v_out[(h * ow + wcol) * 3 + c] / 255.0
                    if v < 0.0:
                        v = 0.0
                    elif v > 1.0:
                        v = 1.0
                    out[orow + wcol] = Float32(v)

    return Tensor.from_host(out, [1, 3, T, oh, ow], STDtype.F32, ctx)
