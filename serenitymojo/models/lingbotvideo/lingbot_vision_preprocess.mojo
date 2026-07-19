# models/lingbotvideo/lingbot_vision_preprocess.mojo
#
# Pure-Mojo replacement for the Qwen3-VL processor's IMAGE branch used by the
# LingBot-Video VLM-grounded i2v/ti2v path (parity/prep_ti2v.py). Produces the
# vision-tower input with NO Python:
#   pixel_values  [grid_t*grid_h*grid_w, 1536]  f32 in [-1, 1]  (device)
#   grid_thw = (grid_t, grid_h, grid_w)
#
# Reference: Qwen2VLImageProcessor(Fast) in the LingBot processor
#   (processor/preprocessor_config.json): patch_size 16, temporal_patch_size 2,
#   merge_size 2, image_mean/std [0.5]x3, min_pixels 65536, max_pixels 16777216,
#   BICUBIC resize. smart_resize(factor=32) -> (h_bar, w_bar); the single image
#   frame is temporally duplicated (T=temporal_patch_size=2); patches are laid out
#   in merge-block order (the vision tower's preshuffle merger consumes exactly this
#   row order):
#     row = ((((t*gh_b + bh)*gw_b + bw)*merge + mh)*merge + mw)     (grid_t == 1)
#     col = ((c*temporal + tp)*patch + ph)*patch + pw
#   with gh_b = grid_h//merge, gw_b = grid_w//merge.
#
# Bicubic reuses the PIL-exact separable resampler from lingbot_image_preprocess
# (torchvision Fast antialiased bicubic ~ PIL bicubic; gated by pixel cos).

from std.gpu.host import DeviceContext
from std.math import sqrt, floor, ceil

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.image.decode import decode_image
from serenitymojo.models.lingbotvideo.lingbot_image_preprocess import (
    _precompute,
    _resample_h,
    _resample_v,
)

comptime PATCH = 16
comptime TEMPORAL = 2
comptime MERGE = 2
comptime FACTOR = 32          # PATCH * MERGE
comptime MIN_PIXELS = 65536
comptime MAX_PIXELS = 16777216
comptime ROW_DIM = 1536       # 3 * TEMPORAL * PATCH * PATCH


@fieldwise_init
struct VisionPreproc(Movable):
    var pixel_values: Tensor   # [seq, 1536] f32 device
    var grid_t: Int
    var grid_h: Int
    var grid_w: Int
    var seq: Int


# Python round() = round-half-to-even (banker's), matching smart_resize's round().
def _py_round(x: Float64) -> Int:
    var fl = floor(x)
    var diff = x - fl
    var fli = Int(fl)
    if diff < 0.5:
        return fli
    if diff > 0.5:
        return fli + 1
    # exactly .5 -> nearest even
    if fli % 2 == 0:
        return fli
    return fli + 1


def _round_by(x: Float64, f: Int) -> Int:
    return _py_round(x / Float64(f)) * f


def _floor_by(x: Float64, f: Int) -> Int:
    return Int(floor(x / Float64(f))) * f


def _ceil_by(x: Float64, f: Int) -> Int:
    return Int(ceil(x / Float64(f))) * f


# Qwen2VL smart_resize: keep aspect, snap each side to FACTOR, clamp pixel area.
def _smart_resize(h: Int, w: Int) -> Tuple[Int, Int]:
    var hb = _round_by(Float64(h), FACTOR)
    var wb = _round_by(Float64(w), FACTOR)
    if hb < FACTOR:
        hb = FACTOR
    if wb < FACTOR:
        wb = FACTOR
    if hb * wb > MAX_PIXELS:
        var beta = sqrt(Float64(h) * Float64(w) / Float64(MAX_PIXELS))
        hb = _floor_by(Float64(h) / beta, FACTOR)
        wb = _floor_by(Float64(w) / beta, FACTOR)
    elif hb * wb < MIN_PIXELS:
        var beta = sqrt(Float64(MIN_PIXELS) / (Float64(h) * Float64(w)))
        hb = _ceil_by(Float64(h) * beta, FACTOR)
        wb = _ceil_by(Float64(w) * beta, FACTOR)
    return (hb, wb)


# Load an image and produce the Qwen3-VL vision-tower pixel_values + grid.
def lingbot_vision_preprocess(
    path: String, ctx: DeviceContext
) raises -> VisionPreproc:
    # drop_alpha=True to match the Qwen3-VL processor's PIL Image.convert("RGB")
    # (raw RGB, ignore alpha) — NOT composite-over-white.
    var img = decode_image(path, True)
    var iw = img.width
    var ih = img.height
    var hw = _smart_resize(ih, iw)
    var h_bar = hw[0]
    var w_bar = hw[1]
    var grid_h = h_bar // PATCH
    var grid_w = w_bar // PATCH
    var grid_t = 1
    var seq = grid_t * grid_h * grid_w

    # uint8 HWC -> f64 HWC in [0, 255]
    var srcf = List[Float64]()
    srcf.resize(ih * iw * 3, Float64(0.0))
    for i in range(ih * iw * 3):
        srcf[i] = Float64(img.rgb[i])

    # BICUBIC resize to (h_bar, w_bar): horizontal then vertical (separable).
    var ch = _precompute(iw, w_bar)
    var cv = _precompute(ih, h_bar)
    var h_out = _resample_h(srcf, ih, iw, w_bar, ch)      # [ih, w_bar, 3]
    var v_out = _resample_v(h_out, ih, w_bar, h_bar, cv)  # [h_bar, w_bar, 3]

    # normalize (v/255 - 0.5)/0.5 = v/127.5 - 1 -> [-1, 1]; patchify to
    # [seq, 1536] in merge-block row order (grid_t == 1). The single frame is
    # temporally duplicated, so tp=0 and tp=1 carry identical values.
    var gh_b = grid_h // MERGE
    var gw_b = grid_w // MERGE
    var vals = List[Float32]()
    vals.resize(seq * ROW_DIM, Float32(0.0))
    for bh in range(gh_b):
        for bw in range(gw_b):
            for mh in range(MERGE):
                for mw in range(MERGE):
                    var gh = bh * MERGE + mh
                    var gw = bw * MERGE + mw
                    var row = ((bh * gw_b + bw) * MERGE + mh) * MERGE + mw
                    var rbase = row * ROW_DIM
                    for c in range(3):
                        for tp in range(TEMPORAL):
                            var cbase = rbase + (c * TEMPORAL + tp) * PATCH * PATCH
                            for ph in range(PATCH):
                                var py = gh * PATCH + ph
                                for pw in range(PATCH):
                                    var px = gw * PATCH + pw
                                    var v = (
                                        v_out[(py * w_bar + px) * 3 + c] / 127.5
                                        - 1.0
                                    )
                                    vals[cbase + ph * PATCH + pw] = Float32(v)

    var pv = Tensor.from_host(vals, [seq, ROW_DIM], STDtype.F32, ctx)
    return VisionPreproc(pv^, grid_t, grid_h, grid_w, seq)


# ── bucketed variant (image captioner): fit ANY image to a fixed grid bucket ───
# Same decode/normalize/patchify as lingbot_vision_preprocess, but instead of
# smart_resize (arbitrary AR-preserving grid → comptime-S dispatch impossible),
# squash-resize to the nearest of FOUR fixed buckets so the caller can comptime-
# instantiate the vision tower per bucket:
#   square small   512x512   (grid 32x32, S=1024)
#   landscape      512x768   (h_bar 512, w_bar 768; grid 32x48, S=1536)
#   portrait       768x512   (h_bar 768, w_bar 512; grid 48x32, S=1536)
#   square large  1024x1024  (grid 64x64, S=4096; picked when min side >= 1024)
# Bucket choice: aspect ratio (>1.25 landscape, <0.8 portrait, else square).
# The residual squash after picking the closest-AR bucket is the standard
# bucketing tradeoff (mild distortion, no cropping, nothing dropped).
def lingbot_vision_preprocess_bucketed(
    path: String, ctx: DeviceContext
) raises -> VisionPreproc:
    var img = decode_image(path, True)
    var iw = img.width
    var ih = img.height
    var ar = Float64(iw) / Float64(ih)
    var mn = ih if ih < iw else iw
    # Size tier by SOURCE min side (never upscale past native): 512-class,
    # 1024-class, 1536-class, 2048-class. AR class picks square/portrait/landscape.
    var h_bar = 512
    var w_bar = 512
    if ar > 1.25:
        # landscape: h x w = 512x768 or 1024x1536
        if mn >= 1024:
            h_bar = 1024
            w_bar = 1536
        else:
            h_bar = 512
            w_bar = 768
    elif ar < 0.8:
        # portrait: h x w = 768x512 or 1536x1024
        if mn >= 1024:
            h_bar = 1536
            w_bar = 1024
        else:
            h_bar = 768
            w_bar = 512
    else:
        # square-ish: 512 / 1024 / 1536 / 2048
        if mn >= 2048:
            h_bar = 2048
            w_bar = 2048
        elif mn >= 1536:
            h_bar = 1536
            w_bar = 1536
        elif mn >= 1024:
            h_bar = 1024
            w_bar = 1024
    var grid_h = h_bar // PATCH
    var grid_w = w_bar // PATCH
    var grid_t = 1
    var seq = grid_t * grid_h * grid_w

    var srcf = List[Float64]()
    srcf.resize(ih * iw * 3, Float64(0.0))
    for i in range(ih * iw * 3):
        srcf[i] = Float64(img.rgb[i])
    var ch = _precompute(iw, w_bar)
    var cv = _precompute(ih, h_bar)
    var h_out = _resample_h(srcf, ih, iw, w_bar, ch)
    var v_out = _resample_v(h_out, ih, w_bar, h_bar, cv)

    var gh_b = grid_h // MERGE
    var gw_b = grid_w // MERGE
    var vals = List[Float32]()
    vals.resize(seq * ROW_DIM, Float32(0.0))
    for bh in range(gh_b):
        for bw in range(gw_b):
            for mh in range(MERGE):
                for mw in range(MERGE):
                    var gh = bh * MERGE + mh
                    var gw = bw * MERGE + mw
                    var row = ((bh * gw_b + bw) * MERGE + mh) * MERGE + mw
                    var rbase = row * ROW_DIM
                    for c in range(3):
                        for tp in range(TEMPORAL):
                            var cbase = rbase + (c * TEMPORAL + tp) * PATCH * PATCH
                            for ph in range(PATCH):
                                var py = gh * PATCH + ph
                                for pw in range(PATCH):
                                    var px = gw * PATCH + pw
                                    var v = (
                                        v_out[(py * w_bar + px) * 3 + c] / 127.5
                                        - 1.0
                                    )
                                    vals[cbase + ph * PATCH + pw] = Float32(v)

    var pv = Tensor.from_host(vals, [seq, ROW_DIM], STDtype.F32, ctx)
    return VisionPreproc(pv^, grid_t, grid_h, grid_w, seq)
