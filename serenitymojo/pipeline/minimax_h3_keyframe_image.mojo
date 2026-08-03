# serenitymojo/pipeline/minimax_h3_keyframe_image.mojo — put a MiniMax-H3
# keyframe onto the target canvas. Host only: no Tensor, no DeviceContext, no
# GPU.
#
# ── WHAT THE VENDOR DOES (packing.py:216-243, `prepare_keyframe_image`) ──────
#   * An image that ALREADY is the canvas is returned untouched — no resampling
#     pass at all, so a pre-sized keyframe is bit-exact by construction.
#   * The FIRST keyframe of a request is the geometry anchor and is STRETCHED
#     onto the canvas: `image.resize((width, height), Image.Resampling.LANCZOS)`.
#     Aspect ratio is NOT preserved. It does not need to be: the canvas itself
#     was resolved from this image's aspect ratio (before_encoder.py:172-173),
#     so the stretch is a no-op up to the 32-pixel canvas rounding.
#   * A SECOND keyframe follows that canvas and is COVER-CROPPED: scale by
#     `max(width/w, height/h)`, LANCZOS-resize to that, then centre-crop.
#
# L2VA IS NOT "THE SECOND KEYFRAME ALONE". `keyframes` is built by filtering out
# the missing one (before_encoder.py:162-166) and `stretch=index == 0`
# (:191) keys off the position in THAT filtered list — so a request with only
# `last_image` has it at index 0 and it is STRETCHED, and the canvas comes from
# ITS aspect ratio. Cover-cropping it would be wrong. The `stretch` flag here is
# therefore the caller's to set from the filtered index, and
# `minimax_h3_prepare_keyframes` below does that filtering itself so the rule
# cannot be re-derived incorrectly at a call site.
#
# ── PIL's LANCZOS, REPRODUCED RATHER THAN APPROXIMATED ───────────────────────
# `ffmpeg -vf scale=...:flags=lanczos` is the obvious route (this repo already
# shells out to ffmpeg for video ingest) and it is NOT the same filter: swscale
# builds its own coefficient table, works in a different intermediate precision,
# and its `param0` window differs from PIL's fixed 3-lobe support. Since the
# resampled keyframe is what the VAE encodes into the conditioning anchor, that
# is a difference the model sees. So this file ports Pillow's own
# `ImagingResample` instead (`src/libImaging/Resample.c`), which is plain host
# integer arithmetic and can be — and is — gated BIT-EXACT against PIL:
#
#   1. `precompute_coeffs`: `filterscale = max(1, in/out)` (upscaling keeps the
#      filter at its natural width, downscaling widens it — this is where the
#      antialiasing comes from), `support = 3 * filterscale`, one coefficient
#      run per output pixel, normalized to sum 1 in DOUBLE.
#   2. `normalize_coeffs_8bpc`: coefficients quantized to fixed point at
#      PRECISION_BITS = 22, rounded HALF AWAY FROM ZERO (`(int)(±0.5 + v*2^22)`).
#   3. Two passes, HORIZONTAL then VERTICAL, with an 8-BIT INTERMEDIATE between
#      them. Doing it in one fused pass, or keeping the intermediate in float,
#      gives a visibly similar but numerically different image.
#   4. Accumulator primed at `1 << (PRECISION_BITS - 1)` (that is the rounding),
#      then `>> PRECISION_BITS` and clamped to [0, 255].
#
# Gate: pipeline/parity/minimax_h3_keyframe_image_probe.mojo, against real PIL
# on real images. Host, no GPU.

from std.collections import List
from std.math import sin, ceil

from serenitymojo.models.minimax_h3.packing import _round_half_even

# Pillow's `struct filter LANCZOS = {lanczos_filter, 3.0}`.
comptime _LANCZOS_SUPPORT = Float64(3.0)
# `#define PRECISION_BITS (32 - 8 - 2)` — 8 bits of result plus two spare, since
# a lanczos kernel has negative lobes and its partial sums overshoot both ends.
comptime _PRECISION_BITS = 22
comptime _PI = Float64(3.14159265358979323846264338327950288)


@fieldwise_init
struct MiniMaxH3RgbImage(Copyable, Movable):
    """8-bit RGB, channels-LAST, flat `[height, width, 3]` row-major — the same
    layout `pipeline/minimax_h3_media_in.mojo` hands back for a decoded frame,
    so a keyframe and a video frame are the same bytes here."""

    var pixels: List[UInt8]
    var height: Int
    var width: Int

    def numel(self) -> Int:
        return self.height * self.width * 3

    def validate(self) raises:
        if self.height < 1 or self.width < 1:
            raise Error("minimax_h3_keyframe_image: image geometry must be positive")
        if len(self.pixels) != self.numel():
            raise Error(
                String("minimax_h3_keyframe_image: buffer holds ")
                + String(len(self.pixels)) + " bytes, geometry implies "
                + String(self.numel())
            )


@fieldwise_init
struct _CoeffTable(Movable):
    """One axis' resampling plan: per output pixel a source start, a run length,
    and `ksize` fixed-point coefficients."""

    var bounds_min: List[Int]
    var bounds_len: List[Int]
    var kk: List[Int32]
    var ksize: Int


def _sinc(x: Float64) -> Float64:
    """`sinc_filter` (Resample.c). The `x == 0` short-circuit is load-bearing:
    the limit is 1, the expression is 0/0."""
    if x == 0.0:
        return 1.0
    var xp = x * _PI
    return sin(xp) / xp


def _lanczos(x: Float64) -> Float64:
    """`lanczos_filter`: a sinc windowed by a 3-lobe sinc. Note the ASYMMETRIC
    guard `-3.0 <= x < 3.0` — Pillow's own, kept rather than symmetrized."""
    if x >= -3.0 and x < 3.0:
        return _sinc(x) * _sinc(x / 3.0)
    return 0.0


def _precompute_coeffs(in_size: Int, out_size: Int) raises -> _CoeffTable:
    """`precompute_coeffs` + `normalize_coeffs_8bpc`, for the full-image box
    (`in0 = 0`, `in1 = in_size`) — the only box a plain `resize` uses."""
    if in_size < 1 or out_size < 1:
        raise Error("minimax_h3_keyframe_image: resample sizes must be positive")

    var scale = Float64(in_size) / Float64(out_size)
    var filterscale = scale
    if filterscale < 1.0:
        filterscale = 1.0
    var support = _LANCZOS_SUPPORT * filterscale
    var ksize = Int(ceil(support)) * 2 + 1

    var bounds_min = List[Int](capacity=out_size)
    var bounds_len = List[Int](capacity=out_size)
    var kk = List[Int32]()
    kk.resize(out_size * ksize, Int32(0))

    var prekk = List[Float64]()
    prekk.resize(ksize, Float64(0.0))

    for xx in range(out_size):
        var center = (Float64(xx) + 0.5) * scale
        var ww = Float64(0.0)
        var ss = 1.0 / filterscale
        # `(int)` truncates toward zero; the clamps then follow.
        var xmin = Int(center - support + 0.5)
        if xmin < 0:
            xmin = 0
        var xmax = Int(center + support + 0.5)
        if xmax > in_size:
            xmax = in_size
        xmax -= xmin

        for x in range(xmax):
            var w = _lanczos((Float64(x + xmin) - center + 0.5) * ss)
            prekk[x] = w
            ww += w
        for x in range(xmax):
            if ww != 0.0:
                prekk[x] = prekk[x] / ww
        for x in range(xmax, ksize):
            prekk[x] = 0.0

        # Fixed point, rounded HALF AWAY FROM ZERO — the sign branch is not
        # cosmetic, lanczos coefficients are routinely negative.
        for x in range(ksize):
            var v = prekk[x]
            var q: Float64
            if v < 0.0:
                q = -0.5 + v * Float64(1 << _PRECISION_BITS)
            else:
                q = 0.5 + v * Float64(1 << _PRECISION_BITS)
            kk[xx * ksize + x] = Int32(Int(q))

        bounds_min.append(xmin)
        bounds_len.append(xmax)

    return _CoeffTable(bounds_min^, bounds_len^, kk^, ksize)


def _clip8(value: Int) -> UInt8:
    """`clip8`: arithmetic shift down by PRECISION_BITS, then clamp. Pillow does
    the clamp with a lookup table centred at 640 purely to stay branchless; the
    arithmetic it implements is this."""
    var shifted = value >> _PRECISION_BITS
    if shifted < 0:
        return UInt8(0)
    if shifted > 255:
        return UInt8(255)
    return UInt8(shifted)


def _resample_horizontal(
    src: List[UInt8], src_h: Int, src_w: Int, out_w: Int, table: _CoeffTable
) raises -> List[UInt8]:
    """`ImagingResampleHorizontal_8bpc`, 3 bands. `[src_h, src_w, 3]` ->
    `[src_h, out_w, 3]`, 8-bit out (the intermediate really is 8-bit)."""
    var out = List[UInt8]()
    out.resize(src_h * out_w * 3, UInt8(0))
    var round_bias = 1 << (_PRECISION_BITS - 1)
    for yy in range(src_h):
        var row_in = yy * src_w * 3
        var row_out = yy * out_w * 3
        for xx in range(out_w):
            var xmin = table.bounds_min[xx]
            var xlen = table.bounds_len[xx]
            var koff = xx * table.ksize
            var s0 = round_bias
            var s1 = round_bias
            var s2 = round_bias
            for x in range(xlen):
                var k = Int(table.kk[koff + x])
                var p = row_in + (xmin + x) * 3
                s0 += Int(src[p]) * k
                s1 += Int(src[p + 1]) * k
                s2 += Int(src[p + 2]) * k
            var q = row_out + xx * 3
            out[q] = _clip8(s0)
            out[q + 1] = _clip8(s1)
            out[q + 2] = _clip8(s2)
    return out^


def _resample_vertical(
    src: List[UInt8], src_h: Int, src_w: Int, out_h: Int, table: _CoeffTable
) raises -> List[UInt8]:
    """`ImagingResampleVertical_8bpc`, 3 bands. `[src_h, src_w, 3]` ->
    `[out_h, src_w, 3]`."""
    var out = List[UInt8]()
    out.resize(out_h * src_w * 3, UInt8(0))
    var round_bias = 1 << (_PRECISION_BITS - 1)
    for yy in range(out_h):
        var ymin = table.bounds_min[yy]
        var ylen = table.bounds_len[yy]
        var koff = yy * table.ksize
        var row_out = yy * src_w * 3
        for xx in range(src_w):
            var s0 = round_bias
            var s1 = round_bias
            var s2 = round_bias
            for y in range(ylen):
                var k = Int(table.kk[koff + y])
                var p = ((ymin + y) * src_w + xx) * 3
                s0 += Int(src[p]) * k
                s1 += Int(src[p + 1]) * k
                s2 += Int(src[p + 2]) * k
            var q = row_out + xx * 3
            out[q] = _clip8(s0)
            out[q + 1] = _clip8(s1)
            out[q + 2] = _clip8(s2)
    return out^


def minimax_h3_lanczos_resize(
    image: MiniMaxH3RgbImage, out_height: Int, out_width: Int
) raises -> MiniMaxH3RgbImage:
    """`PIL.Image.resize((out_width, out_height), Image.Resampling.LANCZOS)`.

    Horizontal pass first, then vertical, with the 8-bit intermediate Pillow
    itself keeps. An axis that does not change size is SKIPPED entirely, as
    Pillow skips it (`need_horizontal` / `need_vertical`,
    `ImagingResampleInner`) — which is why resizing to the same size returns the
    input unchanged rather than passing it through a unity kernel."""
    image.validate()
    if out_height < 1 or out_width < 1:
        raise Error("minimax_h3_keyframe_image: output geometry must be positive")

    var need_h = out_width != image.width
    var need_v = out_height != image.height
    if not need_h and not need_v:
        return MiniMaxH3RgbImage(image.pixels.copy(), image.height, image.width)

    var cur = image.pixels.copy()
    var cur_h = image.height
    var cur_w = image.width
    if need_h:
        var htable = _precompute_coeffs(image.width, out_width)
        cur = _resample_horizontal(cur, cur_h, cur_w, out_width, htable)
        cur_w = out_width
    if need_v:
        var vtable = _precompute_coeffs(image.height, out_height)
        cur = _resample_vertical(cur, cur_h, cur_w, out_height, vtable)
        cur_h = out_height
    return MiniMaxH3RgbImage(cur^, cur_h, cur_w)


def minimax_h3_exif_transpose(
    image: MiniMaxH3RgbImage, orientation: Int
) raises -> MiniMaxH3RgbImage:
    """`PIL.ImageOps.exif_transpose` for an already-parsed EXIF orientation.

    The vendor runs `ImageOps.exif_transpose(keyframe).convert("RGB")` on every
    keyframe BEFORE the canvas is resolved (before_encoder.py:162-166), so the
    orientation decides the aspect ratio the whole request is built around, not
    just which way the picture faces. ffmpeg does NOT apply the JPEG EXIF
    orientation tag on decode (it honours a container display matrix, which is a
    different thing), so skipping this would land a sideways conditioning anchor
    on a canvas resolved from the wrong aspect ratio — with nothing downstream
    able to notice.

    Orientation 0 (absent) and 1 are identity. 5-8 SWAP the axes."""
    image.validate()
    if orientation <= 1:
        return MiniMaxH3RgbImage(image.pixels.copy(), image.height, image.width)
    if orientation > 8:
        raise Error(
            String("minimax_h3_keyframe_image: EXIF orientation ")
            + String(orientation) + " is not one of 1..8"
        )

    var h = image.height
    var w = image.width
    var swaps = orientation >= 5
    var out_h = w if swaps else h
    var out_w = h if swaps else w
    var out = List[UInt8]()
    out.resize(out_h * out_w * 3, UInt8(0))

    for y in range(out_h):
        for x in range(out_w):
            var sy: Int
            var sx: Int
            if orientation == 2:      # FLIP_LEFT_RIGHT
                sy = y
                sx = w - 1 - x
            elif orientation == 3:    # ROTATE_180
                sy = h - 1 - y
                sx = w - 1 - x
            elif orientation == 4:    # FLIP_TOP_BOTTOM
                sy = h - 1 - y
                sx = x
            elif orientation == 5:    # TRANSPOSE
                sy = x
                sx = y
            elif orientation == 6:    # ROTATE_270 (clockwise quarter turn)
                sy = h - 1 - x
                sx = y
            elif orientation == 7:    # TRANSVERSE
                sy = h - 1 - x
                sx = w - 1 - y
            else:                     # 8: ROTATE_90 (counter-clockwise)
                sy = x
                sx = w - 1 - y
            var src = (sy * w + sx) * 3
            var dst = (y * out_w + x) * 3
            out[dst] = image.pixels[src]
            out[dst + 1] = image.pixels[src + 1]
            out[dst + 2] = image.pixels[src + 2]
    return MiniMaxH3RgbImage(out^, out_h, out_w)


def minimax_h3_crop(
    image: MiniMaxH3RgbImage, left: Int, top: Int, out_height: Int, out_width: Int
) raises -> MiniMaxH3RgbImage:
    """`PIL.Image.crop((left, top, left + width, top + height))`, restricted to
    a box that lies inside the image — the vendor's cover-crop always does, and
    a box that did not would need PIL's zero-fill semantics, which are not
    reproduced here so they cannot be silently relied on."""
    image.validate()
    if left < 0 or top < 0 or out_height < 1 or out_width < 1:
        raise Error("minimax_h3_keyframe_image: crop box must be positive")
    if left + out_width > image.width or top + out_height > image.height:
        raise Error(
            String("minimax_h3_keyframe_image: crop box ") + String(out_width)
            + "x" + String(out_height) + "+" + String(left) + "+" + String(top)
            + " leaves the " + String(image.width) + "x" + String(image.height)
            + " image"
        )
    var out = List[UInt8]()
    out.resize(out_height * out_width * 3, UInt8(0))
    for y in range(out_height):
        var src = ((top + y) * image.width + left) * 3
        var dst = y * out_width * 3
        for i in range(out_width * 3):
            out[dst + i] = image.pixels[src + i]
    return MiniMaxH3RgbImage(out^, out_height, out_width)


def minimax_h3_prepare_keyframe_image(
    image: MiniMaxH3RgbImage, height: Int, width: Int, stretch: Bool
) raises -> MiniMaxH3RgbImage:
    """`prepare_keyframe_image` (packing.py:216-243), op for op.

    `stretch` selects the geometry anchor's path; see this file's header for why
    it is the FILTERED index that decides it, not "is this the last frame"."""
    image.validate()
    if image.width == width and image.height == height:
        return MiniMaxH3RgbImage(image.pixels.copy(), image.height, image.width)
    if stretch:
        return minimax_h3_lanczos_resize(image, height, width)

    # Cover crop: aspect-preserving max-scale resize, then centre crop.
    var sx = Float64(width) / Float64(image.width)
    var sy = Float64(height) / Float64(image.height)
    var scale = sx if sx > sy else sy
    # Python's `round` — half to EVEN, which `_round_half_even` already
    # reproduces for the canvas rounding.
    var rw = Int(_round_half_even(Float64(image.width) * scale))
    var rh = Int(_round_half_even(Float64(image.height) * scale))
    if rw < width:
        rw = width
    if rh < height:
        rh = height
    var left = (rw - width) // 2
    if left < 0:
        left = 0
    var top = (rh - height) // 2
    if top < 0:
        top = 0
    var resized = minimax_h3_lanczos_resize(image, rh, rw)
    return minimax_h3_crop(resized, left, top, height, width)


def minimax_h3_prepare_keyframes(
    images: List[MiniMaxH3RgbImage],
    have_first: Bool,
    have_last: Bool,
    height: Int,
    width: Int,
) raises -> List[MiniMaxH3RgbImage]:
    """Every keyframe of a request onto the canvas, in PACKED order.

    `images` is already the filtered list — `[image]`, `[last_image]`, or
    `[image, last_image]` — and `have_first`/`have_last` describe which slots
    the caller filled, so this can check that the two agree instead of trusting
    a length. Index 0 stretches, every later one cover-crops
    (before_encoder.py:190-193)."""
    var expected = 0
    if have_first:
        expected += 1
    if have_last:
        expected += 1
    if len(images) != expected:
        raise Error(
            String("minimax_h3_keyframe_image: ") + String(len(images))
            + " keyframes but have_first/have_last imply " + String(expected)
        )
    var out = List[MiniMaxH3RgbImage]()
    for i in range(len(images)):
        out.append(
            minimax_h3_prepare_keyframe_image(images[i], height, width, i == 0)
        )
    return out^


def minimax_h3_keyframe_anchors(have_first: Bool, have_last: Bool) -> List[Int]:
    """The packed-order anchor codes (`MINIMAX_H3_ANCHOR_FIRST` = 0,
    `..._LAST` = 1) for a request, matching `keyframe_anchors`
    (before_encoder.py:167-171): the tuple keeps the `(first, last)` order and
    drops the missing slots, so a `last_image`-only request is `("last",)` —
    anchored at the END of the timeline even though it is keyframe index 0."""
    var out = List[Int]()
    if have_first:
        out.append(0)
    if have_last:
        out.append(1)
    return out^
