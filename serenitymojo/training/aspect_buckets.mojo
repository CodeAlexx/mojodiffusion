# aspect_buckets.mojo — dynamic aspect-ratio bucket SET generation + stage-time
# bucket ASSIGNMENT, SimpleTuner semantics (Tier-2 campaign phase T2.D).
#
# Reference (read FULL, semantics matched function-by-function):
#   /home/alex/SimpleTuner/simpletuner/helpers/multiaspect/image.py
#     - MultiaspectImage._round_to_nearest_multiple   (lines 93-101)
#     - MultiaspectImage.calculate_new_size_by_pixel_area (lines 178-276)
#     - MultiaspectImage.calculate_image_aspect_ratio (lines 300-328; rounding=2)
#   /home/alex/SimpleTuner/simpletuner/helpers/image_manipulation/training_sample.py
#     - TrainingSample._select_random_aspect, crop_aspect="closest" (lines 266-273):
#       closest = min(buckets, key=|bucket - aspect_ratio|)  (FIRST wins on tie)
#     - TrainingSample.calculate_target_size (lines 561-624): closest aspect ->
#       calculate_new_size_by_pixel_area -> aspect = round(W_t/H_t, 2) ->
#       correct_intermediary_square_size (lines 545-559) -> if aspect == 1.0:
#       target = (pixel_resolution, pixel_resolution)
#     - center crop offsets: ((W_i - W_t)//2, (H_i - H_t)//2)
#       (discovery.py / training_sample.py crop_style="center")
#
# SimpleTuner buckets by the ROUNDED aspect-ratio key str(round(W/H, 2)); with
# crop_aspect="closest" the aspect is first snapped to the nearest entry of a
# user-supplied bucket ladder, then resized (intermediary keeps the ORIGINAL
# raw aspect) and center-cropped to the target. This module reproduces exactly
# that pipeline for a GENERATED ladder under a max-pixel budget:
#
#   generate_aspect_buckets(megapixels, align, ladder)
#     -> List[AspectBucket]  (deduped (W,H) targets; align = SimpleTuner
#        aspect_bucket_alignment, default 64; the VAE-stride(8) x patch(2) = 16
#        divisibility constraint is implied by any align that is a multiple
#        of 16)
#   assign_aspect_bucket(W0, H0, ladder, megapixels, align)
#     -> AspectBucketAssignment (target/intermediary/center-crop offsets)
#
# Python-rounding parity notes (the gate
# training/tests/aspect_buckets_parity.mojo diffs this module against
# SimpleTuner's own python code, exact-match required):
#   - Python round(x) (no digits) is round-half-to-EVEN -> _py_round_half_even.
#   - Python round(x, 2) rounds the EXACT binary value of the float at 2
#     decimal places, ties-to-even -> _py_round2 does the same via integer
#     arithmetic on the IEEE-754 mantissa/exponent (no double-rounding).
#   - Python int(x) truncates toward zero -> _py_int_trunc.
#
# This module is model-agnostic host code (no GPU, no Tensor): stagers call it
# at DATA STAGING time; trainers may consult the same generator to print or
# validate their compiled bucket tables.
#
# Mojo 1.0.0b1.

from std.collections import List
from std.math import sqrt


# ── Python-exact rounding helpers ─────────────────────────────────────────────


def _py_int_trunc(x: Float64) -> Int:
    """Python int(float): truncate toward zero."""
    if x >= 0.0:
        var n = Int(x)
        # Int(Float64) conversion truncates toward zero already; keep explicit.
        return n
    return -Int(-x)


def _py_round_half_even(v: Float64) raises -> Int:
    """Python round(v) for finite v >= 0: nearest int, ties to even.

    Exact: v is a Float64 whose fractional part comparison against 0.5 is
    exact (0.5 is a power of two)."""
    if v < 0.0:
        raise Error("_py_round_half_even: negative input unsupported")
    var f = Int(v)  # truncation = floor for v >= 0
    var frac = v - Float64(f)
    if frac > 0.5:
        return f + 1
    if frac < 0.5:
        return f
    # exact tie -> even
    if f % 2 == 0:
        return f
    return f + 1


def _py_round2(x: Float64) raises -> Float64:
    """Python round(x, 2) for finite x > 0: round the EXACT binary value of x
    at 2 decimal places, ties-to-even, then return the nearest Float64.

    x = M * 2^(e-1075) (normal IEEE-754 double). We need
    n = half_even(x * 100) computed on the exact rational M*100 / 2^k."""
    if not (x > 0.0):
        raise Error("_py_round2: input must be positive")
    var bits = x.to_bits[DType.uint64]()
    var exp_field = Int((bits >> 52) & 0x7FF)
    var mant = bits & UInt64(0xFFFFFFFFFFFFF)
    if exp_field == 0 or exp_field == 0x7FF:
        raise Error("_py_round2: subnormal/inf/nan unsupported")
    var m = UInt64(1) << 52 | mant            # 2^52 <= m < 2^53
    var p = exp_field - 1075                  # x = m * 2^p
    if p >= 0:
        # integral-scaled value; x*100 is an exact integer
        return x
    var k = -p
    if k > 58:
        raise Error("_py_round2: value too small for exact 2dp rounding")
    # n = half_even(m*100 / 2^k); m*100 < 2^60 fits UInt64
    var num = m * UInt64(100)
    var q = num >> UInt64(k)
    var rem = num - (q << UInt64(k))
    var half = UInt64(1) << UInt64(k - 1)
    var n = q
    if rem > half:
        n = q + 1
    elif rem == half:
        if q % 2 == 1:
            n = q + 1
    return Float64(Int(n)) / 100.0


def _round_to_nearest_multiple(value: Float64, multiple: Int) raises -> Int:
    """MultiaspectImage._round_to_nearest_multiple (image.py:93-101):
    round(value / multiple) * multiple, floored at `multiple`."""
    if multiple <= 0:
        raise Error("_round_to_nearest_multiple: multiple must be positive")
    var rounded = _py_round_half_even(value / Float64(multiple)) * multiple
    if rounded < multiple:
        return multiple
    return rounded


# ── bucket set + assignment types ─────────────────────────────────────────────


@fieldwise_init
struct AspectBucket(Copyable, Movable):
    """One generated bucket: the ladder aspect that produced it and the
    target pixel canvas (width, height)."""
    var aspect: Float64       # ladder entry (W/H), e.g. 0.75
    var width: Int            # target W in px
    var height: Int           # target H in px
    var aspect_key_x100: Int  # SimpleTuner bucket key round(W/H, 2) * 100


@fieldwise_init
struct AspectBucketAssignment(Copyable, Movable):
    """Stage-time assignment of one image to a bucket (SimpleTuner
    crop_aspect="closest" + crop_style="center" + resolution_type="area")."""
    var bucket_index: Int     # index into the generated ladder bucket list
    var aspect: Float64       # the selected ladder aspect
    var target_w: Int         # final canvas W (crop output)
    var target_h: Int         # final canvas H
    var inter_w: Int          # intermediary resize W (original aspect kept)
    var inter_h: Int          # intermediary resize H
    var crop_x: Int           # center-crop left offset in the intermediary
    var crop_y: Int           # center-crop top offset
    var aspect_key_x100: Int  # SimpleTuner bucket key round(W_t/H_t, 2) * 100


def default_aspect_ladder() raises -> List[Float64]:
    """The standard ladder from the T2.D task: 1:1, 4:3, 3:4, 16:9, 9:16,
    3:2, 2:3 — expressed as the 2-decimal floats a SimpleTuner
    crop_aspect_buckets list would carry (round(ratio, 2))."""
    var xs = List[Float64]()
    xs.append(1.0)
    xs.append(_py_round2(4.0 / 3.0))   # 1.33
    xs.append(_py_round2(3.0 / 4.0))   # 0.75
    xs.append(_py_round2(16.0 / 9.0))  # 1.78
    xs.append(_py_round2(9.0 / 16.0))  # 0.56
    xs.append(_py_round2(3.0 / 2.0))   # 1.5
    xs.append(_py_round2(2.0 / 3.0))   # 0.67
    return xs^


def pixel_resolution(megapixels: Float64, align: Int) raises -> Int:
    """TrainingSample._set_resolution, resolution_type="area"
    (training_sample.py:222): int(round_to_mult(sqrt(megapixels*1e6)))."""
    return _round_to_nearest_multiple(sqrt(megapixels * 1.0e6), align)


@fieldwise_init
struct _AreaSize(Copyable, Movable):
    var target_w: Int
    var target_h: Int
    var inter_w: Int
    var inter_h: Int
    var adjusted_aspect: Float64


def _calculate_new_size_by_pixel_area(
    aspect_ratio: Float64, megapixels: Float64, w0: Int, h0: Int, align: Int,
) raises -> _AreaSize:
    """MultiaspectImage.calculate_new_size_by_pixel_area (image.py:178-276),
    with the StateTracker aspect-resolution map empty (pure formula)."""
    var target_pixel_area = megapixels * 1.0e6
    var target_pixel_edge = _round_to_nearest_multiple(
        Float64(_py_int_trunc(sqrt(target_pixel_area))), align,
    )
    if w0 <= 0 or h0 <= 0:
        raise Error("_calculate_new_size_by_pixel_area: invalid image dims")

    if aspect_ratio == 1.0 and w0 == h0:
        return _AreaSize(
            target_pixel_edge, target_pixel_edge,
            target_pixel_edge, target_pixel_edge, 1.0,
        )

    var w_target = _round_to_nearest_multiple(
        Float64(target_pixel_edge) * sqrt(aspect_ratio), align,
    )
    var h_target = _round_to_nearest_multiple(
        Float64(target_pixel_edge) / sqrt(aspect_ratio), align,
    )
    var adjusted_aspect = _py_round2(Float64(w_target) / Float64(h_target))

    var ar_raw = Float64(w0) / Float64(h0)
    var w_inter: Int
    var h_inter: Int
    if w_target < h_target:  # portrait or square target
        w_inter = w_target
        h_inter = _py_int_trunc(Float64(w_inter) / ar_raw)
    else:  # landscape target
        h_inter = h_target
        w_inter = _py_int_trunc(Float64(h_inter) * ar_raw)

    # intermediary smaller than target -> grow it along the original aspect
    # (image.py:240-253; single corrective pass, replicated as-is)
    if w_target > w_inter or h_target > h_inter:
        var w_diff: Int
        var h_diff: Int
        if w_target > w_inter:
            w_diff = w_target - w_inter
            h_diff = _py_int_trunc(Float64(w_diff) / ar_raw)
        else:
            h_diff = h_target - h_inter
            w_diff = _py_int_trunc(Float64(h_diff) * ar_raw)
        h_inter += h_diff
        w_inter += w_diff

    return _AreaSize(w_target, h_target, w_inter, h_inter, adjusted_aspect)


def generate_aspect_buckets(
    megapixels: Float64, align: Int, ladder: List[Float64],
) raises -> List[AspectBucket]:
    """Emit the bucket set for a ladder of aspects under the max-pixel budget
    `megapixels` (SimpleTuner resolution_type="area") with bucket alignment
    `align`. Duplicate (W,H) targets are dropped (first ladder entry wins).

    For latent-space models the divisibility constraint is VAE stride x patch
    (Z-Image: 8 x 2 = 16); any `align` that is a multiple of 16 satisfies it
    (SimpleTuner's default 64 does)."""
    if align % 2 != 0:
        raise Error("generate_aspect_buckets: align must be even")
    var pixres = pixel_resolution(megapixels, align)
    var out = List[AspectBucket]()
    for i in range(len(ladder)):
        var a = ladder[i]
        if not (a > 0.0):
            raise Error("generate_aspect_buckets: ladder aspect must be > 0")
        # target dims depend only on (aspect, megapixels, align) — original
        # size only steers the intermediary, not the bucket canvas.
        var sz = _calculate_new_size_by_pixel_area(a, megapixels, 1000, 1000, align)
        var w_t = sz.target_w
        var h_t = sz.target_h
        var key = _py_round2(Float64(w_t) / Float64(h_t))
        # calculate_target_size (training_sample.py:617-618): a final aspect
        # of exactly 1.0 snaps the target to the square pixel resolution.
        if key == 1.0:
            w_t = pixres
            h_t = pixres
        var dup = False
        for j in range(len(out)):
            if out[j].width == w_t and out[j].height == h_t:
                dup = True
        if dup:
            continue
        out.append(AspectBucket(a, w_t, h_t, Int(_py_round2(key) * 100.0 + 0.5)))
    if len(out) == 0:
        raise Error("generate_aspect_buckets: empty bucket set")
    return out^


def assign_aspect_bucket(
    w0: Int, h0: Int, ladder: List[Float64], megapixels: Float64, align: Int,
) raises -> AspectBucketAssignment:
    """Assign one WxH image to a ladder bucket, SimpleTuner
    TrainingSample.calculate_target_size semantics for crop_enabled=True,
    crop_aspect="closest", crop_style="center", resolution_type="area".

    Returns the target canvas, the intermediary resize (original raw aspect
    preserved), and the center-crop offsets within the intermediary."""
    if len(ladder) == 0:
        raise Error("assign_aspect_bucket: empty ladder")
    if w0 <= 0 or h0 <= 0:
        raise Error("assign_aspect_bucket: invalid image dims")

    # calculate_image_aspect_ratio(original_size) — round(W/H, 2)
    var ar_img = _py_round2(Float64(w0) / Float64(h0))

    # _select_random_aspect, crop_aspect="closest": min(|bucket - ar|),
    # python min keeps the FIRST minimal entry on ties.
    var best = 0
    var best_d = ladder[0] - ar_img
    if best_d < 0.0:
        best_d = -best_d
    for i in range(1, len(ladder)):
        var d = ladder[i] - ar_img
        if d < 0.0:
            d = -d
        if d < best_d:
            best = i
            best_d = d
    var aspect = ladder[best]

    var sz = _calculate_new_size_by_pixel_area(aspect, megapixels, w0, h0, align)

    # calculate_target_size lines 610-618: final aspect re-derived from the
    # target; aspect 1.0 snaps intermediary (if smaller) and target to the
    # square pixel resolution.
    var key = _py_round2(Float64(sz.target_w) / Float64(sz.target_h))
    var pixres = pixel_resolution(megapixels, align)
    var w_t = sz.target_w
    var h_t = sz.target_h
    var w_i = sz.inter_w
    var h_i = sz.inter_h
    var crop_x: Int
    var crop_y: Int
    if key == 1.0:
        # correct_intermediary_square_size (training_sample.py:545-559)
        if w_i < pixres:
            w_i = pixres
            h_i = pixres
        w_t = pixres
        h_t = pixres
    crop_x = (w_i - w_t) // 2
    crop_y = (h_i - h_t) // 2
    if crop_x < 0 or crop_y < 0:
        raise Error("assign_aspect_bucket: intermediary smaller than target")
    return AspectBucketAssignment(
        best, aspect, w_t, h_t, w_i, h_i, crop_x, crop_y,
        Int(key * 100.0 + 0.5),
    )


def bucket_index_for_target(
    buckets: List[AspectBucket], target_w: Int, target_h: Int,
) -> Int:
    """Index of the bucket whose canvas is (target_w, target_h); -1 if the
    assignment landed outside the generated set (e.g. the square snap when
    1.0 is not in the ladder)."""
    for i in range(len(buckets)):
        if buckets[i].width == target_w and buckets[i].height == target_h:
            return i
    return -1


# ── comptime integer ladder (T2.D follow-up, 2026-06-11) ─────────────────────
#
# The Z-Image trainer dispatches each step to a COMPTIME-instantiated
# (lat_h, lat_w, cap_len) bucket (train_zimage_real.mojo), so the bucket set
# must be available at COMPILE time — comptime params cannot come from the
# runtime generator above. This section derives the standard 512px / align-64
# ladder with INTEGER-ONLY math (exact nearest-sqrt on rationals, no Float64),
# which the comptime interpreter can evaluate.
#
# Derivation (matches generate_aspect_buckets for megapixels=0.262144,
# align=64; gated EXACT in training/tests/zimage_comptime_ladder_gate.mojo):
#   edge  = pixel_resolution(0.262144, 64) = 512;  e = edge/align = 8 units
#   a     = aspect_x100 / 100   (the 2-decimal ladder entry, e.g. 133)
#   w_units = round(e * sqrt(a))   = nearest_int( sqrt(64*aspect_x100 / 100) )
#   h_units = round(e / sqrt(a))   = nearest_int( sqrt(6400 / aspect_x100) )
#   target  = (64*w_units, 64*h_units); floor-at-multiple => units >= 1
#   square snap: round(W/H, 2) == 1.0 <=> w_units == h_units for 64-aligned
#   dims <= 1024 (|wu/hu - 1| >= 1/16 > 0.005 whenever wu != hu), and the
#   snap target is the square pixel resolution 512x512.
#   latent = target / 8  (VAE stride 8)  =>  lat = 8 * units  (64 for square).
#
# NOTE: the Float64 rounding in generate_aspect_buckets matters only for
# stage-time SimpleTuner parity. The trainer needs just the (lat_h, lat_w)
# SET; the gate proves the two derivations emit the same set for THIS ladder.

comptime ZIMAGE_T2D_LADDER_LEN = 7
# round(ratio, 2) * 100 of default_aspect_ladder(), same order:
# 1:1, 4:3, 3:4, 16:9, 9:16, 3:2, 2:3.
comptime ZIMAGE_T2D_LADDER_X100 = [100, 133, 75, 178, 56, 150, 67]
# trainer caption buckets (cache mask prune + pad32), ascending dispatch order
comptime ZIMAGE_T2D_CAP_LENS = [224, 256]
comptime ZIMAGE_T2D_N_CAPS = 2


def zimage_t2d_nearest_sqrt_units(p: Int, q: Int) -> Int:
    """Nearest integer to sqrt(p/q) for p, q >= 1, ties-to-even — the exact
    integer form of _py_round_half_even(sqrt(p/q)). m is the answer iff
    q*(2m-1)^2 <= 4p <= q*(2m+1)^2; an exact upper tie 4p == q*(2m+1)^2 means
    sqrt(p/q) == m + 0.5 and goes to the even neighbour. Comptime-evaluable
    (integer while loop, no floats, non-raising)."""
    var m = 0
    while q * (2 * m + 1) * (2 * m + 1) < 4 * p:
        m += 1
    if q * (2 * m + 1) * (2 * m + 1) == 4 * p and m % 2 == 1:
        return m + 1
    return m


def zimage_t2d_lat_w(aspect_x100: Int) -> Int:
    """Latent W (= target_w / 8) of the 512px/align-64 bucket for the
    2-decimal ladder aspect aspect_x100/100. Integer-only; comptime-evaluable."""
    var wu = zimage_t2d_nearest_sqrt_units(64 * aspect_x100, 100)
    if wu < 1:
        wu = 1
    var hu = zimage_t2d_nearest_sqrt_units(6400, aspect_x100)
    if hu < 1:
        hu = 1
    if wu == hu:  # 2dp aspect key == 1.0 -> square snap to 512x512
        return 64
    return 8 * wu


def zimage_t2d_lat_h(aspect_x100: Int) -> Int:
    """Latent H (= target_h / 8) of the 512px/align-64 bucket for the
    2-decimal ladder aspect aspect_x100/100. Integer-only; comptime-evaluable."""
    var wu = zimage_t2d_nearest_sqrt_units(64 * aspect_x100, 100)
    if wu < 1:
        wu = 1
    var hu = zimage_t2d_nearest_sqrt_units(6400, aspect_x100)
    if hu < 1:
        hu = 1
    if wu == hu:  # 2dp aspect key == 1.0 -> square snap to 512x512
        return 64
    return 8 * hu
