# serenitymojo/models/vae/parity/minimax_h3_ref_encode_gate.mojo
#
# MiniMax-H3 ref2va reference-ENCODE parity gate — LIVE (GPU).
#
# The CPU half — pixel normalize, fp16 round, latent normalize, patchify, audio
# row packing, the 0.999 mix — is ALREADY gated bit-exact, host-side, with no
# GPU: `minimax_h3_ref_encode_probe.mojo` (12 checks). THIS gate covers what
# that one cannot: the device VAE encode (`minimax_h3_encode_reference_visual_
# seam`'s moments half, i.e. the `_encode` temporal-chunk + spatial-tile path
# on the REAL Ref2VA weights) and the posterior draw — plus the value chain
# re-checked end to end against a REAL reference clip's oracle.
#
# ── ORACLE CONTRACT (the producer matches this, not the reverse) ─────────────
# `scripts/minimax_h3_ref_encode_oracle.py`, run against the REAL Ref2VA
# checkpoint on the GPU, dumping one safetensors with these keys:
#
#   in.frames            uint8  [T, H, W, 3]   the prepared reference frames,
#                                              channels-LAST, at the reference's
#                                              own canvas, already 24 fps and
#                                              already trimmed to 17n+5
#   in.is_image          int32  []             1 -> `_encode_clip` path
#                                              0 -> `_encode` path
#   out.pixels           f32    [3, T, H, W]   after (x/255 - mean)/std
#   out.moments          f32    [1, 2C, T', H', W']  the VAE's raw moments
#   out.sample           f32    [1, C, T', H', W']   posterior.sample(gen=42),
#                                              BEFORE the fp16 round
#   out.rows             f32    [N, C*pt*ph*pw]  final condition rows, after
#                                              fp16 -> normalize -> patchify
#   out.noise            f32    [N, C*pt*ph*pw]  keyframe_condition_noise draw
#   out.rows_mixed       f32    [N, C*pt*ph*pw]  scale_noise(rows, 0.999, noise)
#
# Dumping `out.pixels`, `out.moments` and `out.sample` separately is the point:
# it lands a failure on the normalization, the convolutions, or the draw, rather
# than on "the encode".
#
# ── BARS, AND THE ONE THAT CANNOT BE MET ─────────────────────────────────────
#   out.pixels       BIT-EXACT. Pure elementwise f32; already proven host-side.
#   out.moments      cos >= 0.9999 vs the F32 noise floor. The encoder is
#                    convolutional; this is the same bar the video-encoder gate
#                    already holds.
#   out.sample       *** CANNOT BE MET AS A VALUE COMPARISON ***  The draw is
#                    `torch.Generator().manual_seed(42)` — torch's own CPU RNG
#                    stream. `pipeline/minimax_h3_torch_cpu_rng.mojo` reproduces
#                    the SCALAR reference path of that stream, but torch's AVX2
#                    build diverges from exact libm in the tail (that file's
#                    header quantifies it), so the draw is gated on DISTRIBUTION
#                    (mean/std of (sample - mean_of_moments)/std_of_moments
#                    against N(0,1)); the cos against `out.sample` is REPORTED,
#                    not gated. The VALUE path is gated by FEEDING `out.sample`
#                    back in as an input. See the next line.
#   out.rows         BIT-EXACT *given* `out.sample` as input. This is the check
#                    that matters and it is fully in reach: it isolates the
#                    fp16-round/normalize/patchify chain from the RNG.
#   out.rows_mixed   BIT-EXACT *given* `out.rows` and `out.noise` as inputs,
#                    for exactly the same reason.
#
# Run:
#   /home/alex/torchref/venv/bin/python scripts/minimax_h3_ref_encode_oracle.py
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs -Xlinker -lcudnn \
#     serenitymojo/models/vae/parity/minimax_h3_ref_encode_gate.mojo \
#     -o <scratch>/h3_refenc_gate \
#   && LD_LIBRARY_PATH=serenitymojo/ops/cshim/lib:serenitymojo/ops/cshim/lib/cudnn_stubs \
#     <scratch>/h3_refenc_gate

from std.collections import List
from std.math import exp, sqrt
from std.sys import argv
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.vae.minimax_h3_ref_encode import (
    MINIMAX_H3_VIDEO_LATENT_CHANNELS,
    minimax_h3_encode_reference_visual_moments,
    minimax_h3_mix_condition_rows,
    minimax_h3_pixel_normalize_frames,
    minimax_h3_reference_posterior_sample,
    minimax_h3_video_condition_rows,
)
from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice,
    minimax_h3_video_released_encoder_config,
)
from serenitymojo.pipeline.minimax_h3_video_vae_spatial_tiling import (
    minimax_h3_video_released_tiling_config,
)


comptime ORACLE = (
    "/home/alex/mojodiffusion/output/minimax_h3_ref2va/ref_encode_ref.safetensors"
)
comptime VIDEO_VAE_SOURCE = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/Ref2VA/video_vae/source"
)

comptime MOMENTS_COS_BAR = Float64(0.9999)
# N ~= 4.1M elements: the SEM of the mean is ~1/sqrt(N) ~= 5e-4 and of the std
# ~1/sqrt(2N) ~= 3.5e-4, so 5e-3 is a ~10-sigma distribution bar — loose enough
# never to flake, tight enough that a wrong mean/logvar split, a missed clamp
# or a non-unit draw all fail it by orders of magnitude.
comptime SAMPLE_DIST_BAR = Float64(5.0e-3)


def _read_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    if not st.has_tensor(name):
        raise Error(String("gate: oracle has no tensor ") + name)
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    if tv.dtype != STDtype.F32:
        raise Error(String("gate: ") + name + " is not F32")
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _read_u8(ref st: SafeTensors, name: String) raises -> List[UInt8]:
    if not st.has_tensor(name):
        raise Error(String("gate: oracle has no tensor ") + name)
    var info = st.tensor_info(name)
    if info.dtype != STDtype.U8:
        raise Error(String("gate: ") + name + " is not U8")
    var bytes = st.tensor_bytes(name)
    var out = List[UInt8](capacity=len(bytes))
    for i in range(len(bytes)):
        out.append(bytes[i])
    return out^


def _read_i32_scalar(ref st: SafeTensors, name: String) raises -> Int:
    if not st.has_tensor(name):
        raise Error(String("gate: oracle has no tensor ") + name)
    var info = st.tensor_info(name)
    if info.dtype != STDtype.I32:
        raise Error(String("gate: ") + name + " is not I32")
    var bytes = st.tensor_bytes(name)
    var p = bytes.unsafe_ptr().bitcast[Int32]()
    return Int(p[0])


def _cosine(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("gate: cosine over buffers of different length")
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


def _max_abs_diff(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error("gate: diff over buffers of different length")
    var m = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > m:
            m = d
    return m


def _mismatch_count(a: List[Float32], b: List[Float32]) raises -> Int:
    if len(a) != len(b):
        raise Error("gate: compare over buffers of different length")
    var bad = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            bad += 1
    return bad


def main() raises:
    var args = argv()
    var oracle_path = String(ORACLE)
    if len(args) >= 2:
        oracle_path = String(args[1])

    print("MiniMax-H3 ref2va reference-ENCODE parity gate")
    print("  oracle:", oracle_path)
    print("")

    var st = SafeTensors.open(oracle_path)

    var is_image = _read_i32_scalar(st, String("in.is_image"))
    if is_image != 0:
        raise Error(
            "gate: in.is_image != 0 — this gate covers the VIDEO `_encode`"
            " path; the image `_encode_clip` path is"
            " pipeline/parity/minimax_h3_keyframe_encode_gpu_probe.mojo's"
        )

    var finfo = st.tensor_info(String("in.frames"))
    if len(finfo.shape) != 4 or finfo.shape[3] != 3:
        raise Error("gate: in.frames is not [T, H, W, 3]")
    var t = finfo.shape[0]
    var h = finfo.shape[1]
    var w = finfo.shape[2]
    var frames = _read_u8(st, String("in.frames"))
    print("  in.frames:", t, "frames", w, "x", h, "(video reference, 17n+5)")

    var minfo = st.tensor_info(String("out.moments"))
    if len(minfo.shape) != 5 or minfo.shape[0] != 1:
        raise Error("gate: out.moments is not [1, 2C, T', H', W']")
    var two_c = minfo.shape[1]
    var lt = minfo.shape[2]
    var lh = minfo.shape[3]
    var lw = minfo.shape[4]
    if two_c != 2 * MINIMAX_H3_VIDEO_LATENT_CHANNELS:
        raise Error("gate: out.moments channel count is not 2C")
    print("  latent geometry:", lt, "x", lh, "x", lw, "(", two_c, "channels )")
    print("")

    var pass_all = True

    # ── [1] out.pixels: BIT-EXACT ───────────────────────────────────────────
    var pixels = minimax_h3_pixel_normalize_frames(frames, t, h, w)
    var want_pixels = _read_f32(st, String("out.pixels"))
    var pix_bad = _mismatch_count(pixels, want_pixels)
    if pix_bad == 0:
        print("  ok   out.pixels     — BIT-EXACT over", len(want_pixels), "values")
    else:
        pass_all = False
        print(
            "  FAIL out.pixels     —", pix_bad, "of", len(want_pixels),
            "differ, cos=", _cosine(pixels, want_pixels),
            " max_abs=", _max_abs_diff(pixels, want_pixels),
        )
    want_pixels = List[Float32]()

    # ── [2] out.moments: the device encode (REAL Ref2VA weights) ────────────
    var ctx = DeviceContext()
    var cfg = minimax_h3_video_released_encoder_config()
    var encoder = MiniMaxH3VideoEncoderDevice.load(String(VIDEO_VAE_SOURCE), cfg, ctx)
    var tiling = minimax_h3_video_released_tiling_config()
    print(
        "  encoder loaded (", VIDEO_VAE_SOURCE, "); tiling", tiling.tile_size,
        "px overlap_min", tiling.tile_overlap_min,
    )
    var moments = minimax_h3_encode_reference_visual_moments(
        encoder, pixels, 3, t, h, w, tiling, ctx
    )
    pixels = List[Float32]()
    var want_moments = _read_f32(st, String("out.moments"))
    var mom_cos = _cosine(moments, want_moments)
    var mom_max = _max_abs_diff(moments, want_moments)
    if mom_cos >= MOMENTS_COS_BAR:
        print("  ok   out.moments    — cos=", mom_cos, " max_abs=", mom_max)
    else:
        pass_all = False
        print(
            "  FAIL out.moments    — cos=", mom_cos, " max_abs=", mom_max,
            " (bar cos >=", MOMENTS_COS_BAR, ")",
        )

    # ── [3] out.sample: DISTRIBUTION (the RNG stream is torch's own; the
    #        value chain is gated in [4] by feeding the oracle's sample back).
    var sample = minimax_h3_reference_posterior_sample(moments, lt, lh, lw)
    var zc = MINIMAX_H3_VIDEO_LATENT_CHANNELS
    var lvol = lt * lh * lw
    var zsum = Float64(0.0)
    var zsq = Float64(0.0)
    for i in range(zc * lvol):
        # Whiten OUR draw by OUR moments: exactly N(0,1) iff the draw is unit
        # normal and the mean/logvar split + clamp + exp(0.5*lv) are right.
        var mean = Float64(moments[i])
        var logvar = Float64(moments[zc * lvol + i])
        if logvar < -30.0:
            logvar = -30.0
        if logvar > 20.0:
            logvar = 20.0
        var std = Float64(exp(Float32(0.5) * Float32(logvar)))
        var zval = (Float64(sample[i]) - mean) / std
        zsum += zval
        zsq += zval * zval
    var n = Float64(zc * lvol)
    var zmean = zsum / n
    var zstd = sqrt(zsq / n - zmean * zmean)
    var mean_off = zmean if zmean >= 0 else -zmean
    var std_off = zstd - 1.0 if zstd >= 1.0 else 1.0 - zstd
    var want_sample = _read_f32(st, String("out.sample"))
    var sample_cos = _cosine(sample, want_sample)
    if mean_off <= SAMPLE_DIST_BAR and std_off <= SAMPLE_DIST_BAR:
        print(
            "  ok   out.sample     — DISTRIBUTION mean=", zmean, " std=", zstd,
            " (bar +/-", SAMPLE_DIST_BAR, "); cos vs oracle =", sample_cos,
            "(reported, not gated)",
        )
    else:
        pass_all = False
        print(
            "  FAIL out.sample     — DISTRIBUTION mean=", zmean, " std=", zstd,
            " (bar +/-", SAMPLE_DIST_BAR, "); cos vs oracle =", sample_cos,
        )
    sample = List[Float32]()
    moments = List[Float32]()

    # ── [4] out.rows: BIT-EXACT given out.sample as input ───────────────────
    # `out.sample` is [1, C, T', H', W'] contiguous == [C, T', H', W'] flat,
    # exactly the layout minimax_h3_video_condition_rows consumes.
    var rows = minimax_h3_video_condition_rows(want_sample, zc, lt, lh, lw)
    want_sample = List[Float32]()
    var want_rows = _read_f32(st, String("out.rows"))
    var rows_bad = _mismatch_count(rows, want_rows)
    if rows_bad == 0:
        print("  ok   out.rows       — BIT-EXACT over", len(want_rows), "values (given out.sample)")
    else:
        pass_all = False
        print(
            "  FAIL out.rows       —", rows_bad, "of", len(want_rows),
            "differ, cos=", _cosine(rows, want_rows),
            " max_abs=", _max_abs_diff(rows, want_rows),
        )
    rows = List[Float32]()

    # ── [5] out.rows_mixed: BIT-EXACT given out.rows + out.noise ────────────
    var noise = _read_f32(st, String("out.noise"))
    var mixed = minimax_h3_mix_condition_rows(want_rows, noise)
    var want_mixed = _read_f32(st, String("out.rows_mixed"))
    var mixed_bad = _mismatch_count(mixed, want_mixed)
    if mixed_bad == 0:
        print("  ok   out.rows_mixed — BIT-EXACT over", len(want_mixed), "values (given rows+noise)")
    else:
        pass_all = False
        print(
            "  FAIL out.rows_mixed —", mixed_bad, "of", len(want_mixed),
            "differ, cos=", _cosine(mixed, want_mixed),
            " max_abs=", _max_abs_diff(mixed, want_mixed),
        )

    print("")
    if pass_all:
        print("PASS — reference-ENCODE chain gated against the vendor's own encode")
    else:
        print("FAIL")
        raise Error("minimax_h3_ref_encode_gate FAILED — see per-key lines above")
