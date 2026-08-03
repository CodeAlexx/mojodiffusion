# serenitymojo/pipeline/parity/minimax_h3_keyframe_encode_gpu_probe.mojo
#
# PENDING-GPU as of 2026-08-03 — written and BUILT, not yet run. The overnight
# chain owns the card.
#
# The one stage `minimax_h3_keyframe_encode_probe.mojo` cannot cover: the video
# VAE forward. Runs `minimax_h3_keyframe_encode_device` — pixel normalize, then
# the SPATIALLY TILED single-clip encode — on the same prepared keyframe the
# oracle used, and compares the moments against the vendor's own
# `AutoencoderKLLegacy._adaptive_encode`.
#
# ── WHY THIS IS A SEPARATE BINARY FROM THE HOST PROBE ───────────────────────
# Not style: the host probe must not link a DeviceContext at all, so that it
# stays runnable while the GPU is held. A single file behind a comptime flag
# would still drag the GPU imports into the parse.
#
# ── WHAT A FAILURE HERE WOULD AND WOULD NOT MEAN ────────────────────────────
# The encoder itself is already gated against this same vendor class at
# cos 0.9999999978 (the t2va decode path's own note) — but for a VIDEO clip, and
# WITHOUT the keyframe path's two specifics:
#   * ONE frame, so the causal temporal padding sees T=1 and the two `time_down`
#     stages must leave it at 1 latent frame. If they do not, the moments come
#     back with the wrong frame count and this probe's shape check fires before
#     any numeric comparison.
#   * SPATIAL TILING at a canvas both of whose axes exceed 256 px, which is
#     every real keyframe. `_encode_clip`/`_adaptive_encode` tile and blend;
#     a single-pass encode is a different (and smoother) image.
# So a failure here is about the keyframe path, not about the encoder.
#
# Run (PENDING — needs the GPU free):
#   python3 scripts/minimax_h3_keyframe_encode_oracle.py \
#       <prepared_keyframe.png> output/minimax_h3_keyframe/keyframe_encode_ref.safetensors 7
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/pipeline/parity/minimax_h3_keyframe_encode_gpu_probe.mojo \
#     -o <scratch>/h3_keyframe_encode_gpu_probe \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa -Xlinker -lm \
#   && <scratch>/h3_keyframe_encode_gpu_probe \
#        output/minimax_h3_keyframe/keyframe_encode_ref.safetensors <prepared_keyframe.png>

from std.sys import argv
from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.vae.minimax_h3_ref_encode import (
    MINIMAX_H3_VIDEO_LATENT_CHANNELS,
)
from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice,
    minimax_h3_video_released_encoder_config,
)
from serenitymojo.pipeline.minimax_h3_keyframe_encode import (
    minimax_h3_keyframe_encode_device,
    minimax_h3_keyframe_pixels_ndhwc,
)
from serenitymojo.pipeline.minimax_h3_keyframe_image import MiniMaxH3RgbImage
from serenitymojo.pipeline.minimax_h3_media_in import minimax_h3_ffmpeg_read_image
from serenitymojo.pipeline.minimax_h3_video_vae_spatial_tiling import (
    minimax_h3_video_released_tiling_config,
)

comptime VIDEO_VAE_DIR = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/video_vae/source"
)


def _read_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    if not st.has_tensor(name):
        raise Error(String("probe: reference file has no tensor ") + name)
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    if tv.dtype != STDtype.F32:
        raise Error(String("probe: ") + name + " is not F32")
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _cosine(a: List[Float32], b: List[Float32]) -> Float64:
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


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("usage: minimax_h3_keyframe_encode_gpu_probe <ref.safetensors> <prepared_keyframe.png>")
        return
    var ref_path = String(args[1])
    var image_path = String(args[2])

    print("MiniMax-H3 keyframe unit — VAE forward (DEVICE) probe")
    print("  reference:", ref_path)
    print("  keyframe :", image_path)
    print("")

    var st = SafeTensors.open(ref_path)
    var mshape = st.tensor_info(String("moments")).shape.copy()
    if len(mshape) != 5:
        raise Error("probe: `moments` is not [1, 2C, 1, H, W]")
    var two_c = mshape[1]
    var latent_h = mshape[3]
    var latent_w = mshape[4]
    if mshape[2] != 1:
        raise Error(
            String("probe: the reference moments carry ") + String(mshape[2])
            + " latent frames — a keyframe must produce exactly 1"
        )

    var frames = minimax_h3_ffmpeg_read_image(
        image_path, image_path + ".probe.rgb", image_path + ".probe.txt"
    )
    if frames.num_frames != 1:
        raise Error("probe: the keyframe file is not a single still")
    var image = MiniMaxH3RgbImage(frames.pixels.copy(), frames.height, frames.width)
    print("  keyframe pixels:", image.width, "x", image.height)

    # Stage 1 is host and is compared too — if the pixel normalization drifted,
    # every later number would be wrong for a reason that has nothing to do with
    # the VAE, and the reference dumps it precisely so that cannot hide.
    var pixels = minimax_h3_keyframe_pixels_ndhwc(image)
    var want_pixels_nchw = _read_f32(st, String("pixels_normalized"))
    var plane = image.height * image.width
    var pixels_nchw = List[Float32]()
    pixels_nchw.resize(3 * plane, Float32(0.0))
    for c in range(3):
        for i in range(plane):
            pixels_nchw[c * plane + i] = pixels[i * 3 + c]
    var pix_cos = _cosine(pixels_nchw, want_pixels_nchw)
    var pix_bad = 0
    for i in range(len(want_pixels_nchw)):
        if pixels_nchw[i] != want_pixels_nchw[i]:
            pix_bad += 1
    if pix_bad == 0:
        print("  ok   stage 1 pixel normalize — bit-exact over", len(want_pixels_nchw), "values")
    else:
        print("  FAIL stage 1 pixel normalize —", pix_bad, "differ, cos=", pix_cos)

    var ctx = DeviceContext()
    var cfg = minimax_h3_video_released_encoder_config()
    var encoder = MiniMaxH3VideoEncoderDevice.load(String(VIDEO_VAE_DIR), cfg, ctx)
    var tiling = minimax_h3_video_released_tiling_config()
    print("  encoder loaded; tiling", tiling.tile_size, "px overlap_min", tiling.tile_overlap_min)

    var got = minimax_h3_keyframe_encode_device(encoder, image, tiling, ctx)
    print("  moments:", len(got), "values for", latent_w, "x", latent_h, "x", two_c)

    # The reference is NCDHW; ours is NDHWC. Transposed explicitly — the two
    # buffers have the same length, so a reinterpretation would compare cleanly
    # against the wrong thing.
    var want_ncdhw = _read_f32(st, String("moments"))
    var lplane = latent_h * latent_w
    if len(got) != two_c * lplane or len(want_ncdhw) != two_c * lplane:
        raise Error("probe: moment buffers do not match the declared geometry")
    var want = List[Float32]()
    want.resize(two_c * lplane, Float32(0.0))
    for c in range(two_c):
        for i in range(lplane):
            want[i * two_c + c] = want_ncdhw[c * lplane + i]

    var cos = _cosine(got, want)
    var max_abs = Float32(0.0)
    var at = -1
    for i in range(len(want)):
        var d = got[i] - want[i]
        if d < 0:
            d = -d
        if d > max_abs:
            max_abs = d
            at = i
    print("")
    # The bar is the one the encoder already meets against this same class on a
    # video clip; the keyframe path adds only tiling and T=1, neither of which
    # should cost accuracy.
    if cos > 0.9999999 and max_abs < Float32(1.0e-2):
        print("  ok   VAE moments — cos=", cos, " max_abs=", max_abs)
        print("")
        print("PASS")
    else:
        print("  FAIL VAE moments — cos=", cos, " max_abs=", max_abs, " at", at)
        print("")
        print("FAIL")
        raise Error("minimax_h3_keyframe_encode GPU probe FAILED")
    _ = MINIMAX_H3_VIDEO_LATENT_CHANNELS
