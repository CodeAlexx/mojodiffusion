# serenitymojo/pipeline/minimax_h3_keyframe_encode.mojo — the I2VA / FL2VA /
# L2VA keyframe conditioning chain: a prepared keyframe in, packed conditioning
# rows out.
#
# Everything except one function is HOST arithmetic. `minimax_h3_keyframe_
# encode_device` is the single GPU seam and is the only thing here that needs a
# DeviceContext.
#
# ── THE CHAIN, op for op (encoders.py:264-322, read in full) ─────────────────
#   1. `np.array(image).permute(2,0,1)[None,:,None]`            -> [1,3,1,H,W]
#   2. `(pixels/255 - pixel_mean) / pixel_std`, the VIDEO VAE's ImageNet
#      constants (packing.py:70-71). NOT the conditioner's mean=std=0.5.
#   3. `vae._encode_clip(pixels)` — the SPATIAL encoder alone. A keyframe is one
#      frame and must NOT go through the 17-frames-per-5-latents temporal
#      chunking (:295 and the method's own docstring). Tiling is on by default
#      (`vae_encoder_tiling: 1`, `vae_tile_size: 256`), so at any real canvas
#      this is the TILED spatial encode, not a single pass.
#   4. `DiagonalGaussianDistribution(moments).sample(generator=
#      torch.Generator().manual_seed(42))` — a SAMPLE, off a CPU generator
#      seeded 42 INDEPENDENTLY of the request seed.
#   5. `latents.to(torch.float16).float()` BEFORE normalizing.
#   6. `(latents - latents_mean) / latents_std`, then `patchify_video_latents`.
#   7. `scheduler.scale_noise(rows, 0.999, keyframe_condition_noise(...))`.
#
# Steps 2 and 5-7 are NOT re-derived here: they are `models/vae/minimax_h3_ref_
# encode.mojo`'s, host-gated 12/12, called directly. Steps 3-4 are what this
# file adds — step 3 by calling the vendor-oracle-gated tiled encoder, step 4
# by pairing it with `pipeline/minimax_h3_torch_cpu_rng.mojo` (torch's CPU
# MT19937, gated bit-exact on the uniforms).
#
# ── WHAT MAKES THE KEYFRAME PATH DIFFERENT FROM ref2va's IMAGE PATH ──────────
# Nothing, numerically — `encode_references` (:537-606) runs the identical
# recipe for an image reference, which is why the ref-encode half is shared. The
# difference is entirely in the LAYOUT: a keyframe's rows are anchored to one
# end of the target timeline (`keyframe_anchors`) and inherit the TARGET
# canvas's latent geometry, while a ref2va image reference gets its own canvas
# and its own place on the shared clock.
#
# ── THE ROW TIMESTEP IS NOT THE MIX LEVEL ────────────────────────────────────
# Restated here because the two numbers are equal for most of a run and a
# rewrite that conflates them looks correct until it does not: the LATENTS are
# mixed ONCE, at encode time, at the CONSTANT 0.999. The row TIMESTEP the
# transformer is conditioned on is `max(video_t, 0.999)` and MOVES with the
# schedule (before_denoise.py:413-417). They differ exactly when `video_t >
# 0.999`, i.e. the first step of a run whose schedule starts at 1.0 — and
# re-mixing at the row timestep every step would re-noise the anchor away.
#
# Mojo 1.0.0b1. GPU only inside `minimax_h3_keyframe_encode_device`.

from std.collections import List
from std.math import exp
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.models.minimax_h3.rearrange import minimax_h3_patchify_video
from serenitymojo.models.vae.minimax_h3_ref_encode import (
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_VIDEO_LATENT_CHANNELS,
    minimax_h3_mix_condition_rows,
    minimax_h3_pixel_normalize_frames,
    minimax_h3_video_condition_rows,
)
from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice,
)
from serenitymojo.pipeline.minimax_h3_keyframe_image import MiniMaxH3RgbImage
from serenitymojo.pipeline.minimax_h3_torch_cpu_rng import (
    MiniMaxH3TorchCpuGenerator,
    minimax_h3_torch_cpu_randn,
    minimax_h3_torch_cpu_randn_from,
)
from serenitymojo.pipeline.minimax_h3_video_vae_spatial_tiling import (
    MiniMaxH3TilingConfig,
    minimax_h3_video_tiled_encode,
)

# `DiagonalGaussianDistribution.__init__` clamps the log-variance before it is
# exponentiated (diffusers vae.py:691). Without the clamp a saturated encoder
# output turns into an inf standard deviation.
comptime MINIMAX_H3_LOGVAR_MIN = Float32(-30.0)
comptime MINIMAX_H3_LOGVAR_MAX = Float32(20.0)

# The video VAE's spatial compression, `product(space_down=[2,2,2,2,1,1])`.
comptime MINIMAX_H3_VAE_SPATIAL_RATIO = 16


def minimax_h3_keyframe_pixels_ndhwc(
    image: MiniMaxH3RgbImage,
) raises -> List[Float32]:
    """Steps 1-2: a prepared keyframe -> `[1, 1, H, W, 3]` NDHWC, normalized.

    The normalization itself is `models/vae/minimax_h3_ref_encode.mojo`'s
    gated `minimax_h3_pixel_normalize_frames`, which emits channels-FIRST
    `[3, T, H, W]` (the vendor's own tensor layout). This transposes that to the
    channels-LAST layout `conv3d_fcqrs_cudnn` and the whole device VAE use — a
    pure index shuffle, no arithmetic, so the gated normalization stays the only
    place the pixel math lives."""
    image.validate()
    var chw = minimax_h3_pixel_normalize_frames(
        image.pixels, 1, image.height, image.width
    )
    var plane = image.height * image.width
    var out = List[Float32]()
    out.resize(plane * 3, Float32(0.0))
    for y in range(image.height):
        for x in range(image.width):
            var at = y * image.width + x
            for c in range(3):
                out[at * 3 + c] = chw[c * plane + at]
    return out^


def minimax_h3_keyframe_encode_device(
    encoder: MiniMaxH3VideoEncoderDevice,
    image: MiniMaxH3RgbImage,
    tiling: MiniMaxH3TilingConfig,
    ctx: DeviceContext,
) raises -> List[Float32]:
    """THE GPU SEAM — step 3, `vae._encode_clip`.

    A keyframe is ONE frame, so this calls the spatial encoder path directly and
    never touches `pipeline/minimax_h3_video_vae_temporal.mojo`. That is not an
    optimization: routing a single frame through the temporal chunking would
    apply the 17-frame clip padding and the trailing-token drop to it, which is
    the exact thing `_encode_clip` exists to avoid (its own docstring:
    "a single frame must not go through the temporal chunking").

    Returns the moments as a host `[T', H', W', 2*z]` NDHWC list — the sample
    that consumes them is CPU-side anyway (the generator is a CPU one), so the
    boundary is here rather than after the sample."""
    var pixels_host = minimax_h3_keyframe_pixels_ndhwc(image)
    var shape: List[Int] = [1, 1, image.height, image.width, 3]
    var pixels = Tensor.from_host(pixels_host, shape^, STDtype.F32, ctx)
    var moments = minimax_h3_video_tiled_encode(encoder, pixels, tiling, ctx)

    var ms = moments.shape()
    if ms[0] != 1 or ms[1] != 1:
        raise Error(
            String("minimax_h3_keyframe_encode: _encode_clip returned ")
            + String(ms[1]) + " latent frames for a single-frame keyframe,"
            " expected 1 — the temporal path was taken"
        )
    if ms[4] != 2 * MINIMAX_H3_VIDEO_LATENT_CHANNELS:
        raise Error(
            String("minimax_h3_keyframe_encode: moments carry ") + String(ms[4])
            + " channels, expected " + String(2 * MINIMAX_H3_VIDEO_LATENT_CHANNELS)
            + " (mean + logvar)"
        )
    var expect_h = image.height // MINIMAX_H3_VAE_SPATIAL_RATIO
    var expect_w = image.width // MINIMAX_H3_VAE_SPATIAL_RATIO
    if ms[2] != expect_h or ms[3] != expect_w:
        raise Error(
            String("minimax_h3_keyframe_encode: moments are ") + String(ms[2])
            + "x" + String(ms[3]) + ", expected " + String(expect_h) + "x"
            + String(expect_w) + " for a " + String(image.height) + "x"
            + String(image.width) + " keyframe"
        )
    return moments.to_host(ctx)


def minimax_h3_keyframe_posterior_sample(
    moments_ndhwc: List[Float32],
    latent_height: Int,
    latent_width: Int,
    seed: UInt64 = UInt64(MINIMAX_H3_KEYFRAME_ENCODE_SEED),
) raises -> List[Float32]:
    """Step 4: `DiagonalGaussianDistribution(moments).sample(generator)`.

    `moments_ndhwc` is `[1, H', W', 2*z]` flattened (one latent frame). Returns
    the sample as `[z, 1, H', W']` CHANNELS-FIRST flat, which is the layout
    `minimax_h3_video_condition_rows` consumes.

    Two things that a rewrite gets wrong and that the layout makes easy to get
    wrong here specifically:
      * `torch.chunk(parameters, 2, dim=1)` splits the CHANNEL axis, and our
        channel axis TRAILS where the reference's leads. Mean is the first z
        channels, logvar the last z — per position, not per half-of-buffer.
      * `randn_tensor(self.mean.shape, ...)` draws over the reference's
        `[1, z, 1, H', W']`, i.e. CHANNEL-SLOWEST. The noise index therefore
        matches the CHANNELS-FIRST output layout, not the NDHWC input one."""
    var z = MINIMAX_H3_VIDEO_LATENT_CHANNELS
    var plane = latent_height * latent_width
    var expected = plane * 2 * z
    if len(moments_ndhwc) != expected:
        raise Error(
            String("minimax_h3_keyframe_encode: moments hold ")
            + String(len(moments_ndhwc)) + " values, geometry implies "
            + String(expected)
        )
    var noise = minimax_h3_torch_cpu_randn(z * plane, seed)
    var out = List[Float32]()
    out.resize(z * plane, Float32(0.0))
    for c in range(z):
        for i in range(plane):
            var mean = moments_ndhwc[i * 2 * z + c]
            var logvar = moments_ndhwc[i * 2 * z + z + c]
            if logvar < MINIMAX_H3_LOGVAR_MIN:
                logvar = MINIMAX_H3_LOGVAR_MIN
            if logvar > MINIMAX_H3_LOGVAR_MAX:
                logvar = MINIMAX_H3_LOGVAR_MAX
            var std = exp(Float32(0.5) * logvar)
            out[c * plane + i] = mean + std * noise[c * plane + i]
    return out^


def minimax_h3_keyframe_condition_noise(
    mut gen: MiniMaxH3TorchCpuGenerator,
    num_keyframes: Int,
    latent_height: Int,
    latent_width: Int,
    patch_h: Int = 2,
    patch_w: Int = 2,
) raises -> List[Float32]:
    """`keyframe_condition_noise` (packing.py:501-538).

    ONE `randn` per condition, in PACKED order, off the REQUEST's generator —
    and these are the FIRST draws of a request, ahead of the target video and
    audio noise (the docstring at :511-515 says so explicitly, and
    `before_denoise.py`'s prepare-latents step runs after this one). Each draw
    is patchified on its own and the rows concatenated, so the row order is the
    same one `patchify_video_latents` produces for a single condition."""
    if num_keyframes < 1:
        raise Error("minimax_h3_keyframe_encode: no keyframes to noise")
    var z = MINIMAX_H3_VIDEO_LATENT_CHANNELS
    var out = List[Float32]()
    for _ in range(num_keyframes):
        var noise = minimax_h3_torch_cpu_randn_from(gen, z * latent_height * latent_width)
        var rows = minimax_h3_patchify_video(
            noise, z, 1, latent_height, latent_width, 1, patch_h, patch_w
        )
        for i in range(len(rows)):
            out.append(rows[i])
    return out^


def minimax_h3_keyframe_rows_from_moments(
    moments_per_keyframe: List[List[Float32]],
    latent_height: Int,
    latent_width: Int,
    patch_h: Int = 2,
    patch_w: Int = 2,
) raises -> List[Float32]:
    """Steps 4-6 for every keyframe of a request, concatenated in packed order.

    Each keyframe is sampled off its OWN fresh generator at seed 42 — the
    reference builds `torch.Generator().manual_seed(42)` inside the per-image
    loop (encoders.py:297), so two keyframes of the same size get the SAME
    posterior noise, not two consecutive draws. That is a real property of the
    released model's conditioning, not an oversight to fix."""
    var out = List[Float32]()
    for k in range(len(moments_per_keyframe)):
        var sample = minimax_h3_keyframe_posterior_sample(
            moments_per_keyframe[k], latent_height, latent_width
        )
        var rows = minimax_h3_video_condition_rows(
            sample, MINIMAX_H3_VIDEO_LATENT_CHANNELS, 1, latent_height,
            latent_width, 1, patch_h, patch_w,
        )
        for i in range(len(rows)):
            out.append(rows[i])
    return out^


def minimax_h3_keyframe_condition_rows(
    moments_per_keyframe: List[List[Float32]],
    mut gen: MiniMaxH3TorchCpuGenerator,
    latent_height: Int,
    latent_width: Int,
    patch_h: Int = 2,
    patch_w: Int = 2,
) raises -> List[Float32]:
    """Steps 4-7: moments in, the packed conditioning rows the denoise loop
    prepends to its video rows out.

    `gen` is the REQUEST's generator and is advanced by exactly the conditioning
    draws, so the caller can hand the same generator to the target-latent draw
    afterwards and land on the vendor's own noise for both."""
    var rows = minimax_h3_keyframe_rows_from_moments(
        moments_per_keyframe, latent_height, latent_width, patch_h, patch_w
    )
    var noise = minimax_h3_keyframe_condition_noise(
        gen, len(moments_per_keyframe), latent_height, latent_width, patch_h, patch_w
    )
    return minimax_h3_mix_condition_rows(rows, noise)


def minimax_h3_target_latent_rows(
    mut gen: MiniMaxH3TorchCpuGenerator,
    num_latent_frames: Int,
    latent_height: Int,
    latent_width: Int,
    patch_h: Int = 2,
    patch_w: Int = 2,
) raises -> List[Float32]:
    """The target VIDEO noise, drawn the vendor's way (before_denoise.py:309-316):
    as a `[1, z, T, H, W]` LATENT TENSOR that is patchified afterwards, not
    directly in row space.

    The distinction is invisible in distribution and load-bearing in bytes: for
    a given generator the two orders permute the same values differently, so a
    row-space draw reproduces neither the vendor's latents nor its output. The
    t2va pipeline in this repo deliberately takes the row-space shortcut and
    documents it; a keyframe request cannot, because its whole point is to match
    a conditioned reference."""
    var z = MINIMAX_H3_VIDEO_LATENT_CHANNELS
    var noise = minimax_h3_torch_cpu_randn_from(
        gen, z * num_latent_frames * latent_height * latent_width
    )
    return minimax_h3_patchify_video(
        noise, z, num_latent_frames, latent_height, latent_width, 1, patch_h, patch_w
    )


def minimax_h3_target_audio_rows(
    mut gen: MiniMaxH3TorchCpuGenerator,
    num_audio_latents: Int,
    audio_latent_channels: Int,
) raises -> List[Float32]:
    """The target AUDIO noise (before_denoise.py:318-324): drawn DIRECTLY in row
    layout, `[num_audio_latents * 2, audio_latent_channels]`, unlike the video
    noise. Reproduced as-is — the asymmetry is the reference's."""
    return minimax_h3_torch_cpu_randn_from(
        gen, num_audio_latents * 2 * audio_latent_channels
    )
