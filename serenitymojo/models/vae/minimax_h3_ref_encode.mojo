# serenitymojo/models/vae/minimax_h3_ref_encode.mojo — MiniMax-H3 ref2va
# UNIT 2 (ref-encode). Everything except the seam at the BOTTOM of this file
# is host arithmetic (no Tensor, no DeviceContext, no GPU) and is gated by the
# host probe. The seam — `minimax_h3_encode_reference_visual_seam` and its two
# halves — is the DEVICE VAE encode + posterior draw, and needs a
# DeviceContext; its gate is parity/minimax_h3_ref_encode_gate.mojo (GPU).
#
# ── THE RECIPE, op for op (encoders.py:566-604, read in full) ────────────────
# VIDEO and IMAGE references:
#   1. pixels, channels-first f32:
#        image -> `np.array(image).permute(2,0,1)[None,:,None]`  = [1,3,1,H,W]
#        video -> `frames[:trim].permute(3,0,1,2)[None]`         = [1,3,T,H,W]
#   2. `(pixels/255.0 - pixel_mean) / pixel_std`, ImageNet constants over a
#      [0,1] base range (packing.py:70-71). NOTE these are the VIDEO VAE's
#      constants and are NOT the conditioner's — Qwen3-VL's processor
#      normalizes with mean=std=0.5 (Ref2VA/processor/preprocessor_config.json).
#      Two different normalizations over the same pixels; do not share them.
#   3. moments = `_encode_clip` for an image (the tiled SPATIAL encoder alone,
#      one latent frame) or `_encode` for a video (temporal chunking, 17n+5 ->
#      5n+2).                                       <-- THE SEAM, needs the GPU
#   4. `DiagonalGaussianDistribution(moments).sample(generator=
#      torch.Generator().manual_seed(42))` — a SAMPLE, not the mode, off a
#      generator seeded 42 INDEPENDENTLY of the request seed
#      (MINIMAX_H3_KEYFRAME_ENCODE_SEED, packing.py:87). `torch.Generator()`
#      with no device is a CPU generator, so the draw is CPU-side even when the
#      VAE ran on the GPU.
#   5. `latents.to(torch.float16).float()` — round to fp16 and back BEFORE
#      normalizing. The vendor's own comment: "~11 bits of every conditioning
#      latent, so the released model's conditioning cannot be reproduced
#      without it." Not an optimization; a bit-exactness requirement.
#   6. `(latents - latents_mean) / latents_std`, then
#      `patchify_video_latents(..., patch_size)` — channel-SLOWEST.
#
# AUDIO references (and a video reference's soundtrack), encoders.py:594-600:
#   1. `audio_vae.encode(waveform[:, None])` — the waveform is [2, samples], so
#      [:, None] makes [2, 1, samples]: TWO BATCH ITEMS of a MONO VAE, one per
#      stereo channel. Not a stereo model.
#   2. `posterior.mode()` — the MODE. NOT a sample, no generator, no seed.
#   3. NO fp16 rounding. That step is exclusive to the visual branch.
#   4. `.transpose(1,2)` -> [2, T, C], `(x - mean)/std` per CHANNEL on the last
#      axis (the vendor views them `(1, 1, -1)`), `.reshape(-1, C)` -> channel-
#      major rows: all T rows of the left channel, then all T of the right.
#
# ── THE NOISE MIX (encoders.py:612-628) ─────────────────────────────────────
# Only the VISUAL condition rows are mixed. Audio condition rows go to the
# device CLEAN — `audio_condition_latents.to(device)` and nothing else. That is
# the asymmetry flagged in unit B: audio reference rows are clean latents yet
# are pinned at row timestep 1.0.
#
#     noise = keyframe_condition_noise(shapes, patch_size, latent_channels,
#                                      generator=block_state.generator, ...)
#     condition_latents = scheduler.scale_noise(condition_latents,
#                                               MINIMAX_H3_KEYFRAME_NOISE_AUG,
#                                               noise)
#
# TWO THINGS A REWRITE GETS WRONG HERE:
#
#  (a) The mix is at the CONSTANT 0.999, not at the row timestep. The row
#      timestep the transformer is conditioned on is `max(video_t, 0.999)` and
#      MOVES with the schedule (before_denoise.py:417), but the latents are
#      mixed ONCE, at encode time, at exactly 0.999. The two numbers are equal
#      only when video_t <= 0.999. Mixing at the row timestep instead would
#      re-noise the conditioning every step.
#
#  (b) `MiniMaxH3Scheduler.scale_noise` takes the timestep AT FACE VALUE and
#      does NOT look it up in the schedule (scheduling_minimax_h3.py:196-225,
#      and our gated port's own docstring) — 0.999 is a noise-aug level, not a
#      schedule entry. The formula is the rectified-flow forward process in
#      H3's t convention, `x_t = t*x0 + (1-t)*noise`, so at 0.999 the
#      conditioning keeps 99.9% of itself. This module CALLS the gated
#      `models/minimax_h3/scheduler.mojo::MiniMaxH3Scheduler.scale_noise`
#      rather than restating `0.999*x + 0.001*n`, because that port also
#      reproduces the reference's per-operation f32 rounding.
#
# NOISE DRAW ORDER, and a REAL PARITY RISK. `keyframe_condition_noise`
# (packing.py:501-538) draws one `randn` per condition, in PACKED ORDER, off the
# REQUEST generator, and does so BEFORE the target video/audio noise — "the
# order is part of what a generator reproduces". Reproducing the vendor's exact
# conditioning therefore needs torch's own RNG stream, and `ops/random.mojo`'s
# `randn` is distributionally correct but NOT torch-bit-exact (that file says so
# itself). So a seeded ref2va run will not match the vendor byte for byte
# through the noise, even with everything in this file exact. Flagged here
# rather than discovered at the gate.

from std.benchmark import black_box
from std.collections import List
from std.math import exp
from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.models.minimax_h3.packing import (
    minimax_h3_video_latent_num_frames,
)
from serenitymojo.models.minimax_h3.rearrange import minimax_h3_patchify_video
from serenitymojo.models.minimax_h3.scheduler import MiniMaxH3Scheduler
from serenitymojo.models.dit.minimax_h3_ref_geometry import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
)
from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice,
)
from serenitymojo.pipeline.minimax_h3_torch_cpu_rng import (
    minimax_h3_torch_cpu_randn,
)
from serenitymojo.pipeline.minimax_h3_video_vae_spatial_tiling import (
    MiniMaxH3TilingConfig,
)
from serenitymojo.pipeline.minimax_h3_video_vae_temporal import (
    minimax_h3_video_encode_temporal,
    minimax_h3_video_released_temporal_config,
)


# The video VAE's pixel convention: ImageNet-normalized RGB over a [0,1] base
# range (packing.py:70-71). NOT the conditioner's mean=std=0.5.
comptime MINIMAX_H3_PIXEL_MEAN_R = Float32(0.485)
comptime MINIMAX_H3_PIXEL_MEAN_G = Float32(0.456)
comptime MINIMAX_H3_PIXEL_MEAN_B = Float32(0.406)
comptime MINIMAX_H3_PIXEL_STD_R = Float32(0.229)
comptime MINIMAX_H3_PIXEL_STD_G = Float32(0.224)
comptime MINIMAX_H3_PIXEL_STD_B = Float32(0.225)

# The posterior sample of a visual reference's encode is drawn off a generator
# seeded 42, fixed independently of the request seed (packing.py:87).
comptime MINIMAX_H3_KEYFRAME_ENCODE_SEED = 42

comptime MINIMAX_H3_VIDEO_LATENT_CHANNELS = 24
comptime MINIMAX_H3_AUDIO_CHANNELS = 2


def _round_f32(var x: Float32) -> Float32:
    """Optimization barrier forcing one float32 rounding — the same device as
    `models/minimax_h3/scheduler.mojo::_round_f32`, for the same reason: Mojo
    at -O2 contracts `a*b - c` / `a + b*c` into fused multiply-adds, while the
    reference runs the multiply and the add/sub as separate torch kernels with
    a rounding between them. MEASURED here 2026-08-03 on the /255 reciprocal
    multiply feeding the mean subtraction: contracted, 154 of 256 byte values
    (and 260M of 436M pixels of a real clip, ref-encode GPU gate) land 1 ULP
    off torch; behind this barrier, 0 of 256. `black_box` is an inline-asm
    clobber the optimizer cannot see through — `@no_inline` is NOT enough (the
    scheduler's note: the call site keeps its `contract` fast-math flag)."""
    return black_box(x)


def minimax_h3_video_latents_mean() -> List[Float32]:
    """`latents_mean` of the video VAE, 24 channels.

    Hardcoded from the vendor's own landed `video_vae/config.json` — the same
    convention `minimax_h3_t2va.mojo` uses, and verified BYTE-IDENTICAL between
    the FL2VA and Ref2VA checkpoints, so ref2va needs no separate table."""
    return [
        Float32(0.858090341091156), Float32(-0.9606591463088989),
        Float32(1.0661640167236328), Float32(-0.5090325474739075),
        Float32(-0.2727581858634949), Float32(-1.3675414323806763),
        Float32(-0.2553254961967468), Float32(-0.26907554268836975),
        Float32(-0.5376840829849243), Float32(-0.0464097298681736),
        Float32(0.6657370328903198), Float32(0.19690127670764923),
        Float32(-0.5460608005523682), Float32(-0.4035342037677765),
        Float32(-0.23683024942874908), Float32(0.25928452610969543),
        Float32(-0.30133944749832153), Float32(0.211341992020607),
        Float32(-1.1206848621368408), Float32(0.3581933379173279),
        Float32(-0.04225143790245056), Float32(0.2604829967021942),
        Float32(0.22864092886447906), Float32(0.7056031823158264),
    ]


def minimax_h3_video_latents_std() -> List[Float32]:
    """`latents_std` of the video VAE, 24 channels. See the mean's note."""
    return [
        Float32(1.2223774194717407), Float32(1.2767263650894165),
        Float32(1.6831774711608887), Float32(1.7549455165863037),
        Float32(1.5636216402053833), Float32(2.194143533706665),
        Float32(0.9653137922286987), Float32(1.0569885969161987),
        Float32(0.841948926448822), Float32(0.7729952931404114),
        Float32(1.8955937623977661), Float32(0.946841835975647),
        Float32(0.7996809482574463), Float32(0.44988900423049927),
        Float32(0.7197399735450745), Float32(0.6936293244361877),
        Float32(2.961095094680786), Float32(2.7694199085235596),
        Float32(3.0496184825897217), Float32(2.1088054180145264),
        Float32(3.276226282119751), Float32(3.1627357006073),
        Float32(2.2816812992095947), Float32(2.6127843856811523),
    ]


def minimax_h3_pixel_normalize_frames(
    rgb: List[UInt8], num_frames: Int, height: Int, width: Int
) raises -> List[Float32]:
    """Channels-LAST `uint8` RGB -> channels-FIRST normalized f32.

    Input is `[num_frames, height, width, 3]`, the layout
    `pipeline/minimax_h3_media_in.mojo` produces and the vendor's own numpy
    frame array. Output is `[3, num_frames, height, width]`, i.e. the vendor's
    `[1, 3, T, H, W]` without the batch axis, so the transpose and the
    normalization happen in ONE pass.

    `(x/255 - mean) / std` with the video VAE's ImageNet constants. An image
    reference is just `num_frames = 1`.

    THE /255 IS A RECIPROCAL MULTIPLY, NOT A DIVISION — and that is measured,
    not stylistic. The vendor runs `pixels.to(torch.float32).div(255.0)` ON
    the execution device (encoders.py:576), and torch's CUDA true-div kernel
    computes a CPU-scalar divisor as `a * (1/b)` with the reciprocal formed
    once in f32 (ATen BinaryDivTrueKernel.cu, the `iter.is_cpu_scalar(2)`
    path). Verified on this machine 2026-08-03: CUDA `.div(255.0)` ==
    `x * (1.0f/255.0f)` on all 256 byte values, while true f32 division
    differs on 126 of them — the ref-encode GPU gate caught the mismatch at
    max_abs 7.15e-7 over 18% of a real clip's pixels before this was fixed.
    The subtract and the /std stay true elementwise ops (std is a TENSOR
    divisor, which torch divides for real)."""
    if num_frames < 1 or height < 1 or width < 1:
        raise Error("minimax_h3_ref_encode: frame geometry must be positive")
    var expected = num_frames * height * width * 3
    if len(rgb) != expected:
        raise Error(
            String("minimax_h3_ref_encode: pixel buffer holds ") + String(len(rgb))
            + " bytes, geometry implies " + String(expected)
        )

    var mean = [
        MINIMAX_H3_PIXEL_MEAN_R, MINIMAX_H3_PIXEL_MEAN_G, MINIMAX_H3_PIXEL_MEAN_B
    ]
    var std = [
        MINIMAX_H3_PIXEL_STD_R, MINIMAX_H3_PIXEL_STD_G, MINIMAX_H3_PIXEL_STD_B
    ]
    # torch-CUDA's scalar true-div: reciprocal formed ONCE in f32, then a
    # multiply per element (see this docstring's measured note).
    var inv255 = Float32(1.0) / Float32(255.0)
    var plane = num_frames * height * width
    var out = List[Float32]()
    out.resize(3 * plane, Float32(0.0))
    for f in range(num_frames):
        for y in range(height):
            for x in range(width):
                var src = ((f * height + y) * width + x) * 3
                for c in range(3):
                    # `_round_f32` stops the mul from contracting into the
                    # subtraction as an FMA — see its docstring.
                    var value = _round_f32(Float32(Int(rgb[src + c])) * inv255)
                    var dst = (c * num_frames + f) * height * width + y * width + x
                    out[dst] = (value - mean[c]) / std[c]
    return out^


def minimax_h3_fp16_round(x: Float32) -> Float32:
    """One f32 -> fp16 -> f32 round trip.

    Step 5 of the visual recipe. Drops the mantissa to 10 explicit bits, which
    the released model's conditioning depends on."""
    return Float32(Float16(x))


def minimax_h3_fp16_round_latents(latents: List[Float32]) -> List[Float32]:
    """`minimax_h3_fp16_round` over a whole latent buffer.

    Applied to the SAMPLED latent BEFORE normalization — the order matters:
    rounding after normalizing would quantize a differently-scaled value and
    land on different bits."""
    var out = List[Float32]()
    out.resize(len(latents), Float32(0.0))
    for i in range(len(latents)):
        out[i] = minimax_h3_fp16_round(latents[i])
    return out^


def minimax_h3_normalize_video_latents(
    latents: List[Float32],
    channels: Int,
    num_frames: Int,
    height: Int,
    width: Int,
) raises -> List[Float32]:
    """`(latents - latents_mean) / latents_std`, per channel.

    `latents` is `[C, T, H, W]` flat row-major — the layout
    `minimax_h3_patchify_video` consumes. This is the INVERSE of the
    denormalization the t2va decode tail applies (`z*ls + lm`)."""
    if channels != MINIMAX_H3_VIDEO_LATENT_CHANNELS:
        raise Error(
            String("minimax_h3_ref_encode: expected ")
            + String(MINIMAX_H3_VIDEO_LATENT_CHANNELS) + " latent channels, got "
            + String(channels)
        )
    var expected = channels * num_frames * height * width
    if len(latents) != expected:
        raise Error(
            String("minimax_h3_ref_encode: latent buffer holds ")
            + String(len(latents)) + ", geometry implies " + String(expected)
        )
    var mean = minimax_h3_video_latents_mean()
    var std = minimax_h3_video_latents_std()
    var stride = num_frames * height * width
    var out = List[Float32]()
    out.resize(expected, Float32(0.0))
    for c in range(channels):
        for i in range(stride):
            var at = c * stride + i
            out[at] = (latents[at] - mean[c]) / std[c]
    return out^


def minimax_h3_video_condition_rows(
    sampled_latents: List[Float32],
    channels: Int,
    num_frames: Int,
    height: Int,
    width: Int,
    patch_t: Int = 1,
    patch_h: Int = 2,
    patch_w: Int = 2,
) raises -> List[Float32]:
    """Steps 5-6 for one visual reference: fp16 round -> normalize -> patchify.

    `sampled_latents` is the POSTERIOR SAMPLE straight off the VAE, `[C, T, H,
    W]`, before any rounding. Returns `[T'*H'*W', C*pt*ph*pw]` rows, channel-
    slowest within a row, via the GATED
    `models/minimax_h3/rearrange.mojo::minimax_h3_patchify_video` (17/17 in the
    rearrange parity gate) — the row order is not re-derived here."""
    var rounded = minimax_h3_fp16_round_latents(sampled_latents)
    var normalized = minimax_h3_normalize_video_latents(
        rounded, channels, num_frames, height, width
    )
    return minimax_h3_patchify_video(
        normalized, channels, num_frames, height, width, patch_t, patch_h, patch_w
    )


def minimax_h3_audio_condition_rows(
    posterior_mode: List[Float32],
    latent_channels: Int,
    num_audio_latents: Int,
    latents_mean: List[Float32],
    latents_std: List[Float32],
) raises -> List[Float32]:
    """A soundtrack's clean condition rows: transpose -> normalize -> pack.

    `posterior_mode` is `[2, C, T]` — the audio VAE's MODE for the two stereo
    batch items, the layout the vendor gets before its `.transpose(1, 2)`.
    Returns `[2*T, C]` channel-major rows: all T rows of the left channel, then
    all T of the right, matching `models/minimax_h3/rearrange.mojo::
    minimax_h3_unpack_audio`'s inverse layout.

    NO fp16 rounding and NO sampling — both are exclusive to the visual branch.
    `latents_mean`/`latents_std` are the AUDIO VAE's, passed in rather than
    hardcoded because the t2va pipeline already owns that table."""
    if latent_channels < 1 or num_audio_latents < 1:
        raise Error("minimax_h3_ref_encode: audio geometry must be positive")
    var expected = MINIMAX_H3_AUDIO_CHANNELS * latent_channels * num_audio_latents
    if len(posterior_mode) != expected:
        raise Error(
            String("minimax_h3_ref_encode: audio latent buffer holds ")
            + String(len(posterior_mode)) + ", geometry implies " + String(expected)
        )
    if len(latents_mean) != latent_channels or len(latents_std) != latent_channels:
        raise Error(
            "minimax_h3_ref_encode: audio latents_mean/std must have one entry"
            " per latent channel"
        )

    var out = List[Float32]()
    out.resize(MINIMAX_H3_AUDIO_CHANNELS * num_audio_latents * latent_channels, Float32(0.0))
    for b in range(MINIMAX_H3_AUDIO_CHANNELS):
        for t in range(num_audio_latents):
            var row = b * num_audio_latents + t
            for c in range(latent_channels):
                var src = (b * latent_channels + c) * num_audio_latents + t
                out[row * latent_channels + c] = (
                    posterior_mode[src] - latents_mean[c]
                ) / latents_std[c]
    return out^


def minimax_h3_mix_condition_rows(
    condition_rows: List[Float32], noise_rows: List[Float32]
) raises -> List[Float32]:
    """Noise-augment the VISUAL condition rows at t = 0.999.

    `scale_noise(condition_rows, 0.999, noise_rows)` through the GATED
    scheduler, i.e. `0.999*x + 0.001*noise` with the reference's own per-op f32
    rounding. Applied ONCE, at encode time, at the CONSTANT augmentation level —
    NOT at the moving row timestep the transformer is conditioned on. Audio
    condition rows are never passed through this."""
    if len(condition_rows) != len(noise_rows):
        raise Error(
            String("minimax_h3_ref_encode: condition rows (")
            + String(len(condition_rows)) + ") and noise rows ("
            + String(len(noise_rows)) + ") differ in length"
        )
    # `shift` is irrelevant to `scale_noise` (it never touches the sigma grid),
    # so the schedule is deliberately not built here.
    var scheduler = MiniMaxH3Scheduler(Float32(12.0))
    return scheduler.scale_noise(
        condition_rows, MINIMAX_H3_KEYFRAME_NOISE_AUG, noise_rows
    )


# `DiagonalGaussianDistribution.__init__` clamps the log-variance before it is
# exponentiated (diffusers vae.py:691). Restated here (not imported from
# pipeline/minimax_h3_keyframe_encode.mojo, which would be an import cycle:
# that module imports THIS one).
comptime MINIMAX_H3_LOGVAR_MIN = Float32(-30.0)
comptime MINIMAX_H3_LOGVAR_MAX = Float32(20.0)

# The video VAE's spatial compression, `product(space_down=[2,2,2,2,1,1])`
# (video_vae/source/config.json).
comptime MINIMAX_H3_VAE_SPATIAL_RATIO = 16


def minimax_h3_encode_reference_visual_moments(
    encoder: MiniMaxH3VideoEncoderDevice,
    normalized_pixels: List[Float32],
    channels: Int,
    num_frames: Int,
    height: Int,
    width: Int,
    tiling: MiniMaxH3TilingConfig,
    ctx: DeviceContext,
) raises -> List[Float32]:
    """The DEVICE half of the seam: normalized pixels -> raw VAE moments.

    `normalized_pixels` is `[3, T, H, W]` flat channel-first — exactly what
    `minimax_h3_pixel_normalize_frames` emits — at the reference's own canvas,
    already 24 fps and already trimmed to `17n+5`. This is the VIDEO branch,
    the reference's `_encode` (autoencoder_kl_minimax_h3.py:772-795), i.e. the
    creator's `encode_temporal` (klvae.py:461-512): 17-frame temporal chunking
    with per-clip SPATIALLY TILED encode, concat, trailing `token_drop`
    moment-frames dropped — all via the vendor-gated
    `pipeline/minimax_h3_video_vae_temporal.mojo::minimax_h3_video_encode_
    temporal`, not re-derived here.

    Returns moments `[2C, T', H', W']` flat CHANNEL-FIRST (the vendor's own
    `[1, 2C, T', H', W']` without the batch axis), so the consumer's
    `torch.chunk(dim=1)` mean/logvar split is a contiguous halving."""
    if channels != 3:
        raise Error("minimax_h3_ref_encode: visual references are RGB (3 channels)")
    var expected = channels * num_frames * height * width
    if len(normalized_pixels) != expected:
        raise Error(
            String("minimax_h3_ref_encode: pixel buffer holds ")
            + String(len(normalized_pixels)) + ", geometry implies "
            + String(expected)
        )
    # 17n+5 is enforced by the latent-frame formula itself, which raises on
    # any other length (packing.mojo:243) — the caller must have trimmed.
    var latent_t = minimax_h3_video_latent_num_frames(num_frames)
    var latent_h = height // MINIMAX_H3_VAE_SPATIAL_RATIO
    var latent_w = width // MINIMAX_H3_VAE_SPATIAL_RATIO

    # [3, T, H, W] channel-first -> [1, T, H, W, 3] NDHWC, the device VAE's
    # layout contract. A pure index shuffle, no arithmetic — the gated pixel
    # normalization stays the only place the pixel math lives.
    var plane = num_frames * height * width
    var ndhwc = List[Float32]()
    ndhwc.resize(plane * 3, Float32(0.0))
    for c in range(3):
        for i in range(plane):
            ndhwc[i * 3 + c] = normalized_pixels[c * plane + i]
    var shape: List[Int] = [1, num_frames, height, width, 3]
    var pixels = Tensor.from_host(ndhwc, shape^, STDtype.F32, ctx)

    var tconfig = minimax_h3_video_released_temporal_config()
    var moments = minimax_h3_video_encode_temporal(encoder, pixels, tconfig, tiling, ctx)

    var ms = moments.shape()
    if (
        ms[0] != 1 or ms[1] != latent_t or ms[2] != latent_h or ms[3] != latent_w
        or ms[4] != 2 * MINIMAX_H3_VIDEO_LATENT_CHANNELS
    ):
        raise Error(
            String("minimax_h3_ref_encode: encode_temporal returned [")
            + String(ms[0]) + ", " + String(ms[1]) + ", " + String(ms[2]) + ", "
            + String(ms[3]) + ", " + String(ms[4]) + "], expected [1, "
            + String(latent_t) + ", " + String(latent_h) + ", " + String(latent_w)
            + ", " + String(2 * MINIMAX_H3_VIDEO_LATENT_CHANNELS) + "]"
        )

    # NDHWC host download -> channel-first [2C, T', H', W'].
    var host = moments.to_host(ctx)
    var two_c = 2 * MINIMAX_H3_VIDEO_LATENT_CHANNELS
    var lvol = latent_t * latent_h * latent_w
    var out = List[Float32]()
    out.resize(two_c * lvol, Float32(0.0))
    for i in range(lvol):
        for c in range(two_c):
            out[c * lvol + i] = host[i * two_c + c]
    return out^


def minimax_h3_reference_posterior_sample(
    moments: List[Float32],
    num_latent_frames: Int,
    latent_height: Int,
    latent_width: Int,
    seed: UInt64 = UInt64(MINIMAX_H3_KEYFRAME_ENCODE_SEED),
) raises -> List[Float32]:
    """The DRAW half of the seam: `DiagonalGaussianDistribution(moments)
    .sample(generator=torch.Generator().manual_seed(42))` (encoders.py:584-585).

    `moments` is `[2C, T', H', W']` flat channel-first. Mean is the first C
    channels and logvar the last C (`torch.chunk(parameters, 2, dim=1)`,
    vae.py:690), the logvar is clamped to [-30, 20] before exponentiation
    (vae.py:691), and the noise is `randn` over the mean's own `[1, C, T', H',
    W']` shape — channel-slowest, which is exactly this layout — drawn through
    `pipeline/minimax_h3_torch_cpu_rng.mojo` (torch's own CPU MT19937 stream;
    its probe quantifies the AVX2-vs-scalar tail gap). Returns the sample
    `[C, T', H', W']` flat, BEFORE the fp16 round —
    `minimax_h3_video_condition_rows` consumes it."""
    var z = MINIMAX_H3_VIDEO_LATENT_CHANNELS
    var lvol = num_latent_frames * latent_height * latent_width
    if len(moments) != 2 * z * lvol:
        raise Error(
            String("minimax_h3_ref_encode: moments hold ") + String(len(moments))
            + " values, geometry implies " + String(2 * z * lvol)
        )
    var noise = minimax_h3_torch_cpu_randn(z * lvol, seed)
    var out = List[Float32]()
    out.resize(z * lvol, Float32(0.0))
    for i in range(z * lvol):
        var mean = moments[i]
        var logvar = moments[z * lvol + i]
        if logvar < MINIMAX_H3_LOGVAR_MIN:
            logvar = MINIMAX_H3_LOGVAR_MIN
        if logvar > MINIMAX_H3_LOGVAR_MAX:
            logvar = MINIMAX_H3_LOGVAR_MAX
        var std = exp(Float32(0.5) * logvar)
        # `mean + std*noise` is `x = self.mean + self.std * sample`
        # (vae.py:708) — two torch kernels, two roundings; the barrier stops
        # the FMA contraction that would merge them (see `_round_f32`).
        out[i] = mean + _round_f32(std * noise[i])
    return out^


def minimax_h3_encode_reference_visual_seam(
    encoder: MiniMaxH3VideoEncoderDevice,
    normalized_pixels: List[Float32],
    channels: Int,
    num_frames: Int,
    height: Int,
    width: Int,
    tiling: MiniMaxH3TilingConfig,
    ctx: DeviceContext,
) raises -> List[Float32]:
    """THE SEAM, wired: pixels -> VAE moments -> posterior SAMPLE.

    `minimax_h3_pixel_normalize_frames` feeds it and
    `minimax_h3_video_condition_rows` consumes what it returns (`[C, T', H',
    W']` flat, before the fp16 round). VIDEO references only — the `_encode`
    temporal-chunking path; an image reference's single-frame `_encode_clip`
    path lives in `pipeline/minimax_h3_keyframe_encode.mojo`, which shares
    every host stage of this file. The posterior draw uses torch's own CPU
    MT19937 stream at seed 42, fixed independently of the request seed
    (packing.py:87)."""
    var latent_t = minimax_h3_video_latent_num_frames(num_frames)
    var moments = minimax_h3_encode_reference_visual_moments(
        encoder, normalized_pixels, channels, num_frames, height, width,
        tiling, ctx,
    )
    return minimax_h3_reference_posterior_sample(
        moments,
        latent_t,
        height // MINIMAX_H3_VAE_SPATIAL_RATIO,
        width // MINIMAX_H3_VAE_SPATIAL_RATIO,
    )
