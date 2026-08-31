# MiniMax-H3 one-frame latent-cache math seam, pure Mojo/MAX.
#
# Pinned oracle: kohya-ss/musubi-tuner
# b8717864713c9e4e7ef3d56eba1fc695a9b626a5.
#   src/musubi_tuner/minimax_h3_cache_latents.py
#     sha256 a27d4541add4b256719de530a2daa5a3746d99a32ba168f5579a7e6cb69cb69b
#     _prepare_pixels:209-222
#   src/musubi_tuner/minimax_h3/video_vae.py
#     sha256 96e6698e5072adc258b6610881749d3748173d78c01c9b833e4cc42253165671
#     IMAGENET_MEAN/STDs:28-29, LATENTS_MEAN/STDs:33-85,
#     MiniMaxH3VideoVAE.encode_moments:607-616,
#     _video_posterior_sample:648-659, encode_video_target:663-667.
#
# This module deliberately does NOT implement Torch CPU RNG.  Target posterior
# noise is an explicit F32 Tensor input; the public optional-noise boundary
# rejects None before VAE work.  The cache builder can wire a proven RNG later
# without changing the pinned posterior/layout math here.

from std.benchmark import black_box
from std.collections import List
from std.gpu import global_idx
from std.math import exp
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice,
    minimax_h3_video_encode_device,
)
from serenitymojo.pipeline.minimax_h3_keyframe_image import MiniMaxH3RgbImage
from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.training.minimax_h3.sha256 import minimax_h3_sha256_text


comptime MINIMAX_H3_VIDEO_LATENT_CHANNELS = 24
comptime MINIMAX_H3_VIDEO_MOMENT_CHANNELS = 48
comptime MINIMAX_H3_VIDEO_LOGVAR_MIN = Float32(-30.0)
comptime MINIMAX_H3_VIDEO_LOGVAR_MAX = Float32(20.0)

comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


@always_inline
def _round_f32(value: Float32) -> Float32:
    """Prevent contraction across pinned Torch elementwise-op boundaries."""
    return black_box(value)


@always_inline
def _imagenet_mean(channel: Int) -> Float32:
    if channel == 0:
        return Float32(0.485)
    if channel == 1:
        return Float32(0.456)
    return Float32(0.406)


@always_inline
def _imagenet_std(channel: Int) -> Float32:
    if channel == 0:
        return Float32(0.229)
    if channel == 1:
        return Float32(0.224)
    return Float32(0.225)


def minimax_h3_prepare_pixels_single_frame_ndhwc(
    image: MiniMaxH3RgbImage,
) raises -> List[Float32]:
    """Pinned `_prepare_pixels` for RGB8 HWC, laid out as `[1,1,H,W,3]`.

    Musubi computes `uint8.float().div_(127.5).sub_(1.0)` and then permutes to
    NCFHW.  The existing native encoder consumes NDHWC, so only the logical
    layout differs; every pixel value and the single-frame/batch placement are
    identical.
    """
    image.validate()
    var out = List[Float32](capacity=len(image.pixels))
    for index in range(len(image.pixels)):
        var unit = _round_f32(
            Float32(Int(image.pixels[index])) / Float32(127.5)
        )
        out.append(_round_f32(unit - Float32(1.0)))
    return out^


def minimax_h3_video_vae_imagenet_input_single_frame_ndhwc(
    image: MiniMaxH3RgbImage,
) raises -> List[Float32]:
    """Pinned `encode_moments` pixel conversion in encoder NDHWC layout.

    Starting from `_prepare_pixels` in [-1,1], Musubi performs `(x+1)*0.5`
    and then `(x-IMAGENET_MEAN)/IMAGENET_STD`, each as a distinct F32
    elementwise operation.
    """
    var prepared = minimax_h3_prepare_pixels_single_frame_ndhwc(image)
    var out = List[Float32](capacity=len(prepared))
    for index in range(len(prepared)):
        var channel = index % 3
        var shifted = _round_f32(prepared[index] + Float32(1.0))
        var unit = _round_f32(shifted * Float32(0.5))
        var centered = _round_f32(unit - _imagenet_mean(channel))
        out.append(_round_f32(centered / _imagenet_std(channel)))
    return out^


def minimax_h3_video_vae_input_single_frame_device(
    image: MiniMaxH3RgbImage, ctx: DeviceContext,
) raises -> Tensor:
    """Upload the pinned ImageNet-normalized encoder input as F32 NDHWC."""
    var values = minimax_h3_video_vae_imagenet_input_single_frame_ndhwc(image)
    var shape: List[Int] = [1, 1, image.height, image.width, 3]
    return Tensor.from_host(values, shape^, STDtype.F32, ctx)


def minimax_h3_encode_single_frame_moments_device(
    encoder: MiniMaxH3VideoEncoderDevice,
    image: MiniMaxH3RgbImage,
    ctx: DeviceContext,
) raises -> Tensor:
    """Prepare one RGB8 image and call the existing native device encoder."""
    var pixels = minimax_h3_video_vae_input_single_frame_device(image, ctx)
    return minimax_h3_video_encode_device(encoder, pixels, ctx)


struct MiniMaxH3VideoMomentsF32(Movable):
    """Contiguous NDHWC halves of `[1,1,H,W,48]` moments."""

    var mean: Tensor
    var logvar: Tensor

    def __init__(out self, var mean: Tensor, var logvar: Tensor):
        self.mean = mean^
        self.logvar = logvar^


def minimax_h3_split_single_frame_video_moments_f32(
    moments: Tensor, ctx: DeviceContext,
) raises -> MiniMaxH3VideoMomentsF32:
    """Split NDHWC moments into F32 mean/logvar along the last channel."""
    var shape = moments.shape()
    if moments.dtype() != STDtype.F32:
        raise Error("MiniMax H3 latent cache: video moments must be F32")
    if (
        len(shape) != 5 or shape[0] != 1 or shape[1] != 1
        or shape[2] < 1 or shape[3] < 1
        or shape[4] != MINIMAX_H3_VIDEO_MOMENT_CHANNELS
    ):
        raise Error(
            "MiniMax H3 latent cache: expected one-frame moments "
            "[1,1,H,W,48]"
        )
    var mean = slice(
        moments, 4, 0, MINIMAX_H3_VIDEO_LATENT_CHANNELS, ctx,
    )
    var logvar = slice(
        moments,
        4,
        MINIMAX_H3_VIDEO_LATENT_CHANNELS,
        MINIMAX_H3_VIDEO_LATENT_CHANNELS,
        ctx,
    )
    return MiniMaxH3VideoMomentsF32(mean^, logvar^)


@always_inline
def _latents_mean(channel: Int) -> Float32:
    if channel == 0: return Float32(0.858090341091156)
    if channel == 1: return Float32(-0.9606591463088989)
    if channel == 2: return Float32(1.0661640167236328)
    if channel == 3: return Float32(-0.5090325474739075)
    if channel == 4: return Float32(-0.2727581858634949)
    if channel == 5: return Float32(-1.3675414323806763)
    if channel == 6: return Float32(-0.2553254961967468)
    if channel == 7: return Float32(-0.26907554268836975)
    if channel == 8: return Float32(-0.5376840829849243)
    if channel == 9: return Float32(-0.0464097298681736)
    if channel == 10: return Float32(0.6657370328903198)
    if channel == 11: return Float32(0.19690127670764923)
    if channel == 12: return Float32(-0.5460608005523682)
    if channel == 13: return Float32(-0.4035342037677765)
    if channel == 14: return Float32(-0.23683024942874908)
    if channel == 15: return Float32(0.25928452610969543)
    if channel == 16: return Float32(-0.30133944749832153)
    if channel == 17: return Float32(0.211341992020607)
    if channel == 18: return Float32(-1.1206848621368408)
    if channel == 19: return Float32(0.3581933379173279)
    if channel == 20: return Float32(-0.04225143790245056)
    if channel == 21: return Float32(0.2604829967021942)
    if channel == 22: return Float32(0.22864092886447906)
    return Float32(0.7056031823158264)


@always_inline
def _latents_std(channel: Int) -> Float32:
    if channel == 0: return Float32(1.2223774194717407)
    if channel == 1: return Float32(1.2767263650894165)
    if channel == 2: return Float32(1.6831774711608887)
    if channel == 3: return Float32(1.7549455165863037)
    if channel == 4: return Float32(1.5636216402053833)
    if channel == 5: return Float32(2.194143533706665)
    if channel == 6: return Float32(0.9653137922286987)
    if channel == 7: return Float32(1.0569885969161987)
    if channel == 8: return Float32(0.841948926448822)
    if channel == 9: return Float32(0.7729952931404114)
    if channel == 10: return Float32(1.8955937623977661)
    if channel == 11: return Float32(0.946841835975647)
    if channel == 12: return Float32(0.7996809482574463)
    if channel == 13: return Float32(0.44988900423049927)
    if channel == 14: return Float32(0.7197399735450745)
    if channel == 15: return Float32(0.6936293244361877)
    if channel == 16: return Float32(2.961095094680786)
    if channel == 17: return Float32(2.7694199085235596)
    if channel == 18: return Float32(3.0496184825897217)
    if channel == 19: return Float32(2.1088054180145264)
    if channel == 20: return Float32(3.276226282119751)
    if channel == 21: return Float32(3.1627357006073)
    if channel == 22: return Float32(2.2816812992095947)
    return Float32(2.6127843856811523)


def _posterior_sample_normalize_cthw_kernel(
    mean: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    logvar: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    noise: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    output: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    height_w: Int32,
    width_w: Int32,
    count_w: Int64,
):
    var height = Int(height_w)
    var width = Int(width_w)
    var count = Int(count_w)
    var output_index = Int(global_idx.x)
    if output_index >= count:
        return

    # Output is contiguous [C,1,H,W]; input halves/noise are NDHWC.
    var plane = height * width
    var channel = output_index // plane
    var spatial = output_index - channel * plane
    var source_index = spatial * MINIMAX_H3_VIDEO_LATENT_CHANNELS + channel
    var mean_value = rebind[Scalar[DType.float32]](mean[source_index])
    var logvar_value = rebind[Scalar[DType.float32]](logvar[source_index])
    var noise_value = rebind[Scalar[DType.float32]](noise[source_index])
    if logvar_value < MINIMAX_H3_VIDEO_LOGVAR_MIN:
        logvar_value = MINIMAX_H3_VIDEO_LOGVAR_MIN
    elif logvar_value > MINIMAX_H3_VIDEO_LOGVAR_MAX:
        logvar_value = MINIMAX_H3_VIDEO_LOGVAR_MAX
    var deviation = exp(Float32(0.5) * logvar_value)
    var sampled = mean_value + deviation * noise_value
    var normalized = (
        sampled - _latents_mean(channel)
    ) / _latents_std(channel)
    output[output_index] = rebind[output.element_type](normalized)


def _sample_normalize_single_frame_video_latent(
    mean: Tensor,
    logvar: Tensor,
    noise: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var mean_shape = mean.shape()
    var logvar_shape = logvar.shape()
    var noise_shape = noise.shape()
    if mean.dtype() != STDtype.F32 or logvar.dtype() != STDtype.F32:
        raise Error("MiniMax H3 latent cache: posterior moments must be F32")
    if noise.dtype() != STDtype.F32:
        raise Error("MiniMax H3 latent cache: injected posterior noise must be F32")
    if mean_shape != logvar_shape or mean_shape != noise_shape:
        raise Error(
            "MiniMax H3 latent cache: injected noise shape must equal mean/logvar"
        )
    if (
        len(mean_shape) != 5 or mean_shape[0] != 1 or mean_shape[1] != 1
        or mean_shape[2] < 1 or mean_shape[3] < 1
        or mean_shape[4] != MINIMAX_H3_VIDEO_LATENT_CHANNELS
    ):
        raise Error(
            "MiniMax H3 latent cache: posterior halves must be [1,1,H,W,24]"
        )

    var height = mean_shape[2]
    var width = mean_shape[3]
    var count = MINIMAX_H3_VIDEO_LATENT_CHANNELS * height * width
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](count * 4)
    var layout = RuntimeLayout[_DYN1].row_major(IndexList[1](count))
    var mean_tensor = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(mean.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=layout,
    )
    var logvar_tensor = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(logvar.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=layout,
    )
    var noise_tensor = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(noise.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=layout,
    )
    var output_tensor = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=layout,
    )
    var grid = (count + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_posterior_sample_normalize_cthw_kernel](
        mean_tensor,
        logvar_tensor,
        noise_tensor,
        output_tensor,
        Int32(height),
        Int32(width),
        Int64(count),
        grid_dim=grid,
        block_dim=_BLOCK,
    )
    var out_shape: List[Int] = [
        MINIMAX_H3_VIDEO_LATENT_CHANNELS, 1, height, width,
    ]
    return Tensor(out_buf^, out_shape^, STDtype.F32)


def minimax_h3_video_target_latent_cache_from_moments(
    moments: Tensor,
    noise: Optional[Tensor],
    ctx: DeviceContext,
) raises -> Tensor:
    """Sample and normalize one target latent, requiring injected F32 noise.

    The external noise values correspond to Musubi's CPU-Torch draw but are
    supplied in native NDHWC `[1,1,H,W,24]` layout.  No fallback RNG exists.
    """
    if not noise:
        raise Error(
            "MiniMax H3 latent cache: injected F32 posterior noise is required; "
            "Torch CPU RNG is not implemented"
        )
    var split = minimax_h3_split_single_frame_video_moments_f32(moments, ctx)
    return _sample_normalize_single_frame_video_latent(
        split.mean, split.logvar, noise.value(), ctx,
    )


def minimax_h3_encode_target_single_frame_cache(
    encoder: MiniMaxH3VideoEncoderDevice,
    image: MiniMaxH3RgbImage,
    noise: Optional[Tensor],
    ctx: DeviceContext,
) raises -> Tensor:
    """Bounded image -> existing VAE -> injected posterior -> `[24,1,H,W]`.

    Missing noise is rejected before image upload or encoder execution.
    """
    if not noise:
        raise Error(
            "MiniMax H3 latent cache: injected F32 posterior noise is required "
            "before VAE encode"
        )
    var moments = minimax_h3_encode_single_frame_moments_device(
        encoder, image, ctx,
    )
    return minimax_h3_video_target_latent_cache_from_moments(
        moments, noise, ctx,
    )


@always_inline
def _hex_nibble(value: UInt8) raises -> UInt64:
    if value >= UInt8(ord("0")) and value <= UInt8(ord("9")):
        return UInt64(value - UInt8(ord("0")))
    if value >= UInt8(ord("a")) and value <= UInt8(ord("f")):
        return UInt64(value - UInt8(ord("a")) + UInt8(10))
    raise Error("MiniMax H3 latent cache: internal SHA-256 hex is malformed")


def minimax_h3_target_noise_seed(
    cache_seed: Int, canonical_item_key: String,
) raises -> Int:
    """`sha256(f'{cache_seed}\\0{key}')[:8]` little-endian modulo 2^63."""
    var material = String(cache_seed) + chr(0) + canonical_item_key
    var digest = minimax_h3_sha256_text(material)
    if digest.byte_length() != 71:
        raise Error("MiniMax H3 latent cache: unexpected SHA-256 receipt length")
    var bytes = digest.as_bytes()
    var value = UInt64(0)
    for index in range(8):
        var at = 7 + 2 * index
        var byte = (_hex_nibble(bytes[at]) << UInt64(4)) | _hex_nibble(bytes[at + 1])
        value |= byte << UInt64(8 * index)
    return Int(value & UInt64(0x7FFFFFFFFFFFFFFF))
