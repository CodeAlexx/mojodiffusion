# serenitymojo/pipeline/minimax_h3_video_vae_pixel_norm.mojo
#
# MiniMax-H3 pixel normalize/denormalize — the pre-encode/post-decode pixel
# transform `pixel_norm_type: imagenet` selects (source/config.json).
# Authority: normalize.py, read in full.
#
#   get_normalize_transform(t)   = torchvision.transforms.Normalize(mean,std)
#                                   applied to raw RGB BEFORE the encoder:
#                                   y = (x - mean) / std, per channel.
#   get_denormalize_transform(t) = Normalize(-mean/std, 1/std) applied to the
#                                   decoder's raw output:
#                                   y = (x - (-mean/std)) / (1/std)
#                                     = x*std + mean, per channel — the exact
#                                   inverse of normalize (verified: substitute
#                                   x = normalize(x0) and the algebra returns
#                                   x0).
#
# NOT part of the encoder/decoder network (no weights, no keys) — this is
# raw-pixel preprocessing, deliberately a separate tiny module the pipeline
# calls immediately before `minimax_h3_video_encode_device`/`_temporal` and
# immediately after `minimax_h3_video_decode_device`/`_temporal`.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import add, div, mul, sub


@fieldwise_init
struct MiniMaxH3PixelNormConstants(Copyable, Movable):
    var mean: List[Float32]
    var std: List[Float32]


def minimax_h3_pixel_norm_constants(norm_type: String) raises -> MiniMaxH3PixelNormConstants:
    """`NORM_CONFIGS` (normalize.py:7-20), verbatim. `imagenet` is the
    released `source/config.json` default (`pixel_norm_type: "imagenet"`)."""
    if norm_type == "imagenet":
        return MiniMaxH3PixelNormConstants(
            [Float32(0.485), Float32(0.456), Float32(0.406)],
            [Float32(0.229), Float32(0.224), Float32(0.225)],
        )
    if norm_type == "simple":
        return MiniMaxH3PixelNormConstants(
            [Float32(0.5), Float32(0.5), Float32(0.5)],
            [Float32(0.5), Float32(0.5), Float32(0.5)],
        )
    if norm_type == "raw":
        return MiniMaxH3PixelNormConstants(
            [Float32(0.0), Float32(0.0), Float32(0.0)],
            [Float32(1.0), Float32(1.0), Float32(1.0)],
        )
    raise Error(String("MiniMax-H3 pixel norm: unknown norm_type ") + norm_type)


def minimax_h3_video_pixel_normalize(
    pixels: Tensor,  # [..., 3] NDHWC/NHWC, channel-last (this repo's house layout)
    constants: MiniMaxH3PixelNormConstants,
    ctx: DeviceContext,
) raises -> Tensor:
    """`(pixels - mean) / std`, per channel. Called BEFORE
    `minimax_h3_video_encode_device`/`_temporal`. Channel-last means the
    `[3]` mean/std tensors broadcast directly against the trailing axis
    (NumPy right-align) — no `[1,C,1,1]` reshape needed, unlike the
    reference's NCHW layout."""
    var mean_t = Tensor.from_host(constants.mean, [3], pixels.dtype(), ctx)
    var std_t = Tensor.from_host(constants.std, [3], pixels.dtype(), ctx)
    return div(sub(pixels, mean_t, ctx), std_t, ctx)


def minimax_h3_video_pixel_denormalize(
    pixels: Tensor,  # [..., 3] NDHWC/NHWC, the decoder's raw output
    constants: MiniMaxH3PixelNormConstants,
    ctx: DeviceContext,
) raises -> Tensor:
    """`pixels * std + mean`, per channel — the exact inverse of
    `minimax_h3_video_pixel_normalize`. Called AFTER
    `minimax_h3_video_decode_device`/`_temporal`."""
    var mean_t = Tensor.from_host(constants.mean, [3], pixels.dtype(), ctx)
    var std_t = Tensor.from_host(constants.std, [3], pixels.dtype(), ctx)
    return add(mul(pixels, std_t, ctx), mean_t, ctx)
