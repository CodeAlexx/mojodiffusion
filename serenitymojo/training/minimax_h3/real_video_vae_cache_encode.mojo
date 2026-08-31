# Native real-image to tiled F32 VideoVAE moments seam for MiniMax-H3 caches.
#
# This cache-side wrapper deliberately stops at raw moments.  Posterior RNG,
# cache serialization, trainer loading, and training are separate contracts.

from max.gpu.host import DeviceContext

from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice,
)
from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.minimax_h3.bucket_geometry import (
    minimax_h3_select_bucket,
)
from serenitymojo.training.minimax_h3.image_preprocess import (
    minimax_h3_resize_image_to_bucket,
)
from serenitymojo.training.minimax_h3.latent_cache_math import (
    MINIMAX_H3_VIDEO_MOMENT_CHANNELS,
    minimax_h3_video_vae_input_single_frame_device,
)
from serenitymojo.training.minimax_h3.sha256 import minimax_h3_sha256_bytes
from serenitymojo.training.minimax_h3.source_image import (
    minimax_h3_decode_source_rgb8,
)
from serenitymojo.training.minimax_h3.video_vae_tiling import (
    minimax_h3_encode_one_frame_moments_tiled_f32,
)


comptime MINIMAX_H3_SOURCE_ADMISSION_MULTIPLE = 32
comptime MINIMAX_H3_VIDEO_VAE_SPATIAL_COMPRESSION = 16


struct MiniMaxH3RealOneFrameMomentsF32(Movable):
    var moments: Tensor
    var source_width: Int
    var source_height: Int
    var bucket_width: Int
    var bucket_height: Int
    var moment_width: Int
    var moment_height: Int
    var decoded_rgb_sha256: String
    var prepared_rgb_sha256: String

    def __init__(
        out self,
        var moments: Tensor,
        source_width: Int,
        source_height: Int,
        bucket_width: Int,
        bucket_height: Int,
        moment_width: Int,
        moment_height: Int,
        var decoded_rgb_sha256: String,
        var prepared_rgb_sha256: String,
    ):
        self.moments = moments^
        self.source_width = source_width
        self.source_height = source_height
        self.bucket_width = bucket_width
        self.bucket_height = bucket_height
        self.moment_width = moment_width
        self.moment_height = moment_height
        self.decoded_rgb_sha256 = decoded_rgb_sha256^
        self.prepared_rgb_sha256 = prepared_rgb_sha256^


def minimax_h3_real_one_frame_moments_f32(
    encoder: MiniMaxH3VideoEncoderDevice,
    image_path: String,
    ctx: DeviceContext,
) raises -> MiniMaxH3RealOneFrameMomentsF32:
    """Decode, bucket, preprocess, and tiled-encode one real source image.

    `encoder` must come from `MiniMaxH3VideoEncoderDevice.load_f32_compute`;
    the tiled encoder checks every resident weight before launching.  The
    The source bucket is admitted on `/32` axes because the downstream DiT
    packs the `/16` posterior with a 2x2 spatial patch.  This seam stops before
    that packing: returned raw moments are exactly NDHWC
    `[1,1,H/16,W/16,48]`, never `/32`.
    """
    var decoded = minimax_h3_decode_source_rgb8(image_path)
    var source_width = decoded.width
    var source_height = decoded.height
    var decoded_sha = minimax_h3_sha256_bytes(decoded.pixels)
    var bucket = minimax_h3_select_bucket(source_width, source_height)
    if (
        bucket.width % MINIMAX_H3_SOURCE_ADMISSION_MULTIPLE != 0
        or bucket.height % MINIMAX_H3_SOURCE_ADMISSION_MULTIPLE != 0
    ):
        raise Error("MiniMax H3 real VideoVAE cache bucket must be divisible by 32")
    var prepared = minimax_h3_resize_image_to_bucket(
        decoded, bucket.width, bucket.height,
    )
    var prepared_sha = minimax_h3_sha256_bytes(prepared.pixels)
    var pixels = minimax_h3_video_vae_input_single_frame_device(prepared, ctx)
    var moments = minimax_h3_encode_one_frame_moments_tiled_f32(
        encoder, pixels, ctx,
    )
    var moment_width = bucket.width // MINIMAX_H3_VIDEO_VAE_SPATIAL_COMPRESSION
    var moment_height = bucket.height // MINIMAX_H3_VIDEO_VAE_SPATIAL_COMPRESSION
    if moments.dtype() != STDtype.F32:
        raise Error("MiniMax H3 real VideoVAE moments must be F32")
    if moments.shape() != [
        1, 1, moment_height, moment_width, MINIMAX_H3_VIDEO_MOMENT_CHANNELS,
    ]:
        raise Error(
            "MiniMax H3 real VideoVAE raw moments must preserve /16 geometry"
        )
    return MiniMaxH3RealOneFrameMomentsF32(
        moments^,
        source_width,
        source_height,
        bucket.width,
        bucket.height,
        moment_width,
        moment_height,
        decoded_sha^,
        prepared_sha^,
    )
