# Native MiniMax-H3 one-frame target-cache preparation seam.
#
# Pinned oracle: kohya-ss/musubi-tuner
# b8717864713c9e4e7ef3d56eba1fc695a9b626a5.
#   minimax_h3_cache_latents.py::build_one_frame_latent_tensors (425-507)
#     canonical_item_key = f"{item_key}#1f"; H/W divisible by 32.
#   minimax_h3/video_vae.py::encode_video_target (663-667)
#   minimax_h3/video_vae.py::_video_posterior_sample (648-659)
#     CPU F32 randn(mean.shape) in contiguous NCTHW order.
#   minimax_h3/video_vae.py::MiniMaxH3VideoVAE.__init__ (397-448)
#     space_down=(2,2,2,2,1,1), so spatial compression is exactly 16.
#
# The existing native posterior accepts NDHWC moments/noise.  This wrapper
# draws the pinned CPU profile in the oracle's contiguous [1,24,1,H',W']
# order, transposes values explicitly to [1,1,H',W',24], and verifies the
# final cache shape [24,1,H',W'].  It does not write an artifact or mutate a
# dataset.  The metadata map is a writer seam, not proof an artifact used it.

from std.collections import Dict, List
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice,
)
from serenitymojo.pipeline.minimax_h3_keyframe_image import MiniMaxH3RgbImage
from serenitymojo.tensor import Tensor
from serenitymojo.training.minimax_h3.latent_cache_math import (
    MINIMAX_H3_VIDEO_MOMENT_CHANNELS,
    MINIMAX_H3_VIDEO_LATENT_CHANNELS,
    minimax_h3_encode_target_single_frame_cache,
    minimax_h3_target_noise_seed,
    minimax_h3_video_target_latent_cache_from_moments,
)
from serenitymojo.training.minimax_h3.sha256 import minimax_h3_sha256_text
from serenitymojo.training.minimax_h3.torch_cpu_randn_avx2_v1 import (
    TORCH_CPU_RANDN_F32_AVX2_V1,
    torch_cpu_randn_f32_avx2_v1,
)


comptime MINIMAX_H3_ONE_FRAME_TARGET_NOISE_RECEIPT_V1 = (
    "serenity.minimax_h3.one_frame_target_noise_receipt.v1"
)
comptime MINIMAX_H3_ONE_FRAME_TARGET_NOISE_LAYOUT = (
    "ncthw_contiguous_f32_to_ndhwc_f32"
)
comptime MINIMAX_H3_VIDEO_SPATIAL_COMPRESSION = 16


@fieldwise_init
struct MiniMaxH3OneFrameTargetNoisePlan(Copyable, Movable):
    """Host-only, deterministic plan that can be validated before CUDA."""

    var item_key: String
    var canonical_item_key: String
    var cache_seed: Int
    var derived_seed: Int
    var source_height: Int
    var source_width: Int
    var latent_height: Int
    var latent_width: Int
    var rng_profile: String
    var receipt_schema: String
    var receipt: String

    def noise_shape_ncthw(self) -> List[Int]:
        return [1, MINIMAX_H3_VIDEO_LATENT_CHANNELS, 1,
                self.latent_height, self.latent_width]

    def posterior_shape_ndhwc(self) -> List[Int]:
        return [1, 1, self.latent_height, self.latent_width,
                MINIMAX_H3_VIDEO_LATENT_CHANNELS]

    def moments_shape_ndhwc(self) -> List[Int]:
        return [1, 1, self.latent_height, self.latent_width,
                MINIMAX_H3_VIDEO_MOMENT_CHANNELS]

    def cache_shape_cthw(self) -> List[Int]:
        return [MINIMAX_H3_VIDEO_LATENT_CHANNELS, 1,
                self.latent_height, self.latent_width]


struct MiniMaxH3OneFrameTargetCachePrepared(Movable):
    """One prepared F32 cache tensor plus its validated RNG provenance."""

    var target_cache: Tensor
    var noise_plan: MiniMaxH3OneFrameTargetNoisePlan

    def __init__(
        out self,
        var target_cache: Tensor,
        var noise_plan: MiniMaxH3OneFrameTargetNoisePlan,
    ):
        self.target_cache = target_cache^
        self.noise_plan = noise_plan^


def _shape_string(shape: List[Int]) -> String:
    var output = String("")
    for index in range(len(shape)):
        if index > 0:
            output += String("x")
        output += String(shape[index])
    return output^


def _receipt_material(
    item_key: String,
    canonical_item_key: String,
    cache_seed: Int,
    derived_seed: Int,
    source_height: Int,
    source_width: Int,
    latent_height: Int,
    latent_width: Int,
    rng_profile: String,
) -> String:
    var noise_shape: List[Int] = [
        1, MINIMAX_H3_VIDEO_LATENT_CHANNELS, 1,
        latent_height, latent_width,
    ]
    var posterior_shape: List[Int] = [
        1, 1, latent_height, latent_width,
        MINIMAX_H3_VIDEO_LATENT_CHANNELS,
    ]
    var cache_shape: List[Int] = [
        MINIMAX_H3_VIDEO_LATENT_CHANNELS, 1,
        latent_height, latent_width,
    ]
    return (
        String(MINIMAX_H3_ONE_FRAME_TARGET_NOISE_RECEIPT_V1) + String("\n")
        + String("item_key_sha256=") + minimax_h3_sha256_text(item_key) + String("\n")
        + String("canonical_item_key_sha256=")
        + minimax_h3_sha256_text(canonical_item_key) + String("\n")
        + String("cache_seed=") + String(cache_seed) + String("\n")
        + String("derived_seed=") + String(derived_seed) + String("\n")
        + String("rng_profile=") + rng_profile + String("\n")
        + String("source_shape_hw=") + String(source_height) + String("x")
        + String(source_width) + String("\n")
        + String("noise_shape_ncthw=") + _shape_string(noise_shape) + String("\n")
        + String("posterior_shape_ndhwc=") + _shape_string(posterior_shape)
        + String("\n")
        + String("cache_shape_cthw=") + _shape_string(cache_shape) + String("\n")
    )


def _validate_geometry(source_height: Int, source_width: Int) raises:
    if source_height <= 0 or source_width <= 0:
        raise Error("MiniMax H3 one-frame target axes must be positive")
    if source_height % 32 != 0 or source_width % 32 != 0:
        raise Error(
            String("MiniMax H3 one-frame target axes must be divisible by 32, got ")
            + String(source_width) + String("x") + String(source_height)
        )


def _validate_plan(plan: MiniMaxH3OneFrameTargetNoisePlan) raises:
    if plan.item_key.byte_length() == 0:
        raise Error("MiniMax H3 one-frame target item_key must not be empty")
    _validate_geometry(plan.source_height, plan.source_width)
    if plan.rng_profile != String(TORCH_CPU_RANDN_F32_AVX2_V1):
        raise Error("MiniMax H3 one-frame target RNG profile mismatch")
    if (
        plan.receipt_schema
        != String(MINIMAX_H3_ONE_FRAME_TARGET_NOISE_RECEIPT_V1)
    ):
        raise Error("MiniMax H3 one-frame target RNG receipt schema mismatch")
    var canonical = plan.item_key + String("#1f")
    if plan.canonical_item_key != canonical:
        raise Error("MiniMax H3 one-frame canonical item key mismatch")
    var latent_height = (
        plan.source_height // MINIMAX_H3_VIDEO_SPATIAL_COMPRESSION
    )
    var latent_width = (
        plan.source_width // MINIMAX_H3_VIDEO_SPATIAL_COMPRESSION
    )
    if (
        plan.latent_height != latent_height
        or plan.latent_width != latent_width
    ):
        raise Error("MiniMax H3 one-frame latent geometry mismatch")
    var seed = minimax_h3_target_noise_seed(
        plan.cache_seed, plan.canonical_item_key,
    )
    if plan.derived_seed != seed:
        raise Error("MiniMax H3 one-frame target derived seed mismatch")
    var material = _receipt_material(
        plan.item_key,
        plan.canonical_item_key,
        plan.cache_seed,
        plan.derived_seed,
        plan.source_height,
        plan.source_width,
        plan.latent_height,
        plan.latent_width,
        plan.rng_profile,
    )
    if plan.receipt != minimax_h3_sha256_text(material):
        raise Error("MiniMax H3 one-frame target RNG receipt mismatch")


def minimax_h3_plan_one_frame_target_noise(
    item_key: String,
    cache_seed: Int,
    source_height: Int,
    source_width: Int,
    rng_profile: String = String(TORCH_CPU_RANDN_F32_AVX2_V1),
) raises -> MiniMaxH3OneFrameTargetNoisePlan:
    """Create and validate the exact host plan; safe before DeviceContext."""
    if item_key.byte_length() == 0:
        raise Error("MiniMax H3 one-frame target item_key must not be empty")
    _validate_geometry(source_height, source_width)
    if rng_profile != String(TORCH_CPU_RANDN_F32_AVX2_V1):
        raise Error("MiniMax H3 one-frame target RNG profile is unsupported")
    var canonical_item_key = item_key + String("#1f")
    var derived_seed = minimax_h3_target_noise_seed(
        cache_seed, canonical_item_key,
    )
    var latent_height = source_height // MINIMAX_H3_VIDEO_SPATIAL_COMPRESSION
    var latent_width = source_width // MINIMAX_H3_VIDEO_SPATIAL_COMPRESSION
    var material = _receipt_material(
        item_key,
        canonical_item_key,
        cache_seed,
        derived_seed,
        source_height,
        source_width,
        latent_height,
        latent_width,
        rng_profile,
    )
    var plan = MiniMaxH3OneFrameTargetNoisePlan(
        item_key,
        canonical_item_key,
        cache_seed,
        derived_seed,
        source_height,
        source_width,
        latent_height,
        latent_width,
        rng_profile,
        String(MINIMAX_H3_ONE_FRAME_TARGET_NOISE_RECEIPT_V1),
        minimax_h3_sha256_text(material),
    )
    _validate_plan(plan)
    return plan^


def minimax_h3_one_frame_target_noise_receipt_material(
    plan: MiniMaxH3OneFrameTargetNoisePlan,
) raises -> String:
    _validate_plan(plan)
    return _receipt_material(
        plan.item_key,
        plan.canonical_item_key,
        plan.cache_seed,
        plan.derived_seed,
        plan.source_height,
        plan.source_width,
        plan.latent_height,
        plan.latent_width,
        plan.rng_profile,
    )


def minimax_h3_one_frame_target_cache_rng_metadata(
    plan: MiniMaxH3OneFrameTargetNoisePlan,
) raises -> Dict[String, String]:
    """Metadata additions required when a future writer persists this plan."""
    _validate_plan(plan)
    var metadata = Dict[String, String]()
    metadata[String("h3_target_rng_profile")] = plan.rng_profile.copy()
    metadata[String("h3_target_rng_receipt_schema")] = plan.receipt_schema.copy()
    metadata[String("h3_target_rng_receipt")] = plan.receipt.copy()
    metadata[String("h3_target_rng_canonical_item_key")] = (
        plan.canonical_item_key.copy()
    )
    metadata[String("h3_target_rng_seed")] = String(plan.derived_seed)
    metadata[String("h3_target_rng_noise_layout")] = String(
        MINIMAX_H3_ONE_FRAME_TARGET_NOISE_LAYOUT
    )
    metadata[String("h3_target_cache_shape")] = _shape_string(
        plan.cache_shape_cthw()
    )
    return metadata^


def minimax_h3_one_frame_target_noise_ncthw_values(
    plan: MiniMaxH3OneFrameTargetNoisePlan,
) raises -> List[Float32]:
    """Draw exact contiguous `[1,24,1,H',W']` F32 values on the host."""
    _validate_plan(plan)
    return torch_cpu_randn_f32_avx2_v1(
        plan.noise_shape_ncthw(),
        UInt64(plan.derived_seed),
        plan.rng_profile,
    )


def minimax_h3_one_frame_target_noise_ndhwc_values(
    plan: MiniMaxH3OneFrameTargetNoisePlan,
) raises -> List[Float32]:
    """Draw NCTHW, then explicitly transpose to posterior NDHWC order."""
    _validate_plan(plan)
    var source = minimax_h3_one_frame_target_noise_ncthw_values(plan)
    var plane = plan.latent_height * plan.latent_width
    var output = List[Float32](capacity=len(source))
    output.resize(len(source), Float32(0.0))
    for y in range(plan.latent_height):
        for x in range(plan.latent_width):
            var spatial = y * plan.latent_width + x
            for channel in range(MINIMAX_H3_VIDEO_LATENT_CHANNELS):
                var source_index = channel * plane + spatial
                var output_index = spatial * MINIMAX_H3_VIDEO_LATENT_CHANNELS + channel
                output[output_index] = source[source_index]
    return output^


def minimax_h3_one_frame_target_noise_ndhwc_device(
    plan: MiniMaxH3OneFrameTargetNoisePlan,
    ctx: DeviceContext,
) raises -> Tensor:
    var values = minimax_h3_one_frame_target_noise_ndhwc_values(plan)
    return Tensor.from_host(
        values, plan.posterior_shape_ndhwc(), STDtype.F32, ctx,
    )


def _validate_target_cache(
    target_cache: Tensor,
    plan: MiniMaxH3OneFrameTargetNoisePlan,
) raises:
    if target_cache.dtype() != STDtype.F32:
        raise Error("MiniMax H3 one-frame target cache must be F32")
    if target_cache.shape() != plan.cache_shape_cthw():
        raise Error("MiniMax H3 one-frame target cache shape mismatch")


def minimax_h3_prepare_one_frame_target_cache_from_moments(
    plan: MiniMaxH3OneFrameTargetNoisePlan,
    moments: Tensor,
    ctx: DeviceContext,
) raises -> MiniMaxH3OneFrameTargetCachePrepared:
    """Generate exact native noise and apply the existing NDHWC posterior."""
    _validate_plan(plan)
    if moments.dtype() != STDtype.F32:
        raise Error("MiniMax H3 one-frame target moments must be F32")
    if moments.shape() != plan.moments_shape_ndhwc():
        raise Error("MiniMax H3 one-frame target moments shape mismatch")
    var noise = minimax_h3_one_frame_target_noise_ndhwc_device(plan, ctx)
    var optional_noise = Optional[Tensor](noise^)
    var target_cache = minimax_h3_video_target_latent_cache_from_moments(
        moments, optional_noise, ctx,
    )
    _validate_target_cache(target_cache, plan)
    return MiniMaxH3OneFrameTargetCachePrepared(
        target_cache^, plan.copy(),
    )


def minimax_h3_encode_one_frame_target_cache_native(
    plan: MiniMaxH3OneFrameTargetNoisePlan,
    encoder: MiniMaxH3VideoEncoderDevice,
    image: MiniMaxH3RgbImage,
    ctx: DeviceContext,
) raises -> MiniMaxH3OneFrameTargetCachePrepared:
    """Image/VAE wrapper; caller should create and validate `plan` pre-CUDA."""
    _validate_plan(plan)
    image.validate()
    if image.height != plan.source_height or image.width != plan.source_width:
        raise Error("MiniMax H3 one-frame image geometry differs from RNG plan")
    # Draw/transpose before VAE execution, so profile failures cannot occur
    # after expensive device encoding begins.
    var noise = minimax_h3_one_frame_target_noise_ndhwc_device(plan, ctx)
    var optional_noise = Optional[Tensor](noise^)
    var target_cache = minimax_h3_encode_target_single_frame_cache(
        encoder, image, optional_noise, ctx,
    )
    _validate_target_cache(target_cache, plan)
    return MiniMaxH3OneFrameTargetCachePrepared(
        target_cache^, plan.copy(),
    )
