# sampling/klein_reference_latent_bridge.mojo — no-weight ReferenceLatent bridge plan.
#
# This module turns the daemon's flattened Comfy ReferenceLatent metadata into
# the token/id contract the Klein edit sampler needs. It deliberately does not
# load Qwen, Klein DiT, or the VAE. The weight-bearing source-image decode/VAE
# encode step will feed the `reference_latent_nchw` argument later.

from std.collections import List
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.serve.backend import JobParams
from serenitymojo.sampling.img2img_refpack import (
    pack_latent_tokens,
    prepare_combined_img_ids,
    prepare_combined_tokens,
)


comptime KLEIN_REFERENCE_T_OFFSET = Float32(10.0)
comptime KLEIN_REFERENCE_LATENT_CHANNELS = 128
comptime KLEIN_REFERENCE_TEXT_TOKENS = 512


struct KleinReferenceLatentPlan(Copyable, Movable):
    var width: Int
    var height: Int
    var latent_h: Int
    var latent_w: Int
    var target_tokens: Int
    var reference_tokens: Int
    var combined_image_tokens: Int
    var text_tokens: Int
    var edit_sequence_tokens: Int
    var latent_channels: Int
    var reference_latent_links: Int
    var reference_t_offset: Float32
    var target_token_start: Int
    var reference_token_start: Int

    def __init__(out self):
        self.width = 0
        self.height = 0
        self.latent_h = 0
        self.latent_w = 0
        self.target_tokens = 0
        self.reference_tokens = 0
        self.combined_image_tokens = 0
        self.text_tokens = KLEIN_REFERENCE_TEXT_TOKENS
        self.edit_sequence_tokens = 0
        self.latent_channels = KLEIN_REFERENCE_LATENT_CHANNELS
        self.reference_latent_links = 0
        self.reference_t_offset = KLEIN_REFERENCE_T_OFFSET
        self.target_token_start = 0
        self.reference_token_start = 0


def plan_klein_reference_latent_bridge(params: JobParams) raises -> KleinReferenceLatentPlan:
    """Validate daemon metadata for the bounded SerenityFlow Klein edit shape.

    `reference_latent_count` is the number of Comfy ReferenceLatent conditioning
    links, not the number of distinct source images. The current SerenityFlow
    Klein edit templates have two links (positive and negative conditioning)
    sharing one source image latent.
    """
    if params.reference_image.byte_length() == 0:
        raise Error("klein ReferenceLatent bridge: reference_image is required")
    if params.reference_latent_count <= 0:
        raise Error("klein ReferenceLatent bridge: reference_latent_count must be positive")
    if params.reference_latent_method.byte_length() > 0 and params.reference_latent_method != String("index"):
        raise Error(
            String("klein ReferenceLatent bridge: unsupported reference_latent_method '")
            + params.reference_latent_method
            + String("'")
        )
    if params.init_image.byte_length() > 0 and params.init_image != params.reference_image:
        raise Error("klein ReferenceLatent bridge: init_image and reference_image disagree")
    if not (
        (params.width == 512 and params.height == 512)
        or (params.width == 1024 and params.height == 1024)
    ):
        raise Error(
            String("klein ReferenceLatent bridge: unsupported size ")
            + String(params.width)
            + String("x")
            + String(params.height)
            + String(" (expected 512x512 or 1024x1024)")
        )

    var out = KleinReferenceLatentPlan()
    out.width = params.width
    out.height = params.height
    out.latent_h = params.height // 16
    out.latent_w = params.width // 16
    out.target_tokens = out.latent_h * out.latent_w
    out.reference_tokens = out.target_tokens
    out.combined_image_tokens = out.target_tokens + out.reference_tokens
    out.text_tokens = KLEIN_REFERENCE_TEXT_TOKENS
    out.edit_sequence_tokens = out.text_tokens + out.combined_image_tokens
    out.latent_channels = KLEIN_REFERENCE_LATENT_CHANNELS
    out.reference_latent_links = params.reference_latent_count
    out.reference_t_offset = KLEIN_REFERENCE_T_OFFSET
    out.target_token_start = 0
    out.reference_token_start = out.target_tokens
    return out^


def require_klein_reference_latent_shape(
    latent: Tensor, plan: KleinReferenceLatentPlan, name: String
) raises:
    var sh = latent.shape()
    if (
        len(sh) != 4
        or sh[0] != 1
        or sh[1] != plan.latent_channels
        or sh[2] != plan.latent_h
        or sh[3] != plan.latent_w
    ):
        raise Error(
            name
            + String(" must be [1,")
            + String(plan.latent_channels)
            + String(",")
            + String(plan.latent_h)
            + String(",")
            + String(plan.latent_w)
            + String("]")
        )


def build_klein_reference_combined_img_ids(
    plan: KleinReferenceLatentPlan, ctx: DeviceContext
) raises -> Tensor:
    return prepare_combined_img_ids(
        plan.latent_h, plan.latent_w, plan.reference_t_offset, ctx
    )


def build_klein_reference_combined_tokens(
    noise_latent_nchw: Tensor,
    reference_latent_nchw: Tensor,
    plan: KleinReferenceLatentPlan,
    ctx: DeviceContext,
) raises -> Tensor:
    require_klein_reference_latent_shape(noise_latent_nchw, plan, String("noise_latent_nchw"))
    require_klein_reference_latent_shape(reference_latent_nchw, plan, String("reference_latent_nchw"))
    var noise_tokens = pack_latent_tokens(noise_latent_nchw, ctx)
    var reference_tokens = pack_latent_tokens(reference_latent_nchw, ctx)
    return prepare_combined_tokens(noise_tokens, reference_tokens, ctx)


def synthetic_klein_reference_latent(
    plan: KleinReferenceLatentPlan, offset: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Build a deterministic [1,128,LH,LW] latent for no-heavy bridge gates."""
    var vals = List[Float32]()
    for c in range(plan.latent_channels):
        for h in range(plan.latent_h):
            for w in range(plan.latent_w):
                vals.append(offset + Float32(c * 10000 + h * 100 + w))
    var sh = List[Int]()
    sh.append(1)
    sh.append(plan.latent_channels)
    sh.append(plan.latent_h)
    sh.append(plan.latent_w)
    return Tensor.from_host(vals, sh^, STDtype.F32, ctx)
