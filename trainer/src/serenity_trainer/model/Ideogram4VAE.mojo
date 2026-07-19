# Ideogram4VAE.mojo — Ideogram-4 Flux2-family AutoencoderKL (z=32): training
# encode (image -> packed normalized latent) + sampling decode seam.
#
# ════════════════════════════════════════════════════════════════════════════
# PORT SPEC — the EXACT ai-toolkit code path reproduced.
# ════════════════════════════════════════════════════════════════════════════
# ai-toolkit extensions_built_in/diffusion_models/ideogram4/ideogram4.py
#   encode_images (455-476):
#     moments = vae.encoder(image)        # diffusers AutoencoderKL Encoder,
#                                         #   INCLUDES quant_conv (autoencoder.py:201)
#     mean    = moments[:, :32]           # DiagonalGaussian mode (deterministic;
#                                         #   training uses the mean, NOT a sample)
#     patched = patchify_latents(mean, 2) # [1,32,H8,W8] -> [1,128,gh,gw]
#                                         #   (src/pipeline.py:40 permute 0,3,5,1,2,4)
#     latents = (patched - shift) / scale # per-128-ch latent norm
#                                         #   shift,scale = get_latent_norm() (latent_norm.py)
#   decode_latents (479-493) is the inverse: latents*scale+shift -> unpatchify ->
#     vae.decoder (which INCLUDES post_quant_conv, autoencoder.py:263).
#
# VAE topology (diffusers AutoencoderKL, ideogram-4-fp8/vae):
#   conv_in 3->128 ; down_blocks.0-3 (2 resnets each, 128/256/512/512,
#   downsamplers on 0-2) ; mid_block (resnets.0, attentions.0, resnets.1, 512) ;
#   conv_norm_out -> silu -> conv_out 512->64 ; quant_conv 64->64. latent_ch 32.
#
# BORROW BOUNDARY: the verified compute lives in serenitymojo (gated vs torch —
#   mean cos 0.9999586, normalized latents cos 0.9999568, 2026-06-07;
#   serenitymojo/models/vae/parity/ideogram4_vae_encode_probe.mojo). This file is
#   the serenity-side seam the dataLoader's "EncodeIdeogram4VAE" step calls.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.vae.ldm_encoder import (
    LdmVaeEncoder,
    load_ideogram4_vae_encoder,
    encode_ideogram4_latents,
    ideogram4_patchify_latents,
    ideogram4_normalize_latents,
)
from serenitymojo.models.vae.ldm_decoder import (
    LdmVaeDecoder,
    load_ideogram4_vae_decoder,
)

comptime IDEOGRAM4_VAE_LATENT_CHANNELS = 32
comptime IDEOGRAM4_VAE_PACKED_CHANNELS = 128
comptime IDEOGRAM4_VAE_DEFAULT_PATH = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors"
comptime IDEOGRAM4_LATENT_NORM_PATH = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors"


# Training-side encoder: image NCHW [1,3,8*LH,8*LW] (range [-1,1]) ->
# packed normalized latent [1,128,LH/2,LW/2] (== ai-toolkit batch.latents).
# LH/LW are the VAE-latent spatial dims (image/8); packed grid = LH/2 x LW/2.
struct Ideogram4VaeEncoder[LH: Int, LW: Int](Movable):
    var enc: LdmVaeEncoder[Self.LH, Self.LW, IDEOGRAM4_VAE_LATENT_CHANNELS]
    var latent_shift: Tensor   # [128]
    var latent_scale: Tensor   # [128]

    def __init__(
        out self,
        var enc: LdmVaeEncoder[Self.LH, Self.LW, IDEOGRAM4_VAE_LATENT_CHANNELS],
        var latent_shift: Tensor,
        var latent_scale: Tensor,
    ):
        self.enc = enc^
        self.latent_shift = latent_shift^
        self.latent_scale = latent_scale^

    @staticmethod
    def load(
        vae_path: String,
        latent_norm_path: String,
        ctx: DeviceContext,
    ) raises -> Ideogram4VaeEncoder[Self.LH, Self.LW]:
        var enc = load_ideogram4_vae_encoder[Self.LH, Self.LW](vae_path, ctx)
        var ln = ShardedSafeTensors.open(latent_norm_path)
        var shift = Tensor.from_view(ln.tensor_view(String("latent_shift")), ctx)
        var scale = Tensor.from_view(ln.tensor_view(String("latent_scale")), ctx)
        return Ideogram4VaeEncoder[Self.LH, Self.LW](enc^, shift^, scale^)

    @staticmethod
    def load_default(ctx: DeviceContext) raises -> Ideogram4VaeEncoder[Self.LH, Self.LW]:
        return Ideogram4VaeEncoder[Self.LH, Self.LW].load(
            String(IDEOGRAM4_VAE_DEFAULT_PATH),
            String(IDEOGRAM4_LATENT_NORM_PATH),
            ctx,
        )

    # ai-toolkit encode_images: image -> (patched - shift) / scale.
    def encode(self, image_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        return encode_ideogram4_latents[Self.LH, Self.LW](
            self.enc, image_nchw, self.latent_shift, self.latent_scale, ctx
        )

    # Seam: deterministic VAE mean latent [1,32,LH,LW] (pre-patchify, pre-norm).
    def encode_mean(self, image_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        return self.enc.encode_mean(image_nchw, ctx)


# Sampling-side decoder seam (latent*scale+shift -> unpatchify -> decoder).
# Reuses the gated serenitymojo decoder (chunk8 cos 0.999); exposed here so the
# Ideogram4 sampler/save path stays in serenity_trainer namespace.
def load_ideogram4_vae_decoder_for_sampling[
    LH: Int, LW: Int
](vae_path: String, ctx: DeviceContext) raises -> LdmVaeDecoder[LH, LW, IDEOGRAM4_VAE_LATENT_CHANNELS]:
    return load_ideogram4_vae_decoder[LH, LW](vae_path, ctx)
