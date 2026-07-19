# LensVAE.mojo — the Lens VAE wrapper (AutoencoderKLFlux2), the SAME VAE Klein
# uses (Flux2 family). Lens (LensModel.py) loads `AutoencoderKLFlux2` and applies
# the BatchNorm latent scaling exactly as Flux2Model does:
#     scale_latents   : (z - bn.running_mean) / sqrt(bn.running_var + batch_norm_eps)
#     unscale_latents : z * sqrt(bn.running_var + batch_norm_eps) + bn.running_mean
# (LensModel.py:301-326 scale_latents/unscale_latents; the bn stats live on the
# diffusers AutoencoderKLFlux2, autoencoder_kl_flux2.py:104,138-144,
# batch_norm_eps=1e-4 default.)
#
# BORROW boundary: the Flux2 VAE forward (encoder/decoder + the BN apply + the
# packed (un)patchify) was ALREADY copied into the serenity_trainer namespace as
# model/KleinVAE.mojo (itself borrowed line-for-line from serenitymojo
# klein_decoder.mojo / klein_encoder.mojo). Lens uses the identical VAE, so this
# file REUSES that port copy (NOT a serenitymojo import) and exposes the
# Lens-named seams (decode for the sampler + the bn stat accessors used by
# LensModel.scale_latents/unscale_latents).
#
# DTYPE: BF16 latent storage in/out; BN stats are F32 file params; the conv stack
# runs in the VAE weight dtype (F32 for flux2-vae.safetensors), F32 only in the
# foundation conv/norm registers.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors

# Reuse the port's already-copied Flux2 VAE (model/KleinVAE.mojo). These are
# serenity_trainer symbols, NOT serenitymojo imports.
from serenity_trainer.model.KleinVAE import (
    KleinVaeDecoder,
    _bn_apply,
    _load_bn_inv_scale,
    _load_bn_mean,
)


# ── Lens VAE constants (AutoencoderKLFlux2) ───────────────────────────────────
comptime LENS_LATENT_CH = 32         # autoencoder_kl_flux2.py:97 latent_channels
comptime LENS_PACKED_CH = 128        # prod(patch_size=2)^2 * latent_ch = 4*32
comptime LENS_BN_EPS = Float32(1.0e-4)  # AutoencoderKLFlux2.config.batch_norm_eps


# ── LensVAE[LH, LW] ───────────────────────────────────────────────────────────
# LH/LW are the PACKED latent spatial dims (post-patchify): for 1024x1024 the VAE
# latent is [1,32,128,128] -> patchify 2x2 -> packed [1,128,64,64] => LH=LW=64.
# The decoder upsamples 16x: decode([1,128,LH,LW]) -> [1,3,16*LH,16*LW].
struct LensVAE[LH: Int, LW: Int](Movable):
    var dec: KleinVaeDecoder[Self.LH, Self.LW]   # decode + unscale_latents (bn_scale=sqrt(var+eps), bn_bias=mean)
    var bn_inv_scale: Tensor                     # 1/sqrt(running_var + eps) [128] F32 (for scale_latents)
    var bn_mean: Tensor                          # running_mean              [128] F32

    def __init__(
        out self,
        var dec: KleinVaeDecoder[Self.LH, Self.LW],
        var bn_inv_scale: Tensor,
        var bn_mean: Tensor,
    ):
        self.dec = dec^
        self.bn_inv_scale = bn_inv_scale^
        self.bn_mean = bn_mean^

    @staticmethod
    def load(path: String, ctx: DeviceContext) raises -> LensVAE[Self.LH, Self.LW]:
        var st = ShardedSafeTensors.open(path)
        var inv = _load_bn_inv_scale(st, ctx)   # 1/sqrt(running_var+eps) [128] F32
        var mean = _load_bn_mean(st, ctx)       # running_mean            [128] F32
        var dec = KleinVaeDecoder[Self.LH, Self.LW].load(path, ctx)
        return LensVAE[Self.LH, Self.LW](dec^, inv^, mean^)

    # ── scale_latents (LensModel.py:301-318): (z - running_mean)/sqrt(var+eps) ──
    # Operates on PATCHIFIED+PACKED latents [B,128,h,w] (channel-packed); the bn
    # has 128 channels (== LENS_PACKED_CH). Mirrors _bn_apply[scale_mode=True].
    def scale_latents(self, packed_latent_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        return _bn_apply[True](packed_latent_nchw, self.bn_inv_scale, self.bn_mean, ctx)

    # ── unscale_latents (LensModel.py:321-326): z*sqrt(var+eps) + running_mean ──
    def unscale_latents(self, packed_latent_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        return self.dec.unscale_latents(packed_latent_nchw, ctx)

    # ── decode for the sampler (Flux2Sampler path): unscale -> unpatchify ->
    #    vae.decode. KleinVaeDecoder.decode does the inverse-BN(=unscale) +
    #    packed-unpatchify + conv decoder internally. ────────────────────────────
    def decode(self, packed_latent_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        # [1,128,LH,LW] -> [1,3,16*LH,16*LW]
        return self.dec.decode(packed_latent_nchw, ctx)

    # ── bn stat accessors used by LensModel.scale_latents/unscale_latents ───────
    def running_mean(self) -> Tensor:
        return self.bn_mean

    def bn_scale_sqrt(self) -> Tensor:
        # sqrt(running_var + eps) [128] F32 (the decoder holds this as bn_scale).
        return self.dec.bn_scale

    def batch_norm_eps(self) -> Float32:
        return LENS_BN_EPS
