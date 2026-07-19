# LensModel.mojo — the Lens model wrapper: 1:1 port of Serenity
# modules/model/LensModel.py (pr-1510). Mirrors the ZImage vertical's model-wrapper
# (latent (un)patchify/pack, BatchNorm latent scaling, timestep-shift, encode_text
# structure). The DiT forward lives in model/LensDiT.mojo; the VAE in
# model/LensVAE.mojo; the text encoder in model/LensTextEncoder.mojo.
#
# REFERENCE (the ONLY spec): modules/model/LensModel.py. Each function below first
# pastes the exact LensModel.py source (in a comment), then translates API/namespace
# only — math/structure/order do NOT change. Line cites are vs pr-1510 LensModel.py.
#
# Config (LensTransformer2DModel / checkpoint config.json): patch_size=2,
# in_channels=128, out_channels=32, num_layers=48, inner_dim=1536, num_heads=24,
# head_dim=64, enc_hidden_dim=2880, selected_layer_index=[5,11,17,23].
#
# DTYPE: BF16 storage in/out; F32 only in compute registers (kernels accumulate F32).

from std.math import exp as fexp
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import reshape, permute, slice

from serenity_trainer.model.LensVAE import LensVAE
# Re-export the SLICE-A cold-start LoRA builder so the parity smoke can import it
# from model.LensModel (the contract namespace). Implementation lives in LensDiT.
from serenity_trainer.model.LensDiT import build_lens_lora_set


comptime TArc = ArcPointer[Tensor]


# ── Lens / LensTransformer2DModel config constants (checkpoint config.json) ────
comptime LENS_PATCH_SIZE = 2
comptime LENS_IN_CHANNELS = 128       # img_in in-features (patchified packed latent)
comptime LENS_OUT_CHANNELS = 32       # AutoencoderKLFlux2 latent channels
comptime LENS_INNER_DIM = 1536
comptime LENS_NUM_HEADS = 24
comptime LENS_HEAD_DIM = 64
comptime LENS_NUM_LAYERS = 48
comptime LENS_ENC_HIDDEN_DIM = 2880   # GPT-OSS hidden size (per selected layer)
comptime LENS_N_TEXT_LAYERS = 4       # len(selected_layer_index) = [5,11,17,23]

# encode_text constants (LensModel.py:29-31)
comptime PROMPT_TEMPLATE_CROP_START = 97   # tokens consumed by the chat template prefix
comptime PROMPT_MAX_LENGTH = 512           # caption token budget


# ── shape helpers ─────────────────────────────────────────────────────────────
def _sh(*vals: Int) -> List[Int]:
    var s = List[Int]()
    for i in range(len(vals)):
        s.append(vals[i])
    return s^


# ══════════════════════════════════════════════════════════════════════════════
# noise_scheduler config slice used by calculate_timestep_shift. In Serenity
# these are read from self.noise_scheduler.config (FlowMatchEulerDiscreteScheduler);
# the Mojo loader fills them from the checkpoint scheduler_config.json. Defaults
# below are the diffusers FlowMatchEulerDiscreteScheduler defaults — the loader
# MUST overwrite them with the real checkpoint values before sampling/training.
# ══════════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct LensSchedulerConfig(Copyable, Movable):
    var base_image_seq_len: Int
    var max_image_seq_len: Int
    var base_shift: Float64
    var max_shift: Float64

    @staticmethod
    def diffusers_default() -> LensSchedulerConfig:
        return LensSchedulerConfig(256, 4096, 0.5, 1.15)


# ══════════════════════════════════════════════════════════════════════════════
# LensModel — static methods mirroring the LensModel.py class methods. Stateless
# tensor transforms; the VAE bn stats are passed in (python reads self.vae.bn).
# ══════════════════════════════════════════════════════════════════════════════
struct LensModel:

    # ── calculate_timestep_shift (LensModel.py:259-270) ────────────────────────
    #   base_seq_len = self.noise_scheduler.config.base_image_seq_len
    #   max_seq_len  = self.noise_scheduler.config.max_image_seq_len
    #   base_shift   = self.noise_scheduler.config.base_shift
    #   max_shift    = self.noise_scheduler.config.max_shift
    #   patch_size = 2
    #   image_seq_len = (latent_width // patch_size) * (latent_height // patch_size)
    #   m  = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    #   b  = base_shift - m * base_seq_len
    #   mu = image_seq_len * m + b
    #   return math.exp(mu)
    @staticmethod
    def calculate_timestep_shift(
        latent_height: Int, latent_width: Int, cfg: LensSchedulerConfig
    ) raises -> Float64:
        var base_seq_len = cfg.base_image_seq_len
        var max_seq_len = cfg.max_image_seq_len
        var base_shift = cfg.base_shift
        var max_shift = cfg.max_shift
        var patch_size = LENS_PATCH_SIZE
        var image_seq_len = (latent_width // patch_size) * (latent_height // patch_size)
        var m = (max_shift - base_shift) / Float64(max_seq_len - base_seq_len)
        var b = base_shift - m * Float64(base_seq_len)
        var mu = Float64(image_seq_len) * m + b
        return fexp(mu)

    # ── pack_latents (LensModel.py:273-276) ────────────────────────────────────
    #   batch_size, num_channels, height, width = latents.shape
    #   return latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
    @staticmethod
    def pack_latents(latents: Tensor, ctx: DeviceContext) raises -> Tensor:
        var sh = latents.shape()
        var b = sh[0]; var c = sh[1]; var h = sh[2]; var w = sh[3]
        var r = reshape(latents, _sh(b, c, h * w), ctx)
        return permute(r, _sh(0, 2, 1), ctx)            # [B, H*W, C]

    # ── unpack_latents (LensModel.py:278-281) ──────────────────────────────────
    #   batch_size, seq_len, num_channels = latents.shape
    #   return latents.reshape(batch_size, height, width, num_channels).permute(0, 3, 1, 2)
    @staticmethod
    def unpack_latents(
        latents: Tensor, height: Int, width: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var sh = latents.shape()
        var b = sh[0]; var c = sh[2]
        var r = reshape(latents, _sh(b, height, width, c), ctx)
        return permute(r, _sh(0, 3, 1, 2), ctx)         # [B, C, H, W]

    # ── patchify_latents (LensModel.py:283-289) ────────────────────────────────
    #   batch_size, num_channels_latents, height, width = latents.shape
    #   latents = latents.view(b, c, height // 2, 2, width // 2, 2)
    #   latents = latents.permute(0, 1, 3, 5, 2, 4)
    #   latents = latents.reshape(b, c * 4, height // 2, width // 2)
    #   return latents
    @staticmethod
    def patchify_latents(latents: Tensor, ctx: DeviceContext) raises -> Tensor:
        var sh = latents.shape()
        var b = sh[0]; var c = sh[1]; var h = sh[2]; var w = sh[3]
        var v = reshape(latents, _sh(b, c, h // 2, 2, w // 2, 2), ctx)
        var p = permute(v, _sh(0, 1, 3, 5, 2, 4), ctx)  # [B,C,2,2,h//2,w//2]
        return reshape(p, _sh(b, c * 4, h // 2, w // 2), ctx)

    # ── unpatchify_latents (LensModel.py:291-298) ──────────────────────────────
    #   batch_size, num_channels_latents, height, width = latents.shape
    #   latents = latents.reshape(b, c // (2 * 2), 2, 2, height, width)
    #   latents = latents.permute(0, 1, 4, 2, 5, 3)
    #   latents = latents.reshape(b, c // (2 * 2), height * 2, width * 2)
    #   return latents
    @staticmethod
    def unpatchify_latents(latents: Tensor, ctx: DeviceContext) raises -> Tensor:
        var sh = latents.shape()
        var b = sh[0]; var c = sh[1]; var h = sh[2]; var w = sh[3]
        var r = reshape(latents, _sh(b, c // 4, 2, 2, h, w), ctx)
        var p = permute(r, _sh(0, 1, 4, 2, 5, 3), ctx)  # [B,c//4,h,2,w,2]
        return reshape(p, _sh(b, c // 4, h * 2, w * 2), ctx)

    # ── scale_latents (LensModel.py:301-318) ───────────────────────────────────
    #   latents_bn_mean = self.vae.bn.running_mean.view(1,-1,1,1).to(...)
    #   latents_bn_std  = sqrt(self.vae.bn.running_var.view(1,-1,1,1)
    #                          + self.vae.config.batch_norm_eps).to(...)
    #   return (latents - latents_bn_mean) / latents_bn_std
    # Operates on PATCHIFIED+PACKED latents [B,128,h,w]. Delegated to the VAE bn
    # apply (LensVAE.scale_latents == _bn_apply[scale_mode=True], identical math).
    @staticmethod
    def scale_latents[LH: Int, LW: Int](
        latents: Tensor, vae: LensVAE[LH, LW], ctx: DeviceContext
    ) raises -> Tensor:
        return vae.scale_latents(latents, ctx)

    # ── unscale_latents (LensModel.py:320-326) ─────────────────────────────────
    #   return latents * latents_bn_std + latents_bn_mean
    @staticmethod
    def unscale_latents[LH: Int, LW: Int](
        latents: Tensor, vae: LensVAE[LH, LW], ctx: DeviceContext
    ) raises -> Tensor:
        return vae.unscale_latents(latents, ctx)

    # ── encode_text structure (LensModel.py:171-243) ───────────────────────────
    # The fresh-encode tokenizer path (chat template render, split on "<|return|>",
    # tokenizer max_length=PROMPT_MAX_LENGTH+PROMPT_TEMPLATE_CROP_START) is python
    # tokenizer-bound and lives in the dataLoader caching stage. This port carries
    # the POST-ENCODE structural transform applied to the per-layer features:
    #
    #   layer_outputs = text_encoder.encode_layers(tokens, tokens_mask)   # 4 layers
    #   if tokens.shape[1] > PROMPT_TEMPLATE_CROP_START:
    #       text_encoder_output = [feat[:, CROP:, :] for feat in layer_outputs]
    #       tokens_mask = tokens_mask[:, CROP:]
    #
    # i.e. drop the first PROMPT_TEMPLATE_CROP_START (=97) chat-template positions
    # from every selected-layer feature (and the mask). `feats` is the 4 per-layer
    # [1, S, 2880] features from LensTextEncoder.encode_layers; returns the cropped
    # 4-layer list. (Mask handling — the all-masked prune + pad-to-16 — is done by
    # the caller, which holds tokens_mask; see crop_text_mask below.)
    @staticmethod
    def crop_text_features(
        feats: List[TArc], seq_len: Int, ctx: DeviceContext
    ) raises -> List[TArc]:
        var out = List[TArc]()
        if seq_len > PROMPT_TEMPLATE_CROP_START:
            var keep = seq_len - PROMPT_TEMPLATE_CROP_START
            for i in range(len(feats)):
                # feat[:, CROP:, :]  (slice dim=1 from CROP, length keep)
                var cropped = slice(feats[i][], 1, PROMPT_TEMPLATE_CROP_START, keep, ctx)
                out.append(TArc(cropped^))
        else:
            # guard branch (LensModel.py:206-211): zero-length features. We return
            # the features unchanged (caller treats seq as 0). Kept structural.
            for i in range(len(feats)):
                out.append(feats[i])
        return out^

    # ── cached split (LensModel.py:221-224) ────────────────────────────────────
    #   hidden_dim = self.text_encoder_hidden_size
    #   text_encoder_output = list(text_encoder_output.split(hidden_dim, dim=-1))
    # The dataLoader caches the 4 layers concatenated along dim=-1 (EncodeLensText);
    # split them back into the per-layer list of [1, S, enc_hidden_dim].
    @staticmethod
    def split_cached_text(
        cached: Tensor, hidden_dim: Int, ctx: DeviceContext
    ) raises -> List[TArc]:
        var sh = cached.shape()
        var total = sh[len(sh) - 1]
        var n = total // hidden_dim
        var last = len(sh) - 1
        var out = List[TArc]()
        for i in range(n):
            var part = slice(cached, last, i * hidden_dim, hidden_dim, ctx)
            out.append(TArc(part^))
        return out^
