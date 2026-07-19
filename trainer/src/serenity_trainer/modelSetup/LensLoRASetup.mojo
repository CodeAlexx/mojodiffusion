# LensLoRASetup.mojo — Lens LoRA training spec (ModelSpec conformance). Mirrors
# modelSetup/ZImageLoRASetup.mojo (which holds the ZImageLoRASpec). This is the
# Mojo translation of Serenity's LensLoRASetup + BaseLensSetup.predict.
#
# ── PORT SPEC (1:1) ───────────────────────────────────────────────────────────
# Serenity pr-1510 modules/modelSetup/LensLoRASetup.py — the LoRA wiring:
#   create_parameters:       _create_model_part_parameters("transformer",
#                              model.transformer_lora, config.transformer)   (:43-50)
#   setup_model:             model.transformer_lora = LoRAModuleWrapper(
#                              model.transformer,"transformer",config,
#                              config.layer_filter.split(","))                (:64-66)
#                            + set_dropout, to(lora_weight_dtype), hook_to_module,
#                              create_parameters, __setup_requires_grad,
#                              init_model_parameters                          (:64-78)
#   __setup_requires_grad:   text_encoder/transformer/vae requires_grad_(False);
#                            transformer_lora trainable                       (:52-62)
#   setup_train_device:      materialize/release text encoder, vae/transformer to
#                            device, eval()/train() as configured              (:101-128)
#   after_optimizer_step:    re-assert requires_grad                          (:130-136)
# In this port these wrapper-lifecycle steps collapse into: build a LensLoraSet
# (A~kaiming/B=0 identity-at-init), run the BaseLensSetup.predict math on the
# tape/hand-chain seam, and expose backward_lora() for the driver. The frozen
# transformer/vae/text-encoder are simply never tracked.
#
# Serenity pr-1510 modules/modelSetup/BaseLensSetup.py::predict (:75-160). Body
# step-by-step and where each maps below:
#   batch_seed (:79)            → per-step seed = base_seed + step
#   text_encoder_output (:90-97)→ self.cap_feats (cached, Slice C dataLoader)
#   latent_image = patchify_latents(batch['latent_image'].float()) (:98)
#                               → patchify_latents(self.latent)
#   latent_height/width (:99-100)→ patchified dims HLp, WLp
#   scaled_latent_image = scale_latents(latent_image)  (:101)
#                               → batch-norm scale_latents(...)
#   latent_noise = _create_noise(scaled_latent_image, config, generator) (:103)
#                               → randn(scaled.shape, seed)  (offset/perturb=0.0)
#   shift = model.calculate_timestep_shift(latent_height, latent_width) (:131)
#   timestep = _get_timestep_discrete(N, det, gen, B, config,
#       shift if dynamic_timestep_shifting else config.timestep_shift)  (:132-139)
#   scaled_noisy, sigma = _add_noise_discrete(scaled, noise, timestep, ...)(:141-146)
#                               → sigma=(t+1)/1000; x_t=sigma*noise+(1-sigma)*scaled
#   packed = pack_latents(scaled_noisy)                              (:149)
#   out = transformer(packed, enc, mask, timestep/1000, img_shapes)  (:151-156)
#                               → LoRA-overlaid Lens DiT forward; t-input = t/1000
#   predicted_flow = unpack_latents(out, latent_height, latent_width) (:157)
#   flow = latent_noise - scaled_latent_image                        (:159)
#   predicted = unpatchify_latents(predicted_flow)                   (:163)
#   target    = unpatchify_latents(flow)                             (:164)
# calculate_loss (:166-178): _flow_matching_losses(...).mean() = MSE(pred,target);
#   handled by the shared train_step (tape mse_loss + flow loss weighting).
#
# NB (vs Z-Image): Lens predicted = transformer output directly (NO leading minus;
# Z-Image used predicted = -output). Lens t-input is t/1000 (Z-Image used the
# inverted (1000-t)/1000). Lens scale is batch-norm per-channel (Z-Image used a
# fixed shift/scaling). All three are ported per BaseLensSetup.py / LensModel.py.
#
# DTYPE: BF16 storage; sigma/μ host F32 scalars only. No persistent F32 tensor.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd import Tape
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    add as _add, sub as _sub, mul_scalar as _mul_scalar,
)

from serenity_trainer.util.config.TrainConfig import TrainConfig
from serenity_trainer.modelSetup.BaseModelSetup import StepOutput, ModelSpec
from serenity_trainer.modelSetup.BaseLensSetup import (
    sigma_from_timestep, model_t_from_timestep, LENS_NUM_TRAIN_TIMESTEPS,
    calculate_timestep_shift, get_timestep_discrete,
    LENS_VAE_BATCH_NORM_EPS,
    patchify_latents, unpatchify_latents, pack_latents, unpack_latents,
    scale_latents,
)

# Frozen base transformer weights (Slice C — modelLoader/LensModelLoader.mojo).
# CONTRACT: a LensWeights struct holding the full frozen DiT weights (img_in,
# txt_in, txt_norm[i], time_text_embed.*, 48 blocks, norm_out, proj_out) + the VAE
# batch-norm running_mean/running_var tensors for scale_latents.
from serenity_trainer.modelLoader.LensModelLoader import LensWeights

# LoRA overlay set + saved-tape contract + hand-chained backward (Slice B —
# model/lens/lens_backward.mojo OWNS these, so the forward/backward saved-struct
# dependency stays acyclic: A imports the set+saved from B; B imports nothing of
# A). LensLoraSet holds the 480 host LoraAdapter (48*10 block, attn-mlp preset; flat
# order = lens_lora_target_prefixes); build_lens_lora_set(rank,alpha,seed) makes
# A~kaiming / B=0. LensStackLoraGrads is the backward output (480 d_a/d_b).
from serenity_trainer.model.lens.lens_backward import (
    LensLoraSet, build_lens_lora_set, lens_backward_full_lora, LensStackLoraGrads,
)
# The hand-chained forward (Slice A — model/lens/lens_stack_lora.mojo). CONTRACT:
# lens_forward_full_lora runs the FULL Lens DiT (img_in/txt_in/txt_norm/temb → 48
# blocks → norm_out/proj_out) with the LoRA delta on every wrapped Linear and
# returns LensForwardOut (velocity + LensFullSaved activation bundle, the latter
# defined in lens_backward.mojo, that the hand-chained backward consumes).
from serenity_trainer.model.lens.lens_stack_lora import (
    lens_forward_full_lora, LensForwardOut,
)


# ── the conforming spec ───────────────────────────────────────────────────────
# Comptime-shaped on the PATCHIFIED latent dims (HLp=H//2, WLp=W//2) + caption
# length: the Lens sequence length N_IMG = HLp*WLp must be a compile-time constant
# for the SDPA forward. `latent` is the RAW cached VAE latent [1,32,H,W] (pre
# patchify/scale).
struct LensLoRASpec[HLp: Int, WLp: Int, CAPLEN: Int](Movable, ModelSpec):
    var weights: LensWeights           # frozen base DiT weights (untracked, name→tensor map)
    var loras: LensLoraSet             # trained overlay (480 adapters)
    var latent: Tensor                 # raw VAE latent [1,32,H,W] BF16 (pre-scale)
    var cap_feats: Tensor              # [CAPLEN, 11520] concat text features BF16
    # VAE batch-norm latent-scale stats (LensModel.scale_latents :285-290). These
    # live on the VAE (AutoencoderKLFlux2.bn), NOT in the transformer LensWeights
    # (which only holds DiT tensors), so the spec carries them directly:
    #   vae_bn_mean = vae.bn.running_mean   [128]
    #   vae_bn_var  = vae.bn.running_var    [128]
    var vae_bn_mean: Tensor            # [128] per-channel running_mean
    var vae_bn_var: Tensor             # [128] per-channel running_var
    var timestep_shift: Float32        # legacy carrier; predict() derives the shift
                                       # per Serenity (calculate_timestep_shift
                                       # when config.dynamic_timestep_shifting, else
                                       # config.timestep_shift). Kept for ctor API.
    var base_seed: UInt64              # per-step noise/timestep seed base
    # HAND-CHAIN SEAM: predict() runs the full Lens DiT forward (hand-chained, NOT
    # on the autograd tape) and stashes its saved-for-backward here so the driver
    # can call backward_lora(d_velocity) AFTER it knows d_predicted. Stores the
    # WHOLE forward output (move-only struct → no piecemeal field transfer-out).
    var fwd_out: Optional[LensForwardOut]

    def __init__(
        out self, var weights: LensWeights, var loras: LensLoraSet,
        var latent: Tensor, var cap_feats: Tensor,
        var vae_bn_mean: Tensor, var vae_bn_var: Tensor,
        timestep_shift: Float32, base_seed: UInt64,
    ):
        self.weights = weights^
        self.loras = loras^
        self.latent = latent^
        self.cap_feats = cap_feats^
        self.vae_bn_mean = vae_bn_mean^
        self.vae_bn_var = vae_bn_var^
        self.timestep_shift = timestep_shift
        self.base_seed = base_seed
        self.fwd_out = None

    # ModelSpec.predict — builds the noised latent + flow target 1:1 with
    # BaseLensSetup.predict and runs the LoRA-overlaid forward (hand-chained).
    def predict(
        mut self, mut tape: Tape, config: TrainConfig, step: Int, ctx: DeviceContext
    ) raises -> StepOutput:
        var seed = self.base_seed + UInt64(step)

        # 1) patchify the raw latent (BaseLensSetup.py:98) → [1,128,HLp,WLp].
        var patched = patchify_latents(self.latent, ctx)

        # 2) batch-norm scale (BaseLensSetup.py:101 / LensModel.scale_latents).
        var scaled_latent = scale_latents(
            patched, self.vae_bn_mean, self.vae_bn_var,
            LENS_VAE_BATCH_NORM_EPS, ctx,
        )

        # 3) per-step noise (_create_noise; offset/perturb default 0.0 → plain randn)
        var noise = randn(scaled_latent.shape().copy(), seed, STDtype.BF16, ctx)

        # 4) schedule shift (BaseLensSetup.py:131-139). latent_height/width are the
        #    PATCHIFIED dims HLp, WLp.
        var shift: Float32
        if config.dynamic_timestep_shifting:
            shift = calculate_timestep_shift(Self.HLp, Self.WLp)
        else:
            shift = config.timestep_shift

        # 5) sample discrete timestep (_get_timestep_discrete). RNG NOTE: Serenity
        #    shares ONE torch.Generator(batch_seed) across noise + timestep; we key
        #    the host timestep sampler with the SAME `seed` (documented RNG-stream
        #    divergence; parity verified on dumped tensors — see BaseLensSetup).
        var t = get_timestep_discrete(
            LENS_NUM_TRAIN_TIMESTEPS,
            False,                         # deterministic (predict default)
            seed,
            config.timestep_distribution,
            config.min_noising_strength,
            config.max_noising_strength,
            config.noising_weight,
            config.noising_bias,
            shift,
        )
        var sigma = sigma_from_timestep(t, LENS_NUM_TRAIN_TIMESTEPS)
        var t_model = model_t_from_timestep(t, LENS_NUM_TRAIN_TIMESTEPS)  # t/1000

        # 6) noised input + flow target (_add_noise_discrete + flow), patchified scale.
        #    x_t  = sigma*noise + (1-sigma)*scaled_latent   (NoiseMixin.py:36-37)
        #    flow = noise - scaled_latent                   (BaseLensSetup.py:159)
        var a = _mul_scalar(scaled_latent, Float32(1.0) - sigma, ctx)
        var b = _mul_scalar(noise, sigma, ctx)
        var scaled_noisy = _add(a, b, ctx)                 # [1,128,HLp,WLp]
        var flow = _sub(noise, scaled_latent, ctx)         # [1,128,HLp,WLp]

        # 7) pack to the transformer sequence (BaseLensSetup.py:149) → [1,N_IMG,128].
        var packed = pack_latents(scaled_noisy, ctx)

        # 8) LoRA-overlaid Lens DiT forward → packed model output [1,N_IMG,128].
        var model_out_packed = self._forward_lora(packed, t_model, ctx)

        # 9) unpack (BaseLensSetup.py:157) → [1,128,HLp,WLp].
        var predicted_flow = unpack_latents(model_out_packed, Self.HLp, Self.WLp, ctx)

        # 10) unpatchify both predicted_flow and flow (BaseLensSetup.py:163-164)
        #     → [1,32,H,W] each.
        var predicted = unpatchify_latents(predicted_flow, ctx)
        var target = unpatchify_latents(flow, ctx)

        return StepOutput(predicted^, target^, sigma)

    # The LoRA-overlaid FULL Lens DiT forward (hand-chained; NOT recorded on
    # `tape`). The LoRA grads are produced by backward_lora() (read directly by the
    # driver), NOT by autograd.backward. We accept `tape` to keep the predict()
    # seam signature but record nothing on it. Returns the packed velocity
    # [1,N_IMG,128].
    def _forward_lora(
        mut self, packed: Tensor, t_model: Float32, ctx: DeviceContext
    ) raises -> Tensor:
        var fo = lens_forward_full_lora[Self.HLp, Self.WLp, Self.CAPLEN](
            packed, t_model, self.cap_feats, self.weights, self.loras, ctx
        )
        # Clone the velocity (borrow → owned copy; * 1.0) BEFORE moving `fo` whole
        # into the spec (Movable struct → no partial field transfer-out).
        var velocity = _mul_scalar(fo.velocity, Float32(1.0), ctx)
        self.fwd_out = Optional[LensForwardOut](fo^)
        return velocity^

    # HAND-CHAIN BACKWARD seam the DRIVER calls after computing d_velocity (the grad
    # of the loss wrt the model's packed velocity output). Since Lens predicted ==
    # velocity directly (no minus), d_velocity == d_predicted (after unpack/
    # unpatchify pull-back, handled by the driver). Returns the 480 LoRA d_a/d_b.
    def backward_lora(
        mut self, d_velocity: Tensor, ctx: DeviceContext
    ) raises -> LensStackLoraGrads:
        if not self.fwd_out:
            raise Error(
                "LensLoRASpec.backward_lora: no saved forward state — call "
                "predict()/_forward_lora first."
            )
        return lens_backward_full_lora[Self.HLp, Self.WLp, Self.CAPLEN](
            d_velocity, self.fwd_out.value().saved, self.loras, ctx
        )


# Convenience: build a cold-start spec (A~kaiming, B=0) for a fresh LoRA run,
# mirroring LensLoRASetup.setup_model (transformer_lora = LoRAModuleWrapper(...)).
def make_lens_lora_spec[HLp: Int, WLp: Int, CAPLEN: Int](
    var weights: LensWeights, var latent: Tensor, var cap_feats: Tensor,
    var vae_bn_mean: Tensor, var vae_bn_var: Tensor,
    rank: Int, alpha: Float32, timestep_shift: Float32, base_seed: UInt64,
    ctx: DeviceContext,
) raises -> LensLoRASpec[HLp, WLp, CAPLEN]:
    var loras = build_lens_lora_set(rank, alpha, base_seed, ctx)
    return LensLoRASpec[HLp, WLp, CAPLEN](
        weights^, loras^, latent^, cap_feats^, vae_bn_mean^, vae_bn_var^,
        timestep_shift, base_seed
    )
