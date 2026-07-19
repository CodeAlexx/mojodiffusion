# Flux2LoRASetup.mojo — the Klein (FLUX.2) LoRA ModelSpec: the Serenity
# BaseFlux2Setup.predict math (the modelSetup slice) wired to the BORROWED Klein
# DiT forward (model/KleinModel.mojo).
#
# ── PORT SPEC (1:1) ───────────────────────────────────────────────────────────
#   * modules/modelSetup/Flux2LoRASetup.py (the LoRA wiring: setup_model :52-71,
#       create_parameters :31-39, __setup_requires_grad :41-50).
#   * modules/modelSetup/BaseFlux2Setup.py::predict (:82-179) — the latent/noise/
#       sigma/timestep/target construction this spec OWNS, step by step:
#       latent_image      = patchify_latents(latent_image.float())   (:107)
#       scaled_latent     = scale_latents(latent_image)              (:110)
#       latent_noise      = _create_noise(scaled_latent, generator)  (:112)
#       shift = calculate_timestep_shift(H, W) if dynamic else timestep_shift (:114,121)
#       timestep          = _get_timestep_discrete(N, det, gen, B, config, shift)(:115)
#       scaled_noisy, σ   = _add_noise_discrete(scaled, noise, t, timesteps) (:124)
#       latent_input      = scaled_noisy                              (:130)
#       guidance          = [guidance_scale].expand(B) if guidance_embeds else None (:132-136)
#       text_ids/image_ids/packed = prepare_*                         (:138-140)
#       predicted_flow    = transformer(packed, timestep/1000, guidance, txt, ...) (:142-157)
#       flow              = latent_noise - scaled_latent              (:159)
#       model_output_data = {predicted: unpatchify(predicted_flow),
#                            target:    unpatchify(flow),  timestep}  (:160-166)
#   * calculate_loss (:181-194): _flow_matching_losses(...).mean(), handled by the
#       shared train_step (MSE + LossWeight.SIGMA weighting on `σ`).
#
# ── THE FORWARD SEAM (now WIRED — no longer a placeholder) ────────────────────
# predict() OWNS the latent/noise/sigma/target arithmetic; the DiT forward is the
# BORROWED KleinModel (model/KleinModel.mojo). KleinModel.klein_training_forward
# builds the hand-chained backward tape; KleinModel.klein_backward returns the
# per-adapter d_A/d_B (KleinLoraGrads) the driver scatters into the optimizer. This
# spec stores the KleinStackForward (via fwd_saved) so backward_lora(d_flow) can
# chain it AFTER the loss is known — the same hand-chain seam ZImageLoRASetup uses.
# The LoRA grads do NOT flow through autograd.backward; the driver reads them off
# backward_lora().
#
# _scaled_latent now performs the REAL patchify_latents + scale_latents by REUSING
# the verified VAE-unit kernels (KleinVAE._patchify_packed + _bn_apply[True], the
# 1:1 port of Flux2Model.patchify_latents :297-302 + scale_latents :313-318).
# _forward_lora now performs pack_latents → klein_training_forward (tape) →
# unpack_latents → unpatchify_latents, feeding modvecs built from the INTEGER
# timestep t (see the timestep-trap note below).
#
# FUSED-qkv divergence: the borrowed Klein forward carries adapters at the ORIGINAL
# fused-qkv granularity (one rank-r adapter per qkv, B:[3D,rank]); Serenity wraps
# the diffusers Flux2Transformer2DModel where attn.to_q/to_k/to_v are THREE
# independent LoRAModules (effective rank 3r). The on-disk PEFT shapes match after
# the saver's 1→3 split (shared A, row-split B) and the loader's 3→1 merge, but the
# trained subspace is rank-constrained to a shared down-projection. This is the
# central borrow-vs-faithfulness gap, documented in the saver / loader / targets
# files and returned to the orchestrator as an unresolved item (reference trainer source:
# Flux2Model.py:53 qkv_fusion; LoRAModule.py wrapping per-Linear).
#
# transformer t-input = timestep/1000 (BaseFlux2Setup.py:144 — NOT inverted).
# CRITICAL TIMESTEP TRAP (weights.mojo:351-369): diffusers Flux2 multiplies the
# timestep/1000 BACK by 1000 internally (transformer_flux2.py:1231), so the
# t_embedder sees the INTEGER timestep t. We therefore build the Klein modulation
# vectors from the integer `t` (the int returned by _get_timestep_discrete), NOT
# from t_model = t/1000 and NOT from sigma*1000 (= t+1, off by one).
#
# guidance (BaseFlux2Setup.py:132-136): gated on the LOADED CHECKPOINT'S
# transformer.config.guidance_embeds (:132, a runtime value). The flagship
# FLUX.2-klein-base-9B is guidance_embeds=FALSE (verified vs transformer/config.json)
# ⇒ guidance=None and the branch is a no-op. Only guidance-distilled variants
# (guidance_embeds=True) set guidance = tensor([config.transformer.guidance_scale])
# expanded to B; the transformer then multiplies guidance ×1000 internally
# (transformer_flux2.py:1234) before the guidance_embedder, and the combined
# vec = silu(t_emb + guidance_emb) feeds every modulation
# (transformer_flux2.py:1004-1014,1236-1240). The spec carries guidance_scale and
# threaded guidance_embeds; passes guidance_embedder_value(guidance_scale,
# guidance_embeds) = guidance_scale*1000 (or None) into the modvec builder.
#
# DTYPE: BF16 storage; σ/μ host F32 scalars only. No persistent F32 tensor.

from std.collections import Optional
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd import Tape
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    add as _add, sub as _sub, mul_scalar as _mul_scalar,
    reshape as _reshape, reshape_owned as _reshape_owned, permute as _permute,
)

from serenity_trainer.util.config.TrainConfig import TrainConfig
from serenity_trainer.modelSetup.BaseModelSetup import StepOutput, ModelSpec
from serenity_trainer.modelSetup.BaseFlux2Setup import (
    sigma_from_timestep, FLUX2_NUM_TRAIN_TIMESTEPS,
    calculate_timestep_shift, get_timestep_discrete,
    guidance_embedder_value, FLUX2_GUIDANCE_EMBEDS,
)

# REUSED VAE-unit kernels: the verified 1:1 patchify_latents + scale_latents
# (BatchNorm) ports. _bn_apply[True] = (z - mean) * inv_scale where inv_scale =
# 1/sqrt(var + batch_norm_eps); _patchify_packed = Flux2Model.patchify_latents;
# _unpatchify_packed = Flux2Model.unpatchify_latents.
from serenity_trainer.model.KleinVAE import (
    _patchify_packed, _unpatchify_packed, _bn_apply,
)

# The borrowed Klein forward + the model-unit state it consumes.
from serenity_trainer.model.klein.double_block import DoubleBlockWeights
from serenity_trainer.model.klein.single_block import SingleBlockWeights
from serenity_trainer.model.klein.klein_stack import KleinStackBase, KleinStackForward
from serenity_trainer.model.klein.klein_stack_lora import (
    KleinLoraDeviceSet, KleinLoraGrads,
)
from serenity_trainer.model.klein.weights import (
    KleinStepModWeights, build_klein_step_mods_device_cached,
)
from serenity_trainer.model.KleinModel import (
    klein_training_forward, klein_backward,
    KDIM, KIN_CH, KOUT_CH, KTIMESTEP_DIM,
)


comptime TArc = ArcPointer[Tensor]


# ══════════════════════════════════════════════════════════════════════════════
# Flux2LoRASpec — the ModelSpec conformance for Klein LoRA training.
#
# Holds the FROZEN Klein base (KleinStackBase + per-block DoubleBlockWeights/
# SingleBlockWeights + the timestep/modulation weights), the trained LoRA overlay
# (KleinLoraDeviceSet), the precomputed RoPE tables, the VAE batch-norm stats
# (bn_inv_scale = 1/sqrt(var+eps), bn_mean — both [128] F32), the concatenated Qwen3
# text features for this step, and the clean (raw mean) latent + seed. predict()
# owns the BaseFlux2Setup.predict math and runs the LoRA-overlaid Klein forward;
# backward_lora() chains the hand-built tape.
#
# Comptime-shaped on the PATCHIFIED latent HL/WL (the DiT sequence length must be a
# compile-time constant for the borrowed sdpa). N_IMG = HL*WL is the packed image
# token count; NTXT is the Qwen3 concat seq len; S = NTXT + N_IMG.
#
# `latent` is the RAW VAE mean latent [1, 32, HL*2, WL*2] BF16 (latent caching
# stores the pre-patchify, pre-scale mean; BaseFlux2Setup.predict:107 patchifies it
# to [1,128,HL,WL] then scales). KIN_CH = 128 (the patchified channel count).
struct Flux2LoRASpec[HL: Int, WL: Int, NTXT: Int](Movable):
    # ── frozen Klein base (untracked) ──
    var base: KleinStackBase
    var dbw: List[DoubleBlockWeights]
    var sbw: List[SingleBlockWeights]
    var step_mod_w: KleinStepModWeights      # frozen time/mod weights (Movable)
    # ── trained LoRA overlay (A/B per adapter; hand-chained, NOT on the tape) ──
    var lora: KleinLoraDeviceSet
    # ── precomputed RoPE tables (cos/sin), shaped [S*H, Dh/2] ──
    var cos_t: Tensor
    var sin_t: Tensor
    # ── VAE batch-norm stats for scale_latents (per PATCHIFIED channel, 128 F32) ──
    #    bn_inv_scale = 1/sqrt(running_var + batch_norm_eps); bn_mean = running_mean.
    var bn_inv_scale: Tensor                 # [128] F32
    var bn_mean: Tensor                      # [128] F32
    # ── the concatenated Qwen3 text features [NTXT, KTXT_CH] BF16 ──
    var txt_tokens: Tensor
    # ── per-step inputs ──
    var latent: Tensor                       # raw VAE mean latent [1,32,HL*2,WL*2] BF16 (pre-patchify/scale)
    var base_seed: UInt64
    # ── guidance: config.transformer.guidance_scale (BaseFlux2Setup.py:133), used
    #    only when the loaded checkpoint's config.guidance_embeds is True (:132).
    #    guidance_embeds is the runtime checkpoint value; the flagship
    #    FLUX.2-klein-base-9B is guidance_embeds=FALSE so the guidance branch is a
    #    no-op there. Only guidance-distilled variants take it. ──
    var guidance_scale: Float32
    var guidance_embeds: Bool
    # ── HAND-CHAIN SEAM: the saved-for-backward tape + the host token lists the
    #    backward needs. Stashed by _forward_lora so backward_lora() can chain it
    #    after the loss is known. ──
    var fwd_saved: Optional[KleinStackForward]
    var fwd_img_host: List[Float32]          # packed img tokens (host) for backward
    var fwd_txt_host: List[Float32]          # txt tokens (host) for backward
    var fwd_t_int: Float32                   # the integer timestep the forward used

    def __init__(
        out self,
        var base: KleinStackBase,
        var dbw: List[DoubleBlockWeights], var sbw: List[SingleBlockWeights],
        var step_mod_w: KleinStepModWeights,
        var lora: KleinLoraDeviceSet,
        var cos_t: Tensor, var sin_t: Tensor,
        var bn_inv_scale: Tensor, var bn_mean: Tensor,
        var txt_tokens: Tensor,
        var latent: Tensor, base_seed: UInt64,
        guidance_scale: Float32 = Float32(1.0),  # Serenity training default (TrainConfig.py:289); factory at make_flux2_lora_spec always passes this through
        guidance_embeds: Bool = FLUX2_GUIDANCE_EMBEDS,
    ):
        self.base = base^
        self.dbw = dbw^
        self.sbw = sbw^
        self.step_mod_w = step_mod_w^
        self.lora = lora^
        self.cos_t = cos_t^
        self.sin_t = sin_t^
        self.bn_inv_scale = bn_inv_scale^
        self.bn_mean = bn_mean^
        self.txt_tokens = txt_tokens^
        self.latent = latent^
        self.base_seed = base_seed
        self.guidance_scale = guidance_scale
        self.guidance_embeds = guidance_embeds
        self.fwd_saved = None
        self.fwd_img_host = List[Float32]()
        self.fwd_txt_host = List[Float32]()
        self.fwd_t_int = Float32(0.0)

    # ModelSpec.predict — builds the noised latent + flow target 1:1 with
    # BaseFlux2Setup.predict, and runs the LoRA-overlaid Klein forward.
    def predict(
        mut self, mut tape: Tape, config: TrainConfig, step: Int, ctx: DeviceContext
    ) raises -> StepOutput:
        var seed = self.base_seed + UInt64(step)

        # 1) patchify + scale clean latent (BaseFlux2Setup.py:107,110).
        #    scaled_latent is the PATCHIFIED, batch-norm-scaled latent [1,128,HL,WL].
        var scaled_latent = self._scaled_latent(ctx)

        # 2) per-step noise (_create_noise, NoiseMixin.py:85-90). offset_noise_weight
        #    / perturbation_noise_weight default 0.0 → plain randn.
        var noise = randn(scaled_latent.shape().copy(), seed, STDtype.BF16, ctx)

        # 3) schedule shift (BaseFlux2Setup.py:114,121):
        #    shift = calculate_timestep_shift(H, W) if dynamic_timestep_shifting
        #            else config.timestep_shift.
        #    H = scaled_latent.shape[-2], W = scaled_latent.shape[-1] (PATCHIFIED).
        var lshape = scaled_latent.shape()
        var H = lshape[len(lshape) - 2]
        var W = lshape[len(lshape) - 1]
        var shift: Float32
        if config.dynamic_timestep_shifting:
            shift = calculate_timestep_shift(H, W)
        else:
            shift = config.timestep_shift

        # 4) sample discrete timestep (_get_timestep_discrete, shift μ) (:115-122).
        #    RNG NOTE: Serenity shares ONE torch.Generator(batch_seed) for noise +
        #    timestep (BaseFlux2Setup.py:92-95,112,115); serenitymojo has no shared
        #    host/device torch stream, so values will not bit-match torch (documented
        #    divergence; parity verified against dumped tensors). Same base `seed`.
        var t = get_timestep_discrete(
            FLUX2_NUM_TRAIN_TIMESTEPS,    # model.noise_scheduler.config['num_train_timesteps']
            False,                        # deterministic (predict default)
            seed,
            config.timestep_distribution,
            config.min_noising_strength,
            config.max_noising_strength,
            config.noising_weight,
            config.noising_bias,
            shift,
        )
        var sigma = sigma_from_timestep(t, FLUX2_NUM_TRAIN_TIMESTEPS)

        # 5) noised input + flow target (_add_noise_discrete + flow) (:124,159).
        #    sigmas = (t+1)/N (FlowMatchingMixin.py:24,29)
        #    x_t  = noise*σ + scaled_latent*(1-σ)  (FlowMatchingMixin.py:36-37)
        #    flow = noise - scaled_latent          (BaseFlux2Setup.py:159)
        var a = _mul_scalar(scaled_latent, Float32(1.0) - sigma, ctx)
        var b = _mul_scalar(noise, sigma, ctx)
        var scaled_noisy = _add(a, b, ctx)
        var target_patch = _sub(noise, scaled_latent, ctx)   # flow target (loss_type 'target')
        var target = _unpatchify_packed(target_patch, ctx)

        # 6) LoRA-overlaid Klein forward → predicted flow (the borrowed forward).
        #    transformer t-input = timestep/1000 (BaseFlux2Setup.py:144), but the
        #    t_embedder sees the INTEGER timestep t (diffusers re-scales ×1000;
        #    weights.mojo:351-369). Flux2 does NOT negate the model output (UNLIKE
        #    Z-Image): predicted is the raw flow prediction (no leading minus).
        var predicted = self._forward_lora(tape, scaled_noisy, Float32(t), ctx)

        return StepOutput(predicted^, target^, sigma)

    # ── _scaled_latent: patchify_latents + scale_latents (Flux2Model.py:107,110) ──
    # Reuses the verified VAE-unit kernels: _patchify_packed ([1,32,HL*2,WL*2] →
    # [1,128,HL,WL], Flux2Model.patchify_latents) then _bn_apply[True]
    # ((z - mean)*inv_scale, Flux2Model.scale_latents). bn stats carried by the spec.
    def _scaled_latent(self, ctx: DeviceContext) raises -> Tensor:
        var patchified = _patchify_packed(self.latent, ctx)        # [1,128,HL,WL]
        return _bn_apply[True](patchified, self.bn_inv_scale, self.bn_mean, ctx)

    # ── _forward_lora: pack_latents → klein_training_forward → unpack/unpatchify ──
    # HAND-CHAINED: the borrowed Klein forward saves a KleinStackForward tape; the
    # LoRA grads come back through backward_lora() (read by the driver), NOT through
    # autograd.backward. We accept `tape` to keep the predict() seam signature but
    # record nothing on it. Returns the predicted flow [1, 32, HL*2, WL*2]
    # (unpatchified to match the mask shape, BaseFlux2Setup.py:164).
    def _forward_lora(
        mut self, mut tape: Tape, scaled_noisy: Tensor, t_int: Float32, ctx: DeviceContext
    ) raises -> Tensor:
        comptime N_IMG = Self.HL * Self.WL
        comptime N_TXT = Self.NTXT
        comptime S = N_TXT + N_IMG

        # pack_latents (Flux2Model.py:255-257): [1,128,HL,WL] →
        #   reshape [1,128,HL*WL] → permute [0,2,1] → [1,N_IMG,128].
        #   The borrowed forward consumes the 2D [N_IMG, 128] view.
        var packed3 = _reshape(scaled_noisy, [1, KIN_CH, N_IMG], ctx)
        var packed_p = _permute(packed3, [0, 2, 1], ctx)      # [1, N_IMG, 128]
        var img_tokens = _reshape_owned(packed_p^, [N_IMG, KIN_CH])

        # modulation vectors from the INTEGER timestep t (the t_embedder input;
        # weights.mojo — NOT t/1000, NOT sigma*1000) and the INTEGER guidance value
        # (only when self.guidance_embeds; None for klein-base-9B which is
        # guidance_embeds=False): vec = silu(t_emb + guidance_emb)
        # (BaseFlux2Setup.py:132-136 + transformer_flux2.py:1004-1014,1234). The
        # builder also returns the per-step final-layer shift/scale (elems 3,4); the
        # borrowed forward uses base.final_shift/scale, so we ignore them here.
        var guidance_val = guidance_embedder_value(self.guidance_scale, self.guidance_embeds)
        var mods = build_klein_step_mods_device_cached(
            self.step_mod_w, t_int, guidance_val, KTIMESTEP_DIM, KDIM, ctx
        )
        var img_mod_dev = mods[0].copy()
        var txt_mod_dev = mods[1].copy()
        var single_mod_dev = mods[2].copy()

        # stash host copies of the inputs for the hand-chained backward.
        self.fwd_img_host = img_tokens.to_host(ctx)
        self.fwd_txt_host = self.txt_tokens.to_host(ctx)
        self.fwd_t_int = t_int

        var img_arc = TArc(img_tokens^)
        var txt_arc = TArc(_mul_scalar(self.txt_tokens, Float32(1.0), ctx))

        var fwd = klein_training_forward[N_IMG, N_TXT, S](
            img_arc, txt_arc, self.base, self.dbw, self.sbw, self.lora,
            img_mod_dev, txt_mod_dev, single_mod_dev, self.cos_t, self.sin_t, ctx,
        )

        # the predicted flow tokens [N_IMG, KOUT_CH] are host F32 (KleinStackForward
        # .out is List[Float32]); upload to a BF16 device tensor before unpack.
        var flow_host = fwd.out.copy()
        self.fwd_saved = Optional[KleinStackForward](fwd^)
        var flow_tokens = Tensor.from_host(flow_host^, [1, N_IMG, KOUT_CH], STDtype.BF16, ctx)

        # unpack_latents (Flux2Model.py:260-262): [1,N_IMG,128] →
        #   reshape [1,HL,WL,128] → permute [0,3,1,2] → [1,128,HL,WL].
        var flow_b = _reshape(flow_tokens, [1, Self.HL, Self.WL, KOUT_CH], ctx)
        var flow_perm = _permute(flow_b, [0, 3, 1, 2], ctx)   # [1,128,HL,WL]
        var predicted_flow_patch = _reshape_owned(flow_perm^, [1, KOUT_CH, Self.HL, Self.WL])

        # unpatchify_latents (Flux2Model.py:305-310) via the verified VAE kernel:
        #   [1,128,HL,WL] → [1,32,HL*2,WL*2].
        return _unpatchify_packed(predicted_flow_patch, ctx)

    # ── HAND-CHAIN BACKWARD seam the DRIVER calls after computing d_flow ──────────
    # d_flow: dL/d(predicted flow) in the UNPATCHIFIED layout [1,32,HL*2,WL*2] (the
    # same layout _forward_lora returned). We re-patchify + re-pack it to the
    # borrowed forward's token layout [N_IMG, 128], then chain klein_backward.
    # Returns the per-adapter d_A/d_B (KleinLoraGrads). FROZEN base. Raises if called
    # before _forward_lora.
    def backward_lora(
        mut self, d_flow: Tensor, ctx: DeviceContext
    ) raises -> KleinLoraGrads:
        if not self.fwd_saved:
            raise Error(
                "Flux2LoRASpec.backward_lora: no saved forward state — call "
                "predict()/_forward_lora first."
            )
        comptime N_IMG = Self.HL * Self.WL
        comptime N_TXT = Self.NTXT
        comptime S = N_TXT + N_IMG

        # re-patchify d_flow [1,32,HL*2,WL*2] → [1,128,HL,WL] (inverse of the
        # unpatchify the forward applied to the prediction).
        var d_patch = _patchify_packed(d_flow, ctx)           # [1,128,HL,WL]
        # re-pack → [N_IMG, 128] (inverse of unpack).
        var dpk3 = _reshape(d_patch, [1, KOUT_CH, N_IMG], ctx)
        var dpk_perm = _permute(dpk3, [0, 2, 1], ctx)         # [1,N_IMG,128]
        var d_out_t = _reshape_owned(dpk_perm^, [N_IMG, KOUT_CH])
        var d_out_host = d_out_t.to_host(ctx)

        # rebuild modvecs from the cached integer timestep (same t the forward used)
        # AND the same guidance value (modulation must be byte-identical to forward).
        var guidance_val = guidance_embedder_value(self.guidance_scale, self.guidance_embeds)
        var mods = build_klein_step_mods_device_cached(
            self.step_mod_w, self.fwd_t_int, guidance_val, KTIMESTEP_DIM, KDIM, ctx
        )
        var img_mod_dev = mods[0].copy()
        var txt_mod_dev = mods[1].copy()
        var single_mod_dev = mods[2].copy()

        return klein_backward[N_IMG, N_TXT, S](
            d_out_host, self.fwd_img_host, self.fwd_txt_host,
            self.base, self.dbw, self.sbw, self.lora,
            img_mod_dev, txt_mod_dev, single_mod_dev, self.cos_t, self.sin_t,
            self.fwd_saved.value(), ctx,
        )


# Convenience: build a Klein LoRA spec from a fully-assembled model unit + the
# per-step latent/seed/text features. Mirrors Flux2LoRASetup.setup_model (the LoRA
# set A~kaiming / B=0 is built by build_klein9b_lora_set in the model unit and moved
# to device via klein_lora_set_to_device). HL/WL are the PATCHIFIED latent dims;
# NTXT is the Qwen3 concat sequence length.
def make_flux2_lora_spec[HL: Int, WL: Int, NTXT: Int](
    var base: KleinStackBase,
    var dbw: List[DoubleBlockWeights], var sbw: List[SingleBlockWeights],
    var step_mod_w: KleinStepModWeights,
    var lora: KleinLoraDeviceSet,
    var cos_t: Tensor, var sin_t: Tensor,
    var bn_inv_scale: Tensor, var bn_mean: Tensor,
    var txt_tokens: Tensor,
    var latent: Tensor, base_seed: UInt64,
    guidance_scale: Float32 = Float32(1.0),       # config.transformer.guidance_scale (Serenity training default, TrainConfig.py:289; #flux2 presets do not override)
    guidance_embeds: Bool = FLUX2_GUIDANCE_EMBEDS,  # runtime checkpoint value; default False (klein-base-9B)
) -> Flux2LoRASpec[HL, WL, NTXT]:
    # Source of truth for guidance is the LOADED CHECKPOINT, exactly like
    # Serenity's model.transformer.config.guidance_embeds (BaseFlux2Setup.py:132)
    # and diffusers' `guidance is not None and guidance_embedder is not None`
    # (transformer_flux2.py:1008). step_mod_w.has_guidance() is True iff the
    # guidance_in.* keys were present in the checkpoint; this is the structural
    # equivalent. We AND it with the threaded flag so a guidance-distilled
    # checkpoint activates guidance and klein-base-9B (no guidance_in.*) never does,
    # regardless of the caller-supplied default.
    var effective_guidance_embeds = guidance_embeds and step_mod_w.has_guidance()
    return Flux2LoRASpec[HL, WL, NTXT](
        base^, dbw^, sbw^, step_mod_w^, lora^, cos_t^, sin_t^,
        bn_inv_scale^, bn_mean^, txt_tokens^, latent^, base_seed,
        guidance_scale, effective_guidance_embeds,
    )
