# Flux2Sampler.mojo — Klein (FLUX.2) sampler primitives for the INFERENCE denoise
# path (the Z-Image lesson: a SEPARATE no-grad forward, never the training fwd) +
# the FLUX.2 mu-based FlowMatchEulerDiscreteScheduler.
#
# STATUS (guidance_wiring): the TRAINING guidance path is wired end-to-end
# (predict→build_klein_vec_silu_guidance_device→forward, backward mirrors). The
# SAMPLER here is HELPER-LEVEL ONLY (same maturity as ZImageSampler): it exposes the
# scheduler state (make_flux2_denoise_state), the per-step guidance/timestep values
# (flux2_guidance_value/flux2_step_t_embedder — the INTEGER modvec inputs the
# guidance_embedder + t_embedder see), CFG combine, and one Euler step. It does NOT
# yet assemble these into a denoise loop that calls model/KleinModel.
# klein_inference_forward injecting the guidance modvecs — that driver loop is
# PENDING. So guidance is NOT executing on the sampler path; the helpers exist but
# nothing drives them. The PORT SPEC block below is the Serenity reference loop to
# implement that driver against.
#
# ── PORT SPEC (1:1) ───────────────────────────────────────────────────────────
#   * modules/modelSampler/Flux2Sampler.py::__sample_base (:39-161), under
#       @torch.no_grad() (:39). The denoise loop (:115-137):
#         latent_model_input = cat([latent]*batch_size)              (:116)
#         expanded_timestep  = timestep.expand(B)                    (:117)
#         noise_pred = transformer(latent_input.to(train_dtype),
#                                  timestep=expanded_timestep/1000,  (:120-129)
#                                  guidance, encoder_hidden_states, txt_ids, img_ids)
#         if batch_size==2: pos,neg = chunk(2);
#             noise_pred = neg + cfg*(pos - neg)                     (:131-133)
#         latent = scheduler.step(noise_pred, t, latent)[0]          (:135)
#       batch_size = 2 if cfg>1 and NOT guidance_embeds else 1       (:73)
#       NB: NO leading minus on noise_pred (UNLIKE Z-Image). The transformer output
#       IS the predicted flow; the Euler step adds dt*noise_pred directly.
#   * guidance (Flux2Sampler.py:113-114): guidance = tensor([cfg_scale]) iff the
#       LOADED CHECKPOINT'S transformer.config.guidance_embeds is True (a runtime
#       value, NOT a constant). The flagship FLUX.2-klein-base-9B checkpoint has
#       guidance_embeds=FALSE (verified vs transformer/config.json) ⇒ guidance=None
#       and the cfg>1 pos/neg CFG batch path (flux2_use_cfg/flux2_batch_size,
#       batch_size=2) is used. Only guidance-DISTILLED variants (guidance_embeds=True,
#       guidance_in.* keys present) take batch_size=1 with guidance injected into the
#       t_embedder vec (silu(t_emb + guidance_emb), transformer_flux2.py:1004-1014,1234).
#       flux2_guidance_value() returns the INTEGER guidance value (cfg_scale*1000) for
#       build_klein_step_mods_device_cached; flux2_step_t_embedder() returns the INTEGER
#       timestep (scheduler.timesteps[i], NOT /1000). `guidance_embeds` is threaded from
#       the checkpoint, defaulting to FLUX2_GUIDANCE_EMBEDS (False = klein-base-9B).
#   * scheduler prep (:91-102):
#       latent = patchify_latents(randn(1,32,H/8,W/8, f32))          (:84-91)
#       image_ids = prepare_latent_image_ids(latent)                (:92)
#       latent = pack_latents(latent)                               (:94)
#       image_seq_len = latent.shape[1]                             (:95)
#       mu = compute_empirical_mu(image_seq_len, diffusion_steps)   (:96)
#       sigmas = linspace(1.0, 1/n, n)                              (:100)
#       scheduler.set_timesteps(n, mu=mu, sigmas=sigmas)            (:101)
#   * decode (:143-151):
#       latent = unpack_latents(latent, H/8/2, W/8/2)               (:143)
#       latents = unscale_latents(latent)                          (:148)
#       latents = unpatchify_latents(latents)                      (:149)
#       image = vae.decode(latents)[0]                             (:151)
#
# vae_scale_factor=8, num_latent_channels=32, patch_size=2 (Flux2Sampler.py:66-68).
# The packed latent seq len = (H/8/2)*(W/8/2) (after patchify halves H/8,W/8 again).
#
# DTYPE: the persistent Euler latent stays F32 (Flux2Sampler.py:88 dtype=float32);
# latent_model_input is cast to train_dtype (BF16) before the transformer (:121);
# the scheduler step upcasts to F32 and casts back (diffusers parity). The borrowed
# inference forward (model/KleinModel.klein_inference_forward) returns the packed
# flow; this driver runs the Euler update on the F32 latent.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar

from std.collections import Optional

from serenity_trainer.modelSampler.FlowMatchEulerDiscreteScheduler import (
    FlowMatchEulerDiscreteScheduler, make_flux2_scheduler, flux2_compute_empirical_mu,
)
from serenity_trainer.modelSetup.BaseFlux2Setup import (
    guidance_embedder_value, FLUX2_GUIDANCE_EMBEDS,
)


# ── the sampler output: a decoded image tensor [1, 3, H, W] ───────────────────
struct Flux2SampleOutput(Movable):
    var image: Tensor

    def __init__(out self, var image: Tensor):
        self.image = image^


# ── the denoise state: F32 Euler latent + the FLUX.2 scheduler ────────────────
# The per-step DiT evaluation is the model-unit's klein_inference_forward (the
# inference path: NO tape, NO checkpoints, NO host syncs — Flux2Sampler.py runs the
# transformer under @torch.no_grad()). The packed-token assembly
# (pack_latents/prepare_*_ids/modvecs from timestep/1000) and the VAE decode are the
# model unit's; this driver owns the scheduler + CFG + the F32 Euler state, exactly
# like ZImageSampler owns its loop.
struct Flux2DenoiseState(Movable):
    var latent: Tensor          # packed F32 Euler state [1, seq, out_ch]
    var scheduler: FlowMatchEulerDiscreteScheduler

    def __init__(out self, var latent: Tensor, var scheduler: FlowMatchEulerDiscreteScheduler):
        self.latent = latent^
        self.scheduler = scheduler^


# Build the initial denoise state. `packed_latent` is the already pack_latents'd
# randn latent [1, seq, out_ch] F32 (the model unit owns patchify/pack; the sampler
# owns the randn seed + the scheduler). mu = compute_empirical_mu(image_seq_len,
# diffusion_steps) (Flux2Sampler.py:95-96).
def make_flux2_denoise_state(
    var packed_latent: Tensor, image_seq_len: Int, diffusion_steps: Int,
) raises -> Flux2DenoiseState:
    var scheduler = make_flux2_scheduler(diffusion_steps, image_seq_len)
    return Flux2DenoiseState(packed_latent^, scheduler^)


# Per-step transformer t-input: timestep/1000 (Flux2Sampler.py:122, expanded
# /1000). `timestep` is the scheduler's discrete value (= sigma*1000), NOT the loop
# index. Returns the /1000 value (the literal transformer kwarg) for reference; the
# transformer re-scales it ×1000 internally (transformer_flux2.py:1231).
def flux2_step_t_model(state: Flux2DenoiseState, step_index: Int) -> Float32:
    return state.scheduler.timesteps[step_index] / Float32(1000.0)


# Per-step INTEGER timestep the t_embedder actually sees (Flux2Sampler.py:122 feeds
# timestep/1000; transformer_flux2.py:1231 multiplies back ×1000 ⇒ the embedder sees
# the scheduler's discrete integer value). This is the value to feed the modvec
# build (build_klein_step_mods_device_cached's timestep_value), NOT flux2_step_t_model.
def flux2_step_t_embedder(state: Flux2DenoiseState, step_index: Int) -> Float32:
    return state.scheduler.timesteps[step_index]


# Per-step INTEGER guidance value the guidance_embedder sees (Klein 9B
# guidance_embeds=True). Flux2Sampler.py:113-114 feeds guidance = tensor([cfg_scale])
# when transformer.config.guidance_embeds; transformer_flux2.py:1234 multiplies it
# ×1000 ⇒ the embedder sees cfg_scale*1000. Returns None when not guidance-distilled
# (then the CFG batch path is used instead, see flux2_use_cfg).
def flux2_guidance_value(
    cfg_scale: Float32, guidance_embeds: Bool = FLUX2_GUIDANCE_EMBEDS
) -> Optional[Float32]:
    return guidance_embedder_value(cfg_scale, guidance_embeds)


# Cast the F32 Euler latent to BF16 for the transformer input (Flux2Sampler.py:121).
def flux2_latent_to_train_dtype(state: Flux2DenoiseState, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(state.latent, STDtype.BF16, ctx)


# CFG combine (Flux2Sampler.py:131-133): noise_pred = neg + cfg*(pos - neg).
def flux2_cfg_combine(
    pos: Tensor, neg: Tensor, cfg_scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    var diff = sub(pos, neg, ctx)
    var scaled = mul_scalar(diff, cfg_scale, ctx)
    return add(neg, scaled, ctx)


# Whether CFG uses the batch_size==2 (pos/neg) path: cfg>1 and NOT guidance_embeds
# (Flux2Sampler.py:73). batch_size = 2 if cfg>1 and not guidance_embeds else 1.
# The flagship FLUX.2-klein-base-9B is guidance_embeds=FALSE ⇒ it DOES take the
# pos/neg CFG batch path when cfg>1. Only guidance-distilled variants
# (guidance_embeds=True) take batch_size=1 with guidance injected via
# flux2_guidance_value instead of the pos/neg CFG batch.
def flux2_use_cfg(
    cfg_scale: Float32, guidance_embeds: Bool = FLUX2_GUIDANCE_EMBEDS
) -> Bool:
    return (cfg_scale > Float32(1.0)) and (not guidance_embeds)


# Sampler batch size (Flux2Sampler.py:73): 2 if the CFG batch path is used, else 1.
def flux2_batch_size(
    cfg_scale: Float32, guidance_embeds: Bool = FLUX2_GUIDANCE_EMBEDS
) -> Int:
    if flux2_use_cfg(cfg_scale, guidance_embeds):
        return 2
    return 1


# One Euler step given the (already CFG-combined) predicted flow.
# Flux2Sampler.py:135: latent = scheduler.step(noise_pred, timestep, latent)[0].
# NO leading minus — the transformer output IS the flow, added via dt*noise_pred.
def flux2_euler_step(
    mut state: Flux2DenoiseState, noise_pred: Tensor, step_index: Int, ctx: DeviceContext
) raises:
    state.latent = state.scheduler.step(noise_pred, state.latent, step_index, ctx)
