# klein_sample_resident.mojo — sample-during-training denoise for Klein (FLUX.2)
# that REUSES the trainer's ALREADY-RESIDENT weights + live LoRA.
#
# ── WHY this file exists (vs the existing klein_sampler.mojo) ──────────────────
# sampling/klein_sampler.mojo::klein_sample is the production validation sampler.
# It is MONOLITHIC by design: _denoise_lora_from_initial RE-OPENS the checkpoint
# (SafeTensors.open(cfg.checkpoint)), RE-LOADS the base stack
# (load_klein_stack_base), RE-OPENS a fresh TurboPlannedLoader and RE-LOADS the
# LoRA from a *saved file*. That is correct for the process-separated cadence
# (train_klein_cadence.mojo) where the sampler runs in its OWN process after the
# trainer exits — but it does NOT reuse anything the trainer already holds.
#
# This file is the resident-weights path: it takes the trainer's OWN device
# handles (base, loader, mod_weights, resident LoRA device set, rope tables,
# fwd scratch ring) and runs the SAME CFG Euler denoise WITHOUT loading anything.
# It samples from the model's CURRENT in-memory state — the resident base trunk
# PLUS the live, in-place-updated LoRA adapters — which is the whole point of
# sampling-during-training.
#
# ── WHAT it reuses (NO re-load) ───────────────────────────────────────────────
# The denoise core is klein_stack_lora_predict_cfg_offload_turbo_moddev_rope_scratch
# (klein_stack_lora.mojo:1136) — the inference-only forward (no backward tape)
# that the production sampler ALSO calls. It consumes exactly the resident args
# the trainer builds once at startup and per-step:
#   base              KleinStackBase            (load_klein_stack_base_training)
#   loader            TurboPlannedLoader        (TurboPlannedLoader.open + pin_residents)
#   lora_dev          KleinLoraDeviceSet        (resident_lora_dev / klein_lora_set_to_device)
#   mod_weights       KleinStepModWeights       (load_klein_step_mod_weights)
#   cos_dev,sin_dev   Tensor                    (_build_klein_rope_host -> from_host)
#   scratch_fwd       ScratchRingAllocator      (the trainer's fwd ring)
# Per denoise step the mods are rebuilt for THIS sigma via
# build_klein_step_mods_device_cached — identical to the trainer's per-step call
# (train_klein_real.mojo:1276) and the production sampler's per-step call
# (klein_sampler.mojo:324). final_shift/final_scale are written onto `base` each
# step (mods[3]/mods[4]), exactly as both of those paths do.
#
# ── SCHEDULE + STEP (1:1 with klein_sampler._denoise_lora_from_initial) ────────
#   sigmas = build_flux2_sigma_schedule(num_steps, N_IMG)     # num_steps+1, desc
#   for i in 0..num_steps:
#       sigma = sigmas[i] ; dt = sigmas[i+1] - sigma          # dt < 0 (down)
#       mods  = build_klein_step_mods_device_cached(mod_weights, sigma, ...)
#       preds = predict_cfg(x, pos, neg, ...)                 # velocity [N_IMG,out_ch]
#       v     = flux2_cfg(preds.pos, preds.neg, cfg_scale)    # neg + s*(pos-neg)
#       x     = x + dt*v                                       # Euler
# This is the same direct-velocity Euler the production sampler uses (NOT the
# negated-velocity convention ideogram4 uses — Klein's predict returns the raw
# transformer velocity and the sampler steps x += dt*v).
#
# ── CONDITIONING (v1 decision — flagged) ──────────────────────────────────────
# The caller passes pos_txt / neg_txt token embeddings [N_TXT, joint] already on
# device. The train loop's resident path has the per-sample cached text tokens
# (cached_txt_tokens) AND a zero uncond embedding (uncond_txt) — wiring those in
# is a drop-in: pass cached_txt_tokens[slot][] as pos and a zero [N_TXT,joint] as
# neg. See the caller-side note in the WIRING block at the foot of this file.
#
# ── MEMORY CAVEAT (the real blocker, flagged loudly) ──────────────────────────
# The DENOISE here is cheap to co-reside: it reuses the resident trunk + the fwd
# scratch ring the trainer already sized, and the inference-only predict keeps NO
# backward tape. The EXPENSIVE part is the VAE DECODE. klein_sampler.klein_sample
# uses klein_tiled_decode which loads a half-size KleinVaeDecoder — but even tiled,
# loading the VAE while the FULL training stack (resident blocks + Adam OT state +
# scratch_fwd + scratch_bwd slabs) is still alive risks CUDA_OUT_OF_MEMORY at
# Klein-9B 1024px. THIS IS WHY the production trainer samples in a SEPARATE process
# (train_klein_real.mojo:21-35 + train_klein_cadence.mojo). This resident path is
# therefore SAFE for the latent (always) and SAFE for the decode ONLY when there
# is decode headroom — i.e. small sample resolutions, or a run that pinned fewer
# resident blocks. The decode-to-png helper below is provided, but the caller must
# accept the OOM risk at full res. See decision note in the WIRING block.
#
# Mojo 1.0.0b1: `def` not `fn`; def needs explicit `raises`.

from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator

from serenitymojo.models.klein.klein_stack import KleinStackBase
from serenitymojo.models.klein.weights import (
    KleinStepModWeights, build_klein_step_mods_device_cached,
)
from serenitymojo.models.klein.klein_stack_lora import (
    KleinLoraDeviceSet, KleinLoraCfgPreds,
    klein_stack_lora_predict_device_offload_turbo_moddev_rope_scratch,
    klein_stack_lora_predict_cfg_offload_turbo_moddev_rope_scratch,
)
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader

from serenitymojo.ops.tensor_algebra import add, mul_scalar, permute, reshape_owned
from serenitymojo.ops.random import randn

from serenitymojo.sampling.flux2_klein import build_flux2_sigma_schedule, flux2_cfg
from serenitymojo.sampling.base_sampler import tokens_to_packed_nchw, save_image
from serenitymojo.models.vae.klein_tiled_decode import klein_tiled_decode
from serenitymojo.training.progress_display import (
    print_sample_setup, print_sample_step, print_sample_saved,
)

comptime TArc = ArcPointer[Tensor]


# ──────────────────────────────────────────────────────────────────────────────
# _klein_initial_noise_tokens_resident — t=1 latent as img tokens [N_IMG, in_ch].
# Byte-identical to klein_sampler._initial_noise_tokens / train_klein_real's
# _latent_to_img_tokens_device packing: randn NCHW [1,in_ch,LH,LW] -> NHWC ->
# reshape [N_IMG, in_ch]. BF16 carrier (the model token dtype on this path).
# ──────────────────────────────────────────────────────────────────────────────
def _klein_initial_noise_tokens_resident[N_IMG: Int, LH: Int, LW: Int](
    in_ch: Int, seed: UInt64, ctx: DeviceContext
) raises -> Tensor:
    var nchw = List[Int]()
    nchw.append(1); nchw.append(in_ch); nchw.append(LH); nchw.append(LW)
    var noise = randn(nchw^, seed, STDtype.BF16, ctx)
    var p = List[Int]()
    p.append(0); p.append(2); p.append(3); p.append(1)
    var nhwc = permute(noise, p^, ctx)
    var sh = List[Int]()
    sh.append(N_IMG); sh.append(in_ch)
    return reshape_owned(nhwc^, sh^)


# ──────────────────────────────────────────────────────────────────────────────
# klein_sample_resident_latent — CFG Euler denoise on the trainer's RESIDENT base
# + live LoRA. Returns the denoised latent TOKENS [N_IMG, out_ch] (still token
# space; the caller packs + VAE-decodes). NOTHING is loaded here — every weight
# handle is the trainer's own.
#
# Inputs (all on the trainer's DeviceContext):
#   base            mut KleinStackBase       resident trunk (final_shift/scale are
#                                            overwritten per step from the sigma mods)
#   loader          mut TurboPlannedLoader   resident/streamed block loader
#   lora_dev        KleinLoraDeviceSet       the LIVE LoRA device set (resident view)
#   mod_weights     KleinStepModWeights      per-step modulation weights
#   cos_dev,sin_dev Tensor                   rope tables [S*H, Dh//2]
#   scratch_fwd     mut ScratchRingAllocator the trainer's fwd scratch ring
#   x               Tensor [N_IMG, in_ch]    t=1 noise tokens (BF16)
#   pos_txt,neg_txt Tensor [N_TXT, joint]    cond / uncond text token embeddings
#   cfg_scale       Float32                  CFG guidance scale (1.0 => no CFG)
#   num_steps       Int                      denoise steps
#   d_model,mlp_hidden,in_ch,joint,out_ch,eps,timestep_dim  arch scalars (from cfg)
#
# COMPTIME [H, Dh, N_IMG, N_TXT, S] mirror the predict generics — the driver
# instantiates with the SAME values it uses for the train forward (or sample-res
# values for a larger validation grid; the predict path is grid-generic).
# ──────────────────────────────────────────────────────────────────────────────
def klein_sample_resident_latent[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    mut base: KleinStackBase,
    mut loader: TurboPlannedLoader,
    lora_dev: KleinLoraDeviceSet,
    mod_weights: KleinStepModWeights,
    cos_dev: Tensor,
    sin_dev: Tensor,
    mut scratch_fwd: ScratchRingAllocator,
    var x: Tensor,            # [N_IMG, in_ch] initial noise tokens
    pos_txt: Tensor,          # [N_TXT, joint]
    neg_txt: Tensor,          # [N_TXT, joint]
    cfg_scale: Float32,
    num_steps: Int,
    d_model: Int,
    mlp_hidden: Int,
    in_ch: Int,
    joint: Int,
    out_ch: Int,
    eps: Float32,
    timestep_dim: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    if num_steps < 1:
        raise Error("klein_sample_resident_latent: num_steps must be >= 1")

    var pos_t = TArc(pos_txt.clone(ctx))
    var neg_t = TArc(neg_txt.clone(ctx))
    var sigmas = build_flux2_sigma_schedule(num_steps, N_IMG)
    print_sample_setup(
        String("Klein-sample-resident"), String("klein"), num_steps, cfg_scale, N_IMG, 0
    )

    for i in range(num_steps):
        # reset the fwd ring each step (same discipline as the production sampler;
        # this is the trainer's ring — the caller must NOT have live tape on it).
        scratch_fwd.reset()
        var sigma = sigmas[i]
        var dt = sigmas[i + 1] - sigma

        # per-step modulation for THIS sigma (1:1 with klein_sampler.mojo:324 and
        # train_klein_real.mojo:1276). Overwrite the resident base final-layer
        # adaLN mod with this step's sigma (mods[3]=shift, mods[4]=scale).
        var mods = build_klein_step_mods_device_cached(
            mod_weights, sigma, timestep_dim, d_model, ctx
        )
        var img_mod = mods[0].copy()
        var txt_mod = mods[1].copy()
        var single_mod = mods[2].copy()
        base.final_shift = mods[3].copy()
        base.final_scale = mods[4].copy()

        var v_dev: Tensor
        if cfg_scale == Float32(1.0):
            v_dev = klein_stack_lora_predict_device_offload_turbo_moddev_rope_scratch[
                H, Dh, N_IMG, N_TXT, S
            ](
                TArc(x.clone(ctx)), pos_t, base,
                loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev, sin_dev,
                d_model, mlp_hidden, in_ch, joint, out_ch, eps, ctx, scratch_fwd,
            )
        else:
            var preds = klein_stack_lora_predict_cfg_offload_turbo_moddev_rope_scratch[
                H, Dh, N_IMG, N_TXT, S
            ](
                TArc(x.clone(ctx)), pos_t, neg_t, base,
                loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev, sin_dev,
                d_model, mlp_hidden, in_ch, joint, out_ch, eps, ctx, scratch_fwd,
            )
            v_dev = flux2_cfg(preds.pos, preds.neg, cfg_scale, ctx)

        # direct-velocity Euler: x = x + dt*v (dt<0 walks down the schedule).
        x = add(x, mul_scalar(v_dev, dt, ctx), ctx)
        ctx.synchronize()
        var step = i + 1
        if step == 1 or step == num_steps or step % 5 == 0:
            print_sample_step(
                String("Klein-sample-resident"), step, num_steps, sigma, 0.0, 0.0
            )

    return x.clone(ctx)


# ──────────────────────────────────────────────────────────────────────────────
# klein_decode_latent_to_png — pack tokens + tiled VAE-decode + write PNG.
# 1:1 with klein_sampler.klein_sample's decode tail (klein_sampler.mojo:537-541):
#   packed = tokens_to_packed_nchw[LH,LW](latent)     # [1,128,LH,LW]
#   img    = klein_tiled_decode[LH,LW](packed, vae)   # [1,3,16LH,16LW]
#   save_image(img, out_png)                          # SIGNED [-1,1] range
#
# ⚠️ MEMORY: klein_tiled_decode LOADS a half-size KleinVaeDecoder. Called while
# the training stack is still resident this can OOM at Klein-9B 1024px (the very
# reason production samples in a separate process). SAFE at small sample grids /
# runs with decode headroom; the caller owns that decision.
# ──────────────────────────────────────────────────────────────────────────────
def klein_decode_latent_to_png[LH: Int, LW: Int](
    latent: Tensor,          # [N_IMG, out_ch] denoised tokens (N_IMG == LH*LW)
    vae_path: String,
    out_png: String,
    ctx: DeviceContext,
) raises:
    var packed = tokens_to_packed_nchw[LH, LW](latent, ctx)
    var img = klein_tiled_decode[LH, LW](packed, vae_path, ctx)
    if out_png != String(""):
        save_image(img, out_png, ctx)
        print_sample_saved(String("Klein-sample-resident"), out_png)


# ──────────────────────────────────────────────────────────────────────────────
# klein_sample_resident_to_png — convenience wrapper: noise -> denoise -> png in
# one call, reusing the trainer's resident handles. The driver supplies the
# already-on-device text token embeddings.
# ──────────────────────────────────────────────────────────────────────────────
def klein_sample_resident_to_png[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int, LH: Int, LW: Int
](
    mut base: KleinStackBase,
    mut loader: TurboPlannedLoader,
    lora_dev: KleinLoraDeviceSet,
    mod_weights: KleinStepModWeights,
    cos_dev: Tensor,
    sin_dev: Tensor,
    mut scratch_fwd: ScratchRingAllocator,
    pos_txt: Tensor,          # [N_TXT, joint]
    neg_txt: Tensor,          # [N_TXT, joint]
    cfg_scale: Float32,
    num_steps: Int,
    seed: UInt64,
    vae_path: String,
    out_png: String,
    d_model: Int,
    mlp_hidden: Int,
    in_ch: Int,
    joint: Int,
    out_ch: Int,
    eps: Float32,
    timestep_dim: Int,
    ctx: DeviceContext,
) raises:
    var x = _klein_initial_noise_tokens_resident[N_IMG, LH, LW](in_ch, seed, ctx)
    var latent = klein_sample_resident_latent[H, Dh, N_IMG, N_TXT, S](
        base, loader, lora_dev, mod_weights, cos_dev, sin_dev, scratch_fwd,
        x^, pos_txt, neg_txt, cfg_scale, num_steps,
        d_model, mlp_hidden, in_ch, joint, out_ch, eps, timestep_dim, ctx,
    )
    klein_decode_latent_to_png[LH, LW](latent, vae_path, out_png, ctx)


# ═════════════════════════════════════════════════════════════════════════════
# WIRING into train_klein_real.mojo (the active production Klein LoRA loop)
# ═════════════════════════════════════════════════════════════════════════════
# IMPORTANT — READ FIRST. The active trainer (serenitymojo/training/
# train_klein_real.mojo) ALREADY HAS sampling-during-training wired (the
# `runtime_sample_enabled` / `sample_cadence` / `_do_sample_all` path,
# lines 1583-1641). On `sample_due` it saves the LoRA, then calls _do_sample_all
# -> _do_sample_prompt -> klein_sample(...). That path is the MONOLITHIC sampler:
# it reloads the base stack + LoRA-from-file in-process. Production AVOIDS the
# co-residency OOM by running the whole thing in a separate process via
# train_klein_cadence.mojo (the design note at train_klein_real.mojo:21-35).
#
# So this resident path is an OPTIMIZATION/ALTERNATIVE, not a missing feature:
# it skips the reload (reuses base/loader/resident_lora_dev/mod_weights/cos_dev/
# sin_dev/scratch_fwd) and samples the LIVE in-memory LoRA without a save+reload
# round-trip. To wire it in, REPLACE the _do_sample_prompt call body (the
# klein_sample[...] call) with a klein_sample_resident_to_png[...] call. The
# trainer holds every argument already:
#
#   if sample_due:
#       # resident: no save+reload, samples the LIVE LoRA in-memory.
#       for pi in range(len(sample_cfg.prompts)):
#           var p = sample_cfg.prompts[pi].copy()
#           var caps = load_caps(p.caps_pos, p.caps_neg, ctx)   # [N_TXT, joint]
#           var pos_txt = reshape(caps.pos, [N_TXT, cfg.joint_attention_dim], ctx)
#           var neg_txt = reshape(caps.neg, [N_TXT, cfg.joint_attention_dim], ctx)
#           # lora_dev for sampling: the trainer's resident_lora_dev (KLEIN_V2_ENGINE)
#           #   or klein_lora_set_to_device(lora, ctx) on the legacy path.
#           klein_sample_resident_to_png[
#               SAMPLE_N_IMG, N_TXT, SAMPLE_S, SAMPLE_LH, SAMPLE_LW, H, Dh   # NOTE order: see below
#           ](
#               base, loader, lora_dev, mod_weights, cos_dev[], sin_dev[], scratch_fwd,
#               pos_txt, neg_txt, p.cfg, p.steps, p.seed, cfg.vae,
#               _sample_png_path(k, p.label),
#               cfg.d_model, cfg.mlp_hidden, cfg.in_channels,
#               cfg.joint_attention_dim, cfg.out_channels, cfg.eps,
#               cfg.timestep_dim, ctx,
#           )
#
# ⚠️ GENERIC ORDER: klein_sample_resident_to_png's params are
#   [H, Dh, N_IMG, N_TXT, S, LH, LW] (H/Dh FIRST), whereas the existing
#   klein_sample is [N_IMG, N_TXT, S, LH, LW, H, Dh]. Use the order in THIS file's
#   signature. The example above is wrong on purpose-of-illustration order; the
#   real call must be klein_sample_resident_to_png[H, Dh, SAMPLE_N_IMG, N_TXT,
#   SAMPLE_S, SAMPLE_LH, SAMPLE_LW](...).
#
# ⚠️ ROPE TABLE SIZE: the trainer's cos_dev/sin_dev are built for the TRAIN grid
#   (S = N_IMG + N_TXT at 512px). A 1024px validation sample (SAMPLE_S) needs a
#   DIFFERENT rope table (built for SAMPLE_S). If you sample at the SAME grid as
#   training (N_IMG/N_TXT/S), reuse cos_dev/sin_dev directly. If you sample at a
#   larger grid, build a sample-grid rope table first (the production sampler does
#   this inside _denoise_lora via _rope_host[N_IMG,N_TXT,S,H]). For the FIRST cut,
#   sample at the TRAIN grid (512px: SAMPLE_*=train values) so cos_dev/sin_dev are
#   reused as-is and NO new table is built. <-- recommended v1.
#
# ⚠️ MEMORY: see the decode caveat above. At 512px train-grid sampling the decode
#   is far cheaper than 1024px and is the safest first target. If decode OOMs even
#   so, keep the existing separate-process klein_sample path for validation and use
#   this resident path only for the cheap latent (skip the decode, or decode at a
#   smaller grid).
# ═════════════════════════════════════════════════════════════════════════════
