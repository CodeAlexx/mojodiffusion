# train_ltx2_av.mojo — LTX-2.3 22B VIDEO-MODE LoRA trainer (musubi-tuner oracle).
#
# PRODUCTION MILESTONE-1 (2026-07-09): the real video-mode training loop over the
# lead-gated composition layer:
#   per-block arm   models/ltx2/ltx2_video_backward.mojo   (torch gate cos 1.0)
#   stack + tail    models/ltx2/ltx2_video_stack.mojo      (2-block torch gate cos
#                   1.0 fwd/d_input/16x2 LoRA; 48-block real-depth smoke PASS)
# Oracle = kohya musubi-tuner (SerenityTrainer does not support LTX-2). Recipe pinned
# from the pinned musubi-tuner oracle (docs/ltx_2.md + ltx2_train_network.py):
#   * cache: musubi-NATIVE files ({key}_..._ltx2.safetensors latents +
#     {key}_ltx2_te.safetensors video_prompt_embeds/prompt_attention_mask).
#     The TE embeds are POST-connector (musubi runs Embeddings1DConnector at
#     TE-cache time) => used DIRECTLY as block context; mask asserted all-ones
#     (connector replaces pads with learnable registers — measured 2026-07-09).
#   * sigma: shifted_logit_normal, mode STRETCHED for ltx 2.3 (percentile
#     stretch 3.0902/-2.5758, eps=1e-3 reflect, 10% uniform), std 1.0, shift
#     auto-interpolated from sequence length (0.95@1024tok -> 2.05@4096tok,
#     UNCLAMPED — ltx2_train_network.py:1768/1799). Math in
#     training/ltx2/schedule.mojo (verified line-exact vs musubi).
#   * target = noise - latent; plain MSE mean (weighting_scheme FORCED none,
#     ltx2_train_network.py:7332); video_loss_weight default 1.0.
#   * optimizer: torch-default AdamW (betas .9/.999, eps 1e-8, wd 0.01),
#     lr 1e-4, CONSTANT schedule, global grad clip max_norm=1.0
#     (hv_train_network.py:4141 default / :3581 clip call). Masters are F32
#     (musubi ss_full_bf16=False keeps trainable params F32,
#     hv_train_network.py:2504); bf16 ONLY at save. The bf16-master era
#     (pre-2026-07-16-evening) absorbed 30-57% of A-updates per step — see
#     _adamw_host_list_f32. Anchor digits re-baselined (grad path sees
#     unrounded F32 kaiming A from step 1; loss@step1 unchanged, B=0).
#   * LoRA: rank 32 / alpha 32 (docs/ltx_2.md standard command), video-mode
#     active surface = {attn1,attn2}x{to_q,to_k,to_v,to_out.0} = 8/block x 48
#     = 384 adapters (musubi t2v preset also CREATES audio-module adapters but
#     they receive no grads with audio=None; we train+save the video set).
#     Init: A ~ kaiming-uniform U(-1/sqrt(in), +1/sqrt(in)) (kohya), B = 0.
#   * save keys: diffusion_model.transformer_blocks.N.<module>.lora_A/B.weight
#     (musubi convert_lora_to_comfy target format; training/ltx2/lora_surface).
#
# DTYPE (milestone-1): stack fwd+bwd F32 end-to-end (the frozen block backward
# is an F32 carrier — see the real-depth smoke header); head runs BF16 (the
# musubi network dtype) with outputs cast to F32. Speed pass comes later; the
# 48-block streamed F32 step measured ~58s on the real dev-fp8 checkpoint.
#
# LEVERS NOTE: loss/optimizer stay the literal legacy-default formulas
# (levers C13 default-off is formula-identical); routing through
# training/levers.mojo lands with the config-JSON stage — owed follow-up.
#
# VAL_LOSS (P2): --val_cache_dir + --validate_every run a forward-ONLY eval over
# the val cache at cadence (same F32 MSE-mean loss formula + levers seam as train,
# mean of per-sample scalars — musubi-exact hv:3155-3167). DELIBERATE DEVIATION:
# musubi's val noise+sigma are FRESH UNSEEDED global-RNG draws that ADVANCE the
# training RNG (hv:3027/3029, no guard); ours are DETERMINISTIC per val-index on
# isolated seed streams (sigma seed cfg.seed*10007+1, noise cfg.seed*2000003+idx*
# 104729) — reproducible + isolated by construction, strictly better for overfit
# detection. Same-class, NOT bit-parity. Default-off.
# SAMPLING (P2): --sample_every saves the COMFY/PEFT artifact (<base>.safetensors)
# AND the native musubi artifact (<base>.native.safetensors), plus a ready-to-run
# render command file (NEVER auto-run — staged-loader rule). The render path is
# scripts/ltx2_hq_ref_run.py (loads COMFY keys via LTXV_LORA_COMFY_RENAMING_MAP):
# musubi's OWN generator OOMs on 24GB loading Gemma next to the resident DiT
# (MEASURED 2026-07-16, campaign plan P2 — bf16-single-file AND its 8bit+swap-44
# recipe both, then "Generation complete" over an EMPTY dir). The native artifact
# is kept as the musubi-parity export (that generator returns once precached
# prompts are wired).
# RESUME (musubi parity): musubi keeps LoRA masters in F32 (network_dtype F32,
# hv_train_network.py:2504; bf16 ONLY under opt-in --full_bf16 with stochastic
# rounding) and on resume restores network weights + optimizer state + step
# (metadata priority, scheduler fallback). We match that: --resume loads the
# `.state.safetensors` sidecar (F32-exact `.lora_A/B.master` + AdamW moments ->
# BIT-EXACT masters at the boundary, no bf16 hop). Old-era bf16-only state files
# fall back to a WARM resume with a LOUD banner (moments ARE restored; only the
# masters were bf16-rounded on the prior save — NOT trainer_core's zeroed-moments
# warm case, so its trainer_warn_warm_resume banner does not apply). The loop
# continues at saved_step+1. Because our sigma/noise/sample streams are
# (seed,step)-DERIVED (no RNG pickle), restoring the step alone restores every
# stream EXACTLY — stronger than musubi's RNG-state-pickle approach. Resume scope
# (seed / dataset size+identity / token layout) is guarded through the shared
# trainer_core 5-field __meta__ (trainer_state_meta + trainer_resume_meta_guard);
# legacy 2-field metas still resume (step,seed only).
#
# Geometry is COMPTIME (milestone-1 monomorphization): 512x288x25f ->
# latents [128,4,9,16] -> S_V=576 tokens, N_TXT=1024. Cache samples with a
# different latents key are skipped with a loud count (klein bucket-filter
# pattern).
#
# Run:
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -O2 -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/training/train_ltx2_av.mojo -o /tmp/ltx2_av_trainer
#   /tmp/ltx2_av_trainer --dataset_cache_dir /path/to/ltx2_musubi_v3/cache \
#       --output_dir ./output/ltx2_video_lora --max_steps 4

from std.sys import argv
from std.os import listdir
from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import sqrt
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import (
    sys_mkdirs, sys_open, sys_close, sys_pwrite, BytePtr,
    O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import reshape, permute, sub, mul_scalar, add, slice, concat, mul

from serenitymojo.models.dit.ltx2_dit import LTX2Config, LTX2AVBlockWeights
from serenitymojo.models.ltx2.ltx2_video_stack import (
    LTX2VideoBlockSource, LTX2VideoTail, LTX2VideoStackHead, LTX2VideoHeadOut,
    ltx2_video_stack_lora_forward, ltx2_video_stack_lora_backward,
    video_lora_names, assert_prompt_mask_all_ones, _st_has,
    LTX2VideoStackGrads,
)
# ── AV (audio-video) arm (P6.2): the stack fwd/bwd, the rope builder, the loss
#    combine + masked reducers, and the tri-pair cache helpers. All gated:
#    stack fwd 37/0, bwd 115/0, rope 8/8, loss-combine 4/4, cache round-trip 9/9.
from serenitymojo.models.ltx2.ltx2_av_stack import (
    LTX2AVBlockSource, ltx2_av_build_head, ltx2_av_stack_forward, ltx2_av_stack_backward,
    LTX2AVHead, LTX2AVStackForward, LTX2AVStackGrads, _patchify, _av_lora_slots,
)
from serenitymojo.models.ltx2.ltx2_av_rope import ltx2_av_build_rope, LTX2AVRope
from serenitymojo.training.ltx2.masked_loss import (
    av_combine_loss, masked_loss_unmasked, masked_loss_audio_bt_mask, LTX2_LOSS_MSE,
)
from serenitymojo.training.ltx2.av_cache import (
    audio_cache_path, discover_audio_latents_key, discover_audio_lengths_key,
)
# autograd_v2 engine backward (L2 stack conductor), behind LTX2_V2_GRAPH (C13).
# LTX2_V2_SLAB adds the StepSlab + device-resident LoRA grad path (rung 2).
from serenitymojo.models.ltx2.ltx2_video_stack_graph import (
    ltx2_video_stack_lora_backward_graph,
    ltx2_video_stack_lora_backward_graph_slab,
)
from serenitymojo.models.ltx2.ltx2_video_stack_capture import (
    ltx2_video_stack_lora_backward_graph_capture,
    LTX2CaptureStage,
)

from serenitymojo.training.ltx2.acceptance import run_acceptance
from serenitymojo.training.ltx2.config import (
    LTX2TrainerConfig, print_config_summary, MODE_VIDEO, MODE_AV, SHIFT_STRETCHED,
    SIGMA_SHIFTED_LOGIT_NORMAL, _parse_float32,
    PRESET_V2V, PRESET_AUDIO, PRESET_AUDIO_REF_ONLY_IC,
)
from serenitymojo.training.ltx2.lora_surface import diffusion_lora_prefix, modules_per_block_for_preset
from serenitymojo.training.ltx2.v2v_cache import pair_reference, discover_ref_latent_key
from serenitymojo.training.ltx2.v2v_loss import v2v_target_cotangent
from serenitymojo.training.ltx2.conditioning_mask import (
    temporal_mask, first_frame_mask, spatial_crop_mask, union_into,
)
from serenitymojo.training.ltx2.mask_cache import load_mask_tokens, mask_cache_path
from serenitymojo.training.ltx2.readiness import print_readiness
from serenitymojo.training.ltx2.schedule import (
    shift_for_sequence_length,
    shifted_logit_normal_sigma_legacy,
    shifted_logit_normal_sigma_stretched,
)
from serenitymojo.training.train_step import (
    LoraAdapter, LoraGrads, _adamw_host_list_f32, _bf16_to_f32_list,
)
from serenitymojo.training.lora_save import (
    NamedLora, save_lora_peft, save_lora_serenity_trainer,
    F32LoraState, save_lora_train_state_f32,
    lora_train_state_has_f32_masters, load_lora_train_state_f32,
    load_lora_train_state, load_lora_train_state_meta,
)
from serenitymojo.training.trainer_core import (
    trainer_resolve_resume_path, trainer_state_meta, trainer_resume_meta_guard,
    GradAccumWindow,
)
from serenitymojo.training.levers import (
    levers_loss_active, levers_loss_grad, levers_optimizer_active,
    caption_dropout_pick,
    F32AdapterView, LeversOptimizerState, levers_optimizer_step_host_f32,
    levers_optimizer_validate,
)
from serenitymojo.training.levers_optimizer_sidecar import (
    levers_optimizer_sidecar_save, levers_optimizer_sidecar_load,
    levers_optimizer_sidecar_path,
)
# SPEED PASS Phase A: device-resident LoRA + fused device AdamW + device clip.
from serenitymojo.training.fused_adamw_multitensor import fused_adamw_step
from serenitymojo.training.on_device_global_norm import on_device_grad_stats
from serenitymojo.io.env import (
    env_int, env_or, serenity_checkpoint, serenity_repo_root,
)
from serenitymojo.training.lr_schedule import (
    transformers_lr_for_step, LR_CONSTANT, LR_LINEAR, LR_COSINE,
)
from serenitymojo.training.progress_display import print_trainer_progress

# host N(0,1) draw — the SAME rand-0.8.5 ChaCha12/Box-Muller stream randn uses.
from serenitymojo.ops.random import _std_rng_pair

# ── milestone-1 comptime geometry (512x288x25f disney cache) ─────────────────
comptime NF = 4
comptime NH = 9
comptime NW = 16
comptime S_V = NF * NH * NW          # 576
comptime N_TXT = 1024
comptime C = 128
comptime VD = 4096
comptime FFV = 16384          # video FFN hidden (ff.net.0.proj out / ff.net.2 in)
comptime NUM_LAYERS = 48
comptime FRAME_RATE = Float64(25.0)
comptime EPS = Float32(1e-6)
comptime LAT_KEY = "latents_4x9x16_bfloat16"
comptime ENC_KEY = "video_prompt_embeds_bfloat16"
comptime DEFAULT_CKPT_NAME = "ltx-2.3-22b-dev-fp8.safetensors"
comptime MAX_GRAD_NORM = Float32(1.0)   # musubi hv_train_network default


def _has_arg(args: List[String], name: String) -> Bool:
    for i in range(len(args)):
        if String(args[i]) == name:
            return True
    return False


def _parse_int_arg(s: String) raises -> Int:
    var v = 0
    var b = s.as_bytes()
    for i in range(len(b)):
        var c = b[i]
        if c < UInt8(48) or c > UInt8(57):
            raise Error(String("expected integer, got: ") + s)
        v = v * 10 + Int(c - UInt8(48))
    return v


# ── deterministic host uniforms (splitmix64) for the sigma sampler branches ──
def _splitmix64(state: UInt64) -> UInt64:
    var z = state + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _uniform01(seed: UInt64) -> Float32:
    # top 24 bits -> [0,1)
    var w = _splitmix64(seed)
    return Float32(Float64(w >> 40) * (1.0 / 16777216.0))


# ── LTX2 min/max_timestep sigma-domain AFFINE RESCALE (ltx2:2011-2016) ───────
# σ' = σ·(max−min)/1000 + min/1000, applied to EVERY draw AFTER the sampler
# branch (NOT a clamp; this is the LTX2 form, differs from the base trainer's
# index-range form). Defaults min=0/max=1000 are the arithmetic IDENTITY — the
# guard preserves the bit-identical default sigma stream (no fp perturbation).
def _apply_sigma_rescale(sigma: Float32, cfg: LTX2TrainerConfig) -> Float32:
    if cfg.min_timestep == Float32(0.0) and cfg.max_timestep == Float32(1000.0):
        return sigma
    return sigma * (cfg.max_timestep - cfg.min_timestep) / Float32(1000.0) \
        + cfg.min_timestep / Float32(1000.0)


# ── musubi-recipe sigma for one step (s_v = token count of the geometry) ─────
def _sample_sigma(cfg: LTX2TrainerConfig, step: Int, s_v: Int) raises -> Float32:
    if cfg.timestep_sampling != SIGMA_SHIFTED_LOGIT_NORMAL:
        # SIGMA_UNIFORM
        return _apply_sigma_rescale(
            _uniform01(cfg.seed * UInt64(6364136223846793005) + UInt64(step) * 3 + 11), cfg)
    var shift: Float32
    if cfg.shifted_logit_shift >= Float32(0.0):
        shift = cfg.shifted_logit_shift
    else:
        shift = shift_for_sequence_length(s_v)
    var pair = _std_rng_pair(cfg.seed + UInt64(step) * UInt64(7919), UInt64(0))
    var z = Float32(pair.z0)
    if cfg.shifted_logit_mode != SHIFT_STRETCHED:
        return _apply_sigma_rescale(
            shifted_logit_normal_sigma_legacy(z, shift, cfg.logit_std), cfg)
    var u1 = _uniform01(cfg.seed * UInt64(2654435761) + UInt64(step) * 5 + 1)
    var u2 = _uniform01(cfg.seed * UInt64(1442695040888963407) + UInt64(step) * 7 + 2)
    return _apply_sigma_rescale(
        shifted_logit_normal_sigma_stretched(
            z, u1, u2, shift, cfg.logit_std,
            cfg.shifted_logit_eps, cfg.shifted_logit_uniform_prob,
        ), cfg)


# ── cache enumeration: pair musubi latent/TE files, filter to the comptime
#    geometry (klein bucket-filter pattern) ───────────────────────────────────
struct CacheItem(Copyable, Movable):
    var lat_path: String
    var te_path: String
    # IC-LoRA / v2v: paired downscaled REFERENCE latent cache path ("" = none).
    # Set by _pair_reference_cache when --reference_cache_dir is given; consumed
    # by the v2v forward (P5 unit 2). Non-breaking: the 2-arg ctor keeps it "".
    var ref_lat_path: String
    # P5.5 unit 2 inpaint: paired per-sample binary latent MASK cache path.
    var mask_path: String

    def __init__(out self, var lat_path: String, var te_path: String):
        self.lat_path = lat_path^
        self.te_path = te_path^
        self.ref_lat_path = String("")
        self.mask_path = String("")

    def set_reference(mut self, var ref_lat_path: String):
        self.ref_lat_path = ref_lat_path^

    def has_reference(self) -> Bool:
        return self.ref_lat_path.byte_length() > 0

    def set_mask(mut self, var mask_path: String):
        self.mask_path = mask_path^

    def has_mask(self) -> Bool:
        return self.mask_path.byte_length() > 0


# Pair a latent file with its TE file. Musubi latent names are
# `{itemkey}[_{window}]_{WxH}_ltx2.safetensors` and item keys themselves may
# contain underscores (e.g. `16989751_095_f66d`), so a first-underscore split
# is WRONG. Robust rule: the TE file is `{X}_ltx2_te.safetensors` where X is
# the LONGEST prefix of the latent name followed by '_' — match against the
# actual TE files in the dir.
def _te_for_latent(lat_name: String, te_names: List[String]) raises -> String:
    var best = String("")
    for ref te in te_names:
        # te = "<X>_ltx2_te.safetensors" -> X
        var b = te.as_bytes()
        var keep = len(b) - len("_ltx2_te.safetensors")
        if keep <= 0:
            continue
        var x = String("")
        for i in range(keep):
            x += chr(Int(b[i]))
        if lat_name.startswith(x + "_") and x.byte_length() > best.byte_length():
            best = x
    if best.byte_length() == 0:
        raise Error(String("no TE cache pairs latent ") + lat_name)
    return best + "_ltx2_te.safetensors"


def _enumerate_cache(dir: String, lat_key: String) raises -> List[CacheItem]:
    var names = List[String]()
    var raw = listdir(dir)
    for ref e in raw:
        names.append(String(e))
    # stable order for reproducible sampling
    for i in range(len(names)):
        for j in range(len(names) - 1 - i):
            if names[j] > names[j + 1]:
                var tmp = names[j]
                names[j] = names[j + 1]
                names[j + 1] = tmp
    var te_names = List[String]()
    for ref n in names:
        if n.endswith("_ltx2_te.safetensors"):
            te_names.append(n.copy())
    var items = List[CacheItem]()
    var skipped_geom = 0
    for ref n in names:
        if not n.endswith("_ltx2.safetensors"):
            continue
        if n.endswith("_te.safetensors") or n.endswith("_audio.safetensors") or n.endswith("_dino.safetensors"):
            continue
        var lat_path = dir + "/" + n
        # geometry filter: header-only check for the geometry's latents key
        var st = ShardedSafeTensors.open(lat_path)
        if not _st_has(st, lat_key):
            skipped_geom += 1
            continue
        var te_path = dir + "/" + _te_for_latent(n, te_names)
        # fail loud if the TE pair is malformed (open raises)
        var te = ShardedSafeTensors.open(te_path)
        if not _st_has(te, String(ENC_KEY)):
            raise Error(String("TE cache missing ") + String(ENC_KEY) + ": " + te_path)
        items.append(CacheItem(lat_path^, te_path^))
    if skipped_geom > 0:
        print("  [cache] skipped", skipped_geom,
              "samples not matching geometry key", lat_key)
    return items^


# ── IC-LoRA / v2v: pair a downscaled REFERENCE latent with every target ──────
# Host-side (no ctx): sets CacheItem.ref_lat_path via the musubi basename route
# (serenitymojo/training/ltx2/v2v_cache.mojo). Fails loud on a missing/malformed
# reference cache. The forward that PREPENDS the reference tokens is P5 unit 2;
# this only makes the paired cache readable.
def _pair_reference_cache(mut items: List[CacheItem], ref_cache_dir: String) raises -> Int:
    var n = 0
    for i in range(len(items)):
        var rp = pair_reference(items[i].lat_path, ref_cache_dir, C)
        items[i].set_reference(rp^)
        n += 1
    return n


# P5.5 u2: pair the per-sample inpaint mask cache (same-basename route). Host-side;
# the mask is loaded + thresholded + unioned in the loop (mask_cache.load_mask_tokens).
def _pair_mask_cache(mut items: List[CacheItem], mask_cache_dir: String) -> Int:
    var n = 0
    for i in range(len(items)):
        items[i].set_mask(mask_cache_path(items[i].lat_path, mask_cache_dir))
        n += 1
    return n


# A multiple of `period` falls in (start_update, max_update] — i.e. a
# sample/validation cadence actually LANDS in the run's remaining update range.
# UPDATE space (sample_every/validate_every and max_steps all count updates).
def _cadence_reached(period: Int, start_update: Int, max_update: Int) -> Bool:
    if period <= 0:
        return False
    return (max_update // period) > (start_update // period)


# IC-LoRA / v2v first-frame conditioning (musubi ltx2_train_network.py:3327-3343,
# default p=0.1): a PER-STEP (per-batch) Bernoulli scalar drawn on its OWN seed
# base (distinct from caption-dropout's seed*31 — P7 stream-registry note). Video
# arms only (the num_frames>1 guard lives at the comptime NFp>1 call sites).
def _first_frame_fired(seed: UInt64, step: Int, p: Float32) -> Bool:
    if p <= Float32(0.0):
        return False
    return _uniform01(seed * UInt64(0xA5A5A5A5A5A5A5A5) + UInt64(step) * UInt64(2246822519) + 3) < p


# P5.5: per-condition Bernoulli draw on a DISTINCT seed base (own namespace, P7
# stream registry) — same shape as _first_frame_fired (which is base 0xA5A5..).
# Written as a per-step scalar (B=1); becomes a per-sample [B] vector before true
# batching (the official per-sample rationale: gradient decorrelation at bs>1).
comptime _FF_BASE = UInt64(0xA5A5A5A5A5A5A5A5)   # first-frame (== _first_frame_fired)
comptime _PREFIX_BASE = UInt64(0x9E3779B97F4A7C15)
comptime _SUFFIX_BASE = UInt64(0xC2B2AE3D27D4EB4F)
comptime _SPATIAL_BASE = UInt64(0x165667B19E3779F9)
comptime _MASK_BASE = UInt64(0xD1B54A32D192ED03)   # inpaint (P5.5 u2)


def _cond_fired(seed: UInt64, step: Int, p: Float32, base: UInt64) -> Bool:
    if p <= Float32(0.0):
        return False
    return _uniform01(seed * base + UInt64(step) * UInt64(2246822519) + 3) < p


# P5.5: build the combined per-token intrinsic-conditioning mask [S_TGT] = UNION
# of the ENABLED conditions, each drawn independently (own seed base). first-frame
# is one member (union path, no separate wiring). All-False when nothing fires ->
# the surfaces reduce to the plain t2v/v2v step (C13).
def _build_intrinsic_mask[NFp: Int, NHp: Int, NWp: Int](
    cfg: LTX2TrainerConfig, ff_p: Float32, step: Int
) raises -> List[Bool]:
    comptime S_TGT = NFp * NHp * NWp
    var mask = List[Bool]()
    for _ in range(S_TGT):
        mask.append(False)
    if _cond_fired(cfg.seed, step, ff_p, _FF_BASE):
        union_into(mask, first_frame_mask(NFp, NHp, NWp))
    if _cond_fired(cfg.seed, step, cfg.levers.prefix_conditioning_p, _PREFIX_BASE):
        union_into(mask, temporal_mask(NFp, NHp, NWp, cfg.levers.temporal_boundary, False))
    if _cond_fired(cfg.seed, step, cfg.levers.suffix_conditioning_p, _SUFFIX_BASE):
        union_into(mask, temporal_mask(NFp, NHp, NWp, cfg.levers.temporal_boundary, True))
    if _cond_fired(cfg.seed, step, cfg.levers.spatial_crop_conditioning_p, _SPATIAL_BASE):
        union_into(mask, spatial_crop_mask(NFp, NHp, NWp,
            cfg.levers.spatial_crop_y1, cfg.levers.spatial_crop_x1,
            cfg.levers.spatial_crop_y2, cfg.levers.spatial_crop_x2))
    return mask^


# P5.5: per-token mask [S_TGT] (frame->h->w) -> [1,C,NF,NH,NW] F32, broadcast over
# channels (each channel copies the token mask), for the noising clean-replace
# blend  noisy = mask5*clean + (1-mask5)*noisy  (invert=True gives 1-mask).
def _mask_to_5d_f32[NFp: Int, NHp: Int, NWp: Int](
    mask: List[Bool], invert: Bool, ctx: DeviceContext
) raises -> Tensor:
    comptime S_TGT = NFp * NHp * NWp
    var host = List[Float32]()
    for _c in range(C):
        for t in range(S_TGT):
            var v = mask[t] != invert    # invert=False -> mask; invert=True -> !mask
            host.append(Float32(1.0) if v else Float32(0.0))
    return Tensor.from_host(host, _sh5(1, C, NFp, NHp, NWp), STDtype.F32, ctx)


def _any_true(m: List[Bool]) -> Bool:
    for i in range(len(m)):
        if m[i]:
            return True
    return False


# musubi infer_ic_lora_strategy_from_preset (ltx2_train_network.py:85): the LoRA
# target preset maps the "auto" IC-LoRA strategy — v2v preset -> v2v,
# audio_ref_only_ic -> audio_ref_only_ic, else none.
def _infer_ic_lora_strategy(preset: Int) -> String:
    if preset == PRESET_V2V:
        return String("v2v")
    if preset == PRESET_AUDIO_REF_ONLY_IC:
        return String("audio_ref_only_ic")
    return String("none")


# IC-LoRA / v2v: load the paired CLEAN reference latent (bf16) from the ref cache
# file, discovering its latents_* key (its OWN downscaled geometry). Reshape to
# [1,C,RNF,RNH,RNW] happens at the call site (matching the base latent load).
def _load_ref_latent(item: CacheItem, ctx: DeviceContext) raises -> Tensor:
    if not item.has_reference():
        raise Error("train_ltx2_av: v2v arm reached without a paired reference latent")
    var st = ShardedSafeTensors.open(item.ref_lat_path)
    var key = discover_ref_latent_key(st)
    return Tensor.from_view_as_bf16(st.tensor_view(key), ctx)


def _load_bf16(path: String, name: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(path)
    return Tensor.from_view_as_bf16(st.tensor_view(name), ctx)


def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


def _sh5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d); s.append(e)
    return s^


def _perm021() -> List[Int]:
    var p = List[Int](); p.append(0); p.append(2); p.append(1); return p^


# patchify [1,C,NF,NH,NW] -> [1,s_v,C] (musubi VideoLatentPatchifier p=(1,1,1):
# token order f outer, h, w inner — same as the gated stack head).
def _patchify_tokens(x5: Tensor, s_v: Int, ctx: DeviceContext) raises -> Tensor:
    return permute(reshape(x5, _sh3(1, C, s_v), ctx), _perm021(), ctx)


# F32 master adapter (musubi keeps trainable LoRA params in F32 —
# ss_full_bf16=False default; hv_train_network.py:2504). Params/moments stay
# F32 for the whole run; bf16 happens ONLY at save (LoraAdapter constructor
# downcast). MEASURED 2026-07-16: the previous bf16-master write-back absorbed
# 30-57% of A-updates at this recipe's magnitudes (~28% relerr over 400 steps
# vs the F32 path) — see _adamw_host_list_f32 in train_step.mojo.
# NOTE: anchor digits shift vs the bf16-master era from step 1's grad_norm on
# (device A is no longer bf16-rounded); loss@step1 is unchanged (B=0).
struct F32Lora(Copyable, Movable):
    var a: List[Float32]
    var b: List[Float32]
    var ma: List[Float32]
    var va: List[Float32]
    var mb: List[Float32]
    var vb: List[Float32]

    def __init__(
        out self, var a: List[Float32], var b: List[Float32],
        var ma: List[Float32], var va: List[Float32],
        var mb: List[Float32], var vb: List[Float32],
    ):
        self.a = a^
        self.b = b^
        self.ma = ma^
        self.va = va^
        self.mb = mb^
        self.vb = vb^


# ── video-mode LoRA master set (kohya init) ──────────────────────────────────
# slot count is PRESET-DERIVED: t2v = 8 attention modules; v2v = +2 video FFN
# (ff.net.0.proj, ff.net.2) = 10. Everything below drives off len(mods).
def _module_paths(preset: Int) -> List[String]:
    # slot order MUST match video_lora_names(preset) (stack flat index = bi*n+slot)
    var out = List[String]()
    var names = video_lora_names(preset)
    for ref n in names:
        # strip trailing ".weight"
        var b = n.as_bytes()
        var keep = len(b) - 7
        var s = String("")
        for i in range(keep):
            s += chr(Int(b[i]))
        out.append(s)
    return out^


# Per-slot (in_f, out_f) for a video LoRA module path (mods[s], no ".weight").
# Attention q/k/v/out are VDxVD; the two v2v FFN linears are rectangular.
def _slot_dims(module_path: String) -> Tuple[Int, Int]:
    if module_path == "ff.net.0.proj":
        return (VD, FFV)   # up:   4096 -> 16384
    if module_path == "ff.net.2":
        return (FFV, VD)   # down: 16384 -> 4096
    return (VD, VD)        # attention


# Full ordered prefix list for save/resume — MUST match the save cadence exactly:
# diffusion_lora_prefix(bi, mods[s]) over bi in NUM_LAYERS, s in len(mods) (flat
# index bi*n+s, same order as _make_video_lora_set / the masters).
def _all_prefixes(mods: List[String]) -> List[String]:
    var out = List[String]()
    for bi in range(NUM_LAYERS):
        for s in range(len(mods)):
            out.append(diffusion_lora_prefix(bi, mods[s]))
    return out^


# Strip a known trailing suffix (caller guarantees s.endswith(suf)) — byte-build,
# consistent with the file's other suffix handling.
def _strip_suffix(s: String, suf: String) -> String:
    var b = s.as_bytes()
    var keep = s.byte_length() - suf.byte_length()
    var out = String("")
    for i in range(keep):
        out += chr(Int(b[i]))
    return out^


# LR for the CURRENT (1-based) optimizer step via the config-JSON LR levers.
# Routes through transformers_lr_for_step (musubi's LR path IS transformers), NOT
# the flame lr_for_step whose warmup ramp (step+1)/W diverges. transformers
# optimizer step k consumes lambda(k-1), so we pass step-1 (0-based). base = the
# ltx2-effective cfg.learning_rate, horizon = cfg.max_steps (argv wins). Default
# (LR_CONSTANT, warmup0) returns learning_rate EXACTLY — C13 byte-identity.
def _ltx2_step_lr(cfg: LTX2TrainerConfig, step: Int) raises -> Float32:
    var kind = cfg.levers.lr_scheduler
    if kind != LR_CONSTANT and kind != LR_LINEAR and kind != LR_COSINE:
        raise Error(
            String("train_ltx2_av: lr_scheduler ") + String(kind)
            + " not wired for ltx2 (musubi exposes constant|linear|cosine;"
            + " cosine_with_restarts/polynomial/rex owed follow-up)")
    return transformers_lr_for_step(
        cfg.learning_rate, step - 1, cfg.levers.lr_warmup_steps,
        cfg.max_steps, kind)


# Write a text file via io/ffi (NOT the `open` builtin — it pulls a stdlib mkdir
# external_call that conflicts with io/ffi's sys_mkdirs already in this graph).
def _write_text_file(path: String, body: String) raises:
    var flags = O_WRONLY | O_CREAT | O_TRUNC
    var fd = sys_open(path, flags, Int32(0o644))
    if fd < 0:
        raise Error(String("train_ltx2_av: cannot open for write: ") + path)
    var sb = body.as_bytes()
    var n = body.byte_length()
    var p = BytePtr(unsafe_from_address=Int(sb.unsafe_ptr()))
    var done = 0
    while done < n:
        var w = sys_pwrite(fd, p + done, n - done, done)
        if w <= 0:
            _ = sys_close(fd)
            raise Error(String("train_ltx2_av: write failed: ") + path)
        done += w
    _ = sys_close(fd)


# musubi NATIVE lora_unet_* key stem for (block, module) — dots -> underscores
# (to_out.0 -> to_out_0). DERIVED from mods[s] so it aligns with the master slot
# regardless of module order (same construction the 1152-key native round-trip
# gate exercises: ltx2_lora_musubi_native_roundtrip.mojo). save_lora_serenity_trainer
# appends .alpha/.lora_down.weight/.lora_up.weight.
def _native_prefix(bi: Int, module_path: String) -> String:
    var b = module_path.as_bytes()
    var conv = String("")
    for i in range(module_path.byte_length()):
        var c = Int(b[i])
        if c == 0x2E:  # '.'
            conv += "_"
        else:
            conv += chr(c)
    return String("lora_unet_model_transformer_blocks_") + String(bi) + "_" + conv


# Write a ready-to-run (NOT auto-executed) render command. Render path = the
# PROVEN HQ script with the COMFY/PEFT artifact (ltx2_hq_ref_run.py --user-lora,
# comfy keys via LTXV_LORA_COMFY_RENAMING_MAP) — the musubi generator MEASURED-
# FAILS on 24GB (2026-07-16, campaign plan P2: its in-process Gemma encode OOMs
# next to the resident DiT under both its bf16-single-file AND 8bit+swap-44
# recipes, then prints "Generation complete" over an EMPTY dir). The .native
# artifact is still saved alongside (musubi-parity; that generator path can
# return once precached-prompt support is wired). Kept a command FILE because
# renders must never run concurrently with the trainer (staged-loader rule).
# Write a ready-to-run (NOT auto-executed) render command. Render path = the
# PROVEN HQ script with the COMFY/PEFT artifact (ltx2_hq_ref_run.py --user-lora,
# comfy keys via LTXV_LORA_COMFY_RENAMING_MAP) — the musubi generator MEASURED-
# FAILS on 24GB (2026-07-16, campaign plan P2). Kept a command FILE because
# renders must never run concurrently with the trainer (staged-loader rule).
#
# IC-LoRA / v2v (A2a): when `v2v_ref_latent` is non-empty the command is a v2v
# render — it carries the CLEAN reference latent + an --ic-lora flag. THE PYTHON
# RENDER SIDE (ltx2_hq_ref_run.py) MUST, when --ic-lora is present, CONSUME:
#   --v2v-ref-latent <path>  a safetensors with a single latents_{F}x{H}x{W}_*
#                            key (the SAME format as the training ref cache) —
#                            load it, patchify (patch=1), and PREPEND it on the
#                            token axis (musubi ltx2_train_network.py:3354-3356);
#   --v2v-downscale <int>    reference_downscale — multiply the ref H/W rope
#                            coords by this (co-location, :3402-3405);
#   and build the two-grid rope with the SPINE convention (fixed max_pos
#   [20,2048,2048], causal 1, temporal /frame_rate in bf16 — _build_v2v_rope),
#   set the ref tokens' timestep to 0, denoise the target only, and SLICE the ref
#   off the output before decode (pred[:, ref_seq_len:], :3452). No such Python
#   flags exist yet — this is the command SURFACE; the render impl is P5 tail.
def _write_sample_cmd(
    path: String, cfg: LTX2TrainerConfig, native_path: String, step: Int,
    v2v_ref_latent: String = String(""), v2v_downscale: Int = 1,
) raises:
    var peft_path = cfg.output_dir + "/ltx2_video_lora_step" + String(step) + ".safetensors"
    var is_v2v = v2v_ref_latent.byte_length() > 0
    var body = String("#!/usr/bin/env bash\n")
    body += "# LTX2 in-training sample render (step " + String(step) + ") — comfy LoRA via the HQ script.\n"
    if is_v2v:
        body += "# IC-LoRA / v2v render: conditions on the reference latent below and slices it\n"
        body += "# off before decode. Requires ltx2_hq_ref_run.py --ic-lora support (P5 tail).\n"
    body += "# NOT auto-executed (staged-loader rule; never concurrent with the trainer).\n"
    # Cached-contexts (LTX2_CTX_DUMP flow, campaign plan §P2): when LTX2_CTX_DUMP
    # names a dumped ctx safetensors, emit --contexts so ltx2_hq_ref_run.py INJECTS
    # it (script:531) and SKIPS the in-process Gemma encode — that encode OOMs on
    # 24GB next to the resident DiT (the P2 lesson). Unset (default) -> no flag ->
    # in-process Gemma (fine off-box) and the .sh is byte-identical to the
    # pre-wiring emit (C13).
    var ctx_cache = env_or("LTX2_CTX_DUMP", String(""))
    var sample_python = env_or("SERENITY_SAMPLE_PYTHON", String("python3"))
    var sample_script = env_or(
        "SERENITY_LTX2_SAMPLE_SCRIPT",
        serenity_repo_root() + String("/scripts/ltx2_hq_ref_run.py"),
    )
    body += sample_python + String(" ") + sample_script + String(" \\\n")
    body += "  --user-lora " + peft_path + " \\\n"
    body += "  --user-lora-mult 1.0 \\\n"
    body += '  --prompt "' + cfg.sample_prompt + '" \\\n'
    if ctx_cache.byte_length() > 0:
        body += "  --contexts " + ctx_cache + " \\\n"
    if is_v2v:
        body += "  --ic-lora \\\n"
        body += "  --v2v-ref-latent " + v2v_ref_latent + " \\\n"
        body += "  --v2v-downscale " + String(v2v_downscale) + " \\\n"
    body += "  --num-frames 33 --steps 15 --seed 42 \\\n"
    body += "  --out " + cfg.output_dir + "/samples/step" + String(step) + "\n"
    _write_text_file(path, body)


def _kaiming_a(rank: Int, in_f: Int, seed: UInt64) -> List[Float32]:
    # kohya LoRA A init: kaiming_uniform_(a=sqrt(5)) == U(-1/sqrt(in), +1/sqrt(in))
    var bound = Float32(1.0) / sqrt(Float32(in_f))
    var out = List[Float32]()
    for i in range(rank * in_f):
        var u = _uniform01(seed * UInt64(1099511628211) + UInt64(i) * 2 + 1)
        out.append((u * Float32(2.0) - Float32(1.0)) * bound)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _make_video_lora_set(rank: Int, seed: UInt64, mods: List[String]) raises -> List[F32Lora]:
    var n_slots = len(mods)
    var set_ = List[F32Lora]()
    for bi in range(NUM_LAYERS):
        for s in range(n_slots):
            var dims = _slot_dims(mods[s])
            var in_f = dims[0]
            var out_f = dims[1]
            var sd = seed + UInt64(bi) * 97 + UInt64(s) * 13
            # A [rank,in] kaiming (bound uses THIS slot's in_f); B [out,rank] = 0.
            set_.append(F32Lora(
                _kaiming_a(rank, in_f, sd), _zeros(out_f * rank),
                _zeros(rank * in_f), _zeros(rank * in_f),
                _zeros(out_f * rank), _zeros(out_f * rank),
            ))
    return set_^


# upload the F32 master A/B lists as F32 device tensors for the F32 stack (the
# flat index i -> slot s = i % n_slots gives the slot's rectangular dims).
def _lora_to_device(
    set_: List[F32Lora], rank: Int, mods: List[String], ctx: DeviceContext,
    dt: STDtype = STDtype.F32,
) raises -> Tuple[List[ArcPointer[Tensor]], List[ArcPointer[Tensor]]]:
    # dt = the dtype the fwd/bwd LoRA side-GEMMs consume: F32 (default, C13) or
    # BF16 (bf16-stack — the delta GEMM must match the bf16 activation dtype).
    # The host F32 masters are the numeric source of truth either way.
    var n_slots = len(mods)
    var la = List[ArcPointer[Tensor]]()
    var lb = List[ArcPointer[Tensor]]()
    for i in range(len(set_)):
        ref lo = set_[i]
        var dims = _slot_dims(mods[i % n_slots])
        var in_f = dims[0]
        var out_f = dims[1]
        var at = Tensor.from_host(lo.a.copy(), [rank, in_f], dt, ctx)
        var bt = Tensor.from_host(lo.b.copy(), [out_f, rank], dt, ctx)
        la.append(ArcPointer[Tensor](at^))
        lb.append(ArcPointer[Tensor](bt^))
    return (la^, lb^)


# ═════════════════════════════════════════════════════════════════════════════
# The training run for ONE comptime geometry. Video mode over musubi-native
# caches; `S_Vp = NFp*NHp*NWp` tokens; images are 1-frame samples (NFp=1),
# exactly musubi's image handling ("treated as 1-frame samples (F=1)").
# ═════════════════════════════════════════════════════════════════════════════
def _run_geometry[
    S_Vp: Int, NFp: Int, NHp: Int, NWp: Int,
    S_REF: Int = 0, RNFp: Int = 0, RNHp: Int = 0, RNWp: Int = 0, DS_P: Int = 1,
](
    cfg: LTX2TrainerConfig, args: List[String], lat_key: String,
) raises:
    # v2v arm (S_REF>0): S_Vp is the COMBINED length (ref + target); the target
    # grid is NFp×NHp×NWp, the reference grid is RNFp×RNHp×RNWp, DS_P is the
    # reference_downscale used at cache time. S_REF==0 (the default) is the plain
    # t2v path — every v2v branch below is `comptime if S_REF > 0`, so the no-ref
    # codegen is byte-identical (comptime-eliminated).
    comptime S_TGT = NFp * NHp * NWp                 # target-grid token count
    comptime DS = DS_P                                # reference_downscale (LITERAL)
    comptime if S_REF > 0:
        comptime assert S_REF + S_TGT == S_Vp, "v2v: S_REF + S_TGT must equal S_Vp"
        comptime assert S_REF == RNFp * RNHp * RNWp, "v2v: S_REF must equal RNFp*RNHp*RNWp"
        # NO exact-multiple assert: musubi multiplies ref positions by the INTEGER
        # reference_downscale (ltx2_train_network.py:3402-3405), so ref×DS only
        # APPROXIMATELY co-locates on non-power-of-2 target extents (e.g. NH=9,
        # RNH=4, DS=2: ref reaches 8·32 vs target 9·32) — musubi-faithful.
        comptime assert DS_P >= 1, "v2v: reference_downscale (DS) must be >= 1"
    var ckpt = cfg.ltx2_checkpoint
    if len(ckpt) == 0:
        ckpt = serenity_checkpoint(String(DEFAULT_CKPT_NAME))
    if len(cfg.dataset_cache_dir) == 0:
        raise Error("train_ltx2_av: --dataset_cache_dir is required (musubi-native cache)")
    _ = sys_mkdirs(cfg.output_dir)

    print("[ltx2-video-lora] musubi-oracle trainer — geometry",
          "S_V=", S_Vp, "(", NFp, "x", NHp, "x", NWp, ") N_TXT=", N_TXT,
          "blocks=", NUM_LAYERS, "key=", lat_key)
    var items = _enumerate_cache(cfg.dataset_cache_dir, lat_key)
    if len(items) == 0:
        raise Error("train_ltx2_av: no cache samples matching the comptime geometry")
    print("  [cache]", len(items), "samples @", cfg.dataset_cache_dir)

    # ── IC-LoRA / v2v flag+config surface (P5 (d)) ──────────────────────────────
    # Each knob has an argv flag AND a config (cfg.levers.*) key; per the lead
    # ruling the CONFIG value WINS when set (argv stays for ad-hoc runs). All keys
    # default-off/identity so an absent config is C13 byte-identical.
    var ref_cache_dir = String("")
    for i in range(len(args)):
        if String(args[i]) == "--reference_cache_dir" and i + 1 < len(args):
            ref_cache_dir = String(args[i + 1])
    if cfg.levers.reference_cache_dir.byte_length() > 0:
        ref_cache_dir = cfg.levers.reference_cache_dir
    if ref_cache_dir.byte_length() > 0:
        var npaired = _pair_reference_cache(items, ref_cache_dir)
        print("  [v2v] paired", npaired, "reference latents @", ref_cache_dir)
    # P5.5 u2 inpaint: pair the per-sample mask cache (host-side same-basename).
    if cfg.levers.mask_cache_dir.byte_length() > 0:
        var nmask = _pair_mask_cache(items, cfg.levers.mask_cache_dir)
        print("  [inpaint] paired", nmask, "mask caches @", cfg.levers.mask_cache_dir)
    comptime if S_REF > 0:
        comptime if NFp > 1:
            if cfg.levers.mask_conditioning_p > Float32(0.0) \
               and cfg.levers.mask_cache_dir.byte_length() == 0:
                raise Error("train_ltx2_av: mask_conditioning_p>0 requires mask_cache_dir")
            # WARN (not clamp): a temporal_boundary that spans the whole target grid
            # makes prefix/suffix intrinsic conditioning degenerate to FULL
            # conditioning. Official flexible.py slice-clamps (tb>=NF masks every
            # token) — finding #2 confirmed that IS the upstream behavior, so we
            # mirror it (no clamp) and only surface it. Formula: tb·(NH·NW) >= S_TGT.
            # Gated on prefix/suffix actually being enabled — temporal_boundary is
            # only read there (lines 439-442); the default tb=8 would otherwise trip
            # the "degenerate" print on every default video_v2v run (NFp=4) with no
            # prefix/suffix conditioning active, which the message itself denies.
            if (cfg.levers.prefix_conditioning_p > Float32(0.0)
                or cfg.levers.suffix_conditioning_p > Float32(0.0)) \
               and cfg.levers.temporal_boundary * (NHp * NWp) >= S_TGT:
                print("[ltx2-v2v] WARN: temporal_boundary covers the whole grid — prefix/suffix degenerate to full conditioning")
    # first-frame conditioning probability (default 0.0 OFF; musubi 0.1). Video
    # arms only (comptime NFp>1 gates it).
    var first_frame_p = Float32(0.0)
    for i in range(len(args)):
        if String(args[i]) == "--ltx2_first_frame_conditioning_p" and i + 1 < len(args):
            first_frame_p = _parse_float32(String(args[i + 1]))
    if cfg.levers.first_frame_conditioning_p != Float32(0.0):
        first_frame_p = cfg.levers.first_frame_conditioning_p
    # sample-render reference latent (A2a).
    var sample_v2v_ref = String("")
    for i in range(len(args)):
        if String(args[i]) == "--sample_v2v_ref_latent" and i + 1 < len(args):
            sample_v2v_ref = String(args[i + 1])
    # validation references (A2b): the val cache's OWN reference_cache_directory.
    var val_ref_cache_dir = String("")
    for i in range(len(args)):
        if String(args[i]) == "--val_reference_cache_dir" and i + 1 < len(args):
            val_ref_cache_dir = String(args[i + 1])
    if cfg.levers.val_reference_cache_dir.byte_length() > 0:
        val_ref_cache_dir = cfg.levers.val_reference_cache_dir
    # ic_lora_strategy: config default "auto"; argv override; config WINS when set
    # to a non-"auto" value. "auto" -> infer from the LoRA preset (musubi
    # infer_ic_lora_strategy_from_preset). audio_ref_only_ic fail-loud (P6).
    var ic_strategy = String("auto")
    for i in range(len(args)):
        if String(args[i]) == "--ic_lora_strategy" and i + 1 < len(args):
            ic_strategy = String(args[i + 1])
    if cfg.levers.ic_lora_strategy != String("auto"):
        ic_strategy = cfg.levers.ic_lora_strategy
    if ic_strategy == String("auto"):
        ic_strategy = _infer_ic_lora_strategy(cfg.lora_target_preset)
    if ic_strategy == String("audio_ref_only_ic"):
        raise Error("train_ltx2_av: ic_lora_strategy=audio_ref_only_ic requires the AV arm (P6.2, not yet wired)")
    if ic_strategy != String("v2v") and ic_strategy != String("none"):
        raise Error(String("train_ltx2_av: ic_lora_strategy must be auto|none|v2v|audio_ref_only_ic; got '") + ic_strategy + "'")
    # P6.0: the audio (672) & audio_ref_only_ic (864) LoRA surfaces are DEFINED
    # (lora_surface.mojo, counts verified vs lora_ltx2.py:539-618) but the AV arm
    # that consumes them is not wired until P6.2. This trainer is video+v2v+
    # intrinsic only, and video_lora_names() emits ONLY the video surface, so
    # refuse the audio presets LOUDLY rather than silently training the wrong
    # (video) surface under an audio preset.
    if cfg.lora_target_preset == PRESET_AUDIO or cfg.lora_target_preset == PRESET_AUDIO_REF_ONLY_IC:
        raise Error("train_ltx2_av: LoRA preset audio/audio_ref_only_ic requires the AV arm (P6.2, not yet wired); this trainer is video/v2v only")
    comptime if S_REF > 0:
        if ic_strategy != String("v2v"):
            raise Error(String("train_ltx2_av: a v2v geometry needs ic_lora_strategy=v2v (or auto + a v2v preset); resolved '") + ic_strategy + "'")
        # reference_downscale is the CACHE-side integer downscale; it must match
        # the comptime co-location DS when explicitly set (>1).
        if cfg.levers.reference_downscale > 1 and cfg.levers.reference_downscale != DS:
            raise Error(String("train_ltx2_av: config reference_downscale ") + String(cfg.levers.reference_downscale)
                + " != geometry DS " + String(DS) + " (cache/geometry mismatch)")
        if ref_cache_dir.byte_length() == 0:
            raise Error("train_ltx2_av: v2v geometry requires --reference_cache_dir (or config reference_cache_dir)")
        if sample_v2v_ref.byte_length() == 0 and len(items) > 0:
            sample_v2v_ref = items[0].ref_lat_path
    else:
        if ic_strategy == String("v2v"):
            raise Error("train_ltx2_av: ic_lora_strategy=v2v requires a v2v --geometry (image512_v2v|video_v2v)")

    var ctx = DeviceContext()
    var model_cfg = LTX2Config.ltx2()

    # ── SPEED PASS speed flags (LTX2_FAST is now the DEFAULT — band gate PASSED
    #    n=100 vs the musubi oracle 2026-07-16, lead-authorized for production) ──
    #   DEFAULT (no flag) = LTX2_FAST: bf16 stack + device optimizer (~7.3 s/step,
    #                        ~20% vs the 9.4 F32-host baseline).
    #   LTX2_F32_STACK=1   = ESCAPE HATCH: F32 stack + host AdamW = the old default
    #                        (C13 byte-exact arm; anchors 0.6093/0.6901/0.4911/1.0092).
    #   LTX2_HOST_OPT=1    = bf16 stack + host optimizer (B-alone diagnostic; ~8.2s).
    #   LTX2_FAST=1        = explicit force (redundant with default; overrides
    #                        LTX2_HOST_OPT to guarantee the full fast combo).
    # A levers optimizer keeps its GATED F32 stack + host step (not silently moved
    # to the bf16 class); the F32 escape is implied for it.
    var f32_escape = env_int("LTX2_F32_STACK", 0) == 1
    var fast = env_int("LTX2_FAST", 0) == 1
    # levers optimizer active? (needed EARLY: a lever keeps its gated F32 stack +
    # host step, so bf16_stack/device_opt below both gate on it). MJ-1112 state
    # persists across steps (lazy-init step 1 / RESTORED from the optstate sidecar).
    var lever_opt = levers_optimizer_active(cfg.levers)
    var opt_state = LeversOptimizerState()

    # ── P4 grad-accumulation ─────────────────────────────────────────────────
    # accum_steps micro-steps SUM into one window, MEAN, then ONE optimizer step
    # (musubi-faithful for LoRA: bs1×accum-N grads == musubi's bs-N per-sample-
    # sigma stack — no cross-sample ops). max_steps counts UPDATES (total micros
    # = max_steps×accum). accum==1 -> window bypassed, BYTE-IDENTICAL to the
    # per-step path (sum-of-one/÷1, trainer_core:251-252). true bs>1 is DEFERRED
    # (B=1 comptime-baked in every stack/block/flash shape; see plan P4) — FAIL
    # LOUD up front (before the load) rather than silently ignore.
    var accum_steps = cfg.gradient_accumulation_steps
    if accum_steps < 1:
        accum_steps = 1
    # (batch_size>1 already failed loud in main() before any DeviceContext.)
    var accum_window = GradAccumWindow(accum_steps)
    if accum_steps > 1:
        print("  [accum] gradient_accumulation_steps=", accum_steps, "-> max_steps",
              cfg.max_steps, "UPDATES =", cfg.max_steps * accum_steps, "micros; optimizer/",
              "LR/cadence/save keyed on the UPDATE index")

    # SAVE-ACTS arm (LTX2_SAVE_ACTS=K, default 0=off): retain the first K blocks'
    # train-forward acts so the backward skips their recompute (bwd 4.05s ~= 2.7s
    # re-forward + 1.35s grad-math; saving K blocks removes (K/48)x2.7s). VRAM:
    # per-block bf16 acts ~40.9M elem (image512) -> ~1.31GB for K=16 (measured
    # A+B peak 22.06/24 -> ~1.5-2GB headroom, so K<=16 fits). Pure memory-for-
    # compute (forward already computes+discards these acts); BIT-IDENTICAL to the
    # recompute arm (saved acts == recompute acts, same code + same-step LoRA).
    # NOTE vs krea2 act-OFFLOAD NO-GO: that was PCIe host round-trips (2x slower);
    # this is KEEP-ON-DEVICE, a different class.
    var save_acts_k = env_int("LTX2_SAVE_ACTS", 0)
    if save_acts_k > NUM_LAYERS:
        save_acts_k = NUM_LAYERS
    if save_acts_k > 0:
        print("  [speed] save-acts K=", save_acts_k, "blocks retained (backward skips",
              save_acts_k, "recomputes; bf16 acts ~", save_acts_k, "x 82MB)")

    # Phase B bf16 stack carriers — NOW THE DEFAULT (band gate PASSED). Stack
    # loads bf16 (run_f32=False): fwd + recompute-bwd GEMMs run bf16 (rms_norm
    # still F32-accumulates — ops/norm.mojo:20-21, carrier-safe). LoRA masters stay
    # F32; the compute (side-GEMMs) is bf16. FORK-1 (lead-ruled): bf16-LoRA-compute
    # IS musubi's OWN class — musubi wraps the transformer in accelerator.autocast()
    # bf16 (hv:3846-3853; parity ref-gen replicates with torch.autocast bf16), and
    # LoRA wraps the Linears as org_forward + lora_up(lora_down(x))*scale
    # (lora.py:140-142), so under autocast the F32-master LoRA params run bf16 GEMMs
    # exactly as ours do. F32 stays the MASTER/optimizer dtype; bf16 = compute class
    # = the oracle's. OFF only under LTX2_F32_STACK=1 (escape) or a lever optimizer
    # (kept in its GATED F32 class until re-gated in bf16).
    var bf16_stack = (not f32_escape) and (not lever_opt)
    var stack_f32 = not bf16_stack
    var stack_dt = STDtype.F32 if stack_f32 else STDtype.BF16
    if bf16_stack:
        print("  [speed] bf16 stack carriers ON (DEFAULT; fwd+recompute-bwd GEMMs bf16;",
              "norms F32; LoRA F32-master / bf16-compute = musubi autocast class)")
    else:
        print("  [speed] F32 stack (", "LTX2_F32_STACK escape" if f32_escape else "lever optimizer", ")")

    # autograd_v2 engine backward (LTX2_V2_GRAPH=1): the stack backward runs
    # through the dependency-counted graph engine (coarse OPK_LTX2V_VIDEO_BLOCK,
    # apply arm calls ltx2_video_block_backward WHOLE) — BIT-IDENTICAL to the
    # hand-chain (per-block gate n_mismatch=0). Default OFF (C13, hand-chain).
    # Full-recompute only: incompatible with LTX2_SAVE_ACTS (fails loud in L2).
    var v2_graph = env_int("LTX2_V2_GRAPH", 0) == 1
    if v2_graph:
        print("  [engine] LTX2_V2_GRAPH ON — stack backward via the autograd_v2 graph engine")
    # LTX2_V2_SLAB=1 (rung 2): the graph engine backward runs through the StepSlab
    # + device-resident LoRA grad path — per-block mark/rewind, LoRA d_A/d_B stay
    # on device through the block loop, ONE boundary readback fills the same host
    # d_a/d_b lists (C12: byte-identical grads → anchors must match flag-OFF
    # digits). Default OFF (C13). Removes the per-block DtoH+sync round trip; the
    # launch-storm kill is rung 3 (CUDA-graph capture).
    var v2_slab = env_int("LTX2_V2_SLAB", 0) == 1
    if v2_slab:
        print("  [engine] LTX2_V2_SLAB ON — StepSlab + device-resident LoRA grads (boundary readback)")
    # FAIL LOUD (rung 3): the resident grad store OVERWRITES per micro-step
    # (accum=1 only). Device add-into across micro-steps is a separate gated rung;
    # never silently mix device-overwrite with P4's host-side accumulation.
    if v2_slab and accum_steps > 1:
        raise Error(
            String("LTX2_V2_SLAB=1 is accum=1 ONLY (the resident grad store overwrites")
            + " per micro-step); gradient_accumulation_steps=" + String(accum_steps)
            + " > 1 would silently corrupt grads. Unset LTX2_V2_SLAB or set accum=1.")

    # LTX2_V2_CAPTURE=1 (rung 3): per-block CUDA-graph capture/replay — the
    # launch-storm kill. REQUIRES LTX2_V2_SLAB semantics (persistent StepSlab +
    # resident grad store) and the bf16 stack: the standalone weight stage returns
    # bf16 views the captured graph reads IN PLACE, so an F32 stack (which clones
    # via to_f32) would break the fixed-address refill. Fails LOUD rather than
    # silently falling back. accum=1 is already enforced by the v2_slab guard
    # above. Default OFF (C13). LTX2_RESIDENT_BLOCKS (default 39 under capture,
    # below) budgets VRAM for the extra stage.
    var v2_capture = env_int("LTX2_V2_CAPTURE", 0) == 1
    if v2_capture:
        if not v2_slab:
            raise Error(
                "LTX2_V2_CAPTURE=1 requires LTX2_V2_SLAB=1 (persistent slab +"
                " resident grad store). Set LTX2_V2_SLAB=1 or unset LTX2_V2_CAPTURE.")
        if stack_f32:
            raise Error(
                "LTX2_V2_CAPTURE=1 requires the bf16 stack (the standalone weight"
                " stage is bf16); an F32 stack (LTX2_F32_STACK / lever optimizer)"
                " is not supported under capture.")
        print("  [engine] LTX2_V2_CAPTURE ON — per-block CUDA-graph capture/replay"
              " (warmup step0, capture step1, replay step>=2)")

    # W1 (skeptic): the head/tail/blocks load (29GB fp8 + 17GB resident) is
    # DEFERRED to AFTER the resume block below, so a doomed resume (bad state /
    # dim mismatch / missing optstate sidecar / seed-guard) fails LOUD up front
    # without paying the load. Nothing between here and the loop uses the model
    # except the loop itself (verified: no src/head/tail refs in masters/resume).

    # slot count is preset-derived (t2v 8 / v2v 10). mods drives EVERY slot loop.
    var mods = _module_paths(cfg.lora_target_preset)
    var n_slots = len(mods)
    var masters = _make_video_lora_set(cfg.lora_rank, cfg.seed, mods)
    var lora_scale = cfg.lora_alpha / Float32(cfg.lora_rank)
    print("  [lora] rank", cfg.lora_rank, "alpha", cfg.lora_alpha,
          "preset", cfg.lora_target_preset, "->", NUM_LAYERS * n_slots,
          "adapters (", n_slots, "slots/block; A kaiming, B=0, F32 masters), scale", lora_scale)

    # ── SPEED PASS Phase A: device-resident optimizer — NOW THE DEFAULT ──────
    # device_opt = the fused-device AdamW path (resident F32 LoRA + moments;
    # host-sync only at cadence). ON by default; OFF under LTX2_F32_STACK=1
    # (escape), a lever optimizer (device fusion out of scope), or LTX2_HOST_OPT=1
    # (the B-alone diagnostic — bf16 stack + host AdamW). MEASURED: device-opt
    # ALONE off bf16 is a WASH (resident F32 moments make fwd/bwd memory-bound —
    # the krea2 devgrad-peak class [[project_krea2_devgrad_lora_peak]]), but WITH
    # bf16 (the default) it wins: bf16 halves activations, freeing the headroom
    # (A+B ~20%, see project_ltx2_speed_pass). LTX2_FAST=1 forces it even past
    # LTX2_HOST_OPT.
    var device_opt = (not lever_opt) and (not f32_escape) and (
        fast or (env_int("LTX2_HOST_OPT", 0) != 1))

    # ── resume (F32-exact masters+moments preferred; bf16-master fallback) ────
    var start_step = 1
    if len(cfg.resume_from) > 0:
        var rf = cfg.resume_from
        var prefixes = _all_prefixes(mods)
        var probe_prefix = prefixes[0].copy()

        # ltx2's state sidecar is <stem>.state.safetensors — a REPLACE of the
        # PEFT `.safetensors`, NOT krea2's append. Reduce whatever the user
        # passed (PEFT weights, the state file, or a bare stem) to <stem>, then
        # let the shared trainer_resolve_resume_path locate the moments sidecar.
        var stem = rf.copy()
        if rf.endswith(".state.safetensors"):
            stem = _strip_suffix(rf, String(".state.safetensors"))
        elif rf.endswith(".safetensors"):
            stem = _strip_suffix(rf, String(".safetensors"))
        var expected_state = stem + ".state.safetensors"
        var state_path = trainer_resolve_resume_path(
            stem, probe_prefix, String(".state.safetensors"))

        # read meta first (also our open-probe): fail loud naming the tried path.
        var meta = List[Float32]()
        try:
            meta = load_lora_train_state_meta(state_path, ctx)
        except:
            raise Error(
                String("train_ltx2_av: --resume could not open state file. tried '")
                + expected_state + "' (from --resume_from '" + rf + "')")
        if len(meta) < 2:
            raise Error("train_ltx2_av: state has no __meta__ (step,seed) — cannot infer start step")
        var saved_step = Int(meta[0])
        if saved_step >= cfg.max_steps:
            raise Error(
                String("train_ltx2_av: nothing to resume: state step ")
                + String(saved_step) + " >= max_steps " + String(cfg.max_steps))

        # resume-scope guard. 5-field meta -> shared trainer_core guard (seed +
        # dataset size/identity + token layout + step). Legacy 2-field meta ->
        # minimal seed check (step,seed only).
        if len(meta) >= 5:
            trainer_resume_meta_guard(
                String("ltx2"), String("S_V"), S_Vp, state_path, True,
                saved_step, cfg.seed, len(items), cfg.dataset_cache_dir, ctx)
        else:
            if UInt64(Int(meta[1])) != cfg.seed:
                print("=====================================================================")
                print("[ltx2-resume] WARNING: legacy state seed", Int(meta[1]), "!= config seed", cfg.seed)
                print("[ltx2-resume] the (seed,step)-derived sigma / noise / sample-pick streams")
                print("[ltx2-resume] will NOT match the original continuous run — the loss")
                print("[ltx2-resume] trajectory and samples will diverge from an uninterrupted run.")
                print("=====================================================================")

        if lora_train_state_has_f32_masters(state_path, prefixes[0]):
            var states = load_lora_train_state_f32(prefixes, state_path, ctx)
            if len(states) != NUM_LAYERS * n_slots:
                raise Error(String("train_ltx2_av: resume expected ")
                            + String(NUM_LAYERS * n_slots) + " states, got "
                            + String(len(states)))
            for i in range(len(states)):
                if states[i].rank != cfg.lora_rank:
                    raise Error(String("train_ltx2_av: resume rank ")
                        + String(states[i].rank) + " != cfg.lora_rank "
                        + String(cfg.lora_rank) + " at adapter " + String(i))
                var edims = _slot_dims(mods[i % n_slots])
                if states[i].in_f != edims[0] or states[i].out_f != edims[1]:
                    raise Error(String("train_ltx2_av: resume dims ")
                        + String(states[i].in_f) + "x" + String(states[i].out_f)
                        + " != " + String(edims[0]) + "x" + String(edims[1])
                        + " (wrong-model/preset state?) at adapter " + String(i))
                masters[i].a = states[i].a.copy()
                masters[i].b = states[i].b.copy()
                masters[i].ma = states[i].ma.copy()
                masters[i].va = states[i].va.copy()
                masters[i].mb = states[i].mb.copy()
                masters[i].vb = states[i].vb.copy()
            print("[ltx2-resume] F32-exact masters+moments from", state_path, "@ step", saved_step)
        else:
            # NOTE: NOT trainer_warn_warm_resume — that banner is about ZEROED
            # AdamW moments (krea2's warm case). Our old-era state DOES carry
            # moments; only the MASTERS were bf16-rounded on the prior save.
            var warm = load_lora_train_state(prefixes, lora_scale, state_path, ctx)
            if len(warm) != NUM_LAYERS * n_slots:
                raise Error(String("train_ltx2_av: resume expected ")
                            + String(NUM_LAYERS * n_slots) + " adapters, got "
                            + String(len(warm)))
            for i in range(len(warm)):
                if warm[i].adapter.rank != cfg.lora_rank:
                    raise Error(String("train_ltx2_av: resume rank ")
                        + String(warm[i].adapter.rank) + " != cfg.lora_rank "
                        + String(cfg.lora_rank) + " at adapter " + String(i))
                var wdims = _slot_dims(mods[i % n_slots])
                if warm[i].adapter.in_f != wdims[0] or warm[i].adapter.out_f != wdims[1]:
                    raise Error(String("train_ltx2_av: resume dims ")
                        + String(warm[i].adapter.in_f) + "x" + String(warm[i].adapter.out_f)
                        + " != " + String(wdims[0]) + "x" + String(wdims[1])
                        + " (wrong-model/preset state?) at adapter " + String(i))
                masters[i].a = _bf16_to_f32_list(warm[i].adapter.a)
                masters[i].b = _bf16_to_f32_list(warm[i].adapter.b)
                masters[i].ma = warm[i].adapter.ma.copy()
                masters[i].va = warm[i].adapter.va.copy()
                masters[i].mb = warm[i].adapter.mb.copy()
                masters[i].vb = warm[i].adapter.vb.copy()
            print("=====================================================================")
            print("[ltx2-resume] WARM (bf16-rounded masters — old-era state file)")
            print("[ltx2-resume]  ", state_path, "@ step", saved_step)
            print("[ltx2-resume] AdamW moments ARE restored; only the MASTERS were bf16-rounded")
            print("[ltx2-resume] on the previous save (MJ-1108 class). Resume continues from")
            print("[ltx2-resume] bf16-quantized A/B — not bit-identical to a continuous run.")
            print("=====================================================================")

        # ── levers optimizer state (P3.3): restore the .optstate sidecar ─────
        # Masters came from the .state above; the LeversOptimizerState (moments/
        # ctl/rng/sign-history) lives in <stem>.optstate.safetensors. The load
        # fails loud (missing file / tag / count / version) — that IS the
        # "no sidecar" guard for a lever optimizer resuming off a plain .state.
        if lever_opt:
            var optstate_path = levers_optimizer_sidecar_path(stem)
            var ores = levers_optimizer_sidecar_load(
                optstate_path, cfg.levers.optimizer, NUM_LAYERS * n_slots * 2)
            if ores.k != saved_step:
                raise Error(
                    String("train_ltx2_av: optstate k ") + String(ores.k)
                    + " != .state step " + String(saved_step) + " (mismatched save)")
            opt_state = ores^.take_state()
            print("[ltx2-resume] levers optimizer state restored from", optstate_path,
                  "@ step", saved_step)
        # saved_step is the UPDATE index; resume at the first MICRO of the next
        # window. accum==1 -> saved_step+1 (unchanged). Saves are boundary-aligned
        # (cadence fires on _opt_idx), so resume always lands on a window start.
        start_step = saved_step * accum_steps + 1

    # ── head/tail/blocks load (W1: DEFERRED past the resume fail-loud above) ──
    print("  [load] head/tail/blocks from", ckpt)
    var head = LTX2VideoStackHead.load(ckpt, ctx)
    var tail = LTX2VideoTail.load(ckpt, stack_f32, ctx)
    # SquareQ W4A16 residency arm (env LTX2_SQUAREQ_SLAB=<slab dir>, mirrors the
    # inference pipeline's LTX2_INT4_SLAB precedent at pipeline/ltx2_t2v_av_hq.
    # mojo:5106): blocks reconstruct dense BF16 from the packed slab per visit
    # (ops/squareq, auto-detected inside LTX2Int4BlockStream by the absent
    # `.smooth` key); head/tail/w0 keep loading from the checkpoint, which MUST
    # be the slab's build source (see <slab>/build-state.json) so passthrough
    # tensors match. Fail LOUD if the slab does not open.
    var squareq_slab = env_or("LTX2_SQUAREQ_SLAB", String(""))
    var src: LTX2VideoBlockSource
    if len(squareq_slab) > 0:
        try:
            src = LTX2VideoBlockSource.open_int4(
                ckpt, squareq_slab, model_cfg, stack_f32)
        except err:
            raise Error(
                String("train_ltx2_av: LTX2_SQUAREQ_SLAB='") + squareq_slab
                + "' failed to open: " + String(err))
        print("  [squareq] W4A16 base ON — blocks reconstruct BF16 from",
              squareq_slab)
    else:
        src = LTX2VideoBlockSource.open(ckpt, model_cfg, stack_f32)
    if src.block_count() != NUM_LAYERS:
        raise Error(String("checkpoint block_count != 48: ") + String(src.block_count()))

    # ── fp8-RESIDENT block store (speed): streamed get_block measured 29.2s
    #    x2 per step = the whole 58s; resident materialize is 0.76s/48 (probe
    #    ltx2_stream_cost_probe.mojo 2026-07-09). Full 48-block residency
    #    (20.1GB) OOMs next to the F32 working set on 24GB, so the DEFAULT is a
    #    RANGE: blocks 0..41 resident (~17.6GB), tail 6 streamed through the
    #    de-synced streamed path (one fence/block instead of ~86). Residency
    #    changes NO math — anchor digits must stay IDENTICAL
    #    (video 4-step set re-baselined 2026-07-16 late, post rope-fix MJ-1109:
    #    0.4602/0.2296/0.2224/0.2616 — the 0.4560/... set was the pre-rope-fix
    #    era; see configs/ltx2_video_anchor.json). Tune with --resident_blocks
    #    N (0=off).
    # Under LTX2_V2_CAPTURE the persistent capture stage (standalone weight stage
    # ~0.7GB + fixed LoRA + staging + A/B + 2GB slab + resident grad store) eats
    # into the F32 working set headroom, so the resident-fp8 block count defaults
    # to LTX2_RESIDENT_BLOCKS (39; measured fit) instead of 42. --resident_blocks
    # still overrides.
    var n_resident = 42
    if v2_capture:
        n_resident = env_int("LTX2_RESIDENT_BLOCKS", 39)
    var resident_explicit = False
    for i in range(len(args)):
        if String(args[i]) == "--resident_blocks" and i + 1 < len(args):
            n_resident = Int(_parse_int_arg(String(args[i + 1])))
            resident_explicit = True
    if n_resident > NUM_LAYERS:
        n_resident = NUM_LAYERS
    # SquareQ arm: fp8 residency stages the fp8 CHECKPOINT's raw bytes, but the
    # blocks come from the packed slab — there is nothing to park. The implicit
    # default (42) just switches off; an EXPLICIT --resident_blocks > 0 is a
    # contradiction and fails loud.
    if len(squareq_slab) > 0 and n_resident > 0:
        if resident_explicit:
            raise Error(
                "train_ltx2_av: --resident_blocks > 0 is the fp8 residency"
                " knob; it does not combine with LTX2_SQUAREQ_SLAB (blocks"
                " reconstruct from the packed slab). Pass --resident_blocks 0."
            )
        n_resident = 0
        print("  [squareq] fp8 residency default OFF — blocks reconstruct from"
              " the packed slab per visit")
    if n_resident > 0:
        # PRE-ALLOC over-commit guard: enable_fp8_resident_range below allocates
        # n_resident × ~405 MiB of fp8 up front; an over-committed --resident_blocks
        # / LTX2_RESIDENT_BLOCKS otherwise dies with a GENERIC CUDA OOM in this load
        # phase, BEFORE the post-alloc headroom check (create() at the capture
        # stage). Only engage PAST the measured-fit default (39 capture / 42
        # non-capture) so the known-good default can never false-positive; then
        # byte-math resident vs (free − the post-residency working-set reserve) and
        # fail loud naming the flag. NOTE: the ~405 MiB/block and the reserve are
        # calibrated estimates (20.1GB/48≈419MB; reserve from the capture-stage
        # comment) — not a live measurement; they gate over-commit, the post-alloc
        # check still guards the exact fit.
        var _fit = 39 if v2_capture else 42
        if n_resident > _fit:
            var _blk = 405 * 1024 * 1024                       # ~405 MiB per fp8 block
            var _resident_est = n_resident * _blk
            var _reserve = (3 + (2 if v2_capture else 0)) * 1024 * 1024 * 1024
            var _free = Int(ctx.get_memory_info()[0])
            if _resident_est > _free - _reserve:
                raise Error(String("train_ltx2_av: resident_blocks=") + String(n_resident)
                    + " over-commits VRAM — ~" + String(_resident_est // (1024 * 1024 * 1024))
                    + " GiB of fp8 residency needs more than the ~"
                    + String((_free - _reserve) // (1024 * 1024 * 1024))
                    + " GiB free after the working-set reserve. Lower --resident_blocks"
                    + " / LTX2_RESIDENT_BLOCKS (measured fit: " + String(_fit) + ").")
        var t_res = perf_counter_ns()
        src.stream.enable_fp8_resident_range(0, n_resident - 1, ctx)
        print("  [resident] blocks 0..", n_resident - 1, "raw fp8 in VRAM:",
              Float64(src.stream.resident_bytes()) / 1.0e9, "GB in",
              Float64(perf_counter_ns() - t_res) / 1.0e9, "s;",
              NUM_LAYERS - n_resident, "blocks streamed (de-synced)")

    # ── caption-dropout lever (MJ-1113) resident zero-embeds buffer ──────────
    # Built ONCE (only when the lever is on) and reused: on a drop step we upload
    # a same-shape ZERO bf16 `enc`. Default caption_dropout_prob=0.0 -> never
    # drawn -> no buffer, byte-identical (levers.mojo:100-106 default-off).
    var cap_drop_on = cfg.levers.caption_dropout_prob > Float32(0.0)
    var enc_zeros_host = List[BFloat16]()
    if cap_drop_on:
        for _ in range(N_TXT * VD):
            enc_zeros_host.append(BFloat16(0.0))

    # ── val_loss (P2) setup ──────────────────────────────────────────────────
    # DETERMINISTIC per-val-index sigma+noise on VAL-ISOLATED streams: sigma from
    # _sample_sigma with a val-namespaced seed (cfg.seed*10007+1) keyed on the val
    # index; noise seeded cfg.seed*2000003 + index*104729 (distinct multipliers
    # from the training sigma/noise streams -> decorrelated + never colliding).
    # This is a DOCUMENTED DEVIATION from musubi, whose val noise+sigma are FRESH
    # UNSEEDED global-RNG draws that even ADVANCE the training RNG (hv:3027/3029,
    # no save/restore guard) — deterministic-by-index is strictly better for
    # overfit detection; the loss FORMULA/mean/cadence are musubi-exact
    # (hv:3155-3167). C13: off unless BOTH --val_cache_dir and --validate_every.
    var val_on = len(cfg.val_cache_dir) > 0 and cfg.validate_every > 0

    # v2v (P5): sampling (A2a) + validation (A2b) are both wired. v2v validation
    # requires the val cache's OWN references (--val_reference_cache_dir) ONLY when
    # a validation cadence actually LANDS in this run's remaining update range — a
    # nonzero default that never fires within max_steps must NOT demand it.
    # start_step is micro-space (saved_step*accum+1); recover the resumed UPDATE.
    comptime if S_REF > 0:
        var resume_update = (start_step - 1) // accum_steps
        if val_on and _cadence_reached(cfg.validate_every, resume_update, cfg.max_steps) \
           and val_ref_cache_dir.byte_length() == 0:
            raise Error(
                "train_ltx2_av: v2v validation requires --val_reference_cache_dir"
                " (the val cache's own downscaled references)")

    var val_items = List[CacheItem]()
    var vcfg = cfg.copy()
    if val_on:
        vcfg.seed = cfg.seed * UInt64(10007) + UInt64(1)
        val_items = _enumerate_cache(cfg.val_cache_dir, lat_key)
        if len(val_items) == 0:
            raise Error("train_ltx2_av: --val_cache_dir has no samples matching the geometry")
        print("  [val] enabled: every", cfg.validate_every, "steps over",
              len(val_items), "samples @", cfg.val_cache_dir)
        # v2v (A2b): pair the val cache's own downscaled references.
        comptime if S_REF > 0:
            if val_ref_cache_dir.byte_length() > 0:
                var nvpaired = _pair_reference_cache(val_items, val_ref_cache_dir)
                print("  [v2v] paired", nvpaired, "VAL reference latents @", val_ref_cache_dir)

    # ── SPEED PASS Phase A: resident device LoRA + AdamW moments ─────────────
    # Uploaded ONCE (F32). Per step the forward/backward READ these resident
    # tensors and fused_adamw_step mutates the SAME device buffers + moments in
    # place — no per-step host->device upload, no host AdamW, no host clip loop.
    # ArcPointer .copy() shares the buffer, so the in-place kernel write IS the
    # next step's LoRA weight. host masters are refreshed (to_host) only at
    # save/sample/val cadence. device_opt False (default / lever) -> these stay
    # empty and the host path (C13) runs unchanged.
    var res_a = List[ArcPointer[Tensor]]()
    var res_b = List[ArcPointer[Tensor]]()
    var res_ma = List[ArcPointer[Tensor]]()
    var res_va = List[ArcPointer[Tensor]]()
    var res_mb = List[ArcPointer[Tensor]]()
    var res_vb = List[ArcPointer[Tensor]]()
    if device_opt:
        var dd = _lora_to_device(masters, cfg.lora_rank, mods, ctx)
        res_a = dd[0].copy()
        res_b = dd[1].copy()
        for i in range(len(masters)):
            var vd = _slot_dims(mods[i % n_slots])
            res_ma.append(ArcPointer[Tensor](Tensor.from_host(
                masters[i].ma.copy(), [cfg.lora_rank, vd[0]], STDtype.F32, ctx)))
            res_va.append(ArcPointer[Tensor](Tensor.from_host(
                masters[i].va.copy(), [cfg.lora_rank, vd[0]], STDtype.F32, ctx)))
            res_mb.append(ArcPointer[Tensor](Tensor.from_host(
                masters[i].mb.copy(), [vd[1], cfg.lora_rank], STDtype.F32, ctx)))
            res_vb.append(ArcPointer[Tensor](Tensor.from_host(
                masters[i].vb.copy(), [vd[1], cfg.lora_rank], STDtype.F32, ctx)))
        print("  [speed] device-resident LoRA + fused device AdamW ON (opt-in",
              "LTX2_DEVICE_OPT=1;", len(masters), "adapters; unset for host C13 path)")

    # ── LTX2_V2_CAPTURE stage (persistent, ONE per run) ──────────────────────
    # Sized from the fixed geometry (S_Vp/N_TXT/VD/heads) + the per-slot LoRA
    # shapes (_slot_dims) + the compute dtype (bf16 stack). la_proto/lb_proto are
    # shape-only reference adapters. VRAM fail-loud after all stage allocations.
    var cap_stage = Optional[LTX2CaptureStage](None)
    if v2_capture:
        var lora_dt = STDtype.BF16 if bf16_stack else stack_dt
        var la_proto = List[ArcPointer[Tensor]]()
        var lb_proto = List[ArcPointer[Tensor]]()
        for s in range(n_slots):
            var sd = _slot_dims(mods[s])   # (in, out)
            la_proto.append(ArcPointer[Tensor](Tensor.from_host(
                _zeros(cfg.lora_rank * sd[0]), [cfg.lora_rank, sd[0]], lora_dt, ctx)))
            lb_proto.append(ArcPointer[Tensor](Tensor.from_host(
                _zeros(sd[1] * cfg.lora_rank), [sd[1], cfg.lora_rank], lora_dt, ctx)))
        var _bsz = stack_dt.byte_size()
        var _vcos_nb = S_Vp * src.cfg.num_heads * (src.cfg.head_dim // 2) * _bsz
        # create() does the post-alloc VRAM fail-loud (<512 MiB free -> raise).
        cap_stage = Optional[LTX2CaptureStage](LTX2CaptureStage.create(
            src, video_lora_names(cfg.lora_target_preset), la_proto, lb_proto,
            NUM_LAYERS, lora_scale,
            N_TXT * VD * _bsz,          # enc        [1, N_TXT, VD]
            S_Vp * 9 * VD * _bsz,       # v_temb     [1, S_V, 9*VD]
            N_TXT * 2 * VD * _bsz,      # v_prompt_ts[1, N_TXT, 2*VD]
            _vcos_nb,                   # v_cos      [S_V*heads, head_dim/2]
            _vcos_nb,                   # v_sin
            S_Vp * VD * _bsz,           # d_hidden / block_input [1, S_V, VD]
            2 * 1024 * 1024 * 1024,     # slab bytes (== _slab path)
            ctx))

    var t_start = perf_counter_ns()

    # --timing: per-section wall breakdown (ctx.synchronize() at each boundary
    # so GPU work is attributed to its section — sync itself perturbs overlap,
    # so compare the section SUM against an untimed run's s/step before trusting
    # ratios; see the krea2 ctx-sync timer-artifact lesson).
    var timing = _has_arg(args, "--timing")

    var step = start_step
    while step <= cfg.max_steps * accum_steps:   # step = MICRO; max_steps = UPDATES
        var t0 = perf_counter_ns()

        # ── sample pick (seeded, uniform over the cache) ─────────────────────
        var idx = Int(_splitmix64(cfg.seed * UInt64(31) + UInt64(step)) % UInt64(len(items)))
        ref item = items[idx]
        assert_prompt_mask_all_ones(item.te_path)
        var latent5 = reshape(
            _load_bf16(item.lat_path, lat_key, ctx), _sh5(1, C, NFp, NHp, NWp), ctx)
        var enc = reshape(
            _load_bf16(item.te_path, String(ENC_KEY), ctx), _sh3(1, N_TXT, VD), ctx)

        # ── caption-dropout lever (MJ-1113) ─────────────────────────────────
        # musubi drops a caption by ZEROING the cached video_prompt_embeds row
        # AND collapsing the text mask to idx-0-only; the mask collapse is PROVEN
        # forward-IRRELEVANT once the embeds are zeroed (bit-identical in musubi's
        # own runtime — output/ltx2_dropout_probe/probe.log). So we replace `enc`
        # with a same-shape zero bf16 tensor and leave the mask all-ones
        # (assert_prompt_mask_all_ones stays). DETERMINISM: the drop pick stream is
        # seed*31+step (levers.mojo:100) — musubi's CPython-MT19937 PER-SAMPLE
        # dropout ORDER is NOT reproduced (distribution-class, same treatment as
        # our sigma/noise/sample streams — intake doc LEVERS CONTRACT).
        if cap_drop_on and caption_dropout_pick(
            UInt64(step), cfg.seed, cfg.levers.caption_dropout_prob
        ):
            enc = Tensor.from_host_bf16(
                enc_zeros_host.copy(), _sh3(1, N_TXT, VD), ctx)

        if timing:
            ctx.synchronize()
        var tm_load = perf_counter_ns()

        # ── musubi flow-match noising (F32) ─────────────────────────────────
        # sigma keyed on the TARGET-grid token count (== S_Vp on the t2v path);
        # v2v noises ONLY the target latent, ref flows clean (lead ruling 2).
        var sigma = _sample_sigma(cfg, step, S_TGT)
        var latf = cast_tensor(latent5, STDtype.F32, ctx)
        var noise = randn(_sh5(1, C, NFp, NHp, NWp), cfg.seed + UInt64(step) * UInt64(104729), STDtype.F32, ctx)
        # noisy = (1-sigma)*latent + sigma*noise ; target = noise - latent
        var noisy = add(
            mul_scalar(latf, Float32(1.0) - sigma, ctx),
            mul_scalar(noise, sigma, ctx), ctx)
        var target5 = sub(noise, latf, ctx)

        # ── P5.5 intrinsic conditioning (VIDEO v2v arms) ──────────────────────
        # ONE per-token mask = UNION of the enabled conditions (first-frame +
        # prefix/suffix/spatial-crop), each drawn on its own seed base. It drives
        # the three surfaces: clean-replace (here), ts=0 (head), loss-off (loss) —
        # official LTX-2 flexible.py:_apply_intrinsic_condition:513-517. target5
        # (velocity) is UNCHANGED. cond_mask spans to the head + loss below.
        var cond_mask = List[Bool]()
        comptime if S_REF > 0:
            for _ in range(S_TGT):
                cond_mask.append(False)
            comptime if NFp > 1:
                cond_mask = _build_intrinsic_mask[NFp, NHp, NWp](cfg, first_frame_p, step)
                # INPAINT (P5.5 u2): union the per-sample cached binary mask when
                # drawn (own seed base). load_mask_tokens thresholds >0.5 + token
                # order frame->h->w (official flexible.py:506).
                if item.has_mask() and _cond_fired(
                    cfg.seed, step, cfg.levers.mask_conditioning_p, _MASK_BASE):
                    union_into(cond_mask, load_mask_tokens(item.mask_path, NFp, NHp, NWp))
                if _any_true(cond_mask):
                    # noisy = mask5*clean + (1-mask5)*noisy (per-token clean-replace;
                    # subsumes the frame-0 concat, bit-identical for first-frame).
                    var mask5 = _mask_to_5d_f32[NFp, NHp, NWp](cond_mask, False, ctx)
                    var inv5 = _mask_to_5d_f32[NFp, NHp, NWp](cond_mask, True, ctx)
                    noisy = add(mul(mask5, latf, ctx), mul(inv5, noisy, ctx), ctx)

        if timing:
            ctx.synchronize()
        var tm_noise = perf_counter_ns()

        # ── head (BF16, musubi network dtype) -> F32 stack inputs ────────────
        var noisy_bf16 = cast_tensor(noisy, STDtype.BF16, ctx)
        var ho: LTX2VideoHeadOut
        comptime if S_REF > 0:
            # v2v: patch-embed CLEAN reference + noisy target, ref-prepend concat,
            # per-token ts [0×S_REF, σ×S_TGT], two-grid rope (lead ruling 2/3).
            var ref_latf = cast_tensor(
                reshape(_load_ref_latent(item, ctx), _sh5(1, C, RNFp, RNHp, RNWp), ctx),
                STDtype.BF16, ctx)
            # the combined mask's conditioned target tokens carry ts=0 (clean).
            ho = head.forward_v2v[S_Vp, S_REF, S_TGT, N_TXT](
                ref_latf, noisy_bf16, enc, sigma,
                NFp, NHp, NWp, RNFp, RNHp, RNWp, DS, cond_mask, FRAME_RATE, ctx)
        else:
            ho = head.forward[S_Vp, N_TXT](noisy_bf16, enc, sigma, NFp, NHp, NWp, FRAME_RATE, ctx)
        # stack_dt = F32 (default, C13) or BF16 (LTX2_BF16_STACK). The stack GEMMs
        # follow the block-weight dtype (bf16 when run_f32=False); feeding bf16
        # carriers here keeps the whole stack in musubi's bf16-autocast class.
        var hidden = cast_tensor(ho.hidden, stack_dt, ctx)
        var v_temb = cast_tensor(ho.v_temb, stack_dt, ctx)
        var v_embedded = cast_tensor(ho.v_embedded, stack_dt, ctx)
        var v_prompt_ts = cast_tensor(ho.v_prompt_ts, stack_dt, ctx)
        var v_cos = cast_tensor(ho.v_cos, stack_dt, ctx)
        var v_sin = cast_tensor(ho.v_sin, stack_dt, ctx)
        var encf = cast_tensor(enc, stack_dt, ctx)

        if timing:
            ctx.synchronize()
        var tm_head = perf_counter_ns()

        # ── device LoRA: RESIDENT (device_opt, no upload) or per-step rebuild ─
        # device_opt: fwd/bwd read the resident tensors (la_use/lb_use share
        # buffers via ArcPointer.copy) — kills the per-step upload. host path:
        # rebuild from masters each step (C13). In the bf16 stack the side-GEMM
        # activation is bf16, so the LoRA tensors fed to fwd/bwd must be bf16
        # (downcast of the F32 resident/master — the F32 master stays the source
        # of truth; grads return through the block's F32 host lists).
        var la_use = List[ArcPointer[Tensor]]()
        var lb_use = List[ArcPointer[Tensor]]()
        if device_opt:
            if bf16_stack:
                for i in range(len(res_a)):
                    la_use.append(ArcPointer[Tensor](cast_tensor(res_a[i][], STDtype.BF16, ctx)))
                    lb_use.append(ArcPointer[Tensor](cast_tensor(res_b[i][], STDtype.BF16, ctx)))
            else:
                la_use = res_a.copy()
                lb_use = res_b.copy()
        else:
            var dev = _lora_to_device(masters, cfg.lora_rank, mods, ctx, stack_dt)
            la_use = dev[0].copy()
            lb_use = dev[1].copy()

        if timing:
            ctx.synchronize()
        var tm_loraup = perf_counter_ns()

        # ── stack forward (save-acts: retain first K blocks' acts) ────────────
        var fwd = ltx2_video_stack_lora_forward[S_Vp, N_TXT](
            hidden, encf, v_temb, v_embedded, v_prompt_ts, v_cos, v_sin,
            tail, src, la_use, lb_use, lora_scale, NUM_LAYERS, EPS, ctx,
            cfg.lora_target_preset, save_acts_k,
        )

        if timing:
            ctx.synchronize()
        var tm_fwd = perf_counter_ns()

        # ── loss + d_pred (host F32; 73,728 elems) ───────────────────────────
        # LOSS SEAM: config-JSON lever loss (mae|huber|smooth_l1|min-SNR) when
        # active, else the LEGACY inline MSE kept BIT-IDENTICAL to the pre-levers
        # block. video_loss_weight scales the REPORTED loss on both arms (the
        # legacy block never scaled the grad; kept that convention — weight
        # defaults 1.0 so the grad path is unchanged. See report note).
        var predh = fwd.pred.to_host(ctx)
        var loss = Float32(0.0)
        var dp = List[Float32]()
        comptime if S_REF > 0:
            # v2v (lead ruling 4): SLICE the ref off + target-only masked MSE;
            # d_block = [zeros_ref ; dL/d(target)] at [1,S_COMB,C] — every backward
            # arm consumes it UNCHANGED. loss_mask &= !cond_mask (P5.5): the
            # intrinsic-conditioned target tokens are DROPPED from the loss
            # (official flexible.py:517). cond_mask is all-False for image512
            # (NFp==1) so vmask stays all-True there.
            var targh = _patchify_tokens(target5, S_TGT, ctx).to_host(ctx)
            var vmask = List[Bool]()
            for t in range(S_TGT):
                vmask.append(not cond_mask[t])
            var lc = v2v_target_cotangent(predh, targh, vmask, S_REF, S_TGT, C)
            loss = lc[0] * cfg.video_loss_weight
            dp = lc[1].copy()
        else:
            var targh = _patchify_tokens(target5, S_Vp, ctx).to_host(ctx)
            if len(predh) != len(targh):
                raise Error("pred/target numel mismatch")
            var n_el = len(predh)
            if levers_loss_active(cfg.levers):
                var lg = levers_loss_grad(predh, targh, sigma, cfg.levers)
                loss = lg.loss * cfg.video_loss_weight
                dp = lg.d_pred.copy()
            else:
                var loss64 = 0.0
                for i in range(n_el):
                    var d = Float64(predh[i]) - Float64(targh[i])
                    loss64 += d * d
                    dp.append(Float32(2.0 * d / Float64(n_el)))
                loss = Float32(loss64 / Float64(n_el)) * cfg.video_loss_weight
        var d_pred = Tensor.from_host(dp^, _sh3(1, S_Vp, C), STDtype.F32, ctx)

        if timing:
            ctx.synchronize()
        var tm_loss = perf_counter_ns()

        # ── stack backward: autograd_v2 graph engine (LTX2_V2_GRAPH) or the
        # hand-chain (default, C13). Both feed the SAME grads.d_a/d_b lists into
        # the clip+optimizer seam; the graph path is bit-identical (per-block gate).
        var grads: LTX2VideoStackGrads
        if v2_capture:
            grads = ltx2_video_stack_lora_backward_graph_capture[S_Vp, N_TXT](
                d_pred, fwd.saved_inputs, fwd.x_last,
                encf, v_temb, v_embedded, v_prompt_ts, v_cos, v_sin,
                tail, src, la_use, lb_use, lora_scale, NUM_LAYERS, EPS, ctx,
                cap_stage.value(), cfg.lora_target_preset, save_acts_k,
            )
        elif v2_slab:
            grads = ltx2_video_stack_lora_backward_graph_slab[S_Vp, N_TXT](
                d_pred, fwd.saved_inputs, fwd.x_last,
                encf, v_temb, v_embedded, v_prompt_ts, v_cos, v_sin,
                tail, src, la_use, lb_use, lora_scale, NUM_LAYERS, EPS, ctx,
                cfg.lora_target_preset, save_acts_k,
            )
        elif v2_graph:
            grads = ltx2_video_stack_lora_backward_graph[S_Vp, N_TXT](
                d_pred, fwd.saved_inputs, fwd.x_last,
                encf, v_temb, v_embedded, v_prompt_ts, v_cos, v_sin,
                tail, src, la_use, lb_use, lora_scale, NUM_LAYERS, EPS, ctx,
                cfg.lora_target_preset, save_acts_k,
            )
        else:
            grads = ltx2_video_stack_lora_backward[S_Vp, N_TXT](
                d_pred, fwd.saved_inputs, fwd.x_last,
                encf, v_temb, v_embedded, v_prompt_ts, v_cos, v_sin,
                tail, src, la_use, lb_use, lora_scale, NUM_LAYERS, EPS, ctx,
                cfg.lora_target_preset, save_acts_k, fwd.saved_fwds,
            )
        if grads.nonfinite != 0:
            raise Error(String("non-finite LoRA grads at step ") + String(step))

        if timing:
            ctx.synchronize()
        var tm_bwd = perf_counter_ns()

        # ── P4 grad-accum window: SUM this micro's host grads; step the optimizer
        # + cadence ONLY on a window boundary. accum_steps==1 -> BYPASSED (grads
        # untouched, no continue -> byte-identical). On a non-boundary micro:
        # print a micro sub-line and `continue` (skip clip/opt/save). On the
        # boundary: MEAN the window back into grads.d_a/d_b; the clip+optimizer
        # seam below then runs on the MEANED grads with its OWN norm (device norm
        # on the device arm / host F64 on the host arm — identical to the window's
        # recompute for the host arm). `_opt_idx` (UPDATE index) keys AdamW t / LR
        # / cadence / save; `step` (MICRO) keeps keying sample/sigma/noise/dropout.
        if accum_steps > 1:
            var is_boundary = accum_window.accumulate(grads.d_a, grads.d_b, False)
            if not is_boundary:
                print("   [micro]", step, " window", (step - 1) % accum_steps + 1, "/",
                      accum_steps, " loss", loss, " sample", idx)
                step += 1
                continue
            _ = accum_window.finalize_mean(grads.d_a, grads.d_b)  # mean into grads
        var _opt_idx = (step - 1) // accum_steps + 1

        # ── clip + optimizer: DEVICE-fused (Phase A) or HOST (C13) ───────────
        # step_lr = scheduled LR (default constant/warmup0 -> cfg.learning_rate
        # EXACTLY). Both arms clip at musubi max_norm=1.0 then take one decoupled-
        # WD AdamW step. The fused device kernel is FORMULA-IDENTICAL to
        # _adamw_host_list_f32 (same bias correction, same WD-before-subtract,
        # clip_scale folded into g); value-class vs host (device FMA + F32 norm
        # reduction). Under a lever optimizer -> host path (F32-master lever step).
        # G1 (OWED): a dedicated seam-level gate for this F32-optimizer copy path
        # (host grads -> device upload -> fused AdamW -> resident A/B -> host sync)
        # asserting bit/value parity of the round-tripped masters+moments in
        # isolation. Currently covered indirectly by the 4-step anchor + moment-
        # fidelity gates (device-vs-host max|d| adam_m 5.4e-9 / master 2.7e-6).
        var step_lr = _ltx2_step_lr(cfg, _opt_idx)
        var gnorm = 0.0
        var tm_clip = tm_bwd
        if device_opt:
            # Upload the host grads -> F32 device tensors (order [a...][b...] to
            # match the resident param list). on_device_grad_stats returns the
            # PRE-clip global L2 norm + nonfinite count in ONE device reduction
            # (replaces the host clip loop AND the host nonfinite scan);
            # fused_adamw_step folds clip_scale and mutates the resident A/B +
            # moments in place — they ARE next step's LoRA weights (no re-upload).
            var gdev = List[ArcPointer[Tensor]]()
            for ai in range(len(masters)):
                var vd = _slot_dims(mods[ai % n_slots])
                gdev.append(ArcPointer[Tensor](Tensor.from_host(
                    grads.d_a[ai].copy(), [cfg.lora_rank, vd[0]], STDtype.F32, ctx)))
            for ai in range(len(masters)):
                var vd = _slot_dims(mods[ai % n_slots])
                gdev.append(ArcPointer[Tensor](Tensor.from_host(
                    grads.d_b[ai].copy(), [vd[1], cfg.lora_rank], STDtype.F32, ctx)))
            var stats = on_device_grad_stats(gdev, ctx)
            if stats.nonfinite_count != 0:
                raise Error(String("non-finite LoRA grads (device) at step ") + String(step))
            gnorm = Float64(stats.grad_norm)
            var clip_scale = Float32(1.0)
            if MAX_GRAD_NORM > Float32(0.0) and stats.grad_norm > MAX_GRAD_NORM:
                clip_scale = MAX_GRAD_NORM / stats.grad_norm
            tm_clip = perf_counter_ns()
            var pdev = List[ArcPointer[Tensor]]()
            var mdev = List[ArcPointer[Tensor]]()
            var vdev = List[ArcPointer[Tensor]]()
            for i in range(len(res_a)): pdev.append(res_a[i].copy())
            for i in range(len(res_b)): pdev.append(res_b[i].copy())
            for i in range(len(res_ma)): mdev.append(res_ma[i].copy())
            for i in range(len(res_mb)): mdev.append(res_mb[i].copy())
            for i in range(len(res_va)): vdev.append(res_va[i].copy())
            for i in range(len(res_vb)): vdev.append(res_vb[i].copy())
            fused_adamw_step(
                pdev, gdev, mdev, vdev, _opt_idx, step_lr,
                Float32(0.9), Float32(0.999), Float32(1e-8), cfg.weight_decay,
                ctx, clip_scale)
        else:
            # ── musubi global grad clip (max_norm=1.0, default-ON) — HOST ─────
            var sq = 0.0
            for ai in range(len(grads.d_a)):
                for i in range(len(grads.d_a[ai])):
                    var g = Float64(grads.d_a[ai][i]); sq += g * g
                for i in range(len(grads.d_b[ai])):
                    var g = Float64(grads.d_b[ai][i]); sq += g * g
            gnorm = sqrt(sq)
            if MAX_GRAD_NORM > Float32(0.0) and gnorm > Float64(MAX_GRAD_NORM):
                var sc = Float32(Float64(MAX_GRAD_NORM) / gnorm)
                for ai in range(len(grads.d_a)):
                    for i in range(len(grads.d_a[ai])):
                        grads.d_a[ai][i] = grads.d_a[ai][i] * sc
                    for i in range(len(grads.d_b[ai])):
                        grads.d_b[ai][i] = grads.d_b[ai][i] * sc
            tm_clip = perf_counter_ns()
            # ── OPTIMIZER SEAM (mirrors krea2:5100) — HOST ───────────────────
            # Levers optimizer (adafactor/schedule_free/adamw_8bit/automagic3)
            # routes through the F32-MASTER path (MJ-1112): the *_step fns run in
            # place on F32 view lists, NO bf16 hop. The clipped grads feed both
            # arms. Under a lever optimizer the F32Lora ma/va/mb/vb are DEAD
            # (state lives in opt_state); the .state save still writes them.
            if lever_opt:
                var views = List[F32AdapterView]()
                for i in range(len(masters)):
                    var vd = _slot_dims(mods[i % n_slots])
                    views.append(F32AdapterView(
                        masters[i].a.copy(), masters[i].b.copy(),
                        cfg.lora_rank, vd[0], vd[1]))
                levers_optimizer_step_host_f32(
                    cfg.levers, views, grads.d_a, grads.d_b,
                    _opt_idx, step_lr, 0, len(views), opt_state)
                for i in range(len(masters)):
                    masters[i].a = views[i].a.copy()
                    masters[i].b = views[i].b.copy()
            else:
                for ai in range(len(masters)):
                    _adamw_host_list_f32(
                        masters[ai].a, grads.d_a[ai], masters[ai].ma, masters[ai].va,
                        _opt_idx, step_lr, Float32(0.9), Float32(0.999),
                        Float32(1e-8), cfg.weight_decay)
                    _adamw_host_list_f32(
                        masters[ai].b, grads.d_b[ai], masters[ai].mb, masters[ai].vb,
                        _opt_idx, step_lr, Float32(0.9), Float32(0.999),
                        Float32(1e-8), cfg.weight_decay)

        var t1 = perf_counter_ns()
        if timing:
            print("   [timing] load", Float64(tm_load - t0) / 1.0e9,
                  "| noise", Float64(tm_noise - tm_load) / 1.0e9,
                  "| head", Float64(tm_head - tm_noise) / 1.0e9,
                  "| lora_up", Float64(tm_loraup - tm_head) / 1.0e9,
                  "| fwd", Float64(tm_fwd - tm_loraup) / 1.0e9,
                  "| loss", Float64(tm_loss - tm_fwd) / 1.0e9,
                  "| bwd", Float64(tm_bwd - tm_loss) / 1.0e9,
                  "| clip", Float64(tm_clip - tm_bwd) / 1.0e9,
                  "| opt", Float64(t1 - tm_clip) / 1.0e9)
        print_trainer_progress(
            String("LTX2-video-lora"), _opt_idx, cfg.max_steps, len(items),
            loss, gnorm, Float64(t1 - t0) / 1.0e9, 0.0,
            Float64(t1 - t_start) / 1.0e9,
        )
        print("   sigma=", sigma, " sample=", idx)

        # ── val_loss (P2): forward-ONLY eval on the val cache ───────────────
        # Reuses the training forward path (ltx2_video_stack_lora_forward); the
        # returned saved-activation tensors are simply dropped (no backward). NO
        # extra resident allocations: val_dev is built ONCE per pass and reused
        # across samples, and Mojo ASAP-destroys the just-finished training step's
        # dev/fwd/grads before this block runs. Caption dropout is NOT applied in
        # val (real-caption loss). Same loss seam as training.

        # ── SPEED PASS Phase A: refresh host masters from the resident device ─
        # tensors before ANY cadence block that reads masters (val/save/sample).
        # device_opt leaves masters stale between cadences (state lives on
        # device); sync A/B + moments here so the readers below see the current
        # weights. Amortized (cadence-only), NOT per-step. host path: no-op.
        if device_opt:
            var save_hit = (cfg.save_every > 0 and _opt_idx % cfg.save_every == 0) or _opt_idx == cfg.max_steps
            var samp_hit = cfg.sample_every > 0 and _opt_idx % cfg.sample_every == 0
            var val_hit = val_on and (_opt_idx % cfg.validate_every == 0)
            if save_hit or samp_hit or val_hit:
                for i in range(len(masters)):
                    masters[i].a = res_a[i][].to_host(ctx)
                    masters[i].b = res_b[i][].to_host(ctx)
                    masters[i].ma = res_ma[i][].to_host(ctx)
                    masters[i].va = res_va[i][].to_host(ctx)
                    masters[i].mb = res_mb[i][].to_host(ctx)
                    masters[i].vb = res_vb[i][].to_host(ctx)

        if val_on and (_opt_idx % cfg.validate_every == 0):
            var val_dev = _lora_to_device(masters, cfg.lora_rank, mods, ctx, stack_dt)
            var vloss_sum = Float64(0.0)
            for vi in range(len(val_items)):
                ref vitem = val_items[vi]
                assert_prompt_mask_all_ones(vitem.te_path)
                var vlat5 = reshape(
                    _load_bf16(vitem.lat_path, lat_key, ctx), _sh5(1, C, NFp, NHp, NWp), ctx)
                var venc = reshape(
                    _load_bf16(vitem.te_path, String(ENC_KEY), ctx), _sh3(1, N_TXT, VD), ctx)
                var vsigma = _sample_sigma(vcfg, vi, S_TGT)
                var vlatf = cast_tensor(vlat5, STDtype.F32, ctx)
                var vnoise = randn(
                    _sh5(1, C, NFp, NHp, NWp),
                    cfg.seed * UInt64(2000003) + UInt64(vi) * UInt64(104729),
                    STDtype.F32, ctx)
                var vnoisy = add(
                    mul_scalar(vlatf, Float32(1.0) - vsigma, ctx),
                    mul_scalar(vnoise, vsigma, ctx), ctx)
                var vtarget5 = sub(vnoise, vlatf, ctx)
                var vnoisy_bf16 = cast_tensor(vnoisy, STDtype.BF16, ctx)
                var vho: LTX2VideoHeadOut
                comptime if S_REF > 0:
                    # v2v val (A2b): clean ref + noisy target, ref-prepend, two-grid
                    # rope — mirrors the train seam. First-frame OFF (deterministic
                    # val; it's a training-only augmentation) -> ff_fired=False.
                    var vref_latf = cast_tensor(
                        reshape(_load_ref_latent(vitem, ctx), _sh5(1, C, RNFp, RNHp, RNWp), ctx),
                        STDtype.BF16, ctx)
                    # deterministic val: NO intrinsic conditioning -> all-False mask.
                    var vcond_mask = List[Bool]()
                    for _ in range(S_TGT):
                        vcond_mask.append(False)
                    vho = head.forward_v2v[S_Vp, S_REF, S_TGT, N_TXT](
                        vref_latf, vnoisy_bf16, venc, vsigma,
                        NFp, NHp, NWp, RNFp, RNHp, RNWp, DS, vcond_mask, FRAME_RATE, ctx)
                else:
                    vho = head.forward[S_Vp, N_TXT](
                        vnoisy_bf16, venc, vsigma, NFp, NHp, NWp, FRAME_RATE, ctx)
                # Cast to stack_dt (BF16 by DEFAULT — the bf16-stack class), NOT a
                # hardcoded F32: the block weights/AdaLN tables run at stack_dt, so
                # F32 val carriers collide with the bf16 scale_shift_table ([1,1,VD]
                # BF16 vs [1,S,VD] F32). This mirrors the train seam's cast points
                # and makes val-parity-with-train exact. (Pre-existing latent bug for
                # t2v val too once bf16-stack became default; surfaced by v2v val.)
                var vhidden = cast_tensor(vho.hidden, stack_dt, ctx)
                var vv_temb = cast_tensor(vho.v_temb, stack_dt, ctx)
                var vv_embedded = cast_tensor(vho.v_embedded, stack_dt, ctx)
                var vv_prompt_ts = cast_tensor(vho.v_prompt_ts, stack_dt, ctx)
                var vv_cos = cast_tensor(vho.v_cos, stack_dt, ctx)
                var vv_sin = cast_tensor(vho.v_sin, stack_dt, ctx)
                var vencf = cast_tensor(venc, stack_dt, ctx)
                var vfwd = ltx2_video_stack_lora_forward[S_Vp, N_TXT](
                    vhidden, vencf, vv_temb, vv_embedded, vv_prompt_ts, vv_cos, vv_sin,
                    tail, src, val_dev[0], val_dev[1], lora_scale, NUM_LAYERS, EPS, ctx,
                    cfg.lora_target_preset)
                var vpredh = vfwd.pred.to_host(ctx)
                var vl = Float32(0.0)
                comptime if S_REF > 0:
                    # v2v val loss: slice ref off + target-only masked MSE (loss
                    # value only; the cotangent is discarded — val is forward-only).
                    # First-frame OFF here, so the mask is all-True over the target.
                    var vtargh = _patchify_tokens(vtarget5, S_TGT, ctx).to_host(ctx)
                    var vvmask = List[Bool]()
                    for _ in range(S_TGT):
                        vvmask.append(True)
                    var vlc = v2v_target_cotangent(vpredh, vtargh, vvmask, S_REF, S_TGT, C)
                    vl = vlc[0] * cfg.video_loss_weight
                else:
                    var vtargh = _patchify_tokens(vtarget5, S_Vp, ctx).to_host(ctx)
                    if levers_loss_active(cfg.levers):
                        var lg = levers_loss_grad(vpredh, vtargh, vsigma, cfg.levers)
                        vl = lg.loss * cfg.video_loss_weight
                    else:
                        var s64 = Float64(0.0)
                        for i in range(len(vpredh)):
                            var d = Float64(vpredh[i]) - Float64(vtargh[i])
                            s64 += d * d
                        vl = Float32(s64 / Float64(len(vpredh))) * cfg.video_loss_weight
                vloss_sum += Float64(vl)
            var val_loss = Float32(vloss_sum / Float64(len(val_items)))
            print("[val] step", _opt_idx, "val_loss", val_loss, "over", len(val_items), "samples")

        # ── save cadence (PEFT bf16 file + F32-exact state sidecar) ──────────
        if (cfg.save_every > 0 and _opt_idx % cfg.save_every == 0) or _opt_idx == cfg.max_steps:
            var named = List[NamedLora]()
            var states = List[F32LoraState]()
            for bi in range(NUM_LAYERS):
                for s in range(n_slots):
                    ref msrc = masters[bi * n_slots + s]
                    var sd = _slot_dims(mods[s])
                    var pfx = diffusion_lora_prefix(bi, mods[s])
                    # PEFT file: bf16 happens HERE (LoraAdapter constructor
                    # downcast) — matching musubi's f32-train/bf16-save.
                    named.append(NamedLora(
                        pfx.copy(),
                        LoraAdapter(
                            msrc.a.copy(), msrc.b.copy(),
                            cfg.lora_rank, sd[0], sd[1], lora_scale,
                            msrc.ma.copy(), msrc.va.copy(),
                            msrc.mb.copy(), msrc.vb.copy(),
                        ),
                    ))
                    # State sidecar: F32-EXACT masters (no bf16 downcast) so a
                    # resume continues from the unrounded A/B (MJ-1108 class).
                    states.append(F32LoraState(
                        pfx.copy(),
                        msrc.a.copy(), msrc.b.copy(),
                        msrc.ma.copy(), msrc.va.copy(),
                        msrc.mb.copy(), msrc.vb.copy(),
                        cfg.lora_rank, sd[0], sd[1],
                    ))
            var base = cfg.output_dir + "/ltx2_video_lora_step" + String(_opt_idx)
            var n_pairs = save_lora_peft(named, base + ".safetensors", ctx)
            # shared trainer_core 5-field resume-scope meta. The step field is the
            # UPDATE index (_opt_idx) — resume reconstructs micro = update×accum+1.
            var meta = trainer_state_meta(
                _opt_idx, cfg.seed, len(items), S_Vp, cfg.dataset_cache_dir)
            var n_state = save_lora_train_state_f32(states, base + ".state.safetensors", ctx, meta^)
            print("   [save]", n_pairs, "pairs ->", base + ".safetensors",
                  "(+f32-state", n_state, ")")
            # levers optimizer state (P3.3): host-direct sidecar alongside the
            # .state. Under a lever optimizer the F32Lora ma/va/mb/vb saved above
            # are DEAD (never updated — state lives here). AdamW's masters use
            # them; a lever restore reads opt_state from this file, not those.
            if lever_opt:
                var os_path = levers_optimizer_sidecar_path(base)
                var n_os = levers_optimizer_sidecar_save(
                    opt_state, _opt_idx, cfg.seed, os_path)
                print("   [save] levers optstate", n_os, "states ->", os_path)

        # ── sample cadence (P2): COMFY/PEFT (HQ render input) + NATIVE artifacts ─
        # The render command uses ltx2_hq_ref_run.py, which loads the COMFY/PEFT
        # keys (LTXV_LORA_COMFY_RENAMING_MAP) — so we MUST emit <base>.safetensors
        # here (not only at save cadence) so the .sh's --user-lora always exists.
        # The NATIVE (save_lora_serenity_trainer, alpha/lora_down/lora_up) artifact is
        # kept alongside as the musubi-parity export (the musubi generator path may
        # return once precached-prompt support is wired). The .sh is NOT
        # auto-executed (staged-loader rule; never concurrent with the trainer).
        if cfg.sample_every > 0 and _opt_idx % cfg.sample_every == 0:
            var named = List[NamedLora]()   # comfy/PEFT — the HQ render artifact
            var nat = List[NamedLora]()     # native — musubi-parity export
            for bi in range(NUM_LAYERS):
                for s in range(n_slots):
                    ref msrc = masters[bi * n_slots + s]
                    var sd = _slot_dims(mods[s])
                    named.append(NamedLora(
                        diffusion_lora_prefix(bi, mods[s]),
                        LoraAdapter(
                            msrc.a.copy(), msrc.b.copy(),
                            cfg.lora_rank, sd[0], sd[1], lora_scale,
                            msrc.ma.copy(), msrc.va.copy(),
                            msrc.mb.copy(), msrc.vb.copy(),
                        ),
                    ))
                    nat.append(NamedLora(
                        _native_prefix(bi, mods[s]),
                        LoraAdapter(
                            msrc.a.copy(), msrc.b.copy(),
                            cfg.lora_rank, sd[0], sd[1], lora_scale,
                            msrc.ma.copy(), msrc.va.copy(),
                            msrc.mb.copy(), msrc.vb.copy(),
                        ),
                    ))
            var sbase = cfg.output_dir + "/ltx2_video_lora_step" + String(_opt_idx)
            var peft_path = sbase + ".safetensors"
            var native_path = sbase + ".native.safetensors"
            var n_peft = save_lora_peft(named, peft_path, ctx)
            var n_nat = save_lora_serenity_trainer(nat, native_path, ctx)
            var cmd_path = cfg.output_dir + "/sample_cmd_step" + String(_opt_idx) + ".sh"
            # v2v runs (S_REF>0) emit a v2v-aware render cmd (ref latent + --ic-lora).
            var _v2v_ref = String("")
            var _v2v_ds = 1
            comptime if S_REF > 0:
                _v2v_ref = sample_v2v_ref
                _v2v_ds = DS
            _write_sample_cmd(cmd_path, cfg, peft_path, _opt_idx, _v2v_ref, _v2v_ds)
            print("   [sample] comfy/PEFT", n_peft, "->", peft_path,
                  "; native", n_nat, "->", native_path)
            print("   [sample] render command (run manually) ->", cmd_path)
        step += 1

    print("DONE LTX2 video-mode LoRA training:", cfg.max_steps, "steps")


# ══ AV (audio-video) arm runner (P6.2) ═══════════════════════════════════════
# Joint audio+video LoRA training off PRESET_AUDIO (672 adapters). SELF-CONTAINED
# host-F32 loop: the AV stack streams blocks internally (LTX2AVBlockWeights.load
# per block — no fp8-resident) and the committed backward consumes saved acts for
# ALL blocks, so NONE of the video runner's fp8-resident/device-opt/capture/accum
# machinery applies — F32 masters + _adamw_host_list_f32. Numeric pieces gated at
# the musubi-correct rope (causal_offset=1, max_pos [20,2048,2048]): stack fwd
# 37/0, bwd 115/0, rope 8/8, loss-combine 4/4, cache round-trip 9/9.
comptime AD_INNER = 2048   # audio inner_dim
# AV arm checkpoint: the DISTILLED fp8-dequant-bf16 model carries the audio
# modules (audio_attn/audio_ff/cross-modal) — the video default (dev-fp8)
# does NOT. This is the checkpoint the AV stack/rope gates ran against.
comptime AV_DEFAULT_CKPT_NAME = "ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
# distinct per-modality caption-dropout seed bases (v-drop / a-drop i.i.d.).
comptime _VDROP_MULT = UInt64(2246822519)
comptime _ADROP_MULT = UInt64(3266489917)


# The 672 audio-preset ACTIVE mask over the 28-slot _av_lora_slots() order
# (Option-1, lead 07-18). Active = the 14 audio-preset modules
# (lora_surface.audio_modules_per_block: audio_attn1/audio_attn2/video_to_audio_attn
# x{q,k,v,out} + audio_ff.net.0.proj/net.2). The other 14 are INACTIVE (zero-A +
# zero-B identity — the video path runs frozen-base). Grads on inactive slots are
# EXACTLY 0 (d_A=f(B=0)=0, d_B=f(A=0)=0), asserted zero at the final step.
def _av_active_slots() -> List[Bool]:
    var slots = _av_lora_slots()
    var audio = modules_per_block_for_preset(PRESET_AUDIO)
    var out = List[Bool]()
    for ref s in slots:
        var active = False
        for ref a in audio:
            if s == a:
                active = True
        out.append(active)
    return out^


# Per-slot (in,out) for an AV module — READ from block-0's weight shape [out,in]
# (like the oracle build_lora), since audio/cross-modal dims aren't the video
# VDxVD/FFV pattern that _slot_dims covers.
def _av_slot_dims(w0: LTX2AVBlockWeights, module_path: String) raises -> Tuple[Int, Int]:
    ref wt = w0._w(module_path + ".weight")
    ref sh = wt.shape()
    return (sh[1], sh[0])


# tri-pair audio latent [C,T,mel] -> [1, S_A=T, patch_in=C*mel] (the audio
# patchify input; musubi ltx2_cache_latents.py:473-490 stores [C,T,mel]). Mirrors
# the gated ltx2_av_cache_roundtrip reshape.
def _load_audio_latent_tokens(apath: String, akey: String, ctx: DeviceContext) raises -> Tensor:
    var ast = ShardedSafeTensors.open(apath)
    var raw = Tensor.from_view_as_f32(ast.tensor_view(akey), ctx)   # [C,T,mel]
    ref sh = raw.shape()
    var cc = sh[0]; var tt = sh[1]; var mm = sh[2]
    var host = raw.to_host(ctx)
    var patch_in = cc * mm
    var out = List[Float32]()
    out.resize(tt * patch_in, Float32(0.0))
    for t in range(tt):
        for c in range(cc):
            for m in range(mm):
                out[t * patch_in + c * mm + m] = host[(c * tt + t) * mm + m]
    return Tensor.from_host(out^, _sh3(1, tt, patch_in), STDtype.F32, ctx)


def _load_audio_length(apath: String, alen_key: String) raises -> Int:
    """`audio_lengths_*` is I32 — musubi ltx2_cache_latents.py:483-490 writes
    `torch.tensor(effective_steps, dtype=torch.int32)`. Read the raw bytes:
    from_view_as_f32 REJECTS I32 (io/dtype.mojo:154-166 admits BF16/F16/F32
    only), which is what made the first AV micro-run die with
    "unsupported compute dtype: I32" before any weight load."""
    var ast = ShardedSafeTensors.open(apath)
    var info = ast.tensor_info(alen_key)
    if info.dtype != STDtype.I32:
        raise Error(String("av audio cache key '") + alen_key + "' must be I32"
            + " (musubi writes torch.int32); got " + info.dtype.name())
    var bytes = ast.tensor_bytes(alen_key)
    var v = Int(bytes.unsafe_ptr().bitcast[Int32]()[0])
    return v


def _run_geometry_av[
    NF: Int, NH: Int, NW: Int, S_A: Int, N_TXT_AV: Int,
](
    cfg: LTX2TrainerConfig, args: List[String], lat_key: String,
) raises:
    comptime S_V = NF * NH * NW
    var ckpt = cfg.ltx2_checkpoint
    if len(ckpt) == 0:
        # AV needs the audio-carrying distilled checkpoint.
        ckpt = serenity_checkpoint(String(AV_DEFAULT_CKPT_NAME))
    if len(cfg.dataset_cache_dir) == 0:
        raise Error("train_ltx2_av (AV): --dataset_cache_dir is required (tri-pair cache)")
    _ = sys_mkdirs(cfg.output_dir)

    # PIN1: joint sigma (audio_sigma = video sigma). independent_audio_timestep is
    # P6.3 (separate audio timestep) — fail LOUD if requested here.
    if cfg.independent_audio_timestep:
        raise Error("train_ltx2_av (AV): independent_audio_timestep is P6.3 (joint"
            " sigma only in P6.2); unset --independent_audio_timestep")

    var video_w = cfg.levers.video_loss_weight
    var audio_w = cfg.levers.audio_loss_weight
    # 672 audio preset with audio_weight==0 trains NOTHING (all 672 adapters are
    # audio-side; their grads come from d_audio_vel = raw*audio_w). Guard loud.
    if audio_w <= Float32(0.0):
        raise Error(String("train_ltx2_av (AV): audio_loss_weight must be > 0 for the")
            + " audio preset (all 672 adapters get their grad from the audio loss;"
            + " audio_loss_weight=" + String(audio_w) + " => nothing trains). Set"
            + " audio_loss_weight in the config.")

    # RECOMPUTE conductor (mirrors the video arm's LTX2_SAVE_ACTS knob at :973).
    # Default 0 = recompute every block's acts in the backward from its saved
    # inputs. The AV arm previously retained ALL 48 blocks' acts, which is
    # 375.18 MiB/block x48 = 17.59 GiB and OOM'd a 24 GiB card inside step 1.
    # GATE 0 measured recompute at 35.2ms/block vs 80.1ms/block for a host tape,
    # so recompute is both the cheaper and the only fitting arm. K>0 trades VRAM
    # back for speed on the first K blocks; the two arms are numerically identical.
    var av_save_acts_k = env_int("LTX2_SAVE_ACTS", 0)
    if av_save_acts_k > NUM_LAYERS:
        av_save_acts_k = NUM_LAYERS

    print("[ltx2-av-lora] joint AV trainer — geometry S_V=", S_V, "(", NF, "x", NH,
          "x", NW, ") S_A=", S_A, "N_TXT=", N_TXT_AV, "blocks=", NUM_LAYERS,
          "vkey=", lat_key)
    print("  [vram] save_acts_k=", av_save_acts_k, "of", NUM_LAYERS,
          "blocks retained; the rest recompute in the backward")
    print("  [loss] video_loss_weight=", video_w, "audio_loss_weight=", audio_w,
          "use_audio_length_mask=", cfg.levers.use_audio_length_mask)

    var items = _enumerate_cache(cfg.dataset_cache_dir, lat_key)
    if len(items) == 0:
        raise Error("train_ltx2_av (AV): no cache samples matching the geometry")
    print("  [cache]", len(items), "tri-pair samples @", cfg.dataset_cache_dir)

    var ctx = DeviceContext()
    var model_cfg = LTX2Config.ltx2()

    # ── LoRA masters: 28-slot layout (Option-1). Active audio slots kaiming/B=0;
    #    inactive slots zero-A+zero-B. Dims READ from block-0 weight shapes. ─────
    var slots = _av_lora_slots()
    var active = _av_active_slots()
    var n_active = 0
    for a in active:
        if a: n_active += 1
    var w0 = LTX2AVBlockWeights.load(ckpt, 0, model_cfg, ctx)
    var masters = List[F32Lora]()
    for bi in range(NUM_LAYERS):
        for s in range(len(slots)):
            var dims = _av_slot_dims(w0, slots[s])
            var in_f = dims[0]; var out_f = dims[1]
            var a: List[Float32]
            if active[s]:
                a = _kaiming_a(cfg.lora_rank, in_f, cfg.seed + UInt64(bi * len(slots) + s))
            else:
                a = _zeros(cfg.lora_rank * in_f)
            masters.append(F32Lora(
                a^, _zeros(out_f * cfg.lora_rank),
                _zeros(cfg.lora_rank * in_f), _zeros(cfg.lora_rank * in_f),
                _zeros(out_f * cfg.lora_rank), _zeros(out_f * cfg.lora_rank)))
    var lora_scale = cfg.lora_alpha / Float32(cfg.lora_rank)
    print("  [lora] rank", cfg.lora_rank, "alpha", cfg.lora_alpha, "-> 28 slots/block,",
          n_active, "ACTIVE (audio 672 =", n_active * NUM_LAYERS, "adapters) /",
          (len(slots) - n_active), "inactive zero, scale", lora_scale)

    var st = ShardedSafeTensors.open(ckpt)

    # ── fp8-RESIDENT block store (speed) ─────────────────────────────────────
    # The AV loop materialises all 48 blocks in the forward and 48 again in the
    # backward, every step: MEASURED 141.8 ms/block x96 = 13.6 s of a 23.6 s
    # step (~59%). Residency parks the raw fp8 bytes in VRAM once so each visit
    # dequants from VRAM instead of re-reading the checkpoint.
    #
    # Residency changes NO math: this is the DEQUANT path (load_block_bf16 ->
    # from_fp8_block -> to_f32), the same one the video training arm uses, NOT
    # the raw-fp8 `linear_fp8` inference path (which `to_f32` refuses outright).
    # The fp8 and dequant-bf16 checkpoints were MEASURED bit-identical on this
    # path, 86/86 keys (parity/ltx2_av_ckpt_equiv_probe.mojo), so anchors hold.
    #
    # Mirrors the video arm's knob (:1231): --resident_blocks N / 0=off.
    var n_resident = env_int("LTX2_RESIDENT_BLOCKS", 0)
    for i in range(len(args)):
        if String(args[i]) == "--resident_blocks" and i + 1 < len(args):
            n_resident = Int(_parse_int_arg(String(args[i + 1])))
    if n_resident > NUM_LAYERS:
        n_resident = NUM_LAYERS
    # SquareQ W4A16 residency arm (env LTX2_SQUAREQ_SLAB=<slab dir>, mirrors the
    # video arm + the inference LTX2_INT4_SLAB precedent): blocks reconstruct
    # dense BF16 from the packed slab per visit (ops/squareq via
    # LTX2Int4BlockStream smooth-key auto-detect). The checkpoint MUST be the
    # slab's build source (<slab>/build-state.json) so head/tail/w0 passthrough
    # tensors match. fp8 residency is a contradiction here — fail loud.
    var squareq_slab = env_or("LTX2_SQUAREQ_SLAB", String(""))
    var src: LTX2AVBlockSource
    if len(squareq_slab) > 0:
        if n_resident > 0:
            raise Error(
                "train_ltx2_av (AV): --resident_blocks/LTX2_RESIDENT_BLOCKS > 0"
                " is the fp8 residency knob; it does not combine with"
                " LTX2_SQUAREQ_SLAB (blocks reconstruct from the packed slab)."
                " Set it to 0."
            )
        try:
            src = LTX2AVBlockSource.open_squareq(ckpt, squareq_slab, model_cfg)
        except err:
            raise Error(
                String("train_ltx2_av (AV): LTX2_SQUAREQ_SLAB='") + squareq_slab
                + "' failed to open: " + String(err))
        print("  [squareq] W4A16 base ON — blocks reconstruct BF16 from",
              squareq_slab)
    else:
        src = LTX2AVBlockSource.open(ckpt, model_cfg)
    if n_resident > 0:
        # PRE-ALLOC over-commit guard (mirrors the video arm): the resident range
        # allocates up front, which otherwise dies as a generic CUDA OOM inside
        # the load phase. AV peak measured 7197 MiB of 24564, so budget against
        # the ~17367 MiB of free headroom, NOT the card total.
        #
        # Per-block cost is MEASURED, and is NOT one number — the two block
        # classes differ by 2x (parity/ltx2_av_resident_budget_probe.mojo,
        # exact to the byte at both N=6 and N=12):
        #   BF16-boundary block (0-3, 47): 773390720 B = 737.56 MiB  (2 B/param)
        #   fp8 block           (4-46)   : 386924928 B = 369.00 MiB  (1 B/param)
        # Full 48 = 20504725504 B = 19556 MiB, which does NOT FIT in the 17367
        # MiB headroom — hence the knob is a range, not a bool.
        var n_bf16 = 0
        for b in range(n_resident):
            if src.block_count() > b and b < 4:
                n_bf16 += 1
        if n_resident >= 48:
            n_bf16 += 1                      # block 47 is BF16-boundary too
        var est_mib = Int(
            (Float64(n_bf16) * 737.56) + (Float64(n_resident - n_bf16) * 369.0)
        )
        print("  [resident] preloading blocks 0 ..", n_resident - 1,
              " (est ~", est_mib, "MiB =", n_bf16, "BF16 +",
              n_resident - n_bf16, "fp8)")
        if est_mib > 17367:
            raise Error(
                String("train_ltx2_av: --resident_blocks ") + String(n_resident)
                + " needs ~" + String(est_mib) + " MiB but only ~17367 MiB is"
                " free after the 7197 MiB AV training peak. Lower"
                " --resident_blocks (the fp8 blocks 4+ cost 369 MiB each)."
            )
        src.enable_resident(0, n_resident - 1, ctx)
        print("  [resident] resident storage bytes:", src.resident_bytes(),
              " (=", Float64(src.resident_bytes()) / 1048576.0, "MiB )")
    else:
        print("  [resident] OFF (--resident_blocks 0); blocks stream per visit")

    # rope built ONCE (geometry-fixed, musubi-correct offset=1 + max_pos 2048).
    var rope = ltx2_av_build_rope[NF, NH, NW, S_A](ctx)

    var t_start = perf_counter_ns()
    var step = 1
    while step <= cfg.max_steps:
        var t0 = perf_counter_ns()
        # sample pick (seeded uniform).
        var idx = Int(_splitmix64(cfg.seed * UInt64(31) + UInt64(step)) % UInt64(len(items)))
        ref item = items[idx]
        assert_prompt_mask_all_ones(item.te_path)

        # ── load the tri-pair ────────────────────────────────────────────────
        var v_lat5 = reshape(_load_bf16(item.lat_path, lat_key, ctx),
                             _sh5(1, C, NF, NH, NW), ctx)
        var enc = cast_tensor(reshape(
            _load_bf16(item.te_path, String("video_prompt_embeds_bfloat16"), ctx),
            _sh3(1, N_TXT_AV, VD), ctx), STDtype.F32, ctx)
        var aenc = cast_tensor(reshape(
            _load_bf16(item.te_path, String("audio_prompt_embeds_bfloat16"), ctx),
            _sh3(1, N_TXT_AV, AD_INNER), ctx), STDtype.F32, ctx)
        var apath = audio_cache_path(item.lat_path)
        var ast = ShardedSafeTensors.open(apath)
        var akey = discover_audio_latents_key(ast)
        var alen_key = discover_audio_lengths_key(ast)
        var a_lat = _load_audio_latent_tokens(apath, akey, ctx)   # [1,S_A,patch_in]
        var audio_length = _load_audio_length(apath, alen_key)

        # ── per-modality caption dropout (v-drop / a-drop, DISTINCT seed bases) ─
        if cfg.levers.video_caption_dropout_rate > Float32(0.0) and caption_dropout_pick(
            UInt64(step), cfg.seed * _VDROP_MULT + UInt64(1), cfg.levers.video_caption_dropout_rate):
            enc = mul_scalar(enc, Float32(0.0), ctx)
        if cfg.levers.audio_caption_dropout_rate > Float32(0.0) and caption_dropout_pick(
            UInt64(step), cfg.seed * _ADROP_MULT + UInt64(1), cfg.levers.audio_caption_dropout_rate):
            aenc = mul_scalar(aenc, Float32(0.0), ctx)

        # ── PIN1 joint sigma + PIN2 flow-match noise/target (both streams) ────
        var sigma = _sample_sigma(cfg, step, S_V)
        var v_latf = cast_tensor(v_lat5, STDtype.F32, ctx)
        var v_noise = randn(_sh5(1, C, NF, NH, NW),
            cfg.seed + UInt64(step) * UInt64(104729), STDtype.F32, ctx)
        var v_noisy = add(mul_scalar(v_latf, Float32(1.0) - sigma, ctx),
                          mul_scalar(v_noise, sigma, ctx), ctx)
        var v_target5 = sub(v_noise, v_latf, ctx)          # velocity = noise - latent
        # audio: joint sigma; noise/noisy/target in the [1,S_A,patch_in] patch space.
        ref a_sh = a_lat.shape()
        var a_feat = a_sh[2]
        var a_noise = randn(_sh3(1, S_A, a_feat),
            cfg.seed * UInt64(7919) + UInt64(step) * UInt64(51287), STDtype.F32, ctx)
        var a_noisy = add(mul_scalar(a_lat, Float32(1.0) - sigma, ctx),
                          mul_scalar(a_noise, sigma, ctx), ctx)
        var a_target = sub(a_noise, a_lat, ctx)

        # ── head (per-sigma) + patchify both streams ─────────────────────────
        var head = ltx2_av_build_head[S_V, S_A, N_TXT_AV](st, sigma, ctx)
        var v_tokens = _patchify_tokens(v_noisy, S_V, ctx)             # [1,S_V,C]
        var hidden = _patchify(v_tokens, st, String("patchify_proj"), ctx)   # [1,S_V,VD]
        var ahs = _patchify(a_noisy, st, String("audio_patchify_proj"), ctx) # [1,S_A,AD]

        # ── LoRA -> device (F32; rebuilt from masters each step, host C13 path) ─
        var la = List[ArcPointer[Tensor]]()
        var lb = List[ArcPointer[Tensor]]()
        for i in range(len(masters)):
            var d = _av_slot_dims(w0, slots[i % len(slots)])
            la.append(ArcPointer[Tensor](Tensor.from_host(
                masters[i].a.copy(), [cfg.lora_rank, d[0]], STDtype.F32, ctx)))
            lb.append(ArcPointer[Tensor](Tensor.from_host(
                masters[i].b.copy(), [d[1], cfg.lora_rank], STDtype.F32, ctx)))

        # ── stack forward. save_acts_k = LTX2_SAVE_ACTS (default 0 = recompute
        #    every block in the backward from its saved inputs). Retaining all 48
        #    blocks' acts is what OOM'd this arm: 375.18 MiB/block x48 = 17.59 GiB.
        var fwd = ltx2_av_stack_forward[S_V, S_A, N_TXT_AV](
            st, src, hidden, ahs, enc, aenc, head,
            rope.v_cos, rope.v_sin, rope.a_cos, rope.a_sin,
            rope.ca_v_cos, rope.ca_v_sin, rope.ca_a_cos, rope.ca_a_sin,
            la, lb, lora_scale, NUM_LAYERS, EPS, ctx, av_save_acts_k)

        # ── loss (MSE; video unmasked, audio unmasked or length-masked) + the
        #    WEIGHTED cotangent seam. LOSS SEAM (P6.3 factor point): video_w/audio_w
        #    scale the RAW cotangents; uncertainty mode swaps the scale for
        #    0.5*exp(-log_var) here. Reported loss = av_combine_loss (gated). ─────
        var v_predh = fwd.video_vel.to_host(ctx)
        var v_targh = _patchify_tokens(v_target5, S_V, ctx).to_host(ctx)
        var loss_v = masked_loss_unmasked(v_predh, v_targh, LTX2_LOSS_MSE)
        var d_v = List[Float32]()
        var nv = len(v_predh)
        for i in range(nv):
            d_v.append(Float32(2.0) * (v_predh[i] - v_targh[i]) / Float32(nv) * video_w)

        var a_predh = fwd.audio_vel.to_host(ctx)
        var a_targh = a_target.to_host(ctx)
        var na = len(a_predh)
        var loss_a = Float32(0.0)
        var d_a = List[Float32]()
        if cfg.levers.use_audio_length_mask and audio_length < S_A:
            # bt-mask: frames [0, audio_length) kept (scout: audio pad is LOSS-ONLY).
            var mbt = List[Bool]()
            var true_frames = 0
            for t in range(S_A):
                var keep = t < audio_length
                mbt.append(keep)
                if keep: true_frames += 1
            loss_a = masked_loss_audio_bt_mask(
                a_predh, a_targh, mbt, 1, 1, S_A, a_feat, LTX2_LOSS_MSE)
            var denom = Float32(true_frames) / Float32(S_A)
            for t in range(S_A):
                var keep = t < audio_length
                for f in range(a_feat):
                    if keep:
                        var ix = t * a_feat + f
                        d_a.append(Float32(2.0) * (a_predh[ix] - a_targh[ix])
                                   / Float32(na) / denom * audio_w)
                    else:
                        d_a.append(Float32(0.0))
        else:
            loss_a = masked_loss_unmasked(a_predh, a_targh, LTX2_LOSS_MSE)
            for i in range(na):
                d_a.append(Float32(2.0) * (a_predh[i] - a_targh[i]) / Float32(na) * audio_w)

        var combined = av_combine_loss(loss_v, loss_a, video_w, audio_w)
        var d_video_vel = Tensor.from_host(d_v^, _sh3(1, S_V, len(v_predh) // S_V), STDtype.F32, ctx)
        var d_audio_vel = Tensor.from_host(d_a^, _sh3(1, S_A, a_feat), STDtype.F32, ctx)

        # ── stack backward (tail both streams -> reverse block loop; each block
        #    recomputes its acts from the saved inputs unless it is < save_acts_k) ─
        var v_last = fwd.v_last.clone(ctx)
        var a_last = fwd.a_last.clone(ctx)
        var grads = ltx2_av_stack_backward[S_V, S_A, N_TXT_AV](
            d_video_vel, d_audio_vel, fwd.saved_v_in, fwd.saved_a_in,
            v_last, a_last, enc, aenc, head,
            rope.v_cos, rope.v_sin, rope.a_cos, rope.a_sin,
            rope.ca_v_cos, rope.ca_v_sin, rope.ca_a_cos, rope.ca_a_sin,
            st, src, la, lb, lora_scale, NUM_LAYERS, EPS, ctx,
            av_save_acts_k, fwd.saved)
        if grads.nonfinite != 0:
            raise Error(String("train_ltx2_av (AV): non-finite grads at step ") + String(step))

        # ── clip (global L2 over ACTIVE slots) + host AdamW on ACTIVE slots ────
        var sq = 0.0
        for i in range(len(masters)):
            if active[i % len(slots)]:
                for x in grads.d_a[i]: sq += Float64(x) * Float64(x)
                for x in grads.d_b[i]: sq += Float64(x) * Float64(x)
        var gnorm = sqrt(sq)
        if MAX_GRAD_NORM > Float32(0.0) and gnorm > Float64(MAX_GRAD_NORM):
            var scl = Float32(Float64(MAX_GRAD_NORM) / gnorm)
            for i in range(len(masters)):
                if active[i % len(slots)]:
                    for j in range(len(grads.d_a[i])): grads.d_a[i][j] = grads.d_a[i][j] * scl
                    for j in range(len(grads.d_b[i])): grads.d_b[i][j] = grads.d_b[i][j] * scl
        var step_lr = _ltx2_step_lr(cfg, step)
        for i in range(len(masters)):
            if active[i % len(slots)]:
                _adamw_host_list_f32(masters[i].a, grads.d_a[i], masters[i].ma, masters[i].va,
                    step, step_lr, Float32(0.9), Float32(0.999), Float32(1e-8), cfg.weight_decay)
                _adamw_host_list_f32(masters[i].b, grads.d_b[i], masters[i].mb, masters[i].vb,
                    step, step_lr, Float32(0.9), Float32(0.999), Float32(1e-8), cfg.weight_decay)

        var t1 = perf_counter_ns()
        print_trainer_progress(String("LTX2-AV-lora"), step, cfg.max_steps, len(items),
            combined, gnorm, Float64(t1 - t0) / 1.0e9, 0.0, Float64(t1 - t_start) / 1.0e9)
        print("   sigma=", sigma, " loss_v=", loss_v, " loss_a=", loss_a, " sample=", idx)

        # ── save cadence: EXACTLY the 672 audio keys (the 14 active slots/block) ─
        if (cfg.save_every > 0 and step % cfg.save_every == 0) or step == cfg.max_steps:
            var named = List[NamedLora]()
            for bi in range(NUM_LAYERS):
                for s in range(len(slots)):
                    if active[s]:
                        ref msrc = masters[bi * len(slots) + s]
                        var d = _av_slot_dims(w0, slots[s])
                        named.append(NamedLora(
                            diffusion_lora_prefix(bi, slots[s]),
                            LoraAdapter(msrc.a.copy(), msrc.b.copy(),
                                cfg.lora_rank, d[0], d[1], lora_scale,
                                msrc.ma.copy(), msrc.va.copy(), msrc.mb.copy(), msrc.vb.copy())))
            var base = cfg.output_dir + "/ltx2_av_lora_step" + String(step)
            var n_pairs = save_lora_peft(named, base + ".safetensors", ctx)
            print("   [save]", n_pairs, "audio LoRA pairs (672 keys) ->", base + ".safetensors")

        step += 1

    # ── FINAL inactive-slot zero-assert (pin i): the 14 inactive slots must be
    #    EXACTLY zero (zero-init + provably-zero grads). Converts 'provable' to
    #    'measured every run'. ─────────────────────────────────────────────────
    var maxab = Float32(0.0)
    for i in range(len(masters)):
        if not active[i % len(slots)]:
            for x in masters[i].a:
                var ax = x if x >= Float32(0.0) else -x
                if ax > maxab: maxab = ax
            for x in masters[i].b:
                var bx = x if x >= Float32(0.0) else -x
                if bx > maxab: maxab = bx
    if maxab != Float32(0.0):
        raise Error(String("train_ltx2_av (AV): inactive-slot guard FAILED — max|A|+|B| over the")
            + " 14 inactive slots = " + String(maxab) + " (expected exactly 0); an inactive"
            + " slot trained (kaiming leak or a grad path bug).")
    print("[av] inactive-slot zero-assert PASS: max|A|,|B| over",
          (len(slots) - n_active) * NUM_LAYERS, "inactive slots = 0")
    print("DONE LTX2 AV-mode LoRA training:", cfg.max_steps, "steps")


def main() raises:
    var raw_args = argv()
    var args = List[String]()
    for i in range(len(raw_args)):
        args.append(String(raw_args[i]))
    var cfg = LTX2TrainerConfig.from_args(args)
    cfg.validate()
    var readiness = run_acceptance(True)
    if _has_arg(args, "--acceptance"):
        print_readiness(readiness)
        return
    print_config_summary(cfg)

    # OPTIMIZER VALIDATE (MJ-1112 closed): the F32-master levers path now runs
    # adafactor/schedule_free/adamw_8bit/automagic3 IN PLACE on the F32 masters
    # (levers_optimizer_step_host_f32 — no bf16 hop). This still fails LOUD for
    # optimizer tags with no levers dispatch (e.g. prodigy/came/lion/sgd).
    levers_optimizer_validate(cfg.levers, String("train_ltx2_av"))

    # P4: true bs>1 is DEFERRED (B=1 comptime-baked in every stack/block/flash
    # shape; accum is the musubi-equivalent for LoRA — plan P4). FAIL LOUD here,
    # before ANY DeviceContext, so it's provable without touching the GPU.
    if cfg.batch_size > 1:
        raise Error(String("train_ltx2_av: batch_size>1 is DEFERRED (accum is the")
            + " musubi-equivalent for LoRA — see plan P4). Got batch_size="
            + String(cfg.batch_size) + "; use --gradient_accumulation_steps N instead.")

    if cfg.ltx_mode == MODE_AV:
        # ── AV (audio-video) arm (P6.2): joint audio+video LoRA off PRESET_AUDIO
        #    (672 adapters). Geometry from the tri-pair cache: video latents_4x9x16
        #    (S_V=576) + audio S_A=16 + N_TXT=1024. Self-contained host-F32 runner
        #    (streams blocks internally; no fp8-resident/device-opt/capture). The
        #    audio latent key is DISCOVERED (audio_latents_{T}x{mel}x{C}).
        if cfg.lora_target_preset != PRESET_AUDIO:
            raise Error("train_ltx2_av: --ltx_mode av requires --lora_target_preset"
                " audio (the 672 audio surface; audio_ref_only_ic=864 is P6.4)")
        _run_geometry_av[4, 9, 16, 16, 1024](
            cfg, args, String("latents_4x9x16_bfloat16"))
        return
    if cfg.ltx_mode != MODE_VIDEO:
        raise Error(
            "train_ltx2_av: only --ltx_mode video (production) and --ltx_mode av"
            " (P6.2 audio, --lora_target_preset audio) are wired")
    # ── geometry dispatch (comptime monomorphizations, fleet pattern) ─────────
    #   --resume/--resume_from is handled inside _run_geometry (F32-exact masters
    #   + AdamW moments from the .state sidecar; warm bf16 fallback for old files).
    #   video    : 512x288x25f disney-class caches  -> [128,4,9,16]  = 576 tok
    #   image512 : 512x512 image caches (F=1)       -> [128,1,16,16] = 256 tok
    var geometry = String("video")
    for i in range(len(args)):
        if String(args[i]) == "--geometry" and i + 1 < len(args):
            geometry = String(args[i + 1])
    if geometry == "video":
        _run_geometry[576, 4, 9, 16](cfg, args, String("latents_4x9x16_bfloat16"))
    elif geometry == "image512":
        _run_geometry[256, 1, 16, 16](cfg, args, String("latents_1x16x16_bfloat16"))
    elif geometry == "image512_v2v":
        # IC-LoRA / v2v image512 arm (P5): target 1x16x16 (256) + reference
        # 1x8x8 (64, DS=2) -> S_COMB=320. Target latent key = the base image512
        # key; the CLEAN reference comes from --reference_cache_dir (required).
        _run_geometry[320, 1, 16, 16, 64, 1, 8, 8, 2](
            cfg, args, String("latents_1x16x16_bfloat16"))
    elif geometry == "video_v2v":
        # IC-LoRA / v2v video arm (P5): target 4x9x16 (576) + single-image
        # reference 1x8x8-class downscaled to 1x4x8 (32, DS=2) -> S_COMB=608.
        # Ref grid is APPROXIMATE co-location (9 != 4*2) — musubi-faithful. First-
        # frame Bernoulli fires here (NFp>1); --reference_cache_dir required.
        _run_geometry[608, 4, 9, 16, 32, 1, 4, 8, 2](
            cfg, args, String("latents_4x9x16_bfloat16"))
    else:
        raise Error(String("train_ltx2_av: unknown --geometry: ") + geometry
                    + " (video | image512 | image512_v2v | video_v2v)")
