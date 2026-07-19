# train_chroma_real.mojo — Chroma1-HD LoRA training loop (block-swap offload).
#
# TRANSLATION of EriDiffusion-v2 chroma.rs onto the parity-verified Mojo Chroma
# LoRA OFFLOAD stack (models/chroma/chroma_stack_lora.mojo). Real Chroma1-HD base
# weights (streamed block-by-block via TurboPlannedLoader), real prepared cache
# (latent + T5), full 19+38 block depth. No synthetic tensors. Mirrors
# train_flux_real.mojo's loop structure (timing, grad clip, shared progress
# display) and chroma.rs's recipe.
#
# CHROMA vs FLUX (the deltas; see chroma_stack_lora.mojo header):
#   - NO guidance / CLIP-pooled vector. Modulation comes from the FROZEN
#     distilled_guidance_layer APPROXIMATOR (models/dit/chroma_dit.mojo
#     ChromaDitCache.approximator_forward), producing a per-step pooled_temb
#     table [mod_index=344, D=3072]; each block's ModVecs are sliced rows.
#   - Block math IS the proven Flux block (after the loader's separate->fused
#     row-stack), so the per-block LoRA fwd/bwd is REUSED verbatim and the LoRA
#     carrier / AdamW is shared while save uses Chroma's SerenityTrainer raw-key API.
#
# Per step:
#   1. load cached {latent [1,16,64,64] RAW, t5_embed [1,seq,4096]}
#   2. latent_scaled = (latent - SHIFT) * SCALE  (Chroma VAE shift/scale)
#   3. pack_latents([16,64,64]) -> [N_IMG=1024, 64] channel-major patchify
#   4. sigma_idx = floor(logit_normal_sigma(shift=1.15) * 1000) clamp;
#      sig=(idx+1)/1000 ; t_model=idx/1000
#   5. noisy = noise*sig + latent_packed*(1-sig) ; target = noise - latent_packed
#   6. pooled_temb = approximator(t_model)  (frozen; once per step)
#   7. chroma_stack_lora_forward_offload(noisy, txt, pooled, ...) -> pred [N_IMG,64]
#   8. loss = MSE(pred, target); d_loss = (2/N)(pred - target)
#   9. chroma_stack_lora_backward_offload -> LoRA grads; global-norm clip(1.0)
#  10. flux_lora_adamw_step; print shared progress display
#
# Recipe scalars (configs/chroma.json / chroma.rs): lr=1e-4, rank=16, alpha=16,
#   timestep_shift=1.15, clip_grad_norm=1.0, VAE shift=0.1159 scale=0.3611.
#
# MEMORY: the Chroma transformer is ~8.9B params (BF16 17.8GB on disk). The
# OFFLOAD path streams one block at a time + holds the resident base
# (x_embedder/context_embedder/proj_out ~tiny) + the approximator (~loaded once)
# + LoRA optimizer state. FULL 19+38 depth is the default.
#
# FIXED_SIGMA_SMOKE: every step uses the SAME cache sample AND a fixed
# timestep+noise so a correct LoRA backward MUST drive loss DOWN monotonically
# (the canonical trainer-correctness gate, same probe as train_flux_real).
#
# Run (real smoke):
#   from the repository root, remove any stale local package artifact, then run \
#     pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/training/train_chroma_real.mojo -o /tmp/train_chroma_real && \
#     /tmp/train_chroma_real [steps]

from std.sys import argv
from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from std.time import perf_counter_ns
from std.os import listdir, makedirs
from std.ffi import external_call
from std.memory import alloc, UnsafePointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts

from serenitymojo.models.chroma.weights import load_chroma_stack_base
from serenitymojo.models.chroma.chroma_stack_lora import (
    ChromaStackBase,
    chroma_stack_lora_forward_offload, chroma_stack_lora_backward_offload,
    chroma_stack_lora_forward_device_offload, chroma_stack_lora_backward_device_offload,
    chroma_stack_lora_forward_device_offload_b2, chroma_stack_lora_backward_device_offload_b2,
    build_chroma_direct_dora_set_from_offload, build_chroma_direct_oft_set_for_stack,
    chroma_stack_direct_dora_forward_offload, chroma_stack_direct_dora_backward_offload,
    chroma_stack_direct_oft_forward_offload, chroma_stack_direct_oft_backward_offload,
    save_chroma_lora, save_chroma_lora_state,
    load_chroma_lora_resume, load_chroma_lora_state,
)
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet, FluxLoraGradSet, build_flux_lora_set,
    flux_lora_adamw_step, total_adapters,
    flux_lora_adamw_state_init, flux_lora_adamw_step_resident,
)
from serenitymojo.training.lora_adamw_plain_fused import (
    LoraAdamWPlainDeviceState,
    lora_adamw_plain_device_state_sync_params,
    lora_adamw_plain_device_state_sync_moments,
)
from serenitymojo.models.flux.flux_lycoris_stack import (
    FluxLoKrSet, empty_flux_lokr_set, build_flux_lokr_set,
    flux_lokr_carrier_set, flux_lokr_carrier_total_bytes,
    flux_lokr_chain_all, flux_lokr_adamw_step, flux_lokr_grad_norm,
    flux_lokr_clip_grads, flux_lokr_zero_leg_l1, save_flux_lokr,
    FluxLoHaSet, empty_flux_loha_set, build_flux_loha_set,
    flux_loha_carrier_set, flux_loha_carrier_total_bytes,
    flux_loha_chain_all, flux_loha_adamw_step, flux_loha_grad_norm,
    flux_loha_clip_grads, flux_loha_zero_leg_l1, save_flux_loha,
)
from serenitymojo.models.flux.flux_direct_lycoris_stack import (
    FLUX_DIRECT_24_GIB,
    empty_flux_direct_dora_set, empty_flux_direct_oft_set,
    flux_direct_dense_carrier_bytes,
    flux_direct_dora_preflight,
    flux_direct_oft_preflight,
    flux_direct_dora_grad_norm, flux_direct_dora_clip_grads,
    flux_direct_dora_adamw_step, flux_direct_dora_zero_leg_l1,
    flux_direct_dora_trainable_bytes, save_flux_direct_dora,
    flux_direct_oft_grad_norm, flux_direct_oft_clip_grads,
    flux_direct_oft_adamw_step, flux_direct_oft_vec_l1,
    flux_direct_oft_trainable_bytes, save_flux_direct_oft,
)
from serenitymojo.models.flux.lora_block import DBL_STREAM_SLOTS, SGL_SLOTS
from serenitymojo.models.dit.flux1_dit import build_flux1_rope_tables
from serenitymojo.models.dit.chroma_dit import ChromaDitCache
from serenitymojo.offload.plan import build_chroma1_hd_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.schedule import sample_timestep_logit_normal
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.training.lora_ema import (
    LoraEmaState, lora_ema_track, ema_update,
    lora_ema_adapters, ema_path_for_lora,
)
from serenitymojo.training.sample_prompt_config import (
    SampleCadence, read_sample_cadence_config,
    validate_step_sample_cadence, should_sample_completed_step,
    next_sample_completed_step, sample_time_unit_name,
    SAMPLE_UNIT_STEP, SAMPLE_UNIT_NEVER,
    SamplePrompt, SamplePromptConfig, read_sample_prompt_config,
    caps_sampling_active, assert_enabled_sample_prompts,
    warn_legacy_cached_caption_sampling,
)
from serenitymojo.training.serenity_trainer_train_loop_policy import (
    SERENITY_GRAD_POLICY_ON_ONLY,
    serenity_cache_dir_from_train_config,
    serenity_lr_for_optimizer_step,
    serenity_output_lora_path_from_train_config,
    serenity_sample_cadence_from_train_config,
    serenity_sampling_enabled,
    serenity_should_save_before_sample,
    serenity_should_save_checkpoint,
    serenity_step_lora_path,
    validate_serenity_gradient_checkpointing_policy,
    validate_serenity_train_math_policy,
)
from serenitymojo.training.train_config import (
    TrainConfig, GRADIENT_CHECKPOINTING_ON,
    TRAIN_ADAPTER_ALGO_LORA, TRAIN_ADAPTER_ALGO_FULL,
    TRAIN_ADAPTER_ALGO_LOCON, TRAIN_ADAPTER_ALGO_LOHA,
    TRAIN_ADAPTER_ALGO_DORA, TRAIN_ADAPTER_ALGO_LOKR,
    TRAIN_ADAPTER_ALGO_OFT, TRAIN_ADAPTER_ALGO_BOFT,
)
from serenitymojo.training.adapter_algo_policy import adapter_algo_name
from serenitymojo.training.trainer_core import (
    GradAccumWindow, trainer_prune_target_step, trainer_prune_step_checkpoint,
)
from serenitymojo.training.lokr_stack import LOKR_CARRIER_MAX_DEVICE_BYTES
from serenitymojo.training.serenity_trainer_cache_preflight import (
    create_serenity_trainer_cache_preflight_plan,
    validate_serenity_trainer_cache_preflight_plan,
)
from serenitymojo.training.chroma_sample_resident import (
    chroma_sample_resident, chroma_decode_latent_to_png,
)


# ── arch (chroma1-hd; H/Dh/D fixed comptime, verified vs the checkpoint) ─────
comptime H = 24
comptime Dh = 128
comptime D = H * Dh            # 3072
comptime FMLP = 12288          # mlp_hidden = D*4
comptime IN_CH = 64            # patch_dim = 16ch * 2*2
comptime TXT_CH = 4096         # T5-XXL hidden
comptime OUT_CH = 64
comptime NUM_DOUBLE = 19
comptime NUM_SINGLE = 38
comptime MOD_INDEX = 3 * NUM_SINGLE + 2 * 6 * NUM_DOUBLE + 2   # 344
comptime EPS = Float32(1e-06)

# ── resolution (512px): latent [16,64,64] -> pack2 -> 32x32=1024 img tokens ──
comptime LAT_C = 16
comptime LAT_H = 64
comptime LAT_W = 64
comptime PATCH = 2
comptime HT = LAT_H // PATCH   # 32
comptime WT = LAT_W // PATCH   # 32
comptime N_IMG = HT * WT       # 1024
comptime N_TXT = 512           # T5 padded length
comptime S = N_TXT + N_IMG     # 1536

# ── recipe (configs/chroma.json) ─────────────────────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime LR = Float32(1.0e-4)
comptime TIMESTEP_SHIFT = Float32(1.15)
comptime VAE_SHIFT = Float32(0.1159)
comptime VAE_SCALE = Float32(0.3611)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP_GRAD_NORM = Float32(1.0)
comptime SEED_BASE = UInt64(42)

comptime FIXED_SIGMA_SMOKE = True
comptime FIXED_SIGMA_IDX = 500

comptime CKPT = "models/chroma/transformer"
comptime CACHE_DIR = "output/chroma/cache"
comptime LORA_DIR = "output/chroma"
comptime DEFAULT_CONFIG = "serenitymojo/configs/chroma.json"
comptime DEFAULT_RUN_STEPS = 5

# ── sample-during-training (v1; chroma_sample_resident) ──────────────────────
# When the existing SampleCadence fires (should_sample_completed_step), denoise a
# sample from the CURRENT frozen base + streamed blocks + LIVE LoRA, decode with
# the FLUX VAE, and write <LORA_DIR>/samples/step_<N>.png. Geometry is the
# trainer's 512px latent (LAT_H=LAT_W=64 -> 8x VAE -> 512x512 image).
#   SAMPLE_STEPS / SAMPLE_CFG : denoise loop length + CFG (sampler defaults 30/4.0
#                               — chroma_sample_cli.mojo NUM_STEPS/GUIDANCE).
#   SAMPLE_SEED               : base RNG seed for the t=1 packed init noise.
# v1 CONDITIONING (flagged): no in-tree T5 tokenizer, so the COND text is the
#   CURRENT step's cached caption T5 embeds (the loop's txt_tokens); UNCOND is a
#   zero vector. See chroma_sample_resident.mojo header for the why + drop-in path.
comptime SAMPLE_STEPS = 30
comptime SAMPLE_CFG = Float32(4.0)
comptime SAMPLE_SEED = UInt64(0xC4_303A_5A91)


def _is_nonnegative_int(s: String) -> Bool:
    if s.byte_length() == 0:
        return False
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        if bs[i] < 0x30 or bs[i] > 0x39:
            return False
    return True


def _parse_nonnegative_int(s: String) raises -> Int:
    if not _is_nonnegative_int(s):
        raise Error(String("expected non-negative integer, got ") + s)
    var out = 0
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        out = out * 10 + Int(bs[i] - 0x30)
    return out


def _close_f32(a: Float32, b: Float32, tol: Float32 = Float32(1.0e-7)) -> Bool:
    var d = a - b
    if d < Float32(0.0):
        d = -d
    return d <= tol


def validate_chroma_train_config(cfg: TrainConfig) raises:
    if cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOCON:
        print("[Chroma-locon] network_algorithm=locon: using the linear LoRA-compatible down/up path")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR:
        print("[Chroma-lokr] network_algorithm=lokr: using block-projection carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA:
        print("[Chroma-loha] network_algorithm=loha: using block-projection carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA or cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT:
        print(
            String("[Chroma-direct] network_algorithm=")
            + adapter_algo_name(cfg.adapter_algo)
            + String(": using direct W_eff stack dispatch; sample cadence must be disabled")
        )
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_BOFT:
        raise Error("Chroma trainer: BOFT is intentionally excluded; use lora, locon, loha, lokr, dora, or oft where wired")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_FULL:
        raise Error("Chroma trainer: full finetune is not wired; supported here: lora, locon, loha, lokr, dora, oft")
    elif cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA:
        raise Error(
            String("Chroma trainer: network_algorithm=")
            + adapter_algo_name(cfg.adapter_algo)
            + String(" is not wired; supported here: lora, locon, loha, lokr, dora, oft")
        )
    if cfg.checkpoint == String(""):
        raise Error("Chroma trainer config must set checkpoint")
    if cfg.n_heads != H:
        raise Error(String("Chroma config n_heads ") + String(cfg.n_heads) + String(" != H ") + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("Chroma config head_dim ") + String(cfg.head_dim) + String(" != Dh ") + String(Dh))
    if cfg.d_model != D:
        raise Error(String("Chroma config d_model ") + String(cfg.d_model) + String(" != D ") + String(D))
    if cfg.in_channels != IN_CH:
        raise Error(String("Chroma config in_channels ") + String(cfg.in_channels) + String(" != IN_CH ") + String(IN_CH))
    if cfg.joint_attention_dim != TXT_CH:
        raise Error(String("Chroma config joint_attention_dim ") + String(cfg.joint_attention_dim) + String(" != TXT_CH ") + String(TXT_CH))
    if cfg.out_channels != OUT_CH:
        raise Error(String("Chroma config out_channels ") + String(cfg.out_channels) + String(" != OUT_CH ") + String(OUT_CH))
    if cfg.num_double != NUM_DOUBLE or cfg.num_single != NUM_SINGLE:
        raise Error(
            String("Chroma trainer requires double=") + String(NUM_DOUBLE)
            + String(" single=") + String(NUM_SINGLE)
            + String("; got double=") + String(cfg.num_double)
            + String(" single=") + String(cfg.num_single)
        )
    if cfg.mlp_hidden != FMLP:
        raise Error(String("Chroma config mlp_hidden ") + String(cfg.mlp_hidden) + String(" != FMLP ") + String(FMLP))
    if cfg.lora_rank != RANK:
        raise Error(
            String("Chroma trainer is compiled for lora_rank=")
            + String(RANK)
            + String("; parsed ")
            + String(cfg.lora_rank)
        )
    if not _close_f32(cfg.lora_alpha, ALPHA):
        raise Error("Chroma trainer lora_alpha does not match compiled constant")
    if not _close_f32(cfg.lr, LR, Float32(1.0e-9)):
        raise Error("Chroma trainer learning_rate does not match compiled constant")
    if not _close_f32(cfg.timestep_shift, TIMESTEP_SHIFT):
        raise Error("Chroma trainer timestep_shift does not match compiled constant")
    if not _close_f32(cfg.max_grad_norm, CLIP_GRAD_NORM):
        raise Error("Chroma trainer max_grad_norm does not match compiled constant")
    validate_serenity_train_math_policy(cfg, String("Chroma trainer"))
    validate_serenity_gradient_checkpointing_policy(
        cfg, String("Chroma trainer"), SERENITY_GRAD_POLICY_ON_ONLY
    )


def chroma_checkpoint_from_train_config(cfg: TrainConfig) -> String:
    if cfg.checkpoint != String(""):
        return cfg.checkpoint.copy()
    return String(CKPT)


def chroma_cache_dir_from_train_config(cfg: TrainConfig) -> String:
    return serenity_cache_dir_from_train_config(cfg, String(CACHE_DIR))


def chroma_output_lora_path_from_train_config(cfg: TrainConfig, completed_step: Int) -> String:
    return serenity_output_lora_path_from_train_config(
        cfg, String(LORA_DIR), String("chroma_lora"), completed_step
    )


def chroma_sample_cadence_from_train_config(
    cfg_path: String, cfg: TrainConfig,
) raises -> SampleCadence:
    return serenity_sample_cadence_from_train_config(cfg_path, cfg)


def chroma_sampling_enabled(cadence: SampleCadence) -> Bool:
    return serenity_sampling_enabled(cadence)


def chroma_should_save_checkpoint(cfg: TrainConfig, completed_step: Int) -> Bool:
    return serenity_should_save_checkpoint(cfg, completed_step)


def chroma_should_save_before_sample(
    cadence: SampleCadence, completed_step: Int, saved_this_step: Bool,
) raises -> Bool:
    return serenity_should_save_before_sample(cadence, completed_step, saved_this_step)


def _step_lora_path(base_path: String, step: Int) -> String:
    return serenity_step_lora_path(base_path, step)


# Rolling checkpoint retention (audit item #4), pruned AFTER a periodic save —
# krea2's discipline, thin wrapper over the shared trainer_core machinery. Reuses
# the shared keep-count decision (trainer_prune_target_step), but builds the
# pruned path with chroma's OWN step-path helper (chroma names under LORA_DIR via
# the reference-policy scheme, not krea2's workspace/stem), then removes it + its
# `.state.safetensors` sidecar. keep_default/milestone=0 ⇒ NO prune until the
# webui sets save_max_keep, so keep-all stays byte-unchanged when it is unset.
def _chroma_prune_old_checkpoints(cfg: TrainConfig, run_steps: Int, saved_step: Int) raises:
    var old = trainer_prune_target_step(cfg, saved_step, 0, 0)
    if old > 0:
        trainer_prune_step_checkpoint(
            _step_lora_path(chroma_output_lora_path_from_train_config(cfg, run_steps), old),
            String(".state.safetensors"),
        )


# ── deterministic host gaussian noise (Box-Muller PCG; per-step seed) ─────────
def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^


def _absum(v: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        s += x if x >= 0.0 else -x
    return s


def _absum(v: List[BFloat16]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i].cast[DType.float32]()
        s += x if x >= 0.0 else -x
    return s


comptime _EnvPtr = UnsafePointer[UInt8, MutExternalOrigin]


def _env_is_set(name: String) -> Bool:
    # CHROMA_DEVICE_STACK=1 selects the device-resident fwd/bwd arm; unset/other = host.
    var n = name.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = name.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var cname = _EnvPtr(unsafe_from_address=Int(buf))
    var ret = external_call["getenv", _EnvPtr](cname)
    buf.free()
    if Int(ret) == 0:
        return False
    return ret[0] == UInt8(49) and ret[1] == UInt8(0)


def _global_norm(grads: FluxLoraGradSet) -> Float64:
    var ss = 0.0
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: FluxLoraGradSet, max_norm: Float32) -> Float64:
    var gn = _global_norm(grads)
    if gn <= Float64(max_norm) or gn == 0.0:
        return gn
    var s = Float32(Float64(max_norm) / gn)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s
    return gn


def _list_cache(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var fs = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            fs.append(dir + String("/") + raw[i])
    if len(fs) == 0:
        raise Error(String("chroma cache: no .safetensors in ") + dir)
    for i in range(1, len(fs)):
        var j = i
        while j > 0 and fs[j - 1] > fs[j]:
            var tmp = fs[j - 1]
            fs[j - 1] = fs[j]
            fs[j] = tmp
            j -= 1
    return fs^


def _load_chroma_cache_tensor(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


# T1.B: save the EMA shadow set as the *_ema.safetensors sibling of a plain-LoRA
# checkpoint (SimpleTuner copy_to analog — lora_ema.mojo lora_ema_adapters
# returns bf16-rounded shadows; keep the live set's shapes/scale/prefix). Only
# reached when cfg.ema_enabled; flag-off leaves this uncalled (baseline bytes
# unchanged).
def _save_chroma_lora_ema(
    ema: LoraEmaState, lora: FluxLoraSet, lora_path: String,
    n_adapters: Int, ctx: DeviceContext
) raises:
    var ema_set = lora.copy()
    var shadow_ads = lora_ema_adapters(ema, lora.ad, 0, n_adapters, 0)
    for i in range(len(shadow_ads)):
        ema_set.ad[i] = shadow_ads[i].copy()
    var ema_path = ema_path_for_lora(lora_path)
    _ = save_chroma_lora(ema_set, ema_path, ctx)
    print("[Chroma-lora] save_ema path=", ema_path)


def _host_f32_for_step_math(t: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    """Stage tensors through their stored dtype before host step math."""
    if t.dtype() == STDtype.BF16:
        var bf = t.to_host_bf16(ctx)
        var out = List[Float32]()
        for i in range(len(bf)):
            out.append(bf[i].cast[DType.float32]())
        return out^
    if t.dtype() == STDtype.F16:
        var hf = t.to_host_f16(ctx)
        var out = List[Float32]()
        for i in range(len(hf)):
            out.append(hf[i].cast[DType.float32]())
        return out^
    return t.to_host(ctx)


# pack_latents: [16,LAT_H,LAT_W] flat -> [N_IMG, 64] channel-major patchify.
def _pack_latents(lat: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for ih in range(HT):
        for iw in range(WT):
            for c in range(LAT_C):
                for ph in range(PATCH):
                    for pw in range(PATCH):
                        var hh = ih * PATCH + ph
                        var ww = iw * PATCH + pw
                        var idx = c * LAT_H * LAT_W + hh * LAT_W + ww
                        out.append(lat[idx])
    return out^


# Build the frozen per-step modulation table [MOD_INDEX*D] as a Tensor.
def _pooled_modulation_tensor(
    approx: ChromaDitCache, t_model: Float32, ctx: DeviceContext,
) raises -> Tensor:
    var approx_in = approx._approximator_input(t_model, ctx)
    return approx.approximator_forward(approx_in, ctx)   # [1, MOD_INDEX, D] BF16


# ── deterministic host gaussian PACKED init noise [N_IMG*IN_CH] ──────────────
# Reuses the same Box-Muller PCG as the train loop's _host_noise (so the sample's
# t=1 latent is drawn the same way the training noise is). seed makes it
# deterministic per sampled step.
def _sample_init_noise(seed: UInt64) -> List[Float32]:
    return _host_noise(N_IMG * IN_CH, seed)


# ── _chroma_run_sample — one sample-during-training image ────────────────────
#   cond text    : the current step's cached caption T5 embeds (v1; see header).
#   uncond text  : a zeroed [N_TXT*TXT_CH] vector (CFG empty cond).
#   init noise   : packed gaussian [N_IMG*IN_CH], seed = SAMPLE_SEED + step.
#   denoise      : chroma_sample_resident (frozen base + streamed blocks + live
#                  LoRA + frozen approximator).
#   decode+write : chroma_decode_latent_to_png -> <samples_dir>/step_<N>.png.
# Fail-loud: any raise propagates (no silent skip), matching the trainer's
# fail-loud cadence contract.
def _chroma_run_sample(
    base: ChromaStackBase,
    approx: ChromaDitCache,
    mut loader: TurboPlannedLoader,
    lora: FluxLoraSet,
    cond_txt: List[Float32],     # [N_TXT*TXT_CH] — the step's cached caption embeds
    cos: List[Float32],
    sin: List[Float32],
    samples_dir: String,
    vae_path: String,
    step: Int,
    ctx: DeviceContext,
) raises:
    # UNCOND: zeroed text features (same dtype/shape as cond_txt).
    var uncond_txt = List[Float32]()
    for _ in range(N_TXT * TXT_CH):
        uncond_txt.append(Float32(0.0))

    var init_noise = _sample_init_noise(SAMPLE_SEED + UInt64(step))

    var latent = chroma_sample_resident[H, Dh, N_IMG, N_TXT, S](
        base, approx, loader, lora,
        cond_txt.copy(), uncond_txt^, init_noise^,
        cos.copy(), sin.copy(),
        SAMPLE_STEPS, SAMPLE_CFG, MOD_INDEX,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )

    var out_path = samples_dir + String("/step_") + String(step) + String(".png")
    chroma_decode_latent_to_png[LAT_C, LAT_H, LAT_W, HT, WT, PATCH, N_IMG, IN_CH](
        latent, vae_path, out_path, ctx,
    )
    print("[Chroma-lora] sample step=", step, " -> ", out_path)


# ── per-prompt validation caps (serenity.sample_prompts.v1) ──────────────────
# A prompt's caps_pos/caps_neg point at a CACHE-ENTRY-SHAPED safetensors carrying
# the "t5_embed" key (same key + layout the train loop reads from its cache).
# `_chroma_txt_cond_from_caps` reuses the loop's cache->cond build (pad/truncate
# to [N_TXT, TXT_CH]) sourced from that file, so the sample conditioning is the
# real prompt's T5 embeds, NOT the current step's cached caption (MJ-1045 trap).
def _chroma_txt_cond_from_caps(path: String, label: String, ctx: DeviceContext) raises -> List[Float32]:
    if path == String(""):
        raise Error(String("Chroma sample prompt ") + label + String(": empty caps path"))
    var st = SafeTensors.open(path)
    var info = st.tensor_info(String("t5_embed"))
    if len(info.shape) < 2 or Int(info.shape[len(info.shape) - 1]) != TXT_CH:
        raise Error(
            String("Chroma caps for prompt '") + label + String("' at ") + path
            + String(": expected cache-shaped safetensors key 't5_embed' shape [1,LT,")
            + String(TXT_CH) + String("] (LT<=") + String(N_TXT)
            + String(", zero-padded to ") + String(N_TXT)
            + String("); got last-dim ") + String(Int(info.shape[len(info.shape) - 1]))
        )
    var t5_seq = Int(info.shape[len(info.shape) - 2])
    var t5_cache = _load_chroma_cache_tensor(st, String("t5_embed"), ctx)
    var t5_flat = _host_f32_for_step_math(t5_cache, ctx)
    var txt = List[Float32]()
    for r in range(N_TXT):
        if r < t5_seq:
            for c in range(TXT_CH):
                txt.append(t5_flat[r * TXT_CH + c])
        else:
            for _ in range(TXT_CH):
                txt.append(Float32(0.0))
    return txt^


def _chroma_sample_prompt_config_for_sampler(sample_file: String) raises -> SamplePromptConfig:
    if sample_file == String(""):
        raise Error("Chroma trainer caps sampling requires validation_prompts_file")
    var cfg = read_sample_prompt_config(sample_file)
    assert_enabled_sample_prompts(cfg, String("Chroma"))
    return cfg^


# ctx-free preflight: fail loud (naming the prompt) if any enabled prompt is not a
# valid 512x512 image prompt with a shape-correct caps.positive (and caps.negative
# when its CFG uses one), BEFORE the train loop starts.
def _chroma_preflight_sample_caps(sample_cfg: SamplePromptConfig) raises:
    var checked = 0
    for i in range(len(sample_cfg.prompts)):
        var p = sample_cfg.prompts[i].copy()
        if not p.enabled:
            continue
        if p.frames != 1:
            raise Error(String("Chroma sample prompt ") + p.label + String(": only single-frame image samples supported"))
        if p.width != LAT_W * 8 or p.height != LAT_H * 8:
            raise Error(
                String("Chroma sample prompt ") + p.label + String(": requests ")
                + String(p.width) + String("x") + String(p.height)
                + String(" but this binary samples ") + String(LAT_W * 8)
                + String("x") + String(LAT_H * 8)
            )
        _chroma_check_caps_shape(p.caps_pos, p.label)
        if p.cfg != Float32(1.0) and p.caps_neg != String(""):
            _chroma_check_caps_shape(p.caps_neg, p.label)
        checked += 1
    if checked == 0:
        raise Error("Chroma trainer requires at least one enabled validation prompt when caps sampling is enabled")


def _chroma_check_caps_shape(path: String, label: String) raises:
    if path == String(""):
        raise Error(String("Chroma sample prompt ") + label + String(": empty caps path"))
    var st = SafeTensors.open(path)
    var info = st.tensor_info(String("t5_embed"))
    if len(info.shape) < 2 or Int(info.shape[len(info.shape) - 1]) != TXT_CH:
        raise Error(
            String("Chroma caps for prompt '") + label + String("' at ") + path
            + String(": expected cache-shaped safetensors key 't5_embed' shape [1,LT,")
            + String(TXT_CH) + String("] (LT<=") + String(N_TXT)
            + String(", zero-padded to ") + String(N_TXT)
            + String("); got last-dim ") + String(Int(info.shape[len(info.shape) - 1]))
        )


# One caps-conditioned validation image (per-prompt seed/steps/cfg/label). Mirrors
# _chroma_run_sample but sources COND (and optional negative UNCOND) from a prompt's
# caps instead of the step's cached caption.
def _chroma_run_sample_caps(
    base: ChromaStackBase,
    approx: ChromaDitCache,
    mut loader: TurboPlannedLoader,
    lora: FluxLoraSet,
    cond_txt: List[Float32],
    uncond_txt: List[Float32],     # empty => zeroed CFG-empty cond
    cos: List[Float32],
    sin: List[Float32],
    samples_dir: String,
    vae_path: String,
    step: Int,
    prompt: SamplePrompt,
    seed: UInt64,
    ctx: DeviceContext,
) raises:
    var uncond = uncond_txt.copy()
    if len(uncond) == 0:
        for _ in range(N_TXT * TXT_CH):
            uncond.append(Float32(0.0))
    var init_noise = _sample_init_noise(seed)
    var latent = chroma_sample_resident[H, Dh, N_IMG, N_TXT, S](
        base, approx, loader, lora,
        cond_txt.copy(), uncond^, init_noise^,
        cos.copy(), sin.copy(),
        prompt.steps, prompt.cfg, MOD_INDEX,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )
    var out_path = (
        samples_dir + String("/step_") + String(step)
        + String("_") + prompt.label + String(".png")
    )
    chroma_decode_latent_to_png[LAT_C, LAT_H, LAT_W, HT, WT, PATCH, N_IMG, IN_CH](
        latent, vae_path, out_path, ctx,
    )
    print("[Chroma-lora] caps sample step=", step, " prompt=", prompt.label, " -> ", out_path)


def main() raises:
    var a = argv()
    var cfg_path = String(DEFAULT_CONFIG)
    var arg_base = 1
    if len(a) >= 2:
        var first = String(a[1])
        if first.endswith(String(".json")):
            cfg_path = first.copy()
            arg_base = 2

    var train_cfg = read_model_config(cfg_path)
    validate_chroma_train_config(train_cfg)
    var cache_preflight = create_serenity_trainer_cache_preflight_plan(train_cfg)
    validate_serenity_trainer_cache_preflight_plan(cache_preflight)

    var run_steps = DEFAULT_RUN_STEPS
    var resume_path = String("")
    var start_step = 0
    if len(a) > arg_base:
        run_steps = _parse_nonnegative_int(String(a[arg_base]))
    elif train_cfg.only_cache:
        run_steps = 0
    if len(a) > arg_base + 1:
        resume_path = String(a[arg_base + 1])
    if len(a) > arg_base + 2:
        start_step = _parse_nonnegative_int(String(a[arg_base + 2]))
    if resume_path != String(""):
        if start_step <= 0 or start_step >= run_steps:
            raise Error(
                "train_chroma_real: resume requires 0 < start_step < requested total steps"
            )
    elif start_step != 0:
        raise Error("train_chroma_real: start_step requires a resume path")

    var ckpt = chroma_checkpoint_from_train_config(train_cfg)
    var cache_dir = chroma_cache_dir_from_train_config(train_cfg)
    var sample_cadence = chroma_sample_cadence_from_train_config(cfg_path, train_cfg)
    var sample_enabled = chroma_sampling_enabled(sample_cadence)
    var direct_algo_requested = (
        train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA
        or train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT
    )
    if sample_enabled and direct_algo_requested:
        raise Error(
            "Chroma direct DoRA/OFT sample-during-training is not wired; disable sample cadence for this runtime gate"
        )

    print("=== Chroma (chroma1-hd) REAL LoRA training loop (block-swap offload) ===")
    print("  config:", cfg_path)
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " Fmlp=", FMLP, " out_ch=", OUT_CH)
    print("  depth: NUM_DOUBLE=", NUM_DOUBLE, " NUM_SINGLE=", NUM_SINGLE, " mod_index=", MOD_INDEX)
    print("  tokens: N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S)
    print("  recipe: rank=", train_cfg.lora_rank, " alpha=", train_cfg.lora_alpha,
          " lr=", train_cfg.lr, " shift=", train_cfg.timestep_shift,
          " vae_shift=", VAE_SHIFT, " vae_scale=", VAE_SCALE)
    print("  run_steps=", run_steps, " config_max_steps=", train_cfg.max_steps)
    print(
        "  cadence: save_every=", train_cfg.save_every,
        " sample_after=", sample_cadence.sample_after,
        " unit=", sample_time_unit_name(sample_cadence.sample_after_unit),
        " skip_first=", sample_cadence.sample_skip_first,
        " sample_file=", sample_cadence.sample_definition_file_name,
    )
    print("  fixed_sigma_smoke=", FIXED_SIGMA_SMOKE)
    print("  ckpt:", ckpt)
    print("  cache:", cache_dir)
    if train_cfg.enable_async_offloading:
        print("[offload] async offload requested by config; Chroma trainer currently uses synchronous TurboPlannedLoader")
    if train_cfg.only_cache:
        print("[Chroma] only_cache requested; no train steps will run in this trainer")
        return

    var ctx = DeviceContext()

    # ── stack-level base (frozen; x_embedder/context_embedder/proj_out) ──────
    print("[load] ChromaStackBase (x_embedder, context_embedder, proj_out)")
    var base_st = ShardedSafeTensors.open(ckpt)
    var base = load_chroma_stack_base(base_st, NUM_DOUBLE, NUM_SINGLE, ctx)
    print("[load] base resident")

    # ── frozen approximator (distilled_guidance_layer) ───────────────────────
    print("[load] approximator (distilled_guidance_layer)")
    var approx = ChromaDitCache.load(ckpt, ctx)
    print("[load] approximator resident")

    # ── block-swap offload loader ────────────────────────────────────────────
    var plan = build_chroma1_hd_block_plan()
    var cfg = OffloadConfig.synchronous_single()
    # fp8_e4m3 pins EVERY block device-resident — skip the whole-DiT pinned
    # host block store (~17 GB, never read again; 2× concurrent = host OOM).
    var loader = TurboPlannedLoader.open(
        ckpt, plan^, cfg, ctx,
        fill_block_store=train_cfg.quantized_resident != String("fp8_e4m3"),
    )
    print("[load] offload loader opened (", loader.block_count(), "blocks)")

    # ── T2.B residency policy (MJ-1065, user directive 2026-07-03) ────────────
    # Chroma1-HD is a ~17GB bf16 DiT; the streamed base re-reads the WHOLE DiT
    # from the host pin pool EVERY forward (~136-147s/step, a per-step DISK read
    # forbidden by policy). fp8_e4m3 = quantize the 2-D matmul weights ONCE to
    # E4M3 + per-row F32 scale, held resident (~9GB, fits 24GB with LoRA state),
    # dequant per block in the step → ZERO per-step disk. streamed_base_opt_in =
    # the explicit slow bf16 disk-stream arm (quality-control A/Bs only). Any
    # other value (incl the "OFF" default) FAILS LOUD — the streamed base is not
    # a silent default. All adapter paths (lora/dora/oft) + the inline sampler go
    # through loader.await_block, so pinning here converts every one at once.
    var quant_tag = train_cfg.quantized_resident.copy()
    if quant_tag == String("fp8_e4m3"):
        print("[quant] fp8_e4m3-resident base: quantizing",
              loader.block_count(), "blocks ONCE at load ...")
        var pinned = loader.pin_residents_fp8(20 * 1024 * 1024 * 1024, ctx)
        if pinned != loader.block_count():
            raise Error(
                String("Chroma fp8-resident: pinned ") + String(pinned) + String(" of ")
                + String(loader.block_count())
                + String(" blocks (budget too small) — MJ-1065 forbids a partial-")
                + String("resident base that would per-step disk-stream the rest")
            )
        print("[quant] fp8_e4m3-resident base: DONE (", pinned,
              "blocks; NO per-step disk read in the step).")
    elif quant_tag == String("streamed_base_opt_in"):
        print("[quant] streamed_base_opt_in: per-step bf16 DISK stream",
              "(EXPLICIT slow experiment arm).")
    else:
        raise Error(
            String("Chroma: quantized_resident='") + quant_tag
            + String("' selects the per-step DISK-STREAM base, forbidden by policy ")
            + String("MJ-1065. Use \"fp8_e4m3\" (resident base) or ")
            + String("\"streamed_base_opt_in\" to explicitly run the slow streamed arm.")
        )

    # ── 3-axis RoPE tables (positions fixed for 512px; built once, BF16) ─────
    var rope = build_flux1_rope_tables[N_IMG, N_TXT, H, Dh](HT, WT, ctx, STDtype.BF16)
    var cos = rope[0].to_host(ctx)
    var sin = rope[1].to_host(ctx)
    print("[load] chroma 3-axis rope tables built (S*H x Dh/2)")

    # ── LoRA set (B=0 init -> identity at step 0) ────────────────────────────
    var lokr_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR
    var loha_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA
    var dora_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA
    var oft_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT
    var direct_active = dora_active or oft_active
    var carrier_active = lokr_active or loha_active
    var lycoris_active = carrier_active or direct_active
    var direct_targets = 1 if train_cfg.lokr_targets == 1 else 2
    var direct_oft_block_size = 4
    var lora = build_flux_lora_set(0, 0, D, FMLP, RANK, ALPHA)
    var n_adapters = 0
    if not direct_active:
        lora = build_flux_lora_set(NUM_DOUBLE, NUM_SINGLE, D, FMLP, RANK, ALPHA)
        n_adapters = total_adapters(lora)
    if resume_path != String(""):
        if lycoris_active:
            raise Error("train_chroma_real: resume is currently admitted for LoRA/LoCon only")
        if resume_path.endswith(String(".state")) or resume_path.endswith(String(".state.safetensors")):
            lora = load_chroma_lora_state(
                NUM_DOUBLE, NUM_SINGLE, RANK, ALPHA, resume_path, ctx
            )
            print("[Chroma-resume] full state loaded at step", start_step, "from", resume_path)
        else:
            lora = load_chroma_lora_resume(
                NUM_DOUBLE, NUM_SINGLE, RANK, ALPHA, resume_path, ctx
            )
            print("[Chroma-resume] weights-only state loaded at step", start_step, "from", resume_path)
    var lokr_masters = empty_flux_lokr_set()
    var loha_masters = empty_flux_loha_set()
    var dora_masters = empty_flux_direct_dora_set()
    var oft_masters = empty_flux_direct_oft_set()
    if lokr_active:
        lokr_masters = build_flux_lokr_set(
            NUM_DOUBLE, NUM_SINGLE, D, FMLP,
            RANK, ALPHA,
            train_cfg.lokr_factor, train_cfg.lokr_factor_attn,
            train_cfg.lokr_factor_ff,
            train_cfg.lokr_decompose_both, train_cfg.lokr_full_matrix,
            direct_targets,
            UInt64(910701),
        )
        var carrier_bytes = flux_lokr_carrier_total_bytes(lokr_masters, D, FMLP)
        print("[Chroma-lokr] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("Chroma LoKr: carrier set needs ")
                + String(carrier_bytes)
                + String(" bytes (> budget). Use a smaller lokr_factor/rank or restrict lokr_targets.")
            )
        lora = flux_lokr_carrier_set(lokr_masters, D, FMLP)
        print("[Chroma-lokr] carrier set materialized:", len(lora.ad), "adapters")
    elif loha_active:
        loha_masters = build_flux_loha_set(
            NUM_DOUBLE, NUM_SINGLE, D, FMLP,
            RANK, ALPHA,
            direct_targets,
            UInt64(910801),
        )
        var carrier_bytes = flux_loha_carrier_total_bytes(loha_masters, D, FMLP)
        print("[Chroma-loha] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("Chroma LoHa: carrier set needs ")
                + String(carrier_bytes)
                + String(" bytes (> budget). Reduce lora_rank or restrict lokr_targets.")
        )
        lora = flux_loha_carrier_set(loha_masters, D, FMLP)
        print("[Chroma-loha] carrier set materialized:", len(lora.ad), "adapters")
    elif dora_active:
        var dense_bytes = flux_direct_dense_carrier_bytes(NUM_DOUBLE, NUM_SINGLE, D, FMLP, direct_targets)
        var direct_bytes = flux_direct_dora_preflight(
            NUM_DOUBLE, NUM_SINGLE, D, FMLP, RANK, direct_targets, FLUX_DIRECT_24_GIB, False,
        )
        print("[Chroma-dora] dense carrier bytes:", dense_bytes,
              " direct trainable bytes:", direct_bytes,
              " budget:", FLUX_DIRECT_24_GIB)
        print("[Chroma-dora] initializing DoRA magnitudes from streamed Chroma block weights ...")
        dora_masters = build_chroma_direct_dora_set_from_offload(
            loader, NUM_DOUBLE, NUM_SINGLE, D, FMLP,
            RANK, ALPHA, direct_targets, train_cfg.seed * UInt64(59) + UInt64(7100),
            False, ctx,
        )
        print("[Chroma-dora] trainable bytes:", flux_direct_dora_trainable_bytes(dora_masters),
              " slots:", len(dora_masters.ad))
    elif oft_active:
        var dense_bytes = flux_direct_dense_carrier_bytes(NUM_DOUBLE, NUM_SINGLE, D, FMLP, direct_targets)
        var direct_bytes = flux_direct_oft_preflight(
            NUM_DOUBLE, NUM_SINGLE, D, FMLP, direct_oft_block_size, direct_targets, FLUX_DIRECT_24_GIB,
        )
        print("[Chroma-oft] dense carrier bytes:", dense_bytes,
              " direct trainable bytes:", direct_bytes,
              " block_size:", direct_oft_block_size,
              " budget:", FLUX_DIRECT_24_GIB)
        oft_masters = build_chroma_direct_oft_set_for_stack(
            NUM_DOUBLE, NUM_SINGLE, D, FMLP, direct_oft_block_size, direct_targets,
        )
        print("[Chroma-oft] trainable bytes:", flux_direct_oft_trainable_bytes(oft_masters),
              " slots:", len(oft_masters.ad))
    if dora_active:
        print("[Chroma-dora] direct block slots:", len(dora_masters.ad),
              " (", DBL_STREAM_SLOTS * 2, "x", NUM_DOUBLE, "double +",
              SGL_SLOTS, "x", NUM_SINGLE, "single)")
    elif oft_active:
        print("[Chroma-oft] direct block slots:", len(oft_masters.ad),
              " (", DBL_STREAM_SLOTS * 2, "x", NUM_DOUBLE, "double +",
              SGL_SLOTS, "x", NUM_SINGLE, "single)")
    else:
        print("[lora] adapters:", n_adapters,
              " (", DBL_STREAM_SLOTS * 2, "x", NUM_DOUBLE, "double +",
              SGL_SLOTS, "x", NUM_SINGLE, "single)")

    var files = _list_cache(cache_dir)
    print("[cache] samples:", len(files))

    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    if carrier_active:
        print("[lora] carrier LoRA-B |.|_1 at init =", b_absum_init)
    elif dora_active:
        print("[Chroma-dora] direct trainable L1 at init =", flux_direct_dora_zero_leg_l1(dora_masters))
    elif oft_active:
        print("[Chroma-oft] direct trainable L1 at init =", flux_direct_oft_vec_l1(oft_masters))
    else:
        print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")
    var carrier_zero_init = Float64(0.0)
    if lokr_active:
        carrier_zero_init = flux_lokr_zero_leg_l1(lokr_masters)
        print("[Chroma-lokr] zero-leg L1 at init =", carrier_zero_init)
    elif loha_active:
        carrier_zero_init = flux_loha_zero_leg_l1(loha_masters)
        print("[Chroma-loha] zero-leg L1 at init =", carrier_zero_init)
    elif dora_active:
        carrier_zero_init = flux_direct_dora_zero_leg_l1(dora_masters)
        print("[Chroma-dora] zero-leg L1 at init =", carrier_zero_init)
    elif oft_active:
        carrier_zero_init = flux_direct_oft_vec_l1(oft_masters)
        print("[Chroma-oft] vec L1 at init =", carrier_zero_init)

    # samples dir for sample-during-training PNGs (created once if enabled).
    var samples_dir = train_cfg.workspace_dir + String("/samples")
    var sample_vae_path = train_cfg.vae.copy()
    # STANDARD sample-prompts contract: caps sampling is ACTIVE when the config
    # names a validation_prompts_file. Load+preflight the per-prompt caps here
    # (fail loud before the run); otherwise the seam falls back to the legacy
    # cached-caption render with a LOUD warning.
    var caps_sample_file = sample_cadence.sample_definition_file_name
    var caps_active = caps_sampling_active(caps_sample_file)
    var sample_cfg = SamplePromptConfig()
    if sample_enabled:
        makedirs(samples_dir, exist_ok=True)
        if caps_active:
            sample_cfg = _chroma_sample_prompt_config_for_sampler(caps_sample_file)
            _chroma_preflight_sample_caps(sample_cfg)
            print("[cadence] sample-during-training WIRED (caps) -> ", samples_dir,
                  " prompts=", len(sample_cfg.prompts), " file=", caps_sample_file)
        else:
            print("[cadence] sample-during-training WIRED (legacy cached-caption) -> ", samples_dir,
                  " (steps=", SAMPLE_STEPS, " cfg=", SAMPLE_CFG, ")")
    if sample_enabled and should_sample_completed_step(sample_cadence, 0):
        print("[cadence] step 0 sample due (fires after the first completed step in this bounded loop)")
    var next_sample = next_sample_completed_step(sample_cadence, 0, train_cfg.max_steps)
    print("[cadence] next sample completed_step=", next_sample)

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)

    # ── resident fused-AdamW state (block adapters; MJ-1070/MJ-1085 crash-class
    # fix). Reuses flux's block-adapter resident pair (chroma is flux-family).
    # Persistent device P/M/V + ONE-TIME pinned staging; lazily initialized on
    # first optimizer use (after any resume load) inside the plain-LoRA arm. ────
    var adamw_dev_state = Optional[LoraAdamWPlainDeviceState](None)
    var adamw_state_ready = False

    # ── T1.B EMA (default-off; SimpleTuner EMAModel — training/lora_ema.mojo).
    # F32 shadows over the plain-LoRA adapters, tracked AFTER build/resume
    # (shadow init = clone of current a/b). ema_enabled False => no shadows, and
    # the per-step update + *_ema save below are no-ops (baseline byte-identical).
    # Plain-LoRA arm only: LyCORIS/DoRA/OFT trains carriers, not lora.ad. ───────
    var ema = LoraEmaState(
        train_cfg.ema_decay, train_cfg.ema_min_decay,
        train_cfg.ema_update_after_step, train_cfg.ema_update_step_interval,
    )
    if train_cfg.ema_enabled and not lycoris_active:
        var ema_base = lora_ema_track(ema, lora.ad, 0, n_adapters)
        if ema_base != 0:
            raise Error("train_chroma_real: ema shadow base must be 0")
        print("[ema] tracking", n_adapters, "adapters decay=", train_cfg.ema_decay,
              " min_decay=", train_cfg.ema_min_decay,
              " update_after_step=", train_cfg.ema_update_after_step,
              " interval=", train_cfg.ema_update_step_interval)

    # ── gradient accumulation buffers (SerenityTrainer micro-batch; default-off == 1) ─
    # Each loop iteration is one MICRO-step. SUM the two AdamW-fed LoRA grad
    # groups (d_a/d_b) across `accum_steps` micro-steps, then MEAN (÷N) and run
    # clip+AdamW once on accumulation boundaries. accum_steps=1 => every step is a
    # boundary, buffer holds one grad, mean=÷1 => byte-identical to the per-step
    # path. Buffers are lazily sized on the first micro-step of each window.
    # Wired for PLAIN LoRA only this wave; LyCORIS arms fail loud (mirrors Klein).
    var accum_steps = train_cfg.grad_accum_steps
    if accum_steps < 1:
        accum_steps = 1
    var use_grad_accum = accum_steps > 1
    if use_grad_accum and (lokr_active or loha_active or dora_active or oft_active):
        raise Error(
            "Chroma trainer: grad_accum_steps>1 is wired for plain LoRA only this "
            + "wave; LoKr/LoHa/DoRA/OFT fail loud (mirrors Klein's honest scope). "
            + "Use adapter_algo=0 (plain LoRA) with gradient accumulation."
        )
    # window buffers + micro counter live in the shared trainer_core struct (wraps
    # the grad_accum.mojo SUM/MEAN primitives). accum_steps==1 => every step is a
    # boundary => byte-identical to the per-step path.
    var accum_window = GradAccumWindow(accum_steps)
    if use_grad_accum:
        print("  grad accumulation: accum_steps=", accum_steps, " (mean over micro-steps)")

    # ── TRUE batch-2 (row-stacked device stack): 2 samples/step -> mean gradient.
    #    Fenced to PLAIN LoRA + device stack + accum=1 (mirrors the fleet pattern).
    var use_b2 = train_cfg.batch_size == 2
    if train_cfg.batch_size < 1 or train_cfg.batch_size > 2:
        raise Error(
            "Chroma trainer: only batch_size 1 or 2 supported (TRUE batch-2 max); got "
            + String(train_cfg.batch_size)
        )
    if use_b2:
        if lokr_active or loha_active or dora_active or oft_active:
            raise Error(
                "Chroma trainer: batch_size=2 (TRUE batch-2) is wired for PLAIN LoRA "
                "only — not the LyCORIS/DoRA/OFT arms. Use adapter_algo=0."
            )
        if use_grad_accum:
            raise Error(
                "Chroma trainer: batch_size=2 + grad_accum_steps>1 not wired — "
                "batch_size=2 IS a 2-sample batch; set grad_accum_steps=1."
            )
        if train_cfg.ema_enabled:
            raise Error("Chroma trainer: batch_size=2 + EMA not wired; disable ema.")
        if _env_is_set(String("CHROMA_HOST_STACK")):
            raise Error(
                "Chroma trainer: batch_size=2 requires the DEVICE stack — "
                "unset CHROMA_HOST_STACK."
            )
        print("  TRUE batch-2 (row-stacked device stack): 2 samples/step,",
              "0.5-scaled per-sample d_out -> mean gradient")

    var train_start = perf_counter_ns()
    for k in range(start_step + 1, run_steps + 1):
        var t0 = perf_counter_ns()

        var slot = 0 if FIXED_SIGMA_SMOKE else (k - 1) % len(files)
        var step_seed = UInt64(1) if FIXED_SIGMA_SMOKE else UInt64(k)
        var st = SafeTensors.open(files[slot])
        var latent_cache = _load_chroma_cache_tensor(st, String("latent"), ctx)
        var lat_raw = _host_f32_for_step_math(latent_cache, ctx)         # [16*64*64]

        # t5_embed [1, seq, 4096] -> pad/truncate to [N_TXT, 4096] (zero pad rows).
        var t5_info = st.tensor_info(String("t5_embed"))
        var t5_seq = Int(t5_info.shape[1])
        var t5_cache = _load_chroma_cache_tensor(st, String("t5_embed"), ctx)
        var t5_flat = _host_f32_for_step_math(t5_cache, ctx)
        var txt_tokens = List[Float32]()
        for r in range(N_TXT):
            if r < t5_seq:
                for c in range(TXT_CH):
                    txt_tokens.append(t5_flat[r * TXT_CH + c])
            else:
                for _ in range(TXT_CH):
                    txt_tokens.append(Float32(0.0))

        # ── VAE shift/scale then pack_latents ──
        for i in range(len(lat_raw)):
            lat_raw[i] = (lat_raw[i] - VAE_SHIFT) * VAE_SCALE
        var latent_packed = _pack_latents(lat_raw)

        # ── timestep ──
        var sigma_idx: Int
        if FIXED_SIGMA_SMOKE:
            sigma_idx = FIXED_SIGMA_IDX
        else:
            var sigma = sample_timestep_logit_normal(SEED_BASE + step_seed, TIMESTEP_SHIFT)
            sigma_idx = Int(sigma * Float32(NUM_TRAIN_TIMESTEPS))
            if sigma_idx > NUM_TRAIN_TIMESTEPS - 1:
                sigma_idx = NUM_TRAIN_TIMESTEPS - 1
        var sig = Float32(sigma_idx + 1) / Float32(NUM_TRAIN_TIMESTEPS)
        var t_model = Float32(sigma_idx) / Float32(NUM_TRAIN_TIMESTEPS)

        # ── flow-match in PACKED latent space ──
        var noise = _host_noise(N_IMG * IN_CH, SEED_BASE * UInt64(7919) + step_seed)
        var noisy = List[Float32]()
        var target = List[Float32]()
        for i in range(len(latent_packed)):
            noisy.append(noise[i] * sig + latent_packed[i] * (Float32(1.0) - sig))
            target.append(noise[i] - latent_packed[i])

        # ── frozen approximator -> modulation table ──
        var pooled_tensor = _pooled_modulation_tensor(approx, t_model, ctx)
        var pooled = _host_f32_for_step_math(pooled_tensor, ctx)

        if dora_active:
            var fwd_dora = chroma_stack_direct_dora_forward_offload[H, Dh, N_IMG, N_TXT, S](
                noisy.copy(), txt_tokens.copy(), pooled.copy(), MOD_INDEX,
                base, loader, dora_masters, NUM_DOUBLE, NUM_SINGLE, direct_targets,
                cos.copy(), sin.copy(),
                D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
            )

            var nout_dora = len(fwd_dora.out)
            var d_loss_dora = List[Float32]()
            var sse_dora = 0.0
            var inv_n_dora = Float32(2.0) / Float32(nout_dora)
            for i in range(nout_dora):
                var diff = fwd_dora.out[i] - target[i]
                sse_dora += Float64(diff) * Float64(diff)
                d_loss_dora.append(inv_n_dora * diff)
            var loss_dora = Float32(sse_dora / Float64(nout_dora))
            if k == start_step + 1:
                first_loss = loss_dora
            last_loss = loss_dora

            var grads_dora = chroma_stack_direct_dora_backward_offload[H, Dh, N_IMG, N_TXT, S](
                d_loss_dora, noisy.copy(), txt_tokens.copy(),
                base, loader, dora_masters, NUM_DOUBLE, NUM_SINGLE, direct_targets,
                cos.copy(), sin.copy(), fwd_dora,
                D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
            )
            var dnorm = flux_direct_dora_grad_norm(grads_dora.grads)
            if dnorm > Float64(train_cfg.max_grad_norm):
                flux_direct_dora_clip_grads(grads_dora.grads, train_cfg.max_grad_norm / Float32(dnorm))
            var step_lr = serenity_lr_for_optimizer_step(train_cfg, k)
            flux_direct_dora_adamw_step(
                dora_masters, grads_dora.grads, k, step_lr,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )

            var t1_dora = perf_counter_ns()
            var secs_dora = Float64(t1_dora - t0) / 1.0e9
            print_trainer_progress(
                String("Chroma-dora"), k, run_steps, 1,
                loss_dora, dnorm, secs_dora, 0.0,
                Float64(t1_dora - train_start) / 1.0e9,
            )
            print("[Chroma-dora] step=", k, " grad_norm=", Float32(dnorm),
                  " zero_leg_l1=", flux_direct_dora_zero_leg_l1(dora_masters))
            if grads_dora.nonfinite_lora_grads != 0:
                print("[Chroma-dora] warning nonfinite_lora_grads=", grads_dora.nonfinite_lora_grads)

            if chroma_should_save_checkpoint(train_cfg, k):
                var save_path = _step_lora_path(
                    chroma_output_lora_path_from_train_config(train_cfg, run_steps), k
                )
                var nmods = save_flux_direct_dora(dora_masters, save_path, ctx)
                print("[Chroma-dora] save step=", k, " modules=", nmods, " path=", save_path)
                _chroma_prune_old_checkpoints(train_cfg, run_steps, k)
            continue

        if oft_active:
            var fwd_oft = chroma_stack_direct_oft_forward_offload[H, Dh, N_IMG, N_TXT, S](
                noisy.copy(), txt_tokens.copy(), pooled.copy(), MOD_INDEX,
                base, loader, oft_masters, NUM_DOUBLE, NUM_SINGLE, direct_targets,
                cos.copy(), sin.copy(),
                D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
            )

            var nout_oft = len(fwd_oft.out)
            var d_loss_oft = List[Float32]()
            var sse_oft = 0.0
            var inv_n_oft = Float32(2.0) / Float32(nout_oft)
            for i in range(nout_oft):
                var diff = fwd_oft.out[i] - target[i]
                sse_oft += Float64(diff) * Float64(diff)
                d_loss_oft.append(inv_n_oft * diff)
            var loss_oft = Float32(sse_oft / Float64(nout_oft))
            if k == start_step + 1:
                first_loss = loss_oft
            last_loss = loss_oft

            var grads_oft = chroma_stack_direct_oft_backward_offload[H, Dh, N_IMG, N_TXT, S](
                d_loss_oft, noisy.copy(), txt_tokens.copy(),
                base, loader, oft_masters, NUM_DOUBLE, NUM_SINGLE, direct_targets,
                cos.copy(), sin.copy(), fwd_oft,
                D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
            )
            var onorm = flux_direct_oft_grad_norm(grads_oft.grads)
            if onorm > Float64(train_cfg.max_grad_norm):
                flux_direct_oft_clip_grads(grads_oft.grads, train_cfg.max_grad_norm / Float32(onorm))
            var step_lr = serenity_lr_for_optimizer_step(train_cfg, k)
            flux_direct_oft_adamw_step(
                oft_masters, grads_oft.grads, k, step_lr,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )

            var t1_oft = perf_counter_ns()
            var secs_oft = Float64(t1_oft - t0) / 1.0e9
            print_trainer_progress(
                String("Chroma-oft"), k, run_steps, 1,
                loss_oft, onorm, secs_oft, 0.0,
                Float64(t1_oft - train_start) / 1.0e9,
            )
            print("[Chroma-oft] step=", k, " grad_norm=", Float32(onorm),
                  " vec_l1=", flux_direct_oft_vec_l1(oft_masters))
            if grads_oft.nonfinite_lora_grads != 0:
                print("[Chroma-oft] warning nonfinite_lora_grads=", grads_oft.nonfinite_lora_grads)

            if chroma_should_save_checkpoint(train_cfg, k):
                var save_path = _step_lora_path(
                    chroma_output_lora_path_from_train_config(train_cfg, run_steps), k
                )
                var nmods = save_flux_direct_oft(oft_masters, save_path, ctx)
                print("[Chroma-oft] save step=", k, " modules=", nmods, " path=", save_path)
                _chroma_prune_old_checkpoints(train_cfg, run_steps, k)
            continue

        # ── forward + MSE loss + backward (full depth) ──
        # DEFAULT = device-resident stack (activations stay on-GPU,
        # recompute-in-backward): ~35x faster (139s -> 4.0s/step) and BIT-IDENTICAL
        # to the host stack end-to-end (same loss/grad_norm/LoRA-B on the real
        # model; gated by chroma_stack_device_parity + chroma_block_device_parity).
        # CHROMA_HOST_STACK=1 selects the proven host stack (the parity oracle).
        var loss: Float32
        var grads: FluxLoraGradSet
        if use_b2:
            # ── TRUE batch-2: sample-0 is prepped above; prep sample-1 (its own
            #    cache slot + noise/sigma stream), then run the row-stacked b2
            #    device stack. Each per-sample d_out is 0.5-scaled so the b2
            #    backward's in-GEMM sum = mean(g0, g1) = the 2-sample batch grad;
            #    loss = 0.5*(L0 + L1). ──
            var slot1 = 0 if FIXED_SIGMA_SMOKE else (k % len(files))
            var step_seed1 = UInt64(2) if FIXED_SIGMA_SMOKE else (UInt64(k) + UInt64(7000003))
            var st1 = SafeTensors.open(files[slot1])
            var latent_cache1 = _load_chroma_cache_tensor(st1, String("latent"), ctx)
            var lat_raw1 = _host_f32_for_step_math(latent_cache1, ctx)
            var t5_info1 = st1.tensor_info(String("t5_embed"))
            var t5_seq1 = Int(t5_info1.shape[1])
            var t5_cache1 = _load_chroma_cache_tensor(st1, String("t5_embed"), ctx)
            var t5_flat1 = _host_f32_for_step_math(t5_cache1, ctx)
            var txt_tokens1 = List[Float32]()
            for r in range(N_TXT):
                if r < t5_seq1:
                    for c in range(TXT_CH):
                        txt_tokens1.append(t5_flat1[r * TXT_CH + c])
                else:
                    for _ in range(TXT_CH):
                        txt_tokens1.append(Float32(0.0))
            for i in range(len(lat_raw1)):
                lat_raw1[i] = (lat_raw1[i] - VAE_SHIFT) * VAE_SCALE
            var latent_packed1 = _pack_latents(lat_raw1)
            var sigma_idx1: Int
            if FIXED_SIGMA_SMOKE:
                sigma_idx1 = FIXED_SIGMA_IDX
            else:
                var sigma1 = sample_timestep_logit_normal(SEED_BASE + step_seed1, TIMESTEP_SHIFT)
                sigma_idx1 = Int(sigma1 * Float32(NUM_TRAIN_TIMESTEPS))
                if sigma_idx1 > NUM_TRAIN_TIMESTEPS - 1:
                    sigma_idx1 = NUM_TRAIN_TIMESTEPS - 1
            var sig1 = Float32(sigma_idx1 + 1) / Float32(NUM_TRAIN_TIMESTEPS)
            var t_model1 = Float32(sigma_idx1) / Float32(NUM_TRAIN_TIMESTEPS)
            var noise1 = _host_noise(N_IMG * IN_CH, SEED_BASE * UInt64(7919) + step_seed1)
            var noisy1 = List[Float32]()
            var target1 = List[Float32]()
            for i in range(len(latent_packed1)):
                noisy1.append(noise1[i] * sig1 + latent_packed1[i] * (Float32(1.0) - sig1))
                target1.append(noise1[i] - latent_packed1[i])
            var pooled_tensor1 = _pooled_modulation_tensor(approx, t_model1, ctx)
            var pooled1 = _host_f32_for_step_math(pooled_tensor1, ctx)

            var fwd = chroma_stack_lora_forward_device_offload_b2[H, Dh, N_IMG, N_TXT, S](
                noisy.copy(), txt_tokens.copy(), pooled.copy(),
                noisy1.copy(), txt_tokens1.copy(), pooled1.copy(), MOD_INDEX,
                base, loader, lora, cos.copy(), sin.copy(),
                D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
            )
            var nout = len(fwd.out0)
            var inv_n = Float32(2.0) / Float32(nout)
            var sse0 = 0.0
            var sse1 = 0.0
            var d_out0 = List[Float32]()
            var d_out1 = List[Float32]()
            for i in range(nout):
                var diff0 = fwd.out0[i] - target[i]
                var diff1 = fwd.out1[i] - target1[i]
                sse0 += Float64(diff0) * Float64(diff0)
                sse1 += Float64(diff1) * Float64(diff1)
                d_out0.append(Float32(0.5) * inv_n * diff0)
                d_out1.append(Float32(0.5) * inv_n * diff1)
            loss = Float32(0.5) * (Float32(sse0 / Float64(nout)) + Float32(sse1 / Float64(nout)))
            grads = chroma_stack_lora_backward_device_offload_b2[H, Dh, N_IMG, N_TXT, S](
                d_out0, d_out1, base, loader, lora, cos.copy(), sin.copy(), fwd,
                D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
            )
        elif not _env_is_set(String("CHROMA_HOST_STACK")):
            var fwd = chroma_stack_lora_forward_device_offload[H, Dh, N_IMG, N_TXT, S](
                noisy.copy(), txt_tokens.copy(), pooled.copy(), MOD_INDEX,
                base, loader, lora, cos.copy(), sin.copy(),
                D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
            )
            var nout = len(fwd.out)
            var d_loss = List[Float32]()
            var sse = 0.0
            var inv_n = Float32(2.0) / Float32(nout)
            for i in range(nout):
                var diff = fwd.out[i] - target[i]
                sse += Float64(diff) * Float64(diff)
                d_loss.append(inv_n * diff)
            loss = Float32(sse / Float64(nout))
            grads = chroma_stack_lora_backward_device_offload[H, Dh, N_IMG, N_TXT, S](
                d_loss, noisy.copy(), txt_tokens.copy(), base, loader, lora,
                cos.copy(), sin.copy(), fwd,
                D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
            )
        else:
            var fwd = chroma_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
                noisy.copy(), txt_tokens.copy(), pooled.copy(), MOD_INDEX,
                base, loader, lora, cos.copy(), sin.copy(),
                D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
            )
            var nout = len(fwd.out)
            var d_loss = List[Float32]()
            var sse = 0.0
            var inv_n = Float32(2.0) / Float32(nout)
            for i in range(nout):
                var diff = fwd.out[i] - target[i]
                sse += Float64(diff) * Float64(diff)
                d_loss.append(inv_n * diff)
            loss = Float32(sse / Float64(nout))
            grads = chroma_stack_lora_backward_offload[H, Dh, N_IMG, N_TXT, S](
                d_loss, noisy.copy(), txt_tokens.copy(), base, loader, lora,
                cos.copy(), sin.copy(), fwd,
                D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
            )
        if k == start_step + 1:
            first_loss = loss
        last_loss = loss

        # ── gradient accumulation (SerenityTrainer semantics; default-off when N==1) ─
        # Fast path: accum_steps==1 uses `grads` directly (no zero-clone/copy).
        if use_grad_accum:
            # Treat this loop iteration as one micro-step: SUM its LoRA grad groups
            # into the shared trainer_core window; on a non-boundary micro-step print
            # progress and skip clip/AdamW/save/sample; on the boundary MEAN (÷N)
            # back into `grads` before clip+AdamW. The window math lives in
            # trainer_core.GradAccumWindow (byte-op-identical to the prior inline
            # block); the boundary control flow (progress print + continue) stays
            # here. k==run_steps force-flushes the tail partial window.
            var is_boundary = accum_window.accumulate(grads.d_a, grads.d_b, k == run_steps)
            if not is_boundary:
                # mid-window: skip clip/AdamW/save/sample, keep accumulating.
                var t1m = perf_counter_ns()
                var secsm = Float64(t1m - t0) / 1.0e9
                print_trainer_progress(
                    String("Chroma-lora"), k, run_steps, 1,
                    loss, 0.0, secsm, 0.0,
                    Float64(t1m - train_start) / 1.0e9,
                )
                continue
            # boundary: MEAN the window back into grads' groups. The norm is
            # recomputed by _clip below from the meaned grads, so finalize_mean's
            # returned norm is discarded (behavior-identical to the inline block).
            _ = accum_window.finalize_mean(grads.d_a, grads.d_b)

        # ── grad norm + clip(1.0) ──
        var gn_before = _clip(grads, CLIP_GRAD_NORM)

        # ── AdamW ──
        # grad-accum: LR schedule + AdamW step counter advance per OPTIMIZER step,
        # not per micro-step (N==1 => optimizer_step==k, byte-identical).
        var optimizer_step = ((k - 1) // accum_steps) + 1
        var step_lr = serenity_lr_for_optimizer_step(train_cfg, optimizer_step)
        if lokr_active:
            var mg = flux_lokr_chain_all(lokr_masters, grads.d_a, grads.d_b)
            var mnorm = flux_lokr_grad_norm(mg)
            if mnorm > Float64(CLIP_GRAD_NORM):
                flux_lokr_clip_grads(mg, CLIP_GRAD_NORM / Float32(mnorm))
            flux_lokr_adamw_step(
                lokr_masters, mg, optimizer_step, step_lr,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )
            lora = flux_lokr_carrier_set(lokr_masters, D, FMLP)
            print("[Chroma-lokr] step=", k, " master_grad_norm=", Float32(mnorm),
                  " zero_leg_l1=", flux_lokr_zero_leg_l1(lokr_masters))
        elif loha_active:
            var mg = flux_loha_chain_all(loha_masters, grads.d_a, grads.d_b)
            var mnorm = flux_loha_grad_norm(mg)
            if mnorm > Float64(CLIP_GRAD_NORM):
                flux_loha_clip_grads(mg, CLIP_GRAD_NORM / Float32(mnorm))
            flux_loha_adamw_step(
                loha_masters, mg, optimizer_step, step_lr,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )
            lora = flux_loha_carrier_set(loha_masters, D, FMLP)
            print("[Chroma-loha] step=", k, " master_grad_norm=", Float32(mnorm),
                  " zero_leg_l1=", flux_loha_zero_leg_l1(loha_masters))
        else:
            # MJ-1070/MJ-1085: resident fused AdamW — persistent device P/M/V
            # with ONE-TIME pinned staging (init lazily after resume load); the
            # per-step fused arm segfaulted under the fp8-resident base
            # (unmapped-staging mechanism) and stays retired.
            if not adamw_state_ready:
                adamw_dev_state = Optional[LoraAdamWPlainDeviceState](
                    flux_lora_adamw_state_init(lora, ctx)
                )
                adamw_state_ready = True
                print("[chroma-adamw] resident fused state initialized (",
                      total_adapters(lora), "adapters )")
            flux_lora_adamw_step_resident(
                adamw_dev_state.value(), lora, grads, optimizer_step, step_lr, ctx,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )
            # T1.B: EMA shadow update post-AdamW (plain arm; once per OPTIMIZER
            # step — this branch runs only at grad-accum boundaries). Off => skip.
            if train_cfg.ema_enabled:
                ema_update(ema, lora.ad, optimizer_step)

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        print_trainer_progress(
            String("Chroma-lora"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start) / 1.0e9,
        )
        if grads.nonfinite_lora_grads != 0:
            print("[Chroma-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)

        var saved_this_step = False
        if chroma_should_save_checkpoint(train_cfg, k):
            var save_path = _step_lora_path(
                chroma_output_lora_path_from_train_config(train_cfg, run_steps), k
            )
            if lokr_active:
                _ = save_flux_lokr(lokr_masters, save_path, ctx)
            elif loha_active:
                _ = save_flux_loha(loha_masters, save_path, ctx)
            else:
                if adamw_state_ready:
                    lora_adamw_plain_device_state_sync_params(
                        adamw_dev_state.value(), lora.ad, ctx
                    )
                    lora_adamw_plain_device_state_sync_moments(
                        adamw_dev_state.value(), lora.ad, ctx
                    )
                _ = save_chroma_lora(lora, save_path, ctx)
                if train_cfg.ema_enabled:  # T1.B EMA sibling next to every save
                    _save_chroma_lora_ema(ema, lora, save_path, n_adapters, ctx)
                var state_path = save_path + String(".state.safetensors")
                _ = save_chroma_lora_state(lora, state_path, ctx)
                print("[Chroma-lora] save_state step=", k, " path=", state_path)
            saved_this_step = True
            _chroma_prune_old_checkpoints(train_cfg, run_steps, k)
        if sample_enabled and should_sample_completed_step(sample_cadence, k):
            if chroma_should_save_before_sample(sample_cadence, k, saved_this_step):
                var sample_path = _step_lora_path(
                    chroma_output_lora_path_from_train_config(train_cfg, run_steps), k
                )
                if lokr_active:
                    _ = save_flux_lokr(lokr_masters, sample_path, ctx)
                elif loha_active:
                    _ = save_flux_loha(loha_masters, sample_path, ctx)
                else:
                    if adamw_state_ready:
                        lora_adamw_plain_device_state_sync_params(
                            adamw_dev_state.value(), lora.ad, ctx
                        )
                        lora_adamw_plain_device_state_sync_moments(
                            adamw_dev_state.value(), lora.ad, ctx
                        )
                    _ = save_chroma_lora(lora, sample_path, ctx)
                    if train_cfg.ema_enabled:  # T1.B EMA sibling
                        _save_chroma_lora_ema(ema, lora, sample_path, n_adapters, ctx)
                    var sample_state = sample_path + String(".state.safetensors")
                    _ = save_chroma_lora_state(lora, sample_state, ctx)
                    print("[Chroma-lora] save_before_sample step=", k, " path=", sample_state)
            # Sample from the CURRENT frozen base + streamed blocks + LIVE LoRA.
            print(
                "[cadence] sample due at completed_step=", k,
                " sample_file=", sample_cadence.sample_definition_file_name,
            )
            if caps_active:
                # Prompt-faithful: one render per ENABLED prompt, conditioned on
                # THAT prompt's caps (not the step's cached caption).
                for pi in range(len(sample_cfg.prompts)):
                    var prompt = sample_cfg.prompts[pi].copy()
                    if not prompt.enabled:
                        continue
                    var cond = _chroma_txt_cond_from_caps(prompt.caps_pos, prompt.label, ctx)
                    var uncond = List[Float32]()
                    if prompt.cfg != Float32(1.0) and prompt.caps_neg != String(""):
                        uncond = _chroma_txt_cond_from_caps(prompt.caps_neg, prompt.label, ctx)
                    _chroma_run_sample_caps(
                        base, approx, loader, lora, cond^, uncond^,
                        cos.copy(), sin.copy(), samples_dir, sample_vae_path, k, prompt,
                        prompt.seed + UInt64(pi), ctx,
                    )
            else:
                # Legacy: reuse this step's cached caption T5 embeds (txt_tokens)
                # as COND, zeros as UNCOND — NOT a prompt-faithful validation.
                warn_legacy_cached_caption_sampling(String("Chroma"))
                _chroma_run_sample(
                    base, approx, loader, lora, txt_tokens.copy(),
                    cos.copy(), sin.copy(), samples_dir, sample_vae_path, k, ctx,
                )

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    for i in range(n_adapters):
        b_absum_final += _absum(lora.ad[i].b)
    var trains = (
        ((b_absum_init == 0.0) and (b_absum_final > 0.0))
        if start_step == 0 else b_absum_final != b_absum_init
    )
    var carrier_zero_final = Float64(0.0)
    if lokr_active:
        carrier_zero_final = flux_lokr_zero_leg_l1(lokr_masters)
        trains = carrier_zero_final > carrier_zero_init
    elif loha_active:
        carrier_zero_final = flux_loha_zero_leg_l1(loha_masters)
        trains = carrier_zero_final > carrier_zero_init
    elif dora_active:
        carrier_zero_final = flux_direct_dora_zero_leg_l1(dora_masters)
        trains = carrier_zero_final > carrier_zero_init
    elif oft_active:
        carrier_zero_final = flux_direct_oft_vec_l1(oft_masters)
        trains = carrier_zero_final > carrier_zero_init
    if trains and (last_loss == last_loss):
        if lycoris_active:
            print("RESULT: REAL run OK — LyCORIS trainable grew ",
                  carrier_zero_init, " -> ", carrier_zero_final,
                  "; loss", first_loss, "->", last_loss,
                  (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        else:
            print("RESULT: REAL run OK — LoRA-B changed", b_absum_init, "->", b_absum_final,
                  "; loss", first_loss, "->", last_loss,
                  (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        var lora_out = chroma_output_lora_path_from_train_config(train_cfg, run_steps)
        if lokr_active:
            _ = save_flux_lokr(lokr_masters, lora_out, ctx)
        elif loha_active:
            _ = save_flux_loha(loha_masters, lora_out, ctx)
        elif dora_active:
            var nmods = save_flux_direct_dora(dora_masters, lora_out, ctx)
            print("[Chroma-dora] save final modules=", nmods, " path=", lora_out)
        elif oft_active:
            var nmods = save_flux_direct_oft(oft_masters, lora_out, ctx)
            print("[Chroma-oft] save final modules=", nmods, " path=", lora_out)
        else:
            if adamw_state_ready:
                lora_adamw_plain_device_state_sync_params(
                    adamw_dev_state.value(), lora.ad, ctx
                )
                lora_adamw_plain_device_state_sync_moments(
                    adamw_dev_state.value(), lora.ad, ctx
                )
            _ = save_chroma_lora(lora, lora_out, ctx)
            if train_cfg.ema_enabled:  # T1.B EMA sibling
                _save_chroma_lora_ema(ema, lora, lora_out, n_adapters, ctx)
            var state_out = lora_out + String(".state.safetensors")
            _ = save_chroma_lora_state(lora, state_out, ctx)
            print("[Chroma-lora] save_state step=", run_steps, " path=", state_out)
    else:
        print("RESULT: FAIL trains=", trains)
