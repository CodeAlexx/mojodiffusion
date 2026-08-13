# train_qwenimage_real.mojo — Qwen-Image LoRA training loop (block-swap offload).
#
# TRANSLATION of EriDiffusion-v2 train_qwenimage.rs onto the parity-verified Mojo
# Qwen-Image LoRA OFFLOAD stack. 60 all-double-stream blocks (D=3072, H=24, Dh=128,
# F=12288, in_ch=64, txt_ch=3584, out_ch=64). Mirrors train_chroma_real.mojo's loop
# structure (timing, grad clip, progress display) and qwenimage.rs's recipe.
#
# QWENIMAGE vs CHROMA (key deltas):
#   - ALL double-stream (60 blocks, 0 single). No single-stream loop.
#   - SEPARATE per-block mod-MLPs (img_mod.1 / txt_mod.1). Each frozen mod MLP
#     projects silu(temb) -> [6D] per block. Mod-MLP weights are STREAMED from the
#     block (same Block handle as the attention weights). Frozen: grads discarded.
#   - time_text_embed: sinusoidal(t*1000, 256) -> silu(Linear1) -> Linear2 -> [D].
#     This MLP is applied ONCE per step to get silu_temb_h [1,D].
#   - norm_out.linear: [2D,D] produces final_scale/final_shift from silu_temb_h.
#   - txt_ch = 3584 (Qwen2.5-VL text encoder hidden dim; NOT T5-XXL 4096).
#   - Flow-match recipe (qwenimage.rs:1093-1099):
#       x_t = (1 - sigma)*latent + sigma*noise   (note: opposite sign to Flux)
#       target = noise - latent
#   - Timestep: SerenityTrainer DISCRETE (BaseQwenSetup + ModelSetupNoiseMixin
#       _get_timestep_discrete + ModelSetupFlowMatchingMixin _add_noise_discrete):
#       idx = int(sigmoid(N(0,1)) * 1000 * shift_remap);  shift=1.0 -> identity.
#       sigma = (idx+1)/1000  (blend);  model_t = idx/1000  (transformer input).
#   - out_ch = 64 (latent channels; proj_out [64,D]; target [N_IMG, 64]).
#   - ROPE: 3-axis interleaved, axes=(16,56,56), theta=10000.
#
# Recipe (configs/qwenimage.json, matching SerenityTrainer "#qwen LoRA 24GB" preset):
#   lr=3e-4, rank=16, alpha=1.0 (scale=1/16), timestep_shift=1.0,
#   lr_warmup_steps=200 (constant scheduler), clip_grad_norm=1.0.
#
# MEMORY: 60 * ~648 MB BF16/FP8 blocks + resident base (~tiny) + LoRA + optimizer.
# Block-swap streams one block at a time. A fixed-sigma smoke mode confirms
# loss decreases monotonically with a frozen sample.
#
# FIXED_SIGMA_SMOKE: every step uses the SAME latent+text AND a fixed sigma+noise.
# A correct LoRA backward MUST drive loss DOWN monotonically.
#
# COMPILE-ONLY DELIVERABLE: do NOT execute this binary (full 60-block model).
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/training/train_qwenimage_real.mojo \
#     -o /tmp/train_qwenimage_real
#
# ── UNVERIFIABLE-WITHOUT-CACHE ITEMS (flagged for future parity gate) ──────────
# (1) Checkpoint dtype FP8-E4M3: Qwen loaders dequant FP8 bytes to BF16 on use;
#     parity vs torch FP8 checkpoint still needs a local reference gate.
# (2) txt_ch=3584: the Qwen2.5-VL text encoder; cache dir uses placeholder zeros.
# (3) RoPE total_half = 8+28+28 = 64 = Dh//2: matches config axes (16,56,56);
#     parity to qwenimage.rs RoPE verified per-block cos>=0.999 in the block tests.
# (4) norm_out.linear [2D,D] → chunk 0 scale chunk 1 shift: diffusers layout from
#     config.json; not re-gated here (same as weights.mojo build_qwen_per_block_mods).
# (5) txt_norm.weight [txt_ch]: applied by the caller pre-normalization in inference;
#     in the trainer we skip it (match train_qwenimage.rs which operates on already-
#     normalized text embeddings from the cache).

from std.sys import argv
from std.collections import List, Optional
from max.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from std.time import perf_counter_ns
from std.os import listdir, makedirs

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.activations import silu

from serenitymojo.models.qwenimage.qwenimage_stack import (
    QwenStackBase, _t as _qstack_t,
)
from serenitymojo.models.qwenimage.weights import load_qwen_stack_base, load_qwen_host_bf16
# DEVICE-RESIDENT stack (MJ-1084): activations stay on GPU across the 60
# blocks (block gate 28/28 max_abs=0.0). False = the host oracle conductors.
comptime QWEN_DEVICE_STACK = True

from serenitymojo.training.lora_adamw_plain_fused import LoraAdamWPlainDeviceState
from serenitymojo.models.qwenimage.qwenimage_stack_lora import (
    qwen_offload_lora_adamw_step_resident, qwen_lora_adamw_state_init,
    QwenLoraSet, QwenLoraGradSet, QwenOffloadBase, QwenOffloadForward,
    build_qwen_lora_set, save_qwen_lora, save_qwen_lora_state,
    qwenimage_stack_lora_forward_offload,
    qwenimage_stack_lora_backward_offload,
    qwenimage_stack_lora_forward_offload_device,
    qwenimage_stack_lora_backward_offload_device,
    qwenimage_stack_lora_forward_offload_device_b2,
    qwenimage_stack_lora_backward_offload_device_b2,
    build_qwen_direct_dora_set_from_offload,
    qwenimage_stack_direct_dora_forward_offload,
    qwenimage_stack_direct_dora_backward_offload,
    qwenimage_stack_direct_oft_forward_offload,
    qwenimage_stack_direct_oft_backward_offload,
    qwen_offload_lora_adamw_step,
    DBL_SLOTS,
)
from serenitymojo.models.qwenimage.qwenimage_lycoris_stack import (
    QwenLoKrSet, empty_qwen_lokr_set, build_qwen_lokr_set,
    qwen_lokr_carrier_set, qwen_lokr_carrier_total_bytes,
    qwen_lokr_chain_all, qwen_lokr_adamw_step, qwen_lokr_grad_norm,
    qwen_lokr_clip_grads, qwen_lokr_zero_leg_l1, save_qwen_lokr,
    QwenLoHaSet, empty_qwen_loha_set, build_qwen_loha_set,
    qwen_loha_carrier_set, qwen_loha_carrier_total_bytes,
    qwen_loha_chain_all, qwen_loha_adamw_step, qwen_loha_grad_norm,
    qwen_loha_clip_grads, qwen_loha_zero_leg_l1, save_qwen_loha,
)
from serenitymojo.models.qwenimage.qwenimage_direct_lycoris_stack import (
    QWEN_DIRECT_24_GIB,
    empty_qwen_direct_dora_set, empty_qwen_direct_oft_set,
    build_qwen_direct_oft_set,
    qwen_direct_dense_carrier_bytes,
    qwen_direct_dora_preflight,
    qwen_direct_oft_preflight,
    qwen_direct_dora_grad_norm, qwen_direct_dora_clip_grads,
    qwen_direct_dora_adamw_step, qwen_direct_dora_zero_leg_l1,
    qwen_direct_dora_trainable_bytes, save_qwen_direct_dora,
    qwen_direct_oft_grad_norm, qwen_direct_oft_clip_grads,
    qwen_direct_oft_adamw_step, qwen_direct_oft_vec_l1,
    qwen_direct_oft_trainable_bytes, save_qwen_direct_oft,
)
from serenitymojo.models.dit.qwenimage_dit import (
    QwenImageConfig, build_qwenimage_rope_tables,
)
from serenitymojo.training.qwenimage_sample_resident import (
    qwenimage_sample_resident, qwenimage_decode_packed_to_png,
)
from serenitymojo.offload.qwenimage_plan import build_qwenimage_offload_plan
from serenitymojo.offload.plan import OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.training.schedule import (
    sample_timestep_discrete_qwen, DiscreteTimestep,
)
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.training.lora_ema import (
    LoraEmaState, lora_ema_track, ema_update,
    lora_ema_adapters, ema_path_for_lora,
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
from serenitymojo.io.train_config_reader import read_model_config
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
    serenity_lr_for_optimizer_step,
    serenity_sample_cadence_from_train_config,
    serenity_sampling_enabled,
    serenity_should_save_before_sample,
    serenity_state_path_for_lora,
    serenity_step_lora_path,
    validate_serenity_gradient_checkpointing_policy,
    validate_serenity_train_math_policy,
)


# ── arch (qwen-image; confirmed from config.json + qwenimage_dit.mojo) ────────
comptime H = 24
comptime Dh = 128
comptime D = H * Dh            # 3072
comptime FMLP = 12288          # mlp_hidden = D * 4
comptime IN_CH = 64            # in_channels (patchified latent)
comptime TXT_CH = 3584         # Qwen2.5-VL text encoder hidden
comptime OUT_CH = 64           # proj_out output channels
comptime NUM_DOUBLE = 60       # all-double-stream
comptime TIMESTEP_DIM = 256    # sinusoidal embedding dim
comptime EPS = Float32(1.0e-6)

# ── resolution (512px / patch=2): latent [64,32,32] -> [N_IMG=1024, 64] ───────
comptime LAT_C = 64            # in_channels (VAE latent channels = 16 before patch)
comptime LAT_H = 32            # latent height at patch-2 (512px / 16)
comptime LAT_W = 32            # latent width
comptime N_IMG = LAT_H * LAT_W  # 1024 image tokens
comptime N_TXT = 256           # text token sequence length (padded)
comptime S = N_TXT + N_IMG     # 1280 joint sequence

# ── RoPE frame/height/width for 512px 1-frame ─────────────────────────────────
comptime ROPE_FRAME = 1
comptime ROPE_H = 32           # == LAT_H (latent height in patch coords)
comptime ROPE_W = 32           # == LAT_W

# ── recipe defaults (configs/qwenimage.json is the runtime source of truth) ───
comptime SEED_BASE = UInt64(42)

comptime FIXED_SIGMA_SMOKE = True
comptime FIXED_SIGMA_VAL = Float32(0.5)    # fixed sigma for smoke test

comptime DEFAULT_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/qwenimage.json"
comptime DEFAULT_RUN_STEPS = 5
comptime DEFAULT_CACHE_DIR = "/home/alex/datasets/qwenimage_cache_512"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/qwenimage_lora"
# fp8-resident cap (MJ-1065): the 60 double-blocks (~39 GiB bf16, the biggest
# per-forward disk stream in the fleet) quantized to E4M3 + per-row scale are
# ~20 GiB — held resident, dequant per block, NO per-step disk stream. 22 GiB
# cap holds every block (require pinned==count). ~20 GiB resident is TIGHT on
# 24 GiB with LoRA/optimizer state — the 1-step gate MEASURES the VRAM peak.
comptime QWEN_FP8_RESIDENT_BUDGET_BYTES = 22 * 1024 * 1024 * 1024

# ── sample-during-training (v1; qwenimage_sample_resident) ────────────────────
# SAMPLE_STEPS / SAMPLE_CFG : denoise loop length + true-CFG scale (sampler
#   defaults 30 / 4.0, matching qwenimage_sample_cli.mojo STEPS/CFG).
# SAMPLE_SEED : base RNG seed for the t=1 packed init noise (per-step deterministic).
# DEFAULT_VAE_DIR : fallback Qwen VAE dir if train_cfg.vae is empty (the config's
#   "vae" field is the source of truth; see configs/qwenimage.json).
# Conditioning v1 = the step's cached caption embeds (COND) + zeros (UNCOND); the
#   trainer has no in-tree Qwen2.5-VL encoder to encode an arbitrary sample prompt.
comptime SAMPLE_STEPS = 30
comptime SAMPLE_CFG = Float32(4.0)
comptime SAMPLE_SEED = UInt64(0xC4_303A_5A91)
comptime DEFAULT_VAE_DIR = "/home/alex/.serenity/models/checkpoints/qwen-image-2512/vae"

# ── VAE latent geometry for the 512px bucket (patch=2) ────────────────────────
# unpatchify needs (channels, height, width, patch). The trainer's packed latent
# is [N_IMG=1024, IN_CH=64]; IN_CH = LAT_C(16) * patch(2) * patch(2). The VAE input
# is [1, 16, LAT_H*2, LAT_W*2] = [1,16,64,64] -> tiled-decode -> [1,3,512,512].
comptime SAMPLE_LAT_C = 16              # VAE latent channels (out_channels)
comptime SAMPLE_PATCH = 2               # patch_size (config.json)


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


def qwen_patchified_out_channels(cfg: TrainConfig) -> Int:
    var qcfg = QwenImageConfig.qwen_image()
    return cfg.out_channels * qcfg.patch_size * qcfg.patch_size


def validate_qwen_train_config(cfg: TrainConfig) raises:
    if cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOCON:
        print("[QwenImage-locon] network_algorithm=locon: using the linear LoRA-compatible down/up path")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR:
        print("[QwenImage-lokr] network_algorithm=lokr: using carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA:
        print("[QwenImage-loha] network_algorithm=loha: using carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA or cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT:
        print(
            String("[QwenImage-direct] network_algorithm=")
            + adapter_algo_name(cfg.adapter_algo)
            + String(": using direct W_eff stack dispatch; sample cadence must be disabled")
        )
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_BOFT:
        raise Error("Qwen-Image trainer: BOFT is intentionally excluded; use lora, locon, loha, lokr, dora, or oft where wired")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_FULL:
        raise Error("Qwen-Image trainer: full finetune is not wired; supported here: lora, locon, loha, lokr, dora, oft")
    elif cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA:
        raise Error(
            String("Qwen-Image trainer: network_algorithm=")
            + adapter_algo_name(cfg.adapter_algo)
            + String(" is not wired; supported here: lora, locon, loha, lokr, dora, oft")
        )
    # The hot stack functions are still comptime-specialized for the 512px
    # Qwen-Image bucket. Fail here instead of silently using mismatched metadata.
    if cfg.checkpoint == String(""):
        raise Error("Qwen trainer config must set checkpoint")
    if cfg.n_heads != H:
        raise Error(String("Qwen config n_heads ") + String(cfg.n_heads) + String(" != comptime H ") + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("Qwen config head_dim ") + String(cfg.head_dim) + String(" != comptime Dh ") + String(Dh))
    if cfg.d_model != H * Dh:
        raise Error(String("Qwen config d_model ") + String(cfg.d_model) + String(" != H*Dh ") + String(H * Dh))
    if cfg.in_channels != IN_CH:
        raise Error(String("Qwen config in_channels ") + String(cfg.in_channels) + String(" != IN_CH ") + String(IN_CH))
    if cfg.joint_attention_dim != TXT_CH:
        raise Error(String("Qwen config joint_attention_dim ") + String(cfg.joint_attention_dim) + String(" != TXT_CH ") + String(TXT_CH))
    if cfg.num_double != NUM_DOUBLE or cfg.num_single != 0:
        raise Error(
            String("Qwen trainer requires 60 double-stream blocks and 0 single blocks; got double=")
            + String(cfg.num_double) + String(" single=") + String(cfg.num_single)
        )
    if cfg.mlp_hidden != FMLP:
        raise Error(String("Qwen config mlp_hidden ") + String(cfg.mlp_hidden) + String(" != FMLP ") + String(FMLP))
    if cfg.timestep_dim != TIMESTEP_DIM:
        raise Error(String("Qwen config timestep_dim ") + String(cfg.timestep_dim) + String(" != TIMESTEP_DIM ") + String(TIMESTEP_DIM))
    if qwen_patchified_out_channels(cfg) != OUT_CH:
        raise Error(
            String("Qwen config out_channels ") + String(cfg.out_channels)
            + String(" with patch_size=2 gives ")
            + String(qwen_patchified_out_channels(cfg))
            + String(" patchified channels, expected ") + String(OUT_CH)
        )
    if cfg.lora_rank <= 0:
        raise Error("Qwen trainer config requires lora_rank > 0")
    if cfg.lora_alpha <= Float32(0.0):
        raise Error("Qwen trainer config requires lora_alpha > 0")
    if cfg.lr <= Float32(0.0):
        raise Error("Qwen trainer config requires learning_rate > 0")
    if cfg.max_grad_norm <= Float32(0.0):
        raise Error("Qwen trainer config requires max_grad_norm > 0")
    validate_serenity_train_math_policy(cfg, String("Qwen trainer"))
    validate_serenity_gradient_checkpointing_policy(
        cfg, String("Qwen trainer"), SERENITY_GRAD_POLICY_ON_ONLY
    )


def qwen_offload_config_from_train_config(cfg: TrainConfig) raises -> OffloadConfig:
    validate_qwen_train_config(cfg)
    if cfg.activation_offload_enabled() or cfg.layer_offload_enabled():
        raise Error(
            String("Qwen trainer cannot honor CPU activation/layer offload yet; ")
            + String("set gradient_checkpointing=ON for the current synchronous block loader")
        )
    return OffloadConfig.synchronous_single()


def qwen_sample_cadence_from_train_config(
    cfg_path: String, cfg: TrainConfig,
) raises -> SampleCadence:
    return serenity_sample_cadence_from_train_config(cfg_path, cfg)


def qwen_sampling_enabled(cadence: SampleCadence) -> Bool:
    return serenity_sampling_enabled(cadence)


def qwen_should_save_before_sample(
    cadence: SampleCadence, completed_step: Int, saved_this_step: Bool,
) raises -> Bool:
    return serenity_should_save_before_sample(cadence, completed_step, saved_this_step)


def qwen_state_path_for_lora(lora_path: String) -> String:
    return serenity_state_path_for_lora(lora_path)


def _step_lora_path(base_path: String, step: Int) -> String:
    return serenity_step_lora_path(base_path, step)


# Rolling checkpoint retention (audit item #4), pruned AFTER a periodic save —
# krea2's discipline, thin wrapper over the shared trainer_core machinery. Reuses
# the shared keep-count decision (trainer_prune_target_step), but builds the
# pruned path with qwen's OWN step-path helper (reference-policy naming off
# `output_lora_path`, not krea2's workspace/stem), then removes it + its
# `.state.safetensors` sidecar (the DoRA/OFT arms write no sidecar → no-op).
# keep_default/milestone=0 ⇒ NO prune until the webui sets save_max_keep, so
# keep-all stays byte-unchanged when it is unset.
def _qwen_prune_old_checkpoints(cfg: TrainConfig, output_lora_path: String, saved_step: Int) raises:
    var old = trainer_prune_target_step(cfg, saved_step, 0, 0)
    if old > 0:
        trainer_prune_step_checkpoint(
            _step_lora_path(output_lora_path, old), String(".state.safetensors")
        )


# T1.B: save the EMA shadow set as the *_ema.safetensors sibling of a plain-LoRA
# checkpoint (SimpleTuner copy_to analog — lora_ema.mojo lora_ema_adapters
# returns bf16-rounded shadows over the LIVE set's shapes). Only reached when
# cfg.ema_enabled; flag-off leaves this uncalled (baseline bytes unchanged).
def _save_qwen_lora_ema(
    ema: LoraEmaState, lora: QwenLoraSet, lora_path: String, ctx: DeviceContext
) raises:
    var ema_set = lora.copy()
    var shadow_ads = lora_ema_adapters(ema, lora.dbl, 0, len(lora.dbl), 0)
    for i in range(len(shadow_ads)):
        ema_set.dbl[i] = shadow_ads[i].copy()
    var ema_path = ema_path_for_lora(lora_path)
    _ = save_qwen_lora(ema_set, ema_path, ctx)
    print("[QwenImage-lora] save_ema path=", ema_path)


def _substr(s: String, start: Int, end: Int) -> String:
    var out = String("")
    var i = 0
    for ch in s.codepoint_slices():
        if i >= start and i < end:
            out += String(ch)
        i += 1
    return out^


def _dirname(path: String) -> String:
    var last = -1
    var i = 0
    for ch in path.codepoint_slices():
        if String(ch) == String("/"):
            last = i
        i += 1
    if last <= 0:
        return String(".")
    return _substr(path, 0, last)


def _mkdir_parent(path: String) raises:
    var parent_dir = _dirname(path)
    if parent_dir != String("."):
        makedirs(parent_dir, exist_ok=True)


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


def _global_norm(grads: QwenLoraGradSet) -> Float64:
    var ss = 0.0
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: QwenLoraGradSet, max_norm: Float32) -> Float64:
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
        raise Error(String("qwenimage cache: no .safetensors in ") + dir)
    for i in range(1, len(fs)):
        var j = i
        while j > 0 and fs[j - 1] > fs[j]:
            var tmp = fs[j - 1]
            fs[j - 1] = fs[j]
            fs[j] = tmp
            j -= 1
    return fs^


def _load_cache_preserving_dtype(
    st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def _load_host_f32_sharded(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var tv = st.tensor_view(name)
    var t = Tensor.from_view(tv, ctx)
    return t.to_host(ctx)


def _load_host_bf16_sharded(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> List[BFloat16]:
    return load_qwen_host_bf16(st, name, ctx)


# Sinusoidal timestep embedding (host, returns [timestep_dim] F32).
# t_val: sigma in [0,1]; Qwen scales by 1000 before embedding.
def _sinusoidal_temb(t_val: Float32, ctx: DeviceContext) raises -> List[Float32]:
    var t_h = List[Float32]()
    t_h.append(t_val * Float32(1000.0))
    var t_tensor = Tensor.from_host(t_h, [1], STDtype.F32, ctx)
    var t_emb = timestep_embedding(
        t_tensor, Int(TIMESTEP_DIM), ctx, Float32(10000.0), STDtype.F32
    )
    return t_emb.to_host(ctx)   # [1, TIMESTEP_DIM] flat = [TIMESTEP_DIM] scalars


# Build silu_temb_h: silu(time_text_embed(t)) = silu(MLP(sinusoidal(t))).
# te_lin1_w [D, timestep_dim], te_lin1_b [D], te_lin2_w [D,D], te_lin2_b [D].
def _build_silu_temb(
    t_val: Float32,
    te_lin1_w: List[BFloat16], te_lin1_b: List[BFloat16],
    te_lin2_w: List[BFloat16], te_lin2_b: List[BFloat16],
    ctx: DeviceContext,
) raises -> List[Float32]:
    from serenitymojo.ops.linear import linear
    var sin_emb_h = _sinusoidal_temb(t_val, ctx)        # [TIMESTEP_DIM]
    var t_emb = Tensor.from_host(sin_emb_h, [1, Int(TIMESTEP_DIM)], STDtype.BF16, ctx)
    var b1 = Tensor.from_host_bf16(te_lin1_b.copy(), [Int(D)], ctx)
    var h1 = linear(
        t_emb,
        Tensor.from_host_bf16(te_lin1_w.copy(), [Int(D), Int(TIMESTEP_DIM)], ctx),
        Optional[Tensor](b1^), ctx,
    )
    var h1_silu = silu(h1, ctx)
    var b2 = Tensor.from_host_bf16(te_lin2_b.copy(), [Int(D)], ctx)
    var temb_out = linear(
        h1_silu,
        Tensor.from_host_bf16(te_lin2_w.copy(), [Int(D), Int(D)], ctx),
        Optional[Tensor](b2^), ctx,
    )
    # final silu for use as per-block mod MLP input
    return silu(temb_out, ctx).to_host(ctx)   # [1, D] flat


# pack_latents: [LAT_C, LAT_H, LAT_W] flat F32 -> [N_IMG, LAT_C] (trivial
# patch=1 since latent is already at patch resolution for Qwen-Image 512px).
# Qwen-Image patchify: patch_size=2 applied at VAE decode time (in_channels=64
# = 16ch * 2*2); the latent cache already stores the patchified [N_IMG, 64].
# So no patchify needed — the cache tensor IS [N_IMG, 64] already.
# (Verified: in_channels=64, out_channels=64 in config.json; the latent cache
#  for training stores the pack_latents output, not the raw VAE latent.)


# ── _qwen_run_sample — one sample-during-training image ───────────────────────
#   cond text    : the current step's cached caption embeds (v1; see header of
#                  qwenimage_sample_resident.mojo).
#   uncond text  : a zeroed [N_TXT*TXT_CH] vector (true-CFG empty cond).
#   init noise   : packed gaussian [N_IMG*IN_CH], seed = SAMPLE_SEED + step (the
#                  same Box-Muller PCG the train loop draws noise with).
#   denoise      : qwenimage_sample_resident (frozen base + streamed blocks + live
#                  LoRA; true-CFG flow-match Euler).
#   decode+write : qwenimage_decode_packed_to_png -> <samples_dir>/step_<N>.png.
# Fail-loud: any raise propagates (no silent skip), matching the trainer's
# fail-loud cadence contract.
def _qwen_run_sample(
    base: QwenOffloadBase,
    mut loader: TurboPlannedLoader,
    lora: QwenLoraSet,
    cond_txt: List[Float32],     # [N_TXT*TXT_CH] — the step's cached caption embeds
    cos_h: List[Float32],
    sin_h: List[Float32],
    norm_out_w: List[BFloat16], norm_out_b: List[BFloat16],
    vae_dir: String,
    samples_dir: String,
    step: Int,
    ctx: DeviceContext,
) raises:
    # UNCOND: zeroed text features (same dtype/shape as cond_txt).
    var uncond_txt = List[Float32]()
    for _ in range(Int(N_TXT) * Int(TXT_CH)):
        uncond_txt.append(Float32(0.0))

    var init_noise = _host_noise(Int(N_IMG) * Int(IN_CH), SAMPLE_SEED + UInt64(step))

    var latent = qwenimage_sample_resident[H, Dh, N_IMG, N_TXT, S](
        base, loader, lora,
        cond_txt.copy(), uncond_txt^, init_noise^,
        cos_h.copy(), sin_h.copy(),
        norm_out_w, norm_out_b,
        SAMPLE_STEPS, SAMPLE_CFG,
        Int(D), Int(FMLP), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), Int(TIMESTEP_DIM),
        EPS, ctx,
    )

    var out_path = samples_dir + String("/step_") + String(step) + String(".png")
    qwenimage_decode_packed_to_png[
        N_IMG, ROPE_H, ROPE_W, SAMPLE_LAT_C, SAMPLE_PATCH, IN_CH
    ](
        latent, vae_dir, out_path, ctx,
    )
    print("[QwenImage-lora] sample step=", step, " -> ", out_path)


# ── per-prompt validation caps (serenity.sample_prompts.v1) ──────────────────
# caps_pos/caps_neg point at a CACHE-ENTRY-SHAPED safetensors carrying the
# "text_embedding" key (same key + layout the train loop reads from its cache).
# Reuses the loop's cache->cond pad/truncate to [N_TXT, TXT_CH] so the sample
# conditioning is the real prompt (NOT the step's cached caption; MJ-1045 trap).
def _qwen_txt_cond_from_caps(path: String, label: String, ctx: DeviceContext) raises -> List[Float32]:
    if path == String(""):
        raise Error(String("QwenImage sample prompt ") + label + String(": empty caps path"))
    var st = SafeTensors.open(path)
    var info = st.tensor_info(String("text_embedding"))
    if len(info.shape) < 2 or Int(info.shape[len(info.shape) - 1]) != Int(TXT_CH):
        raise Error(
            String("QwenImage caps for prompt '") + label + String("' at ") + path
            + String(": expected cache-shaped safetensors key 'text_embedding' shape [1,LT,")
            + String(Int(TXT_CH)) + String("] (LT<=512, truncated/zero-padded to ")
            + String(Int(N_TXT)) + String("); got last-dim ")
            + String(Int(info.shape[len(info.shape) - 1]))
        )
    var txt_cache = _load_cache_preserving_dtype(st, String("text_embedding"), ctx)
    var txt_flat = txt_cache.to_host_bf16(ctx)
    var txt_seq = len(txt_flat) // Int(TXT_CH)
    var txt = List[Float32]()
    for r in range(Int(N_TXT)):
        if r < txt_seq:
            for c in range(Int(TXT_CH)):
                txt.append(txt_flat[r * Int(TXT_CH) + c].cast[DType.float32]())
        else:
            for _ in range(Int(TXT_CH)):
                txt.append(Float32(0.0))
    return txt^


def _qwen_check_caps_shape(path: String, label: String) raises:
    if path == String(""):
        raise Error(String("QwenImage sample prompt ") + label + String(": empty caps path"))
    var st = SafeTensors.open(path)
    var info = st.tensor_info(String("text_embedding"))
    if len(info.shape) < 2 or Int(info.shape[len(info.shape) - 1]) != Int(TXT_CH):
        raise Error(
            String("QwenImage caps for prompt '") + label + String("' at ") + path
            + String(": expected cache-shaped safetensors key 'text_embedding' shape [1,LT,")
            + String(Int(TXT_CH)) + String("] (LT<=512, truncated/zero-padded to ")
            + String(Int(N_TXT)) + String("); got last-dim ")
            + String(Int(info.shape[len(info.shape) - 1]))
        )


def _qwen_sample_prompt_config_for_sampler(sample_file: String) raises -> SamplePromptConfig:
    if sample_file == String(""):
        raise Error("QwenImage trainer caps sampling requires validation_prompts_file")
    var cfg = read_sample_prompt_config(sample_file)
    assert_enabled_sample_prompts(cfg, String("QwenImage"))
    return cfg^


def _qwen_preflight_sample_caps(sample_cfg: SamplePromptConfig) raises:
    var checked = 0
    for i in range(len(sample_cfg.prompts)):
        var p = sample_cfg.prompts[i].copy()
        if not p.enabled:
            continue
        if p.frames != 1:
            raise Error(String("QwenImage sample prompt ") + p.label + String(": only single-frame image samples supported"))
        if p.width != Int(ROPE_W) * Int(SAMPLE_PATCH) * 8 or p.height != Int(ROPE_H) * Int(SAMPLE_PATCH) * 8:
            raise Error(
                String("QwenImage sample prompt ") + p.label + String(": requests ")
                + String(p.width) + String("x") + String(p.height)
                + String(" but this binary samples ") + String(Int(ROPE_W) * Int(SAMPLE_PATCH) * 8)
                + String("x") + String(Int(ROPE_H) * Int(SAMPLE_PATCH) * 8)
            )
        _qwen_check_caps_shape(p.caps_pos, p.label)
        if p.cfg != Float32(1.0) and p.caps_neg != String(""):
            _qwen_check_caps_shape(p.caps_neg, p.label)
        checked += 1
    if checked == 0:
        raise Error("QwenImage trainer requires at least one enabled validation prompt when caps sampling is enabled")


def _qwen_run_sample_caps(
    base: QwenOffloadBase,
    mut loader: TurboPlannedLoader,
    lora: QwenLoraSet,
    cond_txt: List[Float32],
    uncond_txt: List[Float32],     # empty => zeroed CFG-empty cond
    cos_h: List[Float32],
    sin_h: List[Float32],
    norm_out_w: List[BFloat16], norm_out_b: List[BFloat16],
    vae_dir: String,
    samples_dir: String,
    step: Int,
    prompt: SamplePrompt,
    seed: UInt64,
    ctx: DeviceContext,
) raises:
    var uncond = uncond_txt.copy()
    if len(uncond) == 0:
        for _ in range(Int(N_TXT) * Int(TXT_CH)):
            uncond.append(Float32(0.0))
    var init_noise = _host_noise(Int(N_IMG) * Int(IN_CH), seed)
    var latent = qwenimage_sample_resident[H, Dh, N_IMG, N_TXT, S](
        base, loader, lora,
        cond_txt.copy(), uncond^, init_noise^,
        cos_h.copy(), sin_h.copy(),
        norm_out_w, norm_out_b,
        prompt.steps, prompt.cfg,
        Int(D), Int(FMLP), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), Int(TIMESTEP_DIM),
        EPS, ctx,
    )
    var out_path = (
        samples_dir + String("/step_") + String(step)
        + String("_") + prompt.label + String(".png")
    )
    qwenimage_decode_packed_to_png[
        N_IMG, ROPE_H, ROPE_W, SAMPLE_LAT_C, SAMPLE_PATCH, IN_CH
    ](
        latent, vae_dir, out_path, ctx,
    )
    print("[QwenImage-lora] caps sample step=", step, " prompt=", prompt.label, " -> ", out_path)


def main() raises:
    var a = argv()
    var cfg_path = String(DEFAULT_CONFIG)
    var run_steps = DEFAULT_RUN_STEPS
    if len(a) >= 2:
        var arg1 = String(a[1])
        if _is_nonnegative_int(arg1):
            run_steps = _parse_nonnegative_int(arg1)
        else:
            cfg_path = arg1^
    if len(a) >= 3:
        run_steps = _parse_nonnegative_int(String(a[2]))

    var train_cfg = read_model_config(cfg_path)
    validate_qwen_train_config(train_cfg)
    var cache_preflight = create_serenity_trainer_cache_preflight_plan(train_cfg)
    validate_serenity_trainer_cache_preflight_plan(cache_preflight)
    var sample_cadence = qwen_sample_cadence_from_train_config(cfg_path, train_cfg)
    var sample_enabled = qwen_sampling_enabled(sample_cadence)
    var direct_algo_requested = (
        train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA
        or train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT
    )
    if sample_enabled and direct_algo_requested:
        raise Error(
            "QwenImage direct DoRA/OFT sample-during-training is not wired; disable sample cadence for this runtime gate"
        )
    var offload_cfg = qwen_offload_config_from_train_config(train_cfg)
    if run_steps <= 0:
        run_steps = train_cfg.max_steps
    if run_steps > train_cfg.max_steps:
        run_steps = train_cfg.max_steps
    var cache_dir = String(DEFAULT_CACHE_DIR)
    if train_cfg.dataset_cache_dir != String(""):
        cache_dir = train_cfg.dataset_cache_dir.copy()
    var output_lora_path = String(LORA_DIR) + String("/qwenimage_lora_smoke.safetensors")
    if train_cfg.output_model_destination != String(""):
        output_lora_path = train_cfg.output_model_destination.copy()
    _mkdir_parent(output_lora_path)

    print("=== Qwen-Image REAL LoRA training loop (block-swap offload) ===")
    print("  config:", cfg_path)
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " F=", FMLP, " in_ch=", IN_CH,
          " txt_ch=", TXT_CH, " out_ch=", OUT_CH)
    print("  depth: NUM_DOUBLE=", NUM_DOUBLE, " (all-double)")
    print("  tokens: N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S)
    print("  recipe: rank=", train_cfg.lora_rank, " alpha=", train_cfg.lora_alpha,
          " lr=", train_cfg.lr, " shift=", train_cfg.timestep_shift,
          " max_grad_norm=", train_cfg.max_grad_norm)
    print("  run_steps=", run_steps, " config_max_steps=", train_cfg.max_steps)
    print(
        "  cadence: save_every=", train_cfg.save_every,
        " sample_after=", sample_cadence.sample_after,
        " unit=", sample_time_unit_name(sample_cadence.sample_after_unit),
        " skip_first=", sample_cadence.sample_skip_first,
        " sample_file=", sample_cadence.sample_definition_file_name,
    )
    if train_cfg.enable_async_offloading:
        print("[offload] async offload requested by config; Qwen trainer currently uses synchronous TurboPlannedLoader")
    print("  LoRA targets: 12/block (img/txt x q,k,v,out,ff_up,ff_down) x 60 = 720")
    print("  fixed_sigma_smoke=", FIXED_SIGMA_SMOKE)
    print("  ckpt:", train_cfg.checkpoint)
    print("  cache:", cache_dir)

    if sample_enabled and should_sample_completed_step(sample_cadence, 0):
        print("[cadence] step 0 sample due; fires after the first completed step (sampler is wired)")
    var next_sample = next_sample_completed_step(sample_cadence, 0, train_cfg.max_steps)
    print("[cadence] next sample completed_step=", next_sample)
    if train_cfg.only_cache:
        print("[QwenImage-lora] only_cache requested; no train steps will run in this trainer")
        return

    var ctx = DeviceContext()

    # ── load frozen stack-level base (img_in/txt_in/proj_out + timestep MLP) ──
    print("[load] QwenStackBase from checkpoint")
    var st = ShardedSafeTensors.open(train_cfg.checkpoint)

    var base_stack = load_qwen_stack_base(
        st,
        train_cfg.d_model,
        train_cfg.in_channels,
        train_cfg.joint_attention_dim,
        qwen_patchified_out_channels(train_cfg),
        ctx,
    )

    # timestep MLP weights (top-level in checkpoint)
    var te_lin1_w = _load_host_bf16_sharded(
        st, "time_text_embed.timestep_embedder.linear_1.weight", ctx
    )   # [D, TIMESTEP_DIM]
    var te_lin1_b = _load_host_bf16_sharded(
        st, "time_text_embed.timestep_embedder.linear_1.bias", ctx
    )   # [D]
    var te_lin2_w = _load_host_bf16_sharded(
        st, "time_text_embed.timestep_embedder.linear_2.weight", ctx
    )   # [D, D]
    var te_lin2_b = _load_host_bf16_sharded(
        st, "time_text_embed.timestep_embedder.linear_2.bias", ctx
    )   # [D]

    # norm_out.linear weights (for final scale/shift)
    var norm_out_w = _load_host_bf16_sharded(st, "norm_out.linear.weight", ctx)  # [2D, D]
    var norm_out_b = _load_host_bf16_sharded(st, "norm_out.linear.bias", ctx)    # [2D]

    var base = QwenOffloadBase(
        base_stack^,
        norm_out_w.copy(), norm_out_b.copy(),
        te_lin1_w.copy(), te_lin1_b.copy(),
        te_lin2_w.copy(), te_lin2_b.copy(),
    )
    print("[load] base resident (img_in/txt_in/proj_out/timestep-MLP/norm_out)")

    # ── block-swap offload loader ────────────────────────────────────────────
    var plan = build_qwenimage_offload_plan()
    # Both fp8 modes make every block permanently resident (device or host-
    # pinned-fp8), so the whole-DiT bf16 pinned block store (~39 GiB!, never
    # read again) must not be allocated. Only the explicit streamed A/B arm
    # needs it. (Two concurrent 17 GiB stores OOM-killed the session 2026-07-04.)
    var loader = TurboPlannedLoader.open(
        train_cfg.checkpoint, plan^, offload_cfg, ctx,
        fill_block_store=(
            train_cfg.quantized_resident == String("streamed_base_opt_in")
        ),
    )
    print("[load] offload loader opened (", loader.block_count(), "blocks)")

    # ── Residency policy (MJ-1065, 2026-07-03) ──────────────────────────────────
    # Base weights MUST be device-resident: per-step disk reads are forbidden.
    # qwenimage streamed ~39 GiB bf16 EVERY forward (the biggest payoff in the
    # fleet) and its bf16 host-pin store risked failing on the 62 GiB box.
    #   "fp8_e4m3" (default): quantize the WHOLE block base ONCE from the bf16
    #     checkpoint to E4M3 + per-row F32 scale (~20 GiB), hold resident, dequant
    #     per block on await — no streaming. (The standalone qwen fp8 file is
    #     UNIT-scale with no per-row scales — a different, lossier scheme — so we
    #     quantize the bf16 checkpoint per-row here, matching krea2/hidream/ideogram.)
    #   "fp8_e4m3_host" (default): same per-row E4M3 quantize-once, but PINNED IN
    #     HOST RAM (~20 GiB) and H2D+dequant per await — device-resident fp8 was
    #     MEASURED to OOM 24 GiB (peak 23.4 GiB, CUDA_ERROR_OUT_OF_MEMORY in the
    #     first forward, 2026-07-04). Half the per-step PCIe of bf16 streaming.
    #   "streamed_base_opt_in": the OLD ~39 GiB per-step bf16 disk stream (A/B arm).
    #   empty/OFF/other: FAIL LOUD (the disk-stream default was the violation).
    if train_cfg.quantized_resident == String("fp8_e4m3"):
        var n_blocks = loader.block_count()
        var pinned = loader.pin_residents_fp8(QWEN_FP8_RESIDENT_BUDGET_BYTES, ctx)
        if pinned != n_blocks:
            raise Error(
                String("qwenimage fp8-resident: pinned ") + String(pinned)
                + " of " + String(n_blocks) + " blocks within budget "
                + String(QWEN_FP8_RESIDENT_BUDGET_BYTES) + " bytes — a block would "
                + "still per-step disk-stream (MJ-1065). Raise the budget or fall "
                + "back to host-pinned fp8."
            )
        print(
            "[quant] fp8_e4m3-resident base: quantized", pinned, "of", n_blocks,
            "blocks ONCE (per-row E4M3; dequant per block; NO per-step disk read).",
        )
    elif train_cfg.quantized_resident == String("fp8_e4m3_host"):
        var n_blocks = loader.block_count()
        var pinned = loader.pin_residents_fp8_host(
            QWEN_FP8_RESIDENT_BUDGET_BYTES, ctx
        )
        if pinned != n_blocks:
            raise Error(
                String("qwenimage fp8-host: pinned ") + String(pinned)
                + " of " + String(n_blocks) + " blocks within budget "
                + String(QWEN_FP8_RESIDENT_BUDGET_BYTES) + " bytes — a block would "
                + "still per-step disk-stream (MJ-1065). Raise the budget."
            )
        print(
            "[quant] fp8_e4m3_host base: quantized", pinned, "of", n_blocks,
            "blocks ONCE into pinned host RAM (H2D+dequant per await;",
            "half the PCIe of bf16 streaming; NO per-step disk read).",
        )
    elif train_cfg.quantized_resident == String("streamed_base_opt_in"):
        print("[quant] streamed_base_opt_in: ~39 GiB per-step bf16 disk stream (A/B arm).")
    else:
        raise Error(
            String("qwenimage: quantized_resident='") + train_cfg.quantized_resident
            + "' selects the per-step DISK-STREAM base, forbidden by policy "
            + "MJ-1065. Use \"fp8_e4m3_host\" (host-pinned fp8, fits 24GB), "
            + "\"fp8_e4m3\" (device-resident, MEASURED OOM at 512px), or "
            + "\"streamed_base_opt_in\" for the explicit streamed A/B arm."
        )

    # ── 3-axis RoPE tables (fixed for 512px / 1 frame) ──────────────────────
    var qcfg = QwenImageConfig.qwen_image()
    var rope = build_qwenimage_rope_tables(
        Int(ROPE_FRAME), Int(ROPE_H), Int(ROPE_W), Int(N_TXT),
        Int(H), qcfg, STDtype.F32, ctx
    )
    var cos_h = rope[0].to_host(ctx)
    var sin_h = rope[1].to_host(ctx)
    print("[load] Qwen-Image 3-axis RoPE tables built (S*H x Dh//2)")

    var lokr_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR
    var loha_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA
    var dora_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA
    var oft_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT
    var direct_active = dora_active or oft_active
    var carrier_active = lokr_active or loha_active
    var lycoris_active = carrier_active or direct_active

    # ── gradient accumulation (item 2h; SerenityTrainer sum-N-then-mean) ───────────
    # Each loop iteration is one MICRO-step. We SUM the plain LoRA host grad
    # groups (grads.d_a/d_b) across grad_accum_steps micro-steps, MEAN (÷N), then
    # run clip+AdamW once on accumulation boundaries. accum_steps=1 => every step
    # is a boundary, buffer holds one grad, mean=÷1 => byte-identical baseline.
    # The LyCORIS algos (lokr/loha carrier + dora/oft direct masters) route the
    # optimizer through separate chain/master paths; fail loud rather than
    # silently mis-accumulating them (mirrors klein's honest scope).
    var accum_steps = train_cfg.grad_accum_steps
    if accum_steps < 1:
        accum_steps = 1
    var use_grad_accum = accum_steps > 1
    if use_grad_accum and lycoris_active:
        raise Error("QwenImage: grad_accum_steps>1 not wired for LyCORIS (lokr/loha/dora/oft) this wave")

    var direct_targets = train_cfg.lokr_targets
    var direct_oft_block_size = 4
    # ── LoRA set (B=0 init -> identity at step 0) ────────────────────────────
    var lora = build_qwen_lora_set(
        0,
        train_cfg.d_model,
        train_cfg.mlp_hidden,
        train_cfg.lora_rank,
        train_cfg.lora_alpha,
    )
    var n_adapters = 0
    if not direct_active:
        lora = build_qwen_lora_set(
            train_cfg.num_double,
            train_cfg.d_model,
            train_cfg.mlp_hidden,
            train_cfg.lora_rank,
            train_cfg.lora_alpha,
        )
        n_adapters = train_cfg.num_double * Int(DBL_SLOTS)
    var lokr_masters = empty_qwen_lokr_set()
    var loha_masters = empty_qwen_loha_set()
    var dora_masters = empty_qwen_direct_dora_set()
    var oft_masters = empty_qwen_direct_oft_set()
    if lokr_active:
        lokr_masters = build_qwen_lokr_set(
            train_cfg.num_double, train_cfg.d_model, train_cfg.mlp_hidden,
            train_cfg.lora_rank, train_cfg.lora_alpha,
            train_cfg.lokr_factor, train_cfg.lokr_factor_attn,
            train_cfg.lokr_factor_ff,
            train_cfg.lokr_decompose_both, train_cfg.lokr_full_matrix,
            direct_targets, UInt64(700001),
        )
        var carrier_bytes = qwen_lokr_carrier_total_bytes(
            lokr_masters, train_cfg.d_model, train_cfg.mlp_hidden
        )
        print("[QwenImage-lokr] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("Qwen-Image LoKr: carrier set needs ")
                + String(carrier_bytes)
                + String(" bytes (> budget). Use a smaller lokr_factor/rank or restrict lokr_targets.")
            )
        lora = qwen_lokr_carrier_set(lokr_masters, train_cfg.d_model, train_cfg.mlp_hidden)
        print("[QwenImage-lokr] carrier set materialized:", len(lora.dbl), "adapters")
    elif loha_active:
        loha_masters = build_qwen_loha_set(
            train_cfg.num_double, train_cfg.d_model, train_cfg.mlp_hidden,
            train_cfg.lora_rank, train_cfg.lora_alpha,
            direct_targets, UInt64(800001),
        )
        var carrier_bytes = qwen_loha_carrier_total_bytes(
            loha_masters, train_cfg.d_model, train_cfg.mlp_hidden
        )
        print("[QwenImage-loha] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("Qwen-Image LoHa: carrier set needs ")
                + String(carrier_bytes)
                + String(" bytes (> budget). Reduce lora_rank or restrict lokr_targets.")
        )
        lora = qwen_loha_carrier_set(loha_masters, train_cfg.d_model, train_cfg.mlp_hidden)
        print("[QwenImage-loha] carrier set materialized:", len(lora.dbl), "adapters")
    elif dora_active:
        var dense_bytes = qwen_direct_dense_carrier_bytes(
            train_cfg.num_double, train_cfg.d_model, train_cfg.mlp_hidden,
            direct_targets,
        )
        var direct_bytes = qwen_direct_dora_preflight(
            train_cfg.num_double, train_cfg.d_model, train_cfg.mlp_hidden,
            train_cfg.lora_rank, direct_targets, QWEN_DIRECT_24_GIB,
            False,
        )
        print("[QwenImage-dora] dense carrier bytes:", dense_bytes,
              " direct trainable bytes:", direct_bytes,
              " budget:", QWEN_DIRECT_24_GIB)
        print("[QwenImage-dora] initializing DoRA magnitudes from streamed Qwen block weights ...")
        dora_masters = build_qwen_direct_dora_set_from_offload(
            loader, train_cfg.num_double, train_cfg.d_model, train_cfg.mlp_hidden,
            Int(Dh), train_cfg.lora_rank, train_cfg.lora_alpha, direct_targets,
            train_cfg.seed * UInt64(61) + UInt64(7200), False, ctx,
        )
        print("[QwenImage-dora] trainable bytes:", qwen_direct_dora_trainable_bytes(dora_masters),
              " slots:", len(dora_masters.ad))
    elif oft_active:
        var dense_bytes = qwen_direct_dense_carrier_bytes(
            train_cfg.num_double, train_cfg.d_model, train_cfg.mlp_hidden,
            direct_targets,
        )
        var direct_bytes = qwen_direct_oft_preflight(
            train_cfg.num_double, train_cfg.d_model, train_cfg.mlp_hidden,
            direct_oft_block_size, direct_targets, QWEN_DIRECT_24_GIB,
        )
        print("[QwenImage-oft] dense carrier bytes:", dense_bytes,
              " direct trainable bytes:", direct_bytes,
              " block_size:", direct_oft_block_size,
              " budget:", QWEN_DIRECT_24_GIB)
        oft_masters = build_qwen_direct_oft_set(
            train_cfg.num_double, train_cfg.d_model, train_cfg.mlp_hidden,
            direct_oft_block_size, direct_targets,
        )
        print("[QwenImage-oft] trainable bytes:", qwen_direct_oft_trainable_bytes(oft_masters),
              " slots:", len(oft_masters.ad))
    if dora_active:
        print("[QwenImage-dora] direct block slots:", len(dora_masters.ad),
              " (", DBL_SLOTS, "x", NUM_DOUBLE, "double)")
    elif oft_active:
        print("[QwenImage-oft] direct block slots:", len(oft_masters.ad),
              " (", DBL_SLOTS, "x", NUM_DOUBLE, "double)")
    else:
        print("[lora] adapters:", n_adapters, " (", DBL_SLOTS, "x", NUM_DOUBLE, "double)")

    var files: List[String]
    var have_cache = True
    try:
        files = _list_cache(cache_dir)
        print("[cache] samples:", len(files))
    except:
        files = List[String]()
        have_cache = False
        print("[cache] WARNING: no cache at", cache_dir, "- using synthetic tokens")

    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.dbl[i].b)
    if carrier_active:
        print("[lora] carrier LoRA-B |.|_1 at init =", b_absum_init)
    elif dora_active:
        print("[QwenImage-dora] direct trainable L1 at init =", qwen_direct_dora_zero_leg_l1(dora_masters))
    elif oft_active:
        print("[QwenImage-oft] direct trainable L1 at init =", qwen_direct_oft_vec_l1(oft_masters))
    else:
        print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")
    var carrier_zero_init = Float64(0.0)
    if lokr_active:
        carrier_zero_init = qwen_lokr_zero_leg_l1(lokr_masters)
        print("[QwenImage-lokr] zero-leg L1 at init =", carrier_zero_init)
    elif loha_active:
        carrier_zero_init = qwen_loha_zero_leg_l1(loha_masters)
        print("[QwenImage-loha] zero-leg L1 at init =", carrier_zero_init)
    elif dora_active:
        carrier_zero_init = qwen_direct_dora_zero_leg_l1(dora_masters)
        print("[QwenImage-dora] zero-leg L1 at init =", carrier_zero_init)
    elif oft_active:
        carrier_zero_init = qwen_direct_oft_vec_l1(oft_masters)
        print("[QwenImage-oft] vec L1 at init =", carrier_zero_init)

    # ── sample-during-training setup ─────────────────────────────────────────
    # Sampling is WIRED (see _qwen_run_sample): denoise the CURRENT base + streamed
    # blocks + LIVE LoRA -> Qwen VAE tiled-decode -> <LORA_DIR>/samples/step_<N>.png.
    # Conditioning v1 = the firing step's cached caption embeds (COND) + zeros (UNCOND).
    var sample_vae_dir = String(DEFAULT_VAE_DIR)
    if train_cfg.vae != String(""):
        sample_vae_dir = train_cfg.vae.copy()
    var samples_dir = String(LORA_DIR) + String("/samples")
    makedirs(samples_dir, exist_ok=True)
    # STANDARD sample-prompts contract: caps sampling is ACTIVE when the config
    # names a validation_prompts_file; load+preflight the per-prompt caps (fail
    # loud before the run). Otherwise the seam uses the legacy cached-caption
    # render with a LOUD warning.
    var caps_sample_file = sample_cadence.sample_definition_file_name
    var caps_active = caps_sampling_active(caps_sample_file)
    var sample_cfg = SamplePromptConfig()
    if caps_active:
        sample_cfg = _qwen_sample_prompt_config_for_sampler(caps_sample_file)
        _qwen_preflight_sample_caps(sample_cfg)
        print("[cadence] sample-during-training WIRED (caps) -> ", samples_dir,
              " prompts=", len(sample_cfg.prompts), " file=", caps_sample_file, " vae=", sample_vae_dir)
    else:
        print("[cadence] sample-during-training WIRED (legacy cached-caption) -> ", samples_dir,
              " (steps=", SAMPLE_STEPS, " cfg=", SAMPLE_CFG, " vae=", sample_vae_dir, ")")

    var adamw_dev_state = Optional[LoraAdamWPlainDeviceState](None)
    var adamw_state_ready = False

    # ── T1.B EMA (default-off; SimpleTuner EMAModel — training/lora_ema.mojo).
    # F32 shadows over the plain-LoRA adapters (lora.dbl), tracked AFTER
    # build/resume. ema_enabled False => no shadows; the per-step update + *_ema
    # save below are no-ops (baseline byte-identical). Plain-LoRA arm only:
    # LyCORIS/DoRA/OFT trains carriers, not lora.dbl. ──────────────────────────
    var ema = LoraEmaState(
        train_cfg.ema_decay, train_cfg.ema_min_decay,
        train_cfg.ema_update_after_step, train_cfg.ema_update_step_interval,
    )
    if train_cfg.ema_enabled and not lycoris_active:
        var ema_base = lora_ema_track(ema, lora.dbl, 0, len(lora.dbl))
        if ema_base != 0:
            raise Error("train_qwenimage_real: ema shadow base must be 0")
        print("[ema] tracking", len(lora.dbl), "adapters decay=", train_cfg.ema_decay,
              " min_decay=", train_cfg.ema_min_decay,
              " update_after_step=", train_cfg.ema_update_after_step,
              " interval=", train_cfg.ema_update_step_interval)
    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)

    # window buffers + micro counter live in the shared trainer_core struct (wraps
    # the grad_accum.mojo SUM/MEAN primitives). accum_steps==1 => every step is a
    # boundary => byte-identical to the per-step path.
    var accum_window = GradAccumWindow(accum_steps)
    if use_grad_accum:
        print("  grad accumulation: accum_steps=", accum_steps, " (mean over micro-steps)")

    # ── TRUE batch-2 (row-stacked device stack): 2 samples/step -> mean gradient.
    #    Fenced to PLAIN LoRA + device stack + accum=1 (mirrors train_chroma_real's
    #    use_b2). qwen's device stack is a comptime constant (QWEN_DEVICE_STACK),
    #    so there is no host-stack env to unset — b2 always has the device path.
    var use_b2 = train_cfg.batch_size == 2
    if train_cfg.batch_size < 1 or train_cfg.batch_size > 2:
        raise Error(
            "QwenImage trainer: only batch_size 1 or 2 supported (TRUE batch-2 max); got "
            + String(train_cfg.batch_size)
        )
    if use_b2:
        if lokr_active or loha_active or dora_active or oft_active:
            raise Error(
                "QwenImage trainer: batch_size=2 (TRUE batch-2) is wired for PLAIN LoRA "
                "only — not the LyCORIS/DoRA/OFT arms. Use adapter_algo=0."
            )
        if use_grad_accum:
            raise Error(
                "QwenImage trainer: batch_size=2 + grad_accum_steps>1 not wired — "
                "batch_size=2 IS a 2-sample batch; set grad_accum_steps=1."
            )
        if train_cfg.ema_enabled:
            raise Error("QwenImage trainer: batch_size=2 + EMA not wired; disable ema.")
        comptime if not QWEN_DEVICE_STACK:
            raise Error(
                "QwenImage trainer: batch_size=2 requires the DEVICE stack "
                "(QWEN_DEVICE_STACK=True)."
            )
        print("  TRUE batch-2 (row-stacked device stack): 2 samples/step,",
              "0.5-scaled per-sample d_out -> mean gradient")

    var train_start = perf_counter_ns()
    for k in range(1, run_steps + 1):
        var t0 = perf_counter_ns()

        # ── timestep (SerenityTrainer DISCRETE: idx -> sigma & model_t) ──
        # SerenityTrainer discretizes: idx = int(sigmoid(N)*1000*shift_remap); then
        #   sigma   = (idx+1)/1000  (noise/latent blend, _add_noise_discrete)
        #   model_t = idx/1000      (transformer timestep input; *1000 internally)
        # The blend coefficient (sigma) and the embedding input (model_t) DIFFER
        # by one quantum — this is the divergence being fixed.
        var sigma: Float32
        var model_t: Float32
        var step_seed = UInt64(1) if FIXED_SIGMA_SMOKE else UInt64(k)
        if FIXED_SIGMA_SMOKE:
            # discretize the fixed smoke sigma the same way (0.5 -> idx=499):
            var smoke_idx = Int(FIXED_SIGMA_VAL * Float32(1000.0))
            if smoke_idx >= 1000:
                smoke_idx = 999
            sigma = Float32(Float64(smoke_idx + 1) / 1000.0)
            model_t = Float32(Float64(smoke_idx) / 1000.0)
        else:
            var dts = sample_timestep_discrete_qwen(
                SEED_BASE + step_seed, train_cfg.timestep_shift, 1000
            )
            sigma = dts.sigma
            model_t = dts.model_t

        # ── load / synthesize tokens ──
        var img_tokens = List[Float32]()   # [N_IMG, IN_CH]
        var txt_tokens = List[Float32]()   # [N_TXT, TXT_CH]

        if have_cache and len(files) > 0:
            var slot = 0 if FIXED_SIGMA_SMOKE else (k - 1) % len(files)
            var cst = SafeTensors.open(files[slot])
            var latent_cache = _load_cache_preserving_dtype(cst, String("latent"), ctx)
            var latent_h = latent_cache.to_host_bf16(ctx)
            for i in range(len(latent_h)):
                img_tokens.append(latent_h[i].cast[DType.float32]())
            # txt embed cache key = "text_embedding" (the reference trainer/producer key; matches
            # ernie/anima/sd35 producers). Latent key "latent" already matches.
            var txt_cache = _load_cache_preserving_dtype(cst, String("text_embedding"), ctx)
            var txt_flat = txt_cache.to_host_bf16(ctx)
            var txt_seq = len(txt_flat) // Int(TXT_CH)
            for r in range(Int(N_TXT)):
                if r < txt_seq:
                    for c in range(Int(TXT_CH)):
                        txt_tokens.append(txt_flat[r * Int(TXT_CH) + c].cast[DType.float32]())
                else:
                    for _ in range(Int(TXT_CH)):
                        txt_tokens.append(Float32(0.0))
        else:
            # synthetic: zeros (smoke compile check only)
            for _ in range(Int(N_IMG) * Int(IN_CH)):
                img_tokens.append(Float32(0.0))
            for _ in range(Int(N_TXT) * Int(TXT_CH)):
                txt_tokens.append(Float32(0.0))

        # ── flow-match: noisy = (1-sigma)*latent + sigma*noise ; target = noise - latent ──
        var noise = _host_noise(Int(N_IMG) * Int(IN_CH), SEED_BASE * UInt64(7919) + step_seed)
        var noisy = List[Float32]()
        var target = List[Float32]()
        var one_minus_sigma = Float32(1.0) - sigma
        for i in range(len(img_tokens)):
            noisy.append(one_minus_sigma * img_tokens[i] + sigma * noise[i])
            target.append(noise[i] - img_tokens[i])

        # ── silu_temb_h: frozen time_text_embed output [1, D] ──
        # Use model_t (= idx/1000), the SerenityTrainer transformer timestep input,
        # NOT sigma (= (idx+1)/1000, the blend coefficient).
        var silu_temb_h = _build_silu_temb(
            model_t,
            te_lin1_w, te_lin1_b, te_lin2_w, te_lin2_b,
            ctx,
        )

        if dora_active:
            var fwd_dora = qwenimage_stack_direct_dora_forward_offload[H, Dh, N_IMG, N_TXT, S](
                noisy.copy(), txt_tokens.copy(), silu_temb_h.copy(),
                base, loader, dora_masters, direct_targets,
                cos_h.copy(), sin_h.copy(),
                norm_out_w, norm_out_b,
                Int(D), Int(FMLP), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), EPS, ctx,
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
            if k == 1:
                first_loss = loss_dora
            last_loss = loss_dora

            var grads_dora = qwenimage_stack_direct_dora_backward_offload[H, Dh, N_IMG, N_TXT, S](
                d_loss_dora,
                noisy.copy(), txt_tokens.copy(), silu_temb_h.copy(),
                base, loader, dora_masters, direct_targets,
                cos_h.copy(), sin_h.copy(),
                norm_out_w, norm_out_b,
                fwd_dora,
                Int(D), Int(FMLP), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), EPS, ctx,
            )
            var dnorm = qwen_direct_dora_grad_norm(grads_dora.grads)
            if dnorm > Float64(train_cfg.max_grad_norm):
                qwen_direct_dora_clip_grads(grads_dora.grads, train_cfg.max_grad_norm / Float32(dnorm))
            var step_lr = serenity_lr_for_optimizer_step(train_cfg, k)
            qwen_direct_dora_adamw_step(
                dora_masters, grads_dora.grads, k, step_lr,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps, train_cfg.weight_decay,
            )

            var t1_dora = perf_counter_ns()
            var secs_dora = Float64(t1_dora - t0) / 1.0e9
            print_trainer_progress(
                String("QwenImage-dora"), k, run_steps, 1,
                loss_dora, dnorm, secs_dora, 0.0,
                Float64(t1_dora - train_start) / 1.0e9,
            )
            print("[QwenImage-dora] step=", k, " grad_norm=", Float32(dnorm),
                  " zero_leg_l1=", qwen_direct_dora_zero_leg_l1(dora_masters))
            if grads_dora.nonfinite_grads != 0:
                print("[QwenImage-dora] warning nonfinite_grads=", grads_dora.nonfinite_grads)

            if train_cfg.save_every > 0 and k % train_cfg.save_every == 0:
                var ckpt_path = _step_lora_path(output_lora_path, k)
                var nmods = save_qwen_direct_dora(dora_masters, ckpt_path, ctx)
                print("[checkpoint] saved QwenImage-dora step=", k,
                      " modules=", nmods, " path=", ckpt_path)
                _qwen_prune_old_checkpoints(train_cfg, output_lora_path, k)
            continue

        if oft_active:
            var fwd_oft = qwenimage_stack_direct_oft_forward_offload[H, Dh, N_IMG, N_TXT, S](
                noisy.copy(), txt_tokens.copy(), silu_temb_h.copy(),
                base, loader, oft_masters, direct_targets,
                cos_h.copy(), sin_h.copy(),
                norm_out_w, norm_out_b,
                Int(D), Int(FMLP), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), EPS, ctx,
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
            if k == 1:
                first_loss = loss_oft
            last_loss = loss_oft

            var grads_oft = qwenimage_stack_direct_oft_backward_offload[H, Dh, N_IMG, N_TXT, S](
                d_loss_oft,
                noisy.copy(), txt_tokens.copy(), silu_temb_h.copy(),
                base, loader, oft_masters, direct_targets,
                cos_h.copy(), sin_h.copy(),
                norm_out_w, norm_out_b,
                fwd_oft,
                Int(D), Int(FMLP), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), EPS, ctx,
            )
            var onorm = qwen_direct_oft_grad_norm(grads_oft.grads)
            if onorm > Float64(train_cfg.max_grad_norm):
                qwen_direct_oft_clip_grads(grads_oft.grads, train_cfg.max_grad_norm / Float32(onorm))
            var step_lr = serenity_lr_for_optimizer_step(train_cfg, k)
            qwen_direct_oft_adamw_step(
                oft_masters, grads_oft.grads, k, step_lr,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps, train_cfg.weight_decay,
            )

            var t1_oft = perf_counter_ns()
            var secs_oft = Float64(t1_oft - t0) / 1.0e9
            print_trainer_progress(
                String("QwenImage-oft"), k, run_steps, 1,
                loss_oft, onorm, secs_oft, 0.0,
                Float64(t1_oft - train_start) / 1.0e9,
            )
            print("[QwenImage-oft] step=", k, " grad_norm=", Float32(onorm),
                  " vec_l1=", qwen_direct_oft_vec_l1(oft_masters))
            if grads_oft.nonfinite_grads != 0:
                print("[QwenImage-oft] warning nonfinite_grads=", grads_oft.nonfinite_grads)

            if train_cfg.save_every > 0 and k % train_cfg.save_every == 0:
                var ckpt_path = _step_lora_path(output_lora_path, k)
                var nmods = save_qwen_direct_oft(oft_masters, ckpt_path, ctx)
                print("[checkpoint] saved QwenImage-oft step=", k,
                      " modules=", nmods, " path=", ckpt_path)
                _qwen_prune_old_checkpoints(train_cfg, output_lora_path, k)
            continue

        # ── forward + loss + backward (device arm DEFAULT, MJ-1084; host arm
        # = the byte-untouched oracle, QWEN_DEVICE_STACK=False to re-run it) ──
        var loss: Float32
        var grads: QwenLoraGradSet
        comptime if QWEN_DEVICE_STACK:
            if use_b2:
                # ── TRUE batch-2: sample-0 is prepped above; prep sample-1 (its own
                #    cache slot + noise/sigma/temb stream), then run the row-stacked
                #    b2 device stack. Each per-sample d_out is 0.5-scaled so the b2
                #    backward's in-GEMM sum = mean(g0, g1) = the 2-sample batch grad;
                #    loss = 0.5*(L0 + L1). Mirrors train_chroma_real.mojo use_b2. ──
                var step_seed1 = UInt64(2) if FIXED_SIGMA_SMOKE else (UInt64(k) + UInt64(7000003))
                var sigma1: Float32
                var model_t1: Float32
                if FIXED_SIGMA_SMOKE:
                    var smoke_idx1 = Int(FIXED_SIGMA_VAL * Float32(1000.0))
                    if smoke_idx1 >= 1000:
                        smoke_idx1 = 999
                    sigma1 = Float32(Float64(smoke_idx1 + 1) / 1000.0)
                    model_t1 = Float32(Float64(smoke_idx1) / 1000.0)
                else:
                    var dts1 = sample_timestep_discrete_qwen(
                        SEED_BASE + step_seed1, train_cfg.timestep_shift, 1000
                    )
                    sigma1 = dts1.sigma
                    model_t1 = dts1.model_t

                var img_tokens1 = List[Float32]()
                var txt_tokens1 = List[Float32]()
                if have_cache and len(files) > 0:
                    var slot1 = 0 if FIXED_SIGMA_SMOKE else (k % len(files))
                    var cst1 = SafeTensors.open(files[slot1])
                    var latent_cache1 = _load_cache_preserving_dtype(cst1, String("latent"), ctx)
                    var latent_h1 = latent_cache1.to_host_bf16(ctx)
                    for i in range(len(latent_h1)):
                        img_tokens1.append(latent_h1[i].cast[DType.float32]())
                    var txt_cache1 = _load_cache_preserving_dtype(cst1, String("text_embedding"), ctx)
                    var txt_flat1 = txt_cache1.to_host_bf16(ctx)
                    var txt_seq1 = len(txt_flat1) // Int(TXT_CH)
                    for r in range(Int(N_TXT)):
                        if r < txt_seq1:
                            for c in range(Int(TXT_CH)):
                                txt_tokens1.append(txt_flat1[r * Int(TXT_CH) + c].cast[DType.float32]())
                        else:
                            for _ in range(Int(TXT_CH)):
                                txt_tokens1.append(Float32(0.0))
                else:
                    for _ in range(Int(N_IMG) * Int(IN_CH)):
                        img_tokens1.append(Float32(0.0))
                    for _ in range(Int(N_TXT) * Int(TXT_CH)):
                        txt_tokens1.append(Float32(0.0))

                var noise1 = _host_noise(Int(N_IMG) * Int(IN_CH), SEED_BASE * UInt64(7919) + step_seed1)
                var noisy1 = List[Float32]()
                var target1 = List[Float32]()
                var one_minus_sigma1 = Float32(1.0) - sigma1
                for i in range(len(img_tokens1)):
                    noisy1.append(one_minus_sigma1 * img_tokens1[i] + sigma1 * noise1[i])
                    target1.append(noise1[i] - img_tokens1[i])

                var silu_temb_h1 = _build_silu_temb(
                    model_t1, te_lin1_w, te_lin1_b, te_lin2_w, te_lin2_b, ctx,
                )

                # cos_h/sin_h are position-based -> identical for both samples.
                var fwd = qwenimage_stack_lora_forward_offload_device_b2[H, Dh, N_IMG, N_TXT, S](
                    noisy.copy(), txt_tokens.copy(), silu_temb_h.copy(),
                    noisy1.copy(), txt_tokens1.copy(), silu_temb_h1.copy(),
                    base, loader, lora,
                    cos_h.copy(), sin_h.copy(),
                    norm_out_w, norm_out_b,
                    Int(D), Int(FMLP), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), EPS, ctx,
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
                grads = qwenimage_stack_lora_backward_offload_device_b2[H, Dh, N_IMG, N_TXT, S](
                    d_out0, d_out1,
                    silu_temb_h.copy(), silu_temb_h1.copy(),
                    base, loader, lora,
                    cos_h.copy(), sin_h.copy(),
                    norm_out_w, norm_out_b,
                    fwd,
                    Int(D), Int(FMLP), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), EPS, ctx,
                )
            else:
                var fwd = qwenimage_stack_lora_forward_offload_device[H, Dh, N_IMG, N_TXT, S](
                    noisy.copy(), txt_tokens.copy(), silu_temb_h.copy(),
                    base, loader, lora,
                    cos_h.copy(), sin_h.copy(),
                    norm_out_w, norm_out_b,
                    Int(D), Int(FMLP), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), EPS, ctx,
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
                grads = qwenimage_stack_lora_backward_offload_device[H, Dh, N_IMG, N_TXT, S](
                    d_loss,
                    noisy.copy(), txt_tokens.copy(), silu_temb_h.copy(),
                    base, loader, lora,
                    cos_h.copy(), sin_h.copy(),
                    norm_out_w, norm_out_b,
                    fwd,
                    Int(D), Int(FMLP), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), EPS, ctx,
                )
        else:
            var fwd = qwenimage_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
                noisy.copy(), txt_tokens.copy(), silu_temb_h.copy(),
                base, loader, lora,
                cos_h.copy(), sin_h.copy(),
                norm_out_w, norm_out_b,
                Int(D), Int(FMLP), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), EPS, ctx,
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
            grads = qwenimage_stack_lora_backward_offload[H, Dh, N_IMG, N_TXT, S](
                d_loss,
                noisy.copy(), txt_tokens.copy(), silu_temb_h.copy(),
                base, loader, lora,
                cos_h.copy(), sin_h.copy(),
                norm_out_w, norm_out_b,
                fwd,
                Int(D), Int(FMLP), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), EPS, ctx,
            )
        if k == 1:
            first_loss = loss
        last_loss = loss

        # ── gradient accumulation (item 2h; default-off when N==1) ────────────
        # SUM this micro-step's grads into the shared trainer_core window; on a
        # non-boundary micro-step print progress and skip clip+AdamW; on the
        # boundary MEAN (÷N) back into grads.d_a/d_b so the UNCHANGED clip+AdamW
        # below run once. The window math lives in trainer_core.GradAccumWindow
        # (byte-op-identical to the prior inline block); the boundary control flow
        # (progress print + continue) stays here. k==run_steps force-flushes the
        # tail partial window. The recomputed norm is discarded (the _clip below
        # recomputes it from the meaned grads).
        if use_grad_accum:
            var is_boundary = accum_window.accumulate(grads.d_a, grads.d_b, k == run_steps)
            if not is_boundary:
                var t1m = perf_counter_ns()
                print_trainer_progress(
                    String("QwenImage-lora"), k, run_steps, 1,
                    loss, 0.0, Float64(t1m - t0) / 1.0e9, 0.0,
                    Float64(t1m - train_start) / 1.0e9,
                )
                continue
            _ = accum_window.finalize_mean(grads.d_a, grads.d_b)

        # ── grad norm + clip(1.0) ──
        var gn_before = _clip(grads, train_cfg.max_grad_norm)

        # ── AdamW ──
        # Wave 2A scheduled lr keys on OPTIMIZER steps, not micro-steps; with
        # accum_steps=1 this is ((k-1)//1)+1 == k => baseline unchanged.
        var optimizer_step = ((k - 1) // accum_steps) + 1
        var step_lr = serenity_lr_for_optimizer_step(train_cfg, optimizer_step)
        if lokr_active:
            var mg = qwen_lokr_chain_all(lokr_masters, grads.d_a, grads.d_b)
            var mnorm = qwen_lokr_grad_norm(mg)
            if mnorm > Float64(train_cfg.max_grad_norm):
                qwen_lokr_clip_grads(mg, train_cfg.max_grad_norm / Float32(mnorm))
            qwen_lokr_adamw_step(
                lokr_masters, mg, k, step_lr,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps, train_cfg.weight_decay,
            )
            lora = qwen_lokr_carrier_set(lokr_masters, train_cfg.d_model, train_cfg.mlp_hidden)
            print("[QwenImage-lokr] step=", k, " master_grad_norm=", Float32(mnorm),
                  " zero_leg_l1=", qwen_lokr_zero_leg_l1(lokr_masters))
        elif loha_active:
            var mg = qwen_loha_chain_all(loha_masters, grads.d_a, grads.d_b)
            var mnorm = qwen_loha_grad_norm(mg)
            if mnorm > Float64(train_cfg.max_grad_norm):
                qwen_loha_clip_grads(mg, train_cfg.max_grad_norm / Float32(mnorm))
            qwen_loha_adamw_step(
                loha_masters, mg, k, step_lr,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps, train_cfg.weight_decay,
            )
            lora = qwen_loha_carrier_set(loha_masters, train_cfg.d_model, train_cfg.mlp_hidden)
            print("[QwenImage-loha] step=", k, " master_grad_norm=", Float32(mnorm),
                  " zero_leg_l1=", qwen_loha_zero_leg_l1(loha_masters))
        else:
            # MJ-1070 close-out: resident fused AdamW (persistent device
            # P/M/V, ONE-TIME pinned staging) replaces the host scalar loop
            # the per-step-staging segfault forced us onto.
            if not adamw_state_ready:
                adamw_dev_state = Optional[LoraAdamWPlainDeviceState](
                    qwen_lora_adamw_state_init(lora, ctx)
                )
                adamw_state_ready = True
                print("[qwen-adamw] resident fused state initialized (",
                      len(lora.dbl), "adapters )")
            var _optimizer_timer = perf_counter_ns()
            qwen_offload_lora_adamw_step_resident(
                adamw_dev_state.value(), lora, grads, optimizer_step, step_lr, ctx,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps, train_cfg.weight_decay,
            )
            print("[qwen-opt-ms]", Float64(perf_counter_ns() - _optimizer_timer) / 1.0e6)
            # T1.B: EMA shadow update post-AdamW (plain arm; once per OPTIMIZER
            # step — this branch runs only at grad-accum boundaries). Off => skip.
            if train_cfg.ema_enabled:
                ema_update(ema, lora.dbl, optimizer_step)

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        print_trainer_progress(
            String("QwenImage-lora"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start) / 1.0e9,
        )
        if grads.nonfinite_lora_grads != 0:
            print("[QwenImage-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)

        var saved_this_step = False
        if train_cfg.save_every > 0 and k % train_cfg.save_every == 0:
            var ckpt_path = _step_lora_path(output_lora_path, k)
            if lokr_active:
                _ = save_qwen_lokr(lokr_masters, ckpt_path, ctx)
            elif loha_active:
                _ = save_qwen_loha(loha_masters, ckpt_path, ctx)
            else:
                _ = save_qwen_lora(lora, ckpt_path, ctx)
                if train_cfg.ema_enabled:  # T1.B EMA sibling next to every save
                    _save_qwen_lora_ema(ema, lora, ckpt_path, ctx)
                var ckpt_state = qwen_state_path_for_lora(ckpt_path)
                _ = save_qwen_lora_state(lora, ckpt_state, ctx)
            saved_this_step = True
            print("[checkpoint] saved step=", k, " path=", ckpt_path)
            _qwen_prune_old_checkpoints(train_cfg, output_lora_path, k)
        if should_sample_completed_step(sample_cadence, k):
            if qwen_should_save_before_sample(sample_cadence, k, saved_this_step):
                var pre_sample_path = _step_lora_path(output_lora_path, k)
                if lokr_active:
                    _ = save_qwen_lokr(lokr_masters, pre_sample_path, ctx)
                elif loha_active:
                    _ = save_qwen_loha(loha_masters, pre_sample_path, ctx)
                else:
                    _ = save_qwen_lora(lora, pre_sample_path, ctx)
                    if train_cfg.ema_enabled:  # T1.B EMA sibling
                        _save_qwen_lora_ema(ema, lora, pre_sample_path, ctx)
                    var pre_sample_state = qwen_state_path_for_lora(pre_sample_path)
                    _ = save_qwen_lora_state(lora, pre_sample_state, ctx)
                print("[checkpoint] saved before sample step=", k, " path=", pre_sample_path)
            print(
                "[cadence] sample due at completed_step=", k,
                " sample_file=", sample_cadence.sample_definition_file_name,
            )
            # Sample from the CURRENT frozen base + streamed blocks + LIVE LoRA.
            if caps_active:
                # Prompt-faithful: one render per ENABLED prompt, conditioned on
                # THAT prompt's caps (not the step's cached caption).
                for pi in range(len(sample_cfg.prompts)):
                    var prompt = sample_cfg.prompts[pi].copy()
                    if not prompt.enabled:
                        continue
                    var cond = _qwen_txt_cond_from_caps(prompt.caps_pos, prompt.label, ctx)
                    var uncond = List[Float32]()
                    if prompt.cfg != Float32(1.0) and prompt.caps_neg != String(""):
                        uncond = _qwen_txt_cond_from_caps(prompt.caps_neg, prompt.label, ctx)
                    _qwen_run_sample_caps(
                        base, loader, lora, cond^, uncond^,
                        cos_h.copy(), sin_h.copy(), norm_out_w, norm_out_b,
                        sample_vae_dir, samples_dir, k, prompt,
                        prompt.seed + UInt64(pi), ctx,
                    )
            elif have_cache and len(files) > 0:
                # Legacy: reuse this step's cached caption embeds (txt_tokens) as
                # COND, zeros as UNCOND — NOT a prompt-faithful validation.
                warn_legacy_cached_caption_sampling(String("QwenImage"))
                _qwen_run_sample(
                    base, loader, lora, txt_tokens.copy(),
                    cos_h.copy(), sin_h.copy(),
                    norm_out_w, norm_out_b,
                    sample_vae_dir, samples_dir, k, ctx,
                )
            else:
                print("[QwenImage-lora] sample skipped step=", k,
                      " (no validation_prompts_file and no cache; synthetic zero conditioning)")

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    for i in range(n_adapters):
        b_absum_final += _absum(lora.dbl[i].b)
    var trains: Bool
    var carrier_zero_final = Float64(0.0)
    if lokr_active:
        carrier_zero_final = qwen_lokr_zero_leg_l1(lokr_masters)
        trains = carrier_zero_final > carrier_zero_init
    elif loha_active:
        carrier_zero_final = qwen_loha_zero_leg_l1(loha_masters)
        trains = carrier_zero_final > carrier_zero_init
    elif dora_active:
        carrier_zero_final = qwen_direct_dora_zero_leg_l1(dora_masters)
        trains = carrier_zero_final > carrier_zero_init
    elif oft_active:
        carrier_zero_final = qwen_direct_oft_vec_l1(oft_masters)
        trains = carrier_zero_final > carrier_zero_init
    else:
        trains = (b_absum_init == 0.0) and (b_absum_final > 0.0)
    if trains and (last_loss == last_loss):
        if lycoris_active:
            print("RESULT: REAL run OK — LyCORIS trainable grew ",
                  carrier_zero_init, " -> ", carrier_zero_final,
                  "; loss", first_loss, "->", last_loss,
                  (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        else:
            print("RESULT: REAL run OK — LoRA-B grew 0 ->", b_absum_final,
                  "; loss", first_loss, "->", last_loss,
                  (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        if lokr_active:
            _ = save_qwen_lokr(lokr_masters, output_lora_path, ctx)
        elif loha_active:
            _ = save_qwen_loha(loha_masters, output_lora_path, ctx)
        elif dora_active:
            var nmods = save_qwen_direct_dora(dora_masters, output_lora_path, ctx)
            print("[QwenImage-dora] save final modules=", nmods, " path=", output_lora_path)
        elif oft_active:
            var nmods = save_qwen_direct_oft(oft_masters, output_lora_path, ctx)
            print("[QwenImage-oft] save final modules=", nmods, " path=", output_lora_path)
        else:
            _ = save_qwen_lora(lora, output_lora_path, ctx)
            if train_cfg.ema_enabled:  # T1.B EMA sibling
                _save_qwen_lora_ema(ema, lora, output_lora_path, ctx)
            _ = save_qwen_lora_state(lora, qwen_state_path_for_lora(output_lora_path), ctx)
    else:
        print("RESULT: FAIL trains=", trains)
