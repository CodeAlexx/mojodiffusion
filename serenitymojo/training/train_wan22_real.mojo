# serenitymojo/training/train_wan22_real.mojo
#
# REAL Wan2.2-T2V (A14B, low-noise expert) LoRA TRAINING LOOP.
# Pure Mojo + MAX, GPU, block-swap offload. Wires the EXISTING, parity-green
# engine (models/wan22/wan22_stack_lora.mojo) into a torchref-recipe training loop.
# This file does NOT reimplement any block/stack math — it is the LOOP only.
#
# ── ORACLE / RECIPE ──────────────────────────────────────────────────────────
# Oracle = torchref-ss/torchref `wan_train_network.py` + `hv_train_network.py`
# (trainer_base). Recipe this loop matches:
#   * Latent + text come from cache (Wan2.1-VAE 16-ch latents; umt5-xxl varlen
#     text). scale_shift_latents is IDENTITY for Wan — NO extra latent scaling
#     (wan_train_network.py; differs from SD3/Flux VAE shift+scale).
#   * Flow-match noised input:  x_t = (1 - t)*x0 + t*noise,  t in (0,1], t=1 is
#     pure noise. Training TARGET = velocity = (noise - x0). Loss = plain
#     per-element MSE then mean (NO min-SNR, weighting_scheme=None by default).
#   * Timestep sampling — BOTH modes, config/env selectable:
#       - "sigma"  (torchref DEFAULT): t ~ Uniform(0,1].
#       - "shift"  (doc-recommended): t = sigmoid(randn * sigmoid_scale), then the
#         discrete-flow shift map  t = (t*S)/(1 + (S-1)*t),  S = discrete_flow_shift
#         (recommended 3.0). Enable with env WAN22_SHIFT_TIMESTEPS=1;
#         discrete_flow_shift is read from config `timestep_shift` (default 3.0).
#   * Optimizer: AdamW (betas .9/.999, eps 1e-8, wd 0.01), lr 2e-4 doc example,
#     max_grad_norm 1.0, bf16 adapters (grads/masters F32 inside LoraAdapter).
#   * SINGLE-EXPERT (low-noise) LoRA on the existing 8 attention targets per
#     block: self_attn.{q,k,v,o} + cross_attn.{q,k,v,o} (40 blocks -> 320
#     adapters). build_wan22_lora_set / wan22_stack_lora_forward_offload /
#     wan22_stack_lora_backward_offload / wan22_lora_adamw_step / save_wan22_lora.
#
# ── A14B DIMS (the WIRED engine — NOT the ti2v_5b config) ────────────────────
#   dim=5120, blocks=40, heads=40, head_dim=128 (40*128=5120), ffn=13824,
#   in_ch=64 (patch 16ch * 1*2*2), out_ch=64, text_dim=4096, freq_dim=256,
#   eps=1e-6, rope_theta=10000. These match weights.mojo (patch_embedding
#   [5120,64], head.head [64,5120]) and offload/wan22_plan.mojo (40 blocks).
#
# ── CHECKPOINT-ABSENT GUARD ──────────────────────────────────────────────────
# The checkpoint symlink
#   /home/alex/.serenity/models/checkpoints/wan2.2_t2v_low_noise_14b_fp16.safetensors
# is DOWNLOADING and may not exist. Weight-loading + the real step are guarded
# behind file existence; if absent we print a clear banner and exit 0 (the
# non-weight code paths — config read, geometry, RNG, flow-match math — still run
# and type-check).
#
# ── TODO SEAMS (this is the FIRST version; do NOT build these now) ────────────
#   (a) FFN LoRA targets — see FFN-LoRA-TODO below (extend the 8-slot attn set to
#       include ffn.0 / ffn.2; requires a stack surface that adapts the FFN
#       linears, not present yet).
#   (b) Dual high/low-noise expert switching at the t=0.875 boundary — DONE.
#       Two resident bases + two streaming loaders (low + high checkpoints); per
#       step t>=boundary selects the HIGH expert, else LOW; ONE shared LoRA trains
#       across both (Torchref wan_train_network.py swap_high_low_weights:584). Enable
#       by presence of the high-noise checkpoint (WAN22_DUAL_EXPERT=0 forces off).
#       This is the switching-mechanism smoke — per-block grad parity is already
#       certified; dual switching is NOT a new parity claim.
#   (c) Real cache dataloader — see CACHE-TODO below (Wan2.1-VAE 16ch latent
#       patchify -> [S,64] + umt5 varlen text -> [TXT,4096]); this version
#       synthesizes x0/text so the loop is runnable once weights land.
#
# ── BUILD / RUN ──────────────────────────────────────────────────────────────
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/training/train_wan22_real.mojo \
#       [config.json]
# Default config: serenitymojo/configs/wan22_direct_dora_1step_smoke.json
# (video model = LoRA/LoCon only; network_algorithm=dora/oft/lokr/loha now FAIL-LOUD
#  via require_lora_or_locon_linear rather than silently training plain LoRA).

from max.gpu.host import DeviceContext
from std.collections import List
from std.math import sqrt, exp
from std.time import perf_counter_ns
from std.sys import argv
from std.ffi import external_call
from std.memory import alloc

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.patchify3d import patchify3d, unpatchify3d
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.io.ffi import sys_system
from serenitymojo.image.png import ValueRange, save_png
from serenitymojo.models.lingbotvideo.vae_decoder import LingBotWanVaeDecoder
from serenitymojo.models.lingbotvideo.vae_encoder import _latents_mean, _latents_std
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.train_config import TrainConfig
from serenitymojo.training.adapter_algo_policy import require_lora_or_locon_linear

# PROVEN inference-geometry RoPE (3-axis interleaved) — reused EXACTLY, not
# reimplemented (models/dit/wan22_dit.mojo).
from serenitymojo.models.dit.wan22_dit import wan22_build_rope

from serenitymojo.models.wan22.weights import load_wan22_stack_base, detect_wan22_prefix
from serenitymojo.training.trainer_core import (
    trainer_resolve_resume_path, trainer_warn_warm_resume,
)
from serenitymojo.training.lora_save import (
    lora_train_state_has_moments, load_lora_train_state_meta,
)
from serenitymojo.models.wan22.wan22_stack_lora import (
    Wan22StackBase, Wan22LoraSet, Wan22LoraGradSet, Wan22StackForward,
    build_wan22_lora_set, wan22_total_adapters,
    wan22_stack_lora_forward_offload, wan22_stack_lora_backward_offload,
    wan22_stack_lora_backward_offload_devnative, Wan22LoraDeviceGradSet,
    wan22_stack_lora_backward_offload_graph,
    wan22_lora_adamw_step, save_wan22_lora, save_wan22_lora_state,
    load_wan22_lora_state, load_wan22_lora_resume,
    Wan22I2VLoraSet, Wan22I2VLoraGradSet, Wan22I2VStackForward,
    build_wan22_i2v_lora_set, wan22_i2v_total_adapters,
    wan22_i2v_stack_lora_forward_offload, wan22_i2v_stack_lora_backward_offload,
    wan22_i2v_lora_adamw_step, save_wan22_i2v_lora, save_wan22_i2v_lora_state,
)
from serenitymojo.training.lora_adamw_plain_fused import (
    LoraAdamWPlainDeviceState, lora_adamw_plain_device_state_init,
    fused_lora_adamw_plain_step_resident_device_grads,
)
from serenitymojo.offload.wan22_plan import (
    build_wan22_block_plan, build_wan21_i2v_block_plan,
)
from serenitymojo.offload.plan import OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader

# ── Wan2.1 I2V-14B: the DUAL cross-attn block (WanI2VCrossAttention) + 12-target
#    LoRA + the frozen CLIP MLPProj context assembler (WAN21_I2V=1 path). The
#    block fwd/bwd is CERTIFIED against Torchref at cos>=0.999 by
#    models/wan22/parity/wan22_i2v21_block_lora_parity.mojo. ──────────────────
from std.collections import Optional
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.activations import gelu, gelu_exact
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.wan22.wan22_block import (
    WanBlockWeights, WanBlockLora, WanModVecs,
    WanI2VBlockWeights, WanI2VBlockLora,
    wan22_i2v_block_lora_forward, wan22_i2v_block_lora_backward,
)


# ── A14B architecture (comptime; the wired engine's dims) ─────────────────────
comptime H = 40
comptime Dh = 128
comptime DIM = H * Dh            # 5120
comptime FFN = 13824
comptime IN_CH = 64              # T2V patch: 16 latent ch * 1 * 2 * 2
comptime OUT_CH = 64             # velocity of the 16-ch latent (BOTH T2V & I2V)
# ── I2V-A14B input channels (WAN22_I2V=1) ─────────────────────────────────────
# I2V feeds concat([noisy_latent(16ch), y(20ch)]) = 36ch to the Conv3d(36->5120,
# k/s (1,2,2)) patch embed. y = concat([mask(4), image_latent(16)]). Patchified
# (1,2,2): noisy_latent -> [S,64], cond_y -> [S,COND_PACK=80], concatenated
# per-token -> [S, IN_CH_I2V=144]. patchify3d packs channel-SLOWEST
# (ops/patchify3d.mojo:19), so per-token concat(patch(16), patch(20)) is BYTE-
# IDENTICAL to patchify(concat(36)) — the packed order is [16 latent chans ->
# cols 0..63][20 y chans -> cols 64..143], exactly the [5120,144] patch-embed
# weight layout. OUT_CH stays 64 (the model predicts velocity of the 16-ch
# latent only; y is clean conditioning, never noised, never a target).
comptime COND_CH = 20            # y = mask(4) + image_latent(16)
comptime COND_PACK = COND_CH * 1 * 2 * 2   # 80
comptime IN_CH_I2V = IN_CH + COND_PACK     # 64 + 80 = 144
comptime I2V_BOUNDARY = Float32(0.900)     # Wan2.2 I2V dual-expert split (vs 0.875 T2V)
comptime DEFAULT_CKPT_I2V =
    "/home/alex/.serenity/models/checkpoints/wan2.2_i2v_low_noise_14b_fp16.safetensors"
comptime TEXT_DIM = 4096
comptime FREQ_DIM = 256
comptime NUM_BLOCKS = 40
comptime EPS = Float32(1.0e-6)
comptime ROPE_THETA = Float32(10000.0)

# ── REAL smoke geometry: latent [16, F=1, H=32, W=32] (a 256px image -> Wan2.1
#    VAE /8 -> 32x32, single latent frame). After the Conv3d patch (1,2,2), the
#    PATCH GRID is (FG, HG, WG) = (1, 16, 16) and S = FG*HG*WG = 256 tokens of
#    packed-dim IN_CH=64. These drive the PROVEN rope/patchify geometry exactly
#    (models/dit/wan22_dit.mojo forward uses wan22_build_rope(FG,HG,WG,...)).
#    CACHE-TODO: real F/H/W come from the cached VAE latent shapes.
comptime FG = 1
comptime HG = 16               # latent H=32 -> patch-grid 32/2 = 16
comptime WG = 16               # latent W=32 -> patch-grid 32/2 = 16
comptime S = FG * HG * WG        # 256 image tokens
comptime TXT = 512               # umt5 cross-attn kv length (torchref text_len)

# ── optimizer / recipe defaults (config overrides where present) ──────────────
comptime DEFAULT_LR = Float32(2.0e-4)          # torchref doc example
comptime DEFAULT_MAX_GRAD_NORM = Float32(1.0)  # hv_train_network default
comptime DEFAULT_SIGMOID_SCALE = Float32(1.0)  # shift-mode sigmoid scale
comptime DEFAULT_DISCRETE_FLOW_SHIFT = Float32(3.0)
comptime DUAL_EXPERT_BOUNDARY = Float32(0.875)  # DUAL-EXPERT-TODO boundary
comptime DEFAULT_CKPT =
    "/home/alex/.serenity/models/checkpoints/wan2.2_t2v_low_noise_14b_fp16.safetensors"
comptime DEFAULT_CONFIG =
    "serenitymojo/configs/wan22_real_smoke_2step.json"

# ── Wan 2.1 T2V (single-expert) checkpoints — original Wan keys (blocks.N.*,
#    patch_embedding, text_embedding, time_embedding, time_projection, head).
#    1.3B: dim1536/30 blocks; 14B: dim5120/40 blocks. Selected by env WAN21_MODEL.
comptime DEFAULT_CKPT_WAN21_1P3B =
    "/home/alex/.serenity/models/checkpoints/wan2.1_t2v_1.3B_fp16.safetensors"
comptime DEFAULT_CKPT_WAN21_14B =
    "/home/alex/.serenity/models/checkpoints/wan2.1_t2v_14B_fp16.safetensors"
# 1.3B geometry deltas (S/IN_CH/TXT/Dh/head_dim identical to A14B).
comptime WAN21_1P3B_DIM = 1536
comptime WAN21_1P3B_FFN = 8960
comptime WAN21_1P3B_HEADS = 12
comptime WAN21_1P3B_BLOCKS = 30
# 14B geometry (dims == A14B; single expert).
comptime WAN21_14B_DIM = 5120
comptime WAN21_14B_FFN = 13824
comptime WAN21_14B_HEADS = 40
comptime WAN21_14B_BLOCKS = 40

# ── Wan2.1 I2V-14B (WAN21_I2V=1) — the DUAL cross-attn variant ────────────────
# dim=5120, ffn=13824, heads=40, blocks=40, in_dim=36 (mask4 + image_latent16 +
# noise16 = the SAME 36-ch concat as Wan2.2 I2V, so packed IN_CH_I2V=144),
# boundary=None (SINGLE expert — NO high/low-noise switch), v2_2=False. The ONLY
# new compute vs T2V is WanI2VCrossAttention: 257 CLIP-image tokens (projected by
# the frozen img_emb/MLPProj) are PREPENDED to the umt5 text context, and every
# block runs a SECOND cross-attn (shared query, image k/v) whose output is added
# before the shared output proj. LoRA = the 10 T2V targets PLUS cross_attn.k_img
# + cross_attn.v_img = 12/block.
comptime WAN21_I2V_14B_DIM = 5120
comptime WAN21_I2V_14B_FFN = 13824
comptime WAN21_I2V_14B_HEADS = 40
comptime WAN21_I2V_14B_BLOCKS = 40
comptime CLIP_TOKENS = 257       # 256 patch + 1 cls (xlm-roberta-vit-h-14)
comptime CLIP_DIM = 1280         # penultimate CLIP visual width
comptime DEFAULT_CKPT_WAN21_I2V =
    "/home/alex/.serenity/models/checkpoints/wan2.1_i2v_480p_14B_fp16.safetensors"


# ── file existence (pure syscall; no builtin open) ────────────────────────────
def _file_exists(path: String) -> Bool:
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


# ── Resident-base file for a checkpoint that may be an fp8 per-block CACHE DIR.
# The Bernini-R E4M3 cache is a DIRECTORY (block_00..39.safetensors + shared
# .safetensors + a model.safetensors.index.json sidecar). The 15 non-block
# tensors (patch/text/time embeddings, time_projection, head) live in
# shared.safetensors, so the resident base is loaded from there; the streaming
# block loader still opens the DIR (ShardedSafeTensors resolves the index →
# per-block fp8 shards). A monolithic .safetensors file has no shared.safetensors
# sibling, so it is returned unchanged (back-compat with the fp16 Wan checkpoints).
def _base_file_for(ckpt: String) -> String:
    var shared = ckpt + String("/shared.safetensors")
    if _file_exists(shared):
        return shared
    return ckpt


# ── libc getenv: True iff env var is exactly "1" (sd35 trainer helper) ────────
comptime _EnvPtr = UnsafePointer[UInt8, MutExternalOrigin]


def _env_is_set(name: String) -> Bool:
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


# ── libc getenv → String ("" if unset). P3 (2026-07-22): the run knobs are
#    CONFIG-FIRST (TrainConfig keys dit_high_noise / timestep_boundary /
#    dual_expert / i2v / wan_variant / cache_dir); the env vars
#    (WAN22_DIT_HIGH_NOISE / WAN22_TIMESTEP_BOUNDARY / WAN22_DUAL_EXPERT /
#    WAN22_I2V / WAN21_MODEL / WAN22_DATA_CACHE / WAN22_DIT / WAN21_DIT) remain
#    as OVERRIDES — env wins ONLY when set (back-compat with smoke scripts).
#    The smoke timestep pins (WAN22_ALTERNATE_T / WAN22_T_LOW / WAN22_T_HIGH /
#    WAN22_SHIFT_TIMESTEPS) stay env-only (test pins). Same pattern as
#    ltx2_t2v_av_hq.mojo::_env_str.
def _env_f32(name: String) -> Float32:
    """Numeric env knob (returns 0.0 when unset/unparseable) — used for
    WAN22_FP8_RESIDENT_GB."""
    var v = _env_str(name)
    if v.byte_length() == 0:
        return Float32(0.0)
    try:
        return Float32(Float64(v))
    except:
        return Float32(0.0)


def _env_str(name: String) -> String:
    var n = name.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = name.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var ret = external_call["getenv", _EnvPtr](_EnvPtr(unsafe_from_address=Int(buf)))
    buf.free()
    var out = String("")
    if Int(ret) == 0:
        return out
    var i = 0
    while ret[i] != UInt8(0):
        out += chr(Int(ret[i]))
        i += 1
    return out^


# ── deterministic host RNG (splitmix64) ───────────────────────────────────────
def _splitmix64(state: UInt64) -> UInt64:
    var z = state + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _u01(seed: UInt64) -> Float32:
    # top 24 bits -> [0,1)
    var w = _splitmix64(seed)
    return Float32(Float64(w >> 40) * (1.0 / 16777216.0))


# Box-Muller N(0,1) from two independent uniforms on distinct seeds.
def _randn(seed: UInt64) -> Float32:
    var u1 = _u01(seed * UInt64(2654435761) + 1)
    var u2 = _u01(seed * UInt64(1442695040888963407) + 2)
    if u1 < Float32(1.0e-7):
        u1 = Float32(1.0e-7)
    var r = sqrt(Float32(-2.0) * _flog(u1))
    var ang = Float32(6.2831853071795864769) * u2
    return r * _fcos(ang)


# small host cos / natural-log helpers (avoid extra imports)
def _fcos(x: Float32) -> Float32:
    from std.math import cos as _c
    return Float32(_c(Float64(x)))


def _flog(x: Float32) -> Float32:
    from std.math import log as _l
    return Float32(_l(Float64(x)))


def _sigmoid(x: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + exp(-x))


# ── torchref timestep sampler (both modes) ──────────────────────────────────────
# "sigma"  : t ~ Uniform(0,1]  (torchref default).
# "shift"  : t = sigmoid(randn*sigmoid_scale); then t = (t*Sd)/(1+(Sd-1)*t),
#            Sd = discrete_flow_shift.
def _sample_t(
    shift_mode: Bool, step: Int, seed: UInt64,
    sigmoid_scale: Float32, discrete_flow_shift: Float32,
) -> Float32:
    var base = seed * UInt64(6364136223846793005) + UInt64(step) * UInt64(7919) + 17
    if not shift_mode:
        var t = _u01(base)
        # map [0,1) -> (0,1]  so t=0 (no-noise) never occurs, t=1 allowed.
        t = Float32(1.0) - t
        return t
    var z = _randn(base)
    var t = _sigmoid(z * sigmoid_scale)
    var num = t * discrete_flow_shift
    var den = Float32(1.0) + (discrete_flow_shift - Float32(1.0)) * t
    return num / den


# uniform noise ~ N(0,1)-ish (host); one scalar per element.
def _noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(_randn(seed + UInt64(i) * 2 + 1))
    return out^


# ── WAN PATCH-LAYOUT BRIDGE (loss/sampler correctness) ────────────────────────
# The patch embed consumes each token in Conv3d/input order `(c, pf, ph, pw)`,
# channel slowest. The trained head emits `(pf, ph, pw, c)`, channel fastest,
# because Wan immediately reshapes/unpatchifies that output before comparing it
# with the canonical latent-space velocity. `patchify3d` and `unpatchify3d` are
# intentionally NOT literal inverses, so the trainer must bridge the two orders.
#
# Wan uses patch (1,2,2), hence patch volume 4. For token `s`, channel `c`, and
# flattened patch coordinate `p`:
#   input index = s*(C*4) + c*4 + p
#   head  index = s*(C*4) + p*C + c
comptime WAN_PATCH_VOLUME = 4


def _wan_patch_input_to_head(
    values: List[Float32], tokens: Int, channels: Int,
) raises -> List[Float32]:
    """Repack `(c,pf,ph,pw)` token rows to WAN head `(pf,ph,pw,c)` rows."""
    if tokens <= 0 or channels <= 0:
        raise Error("WAN patch repack: tokens and channels must be positive")
    var packed = channels * WAN_PATCH_VOLUME
    if len(values) != tokens * packed:
        raise Error(
            String("WAN patch input->head: len ") + String(len(values))
            + String(" != tokens*channels*4 ") + String(tokens * packed)
        )
    var out = List[Float32]()
    out.reserve(len(values))
    for s in range(tokens):
        for p in range(WAN_PATCH_VOLUME):
            for c in range(channels):
                out.append(values[s * packed + c * WAN_PATCH_VOLUME + p])
    return out^


def _wan_patch_head_to_input(
    values: List[Float32], tokens: Int, channels: Int,
) raises -> List[Float32]:
    """Inverse repack: WAN head `(pf,ph,pw,c)` to input `(c,pf,ph,pw)`."""
    if tokens <= 0 or channels <= 0:
        raise Error("WAN patch repack: tokens and channels must be positive")
    var packed = channels * WAN_PATCH_VOLUME
    if len(values) != tokens * packed:
        raise Error(
            String("WAN patch head->input: len ") + String(len(values))
            + String(" != tokens*channels*4 ") + String(tokens * packed)
        )
    var out = List[Float32]()
    out.resize(len(values), Float32(0.0))
    for s in range(tokens):
        for p in range(WAN_PATCH_VOLUME):
            for c in range(channels):
                out[s * packed + c * WAN_PATCH_VOLUME + p] = (
                    values[s * packed + p * channels + c]
                )
    return out^


def _wan_patch_layout_self_gate() raises:
    """Tiny exact non-degenerate guard for the loss-layout bridge."""
    var input_order: List[Float32] = [
        0.0, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0,
        8.0, 9.0, 10.0, 11.0,
    ]
    var expected_head: List[Float32] = [
        0.0, 4.0, 8.0,
        1.0, 5.0, 9.0,
        2.0, 6.0, 10.0,
        3.0, 7.0, 11.0,
    ]
    var head = _wan_patch_input_to_head(input_order, 1, 3)
    for i in range(len(head)):
        if head[i] != expected_head[i]:
            raise Error(
                String("WAN patch layout gate: known mapping mismatch at ")
                + String(i)
            )
    var roundtrip = _wan_patch_head_to_input(head, 1, 3)
    for i in range(len(roundtrip)):
        if roundtrip[i] != input_order[i]:
            raise Error(
                String("WAN patch layout gate: roundtrip mismatch at ")
                + String(i)
            )

    # Permutation must preserve MSE when prediction and target are moved together.
    var pred_head: List[Float32] = [
        11.0, 9.0, 7.0, 5.0,
        3.0, 1.0, 0.0, 2.0,
        4.0, 6.0, 8.0, 10.0,
    ]
    var pred_input = _wan_patch_head_to_input(pred_head, 1, 3)
    var mse_head = Float32(0.0)
    var mse_input = Float32(0.0)
    for i in range(len(head)):
        var dh = pred_head[i] - head[i]
        var di = pred_input[i] - input_order[i]
        mse_head += dh * dh
        mse_input += di * di
    if mse_head != mse_input:
        raise Error("WAN patch layout gate: MSE changed across permutation")
    print("[gate] WAN patch input<->head mapping + MSE invariance: PASS")


# ── LEGACY IN-TRAIN EULER PREVIEW (DISABLED BY MAIN PREFLIGHT) ─────────────────
# Kept temporarily as a documented implementation reference while the
# process-separated request/result manifest is built. `main` rejects every
# sample_every>0 before DeviceContext because this training monomorphization is
# S=256, the standing 1024px sampler needs S=4096, and production validation uses
# the Torchref-aligned UniPC pipeline. The layout math below is nevertheless kept
# correct so this dormant code cannot preserve the old channel-permutation bug.
#
# Historical design choices:
#
# 1. TEXT CONDS COME FROM A FILE, not a text encoder in this process. umt5-xxl is
#    ~11 GB; co-residing it with two A14B experts on a 16 GB card is an OOM. Encode
#    once with `wan22_encode_prompt` and point WAN22_SAMPLE_CONDS at the artifact.
#    This is the same process-separation the rest of the Wan stack uses.
# 2. THE SAMPLER IS THE MODEL'S OWN FLOW-MATCH PARAMETERIZATION, not an imported
#    scheduler. Training defines x_t = (1-t)*x0 + t*noise with the model predicting
#    v = noise - x0 = dx/dt, so integrating t: 1 -> 0 with x -= dt*v is exactly the
#    inverse of the training objective. A preview that used a different scheduler
#    would be measuring the scheduler as much as the LoRA.
# 3. IT RUNS IN PATCHIFIED [S, IN_CH] SPACE, the trainer's native representation,
#    and only unpatchifies once at the end for the VAE.
comptime SAMPLE_STEPS = 30
comptime SAMPLE_CFG = Float32(5.0)
comptime SAMPLE_LAT_C = 16          # Wan2.1 VAE latent channels
comptime SAMPLE_LAT_H = 128         # 1024px / 8  (STANDING ORDER: samples are 1024)
comptime SAMPLE_LAT_W = 128
comptime SAMPLE_VAE = (
    "/mnt/disk1/models/lingbot-video-moe/vae/diffusion_pytorch_model.safetensors"
)


def _pad4(v: Int) -> String:
    var t = String(v)
    while t.byte_length() < 4:
        t = String("0") + t
    return t^


def _load_sample_conds(
    path: String, txt: Int, text_dim: Int, ctx: DeviceContext,
) raises -> List[List[Float32]]:
    """pos/neg [TXT,4096] host lists from a `wan22_encode_prompt` artifact."""
    var st = ShardedSafeTensors.open(path)
    var out = List[List[Float32]]()
    for ref key in [String("pos_embed"), String("neg_embed")]:
        if key not in st.names():
            raise Error(
                String("WAN22_SAMPLE_CONDS file is missing '") + key
                + String("' — produce it with `wan22_encode_prompt <pos> <neg> <out>`")
            )
        var t = Tensor.from_view_as_bf16(st.tensor_view(key), ctx)
        if t.numel() != txt * text_dim:
            raise Error(
                String("WAN22_SAMPLE_CONDS '") + key + String("' numel ")
                + String(t.numel()) + String(" != ") + String(txt * text_dim)
            )
        out.append(cast_tensor(t, STDtype.F32, ctx).to_host(ctx))
    return out^


def _wan22_sample_png[
    H: Int, Dh: Int, S: Int, TXT: Int
](
    tag: String, out_png: String,
    pos: List[Float32], neg: List[Float32],
    base: Wan22StackBase, mut loader: TurboPlannedLoader,
    base_high: Wan22StackBase, mut loader_high: TurboPlannedLoader,
    dual_expert: Bool, boundary: Float32,
    lora: Wan22LoraSet, cos: List[Float32], sin: List[Float32],
    dim: Int, ffn: Int, in_ch: Int, text_dim: Int, out_ch: Int, freq_dim: Int,
    eps: Float32, n_steps: Int, cfg_scale: Float32, shift: Float32, seed: UInt64,
    save_acts: Bool, device_lora: Bool, batched_cross: Bool,
    device_modvecs: Bool, device_modvecs_t: Bool,
    ctx: DeviceContext,
) raises:
    if in_ch != out_ch or out_ch != SAMPLE_LAT_C * WAN_PATCH_VOLUME:
        raise Error(
            "WAN in-train Euler preview only has a T2V 16-channel layout bridge"
        )
    var sample_tokens = (
        (SAMPLE_LAT_H // 2) * (SAMPLE_LAT_W // 2)
    )
    if S != sample_tokens:
        raise Error(
            String("WAN in-train preview token mismatch: training S=") + String(S)
            + String(", 1024px sampling needs S=") + String(sample_tokens)
            + String("; use process-separated pipeline/wan22_lora_sample")
        )
    var t_start = perf_counter_ns()
    var n = S * out_ch
    var x = _noise(n, seed * UInt64(7919) + 17)      # t = 1: pure noise

    # shift-mapped descending t schedule, t: 1 -> 0 (same map the trainer's
    # "shift" timestep mode uses, so preview and training agree on where t lives).
    var ts = List[Float32]()
    for i in range(n_steps + 1):
        var u = Float32(n_steps - i) / Float32(n_steps)
        ts.append((u * shift) / (Float32(1.0) + (shift - Float32(1.0)) * u))

    for i in range(n_steps):
        var t = ts[i]
        var dt = ts[i] - ts[i + 1]
        var t_model = t * Float32(1000.0) + Float32(1.0)
        var use_high = dual_expert and (t >= boundary)

        var fc: Wan22StackForward
        var fu: Wan22StackForward
        if use_high:
            fc = wan22_stack_lora_forward_offload[H, Dh, S, TXT](
                x.copy(), pos.copy(), t_model, base_high, loader_high, lora,
                cos.copy(), sin.copy(), dim, ffn, in_ch, text_dim, out_ch,
                freq_dim, eps, ctx, save_acts, device_lora, batched_cross,
                device_modvecs, device_modvecs_t,
            )
            fu = wan22_stack_lora_forward_offload[H, Dh, S, TXT](
                x.copy(), neg.copy(), t_model, base_high, loader_high, lora,
                cos.copy(), sin.copy(), dim, ffn, in_ch, text_dim, out_ch,
                freq_dim, eps, ctx, save_acts, device_lora, batched_cross,
                device_modvecs, device_modvecs_t,
            )
        else:
            fc = wan22_stack_lora_forward_offload[H, Dh, S, TXT](
                x.copy(), pos.copy(), t_model, base, loader, lora,
                cos.copy(), sin.copy(), dim, ffn, in_ch, text_dim, out_ch,
                freq_dim, eps, ctx, save_acts, device_lora, batched_cross,
                device_modvecs, device_modvecs_t,
            )
            fu = wan22_stack_lora_forward_offload[H, Dh, S, TXT](
                x.copy(), neg.copy(), t_model, base, loader, lora,
                cos.copy(), sin.copy(), dim, ffn, in_ch, text_dim, out_ch,
                freq_dim, eps, ctx, save_acts, device_lora, batched_cross,
                device_modvecs, device_modvecs_t,
            )
        # CFG is in head-output order `(pf,ph,pw,c)`. Repack it to the input/state
        # order `(c,pf,ph,pw)` before integrating the patchified latent.
        var v_head = List[Float32]()
        v_head.reserve(n)
        for j in range(n):
            v_head.append(fu.out[j] + cfg_scale * (fc.out[j] - fu.out[j]))
        var v_input = _wan_patch_head_to_input(v_head^, S, SAMPLE_LAT_C)
        for j in range(n):
            x[j] = x[j] - dt * v_input[j]

    # ── patch space -> latent -> pixels ──
    # `unpatchify3d` consumes the head's channel-fast order, not patch-embed order.
    var x_head = _wan_patch_input_to_head(x^, S, SAMPLE_LAT_C)
    var xt = Tensor.from_host(x_head^, [S, out_ch], STDtype.F32, ctx)
    var lat = unpatchify3d(
        xt, SAMPLE_LAT_C, 1, SAMPLE_LAT_H, SAMPLE_LAT_W, 1, 2, 2, ctx
    )
    var host = lat.to_host(ctx)
    # z = normalized_latent * std + mean (the gated Wan VAE encoder contract).
    var means = _latents_mean()
    var stds = _latents_std()
    var stride = SAMPLE_LAT_H * SAMPLE_LAT_W
    var zhost = List[Float32]()
    zhost.resize(len(host), Float32(0.0))
    for i in range(len(host)):
        var c = (i // stride) % SAMPLE_LAT_C
        zhost[i] = host[i] * stds[c] + means[c]
    var z = Tensor.from_host(
        zhost, [1, SAMPLE_LAT_C, 1, SAMPLE_LAT_H, SAMPLE_LAT_W], STDtype.F32, ctx
    )
    # VAE is loaded and dropped per sample point — it must not co-reside with the
    # experts for the other 99% of the run.
    var dec = LingBotWanVaeDecoder[SAMPLE_LAT_H, SAMPLE_LAT_W].load(SAMPLE_VAE, ctx)
    var video = dec.decode_video(z, ctx)
    var frame = reshape(video, [1, 3, SAMPLE_LAT_H * 8, SAMPLE_LAT_W * 8], ctx)
    save_png(frame, out_png, ctx, ValueRange.SIGNED)
    print("  [sample]", tag, "->", out_png,
          " (", Float64(perf_counter_ns() - t_start) / 1.0e9, "s )")


# CACHE-TODO: synthetic x0 in patch space [S, IN_CH]. Real path: load the cached
# Wan2.1-VAE 16-ch latent, patchify (1,2,2) -> [S,64]. scale_shift_latents is
# IDENTITY for Wan (no VAE shift/scale), so a cached latent is used as-is.
def _synth_x0(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(_randn(seed * UInt64(1099511628211) + UInt64(i) * 3 + 5))
    return out^


# ── RoPE host tables for the stack — the PROVEN 3-axis (frame,h,w) interleaved
#    Wan rope, reused EXACTLY from the parity-green inference geometry
#    (models/dit/wan22_dit.mojo::wan22_build_rope + _expand_rope_per_head).
#    wan22_build_rope(FG,HG,WG,Dh,theta,F32,ctx) returns cos/sin device tables
#    [S, Dh//2] for the (FG,HG,WG) patch grid (axes split [44,42,42] for Dh=128).
#    The stack + block consume cos/sin as the PER-TOKEN table [S, Dh//2] and do
#    the per-head expansion internally (wan22_block.mojo::_expand_rope_per_head,
#    identical to wan22_dit.mojo). So we return the un-expanded [S*Dh//2] host
#    list here — matching what wan22_block_forward's `cos: [S,Dh/2]` contract
#    (line 420) expects.
def _build_rope_proven(ctx: DeviceContext) raises -> List[List[Float32]]:
    var cs = wan22_build_rope(FG, HG, WG, Dh, ROPE_THETA, STDtype.F32, ctx)
    var cos_tok = cs[0].to_host(ctx)   # [S * Dh//2]
    var sin_tok = cs[1].to_host(ctx)   # [S * Dh//2]
    var res = List[List[Float32]]()
    res.append(cos_tok^)
    res.append(sin_tok^)
    return res^


# ── global grad clip over all LoRA d_A/d_B (torchref max_grad_norm) ─────────────
def _clip(mut grads: Wan22LoraGradSet, max_norm: Float32) -> Float32:
    var ss = Float32(0.0)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += grads.d_a[i][j] * grads.d_a[i][j]
        for j in range(len(grads.d_b[i])):
            ss += grads.d_b[i][j] * grads.d_b[i][j]
    var gn = sqrt(ss)
    if gn > max_norm and gn > Float32(0.0):
        var scl = max_norm / gn
        for i in range(len(grads.d_a)):
            for j in range(len(grads.d_a[i])):
                grads.d_a[i][j] = grads.d_a[i][j] * scl
            for j in range(len(grads.d_b[i])):
                grads.d_b[i][j] = grads.d_b[i][j] * scl
    return gn


# ── global grad clip over all 12/block I2V LoRA d_A/d_B ───────────────────────
def _clip_i2v(mut grads: Wan22I2VLoraGradSet, max_norm: Float32) -> Float32:
    var ss = Float32(0.0)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += grads.d_a[i][j] * grads.d_a[i][j]
        for j in range(len(grads.d_b[i])):
            ss += grads.d_b[i][j] * grads.d_b[i][j]
    var gn = sqrt(ss)
    if gn > max_norm and gn > Float32(0.0):
        var scl = max_norm / gn
        for i in range(len(grads.d_a)):
            for j in range(len(grads.d_a[i])):
                grads.d_a[i][j] = grads.d_a[i][j] * scl
            for j in range(len(grads.d_b[i])):
                grads.d_b[i][j] = grads.d_b[i][j] * scl
    return gn


# ══════════════════════════════════════════════════════════════════════════════
# REAL DATA CACHE READER (replaces the synthetic Box-Muller x0/text).
#
# Reads one sample_{i}.safetensors built by
#   models/wan22/parity/wan22_build_data_cache.py
# via the repo's SafeTensors mmap reader (io/safetensors + io/ffi — NEVER builtin
# open). Format (documented in the builder):
#   latent   : f32,  [16,1,32,32]  (Wan2.1-VAE 16ch, ALREADY mean/std-normalized)
#   text     : bf16, [512,4096]    (umt5-xxl, pad/trunc to TXT=512)
#   text_len : i32,  [1]           (true token count before pad)
#
# The clean latent [16,1,32,32] is patchified (1,2,2) via the PROVEN, parity-green
# ops/patchify3d op — the EXACT same unfold the inference DiT uses (wan22_dit.mojo
# line 764) — to [S=256, IN_CH=64], guaranteeing the layout the patch-embedding
# expects (no hand-rolled transpose). The resulting flat [S*IN_CH] list REPLACES
# the synthetic x0 (the flow-match noising/target math downstream is unchanged).
# ══════════════════════════════════════════════════════════════════════════════
# Count contiguous sample_0..sample_{n-1}.safetensors in `cache_dir`.
def _count_cache_samples(cache_dir: String) -> Int:
    var n = 0
    while True:
        var p = cache_dir + String("/sample_") + String(n) + String(".safetensors")
        if not _file_exists(p):
            break
        n += 1
    return n


# Load one cache sample. Reads via the mmap'd SafeTensors handle (origin-bound
# spans), reinterprets the raw bytes on host, patchifies the latent through the
# proven op, and returns clean x0 [S*IN_CH] + text [TXT*TEXT_DIM] + std/mean/txt_len.
def _load_cache_sample(
    cache_dir: String, idx: Int, ctx: DeviceContext,
    mut lat_std: Float32, mut lat_mean: Float32, mut txt_len: Int,
    i2v: Bool = False, read_clip: Bool = False,
) raises -> List[List[Float32]]:
    var path = cache_dir + String("/sample_") + String(idx) + String(".safetensors")
    var st = SafeTensors.open(path)

    # ── latent [16,1,32,32] f32 ────────────────────────────────────────────────
    var lat_info = st.tensor_info(String("latent"))
    if lat_info.dtype != STDtype.F32:
        raise Error(String("cache latent dtype != F32 in ") + path)
    var nlat = 1
    for i in range(len(lat_info.shape)):
        nlat *= lat_info.shape[i]
    if nlat != 16 * 1 * 32 * 32:
        raise Error(String("cache latent numel != 16*1*32*32 in ") + path)
    var lat_bytes = st.tensor_bytes(String("latent"))
    var fp = lat_bytes.unsafe_ptr().bitcast[Float32]()
    var latvals = List[Float32]()
    var s = Float64(0.0)
    var ss = Float64(0.0)
    for i in range(nlat):
        var v = fp[i]
        latvals.append(v)
        s += Float64(v)
        ss += Float64(v) * Float64(v)
    var mean = s / Float64(nlat)
    var varpop = ss / Float64(nlat) - mean * mean
    if varpop < 0.0:
        varpop = 0.0
    lat_mean = Float32(mean)
    lat_std = Float32(sqrt(varpop))

    # ── patchify [16,1,32,32] -> [S=256, IN_CH=64] via the PROVEN op ───────────
    var lat_shape = List[Int]()
    lat_shape.append(16)
    lat_shape.append(1)
    lat_shape.append(32)
    lat_shape.append(32)
    var lat_tensor = Tensor.from_host(latvals^, lat_shape^, STDtype.F32, ctx)
    var patched = patchify3d(lat_tensor, 1, 2, 2, ctx)   # [S, IN_CH]
    var x0 = patched.to_host(ctx)                         # flat [S*IN_CH]
    if len(x0) != S * IN_CH:
        raise Error(String("patchified x0 len != S*IN_CH in ") + path)

    # ── text [512,4096] bf16 -> host f32 ───────────────────────────────────────
    var txt_info = st.tensor_info(String("text"))
    if txt_info.dtype != STDtype.BF16:
        raise Error(String("cache text dtype != BF16 in ") + path)
    var ntxt = 1
    for i in range(len(txt_info.shape)):
        ntxt *= txt_info.shape[i]
    if ntxt != TXT * TEXT_DIM:
        raise Error(String("cache text numel != TXT*TEXT_DIM in ") + path)
    var txt_bytes = st.tensor_bytes(String("text"))
    var bp = txt_bytes.unsafe_ptr().bitcast[BFloat16]()
    var text = List[Float32]()
    for i in range(ntxt):
        text.append(bp[i].cast[DType.float32]())

    # ── text_len i32 [1] ───────────────────────────────────────────────────────
    var tl_bytes = st.tensor_bytes(String("text_len"))
    var ip = tl_bytes.unsafe_ptr().bitcast[Int32]()
    txt_len = Int(ip[0])

    # ── I2V conditioning cond_y [20,1,32,32] f32 -> patchify (1,2,2) -> [S,80] ──
    # Only read when i2v (T2V cache files have no cond_y key). Patchified through
    # the SAME proven op as the latent (channel-slowest packing), giving the clean
    # 80-dim per-token conditioning the caller concatenates after the 64-dim noisy
    # patch to form the 144-dim model input. cond_y is CLEAN (never noised).
    var out = List[List[Float32]]()
    out.append(x0^)
    out.append(text^)
    if i2v:
        var cy_info = st.tensor_info(String("cond_y"))
        if cy_info.dtype != STDtype.F32:
            raise Error(String("cache cond_y dtype != F32 in ") + path)
        var ncy = 1
        for i in range(len(cy_info.shape)):
            ncy *= cy_info.shape[i]
        if ncy != COND_CH * 1 * 32 * 32:
            raise Error(String("cache cond_y numel != 20*1*32*32 in ") + path)
        var cy_bytes = st.tensor_bytes(String("cond_y"))
        var cyp = cy_bytes.unsafe_ptr().bitcast[Float32]()
        var cyvals = List[Float32]()
        var cs = Float64(0.0)
        var css = Float64(0.0)
        for i in range(ncy):
            var v = cyp[i]
            cyvals.append(v)
            cs += Float64(v)
            css += Float64(v) * Float64(v)
        var cy_mean = cs / Float64(ncy)
        var cy_var = css / Float64(ncy) - cy_mean * cy_mean
        if cy_var < 0.0:
            cy_var = 0.0
        # ROUND-TRIP gate (must match the torch-printed cond_y std/mean).
        print("  [cache] cond_y std=", Float32(sqrt(cy_var)),
              " mean=", Float32(cy_mean), " (raw 20ch; must match torch)")
        var cy_shape = List[Int]()
        cy_shape.append(COND_CH)
        cy_shape.append(1)
        cy_shape.append(32)
        cy_shape.append(32)
        var cy_tensor = Tensor.from_host(cyvals^, cy_shape^, STDtype.F32, ctx)
        var cy_patched = patchify3d(cy_tensor, 1, 2, 2, ctx)   # [S, COND_PACK=80]
        var cy_host = cy_patched.to_host(ctx)                  # flat [S*80]
        if len(cy_host) != S * COND_PACK:
            raise Error(String("patchified cond_y len != S*COND_PACK in ") + path)
        out.append(cy_host^)
    # ── I2V-2.1 CLIP image features clip [257,1280] (--i2v21 cache) ─────────────
    # The penultimate xlm-roberta-vit-h-14 CLIP visual tokens. Fed to the frozen
    # MLPProj (img_emb) -> context_img [257,dim]. Read as F32 host (dtype-agnostic;
    # accepts F32/F16/BF16 storage). Appended LAST so the i2v21 caller pops it first.
    if read_clip:
        var clip_h = _st_host_f32(st, String("clip"), ctx)
        if len(clip_h) != CLIP_TOKENS * CLIP_DIM:
            raise Error(String("cache clip numel != 257*1280 in ") + path)
        out.append(clip_h^)
    return out^


# ══════════════════════════════════════════════════════════════════════════════
# WAN 2.1 T2V — SINGLE-EXPERT LoRA TRAINING (comptime-monomorphized on H).
#
# Reuses the CERTIFIED Wan2.2 engine (wan22_stack_lora_forward/backward_offload)
# UNCHANGED. Wan2.1 = single DiT (no dual high/low-noise boundary), same block
# (WanCrossAttention, no CLIP), same flow-match (target=noise-x0), same 10-target
# LoRA set, same Wan2.1-VAE 16-ch latent + umt5 text → the existing 40_woman cache
# is directly reusable. Patch geometry is IDENTICAL to A14B (latent [16,1,32,32]
# → patchify(1,2,2) → S=256, IN_CH=64); ONLY dim/heads/blocks differ.
#
# H is the comptime head count — the ONLY compile-time delta between the 1.3B
# (H=12) and 14B (H=40) monomorphizations. Dh=128, S=256, TXT=512 are shared
# module comptime constants; dim/ffn/num_blocks flow through as RUNTIME args (the
# weights loader + block plan + LoRA builder are all dim/block-generic). The
# top-level dispatch in main() (env WAN21_MODEL) instantiates the matching H.
# ══════════════════════════════════════════════════════════════════════════════
def _run_wan21_train[H: Int](
    cfg: TrainConfig, ckpt: String, dim: Int, ffn: Int, num_blocks: Int,
    variant_name: String, ctx: DeviceContext,
) raises:
    var rank = cfg.lora_rank if cfg.lora_rank > 0 else 16
    var alpha = cfg.lora_alpha if cfg.lora_alpha > Float32(0.0) else Float32(16.0)
    var lr = cfg.lr if cfg.lr > Float32(0.0) else DEFAULT_LR
    var max_grad_norm = cfg.max_grad_norm if cfg.max_grad_norm > Float32(0.0) \
        else DEFAULT_MAX_GRAD_NORM
    var steps = cfg.max_steps if cfg.max_steps > 0 else 1
    var seed = cfg.seed
    var beta1 = cfg.beta1 if cfg.beta1 > Float32(0.0) else Float32(0.9)
    var beta2 = cfg.beta2 if cfg.beta2 > Float32(0.0) else Float32(0.999)
    var opt_eps = cfg.eps if cfg.eps > Float32(0.0) else Float32(1.0e-8)
    var weight_decay = cfg.weight_decay if cfg.weight_decay >= Float32(0.0) \
        else Float32(0.01)
    var discrete_flow_shift = cfg.timestep_shift if cfg.timestep_shift > Float32(0.0) \
        else DEFAULT_DISCRETE_FLOW_SHIFT
    var shift_mode = _env_is_set(String("WAN22_SHIFT_TIMESTEPS"))

    var out_path = cfg.output_model_destination
    if out_path.byte_length() == 0:
        out_path = String("/home/alex/mojodiffusion/output/wan21_lora/") \
            + variant_name + String("_lora.safetensors")

    print("==== Wan2.1-T2V", variant_name, "(SINGLE-EXPERT) LoRA trainer — torchref recipe ====")
    print("VARIANT:", variant_name, " (Wan2.1 single DiT; no dual high/low-noise expert)")
    print("arch: dim=", dim, " blocks=", num_blocks, " heads=", H, " head_dim=", Dh,
          " ffn=", ffn, " in_ch=", IN_CH, " out_ch=", OUT_CH)
    print("geometry: latent[16,1,32,32] -> patch-grid (", FG, "x", HG, "x", WG,
          ") S=", S, " packed IN_CH=", IN_CH, " TXT=", TXT)
    print("lora: rank=", rank, " alpha=", alpha, " (10/block: 8 attn + ffn.0 + ffn.2)")
    print("optim: lr=", lr, " max_grad_norm=", max_grad_norm, " betas=(", beta1, ",", beta2,
          ") eps=", opt_eps, " wd=", weight_decay)
    if shift_mode:
        print("timestep: SHIFT mode (sigmoid_scale=", DEFAULT_SIGMOID_SCALE,
              " discrete_flow_shift=", discrete_flow_shift, ")")
    else:
        print("timestep: SIGMA mode (uniform t) [set WAN22_SHIFT_TIMESTEPS=1 for shift]")
    print("steps:", steps, " seed:", seed)
    print("ckpt:", ckpt)

    # ── checkpoint-absent guard (weights DOWNLOADING) ──────────────────────────
    if not _file_exists(ckpt):
        print("")
        print("[wan21] weights not present yet, skipping real step:")
        print("        ", ckpt)
        print("        (checkpoint downloading — config/geometry/flow-match + the",
              "H=", H, "monomorphization type-checked; re-run once safetensors lands.)")
        print("RESULT: wan21", variant_name, "trainer wired OK (weights-absent path, exit 0)")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # REAL WEIGHTED STEP (single-expert; parity carried over from certified 2.2).
    # ══════════════════════════════════════════════════════════════════════════
    print("")
    print("[wan21] checkpoint present — building resident base + block-swap loader")
    var off_cfg = OffloadConfig.synchronous_single()
    var base_st = SafeTensors.open(ckpt)
    # Auto-detect the checkpoint key prefix: "" for bare Wan2.2 keys, or
    # "model.diffusion_model." for the ComfyUI-repackaged Wan2.1 files. The SAME
    # prefix is threaded into the resident base loader AND the block-swap plan so
    # both agree; 2.2 checkpoints detect "" and stay byte-identical.
    var ckpt_prefix = detect_wan22_prefix(base_st)
    print("[wan21] detected checkpoint key prefix: '", ckpt_prefix, "'", sep="")
    var base = load_wan22_stack_base(base_st, ctx, ckpt_prefix)
    var plan = build_wan22_block_plan(num_blocks, ckpt_prefix)
    var loader = TurboPlannedLoader.open(ckpt, plan^, off_cfg, ctx, False)

    var lora = build_wan22_lora_set(num_blocks, dim, ffn, rank, alpha)
    print("[lora] adapters:", wan22_total_adapters(lora))

    var rope = _build_rope_proven(ctx)
    var cos = rope[0].copy()
    var sin = rope[1].copy()

    # P3: config-first data cache (cfg.dataset_cache_dir ← JSON cache_dir /
    # dataset_cache_dir); env WAN22_DATA_CACHE OVERRIDES when set (back-compat).
    var cache_dir = _env_str(String("WAN22_DATA_CACHE"))
    if cache_dir.byte_length() == 0:
        cache_dir = cfg.dataset_cache_dir
    var n_cache = 0
    var use_cache = False
    if cache_dir.byte_length() > 0:
        n_cache = _count_cache_samples(cache_dir)
        if n_cache == 0:
            # Fail loud: an EXPLICIT cache dir (env or config) with no samples
            # must never silently train on synthetic data (P3 discipline).
            raise Error(
                String("[data] no sample_*.safetensors at cache dir: ")
                + cache_dir
                + String(" (env WAN22_DATA_CACHE > config cache_dir; fail-loud)")
            )
        use_cache = True
    if use_cache:
        print("[data] REAL cache:", cache_dir, " samples=", n_cache,
              " (round-robin; REPLACES synthetic x0/text)")
    else:
        print("[data] SYNTHETIC x0/text (set cache_dir in the config or",
              "env WAN22_DATA_CACHE=<dir> for real data)")

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var have_first = False
    var train_start = perf_counter_ns()

    print("")
    print("step  t        MSE            grad_norm     sec")
    for step in range(steps):
        var t0 = perf_counter_ns()
        var t = _sample_t(shift_mode, step, seed,
                          DEFAULT_SIGMOID_SCALE, discrete_flow_shift)

        var x0: List[Float32]
        var txt_tokens: List[Float32]
        if use_cache:
            var samp_idx = step % n_cache
            var c_std = Float32(0.0)
            var c_mean = Float32(0.0)
            var c_txtlen = 0
            var pair = _load_cache_sample(
                cache_dir, samp_idx, ctx, c_std, c_mean, c_txtlen, False
            )
            print("  [cache] sample", samp_idx, " latent std=", c_std,
                  " mean=", c_mean, " txt_len=", c_txtlen, " (must match torch)")
            txt_tokens = pair.pop()   # [text]
            x0 = pair.pop()           # [x0]
        else:
            x0 = _synth_x0(S * IN_CH, seed + UInt64(step) * 101 + 1)
            txt_tokens = _synth_x0(TXT * TEXT_DIM, seed + UInt64(step) * 777 + 9)

        var noise = _noise(S * IN_CH, seed * UInt64(2000003) + UInt64(step) * 104729 + 3)
        var img_tokens = List[Float32]()
        var target = List[Float32]()
        for i in range(len(x0)):
            img_tokens.append((Float32(1.0) - t) * x0[i] + t * noise[i])
            target.append(noise[i] - x0[i])
        # fwd.out is the pre-unpatchify head layout `(pf,ph,pw,c)`, while x0/noise
        # are patch-embed layout `(c,pf,ph,pw)`.
        target = _wan_patch_input_to_head(target^, S, SAMPLE_LAT_C)

        var t_model = t * Float32(1000.0) + Float32(1.0)

        var fwd = wan22_stack_lora_forward_offload[H, Dh, S, TXT](
            img_tokens.copy(), txt_tokens.copy(), t_model,
            base, loader, lora, cos.copy(), sin.copy(),
            dim, ffn, IN_CH, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
        )

        var nout = len(fwd.out)
        var loss = Float32(0.0)
        var d_out = List[Float32]()
        var inv_n = Float32(2.0) / Float32(nout)
        for i in range(nout):
            var diff = fwd.out[i] - target[i]
            loss += diff * diff
            d_out.append(inv_n * diff)
        loss = loss / Float32(nout)
        if not have_first:
            first_loss = loss
            have_first = True
        last_loss = loss

        var grads = wan22_stack_lora_backward_offload[H, Dh, S, TXT](
            d_out, img_tokens.copy(), txt_tokens.copy(),
            base, loader, lora, cos.copy(), sin.copy(), fwd,
            dim, ffn, IN_CH, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
        )
        var gn = _clip(grads, max_grad_norm)
        wan22_lora_adamw_step(lora, grads, step + 1, lr, ctx,
                              beta1, beta2, opt_eps, weight_decay)
        var secs = Float64(perf_counter_ns() - t0) / 1.0e9
        print(step, "  ", t, "  ", loss, "  ", gn, "  ", secs)
        if grads.nonfinite_lora_grads != 0:
            print("  !! nonfinite lora grads =", grads.nonfinite_lora_grads)

    var npairs = save_wan22_lora(lora, out_path, ctx)
    var meta = List[Float32]()
    meta.append(Float32(steps))
    meta.append(Float32(Int(seed)))
    var state_path = out_path + String(".state")
    var nstate = save_wan22_lora_state(lora, state_path, ctx, meta^)
    print("")
    print("[save] wrote", npairs, "PEFT pairs ->", out_path)
    print("[save] wrote", nstate, "state tensors ->", state_path)
    print("trained steps=", steps, " first MSE=", first_loss, " last MSE=", last_loss,
          " total sec=", Float64(perf_counter_ns() - train_start) / 1.0e9)
    print("RESULT: wan21", variant_name, "single-expert LoRA trainer ran (real weighted",
          "step; block/stack/LoRA math == certified 2.2 engine).")


# ══════════════════════════════════════════════════════════════════════════════
# WAN 2.1 I2V-14B — DUAL CROSS-ATTENTION LoRA TRAINING (WAN21_I2V=1)
#
# The genuinely-new-compute variant: every WanAttentionBlock runs
# WanI2VCrossAttention (models/wan22/wan22_block.mojo wan22_i2v_block_lora_*),
# whose fwd+bwd is CERTIFIED against Torchref at cos>=0.999 by
# models/wan22/parity/wan22_i2v21_block_lora_parity.mojo. Single expert (boundary
# None), 36-ch input (IN_CH_I2V=144 packed), 12-target LoRA (10 + k_img + v_img).
#
# The 257 CLIP-image tokens are projected to `dim` by the FROZEN img_emb/MLPProj
# (mlpproj_forward below) and become the block's context_img; the umt5 tokens are
# context_txt. Assembling those two into every block's cross-attn IS the I2V
# context "prepend" (torchref concats then splits at 257; our block takes the two
# halves directly, so no separate concat is needed).
# ══════════════════════════════════════════════════════════════════════════════

# Frozen CLIP MLPProj (img_emb.proj): LayerNorm(1280) -> Linear(1280,1280) ->
# GELU -> Linear(1280,dim) -> LayerNorm(dim). Maps CLIP [257,1280] -> [257,dim].
# NOTE: torch nn.GELU() here is the EXACT (erf) GELU (torchref model.py:518,
# approximate="none") — NOT the tanh approximation the block FFN uses. mlpproj_forward
# below uses ops.activations.gelu_exact (erf) to match. img_emb.proj keys:
#   proj.0 = LayerNorm(1280)   proj.1 = Linear(1280,1280)   proj.2 = GELU (no params)
#   proj.3 = Linear(1280,dim)  proj.4 = LayerNorm(dim)
struct MLPProjWeights(Copyable, Movable):
    var ln1_w: List[Float32]
    var ln1_b: List[Float32]
    var fc1_w: List[Float32]
    var fc1_b: List[Float32]
    var fc2_w: List[Float32]
    var fc2_b: List[Float32]
    var ln2_w: List[Float32]
    var ln2_b: List[Float32]

    def __init__(
        out self,
        var ln1_w: List[Float32], var ln1_b: List[Float32],
        var fc1_w: List[Float32], var fc1_b: List[Float32],
        var fc2_w: List[Float32], var fc2_b: List[Float32],
        var ln2_w: List[Float32], var ln2_b: List[Float32],
    ):
        self.ln1_w = ln1_w^
        self.ln1_b = ln1_b^
        self.fc1_w = fc1_w^
        self.fc1_b = fc1_b^
        self.fc2_w = fc2_w^
        self.fc2_b = fc2_b^
        self.ln2_w = ln2_w^
        self.ln2_b = ln2_b^


def _zl(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


# clip_h [CLIP_TOKENS, CLIP_DIM] -> context_img [CLIP_TOKENS, dim] (frozen).
def mlpproj_forward(
    clip_h: List[Float32], w: MLPProjWeights, dim: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    var clip = Tensor.from_host(clip_h.copy(), [CLIP_TOKENS, CLIP_DIM], STDtype.BF16, ctx)
    var ln1w = Tensor.from_host(w.ln1_w.copy(), [CLIP_DIM], STDtype.BF16, ctx)
    var ln1b = Tensor.from_host(w.ln1_b.copy(), [CLIP_DIM], STDtype.BF16, ctx)
    var ln1 = layer_norm(clip, ln1w, ln1b, EPS, ctx)
    var fc1w = Tensor.from_host(w.fc1_w.copy(), [CLIP_DIM, CLIP_DIM], STDtype.BF16, ctx)
    var fc1b = Tensor.from_host(w.fc1_b.copy(), [CLIP_DIM], STDtype.BF16, ctx)
    var h = linear(ln1, fc1w, Optional[Tensor](fc1b^), ctx)
    # EXACT (erf) GELU — Torchref MLPProj is torch.nn.GELU() (approximate="none",
    # model.py:518), NOT the tanh-approx the block FFN uses. gelu_exact matches
    # torch's erf form (ops/activations.mojo:440).
    var g = gelu_exact(h, ctx)
    var fc2w = Tensor.from_host(w.fc2_w.copy(), [dim, CLIP_DIM], STDtype.BF16, ctx)
    var fc2b = Tensor.from_host(w.fc2_b.copy(), [dim], STDtype.BF16, ctx)
    var o = linear(g, fc2w, Optional[Tensor](fc2b^), ctx)
    var ln2w = Tensor.from_host(w.ln2_w.copy(), [dim], STDtype.BF16, ctx)
    var ln2b = Tensor.from_host(w.ln2_b.copy(), [dim], STDtype.BF16, ctx)
    var out = layer_norm(o, ln2w, ln2b, EPS, ctx)
    return out.to_host(ctx)


# ── Load one checkpoint tensor to a host F32 list (dtype-preserving read) ──────
def _st_host_f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return cast_tensor(t, STDtype.F32, ctx).to_host(ctx)


# ── Load the RESIDENT frozen CLIP MLPProj (img_emb.proj.*) ─────────────────────
# Sequential(LayerNorm(1280), Linear(1280,1280), GELU, Linear(1280,dim),
# LayerNorm(dim)) — torchref model.py:510-521. Keys img_emb.proj.{0,1,3,4}.{weight,
# bias} (index 2 = GELU, no params). `prefix` honors detect_wan22_prefix ("" for
# bare Wan2.x keys, "model.diffusion_model." for ComfyUI-repackaged files).
# NOTE: exact key layout VERIFIED against torchref source; re-confirm the on-disk
# header once the 14B I2V safetensors lands (weights-absent compile-check here).
def load_wan21_i2v_mlpproj(
    st: SafeTensors, ctx: DeviceContext, prefix: String = String(""),
) raises -> MLPProjWeights:
    var p = prefix + String("img_emb.proj.")
    return MLPProjWeights(
        _st_host_f32(st, p + String("0.weight"), ctx),
        _st_host_f32(st, p + String("0.bias"), ctx),
        _st_host_f32(st, p + String("1.weight"), ctx),
        _st_host_f32(st, p + String("1.bias"), ctx),
        _st_host_f32(st, p + String("3.weight"), ctx),
        _st_host_f32(st, p + String("3.bias"), ctx),
        _st_host_f32(st, p + String("4.weight"), ctx),
        _st_host_f32(st, p + String("4.bias"), ctx),
    )


def _mk_adapter(rank: Int, in_f: Int, out_f: Int, scale: Float32, seed: UInt64) -> Optional[LoraAdapter]:
    return Optional[LoraAdapter](LoraAdapter(
        _synth_x0(rank * in_f, seed), _synth_x0(out_f * rank, seed + 1),
        rank, in_f, out_f, scale,
        _zl(rank * in_f), _zl(rank * in_f), _zl(out_f * rank), _zl(out_f * rank),
    ))


# OPTIONAL exercise of the certified i2v block + MLPProj at REAL dim/ffn, tiny
# sequence (S=16, IMG=257) with SYNTHETIC weights — proves the new compute is
# wired into the trainer binary and runs. Gated behind WAN21_I2V_BLOCKCHECK=1 so
# the default weights-absent run stays instant (the block itself is already
# certified numerically by the parity gate). in=out=dim for the 12 attn+img
# adapters; ffn0 dim->ffn, ffn2 ffn->dim.
def _wan21_i2v_block_selfcheck(dim: Int, ffn: Int, ctx: DeviceContext) raises:
    comptime SH = 40
    comptime SDh = 128
    comptime SS = 16
    comptime STXT = 8
    comptime SIMG = 257
    var rank = 8
    var scale = Float32(1.0)
    print("[wan21-i2v] BLOCKCHECK: exercising certified dual-cross-attn block +",
          "MLPProj at dim=", dim, " ffn=", ffn, " (S=", SS, " IMG=", SIMG, ")")

    var base_w = WanBlockWeights(
        _synth_x0(dim * dim, 11), _synth_x0(dim * dim, 12), _synth_x0(dim * dim, 13), _synth_x0(dim * dim, 14),
        _zl(dim), _zl(dim), _zl(dim), _zl(dim),
        _synth_x0(dim, 15), _synth_x0(dim, 16),
        _synth_x0(dim * dim, 17), _synth_x0(dim * dim, 18), _synth_x0(dim * dim, 19), _synth_x0(dim * dim, 20),
        _zl(dim), _zl(dim), _zl(dim), _zl(dim),
        _synth_x0(dim, 21), _synth_x0(dim, 22),
        _synth_x0(dim, 23), _zl(dim),
        _synth_x0(ffn * dim, 24), _zl(ffn), _synth_x0(dim * ffn, 25), _zl(dim),
        dim, ffn, SDh, ctx,
    )
    var w = WanI2VBlockWeights(
        base_w^,
        _synth_x0(dim * dim, 26), _zl(dim), _synth_x0(dim * dim, 27), _zl(dim),
        _synth_x0(dim, 28),
        dim, ctx,
    )
    var base_lora = WanBlockLora(
        _mk_adapter(rank, dim, dim, scale, 100), _mk_adapter(rank, dim, dim, scale, 102),
        _mk_adapter(rank, dim, dim, scale, 104), _mk_adapter(rank, dim, dim, scale, 106),
        _mk_adapter(rank, dim, dim, scale, 108), _mk_adapter(rank, dim, dim, scale, 110),
        _mk_adapter(rank, dim, dim, scale, 112), _mk_adapter(rank, dim, dim, scale, 114),
        _mk_adapter(rank, dim, ffn, scale, 116), _mk_adapter(rank, ffn, dim, scale, 118),
    )
    var lora = WanI2VBlockLora(
        base_lora^,
        _mk_adapter(rank, dim, dim, scale, 120), _mk_adapter(rank, dim, dim, scale, 122),
    )
    var mv = WanModVecs(
        _synth_x0(SS * dim, 30), _synth_x0(SS * dim, 31), _synth_x0(SS * dim, 32),
        _synth_x0(SS * dim, 33), _synth_x0(SS * dim, 34), _synth_x0(SS * dim, 35),
    )
    var mlpw = MLPProjWeights(
        _synth_x0(CLIP_DIM, 40), _zl(CLIP_DIM),
        _synth_x0(CLIP_DIM * CLIP_DIM, 41), _zl(CLIP_DIM),
        _synth_x0(dim * CLIP_DIM, 42), _zl(dim),
        _synth_x0(dim, 43), _zl(dim),
    )
    var clip = _synth_x0(CLIP_TOKENS * CLIP_DIM, 44)
    var context_img = mlpproj_forward(clip, mlpw, dim, ctx)   # [257,dim]
    var context_txt = _synth_x0(STXT * dim, 45)
    var x = _synth_x0(SS * dim, 46)
    var d_out = _synth_x0(SS * dim, 47)
    var cos = Tensor.from_host(_synth_x0(SS * (SDh // 2), 48), [SS, SDh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_synth_x0(SS * (SDh // 2), 49), [SS, SDh // 2], STDtype.F32, ctx)

    var fwd = wan22_i2v_block_lora_forward[SH, SDh, SS, STXT, SIMG](
        x.copy(), context_txt.copy(), context_img.copy(), mv, w, lora, cos, sin, dim, ffn, EPS, ctx,
    )
    var g = wan22_i2v_block_lora_backward[SH, SDh, SS, STXT, SIMG](
        d_out.copy(), mv, w, lora, fwd.saved, cos, sin, dim, ffn, EPS, ctx,
    )
    print("[wan21-i2v] BLOCKCHECK OK: x_out n=", len(fwd.x_out),
          " d_x n=", len(g.base.base.d_x), " d_context_img n=", len(g.d_context_img),
          " img_k_dA n=", len(g.img_k_da), " img_v_dB n=", len(g.img_v_db))


def _run_wan21_i2v_train[H: Int](
    cfg: TrainConfig, ckpt: String, dim: Int, ffn: Int, num_blocks: Int,
    ctx: DeviceContext,
) raises:
    var rank = cfg.lora_rank if cfg.lora_rank > 0 else 16
    var alpha = cfg.lora_alpha if cfg.lora_alpha > Float32(0.0) else Float32(16.0)
    var lr = cfg.lr if cfg.lr > Float32(0.0) else DEFAULT_LR
    var steps = cfg.max_steps if cfg.max_steps > 0 else 1

    print("==== Wan2.1 I2V-14B (DUAL cross-attn) LoRA trainer — torchref recipe ====")
    print("VARIANT: i2v_14b  (Wan2.1 single DiT; boundary=None; NO high/low-noise expert)")
    print("arch: dim=", dim, " blocks=", num_blocks, " heads=", H, " head_dim=", Dh,
          " ffn=", ffn, " in_dim=36 packed IN_CH_I2V=", IN_CH_I2V, " out_ch=", OUT_CH)
    print("cross-attn: WanI2VCrossAttention — CLIP", CLIP_TOKENS, "img tokens (",
          CLIP_DIM, "-> ", dim, " via frozen MLPProj) PREPENDED to umt5 text context")
    print("lora: rank=", rank, " alpha=", alpha,
          " (12/block: 8 attn + ffn.0 + ffn.2 + cross_attn.k_img + cross_attn.v_img)")
    print("optim: lr=", lr, "  steps:", steps)
    print("ckpt:", ckpt)
    print("[wan21-i2v] block fwd/bwd CERTIFIED vs Torchref (cos>=0.999):",
          "models/wan22/parity/wan22_i2v21_block_lora_parity.mojo")

    # ── checkpoint-absent guard (14B I2V weights not downloaded) ───────────────
    if not _file_exists(ckpt):
        print("")
        print("[wan21-i2v] weights not present yet, skipping real step:")
        print("        ", ckpt)
        print("        (block + 12-target LoRA + MLPProj context assembler +",
              "H=", H, "monomorphization type-checked; re-run once safetensors lands.)")
        if _env_is_set(String("WAN21_I2V_BLOCKCHECK")):
            _wan21_i2v_block_selfcheck(dim, ffn, ctx)
        print("RESULT: wan21 i2v_14b trainer wired OK (weights-absent path, exit 0)")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # REAL WEIGHTED STEP — full 40-block DUAL cross-attn I2V stack fwd/bwd.
    # ══════════════════════════════════════════════════════════════════════════
    comptime IMG = CLIP_TOKENS   # 257 CLIP image tokens (dual-cross-attn kv=IMG)
    var seed = cfg.seed
    var beta1 = cfg.beta1 if cfg.beta1 > Float32(0.0) else Float32(0.9)
    var beta2 = cfg.beta2 if cfg.beta2 > Float32(0.0) else Float32(0.999)
    var opt_eps = cfg.eps if cfg.eps > Float32(0.0) else Float32(1.0e-8)
    var weight_decay = cfg.weight_decay if cfg.weight_decay >= Float32(0.0) \
        else Float32(0.01)
    var max_grad_norm = cfg.max_grad_norm if cfg.max_grad_norm > Float32(0.0) \
        else DEFAULT_MAX_GRAD_NORM
    var discrete_flow_shift = cfg.timestep_shift if cfg.timestep_shift > Float32(0.0) \
        else DEFAULT_DISCRETE_FLOW_SHIFT
    var shift_mode = _env_is_set(String("WAN22_SHIFT_TIMESTEPS"))
    var out_path = cfg.output_model_destination
    if out_path.byte_length() == 0:
        out_path = String("/home/alex/mojodiffusion/output/wan21_lora/i2v_14b_lora.safetensors")

    print("")
    print("[wan21-i2v] checkpoint present — building resident base + MLPProj + block-swap loader")
    var off_cfg = OffloadConfig.synchronous_single()
    var base_st = SafeTensors.open(ckpt)
    var ckpt_prefix = detect_wan22_prefix(base_st)
    print("[wan21-i2v] detected checkpoint key prefix: '", ckpt_prefix, "'", sep="")
    var base = load_wan22_stack_base(base_st, ctx, ckpt_prefix)      # patch-embed autodetects 144-ch
    var mlpw = load_wan21_i2v_mlpproj(base_st, ctx, ckpt_prefix)     # resident frozen CLIP MLPProj
    var plan = build_wan21_i2v_block_plan(num_blocks, ckpt_prefix)   # 32 tensors/block hint
    var loader = TurboPlannedLoader.open(ckpt, plan^, off_cfg, ctx, False)

    var lora = build_wan22_i2v_lora_set(num_blocks, dim, ffn, rank, alpha)
    print("[lora] i2v adapters:", wan22_i2v_total_adapters(lora), " (12/block)")

    var rope = _build_rope_proven(ctx)
    var cos = rope[0].copy()
    var sin = rope[1].copy()

    # P3: config-first data cache; env WAN22_DATA_CACHE overrides when set.
    var cache_dir = _env_str(String("WAN22_DATA_CACHE"))
    if cache_dir.byte_length() == 0:
        cache_dir = cfg.dataset_cache_dir
    var n_cache = 0
    var use_cache = False
    if cache_dir.byte_length() > 0:
        n_cache = _count_cache_samples(cache_dir)
        if n_cache == 0:
            raise Error(
                String("[data] no sample_*.safetensors at cache dir: ")
                + cache_dir
                + String(" (env WAN22_DATA_CACHE > config cache_dir; fail-loud)")
            )
        use_cache = True
    if use_cache:
        print("[data] REAL --i2v21 cache:", cache_dir, " samples=", n_cache,
              " (latent16 + cond_y20 + clip[257,1280])")
    else:
        print("[data] SYNTHETIC x0/cond_y/text/clip (set config cache_dir or",
              "env WAN22_DATA_CACHE=<dir> for real data)")

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var have_first = False
    var train_start = perf_counter_ns()

    print("")
    print("step  t        MSE            grad_norm     sec")
    for step in range(steps):
        var t0 = perf_counter_ns()
        var t = _sample_t(shift_mode, step, seed,
                          DEFAULT_SIGMOID_SCALE, discrete_flow_shift)

        var x0: List[Float32]        # [S*IN_CH]   noisy-latent patch (16ch->64)
        var txt_tokens: List[Float32]  # [TXT*TEXT_DIM] raw umt5
        var cond_y: List[Float32]    # [S*COND_PACK] clean conditioning (20ch->80)
        var clip: List[Float32]      # [257*1280] CLIP visual tokens
        if use_cache:
            var samp_idx = step % n_cache
            var c_std = Float32(0.0)
            var c_mean = Float32(0.0)
            var c_txtlen = 0
            var pair = _load_cache_sample(
                cache_dir, samp_idx, ctx, c_std, c_mean, c_txtlen, True, True
            )
            print("  [cache] sample", samp_idx, " latent std=", c_std,
                  " mean=", c_mean, " txt_len=", c_txtlen)
            clip = pair.pop()       # [clip]
            cond_y = pair.pop()     # [cond_y]
            txt_tokens = pair.pop() # [text]
            x0 = pair.pop()         # [x0]
        else:
            x0 = _synth_x0(S * IN_CH, seed + UInt64(step) * 101 + 1)
            txt_tokens = _synth_x0(TXT * TEXT_DIM, seed + UInt64(step) * 777 + 9)
            cond_y = _synth_x0(S * COND_PACK, seed + UInt64(step) * 313 + 2)
            clip = _synth_x0(CLIP_TOKENS * CLIP_DIM, seed + UInt64(step) * 971 + 7)

        var noise = _noise(S * IN_CH, seed * UInt64(2000003) + UInt64(step) * 104729 + 3)
        var target = List[Float32]()
        for i in range(len(x0)):
            target.append(noise[i] - x0[i])
        # The I2V head still predicts only the 16-channel latent velocity.
        target = _wan_patch_input_to_head(target^, S, SAMPLE_LAT_C)
        # 36-ch model input, packed per-token: noisy(16ch->64) ++ cond_y(20ch->80) = 144.
        var img_tokens = List[Float32]()
        for s in range(S):
            for c in range(IN_CH):
                var j = s * IN_CH + c
                img_tokens.append((Float32(1.0) - t) * x0[j] + t * noise[j])
            for c in range(COND_PACK):
                img_tokens.append(cond_y[s * COND_PACK + c])

        # frozen MLPProj: CLIP [257,1280] -> context_img [257,dim] (assembled once/step).
        var context_img = mlpproj_forward(clip, mlpw, dim, ctx)

        var t_model = t * Float32(1000.0) + Float32(1.0)

        var fwd = wan22_i2v_stack_lora_forward_offload[H, Dh, S, TXT, IMG](
            img_tokens.copy(), txt_tokens.copy(), context_img.copy(), t_model,
            base, loader, lora, cos.copy(), sin.copy(),
            dim, ffn, IN_CH_I2V, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
        )

        var nout = len(fwd.out)
        var loss = Float32(0.0)
        var d_out = List[Float32]()
        var inv_n = Float32(2.0) / Float32(nout)
        for i in range(nout):
            var diff = fwd.out[i] - target[i]
            loss += diff * diff
            d_out.append(inv_n * diff)
        loss = loss / Float32(nout)
        if not have_first:
            first_loss = loss
            have_first = True
        last_loss = loss

        var grads = wan22_i2v_stack_lora_backward_offload[H, Dh, S, TXT, IMG](
            d_out, img_tokens.copy(), txt_tokens.copy(),
            base, loader, lora, cos.copy(), sin.copy(), fwd,
            dim, ffn, IN_CH_I2V, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
        )
        var gn = _clip_i2v(grads, max_grad_norm)
        wan22_i2v_lora_adamw_step(lora, grads, step + 1, lr, ctx,
                                  beta1, beta2, opt_eps, weight_decay)
        var secs = Float64(perf_counter_ns() - t0) / 1.0e9
        print(step, "  ", t, "  ", loss, "  ", gn, "  ", secs)
        if grads.nonfinite_lora_grads != 0:
            print("  !! nonfinite lora grads =", grads.nonfinite_lora_grads)

    var npairs = save_wan22_i2v_lora(lora, out_path, ctx)
    var meta = List[Float32]()
    meta.append(Float32(steps))
    meta.append(Float32(Int(seed)))
    var state_path = out_path + String(".state")
    var nstate = save_wan22_i2v_lora_state(lora, state_path, ctx, meta^)
    print("")
    print("[save] wrote", npairs, "PEFT pairs ->", out_path)
    print("[save] wrote", nstate, "state tensors ->", state_path)
    print("trained steps=", steps, " first MSE=", first_loss, " last MSE=", last_loss,
          " total sec=", Float64(perf_counter_ns() - train_start) / 1.0e9)
    print("RESULT: wan21 i2v_14b DUAL cross-attn LoRA trainer ran (real weighted step;",
          "block math == certified; 12/block LoRA saved diffusion_model.-prefixed).")


def main() raises:
    # ── config (read FIRST — P3: config keys drive arm dispatch; the WAN21_*/
    #    WAN22_* env vars stay as overrides for the legacy smoke scripts) ──────
    var args = argv()
    var config_path = String(DEFAULT_CONFIG)
    if len(args) > 1:
        config_path = String(args[1])
    var cfg = read_model_config(config_path)

    # Pure-host correctness gate and fail-loud sampler preflight happen before
    # DeviceContext/model allocation. The trainer is fixed at S=256, while the
    # standing 1024px preview requires S=4096 and the production Torchref recipe is
    # the process-separated UniPC sampler, not this legacy in-process Euler loop.
    _wan_patch_layout_self_gate()
    if cfg.sample_every > 0:
        raise Error(
            String("WAN sample_every>0 is unsupported in-process: training uses S=256, ")
            + String("the required 1024px sampler uses S=4096, and validation must use ")
            + String("the Torchref-aligned process-separated pipeline/wan22_lora_sample. ")
            + String("Set sample_every=0 and sample saved checkpoints externally.")
        )

    var ctx = DeviceContext()

    # Wan is a VIDEO model — LoRA/LoCon ONLY. LoKr/LoHa/DoRA/OFT are NOT wired
    # here (this loop trains plain attn LoRA); reject them FAIL-LOUD rather than
    # silently training LoRA under a Lycoris config (Alex 2026-07-23: no Lycoris
    # on video models). Image models (krea2/zimage/klein/flux/qwenimage/sd35/
    # chroma/ernie/ideogram4) carry their own direct-Lycoris dispatch.
    require_lora_or_locon_linear(cfg, "wan22")

    # ── WAN 2.1 I2V-14B DISPATCH (env WAN21_I2V=1; env-only test arm) ─────────
    # The DUAL cross-attn (WanI2VCrossAttention) variant. Checked BEFORE the
    # wan_variant / Wan2.2 bodies so those paths stay untouched. Checkpoint via
    # WAN21_DIT override else DEFAULT_CKPT_WAN21_I2V (absent -> weights-absent
    # guard, exit 0).
    if _env_is_set(String("WAN21_I2V")):
        var i2v21_dit_env = _env_str(String("WAN21_DIT"))
        var i2v_ck = i2v21_dit_env if i2v21_dit_env.byte_length() > 0 \
            else String(DEFAULT_CKPT_WAN21_I2V)
        _run_wan21_i2v_train[WAN21_I2V_14B_HEADS](
            cfg, i2v_ck, WAN21_I2V_14B_DIM, WAN21_I2V_14B_FFN,
            WAN21_I2V_14B_BLOCKS, ctx,
        )
        return

    # ── WAN 2.1 T2V DISPATCH (config `wan_variant`; env WAN21_MODEL overrides) ─
    # Single-expert Wan2.1 path, comptime-monomorphized by head count. Selected
    # BEFORE the Wan2.2/I2V body so those paths stay untouched (back-compat).
    #   wan_variant/WAN21_MODEL=t2v_1.3b -> H=12, dim1536/ffn8960/30 blocks.
    #   wan_variant/WAN21_MODEL=t2v_14b  -> H=40, dim5120/ffn13824/40 blocks.
    # Checkpoint precedence: env WAN21_DIT > cfg.checkpoint (ONLY when the
    # config itself selected the variant — a legacy env-selected run against a
    # wan22 config must NOT pick up that config's wan2.2 checkpoint) > the
    # per-variant default.
    var wan21_model = _env_str(String("WAN21_MODEL"))
    if wan21_model.byte_length() == 0:
        wan21_model = cfg.wan_variant
    if wan21_model.byte_length() > 0:
        var dit_env = _env_str(String("WAN21_DIT"))
        var cfg_ck = String("")
        if cfg.wan_variant.byte_length() > 0:
            cfg_ck = cfg.checkpoint
        if wan21_model == String("t2v_1.3b"):
            var ck = dit_env if dit_env.byte_length() > 0 \
                else (cfg_ck if cfg_ck.byte_length() > 0
                      else String(DEFAULT_CKPT_WAN21_1P3B))
            _run_wan21_train[WAN21_1P3B_HEADS](
                cfg, ck, WAN21_1P3B_DIM, WAN21_1P3B_FFN, WAN21_1P3B_BLOCKS,
                String("t2v_1.3b"), ctx,
            )
        elif wan21_model == String("t2v_14b"):
            var ck = dit_env if dit_env.byte_length() > 0 \
                else (cfg_ck if cfg_ck.byte_length() > 0
                      else String(DEFAULT_CKPT_WAN21_14B))
            _run_wan21_train[WAN21_14B_HEADS](
                cfg, ck, WAN21_14B_DIM, WAN21_14B_FFN, WAN21_14B_BLOCKS,
                String("t2v_14b"), ctx,
            )
        else:
            raise Error(String("unknown WAN21_MODEL='") + wan21_model
                + String("' (expected t2v_1.3b | t2v_14b)"))
        return

    print("==== Wan2.2-A14B (low-noise) LoRA trainer — torchref recipe ====")
    print("     (T2V default; config i2v=true or WAN22_I2V=1 for the I2V-A14B 36-ch variant)")
    print("config:", config_path)

    # ── I2V MODE (config `i2v`; env WAN22_I2V overrides when set) ─────────────
    # I2V-A14B differs from T2V-A14B ONLY in the DiT input (36ch = noisy_latent 16
    # + y 20) and the dual-expert boundary (0.900 vs 0.875). The block/stack/LoRA/
    # dual-expert compute is IDENTICAL (WanCrossAttention; no CLIP, no k_img/v_img —
    # torchref model.py:379-380). So I2V is a DATA-PATH + INPUT-CHANNEL change here:
    # patchified input width IN_CH_I2V=144, boundary I2V_BOUNDARY, I2V checkpoint.
    var i2v_env = _env_str(String("WAN22_I2V"))
    var i2v_mode = (i2v_env == String("1")) if i2v_env.byte_length() > 0 \
        else cfg.wan_i2v
    var in_ch_eff = IN_CH_I2V if i2v_mode else IN_CH

    # Checkpoint resolution (P3 precedence: env WAN22_DIT > cfg.checkpoint >
    # default). I2V needs the 36-ch I2V DiT — a T2V smoke config's checkpoint is
    # the WRONG architecture (16-ch patch embed), so in I2V mode cfg.checkpoint
    # is honored ONLY if it names an i2v ckpt, else DEFAULT_CKPT_I2V.
    var ckpt = cfg.checkpoint
    var dit_env = _env_str(String("WAN22_DIT"))
    if i2v_mode:
        if dit_env.byte_length() > 0:
            ckpt = dit_env.copy()
        elif ckpt.byte_length() == 0 \
                or (ckpt.find(String("i2v")) < 0 and ckpt.find(String("I2V")) < 0):
            ckpt = String(DEFAULT_CKPT_I2V)
    else:
        if dit_env.byte_length() > 0:
            ckpt = dit_env.copy()
        elif ckpt.byte_length() == 0:
            ckpt = String(DEFAULT_CKPT)

    var rank = cfg.lora_rank if cfg.lora_rank > 0 else 16
    var alpha = cfg.lora_alpha if cfg.lora_alpha > Float32(0.0) else Float32(16.0)
    var lr = cfg.lr if cfg.lr > Float32(0.0) else DEFAULT_LR
    var max_grad_norm = cfg.max_grad_norm if cfg.max_grad_norm > Float32(0.0) \
        else DEFAULT_MAX_GRAD_NORM
    var steps = cfg.max_steps if cfg.max_steps > 0 else 1
    var seed = cfg.seed
    var beta1 = cfg.beta1 if cfg.beta1 > Float32(0.0) else Float32(0.9)
    var beta2 = cfg.beta2 if cfg.beta2 > Float32(0.0) else Float32(0.999)
    var opt_eps = cfg.eps if cfg.eps > Float32(0.0) else Float32(1.0e-8)
    var weight_decay = cfg.weight_decay if cfg.weight_decay >= Float32(0.0) \
        else Float32(0.01)
    var discrete_flow_shift = cfg.timestep_shift if cfg.timestep_shift > Float32(0.0) \
        else DEFAULT_DISCRETE_FLOW_SHIFT
    var shift_mode = _env_is_set(String("WAN22_SHIFT_TIMESTEPS"))

    # ── DUAL HIGH/LOW-NOISE EXPERT config (Torchref wan_train_network.py) ─────────
    # Wan2.2-A14B ships TWO DiTs: low-noise (`--dit`) + high-noise (`--dit_high_noise`),
    # split at `timestep_boundary` (T2V default 0.875). ONE shared LoRA network trains
    # across BOTH experts (swap_high_low_weights :584 swaps the BASE state_dict under
    # the same LoRA — NOT a LoRA-per-expert), so we keep the single Wan22LoraSet and
    # swap only the (base, loader) pair per step.
    # P3 precedence (env wins ONLY when set; config next; then the defaults):
    #   * boundary:  WAN22_TIMESTEP_BOUNDARY > cfg.timestep_boundary >
    #     per-mode default (0.875 = DUAL_EXPERT_BOUNDARY / 0.900 I2V).
    #   * high ckpt: WAN22_DIT_HIGH_NOISE > cfg.dit_high_noise > convention
    #     low_noise→high_noise.
    #   * dual on iff the high ckpt file exists AND the gate allows it:
    #     env WAN22_DUAL_EXPERT ("0"=off) > cfg.dual_expert (0=off; -1 auto/
    #     1 on) — back-compat: high-noise absent ⇒ single low-noise expert.
    var boundary = I2V_BOUNDARY if i2v_mode else DUAL_EXPERT_BOUNDARY
    if cfg.timestep_boundary > Float32(0.0):
        boundary = cfg.timestep_boundary
    var boundary_env = _env_str(String("WAN22_TIMESTEP_BOUNDARY"))
    if boundary_env.byte_length() > 0:
        boundary = Float32(atof(boundary_env))
    var high_ckpt = _env_str(String("WAN22_DIT_HIGH_NOISE"))
    if high_ckpt.byte_length() == 0:
        high_ckpt = cfg.dit_high_noise
    if high_ckpt.byte_length() == 0:
        high_ckpt = ckpt.replace(String("low_noise"), String("high_noise"))
    var dual_env = _env_str(String("WAN22_DUAL_EXPERT"))
    var dual_gate = (dual_env != String("0")) if dual_env.byte_length() > 0 \
        else (cfg.dual_expert != 0)
    var dual_expert = (
        _file_exists(high_ckpt)
        and high_ckpt != ckpt
        and dual_gate
    )
    # ── smoke timestep pins (GATE): force BOTH branches over ≥2 steps. Even steps
    #    → t_low_pin (<boundary ⇒ LOW expert), odd steps → t_high_pin (≥boundary ⇒
    #    HIGH expert). Enable with WAN22_ALTERNATE_T=1; values overridable via
    #    WAN22_T_LOW / WAN22_T_HIGH (defaults 0.30 / 0.95).
    var alt_t = _env_is_set(String("WAN22_ALTERNATE_T"))
    var t_low_pin = Float32(0.30)
    var t_high_pin = Float32(0.95)
    var tl_env = _env_str(String("WAN22_T_LOW"))
    if tl_env.byte_length() > 0:
        t_low_pin = Float32(atof(tl_env))
    var th_env = _env_str(String("WAN22_T_HIGH"))
    if th_env.byte_length() > 0:
        t_high_pin = Float32(atof(th_env))

    var out_path = cfg.output_model_destination
    if out_path.byte_length() == 0:
        out_path = String("/home/alex/mojodiffusion/output/wan22_lora/wan22_low_noise_lora.safetensors")

    # Periodic checkpoints: `save_every` steps -> "<out_path minus .safetensors>-<step>.safetensors"
    # (+ a .state sibling), so every checkpoint is both samplable and resumable.
    # 0 disables and only the end-of-run save happens (the previous behavior).
    var save_every = cfg.save_every
    var ckpt_stem = String(out_path.removesuffix(String(".safetensors")))
    # In-process sampling is rejected by the preflight before DeviceContext().
    # Keep the legacy variables/calls dormant until a process-separated request/
    # result manifest replaces them.
    var sample_every = cfg.sample_every
    var sample_conds_path = _env_str(String("WAN22_SAMPLE_CONDS"))
    var samples_dir = String(out_path.removesuffix(String(".safetensors"))) + String("_samples")
    if save_every > 0:
        print("[save] periodic checkpoints every", save_every, "steps ->",
              ckpt_stem + String("-<step>.safetensors"))

    if i2v_mode:
        print("VARIANT: I2V-A14B  (in=36ch: noisy_latent 16 + y 20 [mask 4 + img_latent 16])")
    else:
        print("VARIANT: T2V-A14B  (in=16ch latent)")
    print("arch: dim=", DIM, " blocks=", NUM_BLOCKS, " heads=", H, " head_dim=", Dh,
          " ffn=", FFN, " in_ch=", in_ch_eff, " out_ch=", OUT_CH)
    if i2v_mode:
        print("geometry: model input = concat(noisy_latent[16,1,32,32], cond_y[20,1,32,32])",
              "= [36,1,32,32] -> patchify(1,2,2) -> [S=", S, ", IN_CH=", in_ch_eff, "]",
              " (64 noisy + 80 cond) TXT=", TXT)
    else:
        print("geometry: latent[16,1,32,32] -> patch-grid (", FG, "x", HG, "x", WG,
              ") S=", S, " packed IN_CH=", in_ch_eff, " TXT=", TXT)
    print("lora: rank=", rank, " alpha=", alpha, " (SHARED across experts, 10/block: 8 attn + ffn.0 + ffn.2)")
    print("optim: lr=", lr, " max_grad_norm=", max_grad_norm, " betas=(", beta1, ",", beta2,
          ") eps=", opt_eps, " wd=", weight_decay)
    if shift_mode:
        print("timestep: SHIFT mode (sigmoid_scale=", DEFAULT_SIGMOID_SCALE,
              " discrete_flow_shift=", discrete_flow_shift, ")")
    else:
        print("timestep: SIGMA mode (uniform t) [set WAN22_SHIFT_TIMESTEPS=1 for shift]")
    print("steps:", steps, " seed:", seed)
    print("ckpt (low):", ckpt)
    if dual_expert:
        print("DUAL-EXPERT: ON  boundary=", boundary,
              " (t>=boundary ⇒ HIGH expert, else LOW)")
        print("ckpt (high):", high_ckpt)
    else:
        print("DUAL-EXPERT: OFF (single low-noise expert; high ckpt absent or disabled)")
    if alt_t:
        print("smoke timestep pins: ALTERNATE_T (even→", t_low_pin,
              " LOW, odd→", t_high_pin, " HIGH)")

    # ── checkpoint-absent guard (weights DOWNLOADING) ──────────────────────────
    if not _file_exists(ckpt):
        print("")
        print("[wan22] weights not present yet, skipping real step:")
        print("        ", ckpt)
        print("        (checkpoint is downloading — config/geometry/flow-match paths")
        print("         verified; re-run once the safetensors lands.)")
        print("RESULT: wan22 trainer wired OK (weights-absent path, exit 0)")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # REAL WEIGHTED STEP (UNVERIFIED until weights land + orchestrator L2P gate).
    # ══════════════════════════════════════════════════════════════════════════
    print("")
    print("[wan22] checkpoint present — building resident bases + block-swap loaders")

    # ── TWO RESIDENT BASES + TWO STREAMING LOADERS (dual-expert switch) ─────────
    # DESIGN (memory reasoning): the non-block "base" tensors (patch/text/time
    # embeddings, time_projection, head) are small (~0.5 GB/expert F16), so BOTH
    # experts' bases stay GPU-resident. The 40 transformer BLOCKS (~27 GB/expert)
    # are STREAMED one at a time by each expert's TurboPlannedLoader, so only the
    # loader's two rotating slabs (~0.7 GB block ×2) are ever resident per loader.
    # Two resident bases + two loaders' slabs ≈ 4–5 GB device — well under 16 GB.
    #
    # fill_block_store=False for BOTH loaders: the pinned host block store is
    # ~27 GB PER expert; two stores (~54 GB) exceed free host RAM (~49 GB) and
    # would OOM the host. With the store off, prefetch memcpys mmap→slot→device
    # (turbo_loader.mojo:487-495) — both checkpoints stream from the page cache.
    # This is the "two resident bases + stream blocks from the correct checkpoint
    # per step" option from the brief (chosen over swapping one loader's source:
    # two coexisting loaders each keep their own mmap + streams, so per-step
    # expert selection is a branch, not a re-open).
    var off_cfg = OffloadConfig.synchronous_single()

    var base_st = SafeTensors.open(_base_file_for(ckpt))
    var base = load_wan22_stack_base(base_st, ctx)
    var plan_low = build_wan22_block_plan(NUM_BLOCKS)
    var loader = TurboPlannedLoader.open(ckpt, plan_low^, off_cfg, ctx, False)

    # High-noise pair. When dual is OFF we still create the slot (Mojo needs both
    # branches' vars in scope) pointing at the LOW checkpoint — a redundant small
    # base + a streaming loader (blocks stream, so no extra 27 GB). `use_high` is
    # forced False in that case so the redundant loader is never stepped.
    var high_ckpt_eff = high_ckpt if dual_expert else ckpt
    var high_base_st = SafeTensors.open(_base_file_for(high_ckpt_eff))
    var high_base = load_wan22_stack_base(high_base_st, ctx)
    var plan_high = build_wan22_block_plan(NUM_BLOCKS)
    var loader_high = TurboPlannedLoader.open(high_ckpt_eff, plan_high^, off_cfg, ctx, False)
    if dual_expert:
        print("[wan22] high-noise expert base + loader ready (streaming, no host store)")

    # ── fp8 PARTIAL RESIDENCY (env WAN22_FP8_RESIDENT_GB, default 0 = off) ──────
    # MEASURED (WAN22_PHASE_TIMERS): loader.prefetch_next_with_ctx blocks ~50 ms
    # per block, and the forward AND the recompute backward each stream all 40
    # blocks => ~4 s of a 5.6 s step is weight staging, vs 15.5 ms/block for ALL
    # the transformer math. `prefetch_with_ctx` EARLY-RETURNS for any pinned block
    # (turbo_planned_loader.mojo), so each pinned block turns that stall into a
    # free lookup.
    #
    # Uses pin_residents_fp8_PREQUANTIZED — this checkpoint is ALREADY fp8
    # (F8_E4M3 [out,in] + `<key>.__fp8_scale` F32 [out]). The plain
    # pin_residents_fp8 quantizes FROM BF16 and would reinterpret the E4M3 bytes
    # as bf16 (silent corruption). The prequantized pin holds the bytes+scale
    # verbatim and await_block dequants with the SAME kernel the streamed path
    # uses => resident blocks are BIT-IDENTICAL to streamed ones.
    #
    # Pinning is PLAN-ORDER CONTIGUOUS, so a partial pin is fine: blocks
    # [0,pinned) resident, the rest stream. FULL residency is IMPOSSIBLE here —
    # the fp8 checkpoint is ~14 GB PER EXPERT on a 16 GB card. Budget goes to the
    # LOW expert first: only one expert runs per step (use_high = t >= boundary)
    # and LOW is the common case.
    var fp8_res_gb = _env_f32(String("WAN22_FP8_RESIDENT_GB"))
    if fp8_res_gb > Float32(0.0):
        var budget = Int(Float64(fp8_res_gb) * 1024.0 * 1024.0 * 1024.0)
        var n_low = loader.pin_residents_fp8_prequantized(budget, ctx)
        print("[wan22] fp8-resident LOW: pinned", n_low, "/", NUM_BLOCKS,
              "blocks (budget", fp8_res_gb, "GB)")
        var n_high = 0
        if dual_expert:
            var used_gb = Float32(n_low) * Float32(0.35)   # ~350 MB/fp8 block
            var rem = budget - Int(Float64(used_gb) * 1024.0 * 1024.0 * 1024.0)
            if rem > 0:
                n_high = loader_high.pin_residents_fp8_prequantized(rem, ctx)
            print("[wan22] fp8-resident HIGH: pinned", n_high, "/", NUM_BLOCKS, "blocks")
        print("[wan22] fp8-resident TOTAL:", n_low + n_high, "of",
              NUM_BLOCKS * (2 if dual_expert else 1),
              "block loads/step now free (the rest keep streaming)")

    # ── fp8 HOST-pinned tail (env WAN22_FP8_HOST_GB, default 0 = off) ──────────
    # Kills the DOUBLE-STAGING cost for whatever the device budget could not hold.
    # Each block is staged TWICE per step (forward + recompute backward), and each
    # staging is a ~50 ms HOST memcpy out of the mmap page cache into the loader's
    # pinned slab. Blocks pinned in HOST RAM skip that memcpy: await_block H2Ds
    # straight from pinned memory and dequants with the SAME kernel, so the Block
    # is BIT-IDENTICAL to both the streamed and the device-resident path.
    # Runs AFTER the device pin and skips anything already device-resident, so the
    # device budget takes blocks [0,N) and this absorbs the tail.
    # Host cost ~350 MB/block (~14 GB for a full expert) vs the ~27 GB/expert BF16
    # store the design declines — fp8 halves it, so the tails fit in ~49 GB free.
    var fp8_host_gb = _env_f32(String("WAN22_FP8_HOST_GB"))
    if fp8_host_gb > Float32(0.0):
        var hbudget = Int(Float64(fp8_host_gb) * 1024.0 * 1024.0 * 1024.0)
        var h_low = loader.pin_residents_fp8_host_prequantized(hbudget, ctx)
        print("[wan22] fp8-HOST LOW: pinned", h_low, "/", NUM_BLOCKS,
              "blocks (budget", fp8_host_gb, "GB)")
        var h_high = 0
        if dual_expert:
            h_high = loader_high.pin_residents_fp8_host_prequantized(hbudget, ctx)
            print("[wan22] fp8-HOST HIGH: pinned", h_high, "/", NUM_BLOCKS, "blocks")
        print("[wan22] fp8-HOST TOTAL:", h_low + h_high, "blocks now stage from"
              " pinned RAM (no mmap memcpy)")
        # WAN22_FP8H_OVERLAP=1: stage those pinned blocks on the COPY STREAM during
        # the previous block's compute instead of inline on the default stream at
        # await. nsys measured the inline path at 508.6 ms/step of H2D with **0%
        # overlap** (H2D union 508.6 + kernel union 859.2 = 1367.9 = exactly
        # additive), so this is pure serialization, not bandwidth. Bit-identical:
        # same bytes, same dequant kernel — only the stream and timing change.
        if _env_is_set(String("WAN22_FP8H_OVERLAP")):
            loader.set_fp8h_overlap(True)
            var share_slabs = _env_is_set(String("WAN22_FP8H_SHARE_SLABS"))
            if dual_expert:
                loader_high.set_fp8h_overlap(True)
                # WAN22_FP8H_SHARE_SLABS=1: ONE pair of slabs for BOTH experts. Only
                # one expert runs per step, so a pair each leaves ~550 MB idle all run.
                # ⚠ MEASURED A NET LOSS — see the handoff's dead-end list. It does free
                # the memory (peak 14854 → 14300 MiB) but costs 0.08 s/step at equal
                # residency, and spending the freed VRAM on more resident blocks does
                # not buy it back (best shared 1.500 s/step at 7.0 GB / 15186 MiB vs
                # best unshared 1.450 at 6.0 GB / 14854 MiB — slower AND larger).
                # Kept, off by default, for cards or models where the slabs are the
                # binding constraint rather than the pipeline depth.
                # Slots are keyed "<tag>|<prefix>" and the tags MUST differ: the two
                # experts are separate checkpoints that name their blocks identically,
                # so a bare prefix key would let HIGH's blocks.7 answer LOW's blocks.7.
                if share_slabs:
                    loader_high.share_fp8h_stage(loader.fp8h_stage(), String("high"))
                    loader.share_fp8h_stage(loader.fp8h_stage(), String("low"))
            print("[wan22] fp8-HOST: OVERLAPPED staging ON (copy stream, double-buffered"
                  + (String("; slabs SHARED across experts)") if (dual_expert and share_slabs)
                     else String(")")))

    # build_wan22_lora_set creates 10 adapters/block: the 8 ATTENTION projections
    # (self_attn.{q,k,v,o} + cross_attn.{q,k,v,o}) + ffn.0 (dim->ffn) + ffn.2
    # (ffn->dim), matching Torchref's wrapped-linear set (parity-gated cos>=0.999).
    var lora = build_wan22_lora_set(NUM_BLOCKS, DIM, FFN, rank, alpha)
    print("[lora] adapters:", wan22_total_adapters(lora))

    # ── RESUME (config `resume_state` / `start_step` / `warm_resume`) ──────────
    # Mirrors the mageflow contract (train_mageflow_real.mojo:621-646, :740-780).
    # MUST run before the resident AdamW state is built below, because
    # `lora_adamw_plain_device_state_init` seeds dev_m/dev_v from the host adapters
    # — load first and the moments are restored, load after and they are zeros.
    #
    # Continuation is exact because every per-step stream is a pure function of
    # `step`: the sigma draw is `_sample_t(shift_mode, step, seed, …)` and the data
    # order is `step % n_cache`. Running `range(start_step, steps)` therefore
    # continues the uninterrupted sequence rather than restarting it, and AdamW's
    # bias correction keeps counting from `step + 1`.
    var start_step = cfg.start_step
    if start_step < 0:
        start_step = 0
    if cfg.resume_state.byte_length() > 0:
        # Probe with the prefix the wan saver actually writes — `_wan22_lora_prefixes`
        # emits `diffusion_model.blocks.N.…` (wan22_stack_lora.mojo:1866). Probing the
        # bare `blocks.0.…` would never match, silently report "no moments", and
        # downgrade every FULL resume to a warm one.
        var probe = String("diffusion_model.blocks.0.self_attn.q")
        var resolved = trainer_resolve_resume_path(cfg.resume_state, probe)
        var has_moments = lora_train_state_has_moments(resolved, probe)
        if cfg.warm_resume:
            trainer_warn_warm_resume(String("wan22"), resolved)
            lora = load_wan22_lora_resume(NUM_BLOCKS, DIM, rank, alpha, resolved, ctx)
        elif not has_moments:
            raise Error(
                String("wan22 resume: '") + resolved + String("' has no AdamW moments"
                " (it is a bare PEFT file). Resuming from it would restart the"
                " optimizer with zeroed moments — pass warm_resume: true to accept"
                " that explicitly, or point resume_state at the .state sidecar.")
            )
        else:
            lora = load_wan22_lora_state(NUM_BLOCKS, DIM, rank, alpha, resolved, ctx)
            var rmeta = load_lora_train_state_meta(resolved, ctx)
            if len(rmeta) >= 2:
                if Int(rmeta[0]) != start_step:
                    print("  [resume] WARNING: checkpoint was written at step",
                          Int(rmeta[0]), "but start_step is", start_step,
                          "— the sigma/data streams will NOT line up with the"
                          " uninterrupted run")
                if Int(rmeta[1]) != Int(seed):
                    print("  [resume] WARNING: checkpoint seed", Int(rmeta[1]),
                          "!= configured seed", Int(seed))
        print("[resume] loaded", wan22_total_adapters(lora), "adapters from", resolved,
              "(", String("WARM, moments zeroed") if cfg.warm_resume
              else String("FULL, moments restored"), ") — resuming at step", start_step)
    if start_step >= steps:
        raise Error(
            String("wan22 resume: start_step ") + String(start_step)
            + String(" >= max_steps ") + String(steps) + String(" — nothing to do")
        )

    var rope = _build_rope_proven(ctx)
    var cos = rope[0].copy()
    var sin = rope[1].copy()

    var sample_pos = List[Float32]()
    var sample_neg = List[Float32]()
    if sample_every > 0:
        var sc = _load_sample_conds(sample_conds_path, TXT, TEXT_DIM, ctx)
        sample_pos = sc[0].copy()
        sample_neg = sc[1].copy()
        print("[sample] conds loaded from", sample_conds_path)

    # ── REAL DATA CACHE (P3: config cache_dir/dataset_cache_dir; env
    #    WAN22_DATA_CACHE OVERRIDES when set). When the resolved dir is
    #    non-empty + has sample_*.safetensors, x0/text come from the cache
    #    (real VAE latents + umt5 text), iterated ROUND-ROBIN across steps.
    #    When neither is set, the synthetic Box-Muller path runs. ──────────────
    var cache_dir = _env_str(String("WAN22_DATA_CACHE"))
    if cache_dir.byte_length() == 0:
        cache_dir = cfg.dataset_cache_dir
    var n_cache = 0
    var use_cache = False
    if cache_dir.byte_length() > 0:
        n_cache = _count_cache_samples(cache_dir)
        if n_cache == 0:
            raise Error(
                String("[data] no sample_*.safetensors at cache dir: ")
                + cache_dir
                + String(" (env WAN22_DATA_CACHE > config cache_dir; fail-loud)")
            )
        use_cache = True
    if use_cache:
        print("[data] REAL cache:", cache_dir, " samples=", n_cache,
              " (round-robin; REPLACES synthetic x0/text)")
    else:
        print("[data] SYNTHETIC x0/text (set config cache_dir or",
              "env WAN22_DATA_CACHE=<dir> for real data)")

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var have_first = False
    var train_start = perf_counter_ns()

    # Phase 1c device-native path (env WAN22_DEVNATIVE=1): device-resident LoRA
    # grads + resident AdamW (dev_p IS the live params), killing the per-step host
    # grad round-trips (~25k memcpys) of the offload path. dev_state is always
    # inited (cheap); the last step syncs dev_p -> lora.ad so save sees fresh weights.
    # Phase 2 autograd_v2 graph path (env WAN22_V2_GRAPH=1): the per-block backward
    # is the FINE-GRAINED engine graph (recompute through record_* wrappers) instead
    # of the hand-chain save-all tape + devnative backward. Same device-grad output
    # type, so it reuses the devnative consumer (resident AdamW) verbatim. Per-block
    # BIT-gated vs devnative on all 10 LoRA slots
    # (autograd_v2/tests/wan_block_graph_parity.mojo). Default OFF (C13).
    var use_devnative = _env_is_set(String("WAN22_DEVNATIVE"))
    var use_graph = _env_is_set(String("WAN22_V2_GRAPH"))
    var use_devpath = use_devnative or use_graph
    # The recompute backward drivers (devnative, graph) re-derive every activation
    # from the block input and NEVER read the forward's save-all host tape — only
    # the host offload backward does. Skipping it removes ~5.5 GB/step of host list
    # traffic (23 lists/block built + copied, 40 blocks). Pure omission: the forward
    # OUTPUT is byte-identical, so this cannot move loss or grads.
    var save_block_acts = not use_devpath
    # Device-resident LoRA forward (no host round trip) — benched 26x, BIT-EQUAL
    # (models/wan22/parity/wan22_lora_fwd_hostvsdev_bench.mojo). Enabled with the
    # device paths; the host offload path keeps the old forward byte-identical (C13).
    var use_device_lora = use_devpath
    # Batched cross-attention (kills the per-head loop: ~28.8k launches/step).
    # ⚠ VALUE-CLASS numerics change (~1e-3), unlike device_lora which is bit-equal.
    # OFF BY DEFAULT — MEASURED 8.167 → 8.103 s/step = 0.8%, i.e. removing 36% of the
    # step's kernel launches bought under 1%: this step is NOT launch-bound (nsys:
    # ~1.6 s/step CUDA API + ~0.69 s/step GPU out of a 14 s step). Not worth shifting
    # the numerics of a chassis shared with scail2/bernini for 0.8%. The flag, the
    # batched path and the both-modes gate stay in place: revisit at CUDA-graph
    # capture time, where the per-head loop's per-head slices/permutes (allocating
    # ops) and the node count actually matter for StepSlab/replay.
    var use_batched_cross = False
    # Device modvec builder: replaces a pure-CPU broadcast-add loop measured at
    # 2.586 s/step (32% of the step). BIT-EQUAL (same F32 adds, on GPU).
    var use_device_modvecs = use_devpath
    # WAN22_DEV_MODVECS16=1: keep the six AdaLN vectors as BF16 DEVICE tensors for
    # their whole life instead of reading them back and re-uploading them in every
    # consumer. Benched (wan22_modvecs_split_bench.mojo) at 0.211 s/step of readback
    # + 0.101 s/step per consumer pass (forward AND graph recompute both consume)
    # against 0.011 s/step of actual arithmetic. BIT-EQUAL — same broadcast add,
    # same `.cast[bfloat16]()`. Opt-in so the two representations can be A/B'd on
    # the same build; requires a device path (the host offload backward is untouched).
    var use_device_modvecs_t = use_devpath and _env_is_set(String("WAN22_DEV_MODVECS16"))
    # WAN22_PHASE_TIMERS=1: per-step wall-clock split (data / forward / backward+opt).
    # nsys CPU sampling is unavailable on this box, so this is how host time is
    # localized. Prints only; no effect on the computation.
    var phase_timers = _env_is_set(String("WAN22_PHASE_TIMERS"))
    var n_adapters = wan22_total_adapters(lora)
    var dev_state = lora_adamw_plain_device_state_init(lora.ad, 0, n_adapters, ctx)
    if use_graph:
        print("[wan22] V2_GRAPH: autograd_v2 fine-grained block backward + resident AdamW")
    elif use_devnative:
        print("[wan22] DEVNATIVE: resident device grads + AdamW (no host round-trip)")

    print("")
    print("step  expert    t        MSE            grad_norm     sec")
    # sample AT START (fresh runs only — a resumed segment already has its earlier
    # points on disk). This is the base-model reference later points compare against.
    var sample_shift = discrete_flow_shift
    if sample_every > 0 and start_step == 0:
        var tag = String("step0000_start")
        try:
            _wan22_sample_png[H, Dh, S, TXT](
                    tag, samples_dir + String("/") + tag + String(".png"),
                    sample_pos, sample_neg, base, loader, high_base, loader_high,
                    dual_expert, boundary, lora, cos.copy(), sin.copy(),
                    DIM, FFN, in_ch_eff, TEXT_DIM, OUT_CH, FREQ_DIM, EPS,
                    SAMPLE_STEPS, SAMPLE_CFG, sample_shift, seed,
                    False, use_device_lora, use_batched_cross,
                    use_device_modvecs, use_device_modvecs_t, ctx,
                )
        except e:
            print("  [sample] FAILED at start:", e, "— training continues")

    for step in range(start_step, steps):
        var t0 = perf_counter_ns()

        # ── timestep + flow-match target (torchref: x_t=(1-t)x0+t*noise, tgt=noise-x0)
        # `step` is the GLOBAL optimizer step, so on a resume the sigma draw and the
        # `step % n_cache` data order continue the original sequence exactly.
        var t = _sample_t(shift_mode, step, seed,
                          DEFAULT_SIGMOID_SCALE, discrete_flow_shift)
        # smoke pin: force the expert branch deterministically (see alt_t above).
        if alt_t:
            t = t_low_pin if (step % 2 == 0) else t_high_pin

        # ── EXPERT SELECT (Torchref wan_train_network.py:546-569). Torchref decides
        #    high_noise from the FIRST sampled timestep and rejection-samples the
        #    whole batch to that side; for our batch=1 loop the single sampled t
        #    directly selects the expert (no rejection needed). Sample/inference
        #    always uses the LOW expert (wan_train_network.py:224-225) — this is a
        #    TRAIN-ONLY switch. `use_high` requires dual_expert so the single-expert
        #    (back-compat) path never touches the redundant high loader.
        var use_high = dual_expert and (t >= boundary)

        # x0/text: REAL cache (round-robin) when enabled, else synthetic. The
        # flow-match noising below (x_t=(1-t)x0+t*noise, tgt=noise-x0) is unchanged.
        # For I2V we ALSO fetch cond_y_patch [S,80] (clean, never noised).
        var x0: List[Float32]
        var txt_tokens: List[Float32]
        var cond_y_patch = List[Float32]()   # [S*COND_PACK]; populated only for i2v
        if use_cache:
            var samp_idx = step % n_cache
            var c_std = Float32(0.0)
            var c_mean = Float32(0.0)
            var c_txtlen = 0
            var pair = _load_cache_sample(
                cache_dir, samp_idx, ctx, c_std, c_mean, c_txtlen, i2v_mode
            )
            # ROUND-TRIP GATE: this std/mean is over the RAW cached latent and must
            # match the torch-printed std/mean for sample_{samp_idx} (proves the
            # bytes/layout are read correctly — no transpose/scramble).
            print("  [cache] sample", samp_idx, " latent std=", c_std,
                  " mean=", c_mean, " txt_len=", c_txtlen, " (must match torch)")
            if i2v_mode:
                cond_y_patch = pair.pop()  # [cond_y_patch] appended last when i2v
            txt_tokens = pair.pop()   # [text]
            x0 = pair.pop()           # [x0]
        else:
            x0 = _synth_x0(S * IN_CH, seed + UInt64(step) * 101 + 1)
            txt_tokens = _synth_x0(TXT * TEXT_DIM, seed + UInt64(step) * 777 + 9)
            if i2v_mode:
                cond_y_patch = _synth_x0(S * COND_PACK,
                                         seed + UInt64(step) * 313 + 7)
        # Flow-match noising is on the 16-ch latent patch ONLY (x0 is [S,64]); y is
        # clean conditioning. target/loss stay on the 64-dim (16-ch) velocity.
        # Smoke overfit pin: WAN22_FIXED_NOISE=1 drops the per-step term from the
        # noise seed so every step draws the SAME noise. Combined with a 1-sample
        # cache + WAN22_ALTERNATE_T equal pins (fixed t, single expert), the
        # (x_t, target) pair is identical every step → the raw MSE must decrease
        # monotonically, isolating the fwd/bwd/AdamW-LoRA update on the real Bernini
        # fp8 weights (a clean loss-drop proof). Default OFF = the stochastic recipe.
        var _noise_seed = seed * UInt64(2000003) + UInt64(step) * 104729 + 3
        if _env_is_set(String("WAN22_FIXED_NOISE")):
            _noise_seed = seed * UInt64(2000003) + 3
        var noise = _noise(S * IN_CH, _noise_seed)
        var img_tokens = List[Float32]()
        var target = List[Float32]()
        for i in range(len(x0)):
            img_tokens.append((Float32(1.0) - t) * x0[i] + t * noise[i])
            target.append(noise[i] - x0[i])
        # Torchref compares the unpatchified prediction with canonical velocity.
        # This raw-stack loop therefore repacks the target to the head's c-fast
        # output order before MSE/backward.
        target = _wan_patch_input_to_head(target^, S, SAMPLE_LAT_C)

        # ── MODEL INPUT ────────────────────────────────────────────────────────
        # T2V: model input = the noised 16-ch latent patch [S,64].
        # I2V: per-token concat(noisy_latent_patch[64], cond_y_patch[80]) -> [S,144].
        # patchify3d packs channel-SLOWEST, so this per-token concat equals
        # patchify(concat([noisy_latent(16), y(20)])) byte-for-byte (cols 0..63 =
        # 16 latent chans, cols 64..143 = 20 y chans) — the [5120,144] patch-embed
        # weight layout. y is NOT noised (only the 16-ch latent is).
        var model_in = List[Float32]()
        if i2v_mode:
            for tok in range(S):
                for c in range(IN_CH):          # 64 noisy latent
                    model_in.append(img_tokens[tok * IN_CH + c])
                for c in range(COND_PACK):      # 80 clean cond_y
                    model_in.append(cond_y_patch[tok * COND_PACK + c])
        else:
            model_in = img_tokens.copy()

        # Timestep-embedding input scale — CONFIRMED against the torchref oracle
        # (torchref-ss/torchref src/torchref/hv_train.py::
        #  get_noisy_model_input_and_timesteps, lines ~703-708):
        #     noisy_model_input = (1 - t) * latents + t * noise   # t = sigma in [0,1]
        #     timesteps = t * 1000.0
        #     timesteps += 1                                       # range 1..1000
        # i.e. the flow-match noising uses the sigma t (done above), while the DiT
        # timestep argument fed to sinusoidal_embedding_1d is t*1000 + 1. The Wan
        # model (wan/modules/model.py:851) then calls
        #   sinusoidal_embedding_1d(freq_dim, timesteps) -> time_embedding(...).
        var t_model = t * Float32(1000.0) + Float32(1.0)

        # Feed the ACTIVE expert's (base, loader) into the SHARED LoRA engine.
        var t_data_done = perf_counter_ns()
        var fwd: Wan22StackForward
        if use_high:
            fwd = wan22_stack_lora_forward_offload[H, Dh, S, TXT](
                model_in.copy(), txt_tokens.copy(), t_model,
                high_base, loader_high, lora, cos.copy(), sin.copy(),
                DIM, FFN, in_ch_eff, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
                save_block_acts, use_device_lora, use_batched_cross,
                use_device_modvecs, use_device_modvecs_t,
            )
        else:
            fwd = wan22_stack_lora_forward_offload[H, Dh, S, TXT](
                model_in.copy(), txt_tokens.copy(), t_model,
                base, loader, lora, cos.copy(), sin.copy(),
                DIM, FFN, in_ch_eff, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
                save_block_acts, use_device_lora, use_batched_cross,
                use_device_modvecs, use_device_modvecs_t,
            )

        var t_fwd_done = perf_counter_ns()
        # plain per-element MSE then mean; d_out = 2*(pred-target)/N (weighting None).
        var nout = len(fwd.out)
        var loss = Float32(0.0)
        var d_out = List[Float32]()
        var inv_n = Float32(2.0) / Float32(nout)
        for i in range(nout):
            var diff = fwd.out[i] - target[i]
            loss += diff * diff
            d_out.append(inv_n * diff)
        loss = loss / Float32(nout)
        if not have_first:
            first_loss = loss
            have_first = True
        last_loss = loss

        # Backward through the SAME active expert (base+loader) that ran forward;
        # AdamW steps the SHARED LoRA. WAN22_DEVNATIVE: device-resident grads +
        # resident AdamW (no host round-trip); else the host-list offload path.
        var gn: Float32 = Float32(0.0)
        if use_devpath:
            var dgrads: Wan22LoraDeviceGradSet
            if use_graph:
                if use_high:
                    dgrads = wan22_stack_lora_backward_offload_graph[H, Dh, S, TXT](
                        d_out, model_in.copy(), txt_tokens.copy(),
                        high_base, loader_high, lora, cos.copy(), sin.copy(), fwd,
                        DIM, FFN, in_ch_eff, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
                        use_batched_cross,
                    )
                else:
                    dgrads = wan22_stack_lora_backward_offload_graph[H, Dh, S, TXT](
                        d_out, model_in.copy(), txt_tokens.copy(),
                        base, loader, lora, cos.copy(), sin.copy(), fwd,
                        DIM, FFN, in_ch_eff, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
                        use_batched_cross,
                    )
            elif use_high:
                dgrads = wan22_stack_lora_backward_offload_devnative[H, Dh, S, TXT](
                    d_out, model_in.copy(), txt_tokens.copy(),
                    high_base, loader_high, lora, cos.copy(), sin.copy(), fwd,
                    DIM, FFN, in_ch_eff, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
                )
            else:
                dgrads = wan22_stack_lora_backward_offload_devnative[H, Dh, S, TXT](
                    d_out, model_in.copy(), txt_tokens.copy(),
                    base, loader, lora, cos.copy(), sin.copy(), fwd,
                    DIM, FFN, in_ch_eff, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
                )
            # ⚠ MUST BE EVERY STEP — this is a CORRECTNESS requirement, not a save
            # cadence. The forward reads the HOST adapters: the stack does
            # `_wan_block_lora_for(lora, bi)` -> `lora.ad[...]` (wan22_stack_lora.mojo:296)
            # and `lora_adapter_to_device` uploads those host values
            # (klein/lora_block.mojo:84). The resident AdamW only writes dev_p and
            # touches lora.ad when this flag is set, so syncing on a cadence means the
            # forward trains against a LoRA frozen at its last sync.
            #
            # MEASURED with the old `step == steps - 1`: over 500 steps the loss was
            # DEAD FLAT (mean 2.146 / 2.221 / 2.160 / 2.140 / 2.147 per 100-step
            # window) and grad_norm sat at ~0.02 — because B starts at 0, so an
            # un-synced LoRA contributes exactly nothing and the model is the frozen
            # base. At the first sync grad_norm jumped 0.018 -> 4.46 and the loss
            # finally moved (1.859, then 1.803). The optimizer was working the whole
            # time; the forward just never saw it.
            var sync_now = True
            gn = fused_lora_adamw_plain_step_resident_device_grads(
                dev_state, lora.ad, dgrads.grad_indices, dgrads.d_a, dgrads.d_b,
                step + 1, lr, beta1, beta2, opt_eps, weight_decay, ctx,
                Float32(1.0), sync_now, max_grad_norm,
            )
        else:
            var grads: Wan22LoraGradSet
            if use_high:
                grads = wan22_stack_lora_backward_offload[H, Dh, S, TXT](
                    d_out, model_in.copy(), txt_tokens.copy(),
                    high_base, loader_high, lora, cos.copy(), sin.copy(), fwd,
                    DIM, FFN, in_ch_eff, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
                )
            else:
                grads = wan22_stack_lora_backward_offload[H, Dh, S, TXT](
                    d_out, model_in.copy(), txt_tokens.copy(),
                    base, loader, lora, cos.copy(), sin.copy(), fwd,
                    DIM, FFN, in_ch_eff, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
                )
            gn = _clip(grads, max_grad_norm)
            wan22_lora_adamw_step(lora, grads, step + 1, lr, ctx,
                                  beta1, beta2, opt_eps, weight_decay)
            if grads.nonfinite_lora_grads != 0:
                print("  !! nonfinite lora grads =", grads.nonfinite_lora_grads)
        var t_bwd_done = perf_counter_ns()
        if phase_timers:
            print("  [phase] data=", Float64(t_data_done - t0) / 1.0e9,
                  " fwd=", Float64(t_fwd_done - t_data_done) / 1.0e9,
                  " bwd+opt=", Float64(t_bwd_done - t_fwd_done) / 1.0e9, "s")
        var secs = Float64(perf_counter_ns() - t0) / 1.0e9
        var expert = String("HIGH") if use_high else String("LOW ")
        print(step, " ", expert, " ", t, "  ", loss, "  ", gn, "  ", secs)

        # ── periodic checkpoint (config `save_every`, 0 = only at the end) ──
        # Writes the SAME two artifacts the final save writes, suffixed with the
        # step, so any checkpoint can be sampled or resumed from. On the device
        # paths the live params are the device copies, so sync them back into
        # lora.ad first — the same sync the final save relies on at the last step.
        if save_every > 0 and (step + 1) % save_every == 0 and (step + 1) < steps:
            var tag = String("-") + String(step + 1)
            var ck = ckpt_stem + tag + String(".safetensors")
            var nck = save_wan22_lora(lora, ck, ctx)
            var cmeta = List[Float32]()
            cmeta.append(Float32(step + 1))
            cmeta.append(Float32(Int(seed)))
            var nst = save_wan22_lora_state(lora, ck + String(".state"), ctx, cmeta^)
            print("  [ckpt] step", step + 1, "->", ck, "(", nck, "pairs,", nst, "state )")

        # ── sample at the cadence. Placed next to the checkpoint so an image and
        # the exact weights that produced it are written at the same step.
        if sample_every > 0 and (step + 1) % sample_every == 0:
            var tag = String("step") + _pad4(step + 1)
            try:
                _wan22_sample_png[H, Dh, S, TXT](
                    tag, samples_dir + String("/") + tag + String(".png"),
                    sample_pos, sample_neg, base, loader, high_base, loader_high,
                    dual_expert, boundary, lora, cos.copy(), sin.copy(),
                    DIM, FFN, in_ch_eff, TEXT_DIM, OUT_CH, FREQ_DIM, EPS,
                    SAMPLE_STEPS, SAMPLE_CFG, sample_shift, seed,
                    False, use_device_lora, use_batched_cross,
                    use_device_modvecs, use_device_modvecs_t, ctx,
                )
            except e:
                print("  [sample] FAILED at step", step + 1, ":", e,
                      "— training continues")

    # ── save PEFT weights + full resume state (A/B + AdamW moments) ────────────
    var npairs = save_wan22_lora(lora, out_path, ctx)
    var meta = List[Float32]()
    meta.append(Float32(steps))
    meta.append(Float32(Int(seed)))
    var state_path = out_path + String(".state")
    var nstate = save_wan22_lora_state(lora, state_path, ctx, meta^)
    print("")
    print("[save] wrote", npairs, "PEFT pairs ->", out_path)
    print("[save] wrote", nstate, "state tensors ->", state_path)
    print("trained steps=", steps, " first MSE=", first_loss, " last MSE=", last_loss,
          " total sec=", Float64(perf_counter_ns() - train_start) / 1.0e9)
    if dual_expert:
        print("RESULT: wan22 DUAL-EXPERT LoRA trainer ran (low+high experts switched",
              "at t>=", boundary, "; shared LoRA; switching-mechanism smoke — per-block",
              "grad parity already certified, dual switch NOT a new parity claim).")
    else:
        print("RESULT: wan22 low-noise LoRA trainer ran (single expert; real weighted",
              "step; parity UNVERIFIED — orchestrator L2P gate pending).")
