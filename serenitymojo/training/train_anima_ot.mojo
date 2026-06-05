# serenitymojo/training/train_anima_ot.mojo
#
# ANIMA LoRA TRAINING — OneTrainer-FAITHFUL recipe (Chunk A).
#
# This is train_anima_real.mojo RE-TARGETED to the OneTrainer reference
# (/home/alex/OneTrainer-anima-ref), changing EXACTLY the 5 recipe deltas, each
# cited to OT source. The Anima block/stack/attn-mlp-LoRA math is REUSED unchanged
# (parity-gated cos>=0.99999999). The original train_anima_real.mojo is left intact.
#
# THE 5 OT DELTAS (vs the inference-flame-anchored train_anima_real.mojo):
#  1. Context length 256 -> 512.  AnimaModel.PROMPT_MAX_LENGTH=512 (AnimaModel.py:23);
#     the conditioner output is always (B,512,1024) (AnimaModel.py:229). The cross-attn
#     SDPA is comptime-shaped on S_TXT, so setting S_TXT=512 just monomorphizes it
#     (proven to compile+run at 512 by parity/anima_ot_step_parity.mojo).
#  2. scale_latents BEFORE flow.  scaled = (latent - mean) * (1/std) per-channel, 16 ch
#     (AnimaModel.py:233-236). mean/std READ from the Qwen-Image VAE config:
#       /home/alex/.serenity/models/checkpoints/qwen-image-2512/vae/config.json
#     (latents_mean / latents_std, z_dim=16) — baked below with the path cited.
#  3. OT discrete timestep + sigma.  _get_timestep_discrete for the LOGIT_NORMAL
#     distribution the Anima LoRA preset selects (training_presets/#anima LoRA.json:24
#     "timestep_distribution":"LOGIT_NORMAL"; defaults noising_bias=0, noising_weight=0,
#     timestep_shift=1.0, min/max_noising_strength=0/1, dynamic_timestep_shifting=False
#     — TrainConfig.py:1029-1035). For LOGIT_NORMAL: bias=noising_bias=0,
#     scale=noising_weight+1=1, t = sigmoid(normal(0,1))*N + 0, then the shift map with
#     shift=1.0 is identity, then .int() (ModelSetupNoiseMixin.py:156-162,213,253).
#     Then sigma = (timestep_index + 1)/num_train_timesteps (all_timesteps=arange(1,N+1);
#     sigma=all/N; sigma[ts] -> (ts+1)/N) (ModelSetupFlowMatchingMixin.py:23-29).
#       noisy  = noise*sigma + scaled*(1-sigma)   (ModelSetupFlowMatchingMixin.py:36-37)
#       target = noise - scaled                   (BaseAnimaSetup.py:143)
#     Pass timestep/1000 into the sinusoidal t-embedder (BaseAnimaSetup.py:137 —
#     the old trainer passed RAW sigma; OT passes timestep/1000).
#  4. Loss = unmasked MSE mean((pred - target)^2)  (BaseAnimaSetup.calculate_loss ->
#     _flow_matching_losses -> __unmasked_losses F.mse_loss reduction='none'.mean;
#     mse_strength=1, loss_weight_fn=CONSTANT, loss_weight=1). Same MSE the old
#     trainer used — kept, re-verified by the parity gate.
#  5. LoRA save in OT diffusers key naming (save_anima_lora_ot) — mirrors what
#     AnimaLoRASaver dumps (transformer_lora.state_dict() diffusers names attn1/attn2/
#     ff with lora_down/lora_up + alpha; AnimaLoRASaver.py:25, LoRAModule.py:547-551).
#     The existing kohya save (save_anima_lora) is ALSO emitted (cheap).
#
# Run (SEPARATE command, after build):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/training/train_anima_ot.mojo -o /tmp/train_anima_ot
#   /tmp/train_anima_ot

from std.sys import argv
from std.collections import List, Optional
from std.os import listdir
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, exp as fexp, isfinite
from std.memory import ArcPointer
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.ffi import sys_system
from serenitymojo.ops.cast import cast_tensor

from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.norm import rms_norm

from serenitymojo.models.anima.config import anima
from serenitymojo.models.anima.weights import (
    AnimaBlockWeights, AnimaStackBase, load_anima_stack_base,
    load_anima_block_weights_bf16_normf32, verify_anima_stack_shapes,
)
from serenitymojo.models.anima.lora_block import ANIMA_SLOTS
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.anima.anima_stack_lora import (
    AnimaLoraSet, AnimaLoraGrads, build_anima_lora_set,
    anima_stack_lora_forward_streamed, anima_stack_lora_backward_streamed,
    anima_stack_lora_forward_device_resident, anima_stack_lora_backward_device_resident,
    anima_lora_set_to_device,
    anima_lora_adamw_step, save_anima_lora, save_anima_lora_ot,
    load_anima_lora_resume, save_anima_lora_state, load_anima_lora_state,
)
from serenitymojo.models.dit.anima_contract import (
    ANIMA_HIDDEN, ANIMA_NUM_HEADS, ANIMA_HEAD_DIM, ANIMA_DEPTH,
    ANIMA_LATENT_CHANNELS, ANIMA_PATCH_SIZE, ANIMA_ADAPTER_DIM,
)
# GAP 2: decode sample/target latents -> RGB PNGs for the visual/directional check.
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.training.progress_display import print_trainer_progress


comptime TArc = ArcPointer[Tensor]

# ── REAL Anima dims (anima_contract) ──────────────────────────────────────────
comptime B = 1
comptime H = ANIMA_NUM_HEADS        # 16
comptime Dh = ANIMA_HEAD_DIM        # 128
comptime D = ANIMA_HIDDEN           # 2048
comptime F = 8192                   # GELU MLP hidden
comptime JOINT = ANIMA_ADAPTER_DIM  # 1024 cross-attn context dim
comptime C = ANIMA_LATENT_CHANNELS  # 16
comptime PS = ANIMA_PATCH_SIZE      # 2
comptime IN_PATCH = (C + 1) * PS * PS   # 68
comptime OUT_PATCH = C * PS * PS        # 64
# DELTA 1: OT context length 256 -> 512 (AnimaModel.py:23 PROMPT_MAX_LENGTH=512).
comptime S_TXT = 512
comptime EPS = Float32(1e-06)

# ── Training config ───────────────────────────────────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime LR = Float32(3.0e-5)          # #anima LoRA.json:4 learning_rate 3e-05
comptime CLIP_NORM = Float32(1.0)
comptime SEED = UInt64(42)
comptime RUN_STEPS = 16                 # Chunk D run default (~21.8 s/step at S_IMG=256).

# DELTA 3: OT discrete timestep schedule constants.
#   num_train_timesteps: FlowMatchEulerDiscreteScheduler default (1000).
#   LOGIT_NORMAL params from the preset + TrainConfig defaults:
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime NOISING_BIAS = Float32(0.0)   # TrainConfig default; preset leaves unset
comptime NOISING_WEIGHT = Float32(0.0) # TrainConfig default
comptime TIMESTEP_SHIFT = Float32(1.0) # TrainConfig default (identity shift map)
comptime MIN_NOISING = Float32(0.0)    # TrainConfig default
comptime MAX_NOISING = Float32(1.0)    # TrainConfig default

# Thermal/memory crop: latent grid -> LATENT_HW x LATENT_HW -> S_IMG=(LATENT_HW/2)^2.
# Chunk D: REAL 256x256 image -> VAE latent 32x32 -> S_IMG=256 (3-axis rope correct
# at any grid). The real cache (anima_prepare_ot) stores a [1,16,32,32] latent.
# 512px training: latent 64x64 -> S_IMG = (64/2)^2 = 1024 (the OT 512px target).
comptime LATENT_HW = 64
comptime S_IMG = (LATENT_HW // PS) * (LATENT_HW // PS)

# Smoke mode: hold the timestep + noise draw CONSTANT so the flow target is identical
# every step (isolates the LoRA learning signal: loss MUST fall as adapters fit a
# fixed target). The REAL OT schedule (per-step logit-normal discrete timestep) runs
# when this is False.
# FIXED_STEP_SMOKE: hold (timestep, noise) constant so the flow target is identical
# every step — isolates the LoRA learning signal (loss MUST fall as adapters fit a
# fixed target). False = the REAL OT per-step logit-normal discrete-timestep schedule
# (loss bounces per step but the headline learning trend is real). Picked at runtime:
#   argv[1] == "real"  -> REAL schedule (default)
#   argv[1] == "fixed" -> fixed-sigma smoke (clean learning curve)
comptime FIXED_TIMESTEP = 500           # fixed discrete index for the smoke

# ── Data paths ────────────────────────────────────────────────────────────────
# Chunk D REAL cache (anima_prepare_ot output): a real VAE-encoded latent.
comptime CACHE_DIR = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/anima_real_ot"
# Chunk C wired (GAP 1): context produced ENTIRELY by the Mojo pipeline
# (pipeline/anima_text_context.mojo: real Qwen3-0.6B encoder -> zero-pad ->
# net.llm_adapter) from the real caption tokenized at max_len 512. This REPLACES
# the captured inference-flame sidecar (anima_embeddings.safetensors, a different
# caption captured at 256 tokens). The Mojo context is a real [1,512,1024].
comptime CONTEXT_PATH = "/home/alex/mojodiffusion/output/anima_gap2/anima_context_mojo.safetensors"
comptime LORA_OUT = "/home/alex/mojodiffusion/output/anima_lora_ot_real.safetensors"
comptime LORA_OUT_OT = "/home/alex/mojodiffusion/output/anima_lora_ot_real_otkeys.safetensors"
# GAP 2: Qwen-Image VAE (wan21 keys) to DECODE sample/target latents -> RGB.
comptime VAE_FILE = "/home/alex/.serenity/models/anima/split_files/vae/qwen_image_vae.safetensors"
comptime GAP2_DIR = "/home/alex/mojodiffusion/output/anima_gap2"
# CADENCE: per-segment step-tagged LoRA checkpoints + cadence sample PNGs.
comptime CADENCE_DIR = "/home/alex/mojodiffusion/output/anima_cadence"
# STAGE 2 (giger3): the 70-sample dataset cache (each file has `latent`[1,16,64,64]
# RAW + `context_cond`[1,512,1024]) and the step-tagged checkpoint output dir.
# DATASET_DIR is the comptime default; argv[5] overrides it (empty/"-" = single-sample).
comptime DATASET_DIR = "/home/alex/mojodiffusion/output/giger3_cache"
comptime GIGER3_TRAIN_DIR = "/home/alex/mojodiffusion/output/giger3_train"
comptime CKPT_EVERY_DEFAULT = 500       # checkpoint cadence (argv[6] overrides)

# ── DELTA 2: VAE scale_latents constants (16 ch), READ from the Qwen-Image VAE
#    config.json latents_mean / latents_std (z_dim=16). File path:
#      /home/alex/.serenity/models/checkpoints/qwen-image-2512/vae/config.json
#    These are the verbatim values from that file (read 2026-06-02). scale_latents:
#      scaled = (latent - mean) * (1/std)   (AnimaModel.py:233-236).
def _vae_latents_mean() -> List[Float32]:
    var v = List[Float32]()
    v.append(-0.7571); v.append(-0.7089); v.append(-0.9113); v.append(0.1075)
    v.append(-0.1745); v.append(0.9653); v.append(-0.1517); v.append(1.5508)
    v.append(0.4134); v.append(-0.0715); v.append(0.5517); v.append(-0.3632)
    v.append(-0.1922); v.append(-0.9497); v.append(0.2503); v.append(-0.2921)
    return v^


def _vae_latents_std() -> List[Float32]:
    var v = List[Float32]()
    v.append(2.8184); v.append(1.4541); v.append(2.3275); v.append(2.6558)
    v.append(1.2196); v.append(1.7708); v.append(2.6052); v.append(2.0743)
    v.append(3.2687); v.append(2.1526); v.append(2.8652); v.append(1.5579)
    v.append(1.6382); v.append(1.1253); v.append(2.8251); v.append(1.916)
    return v^


# ── deterministic host gaussian noise (Box-Muller on a PCG stream) ────────────
def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int((state >> 11) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int((state >> 11) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^


# ── DELTA 3: OT discrete timestep for the LOGIT_NORMAL distribution ───────────
# Faithful to ModelSetupNoiseMixin._get_timestep_discrete (lines 142-162,213,253):
#   min_t = int(N*min_noising); max_t = int(N*max_noising); num_t = max_t - min_t
#   LOGIT_NORMAL: bias = noising_bias; scale = noising_weight + 1.0
#     z = normal(bias, scale); t = sigmoid(z) * num_t + min_t
#   shift map: t = N*shift*t / ((shift-1)*t + N)  [identity when shift=1]
#   return int(t)
# We supply normal(0,1) draws via Box-Muller (the trainer's deterministic stream).
def _normal_sample(seed: UInt64, bias: Float32, scale: Float32) -> Float32:
    var n = _host_noise(2, seed)   # standard normals
    return bias + scale * n[0]


def _sigmoid(x: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + fexp(-x))


def _ot_timestep_discrete(seed: UInt64) -> Int:
    var min_t = Int(Float32(NUM_TRAIN_TIMESTEPS) * MIN_NOISING)
    var max_t = Int(Float32(NUM_TRAIN_TIMESTEPS) * MAX_NOISING)
    var num_t = Float32(max_t - min_t)
    var scale = NOISING_WEIGHT + Float32(1.0)   # LOGIT_NORMAL: noising_weight+1
    var z = _normal_sample(seed, NOISING_BIAS, scale)
    var t = _sigmoid(z) * num_t + Float32(min_t)
    # shift map (identity when TIMESTEP_SHIFT == 1.0)
    var sh = TIMESTEP_SHIFT
    if not ((sh - Float32(1.0)) < Float32(1.0e-6) and (sh - Float32(1.0)) > Float32(-1.0e-6)):
        var N = Float32(NUM_TRAIN_TIMESTEPS)
        t = N * sh * t / ((sh - Float32(1.0)) * t + N)
    var ti = Int(t)
    if ti < 0:
        ti = 0
    if ti >= NUM_TRAIN_TIMESTEPS:
        ti = NUM_TRAIN_TIMESTEPS - 1
    return ti


# ── load a cache tensor as host F32 (BF16/F32 view -> upcast) ─────────────────
def _cache_f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return cast_tensor(t^, STDtype.F32, ctx)


# ── patchify INPUT layout: 5D channels-last -> patches [B*N,68] (mask ch=0) ────
def _patchify_in(
    x: List[Float32], Bd: Int, T: Int, Hd: Int, Wd: Int, Cd: Int
) -> List[Float32]:
    var pH = PS
    var pW = PS
    var nH = Hd // pH
    var nW = Wd // pW
    var Cp = Cd + 1
    var N = T * nH * nW
    var pd = Cp * pH * pW
    var out = List[Float32]()
    out.reserve(Bd * N * pd)
    for _ in range(Bd * N * pd):
        out.append(Float32(0.0))
    for b in range(Bd):
        for t in range(T):
            for ih in range(nH):
                for iw in range(nW):
                    var pn = (t * nH + ih) * nW + iw
                    for c in range(Cp):
                        for ph in range(pH):
                            for pw in range(pW):
                                var od = (b * N + pn) * pd + (c * pH * pW + ph * pW + pw)
                                if c < Cd:
                                    var hh = ih * pH + ph
                                    var ww = iw * pW + pw
                                    var src = ((b * T + t) * Hd + hh) * Wd * Cd + ww * Cd + c
                                    out[od] = x[src]
    return out^


# ── patchify OUTPUT layout: -> target patches [B*N,64] (C fastest) ────────────
def _patchify_out(
    x: List[Float32], Bd: Int, T: Int, Hd: Int, Wd: Int, Cd: Int
) -> List[Float32]:
    var pH = PS
    var pW = PS
    var nH = Hd // pH
    var nW = Wd // pW
    var N = T * nH * nW
    var pd = Cd * pH * pW
    var out = List[Float32]()
    out.reserve(Bd * N * pd)
    for _ in range(Bd * N * pd):
        out.append(Float32(0.0))
    for b in range(Bd):
        for t in range(T):
            for ih in range(nH):
                for iw in range(nW):
                    var pn = (t * nH + ih) * nW + iw
                    for ph in range(pH):
                        for pw in range(pW):
                            for c in range(Cd):
                                var od = (b * N + pn) * pd + (ph * pW * Cd + pw * Cd + c)
                                var hh = ih * pH + ph
                                var ww = iw * pW + pw
                                var src = ((b * T + t) * Hd + hh) * Wd * Cd + ww * Cd + c
                                out[od] = x[src]
    return out^


# ── t_embedder forward (anima_dit.mojo:822-843), F32 on device ────────────────
struct _TEmb(Movable):
    var t_cond: List[Float32]      # [B, 2048] RAW (un-silu'd) sinusoidal RMSNorm
    var base_adaln: List[Float32]  # [B, 6144]

    def __init__(out self, var t_cond: List[Float32], var base_adaln: List[Float32]):
        self.t_cond = t_cond^
        self.base_adaln = base_adaln^


# cos-first sinusoidal embedding [B] -> [B, dim] (anima_dit _anima_sinusoidal).
def _sinusoidal_host(val: Float32, dim: Int) -> List[Float32]:
    var half = dim // 2
    var neg_ln = -flog(Float32(10000.0))
    var out = List[Float32]()
    out.reserve(dim)
    for _ in range(dim):
        out.append(Float32(0.0))
    for i in range(half):
        var freq = fexp(neg_ln * (Float32(i) / Float32(half)))
        var angle = val * freq
        out[i] = fcos(angle)          # cos first
        out[half + i] = fsin(angle)   # sin second
    return out^


# DELTA 4: the embedder consumes `timestep/1000` (NOT raw sigma) — BaseAnimaSetup.py:137.
def _prepare_timestep(
    t_in: Float32, base: AnimaStackBase, ctx: DeviceContext
) raises -> _TEmb:
    var emb_l = _sinusoidal_host(t_in, D)
    var emb = Tensor.from_host(emb_l, [B, D], STDtype.F32, ctx)
    var h = linear(emb, base.te_lin1[], Optional[Tensor](None), ctx)
    var hidden = silu(h, ctx)
    var base_adaln = linear(hidden, base.te_lin2[], Optional[Tensor](None), ctx)
    var t_cond = rms_norm(emb, base.t_norm[], EPS, ctx)
    return _TEmb(t_cond.to_host(ctx), base_adaln.to_host(ctx))


# ── 3D-RoPE (T,H,W) NTK tables [B*S_IMG*H, Dh/2] ─────────────────────────────
# REAL Anima forward (matches anima_dit.build_anima_3d_rope + diffusers
# CosmosRotaryPosEmbed): the head_dim is split into 3 axis bands
#   dim_h = dim_w = full_d//6*2,  dim_t = full_d - 2*dim_h
# with per-axis NTK-scaled theta from rope_scale (t=1.0, h=4.0, w=4.0).
# Each image token's rotation depends on its (T, ih, iw) grid coordinate, NOT a
# flat linear index. The previous single-axis table (theta=10000 over the flat
# token index) was OT-UNFAITHFUL (cos≈0.71 vs the real 3-axis table); replaced.
# Output layout is identical to before: [B*S_IMG*H, Dh/2] F32, one row per
# (b, s, h) with the per-position angle replicated across the H heads, columns
# ordered [t-bins | h-bins | w-bins] so rope_halfsplit consumes column-i as the
# angle for pair i — same column contract as build_anima_3d_rope.
struct _Rope(Movable):
    var cos: Tensor
    var sin: Tensor

    def __init__(out self, var cos: Tensor, var sin: Tensor):
        self.cos = cos^
        self.sin = sin^


def _rope_tables(s_img: Int, ctx: DeviceContext) raises -> _Rope:
    var half = Dh // 2            # 64
    var full_d = Dh              # 128

    # patch grid: T=1, nH=nW=sqrt(s_img). For LATENT_HW=16, PS=2 -> nH=nW=8, S_IMG=64.
    var t_frames = 1
    var nh = LATENT_HW // PS
    var nw = LATENT_HW // PS
    if nh * nw != s_img:
        raise Error("rope grid mismatch: nh*nw=" + String(nh * nw)
                    + " != S_IMG=" + String(s_img))

    # Cosmos 3D axis split (matches anima_dit.build_anima_3d_rope:314-320).
    var dim_h = full_d // 6 * 2   # 42
    var dim_w = dim_h             # 42
    var dim_t = full_d - 2 * dim_h  # 44
    var bins_t = dim_t // 2       # 22
    var bins_h = dim_h // 2       # 21
    var bins_w = dim_w // 2       # 21
    # bins_t + bins_h + bins_w == half (22+21+21 == 64)

    # NTK-scaled thetas (rope_scale t=1.0, h=4.0, w=4.0; anima_dit:322-334).
    var base_theta: Float64 = 10000.0
    var h_exp = Float64(dim_h) / (Float64(dim_h) - 2.0)
    var w_exp = Float64(dim_w) / (Float64(dim_w) - 2.0)
    var h_ntk = fexp(flog(Float64(4.0)) * h_exp)
    var w_ntk = fexp(flog(Float64(4.0)) * w_exp)
    var t_ntk = Float64(1.0)      # 1.0^any = 1.0 (t_extra=1.0)
    var theta_h = Float64(base_theta * h_ntk)
    var theta_w = Float64(base_theta * w_ntk)
    var theta_t = Float64(base_theta * t_ntk)

    # Per-bin frequencies: freq = 1 / theta^(2i/dim) = exp(-log(theta)*(2i/dim)).
    var freqs_t = List[Float32]()
    for i in range(bins_t):
        var ev = Float64(2 * i) / Float64(dim_t)
        freqs_t.append(Float32(fexp(-flog(theta_t) * ev)))
    var freqs_h = List[Float32]()
    for i in range(bins_h):
        var ev = Float64(2 * i) / Float64(dim_h)
        freqs_h.append(Float32(fexp(-flog(theta_h) * ev)))
    var freqs_w = List[Float32]()
    for i in range(bins_w):
        var ev = Float64(2 * i) / Float64(dim_w)
        freqs_w.append(Float32(fexp(-flog(theta_w) * ev)))

    var cosl = List[Float32]()
    var sinl = List[Float32]()
    for _b in range(B):
        # position s runs t-major then ih then iw (T=1 here) — matches the
        # patchify token order [T, nH, nW] used by the forward.
        for tf in range(t_frames):
            for ih in range(nh):
                for iw in range(nw):
                    for _h in range(H):
                        for fi in range(bins_t):
                            var ang = Float32(tf) * freqs_t[fi]
                            cosl.append(fcos(ang))
                            sinl.append(fsin(ang))
                        for fi in range(bins_h):
                            var ang = Float32(ih) * freqs_h[fi]
                            cosl.append(fcos(ang))
                            sinl.append(fsin(ang))
                        for fi in range(bins_w):
                            var ang = Float32(iw) * freqs_w[fi]
                            cosl.append(fcos(ang))
                            sinl.append(fsin(ang))
    var cos = Tensor.from_host(cosl, [B * s_img * H, half], STDtype.F32, ctx)
    var sin = Tensor.from_host(sinl, [B * s_img * H, half], STDtype.F32, ctx)
    return _Rope(cos^, sin^)


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


# Sum |m|_1 of the AdamW first-moment buffers across the whole LoRA set — the
# faithful-resume evidence (zeroed-Adam resume => this is 0 at the segment start;
# a faithful resume => it equals the value saved at the checkpoint).
def _mom_absum(set: AnimaLoraSet) -> Float32:
    var s = Float32(0.0)
    for i in range(len(set.ad)):
        s += _absum(set.ad[i].ma) + _absum(set.ad[i].mb)
    return s


def _global_norm(grads: AnimaLoraGrads) -> Float32:
    var ss = Float32(0.0)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += grads.d_a[i][j] * grads.d_a[i][j]
        for j in range(len(grads.d_b[i])):
            ss += grads.d_b[i][j] * grads.d_b[i][j]
    return sqrt(ss)


def _clip(mut grads: AnimaLoraGrads, max_norm: Float32):
    var gn = _global_norm(grads)
    if gn <= max_norm or gn == 0.0:
        return
    var s = max_norm / gn
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s


# ── inverse of _patchify_out: target/velocity patches [B*N,64] -> channels-last
#    [B,Hd,Wd,C] (C fastest in the patch, matches _patchify_out layout) ──────────
def _unpatchify_out(
    p: List[Float32], Bd: Int, T: Int, Hd: Int, Wd: Int, Cd: Int
) -> List[Float32]:
    var pH = PS
    var pW = PS
    var nH = Hd // pH
    var nW = Wd // pW
    var pd = Cd * pH * pW
    var out = List[Float32]()
    out.reserve(Bd * Hd * Wd * Cd)
    for _ in range(Bd * Hd * Wd * Cd):
        out.append(Float32(0.0))
    for b in range(Bd):
        for t in range(T):
            for ih in range(nH):
                for iw in range(nW):
                    var pn = (t * nH + ih) * nW + iw
                    for ph in range(pH):
                        for pw in range(pW):
                            for c in range(Cd):
                                var od = (b * (T * nH * nW) + pn) * pd + (ph * pW * Cd + pw * Cd + c)
                                var hh = ih * pH + ph
                                var ww = iw * pW + pw
                                var dst = ((b * Hd + hh) * Wd + ww) * Cd + c
                                out[dst] = p[od]
    return out^


# ── build a "base" LoRA set: identical adapters but every B zeroed, so the LoRA
#    contribution scale·B·A == 0 and the forward reduces bit-exactly to the frozen
#    base DiT (the WITHOUT-LoRA pass). NEVER fuses/bakes — pure in-memory overlay. ─
def _zero_b_set(set: AnimaLoraSet) -> AnimaLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        var src = set.ad[i].copy()   # carries a/b + Adam moments
        for j in range(len(src.b)):
            src.b[j] = Float32(0.0)  # zero the up-projection -> scale·B·A == 0
        ad.append(src^)
    return AnimaLoraSet(ad^, set.num_blocks, set.rank)


# ── sample-shift denoise: a few-step Euler sample using the EXACT trainer forward (which is
#    the inference math at S_TXT=512, S_IMG=256). Same seed + context for both
#    passes; only the LoRA set differs (base = zeroed-B overlay, lora = trained).
#    Returns the final scaled-latent channels-last [B*Hd*Wd*C]. ──────────────────
def _anima_sample_shift(
    lora: AnimaLoraSet, n_steps: Int,
    scaled_bthwc_n: Int, Hd: Int, Wd: Int, Cd: Int,
    context: List[Float32], base: AnimaStackBase, st: SafeTensors,
    cos: Tensor, sin: Tensor,
    ctx: DeviceContext,
) raises -> List[Float32]:
    # fixed initial noise (sample seed independent of the training noise stream)
    var x = _host_noise(scaled_bthwc_n, UInt64(777))   # [B,Hd,Wd,C] channels-last
    var i = 0
    while i < n_steps:
        # linear sigma schedule 1->0 (anima_sampling.build_anima_sigma_schedule)
        var sigma = Float32(1.0) - Float32(i) / Float32(n_steps)
        var sigma_next = Float32(1.0) - Float32(i + 1) / Float32(n_steps)
        var dt = sigma_next - sigma
        var patches = _patchify_in(x, B, 1, Hd, Wd, Cd)
        # discrete ts for the t-embedder: ts/1000 == sigma-ish; OT passes ts/1000.
        var ts_idx = Int(sigma * Float32(NUM_TRAIN_TIMESTEPS))
        if ts_idx >= NUM_TRAIN_TIMESTEPS:
            ts_idx = NUM_TRAIN_TIMESTEPS - 1
        if ts_idx < 0:
            ts_idx = 0
        var temb = _prepare_timestep(Float32(ts_idx) / Float32(1000.0), base, ctx)
        var fwd = anima_stack_lora_forward_streamed[H, Dh, S_IMG, S_TXT](
            patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(), context.copy(),
            base, st, lora, cos, sin,
            B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
        )
        var v = _unpatchify_out(fwd.out, B, 1, Hd, Wd, Cd)   # velocity channels-last
        # direct-velocity Euler: x = x + dt*v
        for j in range(len(x)):
            x[j] = x[j] + dt * v[j]
        i += 1
    return x^


def _mean_abs_diff(a: List[Float32], b: List[Float32]) -> Float32:
    var s = Float64(0.0)
    for i in range(len(a)):
        var d = Float64(a[i] - b[i])
        s += d if d >= 0.0 else -d
    return Float32(s / Float64(len(a)))


def _cosine(a: List[Float32], b: List[Float32]) -> Float32:
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na <= 0.0 or nb <= 0.0:
        return Float32(0.0)
    return Float32(dot / ((na ** 0.5) * (nb ** 0.5)))


def _zeros_list(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _count_nonfinite(a: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(a)):
        var x = a[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


# ── GAP 2 directional metric: L2 distance between two latents (same layout) ───
# For genuine overfit learning the LoRA sample must land CLOSER to the cached
# training target than the base sample does. Both samples + the target are in the
# SAME scaled-latent channels-last [B*Hd*Wd*C] space.
def _l2_dist(a: List[Float32], b: List[Float32]) -> Float32:
    var ss = Float64(0.0)
    for i in range(len(a)):
        var d = Float64(a[i] - b[i])
        ss += d * d
    return Float32(ss ** 0.5)


# ── GAP 2 decode: scaled channels-last latent [B,Hd,Wd,C] -> RGB PNG. ─────────
# The samples (base_sample, lora_sample) and the cached target are in SCALED
# latent space (scale_latents applied). The decoder's internal unnormalize is
# z = z/inv_std + mean == z*std + mean == AnimaModel.unscale_latents, so feeding
# the SCALED latent straight in reproduces unscale -> decode (no separate unscale
# step needed; doing it here would double-apply). Convert channels-last
# [B,Hd,Wd,C] -> NCHW [1,16,Hd,Wd], decode, save PNG ([-1,1] SIGNED range).
def _decode_latent_to_png[LH: Int, LW: Int](
    scaled_bhwc: List[Float32], Hd: Int, Wd: Int, Cd: Int,
    dec: QwenImageVaeDecoder[LH, LW], out_png: String, ctx: DeviceContext,
) raises:
    # channels-last [B,Hd,Wd,C] -> NCHW [1,C,Hd,Wd]
    var nchw = List[Float32]()
    nchw.resize(B * Cd * Hd * Wd, Float32(0.0))
    for b in range(B):
        for h in range(Hd):
            for w in range(Wd):
                for c in range(Cd):
                    var src = ((b * Hd + h) * Wd + w) * Cd + c
                    var dst = ((b * Cd + c) * Hd + h) * Wd + w
                    nchw[dst] = scaled_bhwc[src]
    # The wan21 VAE weights + mean/inv_std are BF16; build the latent BF16 so the
    # decoder's internal div/add (z/inv_std + mean) dtype-matches (else elementwise
    # a/b dtype mismatch). The roundtrip parity feeds BF16 the same way.
    var lat = Tensor.from_host(nchw, [1, Cd, Hd, Wd], STDtype.BF16, ctx)
    var rgb = dec.decode_wan21_keys(lat, ctx)   # NCHW [1,3,8*LH,8*LW], [-1,1]
    save_png(rgb, out_png, ctx, ValueRange.SIGNED)


# ── STAGE 2 dataset: enumerate the 70-sample giger3 cache (sorted = reproducible
#    order), then per-step pick ONE file by a seeded index. Each file holds BOTH
#    `latent`[1,16,64,64] RAW and `context_cond`[1,512,1024] in the SAME file. ─
def _sort_strings(mut xs: List[String]):
    for i in range(1, len(xs)):
        var key = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > key:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = key


def _scan_dataset(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var fs = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            fs.append(dir + String("/") + raw[i])
    if len(fs) == 0:
        raise Error(String("_scan_dataset: no .safetensors in ") + dir)
    _sort_strings(fs)
    return fs^


# A cheap order-independent fingerprint of a host list (for the dataset-variety
# evidence: confirm the per-step latent/context actually differ file to file).
def _hash_list(v: List[Float32]) -> Float32:
    var s = Float64(0.0)
    for i in range(len(v)):
        s += Float64(v[i]) * Float64((i % 97) + 1)
    return Float32(s)


# Load ONE giger3 cache file -> (scaled_bthwc, context). The latent is RAW
# [1,16,64,64] (do NOT pre-scale on disk); scale_latents is applied HERE exactly
# as the single-sample path does (DELTA 2). The context_cond [1,512,1024] is
# loaded + zero-padded to S_TXT if shorter. Returns scaled[B*Hd*Wd*C] channels-last
# and context[B*S_TXT*JOINT].
struct _Sample(Movable):
    var scaled: List[Float32]
    var context: List[Float32]
    var lat_hash: Float32
    var ctx_hash: Float32

    def __init__(out self, var scaled: List[Float32], var context: List[Float32],
                 lat_hash: Float32, ctx_hash: Float32):
        self.scaled = scaled^
        self.context = context^
        self.lat_hash = lat_hash
        self.ctx_hash = ctx_hash


def _load_giger3_sample(
    path: String, Hd: Int, Wd: Int, Cd: Int,
    vae_mean: List[Float32], vae_std: List[Float32],
    ctx: DeviceContext,
) raises -> _Sample:
    var st = SafeTensors.open(path)
    var lat_info = st.tensor_info("latent")
    var lat_sh = lat_info.shape.copy()
    if len(lat_sh) != 4:
        raise Error("giger3 latent must be rank-4 [B,C,H,W]")
    var full_H = lat_sh[2]
    var full_W = lat_sh[3]
    var lat_full = _cache_f32(st, "latent", ctx).to_host(ctx)   # BCHW flat
    # channels-last raw [B,Hd,Wd,C] (tile/crop via modulo source index).
    var raw_bthwc = List[Float32]()
    raw_bthwc.reserve(B * Hd * Wd * Cd)
    for _ in range(B * Hd * Wd * Cd):
        raw_bthwc.append(Float32(0.0))
    for b in range(B):
        for h in range(Hd):
            for w in range(Wd):
                for c in range(Cd):
                    var sh = h % full_H
                    var sw = w % full_W
                    var src = ((b * Cd + c) * full_H + sh) * full_W + sw
                    var dst = ((b * Hd + h) * Wd + w) * Cd + c
                    raw_bthwc[dst] = lat_full[src]
    # DELTA 2: scale_latents = (raw - mean) * (1/std), per-channel.
    var scaled = List[Float32]()
    scaled.reserve(B * Hd * Wd * Cd)
    for _ in range(B * Hd * Wd * Cd):
        scaled.append(Float32(0.0))
    for b in range(B):
        for h in range(Hd):
            for w in range(Wd):
                for c in range(Cd):
                    var idx = ((b * Hd + h) * Wd + w) * Cd + c
                    scaled[idx] = (raw_bthwc[idx] - vae_mean[c]) * (Float32(1.0) / vae_std[c])
    var lat_hash = _hash_list(scaled)
    # context_cond [1,S,1024] -> zero-pad to [B,S_TXT,JOINT].
    var context_raw = _cache_f32(st, "context_cond", ctx).to_host(ctx)
    var ctx_n = len(context_raw)
    var context = List[Float32]()
    context.reserve(B * S_TXT * JOINT)
    for _ in range(B * S_TXT * JOINT):
        context.append(Float32(0.0))
    var src_tokens = ctx_n // (B * JOINT)
    var copy_tokens = src_tokens if src_tokens < S_TXT else S_TXT
    for b in range(B):
        for s in range(copy_tokens):
            for j in range(JOINT):
                context[(b * S_TXT + s) * JOINT + j] = context_raw[(b * src_tokens + s) * JOINT + j]
    var ctx_hash = _hash_list(context)
    return _Sample(scaled^, context^, lat_hash, ctx_hash)


def _parse_int(s: String) -> Int:
    var v = 0
    var bs = s.as_bytes()
    for i in range(len(bs)):
        if bs[i] >= 0x30 and bs[i] <= 0x39:
            v = v * 10 + Int(bs[i] - 0x30)
    return v


# Step-tagged checkpoint paths under output/giger3_train/.
def _giger3_ckpt_paths(step: Int) -> List[String]:
    var tag = String(GIGER3_TRAIN_DIR) + String("/giger3_lora_step") + String(step)
    var out = List[String]()
    out.append(tag + String(".safetensors"))          # 0: PEFT kohya keys
    out.append(tag + String("_otkeys.safetensors"))   # 1: OneTrainer diffusers keys
    out.append(tag + String(".opt.safetensors"))      # 2: Adam-state sidecar (A/B+m/v)
    return out^


def main() raises:
    var ctx = DeviceContext()
    # Schedule mode: "real" (default) = OT per-step logit-normal; "fixed" = fixed-σ
    # smoke (clean learning curve). Optional argv[2] overrides RUN_STEPS.
    var args = argv()
    var fixed_step = False
    if len(args) > 1 and String(args[1]) == String("fixed"):
        fixed_step = True
    var run_steps = RUN_STEPS
    if len(args) > 2:
        var v = 0
        var bs = String(args[2]).as_bytes()
        for i in range(len(bs)):
            v = v * 10 + Int(bs[i] - 0x30)
        if v > 0:
            run_steps = v
    # ── RESUME CADENCE: argv[3] = start_step (loop starts here), argv[4] =
    #    resume_lora_path (optional; if given, load that LoRA set BEFORE the loop). ─
    var start_step = 0
    if len(args) > 3:
        var v = 0
        var bs = String(args[3]).as_bytes()
        for i in range(len(bs)):
            v = v * 10 + Int(bs[i] - 0x30)
        start_step = v
    if start_step > run_steps:
        raise Error(String("start_step ") + String(start_step)
                    + " > run_steps " + String(run_steps))
    var resume_lora_path = String("")
    if len(args) > 4:
        var rp = String(args[4])
        if rp != String("-") and rp != String(""):
            resume_lora_path = rp^
    # ── STAGE 2 (giger3): argv[5] = dataset dir (or "-"/"smoke" for single-sample;
    #    "giger3" expands to the comptime DATASET_DIR). argv[6] = CKPT_EVERY. ─
    var dataset_dir = String("")
    if len(args) > 5:
        var dd = String(args[5])
        if dd == String("giger3"):
            dataset_dir = String(DATASET_DIR)
        elif dd != String("-") and dd != String("") and dd != String("smoke"):
            dataset_dir = dd^
    var ckpt_every = CKPT_EVERY_DEFAULT
    if len(args) > 6:
        var ce = _parse_int(String(args[6]))
        if ce > 0:
            ckpt_every = ce
    var dataset_mode = dataset_dir != String("")
    print("==== train_anima_ot — ANIMA LoRA OneTrainer-recipe training run ====")
    print("schedule:", "FIXED-sigma smoke" if fixed_step else "REAL OT logit-normal",
          " run_steps=", run_steps, " start_step=", start_step,
          " resume=", resume_lora_path if resume_lora_path != String("") else String("(none)"))
    print("dims: D=", D, " H=", H, " Dh=", Dh, " F=", F, " DEPTH=", ANIMA_DEPTH,
          " RANK=", RANK, " ALPHA=", ALPHA, " S_TXT=", S_TXT)
    print("lr=", LR, " N=", NUM_TRAIN_TIMESTEPS, " dist=LOGIT_NORMAL",
          " steps=", run_steps)
    # ── STAGE 2 dataset scan (sorted, once) ──
    var ds_files = List[String]()
    if dataset_mode:
        ds_files = _scan_dataset(dataset_dir)
        print("DATASET mode: dir=", dataset_dir, " files=", len(ds_files),
              " ckpt_every=", ckpt_every, " out=", GIGER3_TRAIN_DIR)
    else:
        print("SINGLE-SAMPLE mode (smoke): CACHE_DIR + CONTEXT_PATH")

    var cfg = anima()
    print("checkpoint:", cfg.checkpoint)
    var st = SafeTensors.open(cfg.checkpoint)
    verify_anima_stack_shapes(st, ANIMA_DEPTH)
    var base = load_anima_stack_base(st, ctx)
    print("base projections + t_embedder loaded (F32 resident)")

    # ── HOT PATH: all 28 blocks BF16-resident on device (≈3.7 GiB). The F32
    #    residual stream is preserved via linear's mixed_base path. LoRA A/B are
    #    uploaded to device ONCE per step (anima_lora_set_to_device). This is the
    #    Z-Image/Klein device-resident fast path — no per-block to_host/from_host. ─
    var blocks = List[AnimaBlockWeights]()
    for bi in range(ANIMA_DEPTH):
        blocks.append(load_anima_block_weights_bf16_normf32(st, bi, ctx))  # BF16 proj + F32 norms
    print("blocks resident (BF16 proj, F32 norms):", len(blocks), "x 20 weights")

    # ── load cached sample (latent + frozen context). In dataset mode the
    #    "sample 0" file is ds_files[0] (latent + context_cond live in ONE file);
    #    in smoke mode it's the single-sample CACHE_DIR/sample0 + separate context. ─
    var cache_path = String(ds_files[0]) if dataset_mode else (String(CACHE_DIR) + "/sample0.safetensors")
    print("cache:", cache_path)
    var cache = SafeTensors.open(cache_path)
    var lat_info = cache.tensor_info("latent")
    var lat_sh = lat_info.shape.copy()
    if len(lat_sh) != 4:
        raise Error("expected cached latent [B,C,H,W] (4D); got rank " + String(len(lat_sh)))
    var Cd = lat_sh[1]
    var full_H = lat_sh[2]
    var full_W = lat_sh[3]
    if Cd != C:
        raise Error("cached latent channels " + String(Cd) + " != " + String(C))
    var Hd = LATENT_HW
    var Wd = LATENT_HW
    # If the cached latent is SMALLER than the target grid (e.g. a 32x32 cache used
    # to run the real 512px/64x64 compute shape for a speed probe), tile it up via
    # modulo indexing; if larger, crop. The math/parity is independent of content.
    if full_H < LATENT_HW or full_W < LATENT_HW:
        print("latent [B,", Cd, full_H, full_W, "] tiled up ->", Hd, "x", Wd,
              " -> S_IMG =", S_IMG, " (speed-shape probe: cache < target grid)")
    else:
        print("latent [B,", Cd, full_H, full_W, "] cropped ->", Hd, "x", Wd,
              " -> S_IMG =", S_IMG)

    var lat_full = _cache_f32(cache, "latent", ctx).to_host(ctx)  # BCHW flat
    # channels-last raw latent [B,Hd,Wd,C] (tile/crop via modulo source index).
    var raw_bthwc = List[Float32]()
    raw_bthwc.reserve(B * Hd * Wd * Cd)
    for _ in range(B * Hd * Wd * Cd):
        raw_bthwc.append(Float32(0.0))
    for b in range(B):
        for h in range(Hd):
            for w in range(Wd):
                for c in range(Cd):
                    var sh = h % full_H
                    var sw = w % full_W
                    var src = ((b * Cd + c) * full_H + sh) * full_W + sw
                    var dst = ((b * Hd + h) * Wd + w) * Cd + c
                    raw_bthwc[dst] = lat_full[src]

    # ── DELTA 2: scale_latents per-channel BEFORE flow ──
    # scaled = (raw - mean) * (1/std), 16 ch (AnimaModel.py:233-236).
    var vae_mean = _vae_latents_mean()
    var vae_std = _vae_latents_std()
    var scaled_bthwc = List[Float32]()
    scaled_bthwc.reserve(B * Hd * Wd * Cd)
    for _ in range(B * Hd * Wd * Cd):
        scaled_bthwc.append(Float32(0.0))
    for b in range(B):
        for h in range(Hd):
            for w in range(Wd):
                for c in range(Cd):
                    var idx = ((b * Hd + h) * Wd + w) * Cd + c
                    var m = vae_mean[c]
                    var inv_s = Float32(1.0) / vae_std[c]
                    scaled_bthwc[idx] = (raw_bthwc[idx] - m) * inv_s
    # quick std receipt (≈0.96 healthy; ≈0.85 would flag a HWC/CHW scramble)
    var smean = Float32(0.0)
    for i in range(len(scaled_bthwc)):
        smean += scaled_bthwc[i]
    smean /= Float32(len(scaled_bthwc))
    var svar = Float32(0.0)
    for i in range(len(scaled_bthwc)):
        var d = scaled_bthwc[i] - smean
        svar += d * d
    svar /= Float32(len(scaled_bthwc))
    print("scaled latent: mean=", smean, " std=", sqrt(svar))

    # ── frozen LLM-adapter context [B,512,1024] (DELTA 1). In dataset mode it's
    #    `context_cond` from the SAME cache file; in smoke mode the CONTEXT_PATH sidecar. ─
    var context_src = String(cache_path) if dataset_mode else String(CONTEXT_PATH)
    print("context:", context_src, " (key=context_cond)" if dataset_mode else String(""))
    var ctx_st = SafeTensors.open(context_src)
    var context_raw = _cache_f32(ctx_st, "context_cond", ctx).to_host(ctx)
    var ctx_n = len(context_raw)
    # The captured sidecar may be [1,256,1024]; OT needs [1,512,1024]. If shorter,
    # zero-pad the extra query positions (the conditioner emits dense zeros for pad
    # positions — AnimaModel.py:229). If already 512, use as-is.
    var context = List[Float32]()
    context.reserve(B * S_TXT * JOINT)
    for _ in range(B * S_TXT * JOINT):
        context.append(Float32(0.0))
    var src_tokens = ctx_n // (B * JOINT)
    var copy_tokens = src_tokens if src_tokens < S_TXT else S_TXT
    for b in range(B):
        for s in range(copy_tokens):
            for j in range(JOINT):
                context[(b * S_TXT + s) * JOINT + j] = context_raw[(b * src_tokens + s) * JOINT + j]
    print("context src_tokens=", src_tokens, " -> [B,", S_TXT, JOINT,
          "] (zero-padded to 512 if needed)")

    # ── RoPE tables ──
    var ropes = _rope_tables(S_IMG, ctx)

    # ── build the LoRA set (B=0 init -> PEFT identity at step 0) ──
    var lora = build_anima_lora_set(ANIMA_DEPTH, D, JOINT, F, RANK, ALPHA)
    var n_adapters = ANIMA_DEPTH * ANIMA_SLOTS
    var b_init = Float32(0.0)
    for i in range(n_adapters):
        b_init += _absum(lora.ad[i].b)
    print("LoRA adapters:", n_adapters, "(10 slots x", ANIMA_DEPTH, "blocks)  |B|_1 init =", b_init)

    # ── RESUME: if a resume path is given, REPLACE the fresh set with the loaded
    #    state BEFORE the loop. FAITHFUL resume: a `.opt.safetensors` sidecar carries
    #    A/B + AdamW moments (ma/va/mb/vb) -> load_anima_lora_state RESTORES momentum
    #    (NOT zeroed). A plain PEFT path restores A/B only (momentum resets — the old
    #    fallback). The AdamW bias-correction step t is reconstructed from start_step
    #    (the loop passes t = step+1), continuous across the boundary. ─
    var faithful_resume = resume_lora_path.endswith(".opt.safetensors")
    if resume_lora_path != String(""):
        if faithful_resume:
            print("RESUME (FAITHFUL): loading LoRA + AdamW m/v from", resume_lora_path)
            lora = load_anima_lora_state(ANIMA_DEPTH, RANK, ALPHA, resume_lora_path, ctx)
        else:
            print("RESUME (A/B only): loading LoRA from", resume_lora_path,
                  " (Adam m/v NOT restored — momentum resets)")
            lora = load_anima_lora_resume(ANIMA_DEPTH, RANK, ALPHA, resume_lora_path, ctx)
    var b_loaded = Float32(0.0)
    for i in range(n_adapters):
        b_loaded += _absum(lora.ad[i].b)
    var m_loaded = _mom_absum(lora)
    print("CADENCE: LoRA-B |.|_1 at START of segment (post-load) =", b_loaded,
          " start_step=", start_step)
    print("RESUME: AdamW |m|_1 at START of segment (post-load) =", m_loaded,
          " (faithful_resume=", faithful_resume, ")")

    var n_lat = B * (C * S_IMG * PS * PS)

    # ── CADENCE decoder + sampler: build the VAE decoder ONCE so the cadence
    #    samples can be decoded to PNGs at start (step0) and at the end of the
    #    segment. mkdir the cadence output dir up front. ─
    _ = sys_system(String("mkdir -p ") + String(CADENCE_DIR))
    if dataset_mode:
        _ = sys_system(String("mkdir -p ") + String(GIGER3_TRAIN_DIR))
    var cad_dec = QwenImageVaeDecoder[LATENT_HW, LATENT_HW].load_wan21_keys(String(VAE_FILE), ctx)

    # INITIAL cadence sample: only when this is the very first segment (start_step==0).
    if start_step == 0:
        var s0 = _anima_sample_shift(
            lora, 4, n_lat, Hd, Wd, Cd,
            context.copy(), base, st, ropes.cos, ropes.sin, ctx,
        )
        var p0 = String(CADENCE_DIR) + "/sample_step0.png"
        _decode_latent_to_png[LATENT_HW, LATENT_HW](
            s0.copy(), Hd, Wd, Cd, cad_dec, p0, ctx)
        print("SAMPLE step=0 ->", p0, " nonfinite=", _count_nonfinite(s0),
              " |.|mean=", _mean_abs_diff(s0, _zeros_list(len(s0))))

    var t0 = perf_counter_ns()
    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var loss_hist = List[Float32]()

    for step in range(start_step, run_steps):
        var step_t0 = perf_counter_ns()
        # ── STAGE 2 DATASET: per-step pick ONE file by a SEEDED index
        #    (SEED + step) % N, load latent + context_cond from THAT file. The
        #    single-sample (smoke) path keeps the pre-loop scaled_bthwc/context. ─
        var cur_scaled = scaled_bthwc.copy()
        var cur_context = context.copy()
        if dataset_mode:
            var ds_idx = Int((SEED + UInt64(step)) % UInt64(len(ds_files)))
            var samp = _load_giger3_sample(
                ds_files[ds_idx], Hd, Wd, Cd, vae_mean, vae_std, ctx)
            cur_scaled = samp.scaled.copy()
            cur_context = samp.context.copy()
            var lat_hash = samp.lat_hash
            var ctx_hash = samp.ctx_hash
            print("DATA step=", step, " ds_idx=", ds_idx, "/", len(ds_files),
                  " file=", ds_files[ds_idx],
                  " lat_hash=", lat_hash, " ctx_hash=", ctx_hash)

        # ── DELTA 3: OT discrete timestep -> sigma = (ts+1)/N ──
        var ts = FIXED_TIMESTEP
        if not fixed_step:
            var ts_seed = SEED * UInt64(7919) + UInt64(step) * UInt64(2654435761)
            ts = _ot_timestep_discrete(ts_seed)
        var sigma = Float32(ts + 1) / Float32(NUM_TRAIN_TIMESTEPS)

        # ── flow-matching noise + target at LATENT level ──
        #   noisy  = noise*sigma + scaled*(1-sigma)
        #   target = noise - scaled            (DELTA 3, BaseAnimaSetup.py:143)
        var noise_seed = SEED * UInt64(104729)
        if not fixed_step:
            noise_seed += UInt64(step)
        var noise = _host_noise(n_lat, noise_seed)
        var noisy = List[Float32]()
        var target = List[Float32]()
        noisy.reserve(n_lat)
        target.reserve(n_lat)
        for i in range(n_lat):
            noisy.append(sigma * noise[i] + (Float32(1.0) - sigma) * cur_scaled[i])
            target.append(noise[i] - cur_scaled[i])

        var patches = _patchify_in(noisy, B, 1, Hd, Wd, Cd)
        var target_patches = _patchify_out(target, B, 1, Hd, Wd, Cd)

        # ── DELTA 4: t_embedder consumes timestep/1000 ──
        var t_in = Float32(ts) / Float32(1000.0)
        var temb = _prepare_timestep(t_in, base, ctx)

        # ── upload LoRA set to device ONCE this step (BF16) ──
        var lora_dev = anima_lora_set_to_device(lora, STDtype.BF16, ctx)

        var prof = (step == 2)
        var t_fwd0 = perf_counter_ns()
        # ── forward (device-resident 28-block) at S_TXT=512 / S_IMG=1024 ──
        var fwd = anima_stack_lora_forward_device_resident[H, Dh, S_IMG, S_TXT](
            patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(), cur_context.copy(),
            base, blocks, lora_dev, ropes.cos, ropes.sin,
            B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
        )
        if prof:
            ctx.synchronize()
            print("[PROF] forward =", Float32(Float64(perf_counter_ns() - t_fwd0) / 1.0e9), "s")

        # ── DELTA 4/5: unmasked MSE + d_out (dMSE/dpred = 2/N*(pred-target)) ──
        var npred = len(fwd.out)
        var sse = Float32(0.0)
        var d_out = List[Float32]()
        d_out.reserve(npred)
        var inv_n = Float32(2.0) / Float32(npred)
        for i in range(npred):
            var diff = fwd.out[i] - target_patches[i]
            sse += diff * diff
            d_out.append(inv_n * diff)
        var loss = sse / Float32(npred)

        # ── backward (device-resident); per-block trace on first step, fwd/bwd split on step 2 ──
        var t_bwd0 = perf_counter_ns()
        var grads = anima_stack_lora_backward_device_resident[H, Dh, S_IMG, S_TXT](
            d_out, patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(), cur_context.copy(),
            base, blocks, lora_dev, ropes.cos, ropes.sin, fwd,
            B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx, step == 1,
        )
        if prof:
            ctx.synchronize()
            print("[PROF] backward(reads saved activations; no recompute) =",
                  Float32(Float64(perf_counter_ns() - t_bwd0) / 1.0e9), "s")

        # ── global-norm clip then AdamW over all adapters ──
        var t_opt0 = perf_counter_ns()
        var gn_before = _global_norm(grads)
        _clip(grads, CLIP_NORM)
        anima_lora_adamw_step(lora, grads, step + 1, LR, ctx)
        if prof:
            ctx.synchronize()
            print("[PROF] optimizer(global_norm+clip+host AdamW) =",
                  Float32(Float64(perf_counter_ns() - t_opt0) / 1.0e9),
                  "s  (lever-2 target: device-resident LoRA optimizer)")

        var b_after = Float32(0.0)
        var b_nonzero = 0
        for i in range(n_adapters):
            var s = _absum(lora.ad[i].b)
            b_after += s
            if s > 0.0:
                b_nonzero += 1
        var da = Float32(0.0)
        var db = Float32(0.0)
        for i in range(n_adapters):
            da += _absum(grads.d_a[i])
            db += _absum(grads.d_b[i])

        if step == start_step:
            first_loss = loss
        last_loss = loss
        loss_hist.append(loss)

        var step_now = perf_counter_ns()
        var step_secs = Float64(step_now - step_t0) / 1.0e9
        var elapsed_secs = Float64(step_now - t0) / 1.0e9
        var samples_per_epoch = 1
        if dataset_mode:
            samples_per_epoch = len(ds_files)
        print_trainer_progress(
            String("Anima-lora"), step + 1, run_steps, samples_per_epoch,
            loss, Float64(gn_before), step_secs, 0.0, elapsed_secs,
        )
        if grads.nonfinite_lora_grads != 0:
            print("[Anima-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)

        # ── CHECKPOINT EVERY N (dataset mode): at (step+1)%ckpt_every==0 save the
        #    LoRA (PEFT kohya + OneTrainer diffusers keys) AND the Adam-state sidecar
        #    (.opt.safetensors: A/B + m/v) to step-tagged paths under giger3_train/.
        #    The tag is the COMPLETED step count (step+1) so resume passes start_step
        #    = tag and t = step+1 stays continuous. ─
        if dataset_mode and ((step + 1) % ckpt_every == 0):
            var ckp = _giger3_ckpt_paths(step + 1)
            var _nk = save_anima_lora(lora, ckp[0], ctx)
            var _no = save_anima_lora_ot(lora, ALPHA, ckp[1], ctx)
            var _ns = save_anima_lora_state(lora, ckp[2], ctx)
            print("CKPT step=", step + 1, " saved PEFT=", ckp[0],
                  " OT=", ckp[1], " opt=", ckp[2],
                  " (AdamW |m|_1=", _mom_absum(lora), ")")

    var t1 = perf_counter_ns()
    var secs = Float64(t1 - t0) / 1.0e9

    # ── END-OF-SEGMENT CADENCE: save a step-tagged LoRA checkpoint so the next
    #    segment can resume from it, sample+decode a PNG, and print LoRA-B |.|_1 at
    #    END so resume-continuity is visible (next segment's START |.|_1 must EQUAL
    #    this END value). The step tag is run_steps (the last step trained this seg). ─
    var seg_steps_run = run_steps - start_step
    var b_end = Float32(0.0)
    for i in range(n_adapters):
        b_end += _absum(lora.ad[i].b)
    var seg_lora_path = String(CADENCE_DIR) + "/anima_lora_step" + String(run_steps) + ".safetensors"
    var n_seg = save_anima_lora(lora, seg_lora_path, ctx)
    print("CADENCE: saved", n_seg, "LoRA pairs (step-tagged) to", seg_lora_path)
    print("CADENCE: LoRA-B |.|_1 at END of segment (step", run_steps, ") =", b_end)
    # ── FINAL giger3 checkpoint (dataset mode): PEFT + OT + Adam-state sidecar at
    #    the END of the segment (tag = run_steps). Always emitted so the next
    #    segment can resume FAITHFULLY from <...>_step<run_steps>.opt.safetensors. ─
    if dataset_mode:
        var fckp = _giger3_ckpt_paths(run_steps)
        var _fk = save_anima_lora(lora, fckp[0], ctx)
        var _fo = save_anima_lora_ot(lora, ALPHA, fckp[1], ctx)
        var _fs = save_anima_lora_state(lora, fckp[2], ctx)
        print("CKPT (final) step=", run_steps, " saved PEFT=", fckp[0],
              " OT=", fckp[1], " opt=", fckp[2],
              " (AdamW |m|_1=", _mom_absum(lora), ")")
    if seg_steps_run > 0:
        var seg_sample = _anima_sample_shift(
            lora, 4, n_lat, Hd, Wd, Cd,
            context.copy(), base, st, ropes.cos, ropes.sin, ctx,
        )
        var seg_png = String(CADENCE_DIR) + "/sample_step" + String(run_steps) + ".png"
        _decode_latent_to_png[LATENT_HW, LATENT_HW](
            seg_sample.copy(), Hd, Wd, Cd, cad_dec, seg_png, ctx)
        print("SAMPLE step=", run_steps, " ->", seg_png,
              " nonfinite=", _count_nonfinite(seg_sample),
              " |.|mean=", _mean_abs_diff(seg_sample, _zeros_list(len(seg_sample))))

    print("")
    print("---- training summary ----")
    print("steps:", seg_steps_run, " (", start_step, "->", run_steps,
          ")  wall:", secs, "s")
    if seg_steps_run > 0:
        print("  (", secs / Float64(seg_steps_run), "s/step)")
    print("loss first =", first_loss, "  last =", last_loss,
          "  delta =", last_loss - first_loss)
    # REAL random-σ schedule: per-step loss bounces with σ, so compare first-half
    # vs second-half MEAN (the σ-noise averages out -> the true learning trend).
    var nh = len(loss_hist)
    var half = nh // 2
    if half < 1:
        half = 1
    var first_half_mean = Float32(0.0)
    for i in range(half):
        first_half_mean += loss_hist[i]
    first_half_mean /= Float32(half)
    var second_half_mean = Float32(0.0)
    var cnt2 = 0
    for i in range(half, nh):
        second_half_mean += loss_hist[i]
        cnt2 += 1
    if cnt2 > 0:
        second_half_mean /= Float32(cnt2)
    print("loss first-half mean =", first_half_mean,
          "  second-half mean =", second_half_mean,
          "  trend delta =", second_half_mean - first_half_mean)

    # DATASET mode: the in-loop cadence already saved step-tagged checkpoints
    # (+ Adam .opt sidecar). Skip the single-sample final save + the 512 sample-shift
    # smoke verdict (the giger3 run samples via the 1024 LoRA sampler, stage 3).
    if dataset_mode:
        print("dataset run complete — checkpoints under", GIGER3_TRAIN_DIR)
        return

    # ── DELTA 5: OT diffusers-keyed save (OT-loadable) + kohya save (cheap) ──
    var n_kohya = save_anima_lora(lora, String(LORA_OUT), ctx)
    print("saved", n_kohya, "LoRA adapter pairs (kohya keys) to", LORA_OUT)
    var n_ot = save_anima_lora_ot(lora, ALPHA, String(LORA_OUT_OT), ctx)
    print("saved", n_ot, "LoRA adapters (OneTrainer diffusers keys) to", LORA_OUT_OT)

    var b_final = Float32(0.0)
    var b_nz = 0
    for i in range(n_adapters):
        var s = _absum(lora.ad[i].b)
        b_final += s
        if s > 0.0:
            b_nz += 1
    var grew = (b_init == Float32(0.0)) and (b_final > Float32(0.0)) and (b_nz == n_adapters)
    # fixed-σ: clean monotone first<last. real σ: use the half-window trend (the
    # per-step σ-bounce averages out over the run).
    var loss_down = (last_loss < first_loss) if fixed_step else (second_half_mean < first_half_mean)

    # ── LEARNING VERDICT: SAMPLE-SHIFT GATE (Chunk D wires 3+4) ───────────────
    # Sample TWICE from the same seed + frozen context: once with the trained LoRA
    # OVERLAID (additive scale·B·A at the inference linears via lora_block), once
    # with a zeroed-B overlay (== the frozen base DiT). NEVER fuses into the saved
    # weights — pure in-memory overlay. The shift between the two final latents is
    # the quantified sample-shift metric (must be non-trivial + finite, not blown up).
    print("")
    print("---- sample-shift (base vs LoRA-overlay) ----")
    comptime SAMPLE_STEPS = 4
    print("sample steps:", SAMPLE_STEPS, " S_IMG=", S_IMG, " (same seed+context both passes)")
    var base_set = _zero_b_set(lora)
    var base_sample = _anima_sample_shift(
        base_set, SAMPLE_STEPS, n_lat, Hd, Wd, Cd,
        context.copy(), base, st, ropes.cos, ropes.sin, ctx,
    )
    var lora_sample = _anima_sample_shift(
        lora, SAMPLE_STEPS, n_lat, Hd, Wd, Cd,
        context.copy(), base, st, ropes.cos, ropes.sin, ctx,
    )
    var base_nf = _count_nonfinite(base_sample)
    var lora_nf = _count_nonfinite(lora_sample)
    var shift_mad = _mean_abs_diff(base_sample, lora_sample)
    var shift_cos = _cosine(base_sample, lora_sample)
    var base_mad0 = _mean_abs_diff(base_sample, _zeros_list(len(base_sample)))
    var lora_mad0 = _mean_abs_diff(lora_sample, _zeros_list(len(lora_sample)))
    print("base sample |.|mean=", base_mad0, " nonfinite=", base_nf)
    print("lora sample |.|mean=", lora_mad0, " nonfinite=", lora_nf)
    print("SAMPLE-SHIFT mean_abs_diff(base,lora)=", shift_mad,
          " cosine(base,lora)=", shift_cos)
    var sample_finite = (base_nf == 0) and (lora_nf == 0)
    # non-trivial: the LoRA actually moved the sample; degenerate guard: cosine<0.99999
    var sample_shifted = (shift_mad > Float32(1e-5)) and sample_finite and (shift_cos < Float32(0.99999))

    # ── GAP 2: DIRECTIONAL learning proof ─────────────────────────────────────
    # The cached training target is `scaled_bthwc` (the real VAE latent in scaled
    # space). Genuine overfit learning => the LoRA sample lands CLOSER to that
    # target than the base sample (lora_dist < base_dist). This is the metric
    # Tenet 4 requires before any "learns" claim — sample-shift alone only proves
    # the LoRA CHANGED the output, not that it moved TOWARD the target.
    print("")
    print("---- DIRECTIONAL learning metric (lora vs base, distance to target) ----")
    var base_dist = _l2_dist(base_sample, scaled_bthwc)
    var lora_dist = _l2_dist(lora_sample, scaled_bthwc)
    print("target = cached scaled training latent (|.|mean=",
          _mean_abs_diff(scaled_bthwc, _zeros_list(len(scaled_bthwc))), ")")
    print("L2(base_sample -> target) =", base_dist)
    print("L2(lora_sample -> target) =", lora_dist)
    var directional_pass = (lora_dist < base_dist) and sample_finite
    print("DIRECTIONAL: lora_closer_than_base =", directional_pass,
          " (delta = base - lora =", base_dist - lora_dist, ")")

    # ── GAP 2: decode base/lora/target latents -> 3 PNGs (visual inspection) ───
    print("")
    print("---- decode base/lora/target latents -> PNG ----")
    var dec = QwenImageVaeDecoder[LATENT_HW, LATENT_HW].load_wan21_keys(String(VAE_FILE), ctx)
    _ = sys_system(String("mkdir -p ") + String(GAP2_DIR))
    _decode_latent_to_png[LATENT_HW, LATENT_HW](
        base_sample.copy(), Hd, Wd, Cd, dec, String(GAP2_DIR) + "/base_sample.png", ctx)
    _decode_latent_to_png[LATENT_HW, LATENT_HW](
        lora_sample.copy(), Hd, Wd, Cd, dec, String(GAP2_DIR) + "/lora_sample.png", ctx)
    _decode_latent_to_png[LATENT_HW, LATENT_HW](
        scaled_bthwc.copy(), Hd, Wd, Cd, dec, String(GAP2_DIR) + "/target.png", ctx)
    print("wrote", String(GAP2_DIR) + "/base_sample.png")
    print("wrote", String(GAP2_DIR) + "/lora_sample.png")
    print("wrote", String(GAP2_DIR) + "/target.png")

    print("")
    # Tenet 4: a LEARNING PASS requires loss down + LoRA grew + sample shifted +
    # the DIRECTIONAL metric (lora closer to target than base). No "learns" claim
    # without the directional number holding.
    var verdict_pass = grew and loss_down and sample_shifted and directional_pass
    if verdict_pass:
        print("VERDICT: LEARNING PASS — loss dropped (", first_loss, "->", last_loss,
              "), LoRA-B grew 0->nonzero (", b_nz, "/", n_adapters,
              "), sample SHIFTED (mad=", shift_mad, " cos=", shift_cos,
              "), AND lora CLOSER to target (lora_dist=", lora_dist,
              " < base_dist=", base_dist, ")")
    else:
        print("VERDICT: review — loss_down=", loss_down, " B_grew=", grew,
              " sample_shifted=", sample_shifted,
              " directional(lora<base)=", directional_pass,
              " (shift_mad=", shift_mad, " cos=", shift_cos,
              " base_dist=", base_dist, " lora_dist=", lora_dist,
              " finite=", sample_finite, ")")
