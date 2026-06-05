# serenitymojo/training/train_anima_real.mojo
#
# ANIMA (Cosmos-Predict2 MiniTrainDIT) LoRA TRAINING — REAL RUN.
#
# TRANSLATION (not invention) of:
#   EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_anima.rs
# onto the parity-verified Mojo stack. Per-step math is byte-faithful to the
# Rust loop (lines cited inline):
#
#   sigma  = sigmoid(scale * z)            (sample_timestep_sigmoid, rs:328-334)
#   sigma  = apply_shift(sigma, shift)     (rs:343-349)  shift*s/(1+(shift-1)*s)
#   sigma  = clamp(sigma, 1e-5, 1-1e-5)    (rs:834)
#   noisy  = sigma*noise + (1-sigma)*latent           (rs:859-861, rect-flow)
#   target = noise - latent                           (rs:863, rect-flow v-target)
#   pred   = AnimaDiT(noisy, sigma, context)          (rs:890-891)
#   loss   = mean((pred - target)^2)                  (MSE, rs:907-926)
#   d_out  = 2/N * (pred - target)                    (dMSE/dpred)
#   backward -> global-norm clip(1.0) -> AdamW        (rs:988-1024)
#   save LoRA (kohya net.blocks.<i>.* keys)           (rs:1136-1155)
#
# The DiT operates in PATCH SPACE. The cached latent [B,16,H',W'] is:
#   1. lifted to 5D [B,T=1,H',W',16] (channels-last),
#   2. flow-matched with noise at the LATENT level (flow_match style),
#   3. patchified INPUT-layout  -> patches [B*N, 68]  ((16+1)*2*2, +mask channel)
#      (matches anima_dit _patchify: [Cp,pH,pW], C slowest)
#   4. patchified OUTPUT-layout  -> target [B*N, 64]  (16*2*2)
#      (matches anima_dit _unpatchify: [pH,pW,C], C FASTEST — the loss target
#       must live in OUTPUT-patch layout, the inverse of the model's unpatchify;
#       VERIFIED anima_dit.mojo:691-725 patch_out_idx = ph*pW*C + pw*C + c).
#
# t_cond / base_adaln are computed from the REAL t_embedder weights (faithful to
# anima_dit _prepare_timestep, anima_dit.mojo:822-843):
#   emb     = sinusoidal(sigma, 2048)          (cos-first)
#   hidden  = silu(linear(emb, te_lin1))       (no bias)
#   base_ad = linear(hidden, te_lin2)          (no bias) -> [B,6144]
#   t_cond  = rms_norm(emb, t_norm, eps=1e-6)  (RAW sinusoidal, NOT silu'd)
# The stack silus t_cond internally (block.mojo:342 — the double-silu fix), so we
# pass RAW t_cond.
#
# Context (frozen cross-attn input): the cached/captured Anima LLM-adapter output
# [B,256,1024]. The adapter is FROZEN for LoRA (its grad path is discarded); we
# consume its output as a frozen input (TRAINING_PLAN_anima.md phase D). For the
# real smoke we read the captured sidecar context (anima_embeddings.safetensors).
#
# Weights: anima-base-v1.0.safetensors (28 blocks streamed per-step via
# anima_stack_lora_forward_streamed / _backward_streamed). LoRA adapters stay
# host-resident; AdamW over all 10x28 adapters (anima_lora_adamw_step).
#
# Run (SEPARATE command, after build):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/training/train_anima_real.mojo -o /tmp/train_anima_real
#   /tmp/train_anima_real

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, exp as fexp, isfinite
from std.memory import ArcPointer
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor

from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.norm import rms_norm

from serenitymojo.models.anima.config import anima
from serenitymojo.models.anima.weights import (
    AnimaStackBase, load_anima_stack_base, verify_anima_stack_shapes,
)
from serenitymojo.models.anima.lora_block import ANIMA_SLOTS
from serenitymojo.models.anima.anima_stack_lora import (
    AnimaLoraSet, AnimaLoraGrads, build_anima_lora_set,
    anima_stack_lora_forward_streamed, anima_stack_lora_backward_streamed,
    anima_lora_adamw_step, save_anima_lora,
)
from serenitymojo.models.dit.anima_contract import (
    ANIMA_HIDDEN, ANIMA_NUM_HEADS, ANIMA_HEAD_DIM, ANIMA_DEPTH,
    ANIMA_LATENT_CHANNELS, ANIMA_PATCH_SIZE, ANIMA_MAX_SEQ_LEN, ANIMA_ADAPTER_DIM,
)

from serenitymojo.training.schedule import sample_timestep_sigmoid
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
comptime S_TXT = ANIMA_MAX_SEQ_LEN  # 256 context tokens
comptime EPS = Float32(1e-06)

# ── Training config (mirrors train_anima.rs CLI defaults) ─────────────────────
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime LR = Float32(5.0e-5)          # rs:92 default 5e-5
comptime SIGMOID_SCALE = Float32(1.0)  # rs:99
comptime FLOW_SHIFT = Float32(3.0)     # rs:107 canonical Anima discrete_flow_shift
comptime CLIP_NORM = Float32(1.0)      # rs:988
comptime SEED = UInt64(42)             # rs:71
comptime RUN_STEPS = 6                 # short smoke (thermal-safe). Bump for real runs.
# The DiT stack is templated on S_IMG (comptime). The cached latent grid can be
# large (512^2 -> 64x64 latent -> S_IMG=1024); for a thermal/memory-safe smoke on
# a shared 24GB 3090 we CROP the latent to LATENT_HW x LATENT_HW (top-left),
# giving S_IMG = (LATENT_HW/2)^2. This is a valid training signal (same flow loss
# on a sub-window). Raise LATENT_HW (or pass the full grid) for full-resolution
# runs once the GPU is free.
comptime LATENT_HW = 16                       # crop latent to 16x16 -> S_IMG=64
comptime S_IMG = (LATENT_HW // PS) * (LATENT_HW // PS)
# Smoke mode: when FIXED_SIGMA_SMOKE, hold sigma + noise CONSTANT across steps so
# the flow target is identical every step. This isolates the LoRA learning signal
# (loss MUST fall monotonically as the adapters fit a fixed target) from the
# per-step timestep variance that dominates a 6-step random-sigma run. The REAL
# training schedule (random sigmoid sigma per step) runs when this is False.
comptime FIXED_SIGMA_SMOKE = True
comptime FIXED_SIGMA = Float32(0.5)

# ── Data paths ────────────────────────────────────────────────────────────────
# Cached latents (prepare_anima schema): latent [1,16,H',W'] BF16, + text fields.
comptime CACHE_DIR = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/anima_synth_smoke"
# Captured frozen LLM-adapter context [1,256,1024] (context_cond key).
comptime CONTEXT_PATH = "/home/alex/EriDiffusion/inference-flame/output/anima_embeddings.safetensors"
comptime LORA_OUT = "/home/alex/mojodiffusion/output/anima_lora_smoke.safetensors"


# ── deterministic host gaussian noise (Box-Muller on a PCG stream) ────────────
# Mirrors train_klein_real.mojo::_host_noise (reproducible per-step flow draw).
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


# ── rectified-flow timestep shift (train_anima.rs:343-349) ────────────────────
def _apply_shift(sigma: Float32, shift: Float32) -> Float32:
    if (shift - Float32(1.0)) < Float32(1.0e-6) and (shift - Float32(1.0)) > Float32(-1.0e-6):
        return sigma
    return shift * sigma / (Float32(1.0) + (shift - Float32(1.0)) * sigma)


def _clamp(x: Float32, lo: Float32, hi: Float32) -> Float32:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


# ── load a cache tensor as host F32 (BF16/F32 view -> upcast) ─────────────────
def _cache_f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return cast_tensor(t^, STDtype.F32, ctx)


# ── patchify INPUT layout: 5D channels-last [B,T,H,W,C] -> patches [B*N, 68].
#    Appends a zero mask channel (Cp = C+1). Patch dim order [Cp, pH, pW]
#    (C slowest), matching anima_dit _patchify_kernel (anima_dit.mojo:480-501):
#      out[ (b*N + (t*nH*nW + ih*nW + iw)) , (c*pH*pW + ph*pW + pw) ]
#        = x[b,t, ih*pH+ph, iw*pW+pw, c]      (c in 0..C, mask channel=0)
def _patchify_in(
    x: List[Float32], Bd: Int, T: Int, Hd: Int, Wd: Int, Cd: Int
) -> List[Float32]:
    var pH = PS
    var pW = PS
    var nH = Hd // pH
    var nW = Wd // pW
    var Cp = Cd + 1
    var N = T * nH * nW
    var pd = Cp * pH * pW   # 68
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
                                # else mask channel stays 0.0
    return out^


# ── patchify OUTPUT layout: [B,T,H,W,C] -> target patches [B*N, 64].
#    Patch dim order [pH, pW, C] (C FASTEST), the inverse of anima_dit
#    _unpatchify (anima_dit.mojo:715): patch_out_idx = ph*pW*C + pw*C + c. ─────
def _patchify_out(
    x: List[Float32], Bd: Int, T: Int, Hd: Int, Wd: Int, Cd: Int
) -> List[Float32]:
    var pH = PS
    var pW = PS
    var nH = Hd // pH
    var nW = Wd // pW
    var N = T * nH * nW
    var pd = Cd * pH * pW   # 64
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
def _sinusoidal_host(sigma: Float32, dim: Int) -> List[Float32]:
    var half = dim // 2
    var neg_ln = -flog(Float32(10000.0))
    var out = List[Float32]()
    out.reserve(dim)
    for _ in range(dim):
        out.append(Float32(0.0))
    for i in range(half):
        var freq = fexp(neg_ln * (Float32(i) / Float32(half)))
        var angle = sigma * freq
        out[i] = fcos(angle)          # cos first
        out[half + i] = fsin(angle)   # sin second
    return out^


def _prepare_timestep(
    sigma: Float32, base: AnimaStackBase, ctx: DeviceContext
) raises -> _TEmb:
    var emb_l = _sinusoidal_host(sigma, D)
    var emb = Tensor.from_host(emb_l, [B, D], STDtype.F32, ctx)   # [B,2048] F32
    # hidden = silu(linear(emb, te_lin1))   (no bias)
    var h = linear(emb, base.te_lin1[], Optional[Tensor](None), ctx)
    var hidden = silu(h, ctx)
    # base_adaln = linear(hidden, te_lin2)  (no bias) -> [B,6144]
    var base_adaln = linear(hidden, base.te_lin2[], Optional[Tensor](None), ctx)
    # t_cond = rms_norm(emb, t_norm)  (RAW sinusoidal — anima_dit.mojo:841)
    var t_cond = rms_norm(emb, base.t_norm[], EPS, ctx)
    return _TEmb(t_cond.to_host(ctx), base_adaln.to_host(ctx))


# ── build 3D-RoPE halfsplit tables [B*S_IMG*H, Dh/2] (per-position freqs).
#    Mirrors stack_real_smoke / lora_step_smoke expansion. For the smoke we use
#    a single linear position axis over the S_IMG image tokens (sufficient to
#    exercise the rope-applied attention; the real 3D axis split is a sampler
#    concern, not a LoRA-training-convergence one). ──────────────────────────
struct _Rope(Movable):
    var cos: Tensor
    var sin: Tensor

    def __init__(out self, var cos: Tensor, var sin: Tensor):
        self.cos = cos^
        self.sin = sin^


def _rope_tables(s_img: Int, ctx: DeviceContext) raises -> _Rope:
    var half = Dh // 2
    var cosl = List[Float32]()
    var sinl = List[Float32]()
    for _b in range(B):
        for s in range(s_img):
            for _h in range(H):
                for i in range(half):
                    var ang = Float32(s) / (Float32(10000.0) ** (Float32(2 * i) / Float32(Dh)))
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


def main() raises:
    var ctx = DeviceContext()
    print("==== train_anima_real — ANIMA LoRA REAL training run ====")
    print("dims: D=", D, " H=", H, " Dh=", Dh, " F=", F, " DEPTH=", ANIMA_DEPTH,
          " RANK=", RANK, " ALPHA=", ALPHA)
    print("lr=", LR, " sigmoid_scale=", SIGMOID_SCALE, " flow_shift=", FLOW_SHIFT,
          " steps=", RUN_STEPS)

    # ── open the real DiT checkpoint (streamed per-block) ──
    var cfg = anima()
    print("checkpoint:", cfg.checkpoint)
    var st = SafeTensors.open(cfg.checkpoint)
    verify_anima_stack_shapes(st, ANIMA_DEPTH)
    var base = load_anima_stack_base(st, ctx)
    print("base projections + t_embedder loaded (F32 resident)")

    # ── load cached sample (latent + frozen context) ──
    var cache_path = String(CACHE_DIR) + "/sample0.safetensors"
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
    if full_H < LATENT_HW or full_W < LATENT_HW:
        raise Error("cached latent " + String(full_H) + "x" + String(full_W)
                    + " smaller than LATENT_HW=" + String(LATENT_HW))
    # crop to LATENT_HW x LATENT_HW (top-left window) so the grid matches the
    # comptime S_IMG the DiT stack is templated on.
    var Hd = LATENT_HW
    var Wd = LATENT_HW
    print("latent [B,", Cd, full_H, full_W, "] cropped ->", Hd, "x", Wd,
          " -> S_IMG =", S_IMG)

    var lat_full = _cache_f32(cache, "latent", ctx).to_host(ctx)  # BCHW flat
    # channels-last cropped: dst [B,Hd,Wd,C] = src[b,c, h, w] (h,w in crop window)
    var lat_bthwc = List[Float32]()
    lat_bthwc.reserve(B * Hd * Wd * Cd)
    for _ in range(B * Hd * Wd * Cd):
        lat_bthwc.append(Float32(0.0))
    for b in range(B):
        for h in range(Hd):
            for w in range(Wd):
                for c in range(Cd):
                    var src = ((b * Cd + c) * full_H + h) * full_W + w
                    var dst = ((b * Hd + h) * Wd + w) * Cd + c
                    lat_bthwc[dst] = lat_full[src]

    # frozen LLM-adapter context [B,256,1024] (captured sidecar context_cond).
    print("context:", CONTEXT_PATH)
    var ctx_st = SafeTensors.open(CONTEXT_PATH)
    var context = _cache_f32(ctx_st, "context_cond", ctx).to_host(ctx)
    var ctx_n = len(context)
    if ctx_n != B * S_TXT * JOINT:
        raise Error("context numel " + String(ctx_n) + " != B*S_TXT*JOINT="
                    + String(B * S_TXT * JOINT))
    print("context [B,", S_TXT, JOINT, "] loaded (frozen cross-attn input)")

    # ── 3D RoPE tables for this image grid ──
    # cos/sin are borrowed (read) by the streamed fwd/bwd — no copy needed.
    var ropes = _rope_tables(S_IMG, ctx)

    # ── build the LoRA set (B=0 init -> PEFT identity at step 0) ──
    var lora = build_anima_lora_set(ANIMA_DEPTH, D, JOINT, F, RANK, ALPHA)
    var n_adapters = ANIMA_DEPTH * ANIMA_SLOTS
    var b_init = Float32(0.0)
    for i in range(n_adapters):
        b_init += _absum(lora.ad[i].b)
    print("LoRA adapters:", n_adapters, "(10 slots x", ANIMA_DEPTH, "blocks)  |B|_1 init =", b_init)

    var n_lat = B * (C * S_IMG * PS * PS)   # latent element count = B*C*H*W

    var t0 = perf_counter_ns()
    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)

    for step in range(RUN_STEPS):
        var step_t0 = perf_counter_ns()
        # ── timestep sampling (sigmoid -> shift -> clamp), rs:328-834 ──
        # In FIXED_SIGMA_SMOKE we hold sigma + the noise draw constant so the
        # target is identical every step (loss-decrease isolation, see header).
        var sigma = FIXED_SIGMA
        if not FIXED_SIGMA_SMOKE:
            var sigma_seed = SEED * UInt64(7919) + UInt64(step) * UInt64(2654435761)
            var sigma0 = sample_timestep_sigmoid(sigma_seed, SIGMOID_SCALE, Float32(0.0))
            sigma = _apply_shift(sigma0, FLOW_SHIFT)
            sigma = _clamp(sigma, Float32(1.0e-5), Float32(1.0) - Float32(1.0e-5))

        # ── flow-matching noise + target at LATENT level (rs:859-863) ──
        #   noisy  = sigma*noise + (1-sigma)*latent
        #   target = noise - latent
        var noise_seed = SEED * UInt64(104729)
        if not FIXED_SIGMA_SMOKE:
            noise_seed += UInt64(step)
        var noise = _host_noise(n_lat, noise_seed)
        var noisy = List[Float32]()
        var target = List[Float32]()
        noisy.reserve(n_lat)
        target.reserve(n_lat)
        for i in range(n_lat):
            noisy.append(sigma * noise[i] + (Float32(1.0) - sigma) * lat_bthwc[i])
            target.append(noise[i] - lat_bthwc[i])

        # ── patchify: noisy -> input patches [B*N,68]; target -> output [B*N,64] ──
        var patches = _patchify_in(noisy, B, 1, Hd, Wd, Cd)
        var target_patches = _patchify_out(target, B, 1, Hd, Wd, Cd)

        # ── t_embedder: sigma -> (t_cond RAW, base_adaln) ──
        var temb = _prepare_timestep(sigma, base, ctx)

        # ── forward (streamed 28-block) ──
        var fwd = anima_stack_lora_forward_streamed[H, Dh, S_IMG, S_TXT](
            patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(), context.copy(),
            base, st, lora, ropes.cos, ropes.sin,
            B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
        )

        # ── MSE loss + d_out (dMSE/dpred = 2/N*(pred-target)) ──
        # fwd.out is [B*N, 64] host; read it by reference (fwd is consumed by
        # the backward call below, so we don't bind/copy it out here).
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

        # ── backward (streamed) ──
        var grads = anima_stack_lora_backward_streamed[H, Dh, S_IMG, S_TXT](
            d_out, patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(), context.copy(),
            base, st, lora, ropes.cos, ropes.sin, fwd,
            B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
        )

        # ── global-norm clip (1.0) then AdamW over all adapters ──
        var gn_before = _global_norm(grads)
        _clip(grads, CLIP_NORM)
        anima_lora_adamw_step(lora, grads, step + 1, LR, ctx)

        # diagnostics
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

        if step == 0:
            first_loss = loss
        last_loss = loss

        var step_now = perf_counter_ns()
        print_trainer_progress(
            String("Anima-lora"), step + 1, RUN_STEPS, 1,
            loss, Float64(gn_before),
            Float64(step_now - step_t0) / 1.0e9, 0.0,
            Float64(step_now - t0) / 1.0e9,
        )
        if grads.nonfinite_lora_grads != 0:
            print("[Anima-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)

    var t1 = perf_counter_ns()
    var secs = Float64(t1 - t0) / 1.0e9
    print("")
    print("---- training summary ----")
    print("steps:", RUN_STEPS, " wall:", secs, "s  (", secs / Float64(RUN_STEPS), "s/step)")
    print("loss first =", first_loss, "  last =", last_loss,
          "  delta =", last_loss - first_loss)

    # ── final LoRA save (kohya net.blocks.<i>.* keys, rs:1136-1155) ──
    var npairs = save_anima_lora(lora, String(LORA_OUT), ctx)
    print("saved", npairs, "LoRA adapter pairs to", LORA_OUT)

    var b_final = Float32(0.0)
    var b_nz = 0
    for i in range(n_adapters):
        var s = _absum(lora.ad[i].b)
        b_final += s
        if s > 0.0:
            b_nz += 1
    var grew = (b_init == Float32(0.0)) and (b_final > Float32(0.0)) and (b_nz == n_adapters)
    var loss_down = last_loss < first_loss
    print("")
    if grew and loss_down:
        print("VERDICT: PASS — loss decreased (", first_loss, "->", last_loss,
              "), LoRA-B grew 0 -> nonzero (ratio", Float32(b_nz) / Float32(n_adapters), ")")
    else:
        print("VERDICT: review — loss_down=", loss_down, " B_grew=", grew,
              " (B init", b_init, " final", b_final, " nonzero", b_nz, "/", n_adapters, ")")
