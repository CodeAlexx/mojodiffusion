# serenitymojo/training/anima_sample_streamed.mojo
#
# ANIMA sample-during-training — denoise from the model's CURRENT state
# (streamed base trunk + live, in-place-updated host LoRA) and decode to PNG.
#
# Follows the PROVEN ideogram4 sampling-during-training template
# (serenity-trainer/.../Ideogram4SampleResident.mojo + Ideogram4LoRATrainer.mojo)
# but adapted to Anima's STREAMED weight path (the difference flagged in the build
# request: Anima's trainer does NOT hold the DiT resident; it streams each block
# from the safetensors handle per forward).
#
# ── WHY this reuses the TRAINING forward (not a fresh inference forward) ───────
# The whole point of sampling-during-training is to see what the model produces
# WITH the LoRA currently being trained. The Anima trainer's per-step forward IS
#   anima_stack_lora_forward_streamed[H, Dh, S_IMG, S_TXT](
#       patches, t_cond, base_adaln, context, base, st, lora, cos, sin, ...)
# (train_anima_real.mojo:728). We call the SAME function inside a CFG Euler
# denoise loop, passing the SAME live `base` / `st` (SafeTensors handle) / `lora`
# (host AnimaLoraSet, mutated in place by anima_lora_adamw_step) / `ropes`. So
# calling it IS the model+LoRA forward at the live weights. No overlay, no device
# LoRA conversion, no extra checkpoint load — the streamed forward already routes
# the LoRA additively per block (anima_stack_lora.mojo:359-412).
#
# ── VELOCITY CONVENTION (direct, NOT negated) ─────────────────────────────────
# The training target is  target = noise - latent  (rectified-flow v-target,
# train_anima_real.mojo:718), and `.out` (the streamed forward result) is trained
# to MATCH that target. So `.out` is the DIRECT rectified-flow velocity
#   v = noise - latent  (== d(x)/d(sigma) along the flow).
# The proven 1024 inference sampler (anima_sample_cli.mojo:_denoise:427-430) uses
# exactly this direct-velocity Euler:
#   pred = v_uncond + cfg*(v_cond - v_uncond)
#   x    = x + dt*pred           dt = sigma_next - sigma   (negative, 1->0)
# We reproduce that 1:1 here — the ONLY change is the forward fn (streamed-train
# instead of the inference resident-nosave forward), which is numerically the
# same block math (composition-gate-equivalent per anima_stack_lora.mojo:357).
#
# ── SCHEDULE (1:1 with the inference sampler) ─────────────────────────────────
# Linear sigma 1->0 over NUM_STEPS, RAW sigma into the t_embedder (NOT *1000),
# matching anima_sample_cli.mojo:404-409. CFG default 4.5, 30 steps
# (anima_contract NUM_STEPS / CFG_SCALE_X10) — overridable by the caller.
#
# ── GRID (trainer's cropped latent) ───────────────────────────────────────────
# The Anima trainer trains on a LATENT_HW x LATENT_HW crop (train_anima_real.mojo
# LATENT_HW, default 16 -> S_IMG=64). The live LoRA + streamed forward are comptime
# on that S_IMG / S_TXT, so the sample MUST run at the same grid (you cannot feed
# a 128x128 latent through a forward monomorphized for S_IMG=64). We therefore
# sample at LATENT_HW x LATENT_HW and decode to a (LATENT_HW*8) px preview PNG.
# This is a valid model+LoRA preview at the trained sub-window resolution.
#
# ── CONDITIONING (v1 decision — flagged) ──────────────────────────────────────
# The trainer holds ONE cached frozen LLM-adapter context [B,256,1024] (the
# captured anima_embeddings sidecar, train_anima_real.mojo:667). v1 reuses THAT
# cached context as the COND conditioning — no Qwen3-VL load, no tokenizer wiring.
# UNCOND is a zeroed [B,S_TXT*JOINT] context (the empty-prompt CFG cond, matching
# anima_sample_cli.mojo:485-487). Swapping in a real encode(prompt) later is a
# drop-in replacement of `context_cond` only.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, exp as fexp
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.norm import rms_norm

from serenitymojo.models.anima.weights import AnimaStackBase
from serenitymojo.models.anima.anima_stack_lora import (
    AnimaLoraSet, anima_stack_lora_forward_streamed,
)
from serenitymojo.models.dit.anima_contract import (
    ANIMA_HIDDEN, ANIMA_LATENT_CHANNELS, ANIMA_PATCH_SIZE, ANIMA_VAE_PATH,
)
from serenitymojo.models.vae.qwenimage_tiled_decode import wan21_image_tiled_decode
from serenitymojo.image.png import save_png, ValueRange


# ── Anima dims (anima_contract); B=1, the trainer's batch ─────────────────────
comptime B = 1
comptime D = ANIMA_HIDDEN            # 2048
comptime F = 8192                    # GELU MLP hidden (matches train_anima_real.F)
comptime JOINT = 1024                # cross-attn context dim (ANIMA_ADAPTER_DIM)
comptime C = ANIMA_LATENT_CHANNELS   # 16
comptime PS = ANIMA_PATCH_SIZE       # 2
comptime IN_PATCH = (C + 1) * PS * PS   # 68
comptime OUT_PATCH = C * PS * PS        # 64
comptime EPS = Float32(1e-06)


# ── deterministic host gaussian noise (Box-Muller on a PCG stream) ────────────
# IDENTICAL layout to train_anima_real._host_noise / anima_sample_cli._host_noise.
def _sample_host_noise(n: Int, seed: UInt64) -> List[Float32]:
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


# ── t_embedder forward (anima_dit.mojo:822-843), RAW sigma — t_cond returned RAW
#    (un-silu'd) sinusoidal RMSNorm; the streamed block silus it internally. ────
struct _SampleTEmb(Movable):
    var t_cond: List[Float32]      # [B, 2048] RAW
    var base_adaln: List[Float32]  # [B, 6144]

    def __init__(out self, var t_cond: List[Float32], var base_adaln: List[Float32]):
        self.t_cond = t_cond^
        self.base_adaln = base_adaln^


def _sample_sinusoidal(val: Float32, dim: Int) -> List[Float32]:
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


def _sample_prepare_timestep(
    sigma: Float32, base: AnimaStackBase, ctx: DeviceContext
) raises -> _SampleTEmb:
    var emb_l = _sample_sinusoidal(sigma, D)
    var emb = Tensor.from_host(emb_l, [B, D], STDtype.F32, ctx)
    var h = linear(emb, base.te_lin1[], Optional[Tensor](None), ctx)
    var hidden = silu(h, ctx)
    var base_adaln = linear(hidden, base.te_lin2[], Optional[Tensor](None), ctx)
    var t_cond = rms_norm(emb, base.t_norm[], EPS, ctx)
    return _SampleTEmb(t_cond.to_host(ctx), base_adaln.to_host(ctx))


# ── patchify INPUT: channels-last [B,Hd,Wd,C] -> patches [B*N,68] (mask ch=0) ──
# IDENTICAL to anima_sample_cli._patchify_in (which equals train_anima_real
# _patchify_in for T=1). Patch dim order [Cp,pH,pW] (C slowest), matching
# anima_dit _patchify.
def _sample_patchify_in(
    x: List[Float32], Hd: Int, Wd: Int
) -> List[Float32]:
    var pH = PS
    var pW = PS
    var nH = Hd // pH
    var nW = Wd // pW
    var Cp = C + 1
    var N = nH * nW
    var pd = Cp * pH * pW   # 68
    var out = List[Float32]()
    out.reserve(B * N * pd)
    for _ in range(B * N * pd):
        out.append(Float32(0.0))
    for b in range(B):
        for ih in range(nH):
            for iw in range(nW):
                var pn = ih * nW + iw
                for c in range(Cp):
                    for ph in range(pH):
                        for pw in range(pW):
                            var od = (b * N + pn) * pd + (c * pH * pW + ph * pW + pw)
                            if c < C:
                                var hh = ih * pH + ph
                                var ww = iw * pW + pw
                                var src = ((b * Hd + hh) * Wd + ww) * C + c
                                out[od] = x[src]
                            # else mask channel stays 0.0
    return out^


# ── unpatchify velocity patches [B*N,64] -> channels-last [B,Hd,Wd,C] ─────────
# IDENTICAL to anima_sample_cli._unpatchify_out. Patch dim order [pH,pW,C]
# (C fastest), the inverse of anima_dit _unpatchify (the layout `.out` lives in).
def _sample_unpatchify_out(
    p: List[Float32], Hd: Int, Wd: Int
) -> List[Float32]:
    var pH = PS
    var pW = PS
    var nH = Hd // pH
    var nW = Wd // pW
    var pd = C * pH * pW   # 64
    var N = nH * nW
    var out = List[Float32]()
    out.reserve(B * Hd * Wd * C)
    for _ in range(B * Hd * Wd * C):
        out.append(Float32(0.0))
    for b in range(B):
        for ih in range(nH):
            for iw in range(nW):
                var pn = ih * nW + iw
                for ph in range(pH):
                    for pw in range(pW):
                        for c in range(C):
                            var od = (b * N + pn) * pd + (ph * pW * C + pw * C + c)
                            var hh = ih * pH + ph
                            var ww = iw * pW + pw
                            var dst = ((b * Hd + hh) * Wd + ww) * C + c
                            out[dst] = p[od]
    return out^


# ──────────────────────────────────────────────────────────────────────────────
# anima_sample_streamed — CFG Euler denoise on the STREAMED base + live host LoRA.
#
# Comptime params (MUST match the trainer's monomorphization so the live LoRA +
# streamed forward type-check):
#   H, Dh        attention heads / head dim (16 / 128)
#   S_IMG        image-token count  = (LATENT_HW/PS)^2  (the trainer's crop grid)
#   S_TXT        context-token count = ANIMA_MAX_SEQ_LEN (256 for this trainer)
#   LATENT_HW    latent grid edge (the trainer's crop, default 16)
#
# Inputs (all live in the train loop):
#   base             AnimaStackBase   (F32 resident projections + t_embedder)
#   st               SafeTensors      (the DiT checkpoint handle; blocks streamed)
#   lora             AnimaLoraSet     (host-resident, mutated in place each step)
#   cos, sin         RoPE tables for the trainer's grid (ropes.cos / ropes.sin)
#   context_cond     [B*S_TXT*JOINT]  cached frozen context (COND conditioning)
#   context_uncond   [B*S_TXT*JOINT]  zeroed context (UNCOND, empty-prompt CFG)
#   n_steps          Euler steps (inference default 30)
#   cfg              CFG scale (inference default 4.5)
#   seed             Box-Muller seed for the t=1 init noise
#
# Returns the denoised channels-last latent [B*LATENT_HW*LATENT_HW*C] F32 (SCALED
# latent space — the decode does z/inv_std+mean internally, NOT pre-unscaled).
# ──────────────────────────────────────────────────────────────────────────────
def anima_sample_streamed[
    H: Int, Dh: Int, S_IMG: Int, S_TXT: Int, LATENT_HW: Int
](
    base: AnimaStackBase,
    st: SafeTensors,
    lora: AnimaLoraSet,
    cos: Tensor,
    sin: Tensor,
    context_cond: List[Float32],     # [B*S_TXT*JOINT]
    context_uncond: List[Float32],   # [B*S_TXT*JOINT]
    n_steps: Int,
    cfg: Float32,
    seed: UInt64,
    ctx: DeviceContext,
) raises -> List[Float32]:
    if n_steps < 1:
        raise Error("anima_sample_streamed: n_steps must be >= 1")

    var n_lat = B * LATENT_HW * LATENT_HW * C   # channels-last [B,H,W,C]
    var x = _sample_host_noise(n_lat, seed)

    # high -> low sigma (linear 1->0), direct-velocity Euler, 1:1 with the proven
    # 1024 inference sampler (anima_sample_cli._denoise).
    for step in range(n_steps):
        var sigma = Float32(1.0) - Float32(step) / Float32(n_steps)
        var sigma_next = Float32(1.0) - Float32(step + 1) / Float32(n_steps)
        var dt = sigma_next - sigma   # negative

        var patches = _sample_patchify_in(x, LATENT_HW, LATENT_HW)
        var temb = _sample_prepare_timestep(sigma, base, ctx)   # RAW sigma

        # COND pass: streamed base + live LoRA, conditioned on context_cond.
        var fwd_c = anima_stack_lora_forward_streamed[H, Dh, S_IMG, S_TXT](
            patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(),
            context_cond.copy(),
            base, st, lora, cos, sin,
            B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
        )
        var v_c = _sample_unpatchify_out(fwd_c.out, LATENT_HW, LATENT_HW)

        # UNCOND pass: same trunk + LoRA, zeroed context.
        var fwd_u = anima_stack_lora_forward_streamed[H, Dh, S_IMG, S_TXT](
            patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(),
            context_uncond.copy(),
            base, st, lora, cos, sin,
            B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
        )
        var v_u = _sample_unpatchify_out(fwd_u.out, LATENT_HW, LATENT_HW)

        # CFG: pred = uncond + cfg*(cond - uncond); direct-velocity Euler x += dt*pred.
        for j in range(len(x)):
            var pred = v_u[j] + cfg * (v_c[j] - v_u[j])
            x[j] = x[j] + dt * pred

        # Drain enqueued buffer frees from this step's two streamed forwards before
        # the next step allocates (the per-forward SDPA scratch + per-block swap
        # buffers accumulate otherwise) — holds peak VRAM flat across the loop,
        # exactly as anima_sample_cli._denoise:436 does.
        ctx.synchronize()

    return x^


# ──────────────────────────────────────────────────────────────────────────────
# anima_decode_latent_to_png — in-process tiled Qwen/Wan VAE decode + write PNG.
#
# 1:1 with anima_decode_cli's decode tail, but IN-PROCESS (the build request
# prefers an in-process decode for sampling). At the trainer's small LATENT_HW
# grid the decode is cheap (LATENT_HW*8 px image), so no process separation is
# needed — the streamed forward already freed its per-step scratch via
# ctx.synchronize at the end of anima_sample_streamed.
#
# x is channels-last [B,LATENT_HW,LATENT_HW,C] SCALED latent; we repack to NCHW
# [1,16,LATENT_HW,LATENT_HW] BF16 and feed wan21_image_tiled_decode, whose
# internal z/inv_std+mean == OneTrainer AnimaModel.unscale_latents (do NOT
# pre-unscale). save_png writes the [-1,1] SIGNED VAE output.
# ──────────────────────────────────────────────────────────────────────────────
def anima_decode_latent_to_png[LATENT_HW: Int](
    x: List[Float32],               # [B*LATENT_HW*LATENT_HW*C] channels-last F32
    out_path: String,
    ctx: DeviceContext,
) raises:
    # channels-last [B,H,W,C] -> NCHW [1,C,H,W]
    var nchw = List[Float32]()
    nchw.reserve(B * C * LATENT_HW * LATENT_HW)
    for _ in range(B * C * LATENT_HW * LATENT_HW):
        nchw.append(Float32(0.0))
    for b in range(B):
        for h in range(LATENT_HW):
            for w in range(LATENT_HW):
                for c in range(C):
                    var src = ((b * LATENT_HW + h) * LATENT_HW + w) * C + c
                    var dst = ((b * C + c) * LATENT_HW + h) * LATENT_HW + w
                    nchw[dst] = x[src]
    var lat = Tensor.from_host(nchw, [1, C, LATENT_HW, LATENT_HW], STDtype.BF16, ctx)

    # tiled decode (3x3 half-tile crops). TILE = LATENT_HW//2; valid for the
    # trainer's LATENT_HW (16 -> TILE 8). -> NCHW [1,3,LATENT_HW*8,LATENT_HW*8].
    var rgb = wan21_image_tiled_decode[LATENT_HW, LATENT_HW](
        lat, String(ANIMA_VAE_PATH), ctx
    )
    save_png(rgb, out_path, ctx, ValueRange.SIGNED)
