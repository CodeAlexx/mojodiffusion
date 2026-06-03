# serenitymojo/pipeline/anima_sample_cli.mojo
#
# STANDALONE Anima 1024x1024 LoRA sampler CLI (process-SEPARATED from the trainer).
#
#   anima_sample_cli <lora_peft.safetensors|base> <context.safetensors|sample_prompts.json> <out.png> [seed] [prompt_id]
#
# Loads the Anima base DiT (BF16 resident base + F32 stack base), OVERLAYS a PEFT
# LoRA additively (scale*B*A at the inference linears, ai-toolkit forward overlay,
# NEVER fused — [[LoRA never fused]]), runs the 30-step 1024 rectified-flow Euler
# CFG denoise, and writes a scaled latent for fresh-process 1024 VAE decode.
#
# This MIRRORS the proven 1024 inference pipeline (anima_pipeline_1024_multistep.mojo
# == anima_infer.rs port): scaled latent space, RAW sigma (NOT *1000), CFG 4.5,
# 30 steps, seed Box-Muller noise. The paired anima_decode_cli feeds that scaled
# latent directly to the Mojo Qwen VAE decoder, whose internal unnormalize is
# z/inv_std + mean == OneTrainer AnimaModel.unscale_latents. The ONLY deltas vs that
# pipeline are: (1) the forward routes through anima_stack_lora_forward_streamed (the
# no-save, load->use->drop block path — fits 24GB at S_IMG=4096) with a LoRA overlay;
# (2) the context (cond + uncond) is a CLI arg, zero-padded to S_TXT=512 (the trained
# context length); (3) `base` mode uses a zero-B overlay so the forward reduces
# bit-exactly to the frozen base DiT.
#
# S_TXT: the trained LoRA/context are 512. The cross-attn SDPA is comptime on S_TXT,
# so we monomorphize the streamed forward at S_TXT=512 (the trainer proved this
# compiles+runs). The base inference (forward_with_context) uses 256 — we do NOT use
# that path; the streamed LoRA forward at 512 is the single code path for both base
# (zero-B) and lora modes.
#
# Build:
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda -Xlinker -lpng16 \
#       serenitymojo/pipeline/anima_sample_cli.mojo -o /tmp/anima_sample_cli
#   /tmp/anima_sample_cli base output/giger3_prompts/p1.safetensors /tmp/base.png 42

from std.sys import argv
from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, exp as fexp
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
    AnimaBlockWeights, load_anima_block_weights_bf16_normf32,
)
from serenitymojo.models.anima.lora_block import ANIMA_SLOTS
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.anima.anima_stack_lora import (
    AnimaLoraSet, build_anima_lora_set, load_anima_lora_resume,
    AnimaLoraDeviceSet, anima_lora_set_to_device,
    anima_stack_lora_forward_device_resident_nosave,
)
from serenitymojo.models.dit.anima_contract import (
    ANIMA_HIDDEN, ANIMA_NUM_HEADS, ANIMA_HEAD_DIM, ANIMA_DEPTH,
    ANIMA_LATENT_CHANNELS, ANIMA_PATCH_SIZE, ANIMA_ADAPTER_DIM,
)
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.training.sample_prompt_config import (
    SamplePrompt, SamplePromptConfig, read_sample_prompt_config,
)
from serenitymojo.training.progress_display import (
    print_sample_setup, print_sample_step, print_sample_saved,
)


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
comptime EPS = Float32(1e-06)

# ── 1024x1024 inference contract ──────────────────────────────────────────────
comptime LATENT_HW = 128                       # 1024 / 8 = 128
comptime NH = LATENT_HW // PS                  # 64
comptime NW = LATENT_HW // PS                  # 64
comptime S_IMG = NH * NW                        # 4096 image tokens
comptime S_TXT = 512                            # trained context length
comptime NUM_STEPS = 30
comptime CFG_SCALE = Float32(4.5)
comptime DEFAULT_SEED = UInt64(42)

# ── LoRA recipe (matches train_anima_ot_res512: RANK=16, ALPHA=16) ────────────
comptime RANK = 16
comptime ALPHA = Float32(16.0)


# ── deterministic host gaussian noise (Box-Muller on a PCG stream) ────────────
# IDENTICAL to ops/random.randn's Rust-StdRng layout is NOT required here — the
# proven multistep pipeline uses ops.random.randn(seed=42). For a standalone CLI
# we use the same Box-Muller PCG the trainer uses; the gate is finite+non-blank +
# base-vs-lora diff, not bit-parity vs the Rust oracle.
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


# ── t_embedder forward (anima_dit.mojo:822-843), RAW sigma (NOT /1000) ────────
# Matches the PROVEN 1024 inference (anima_pipeline_1024_multistep.mojo / anima_infer.rs):
# the embedder consumes the RAW sigma value. (The trainer's res512 path uses ts/1000;
# the production INFERENCE path uses raw sigma — we mirror inference here.)
struct _TEmb(Movable):
    var t_cond: List[Float32]      # [B, 2048] RAW sinusoidal RMSNorm
    var base_adaln: List[Float32]  # [B, 6144]

    def __init__(out self, var t_cond: List[Float32], var base_adaln: List[Float32]):
        self.t_cond = t_cond^
        self.base_adaln = base_adaln^


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


def _prepare_timestep(
    sigma: Float32, base: AnimaStackBase, ctx: DeviceContext
) raises -> _TEmb:
    var emb_l = _sinusoidal_host(sigma, D)
    var emb = Tensor.from_host(emb_l, [B, D], STDtype.F32, ctx)
    var h = linear(emb, base.te_lin1[], Optional[Tensor](None), ctx)
    var hidden = silu(h, ctx)
    var base_adaln = linear(hidden, base.te_lin2[], Optional[Tensor](None), ctx)
    var t_cond = rms_norm(emb, base.t_norm[], EPS, ctx)
    return _TEmb(t_cond.to_host(ctx), base_adaln.to_host(ctx))


# ── 3D-RoPE (T,H,W) NTK tables [B*S_IMG*H, Dh/2], parameterized on nh/nw ───────
# Identical math to train_anima_ot_res512._rope_tables + anima_dit.build_anima_3d_rope,
# but takes the patch-grid (nh,nw) as args so it works at the 1024 grid (64x64).
struct _Rope(Movable):
    var cos: Tensor
    var sin: Tensor

    def __init__(out self, var cos: Tensor, var sin: Tensor):
        self.cos = cos^
        self.sin = sin^


def _rope_tables(nh: Int, nw: Int, ctx: DeviceContext) raises -> _Rope:
    var half = Dh // 2            # 64
    var full_d = Dh              # 128
    var t_frames = 1
    var s_img = nh * nw

    var dim_h = full_d // 6 * 2   # 42
    var dim_w = dim_h             # 42
    var dim_t = full_d - 2 * dim_h  # 44
    var bins_t = dim_t // 2       # 22
    var bins_h = dim_h // 2       # 21
    var bins_w = dim_w // 2       # 21

    var base_theta: Float64 = 10000.0
    var h_exp = Float64(dim_h) / (Float64(dim_h) - 2.0)
    var w_exp = Float64(dim_w) / (Float64(dim_w) - 2.0)
    var h_ntk = fexp(flog(Float64(4.0)) * h_exp)
    var w_ntk = fexp(flog(Float64(4.0)) * w_exp)
    var t_ntk = Float64(1.0)      # t_extra=1.0
    var theta_h = Float64(base_theta * h_ntk)
    var theta_w = Float64(base_theta * w_ntk)
    var theta_t = Float64(base_theta * t_ntk)

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


# ── patchify INPUT: channels-last [B,Hd,Wd,C] -> patches [B*N,68] (mask ch=0) ──
def _patchify_in(
    x: List[Float32], Bd: Int, Hd: Int, Wd: Int, Cd: Int
) -> List[Float32]:
    var pH = PS
    var pW = PS
    var nH = Hd // pH
    var nW = Wd // pW
    var Cp = Cd + 1
    var N = nH * nW
    var pd = Cp * pH * pW
    var out = List[Float32]()
    out.reserve(Bd * N * pd)
    for _ in range(Bd * N * pd):
        out.append(Float32(0.0))
    for b in range(Bd):
        for ih in range(nH):
            for iw in range(nW):
                var pn = ih * nW + iw
                for c in range(Cp):
                    for ph in range(pH):
                        for pw in range(pW):
                            var od = (b * N + pn) * pd + (c * pH * pW + ph * pW + pw)
                            if c < Cd:
                                var hh = ih * pH + ph
                                var ww = iw * pW + pw
                                var src = ((b * Hd + hh) * Wd + ww) * Cd + c
                                out[od] = x[src]
    return out^


# ── unpatchify velocity patches [B*N,64] -> channels-last [B,Hd,Wd,C] ─────────
def _unpatchify_out(
    p: List[Float32], Bd: Int, Hd: Int, Wd: Int, Cd: Int
) -> List[Float32]:
    var pH = PS
    var pW = PS
    var nH = Hd // pH
    var nW = Wd // pW
    var pd = Cd * pH * pW
    var N = nH * nW
    var out = List[Float32]()
    out.reserve(Bd * Hd * Wd * Cd)
    for _ in range(Bd * Hd * Wd * Cd):
        out.append(Float32(0.0))
    for b in range(Bd):
        for ih in range(nH):
            for iw in range(nW):
                var pn = ih * nW + iw
                for ph in range(pH):
                    for pw in range(pW):
                        for c in range(Cd):
                            var od = (b * N + pn) * pd + (ph * pW * Cd + pw * Cd + c)
                            var hh = ih * pH + ph
                            var ww = iw * pW + pw
                            var dst = ((b * Hd + hh) * Wd + ww) * Cd + c
                            out[dst] = p[od]
    return out^


# ── load a context safetensors key as host F32 [n], zero-padded to S_TXT=512 ───
# The text-context pipeline writes `context_cond` [1, S, 1024] (S may be 256 or 512).
# We zero-pad/truncate to exactly [1, 512, 1024] (the conditioner emits dense zeros
# for pad positions — AnimaModel.py:229).
def _load_context_512(path: String, key: String, ctx: DeviceContext) raises -> List[Float32]:
    var st = SafeTensors.open(path)
    var info = st.tensor_info(key)
    var bytes = st.tensor_bytes(key)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    var raw = cast_tensor(t^, STDtype.F32, ctx).to_host(ctx)
    var src_tokens = len(raw) // (B * JOINT)
    var out = List[Float32]()
    out.reserve(B * S_TXT * JOINT)
    for _ in range(B * S_TXT * JOINT):
        out.append(Float32(0.0))
    var copy_tokens = src_tokens if src_tokens < S_TXT else S_TXT
    for b in range(B):
        for s in range(copy_tokens):
            for j in range(JOINT):
                out[(b * S_TXT + s) * JOINT + j] = raw[(b * src_tokens + s) * JOINT + j]
    print("  context key", key, "src_tokens=", src_tokens, "-> padded to", S_TXT)
    return out^


def _ctx_std(c: List[Float32]) -> Float32:
    var m = Float64(0.0)
    for i in range(len(c)):
        m += Float64(c[i])
    m /= Float64(len(c))
    var v = Float64(0.0)
    for i in range(len(c)):
        var d = Float64(c[i]) - m
        v += d * d
    v /= Float64(len(c))
    return Float32(sqrt(v))


# ── zero-B overlay set: every B = 0 -> scale*B*A == 0 -> reduces bit-exact to
#    the frozen base DiT (the WITHOUT-LoRA pass). NEVER fuses. ──────────────────
def _zero_b_set(set: AnimaLoraSet) -> AnimaLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        var src = set.ad[i].copy()
        for j in range(len(src.b)):
            src.b[j] = Float32(0.0)
        ad.append(src^)
    return AnimaLoraSet(ad^, set.num_blocks, set.rank)


def _mean_abs(v: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(v)):
        var x = v[i]
        s += Float64(x) if x >= 0.0 else Float64(-x)
    return s / Float64(len(v))


def _std(v: List[Float32]) -> Float64:
    var m = Float64(0.0)
    for i in range(len(v)):
        m += Float64(v[i])
    m /= Float64(len(v))
    var vv = Float64(0.0)
    for i in range(len(v)):
        var d = Float64(v[i]) - m
        vv += d * d
    return sqrt(vv / Float64(len(v)))


def _count_nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


# ── the 30-step Euler CFG denoise using the RESIDENT no-save LoRA forward ──────
# Scaled latent x, RAW sigma (1->0 linear), CFG 4.5, direct-velocity Euler
# x = x + dt*pred. The
# forward routes through anima_stack_lora_forward_device_resident_nosave: all 28
# blocks stay GPU-resident (BF16, ~3.7 GiB — loaded ONCE, no per-forward disk
# reload), each block's saved activations are discarded immediately (only one
# block live at a time, so VRAM stays flat). This is the 24GB-safe inference
# forward (NOT the training saved-activation forward). LoRA overlay = additive
# scale·B·A per block (NEVER fused).
def _denoise(
    context_cond: List[Float32], context_uncond: List[Float32],
    lora_dev: AnimaLoraDeviceSet, base: AnimaStackBase,
    blocks: List[AnimaBlockWeights],
    rope: _Rope, seed: UInt64, ctx: DeviceContext,
) raises -> List[Float32]:
    var n_lat = B * LATENT_HW * LATENT_HW * C   # channels-last [B,H,W,C]
    var x = _host_noise(n_lat, seed)
    print("  init noise mean_abs=", _mean_abs(x), " std=", _std(x))

    for step in range(NUM_STEPS):
        var step_t0 = perf_counter_ns()
        var sigma = Float32(1.0) - Float32(step) / Float32(NUM_STEPS)
        var sigma_next = Float32(1.0) - Float32(step + 1) / Float32(NUM_STEPS)
        var dt = sigma_next - sigma   # negative

        var patches = _patchify_in(x, B, LATENT_HW, LATENT_HW, C)
        var temb = _prepare_timestep(sigma, base, ctx)   # RAW sigma

        # conditional
        var out_c = anima_stack_lora_forward_device_resident_nosave[H, Dh, S_IMG, S_TXT](
            patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(), context_cond.copy(),
            base, blocks, lora_dev, rope.cos, rope.sin,
            B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
        )
        var v_c = _unpatchify_out(out_c, B, LATENT_HW, LATENT_HW, C)

        # unconditional
        var out_u = anima_stack_lora_forward_device_resident_nosave[H, Dh, S_IMG, S_TXT](
            patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(), context_uncond.copy(),
            base, blocks, lora_dev, rope.cos, rope.sin,
            B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
        )
        var v_u = _unpatchify_out(out_u, B, LATENT_HW, LATENT_HW, C)

        # CFG: pred = uncond + scale*(cond - uncond);  Euler: x = x + dt*pred
        for j in range(len(x)):
            var pred = v_u[j] + CFG_SCALE * (v_c[j] - v_u[j])
            x[j] = x[j] + dt * pred

        # Drain the enqueued buffer frees from this step's two forwards before the
        # next step allocates — otherwise the DeviceContext accumulates the
        # per-forward SDPA scratch ([16,4096,4096] F32 ≈ 1 GiB) across steps and
        # OOMs at 24 GiB. The synchronize holds peak VRAM flat across the loop.
        ctx.synchronize()
        var step_secs = Float64(perf_counter_ns() - step_t0) / 1.0e9
        var rate = Float64(0.0)
        if step_secs > 0.0:
            rate = Float64(1.0) / step_secs
        print_sample_step(
            String("Anima-lora"), step + 1, NUM_STEPS, sigma, step_secs, rate,
        )
    return x^


# ── SCOPED load + denoise: owns ALL DiT device tensors; returns only the host
#    latent. Everything (base, 28 resident blocks, LoRA device set, RoPE) frees on
#    return so the VAE decode gets the full GPU. ────────────────────────────────
def _load_and_denoise(
    use_base: Bool, lora_arg: String, context_path: String,
    uncond_context_path: String, seed: UInt64, ctx: DeviceContext,
) raises -> List[Float32]:
    # ── base DiT (BF16 checkpoint; stack base F32 resident) ──
    var cfg = anima()
    print("\n--- Load base DiT:", cfg.checkpoint, "---")
    var st = SafeTensors.open(cfg.checkpoint)
    verify_anima_stack_shapes(st, ANIMA_DEPTH)
    var base = load_anima_stack_base(st, ctx)
    print("  base projections + t_embedder loaded")

    # ── context (cond from the prompt; uncond from the empty-prompt file) ──
    print("\n--- Load context (cond + uncond) ---")
    var context_cond = _load_context_512(context_path, String("context_cond"), ctx)
    print("  ctx_cond std =", _ctx_std(context_cond))
    var context_uncond = List[Float32]()
    var st_ctx = SafeTensors.open(context_path)
    var names = st_ctx.names()
    var has_uncond = False
    for i in range(len(names)):
        if names[i] == String("context_uncond"):
            has_uncond = True
    if has_uncond:
        context_uncond = _load_context_512(context_path, String("context_uncond"), ctx)
    else:
        var uncond_path = uncond_context_path.copy()
        if uncond_path == String(""):
            uncond_path = String("output/giger3_prompts/uncond.safetensors")
        var ok = True
        try:
            context_uncond = _load_context_512(uncond_path, String("context_cond"), ctx)
        except:
            ok = False
        if not ok:
            print("  no uncond file -> using zero context (empty-prompt CFG)")
            for _ in range(B * S_TXT * JOINT):
                context_uncond.append(Float32(0.0))
    print("  ctx_uncond std =", _ctx_std(context_uncond))

    # ── LoRA set: real overlay (load_anima_lora_resume) or zero-B (base) ──
    print("\n--- Build LoRA overlay ---")
    var lora = build_anima_lora_set(ANIMA_DEPTH, D, JOINT, F, RANK, ALPHA)
    if use_base:
        lora = _zero_b_set(lora)
        print("  BASE mode: zero-B overlay (forward == frozen base DiT)")
    else:
        lora = load_anima_lora_resume(ANIMA_DEPTH, RANK, ALPHA, lora_arg, ctx)
        var babs = Float32(0.0)
        for i in range(len(lora.ad)):
            for j in range(len(lora.ad[i].b)):
                var x = lora.ad[i].b[j]
                babs += x if x >= 0.0 else -x
        print("  LoRA overlay loaded:", len(lora.ad), "adapters  |B|_1 =", babs,
              " scale=alpha/rank=", ALPHA / Float32(RANK))

    # upload the LoRA set to device ONCE (BF16 adapters; mixed_base F32 forward).
    var lora_dev = anima_lora_set_to_device(lora, STDtype.BF16, ctx)

    # ── load ALL 28 blocks RESIDENT (BF16 projections + F32 norms, ~3.7 GiB) ──
    print("\n--- Load 28 blocks resident (BF16) ---")
    var blocks = List[AnimaBlockWeights]()
    for bi in range(ANIMA_DEPTH):
        blocks.append(load_anima_block_weights_bf16_normf32(st, bi, ctx))
    print("  resident blocks:", len(blocks))

    # ── RoPE tables for the 1024 grid ──
    var rope = _rope_tables(NH, NW, ctx)

    # ── 30-step Euler CFG denoise ──
    print("\n--- Denoise (", NUM_STEPS, "steps, CFG=", CFG_SCALE, ") ---")
    var x = _denoise(context_cond, context_uncond, lora_dev, base, blocks, rope, seed, ctx)
    ctx.synchronize()
    return x^


# ── shared sample-prompt JSON bridge. Anima prompts store precomputed
# `context_cond` sidecars in caps.positive, and optional empty/negative context
# sidecars in caps.negative. Text encoders stay out of the sampler/trainer path.
def _select_prompt(sample_cfg: SamplePromptConfig, wanted: String) raises -> SamplePrompt:
    if len(sample_cfg.prompts) == 0:
        raise Error("anima_sample_cli: sample prompt JSON has no prompts")
    if wanted == String(""):
        return sample_cfg.prompts[0].copy()
    for i in range(len(sample_cfg.prompts)):
        if sample_cfg.prompts[i].label == wanted:
            return sample_cfg.prompts[i].copy()
    raise Error(String("anima_sample_cli: prompt id not found: ") + wanted)


# ── save channels-last latent [B,H,W,C] -> NCHW [1,16,128,128] F32 safetensors ─
# Key `latent` (same schema anima_decode_cli + the VAE smoke read).
def _save_latent_nchw(
    bhwc: List[Float32], path: String, ctx: DeviceContext,
) raises:
    var nchw = List[Float32]()
    nchw.resize(B * C * LATENT_HW * LATENT_HW, Float32(0.0))
    for b in range(B):
        for h in range(LATENT_HW):
            for w in range(LATENT_HW):
                for c in range(C):
                    var src = ((b * LATENT_HW + h) * LATENT_HW + w) * C + c
                    var dst = ((b * C + c) * LATENT_HW + h) * LATENT_HW + w
                    nchw[dst] = bhwc[src]
    var lat = Tensor.from_host(nchw, [1, C, LATENT_HW, LATENT_HW], STDtype.F32, ctx)
    var names = List[String]()
    names.append(String("latent"))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer(lat^))
    save_safetensors(names, tensors, path, ctx)


def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error(
            "usage: anima_sample_cli <lora_peft.safetensors|base>"
            " <context.safetensors|sample_prompts.json> <out.png> [seed] [prompt_id]"
        )
    var lora_arg = String(args[1])
    var context_path = String(args[2])
    var uncond_context_path = String("")
    var out_png = String(args[3])
    var seed = DEFAULT_SEED
    var seed_overridden = False
    if len(args) > 4:
        var v = UInt64(0)
        var bs = String(args[4]).as_bytes()
        for i in range(len(bs)):
            v = v * 10 + UInt64(Int(bs[i] - 0x30))
        seed = v
        seed_overridden = True
    var prompt_id = String("")
    if len(args) > 5:
        prompt_id = String(args[5])

    if context_path.endswith(".json"):
        var sample_cfg = read_sample_prompt_config(context_path)
        var prompt = _select_prompt(sample_cfg, prompt_id)
        if prompt.width != 1024 or prompt.height != 1024 or prompt.frames != 1:
            raise Error("anima_sample_cli currently supports 1024x1024 image prompts only")
        print("  sample prompt JSON:", context_path, " prompt=", prompt.label)
        context_path = prompt.caps_pos.copy()
        uncond_context_path = prompt.caps_neg.copy()
        if not seed_overridden:
            seed = prompt.seed

    var use_base = (lora_arg == String("base"))

    print("============================================================")
    print("Anima 1024x1024 LoRA sampler CLI")
    print("  lora      :", lora_arg, "(base-only)" if use_base else "(PEFT overlay)")
    print("  context   :", context_path)
    if uncond_context_path != String(""):
        print("  uncond    :", uncond_context_path)
    print("  out       :", out_png)
    print("  seed      :", seed)
    print("  S_IMG=", S_IMG, " S_TXT=", S_TXT, " steps=", NUM_STEPS,
          " CFG=", CFG_SCALE, " RANK=", RANK, " ALPHA=", ALPHA)
    print("============================================================")
    print_sample_setup(
        String("Anima-lora"), String("anima"), NUM_STEPS, CFG_SCALE, S_IMG, ANIMA_DEPTH,
    )

    var ctx = DeviceContext()
    var t0 = perf_counter_ns()

    # Run the full DiT load + 30-step denoise in a SCOPED helper so every device
    # tensor it owns (base, the 28 resident blocks ~3.7 GiB, the LoRA device set,
    # RoPE tables, all forward scratch) is FREED when it returns — leaving only the
    # host latent. The VAE decode at 1024 needs the GPU to itself (the 3D-conv
    # upsample is multi-GiB); keeping the DiT blocks resident through decode OOMs
    # 24 GiB. Freeing first holds the decode peak well under cap.
    var x = _load_and_denoise(
        use_base, lora_arg, context_path, uncond_context_path, seed, ctx,
    )

    var nbad = _count_nonfinite(x)
    print("\n  final latent: mean_abs=", _mean_abs(x), " std=", _std(x),
          " nonfinite=", nbad)
    if nbad > 0:
        raise Error("final latent contains non-finite values")

    # ── SAVE the final latent (NCHW [1,16,128,128] F32) for SEPARATE-PROCESS
    #    decode. The VAE 1024 decode (3D-conv upsample) is multi-GiB and OOMs 24
    #    GiB if it shares the process with the freed-but-pooled DiT allocations, so
    #    anima_sample_cli WRITES the latent and a fresh-process anima_decode_cli
    #    decodes it (process separation, as the handoff endorses). ───────────────
    var latent_path = out_png + String(".latent.safetensors")
    _save_latent_nchw(x, latent_path, ctx)

    var dt_s = Float64(perf_counter_ns() - t0) / 1.0e9
    print("\n============================================================")
    print("LATENT SAVED:", latent_path)
    print_sample_saved(String("Anima-lora"), latent_path)
    print("  decode with: anima_decode_cli", latent_path, out_png)
    print("  latent mean_abs=", _mean_abs(x), " std=", _std(x))
    print("  wall time:", dt_s, "s")
    print("============================================================")
