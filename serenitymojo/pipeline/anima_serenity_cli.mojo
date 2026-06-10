# serenitymojo/pipeline/anima_serenity_cli.mojo
#
# UI-driven CLI adapter for the Anima 1024×1024 image model.
#
# ──────────────────────────────────────────────────────────────────────────────
# Contract (the UI bridge calls it exactly this way):
#
#   anima_serenity_cli  <config.json>  <lora|->  <sample_prompts.json>  <prompt_id>  <out.png>
#
#   argv[1]  config JSON path — ACCEPTED, IGNORED TODAY (model dirs are comptime
#            constants in anima_contract.mojo).
#
#   argv[2]  LoRA safetensors path, or "-" / "base" / "" for the frozen base model.
#            Passed through to _load_and_denoise as `lora_arg`.
#
#   argv[3]  sample_prompts JSON (serenity.sample_prompts.v1 schema).
#            Read with `read_sample_prompt_config`.
#
#   argv[4]  Prompt id/label to select from the JSON, or "" for the first entry.
#
#   argv[5]  Output PNG path.
#
# ──────────────────────────────────────────────────────────────────────────────
# KEY ARCHITECTURAL FACT: Anima CANNOT encode a prompt string at runtime in
# pure Mojo.  The `anima_text_context.mojo` pipeline requires THREE pre-tokenized
# integer arrays (qwen_ids, qwen_mask, t5_ids) that come from a Python sidecar
# (parity/anima_text_context_tokens.py), because Qwen2Tokenizer and
# T5TokenizerFast are NOT ported to Mojo.  Therefore this adapter reads the
# pre-encoded context-sidecar path from the SamplePrompt.caps_pos field and
# treats the JSON `prompt` text as a human-readable label only.  The
# `caps_neg` field supplies the unconditional context (empty-prompt sidecar).
# This mirrors what `anima_sample_cli.mojo` does when passed a .json arg.
#
# ──────────────────────────────────────────────────────────────────────────────
# Request fields honored vs fixed:
#
#   HONORED at runtime:
#     • lora path (argv[2])      — forwarded to _load_and_denoise; "base"/"-"
#                                  uses the zero-B overlay (pure base DiT).
#     • caps_pos                 — context-cond sidecar read from SamplePrompt.
#     • caps_neg                 — uncond context sidecar read from SamplePrompt.
#     • seed                     — from SamplePrompt.seed.
#
#   FIXED at comptime (from anima_sample_cli / anima_contract):
#     • steps  = NUM_STEPS  (30)
#     • cfg    = CFG_SCALE  (4.5)
#     • width  = 1024       (LATENT_HW * 8)
#     • height = 1024       (LATENT_HW * 8)
#
# ──────────────────────────────────────────────────────────────────────────────
# Generate path: REAL, not a stub.
#   Reuses the full _load_and_denoise helper from anima_sample_cli:
#     load base DiT (BF16) → load LoRA overlay → 28 resident blocks → RoPE →
#     30-step Euler CFG denoise → latent host F32.
#   Then immediately VAE-decodes in the SAME process (DiT tensors freed by
#   _load_and_denoise before it returns, matching the process-separation intent
#   of anima_decode_cli without needing a second executable).
#   On 24 GB GPUs the DiT resident blocks (~3.7 GiB BF16) must be freed before
#   the VAE's 3D-conv upsample (~multi-GiB peak).  The scoped helper ensures
#   this automatically.
#
# Build:
#   cd /home/alex/mojodiffusion && pixi run mojo build -I . \
#     -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/pipeline/anima_serenity_cli.mojo \
#     -o /tmp/anima_serenity_cli

from std.sys import argv
from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, exp as fexp
from std.memory import ArcPointer
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.tensor_algebra import reshape

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
    ANIMA_LATENT_H, ANIMA_LATENT_W, ANIMA_VAE_PATH,
)
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.training.sample_prompt_config import (
    SamplePrompt, SamplePromptConfig, read_sample_prompt_config,
)
from serenitymojo.training.progress_display import (
    print_sample_setup, print_sample_step, print_sample_saved,
)


comptime TArc = ArcPointer[Tensor]

# ── Anima dims (from anima_contract) ─────────────────────────────────────────
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

# ── 1024×1024 inference contract ─────────────────────────────────────────────
comptime LATENT_HW = 128                    # 1024 / 8 = 128
comptime NH = LATENT_HW // PS              # 64
comptime NW = LATENT_HW // PS              # 64
comptime S_IMG = NH * NW                    # 4096 image tokens
comptime S_TXT = 512                        # trained context length
comptime NUM_STEPS = 30
comptime CFG_SCALE = Float32(4.5)
comptime DEFAULT_SEED = UInt64(42)

# ── LoRA recipe (rank=16, alpha=16, matches the trainer) ─────────────────────
comptime RANK = 16
comptime ALPHA = Float32(16.0)


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


# ── t_embedder forward (RAW sigma, mirrors the proven 1024 inference path) ────
struct _TEmb(Movable):
    var t_cond: List[Float32]
    var base_adaln: List[Float32]

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
        out[i] = fcos(angle)
        out[half + i] = fsin(angle)
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


# ── 3D-RoPE tables ─────────────────────────────────────────────────────────────
struct _Rope(Movable):
    var cos: Tensor
    var sin: Tensor

    def __init__(out self, var cos: Tensor, var sin: Tensor):
        self.cos = cos^
        self.sin = sin^


def _rope_tables(nh: Int, nw: Int, ctx: DeviceContext) raises -> _Rope:
    var half = Dh // 2
    var full_d = Dh
    var t_frames = 1
    var s_img = nh * nw

    var dim_h = full_d // 6 * 2
    var dim_w = dim_h
    var dim_t = full_d - 2 * dim_h
    var bins_t = dim_t // 2
    var bins_h = dim_h // 2
    var bins_w = dim_w // 2

    var base_theta: Float64 = 10000.0
    var h_exp = Float64(dim_h) / (Float64(dim_h) - 2.0)
    var w_exp = Float64(dim_w) / (Float64(dim_w) - 2.0)
    var h_ntk = fexp(flog(Float64(4.0)) * h_exp)
    var w_ntk = fexp(flog(Float64(4.0)) * w_exp)
    var t_ntk = Float64(1.0)
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
                    for _hd in range(H):
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
    var cos_t = Tensor.from_host(cosl, [B * s_img * H, half], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sinl, [B * s_img * H, half], STDtype.F32, ctx)
    return _Rope(cos_t^, sin_t^)


# ── patchify INPUT: channels-last [B,Hd,Wd,C] -> patches [B*N,68] ─────────────
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


# ── load a context key from a safetensors file, zero-padded to S_TXT=512 ────────
# Replicates _load_context_512 from anima_sample_cli verbatim.
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


# ── zero-B overlay: every B = 0 -> forward reduces to frozen base DiT ─────────
def _zero_b_set(set: AnimaLoraSet) -> AnimaLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        var src = set.ad[i].copy()
        for j in range(len(src.b)):
            src.b[j] = BFloat16(0.0)
        ad.append(src^)
    return AnimaLoraSet(ad^, set.num_blocks, set.rank)


# ── 30-step Euler CFG denoise ──────────────────────────────────────────────────
def _denoise(
    context_cond: List[Float32], context_uncond: List[Float32],
    lora_dev: AnimaLoraDeviceSet, base: AnimaStackBase,
    blocks: List[AnimaBlockWeights],
    rope: _Rope, seed: UInt64, ctx: DeviceContext,
) raises -> List[Float32]:
    var n_lat = B * LATENT_HW * LATENT_HW * C
    var x = _host_noise(n_lat, seed)
    print("  init noise mean_abs=", _mean_abs(x), " std=", _std(x))

    for step in range(NUM_STEPS):
        var step_t0 = perf_counter_ns()
        var sigma = Float32(1.0) - Float32(step) / Float32(NUM_STEPS)
        var sigma_next = Float32(1.0) - Float32(step + 1) / Float32(NUM_STEPS)
        var dt = sigma_next - sigma

        var patches = _patchify_in(x, B, LATENT_HW, LATENT_HW, C)
        var temb = _prepare_timestep(sigma, base, ctx)

        var out_c = anima_stack_lora_forward_device_resident_nosave[H, Dh, S_IMG, S_TXT](
            patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(), context_cond.copy(),
            base, blocks, lora_dev, rope.cos, rope.sin,
            B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
        )
        var v_c = _unpatchify_out(out_c, B, LATENT_HW, LATENT_HW, C)

        var out_u = anima_stack_lora_forward_device_resident_nosave[H, Dh, S_IMG, S_TXT](
            patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(), context_uncond.copy(),
            base, blocks, lora_dev, rope.cos, rope.sin,
            B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
        )
        var v_u = _unpatchify_out(out_u, B, LATENT_HW, LATENT_HW, C)

        for j in range(len(x)):
            var pred = v_u[j] + CFG_SCALE * (v_c[j] - v_u[j])
            x[j] = x[j] + dt * pred

        ctx.synchronize()
        var step_secs = Float64(perf_counter_ns() - step_t0) / 1.0e9
        var rate = Float64(0.0)
        if step_secs > 0.0:
            rate = Float64(1.0) / step_secs
        print_sample_step(
            String("Anima-serenity"), step + 1, NUM_STEPS, sigma, step_secs, rate,
        )
    return x^


# ── Scoped load + denoise: owns ALL DiT device tensors; returns only host
#    latent (channels-last [B,H,W,C] F32).  ALL DiT tensors are freed before
#    return so the VAE decode can use the full GPU budget.  ────────────────────
def _load_and_denoise(
    use_base: Bool, lora_arg: String,
    context_cond_path: String, context_uncond_path: String,
    seed: UInt64, ctx: DeviceContext,
) raises -> List[Float32]:
    var cfg = anima()
    print("\n--- Load base DiT:", cfg.checkpoint, "---")
    var st = SafeTensors.open(cfg.checkpoint)
    verify_anima_stack_shapes(st, ANIMA_DEPTH)
    var base = load_anima_stack_base(st, ctx)
    print("  base projections + t_embedder loaded")

    print("\n--- Load context (cond + uncond) ---")
    var context_cond = _load_context_512(context_cond_path, String("context_cond"), ctx)
    print("  ctx_cond std =", _ctx_std(context_cond))

    var context_uncond = List[Float32]()
    if context_uncond_path != String(""):
        # Try as context_uncond key first, then fall back to context_cond key
        # (uncond sidecars may store the empty-prompt embedding under either key).
        var st_u = SafeTensors.open(context_uncond_path)
        var names = st_u.names()
        var has_uncond_key = False
        for i in range(len(names)):
            if names[i] == String("context_uncond"):
                has_uncond_key = True
        if has_uncond_key:
            context_uncond = _load_context_512(
                context_uncond_path, String("context_uncond"), ctx
            )
        else:
            context_uncond = _load_context_512(
                context_uncond_path, String("context_cond"), ctx
            )
    else:
        print("  no uncond path -> using zero context (empty-prompt CFG)")
        for _ in range(B * S_TXT * JOINT):
            context_uncond.append(Float32(0.0))
    print("  ctx_uncond std =", _ctx_std(context_uncond))

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
                var x = Float32(lora.ad[i].b[j])
                babs += x if x >= 0.0 else -x
        print("  LoRA overlay loaded:", len(lora.ad), "adapters  |B|_1 =", babs,
              " scale=alpha/rank=", ALPHA / Float32(RANK))

    var lora_dev = anima_lora_set_to_device(lora, STDtype.BF16, ctx)

    print("\n--- Load 28 blocks resident (BF16) ---")
    var blocks = List[AnimaBlockWeights]()
    for bi in range(ANIMA_DEPTH):
        blocks.append(load_anima_block_weights_bf16_normf32(st, bi, ctx))
    print("  resident blocks:", len(blocks))

    var rope = _rope_tables(NH, NW, ctx)

    print("\n--- Denoise (", NUM_STEPS, "steps, CFG=", CFG_SCALE, ") ---")
    var x = _denoise(context_cond, context_uncond, lora_dev, base, blocks, rope, seed, ctx)
    ctx.synchronize()
    return x^


# ── convert channels-last [B,H,W,C] F32 to NCHW Tensor BF16 ─────────────────
def _bhwc_to_nchw_tensor(
    bhwc: List[Float32], ctx: DeviceContext
) raises -> Tensor:
    var nchw = List[Float32]()
    nchw.resize(B * C * LATENT_HW * LATENT_HW, Float32(0.0))
    for b in range(B):
        for hd in range(LATENT_HW):
            for wd in range(LATENT_HW):
                for c in range(C):
                    var src = ((b * LATENT_HW + hd) * LATENT_HW + wd) * C + c
                    var dst = ((b * C + c) * LATENT_HW + hd) * LATENT_HW + wd
                    nchw[dst] = bhwc[src]
    var sh = List[Int]()
    sh.append(1)
    sh.append(C)
    sh.append(LATENT_HW)
    sh.append(LATENT_HW)
    var t = Tensor.from_host(nchw, sh^, STDtype.BF16, ctx)
    return t^


# ── Prompt selection ──────────────────────────────────────────────────────────

def _select_prompt(
    sample_cfg: SamplePromptConfig, wanted: String
) raises -> SamplePrompt:
    if len(sample_cfg.prompts) == 0:
        raise Error("anima_serenity_cli: sample prompt JSON has no prompts")
    if wanted == String(""):
        return sample_cfg.prompts[0].copy()
    for i in range(len(sample_cfg.prompts)):
        if sample_cfg.prompts[i].label == wanted:
            return sample_cfg.prompts[i].copy()
    raise Error(String("anima_serenity_cli: prompt id not found: ") + wanted)


def _load_prompt(
    path: String, wanted: String,
    mut context_cond_path: String,
    mut context_uncond_path: String,
    mut seed: UInt64,
    mut prompt_text: String,
) raises:
    var sample_cfg = read_sample_prompt_config(path)
    var p = _select_prompt(sample_cfg, wanted)
    if p.frames != 1:
        raise Error("anima_serenity_cli: only image prompts (frames=1) are supported")
    if p.caps_pos == String(""):
        raise Error(
            String("anima_serenity_cli: prompt '") + p.label
            + String("' has no caps.positive sidecar; Anima requires a ")
            + String("pre-encoded context (Qwen2+T5 tokenizers are not yet ")
            + String("ported to Mojo — run parity/anima_text_context_tokens.py)")
        )
    context_cond_path = p.caps_pos.copy()
    context_uncond_path = p.caps_neg.copy()
    seed = p.seed
    prompt_text = p.prompt.copy()
    print(
        "  [info] sample prompt requests steps=", p.steps,
        "cfg=", p.cfg, "seed=", p.seed,
        "size=", p.width, "x", p.height,
        "-> steps/cfg/size fixed at comptime; caps_pos/caps_neg/seed honored.",
    )


# ── Main entry ────────────────────────────────────────────────────────────────

def main() raises:
    var a = argv()
    if len(a) < 6:
        print(
            "usage: anima_serenity_cli <config.json> <lora|-> <sample_prompts.json>"
            " <prompt_id> <out.png>"
        )
        print("  argv[1] config       — accepted, ignored (model dirs are comptime)")
        print("  argv[2] lora         — LoRA .safetensors path, or '-'/'base' for frozen base")
        print("  argv[3] prompts      — serenity.sample_prompts.v1 JSON")
        print("  argv[4] id           — prompt label, or '' for first")
        print("  argv[5] out.png      — output image path")
        print("")
        print("  NOTE: Anima requires a pre-encoded text context sidecar (caps_pos /")
        print("  caps_neg in the sample_prompts JSON).  Raw prompt strings cannot be")
        print("  encoded at runtime because Qwen2Tokenizer and T5TokenizerFast are not")
        print("  ported to Mojo.  Use parity/anima_text_context_tokens.py to build the")
        print("  sidecar .safetensors, then anima_text_context to produce the context.")
        raise Error("anima_serenity_cli: need exactly 5 arguments")

    # argv[1]: config — accepted, not used today.
    var _config_path = String(a[1])

    # argv[2]: lora path or sentinel.
    var lora_raw = String(a[2])
    var lora_arg = String("")
    var use_base = False
    if lora_raw == String("-") or lora_raw == String("base") or lora_raw == String(""):
        use_base = True
    else:
        lora_arg = lora_raw

    # argv[3]: sample prompts JSON.
    var prompts_json = String(a[3])

    # argv[4]: prompt id.
    var prompt_id = String(a[4])

    # argv[5]: output PNG.
    var out_png = String(a[5])

    # Load prompt metadata from the JSON.
    var context_cond_path = String("")
    var context_uncond_path = String("")
    var seed = DEFAULT_SEED
    var prompt_text = String("")
    _load_prompt(
        prompts_json, prompt_id,
        context_cond_path, context_uncond_path, seed, prompt_text,
    )

    print("============================================================")
    print("Anima 1024x1024 Serenity CLI")
    print("  config    :", _config_path)
    print("  lora      :", lora_raw, "(base-only)" if use_base else "(PEFT overlay)")
    print("  prompts   :", prompts_json, " id:", prompt_id)
    print("  [prompt]  :", prompt_text, "  (log-only; runtime string not encoded)")
    print("  caps_pos  :", context_cond_path)
    print("  caps_neg  :", context_uncond_path)
    print("  seed      :", seed)
    print("  out       :", out_png)
    print("  S_IMG=", S_IMG, " S_TXT=", S_TXT, " steps=", NUM_STEPS,
          " CFG=", CFG_SCALE, " RANK=", RANK, " ALPHA=", ALPHA)
    print("============================================================")
    print_sample_setup(
        String("Anima-serenity"), String("anima"), NUM_STEPS, CFG_SCALE,
        S_IMG, ANIMA_DEPTH,
    )

    var ctx = DeviceContext()
    var t0 = perf_counter_ns()

    # ── Phase 1: DiT denoise ──────────────────────────────────────────────────
    # _load_and_denoise is scoped: ALL DiT device tensors (base projections, 28
    # resident blocks ~3.7 GiB BF16, LoRA device set, RoPE tables, per-forward
    # SDPA scratch) are FREED when it returns.  Only the host F32 latent survives.
    var x = _load_and_denoise(
        use_base, lora_arg, context_cond_path, context_uncond_path, seed, ctx,
    )

    var nbad = _count_nonfinite(x)
    print("\n  final latent: mean_abs=", _mean_abs(x), " std=", _std(x),
          " nonfinite=", nbad)
    if nbad > 0:
        raise Error("anima_serenity_cli: final latent contains non-finite values")

    # ── Phase 2: VAE decode ───────────────────────────────────────────────────
    # The DiT tensors are now freed.  Load the Qwen-Image VAE (wan21 keys) and
    # decode the latent to RGB.  decode_wan21_keys applies z/inv_std + mean
    # internally (OneTrainer AnimaModel.unscale_latents equivalent) — do NOT
    # pre-unscale here.
    print("\n--- VAE decode (Qwen-Image, wan21 keys) ---")
    var lat = _bhwc_to_nchw_tensor(x, ctx)
    var dec = QwenImageVaeDecoder[ANIMA_LATENT_H, ANIMA_LATENT_W].load_wan21_keys(
        String(ANIMA_VAE_PATH), ctx
    )
    var rgb = dec.decode_wan21_keys(lat, ctx)
    var sh = rgb.shape()
    print("  decoded RGB:", sh[2], "x", sh[3])

    save_png(rgb, out_png, ctx, ValueRange.SIGNED)

    var dt_s = Float64(perf_counter_ns() - t0) / 1.0e9
    print("\n============================================================")
    print_sample_saved(String("Anima-serenity"), out_png)
    print("  wall time:", dt_s, "s")
    print("============================================================")
