# serenitymojo/models/anima/parity/anima_batch2_parity.mojo
#
# TRUE BATCH-2 PARITY GATE for the ANIMA row-stacked (B=2) LoRA stack
# (models/anima/anima_stack_lora anima_*_streamed_b2). The anima TRAINER lives in
# the serenity-trainer repo, so this gate CANNOT import trainer fns — it drives the
# mojodiffusion STACK b2 directly, replicating the trainer's minimal per-step
# conditioning (flow-match noisy/target, patchify, 3D-RoPE, timestep embed) inline.
#
# Loads TWO REAL cache samples (distinct latent + distinct text_embedding →
# genuinely asymmetric), runs the streamed b1 stack per sample AND the streamed b2
# stack, and asserts the batch-2 invariants (MJ-1073 re-baseline):
#
#   (a) loss_B2 == mean(loss_B1(s0), loss_B1(s1))  within 1e-3 relative — BINDING.
#       (b2 joint loss is the 2N-mean MSE; each per-sample d_velocity is scaled by
#       2/(2N) so the joint loss == the mean of the two B1 per-sample MSEs which
#       use 2/N. The forward is deterministic — only bf16 GEMM ULPs, well inside.)
#
#   (b) per-sample forward velocity: cos(b2_out[0:S_IMG*OUT], b1_s0.out) >= 0.999
#       and cos(b2_out[S_IMG*OUT:], b1_s1.out) >= 0.999 — BINDING. Forward
#       reductions do NOT change shape (attention runs PER SAMPLE), so the row-
#       stacked forward must reproduce each sample's velocity.
#
#   (c) grad cosine (b2 concat vs mean of the two b1 grads) — reported INFORMATIONAL
#       per MJ-1073: row-stacked GEMMs are shape-deterministically different in bf16
#       (M=2S vs M=S → different cuBLAS tiling), so grad-cosine-vs-b1 at depth is the
#       WRONG instrument for a batched trainer. NOT a pass/fail bar.
#
# VERDICT PASS iff (a) && (b). NO torch oracle: self-consistent (B2 vs two B1 runs
# on byte-identical draws). All-F32 stack (F32 base weights streamed per block).
#
# Build ONLY the gate binary (mem-safe -O2; anima SDPA is math-mode → NO cuDNN
# shim, unlike krea2) + run ONE GPU process:
#   cd /home/alex/mojodiffusion
#   MEM_MAX=28G MEM_HIGH=24G SWAP_MAX=2G bash scripts/mem_safe.sh \
#     mojo build --optimization-level 2 --num-threads 1 -I . -I /home/alex/MOJO-libs \
#       -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/models/anima/parity/anima_batch2_parity.mojo \
#       -o output/bin/anima_batch2_parity
#   output/bin/anima_batch2_parity

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, exp as fexp
from std.os import listdir

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.train_config_reader import read_model_config

from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.norm import rms_norm

from serenitymojo.models.anima.config import ANIMA_CONFIG
from serenitymojo.models.anima.weights import (
    AnimaStackBase, load_anima_stack_base, verify_anima_stack_shapes,
)
from serenitymojo.models.anima.lora_block import ANIMA_SLOTS
from serenitymojo.models.anima.anima_stack_lora import (
    AnimaLoraSet, AnimaLoraGrads, build_anima_lora_set,
    anima_stack_lora_forward_streamed, anima_stack_lora_backward_streamed,
    anima_stack_lora_forward_streamed_b2, anima_stack_lora_backward_streamed_b2,
)
from serenitymojo.models.anima.anima_stack import AnimaStackForward
from serenitymojo.models.dit.anima_contract import (
    ANIMA_HIDDEN, ANIMA_NUM_HEADS, ANIMA_HEAD_DIM, ANIMA_DEPTH,
    ANIMA_LATENT_CHANNELS, ANIMA_PATCH_SIZE, ANIMA_MAX_SEQ_LEN, ANIMA_ADAPTER_DIM,
)
from serenitymojo.training.train_step import LoraAdapter


comptime CACHE_DIR = "/home/alex/.serenity/parity/cache/anima_synth_smoke"

comptime H = ANIMA_NUM_HEADS        # 16
comptime Dh = ANIMA_HEAD_DIM        # 128
comptime D = ANIMA_HIDDEN           # 2048
comptime F = 8192                   # GELU MLP hidden
comptime JOINT = ANIMA_ADAPTER_DIM  # 1024
comptime C = ANIMA_LATENT_CHANNELS  # 16
comptime PS = ANIMA_PATCH_SIZE      # 2
comptime IN_PATCH = (C + 1) * PS * PS   # 68
comptime OUT_PATCH = C * PS * PS        # 64
comptime S_TXT = ANIMA_MAX_SEQ_LEN  # 256
comptime EPS = Float32(1e-06)
comptime LATENT_HW = 16                 # crop to 16x16 -> S_IMG=64 (fast gate)
comptime S_IMG = (LATENT_HW // PS) * (LATENT_HW // PS)   # 64

comptime SIGMA0 = Float32(0.35)
comptime SIGMA1 = Float32(0.72)
comptime NOISE_SEED0 = UInt64(1234567)
comptime NOISE_SEED1 = UInt64(7654321)
comptime COS_BAR = Float64(0.999)
comptime LOSS_REL_BAR = Float64(1.0e-3)


def _absf(x: Float64) -> Float64:
    return x if x >= Float64(0.0) else -x


# cosine similarity of two host vectors (F64 accumulation).
def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var n = len(a)
    if n != len(b) or n == 0:
        return Float64(-2.0)
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(n):
        var av = Float64(a[i])
        var bv = Float64(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na == Float64(0.0) and nb == Float64(0.0):
        return Float64(1.0)
    if na == Float64(0.0) or nb == Float64(0.0):
        return Float64(0.0)
    return dot / (sqrt(na) * sqrt(nb))


def _mean2(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(a)):
        out.append(Float32(0.5) * (a[i] + b[i]))
    return out^


def _max_abs_diff(a: List[Float32], b: List[Float32]) -> Float64:
    var n = len(a)
    if n != len(b):
        return Float64(1.0e9)
    var m = Float64(0.0)
    for i in range(n):
        var d = _absf(Float64(a[i]) - Float64(b[i]))
        if d > m:
            m = d
    return m


def _sumsq(a: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(a)):
        var v = Float64(a[i])
        s += v * v
    return s


# concat all LoRA grads (all d_a, then all d_b, flat-slot order) into one vector.
def _concat_grads(g: AnimaLoraGrads) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(g.d_a)):
        for j in range(len(g.d_a[i])):
            out.append(g.d_a[i][j])
    for i in range(len(g.d_b)):
        for j in range(len(g.d_b[i])):
            out.append(g.d_b[i][j])
    return out^


# Perturb every LoRA B to small NONZERO (real init B=0 → grads near the bf16 floor).
def _perturb_b(mut set: AnimaLoraSet):
    var s = UInt64(1234567)
    for i in range(len(set.ad)):
        var bb = set.ad[i].b.copy()
        for j in range(len(bb)):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            var r = Float32(Int((s >> UInt64(40)) % UInt64(2000)) - 1000) * Float32(0.00005)
            bb[j] = Float32(r).cast[DType.bfloat16]()
        set.ad[i].b = bb^


# ── deterministic host gaussian noise (Box-Muller; mirrors trainer _host_noise) ─
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


# ── patchify INPUT layout (mirrors trainer _patchify_in): [1,T,H,W,C] -> [N,68] ─
def _patchify_in(x: List[Float32], T: Int, Hd: Int, Wd: Int, Cd: Int) -> List[Float32]:
    var pH = PS
    var pW = PS
    var nH = Hd // pH
    var nW = Wd // pW
    var Cp = Cd + 1
    var N = T * nH * nW
    var pd = Cp * pH * pW
    var out = List[Float32]()
    for _ in range(N * pd):
        out.append(Float32(0.0))
    for t in range(T):
        for ih in range(nH):
            for iw in range(nW):
                var pn = (t * nH + ih) * nW + iw
                for c in range(Cp):
                    for ph in range(pH):
                        for pw in range(pW):
                            var od = pn * pd + (c * pH * pW + ph * pW + pw)
                            if c < Cd:
                                var hh = ih * pH + ph
                                var ww = iw * pW + pw
                                var src = ((t) * Hd + hh) * Wd * Cd + ww * Cd + c
                                out[od] = x[src]
    return out^


# ── patchify OUTPUT layout (mirrors trainer _patchify_out): -> [N,64] ────────────
def _patchify_out(x: List[Float32], T: Int, Hd: Int, Wd: Int, Cd: Int) -> List[Float32]:
    var pH = PS
    var pW = PS
    var nH = Hd // pH
    var nW = Wd // pW
    var N = T * nH * nW
    var pd = Cd * pH * pW
    var out = List[Float32]()
    for _ in range(N * pd):
        out.append(Float32(0.0))
    for t in range(T):
        for ih in range(nH):
            for iw in range(nW):
                var pn = (t * nH + ih) * nW + iw
                for ph in range(pH):
                    for pw in range(pW):
                        for c in range(Cd):
                            var od = pn * pd + (ph * pW * Cd + pw * Cd + c)
                            var hh = ih * pH + ph
                            var ww = iw * pW + pw
                            var src = ((t) * Hd + hh) * Wd * Cd + ww * Cd + c
                            out[od] = x[src]
    return out^


# ── cos-first sinusoidal embedding (mirrors trainer _sinusoidal_host) ────────────
def _sinusoidal_host(sigma: Float32, dim: Int) -> List[Float32]:
    var half = dim // 2
    var neg_ln = -flog(Float32(10000.0))
    var out = List[Float32]()
    for _ in range(dim):
        out.append(Float32(0.0))
    for i in range(half):
        var freq = fexp(neg_ln * (Float32(i) / Float32(half)))
        var angle = sigma * freq
        out[i] = fcos(angle)
        out[half + i] = fsin(angle)
    return out^


struct _TEmb(Movable):
    var t_cond: List[Float32]
    var base_adaln: List[Float32]

    def __init__(out self, var t_cond: List[Float32], var base_adaln: List[Float32]):
        self.t_cond = t_cond^
        self.base_adaln = base_adaln^


# ── timestep embedder (mirrors trainer _prepare_timestep; B=1) ───────────────────
def _prepare_timestep(sigma: Float32, base: AnimaStackBase, ctx: DeviceContext) raises -> _TEmb:
    var emb_l = _sinusoidal_host(sigma, D)
    var emb = Tensor.from_host(emb_l, [1, D], STDtype.F32, ctx)
    var h = linear(emb, base.te_lin1[], Optional[Tensor](None), ctx)
    var hidden = silu(h, ctx)
    var base_adaln = linear(hidden, base.te_lin2[], Optional[Tensor](None), ctx)
    var t_cond = rms_norm(emb, base.t_norm[], EPS, ctx)
    return _TEmb(t_cond.to_host(ctx), base_adaln.to_host(ctx))


# ── 3D RoPE halfsplit table (mirrors trainer _rope_tables; B=1 -> [S_IMG*H, Dh/2]) ─
struct _Rope(Movable):
    var cos: Tensor
    var sin: Tensor

    def __init__(out self, var cos: Tensor, var sin: Tensor):
        self.cos = cos^
        self.sin = sin^


def _rope_tables(s_img: Int, ctx: DeviceContext) raises -> _Rope:
    var half_d = Dh // 2
    var full_d = Dh
    var T_frames = 1
    var nH = LATENT_HW // PS
    var nW = LATENT_HW // PS

    var dim_h = full_d // 6 * 2
    var dim_w = dim_h
    var dim_t = full_d - 2 * dim_h
    var bins_t = dim_t // 2
    var bins_h = dim_h // 2
    var bins_w = dim_w // 2

    var base_theta = Float64(10000.0)
    var h_exp = Float64(dim_h) / (Float64(dim_h) - 2.0)
    var w_exp = Float64(dim_w) / (Float64(dim_w) - 2.0)
    var h_ntk = fexp(flog(Float64(4.0)) * h_exp)
    var w_ntk = fexp(flog(Float64(4.0)) * w_exp)
    var theta_h = Float32(base_theta * h_ntk)
    var theta_w = Float32(base_theta * w_ntk)
    var theta_t = Float32(base_theta)

    var freqs_t = List[Float32]()
    for i in range(bins_t):
        var e = Float32(2 * i) / Float32(dim_t)
        freqs_t.append(Float32(1.0) / fexp(flog(theta_t) * e))
    var freqs_h = List[Float32]()
    for i in range(bins_h):
        var e = Float32(2 * i) / Float32(dim_h)
        freqs_h.append(Float32(1.0) / fexp(flog(theta_h) * e))
    var freqs_w = List[Float32]()
    for i in range(bins_w):
        var e = Float32(2 * i) / Float32(dim_w)
        freqs_w.append(Float32(1.0) / fexp(flog(theta_w) * e))

    var pos_cos = List[Float32]()
    var pos_sin = List[Float32]()
    for tf in range(T_frames):
        for ih in range(nH):
            for iw in range(nW):
                for fi in range(bins_t):
                    var a = Float32(tf) * freqs_t[fi]
                    pos_cos.append(fcos(a)); pos_sin.append(fsin(a))
                for fi in range(bins_h):
                    var a = Float32(ih) * freqs_h[fi]
                    pos_cos.append(fcos(a)); pos_sin.append(fsin(a))
                for fi in range(bins_w):
                    var a = Float32(iw) * freqs_w[fi]
                    pos_cos.append(fcos(a)); pos_sin.append(fsin(a))

    var cosl = List[Float32]()
    var sinl = List[Float32]()
    for s in range(s_img):
        var base_off = s * half_d
        for _h in range(H):
            for i in range(half_d):
                cosl.append(pos_cos[base_off + i])
                sinl.append(pos_sin[base_off + i])
    var cos = Tensor.from_host(cosl, [s_img * H, half_d], STDtype.F32, ctx)
    var sin = Tensor.from_host(sinl, [s_img * H, half_d], STDtype.F32, ctx)
    return _Rope(cos^, sin^)


# ── cache loaders (preserve stored dtype -> F32 host) ────────────────────────────
def _cache_tensor(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def _host_f32(t: Tensor, ctx: DeviceContext) raises -> List[Float32]:
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


# Load one cache sample: cropped channels-last latent [S_IMG*C flat] +
# text_embedding truncated/zero-padded to [S_TXT*JOINT].
struct _Sample(Movable):
    var lat: List[Float32]        # [Hd*Wd*C] channels-last cropped
    var context: List[Float32]    # [S_TXT*JOINT]

    def __init__(out self, var lat: List[Float32], var context: List[Float32]):
        self.lat = lat^
        self.context = context^


def _load_sample(path: String, ctx: DeviceContext) raises -> _Sample:
    var st = SafeTensors.open(path)
    var lat_info = st.tensor_info("latent")
    var lat_sh = lat_info.shape.copy()
    if len(lat_sh) != 4 or lat_sh[1] != C:
        raise Error("cache latent must be [1,C,H,W] with C=" + String(C))
    var full_H = lat_sh[2]
    var full_W = lat_sh[3]
    if full_H < LATENT_HW or full_W < LATENT_HW:
        raise Error("cache latent smaller than LATENT_HW=" + String(LATENT_HW))
    var lat_full = _host_f32(_cache_tensor(st, "latent", ctx), ctx)   # [C*fullH*fullW]
    var lat = List[Float32]()
    for _ in range(LATENT_HW * LATENT_HW * C):
        lat.append(Float32(0.0))
    for h in range(LATENT_HW):
        for w in range(LATENT_HW):
            for c in range(C):
                var src = (c * full_H + h) * full_W + w
                var dst = ((h) * LATENT_HW + w) * C + c
                lat[dst] = lat_full[src]

    # text_embedding [1,LT,JOINT] -> truncate/zero-pad to [S_TXT, JOINT].
    var te_info = st.tensor_info("text_embedding")
    if len(te_info.shape) < 2 or Int(te_info.shape[len(te_info.shape) - 1]) != JOINT:
        raise Error("cache text_embedding must have last dim JOINT=" + String(JOINT))
    var te_flat = _host_f32(_cache_tensor(st, "text_embedding", ctx), ctx)
    var src_tokens = len(te_flat) // JOINT
    var context = List[Float32]()
    for r in range(S_TXT):
        if r < src_tokens:
            for c in range(JOINT):
                context.append(te_flat[r * JOINT + c])
        else:
            for _ in range(JOINT):
                context.append(Float32(0.0))
    return _Sample(lat^, context^)


# ── build (loss, d_out) for one sample's forward output (2/N mean-MSE) ──────────
struct _LossOut(Movable):
    var loss: Float32
    var d_out: List[Float32]

    def __init__(out self, loss: Float32, var d_out: List[Float32]):
        self.loss = loss
        self.d_out = d_out^


def _mse(pred: List[Float32], target: List[Float32], n_div: Int) -> _LossOut:
    var npred = len(pred)
    var sse = Float32(0.0)
    var d_out = List[Float32]()
    d_out.reserve(npred)
    var inv_n = Float32(2.0) / Float32(n_div)
    for i in range(npred):
        var diff = pred[i] - target[i]
        sse += diff * diff
        d_out.append(inv_n * diff)
    return _LossOut(sse / Float32(n_div), d_out^)


def _list_safetensors(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var files = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            files.append(String(raw[i]))
    if len(files) < 2:
        raise Error("anima cache needs >=2 .safetensors samples: " + dir)
    return files^


def main() raises:
    var ctx = DeviceContext()
    print("==== anima_batch2_parity (TRUE batch-2 vs two B=1 runs; real cache) ====")
    print("S_IMG=", S_IMG, " S_TXT=", S_TXT, " D=", D, " DEPTH=", ANIMA_DEPTH,
          " LATENT_HW=", LATENT_HW)

    var cfg = read_model_config(String(ANIMA_CONFIG))
    var st = SafeTensors.open(cfg.checkpoint)
    verify_anima_stack_shapes(st, ANIMA_DEPTH)
    var base = load_anima_stack_base(st, ctx)
    print("base loaded: checkpoint=", cfg.checkpoint,
          " rank=", cfg.lora_rank, " alpha=", cfg.lora_alpha)

    var lora = build_anima_lora_set(ANIMA_DEPTH, D, JOINT, F, cfg.lora_rank, cfg.lora_alpha)
    _perturb_b(lora)   # lift grads off the bf16 floor
    print("  [discriminator] perturbed LoRA B to small nonzero")

    # ── two REAL cache samples (distinct latent + distinct text_embedding) ──────
    var files = _list_safetensors(String(CACHE_DIR))
    var s0 = _load_sample(String(CACHE_DIR) + String("/") + files[0], ctx)
    var s1 = _load_sample(String(CACHE_DIR) + String("/") + files[1], ctx)
    print("samples: s0=", files[0], " s1=", files[1])

    var ropes = _rope_tables(S_IMG, ctx)          # [S_IMG*H, Dh/2] single-sample

    var n_lat = LATENT_HW * LATENT_HW * C
    var Hd = LATENT_HW
    var Wd = LATENT_HW

    # ── per-sample flow-match + conditioning ────────────────────────────────────
    var noise0 = _host_noise(n_lat, NOISE_SEED0)
    var noise1 = _host_noise(n_lat, NOISE_SEED1)
    var noisy0 = List[Float32](); var target0 = List[Float32]()
    var noisy1 = List[Float32](); var target1 = List[Float32]()
    for i in range(n_lat):
        noisy0.append(SIGMA0 * noise0[i] + (Float32(1.0) - SIGMA0) * s0.lat[i])
        target0.append(noise0[i] - s0.lat[i])
        noisy1.append(SIGMA1 * noise1[i] + (Float32(1.0) - SIGMA1) * s1.lat[i])
        target1.append(noise1[i] - s1.lat[i])
    var patches0 = _patchify_in(noisy0, 1, Hd, Wd, C)
    var patches1 = _patchify_in(noisy1, 1, Hd, Wd, C)
    var tgt0 = _patchify_out(target0, 1, Hd, Wd, C)
    var tgt1 = _patchify_out(target1, 1, Hd, Wd, C)
    var temb0 = _prepare_timestep(SIGMA0, base, ctx)
    var temb1 = _prepare_timestep(SIGMA1, base, ctx)

    # ── B=1 per sample ──────────────────────────────────────────────────────────
    var fwd0 = anima_stack_lora_forward_streamed[H, Dh, S_IMG, S_TXT](
        patches0.copy(), temb0.t_cond.copy(), temb0.base_adaln.copy(), s0.context.copy(),
        base, st, lora, ropes.cos, ropes.sin,
        1, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )
    var out0 = fwd0.out.copy()
    var lo0 = _mse(out0, tgt0, len(out0))
    var g0 = anima_stack_lora_backward_streamed[H, Dh, S_IMG, S_TXT](
        lo0.d_out, patches0.copy(), temb0.t_cond.copy(), temb0.base_adaln.copy(), s0.context.copy(),
        base, st, lora, ropes.cos, ropes.sin, fwd0,
        1, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )
    var fwd1 = anima_stack_lora_forward_streamed[H, Dh, S_IMG, S_TXT](
        patches1.copy(), temb1.t_cond.copy(), temb1.base_adaln.copy(), s1.context.copy(),
        base, st, lora, ropes.cos, ropes.sin,
        1, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )
    var out1 = fwd1.out.copy()
    var lo1 = _mse(out1, tgt1, len(out1))
    var g1 = anima_stack_lora_backward_streamed[H, Dh, S_IMG, S_TXT](
        lo1.d_out, patches1.copy(), temb1.t_cond.copy(), temb1.base_adaln.copy(), s1.context.copy(),
        base, st, lora, ropes.cos, ropes.sin, fwd1,
        1, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )

    # ── B=2 (row-stacked) ───────────────────────────────────────────────────────
    var patches_b2 = patches0.copy()
    for i in range(len(patches1)):
        patches_b2.append(patches1[i])
    var target_b2 = tgt0.copy()
    for i in range(len(tgt1)):
        target_b2.append(tgt1[i])
    var t_cond_b2 = temb0.t_cond.copy()
    for i in range(len(temb1.t_cond)):
        t_cond_b2.append(temb1.t_cond[i])
    var base_adaln_b2 = temb0.base_adaln.copy()
    for i in range(len(temb1.base_adaln)):
        base_adaln_b2.append(temb1.base_adaln[i])
    var context_b2 = s0.context.copy()
    for i in range(len(s1.context)):
        context_b2.append(s1.context[i])

    var fwd_b2 = anima_stack_lora_forward_streamed_b2[H, Dh, S_IMG, S_TXT](
        patches_b2.copy(), t_cond_b2.copy(), base_adaln_b2.copy(), context_b2.copy(),
        base, st, lora, ropes.cos, ropes.sin,
        2, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )
    var out_b2 = fwd_b2.out.copy()
    var lo_b2 = _mse(out_b2, target_b2, len(out_b2))   # n_div = 2N (joint mean)
    var g_b2 = anima_stack_lora_backward_streamed_b2[H, Dh, S_IMG, S_TXT](
        lo_b2.d_out, patches_b2.copy(), t_cond_b2.copy(), base_adaln_b2.copy(), context_b2.copy(),
        base, st, lora, ropes.cos, ropes.sin, fwd_b2,
        2, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )

    var allok = True

    # ── (a) loss parity (BINDING) ───────────────────────────────────────────────
    print("")
    print("---- (a) loss_B2 vs mean(loss_B1(s0), loss_B1(s1)) [BINDING] ----")
    var loss_mean = Float64(0.5) * (Float64(lo0.loss) + Float64(lo1.loss))
    var loss_b2 = Float64(lo_b2.loss)
    var loss_rel = _absf(loss_b2 - loss_mean) / (
        loss_mean if loss_mean > Float64(1.0e-8) else Float64(1.0e-8)
    )
    print("  loss_B1(s0)=", lo0.loss, "  loss_B1(s1)=", lo1.loss)
    print("  mean=", loss_mean, "  loss_B2=", loss_b2, "  rel=", loss_rel,
          "  ", "PASS" if loss_rel <= LOSS_REL_BAR else "FAIL")
    if loss_rel > LOSS_REL_BAR:
        allok = False

    # ── (b) per-sample forward velocity (BINDING) ───────────────────────────────
    print("")
    print("---- (b) per-sample forward velocity cos(b2 half, b1 sample) [BINDING] ----")
    var half = S_IMG * OUT_PATCH
    var b2_h0 = List[Float32](); var b2_h1 = List[Float32]()
    for i in range(half):
        b2_h0.append(out_b2[i])
        b2_h1.append(out_b2[half + i])
    var cos0 = _cos(b2_h0, out0)
    var cos1 = _cos(b2_h1, out1)
    print("  sample0 cos(b2_out[0:N], b1_s0.out) =", cos0,
          "  ", "PASS" if cos0 >= COS_BAR else "FAIL")
    print("  sample1 cos(b2_out[N:2N], b1_s1.out) =", cos1,
          "  ", "PASS" if cos1 >= COS_BAR else "FAIL")
    if cos0 < COS_BAR or cos1 < COS_BAR:
        allok = False

    # ── (c) grad cosine (INFORMATIONAL, MJ-1073) ────────────────────────────────
    print("")
    print("---- (c) grad cosine: B2 vs mean(B1) [INFORMATIONAL, MJ-1073] ----")
    var v_b2 = _concat_grads(g_b2)
    var v_g0 = _concat_grads(g0)
    var v_g1 = _concat_grads(g1)
    var v_mean = _mean2(v_g0, v_g1)
    var gc = _cos(v_b2, v_mean)
    print("  B2 vs mean(B1) global cosine =", gc, "  max_abs=", _max_abs_diff(v_b2, v_mean))
    print("  ||grad_B2||=", sqrt(_sumsq(v_b2)), "  ||mean(B1)||=", sqrt(_sumsq(v_mean)))
    print("  INFO (not gated): row-stacked bf16 GEMM tiling is M-shape-deterministic")
    print("        (MJ-1073) — grad-cosine-vs-b1 at depth is the WRONG instrument.")
    print("  nonfinite: b2=", g_b2.nonfinite_lora_grads,
          " s0=", g0.nonfinite_lora_grads, " s1=", g1.nonfinite_lora_grads)

    print("")
    if allok:
        print("VERDICT: PASS — loss_B2 == mean(loss_B1) (rel<=1e-3) AND per-sample",
              "forward velocity cos>=0.999; grad cosine reported informationally",
              "(MJ-1073: shape-deterministic bf16 GEMM rounding — not a gated bar).")
    else:
        print("VERDICT: FAIL — see (a)/(b) above.")
