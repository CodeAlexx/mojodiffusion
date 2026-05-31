# training/schedule.mojo — training-loop POLICY primitives (Phase T6).
#
# FULL_PORT_TRAINING_PLAN.md §4 Phase T6: "Port EDv2 training/ (schedule,
# timestep dist, EMA, ...) once the engine is proven." These are the numeric
# policy pieces of the loop — NOT autograd, NOT model kernels. They sit on top
# of the tape engine + optimizers (training/optim.mojo).
#
# Ported to match the EDv2 qwenimage trainer EXACTLY
# (EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_qwenimage.rs):
#
#   timestep:   t = sigmoid(N(0,1)); t_shifted = t*shift / (1 + (shift-1)*t)
#               (logit_normal sampler `sample_one`, weight=0/scale=1 path, then
#                `apply_qwen_shift`, clamp to [1/1000, 1]).  (lines 391-413, 1075)
#   flow-match:  x_t = (1 - sigma)*latent + sigma*noise   (line 1093-1097)
#                target = noise - latent                  (line 1099)
#   loss target documented at the trainer header line 14: "MSE in F32 between
#   pred and target = noise - latent".
#
#   EMA:   shadow = decay*shadow + (1-decay)*live   (training/training_features/
#          ema.rs hand-check: decay=0.999, init=1, live=2 -> 1.001).
#   grad-accum:  acc += new_grad  (micro-batch gradient accumulation).
#
# ── Why a host-side RNG for the timestep ──────────────────────────────────────
# The timestep is a single F32 scalar per step. ops/random.mojo provides a
# DEVICE randn for filling latent-sized noise tensors; for one scalar we reuse
# the SAME underlying stream (rand 0.8.5 StdRng -> ChaCha12 -> Box-Muller) on the
# host so the draw is deterministic and reproducible across runs. The
# distribution (sigmoid(N(0,1)) then shift) is what gets gated statistically.
#
# ── Tensor mutation contract (mirrors optim.mojo) ────────────────────────────
# Tensor is move-only (Mojo 1.0.0b1). `ema_update` and `grad_accumulate` mutate
# the destination device buffer IN PLACE via `mut` (the move-friendly analogue
# of &mut Tensor). `flow_match_noise_target` builds two NEW tensors and returns
# them in a Movable struct (multi-return for move-only types).
#
# Mojo 1.0.0b1, NVIDIA GPU. F32 device buffers; LayoutTensor flat-index kernels
# built INLINE at each call site (a `def` helper cannot return a LayoutTensor —
# origin inference — same discipline as optim.mojo / linalg_backward).

from std.math import sqrt, log, cos, exp
from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _BLK = 256
comptime _DYN1 = Layout.row_major(-1)
comptime _TWO_PI = Float64(6.283185307179586476925286766559)


# ─────────────────────────────────────────────────────────────────────────────
# Host-side RNG: rand 0.8.5 StdRng(seed) -> ChaCha12 stream -> Box-Muller N(0,1).
# Mirrors ops/random.mojo `_std_rng_pair` so the scalar timestep draw matches the
# device noise stream byte-for-byte. Pure host arithmetic (no device).
# ─────────────────────────────────────────────────────────────────────────────
comptime _U24_SCALE = Float64(5.9604644775390625e-8)  # 1 / 2^24


@fieldwise_init
struct _PcgOut(Copyable, Movable):
    var state: UInt64
    var word: UInt32


def _rotl32(x: UInt32, n: Int) -> UInt32:
    return (x << UInt32(n)) | (x >> UInt32(32 - n))


def _rotr32_var(x: UInt32, n: Int) -> UInt32:
    var r = n & 31
    if r == 0:
        return x
    return (x >> UInt32(r)) | (x << UInt32(32 - r))


def _pcg32(state: UInt64) -> _PcgOut:
    var st = state * 6364136223846793005 + 11634580027462260723
    var xorshifted = UInt32(((st >> 18) ^ st) >> 27)
    var rot = Int((st >> 59) & 31)
    return _PcgOut(st, _rotr32_var(xorshifted, rot))


@fieldwise_init
struct _QrOut(Copyable, Movable):
    var a: UInt32
    var b: UInt32
    var c: UInt32
    var d: UInt32


def _quarter(a_in: UInt32, b_in: UInt32, c_in: UInt32, d_in: UInt32) -> _QrOut:
    var a = a_in
    var b = b_in
    var c = c_in
    var d = d_in
    a += b
    d = _rotl32(d ^ a, 16)
    c += d
    b = _rotl32(b ^ c, 12)
    a += b
    d = _rotl32(d ^ a, 8)
    c += d
    b = _rotl32(b ^ c, 7)
    return _QrOut(a, b, c, d)


def _chacha12_word_from_key(
    k0: UInt32, k1: UInt32, k2: UInt32, k3: UInt32,
    k4: UInt32, k5: UInt32, k6: UInt32, k7: UInt32,
    block: UInt64, offset: Int,
) -> UInt32:
    var s0 = UInt32(0x61707865)
    var s1 = UInt32(0x3320646E)
    var s2 = UInt32(0x79622D32)
    var s3 = UInt32(0x6B206574)
    var s4 = k0
    var s5 = k1
    var s6 = k2
    var s7 = k3
    var s8 = k4
    var s9 = k5
    var s10 = k6
    var s11 = k7
    var s12 = UInt32(block & 0xFFFFFFFF)
    var s13 = UInt32(block >> 32)
    var s14 = UInt32(0)
    var s15 = UInt32(0)

    var x0 = s0
    var x1 = s1
    var x2 = s2
    var x3 = s3
    var x4 = s4
    var x5 = s5
    var x6 = s6
    var x7 = s7
    var x8 = s8
    var x9 = s9
    var x10 = s10
    var x11 = s11
    var x12 = s12
    var x13 = s13
    var x14 = s14
    var x15 = s15

    for _ in range(6):
        var q = _quarter(x0, x4, x8, x12)
        x0 = q.a; x4 = q.b; x8 = q.c; x12 = q.d
        q = _quarter(x1, x5, x9, x13)
        x1 = q.a; x5 = q.b; x9 = q.c; x13 = q.d
        q = _quarter(x2, x6, x10, x14)
        x2 = q.a; x6 = q.b; x10 = q.c; x14 = q.d
        q = _quarter(x3, x7, x11, x15)
        x3 = q.a; x7 = q.b; x11 = q.c; x15 = q.d

        q = _quarter(x0, x5, x10, x15)
        x0 = q.a; x5 = q.b; x10 = q.c; x15 = q.d
        q = _quarter(x1, x6, x11, x12)
        x1 = q.a; x6 = q.b; x11 = q.c; x12 = q.d
        q = _quarter(x2, x7, x8, x13)
        x2 = q.a; x7 = q.b; x8 = q.c; x13 = q.d
        q = _quarter(x3, x4, x9, x14)
        x3 = q.a; x4 = q.b; x9 = q.c; x14 = q.d

    if offset == 0:
        return x0 + s0
    if offset == 1:
        return x1 + s1
    if offset == 2:
        return x2 + s2
    if offset == 3:
        return x3 + s3
    if offset == 4:
        return x4 + s4
    if offset == 5:
        return x5 + s5
    if offset == 6:
        return x6 + s6
    if offset == 7:
        return x7 + s7
    if offset == 8:
        return x8 + s8
    if offset == 9:
        return x9 + s9
    if offset == 10:
        return x10 + s10
    if offset == 11:
        return x11 + s11
    if offset == 12:
        return x12 + s12
    if offset == 13:
        return x13 + s13
    if offset == 14:
        return x14 + s14
    return x15 + s15


def _standard_f64(word: UInt32) -> Float64:
    # rand 0.8.5 Standard<f32>: top 24 bits / 2^24, widened to F64 for the
    # host transform (matches ops/random.mojo `_standard_f32` then F64 math).
    return Float64(Int(word >> 8)) * _U24_SCALE


@fieldwise_init
struct _NormalDraw(Copyable, Movable):
    """A single N(0,1) draw plus the advanced word position for the next draw."""
    var z: Float64
    var next_pair: UInt64


def _standard_normal_at(
    k0: UInt32, k1: UInt32, k2: UInt32, k3: UInt32,
    k4: UInt32, k5: UInt32, k6: UInt32, k7: UInt32,
    pair: UInt64,
) -> _NormalDraw:
    # One Box-Muller pair consumes two ChaCha words (u1, u2); we keep z0 only.
    var word_pos = pair * 2
    var block = word_pos // 16
    var offset = Int(word_pos % 16)
    var w0 = _chacha12_word_from_key(k0, k1, k2, k3, k4, k5, k6, k7, block, offset)
    var w1 = _chacha12_word_from_key(k0, k1, k2, k3, k4, k5, k6, k7, block, offset + 1)
    var u1 = _standard_f64(w0)
    var u2 = _standard_f64(w1)
    if u1 < Float64(1.0e-10):
        u1 = Float64(1.0e-10)
    var r = sqrt(Float64(-2.0) * log(u1))
    var theta = _TWO_PI * u2
    return _NormalDraw(r * cos(theta), pair + 1)


def _expand_key(seed: UInt64) -> List[UInt32]:
    # rand 0.8.5 SeedableRng::seed_from_u64 -> PCG32-expanded 32-byte ChaCha key.
    var ks = List[UInt32]()
    var p = _pcg32(seed)
    ks.append(p.word)
    for _ in range(7):
        p = _pcg32(p.state)
        ks.append(p.word)
    return ks^


def _sigmoid64(x: Float64) -> Float64:
    return Float64(1.0) / (Float64(1.0) + exp(-x))


# ─────────────────────────────────────────────────────────────────────────────
# T6.1 — Timestep sampling: logit-normal then qwen-shift remap.
# ─────────────────────────────────────────────────────────────────────────────
def sample_timestep_logit_normal(seed: UInt64, shift: Float32) -> Float32:
    """Sample one training timestep: t = sigmoid(N(0,1)) then qwen-shift remap.

    Matches the EDv2 qwenimage trainer default path EXACTLY:
      - `TimestepDistribution::LogitNormal` with weight=0 (scale=1.0), bias=0
        degenerates to `t = sigmoid(N(0,1))` (timestep_dist.rs:181-189).
      - `apply_qwen_shift(t, shift)` = `shift*t / (1 + (shift-1)*t)`, clamped to
        [1/1000, 1] (train_qwenimage.rs:411-414).
    With shift=1.0 (OneTrainer qwen preset default) the remap is the identity, so
    the output is exactly sigmoid(N(0,1)) clamped to [1/1000, 1].

    `seed` selects the deterministic draw (caller advances it per step). Returns
    sigma in [1/1000, 1].
    """
    var ks = _expand_key(seed)
    var d = _standard_normal_at(
        ks[0], ks[1], ks[2], ks[3], ks[4], ks[5], ks[6], ks[7], UInt64(0)
    )
    var t = _sigmoid64(d.z)  # logit-normal, scale=1, bias=0
    var shift64 = Float64(shift)
    var shifted = shift64 * t / (Float64(1.0) + (shift64 - Float64(1.0)) * t)
    # clamp to [1/1000, 1] (train_qwenimage.rs apply_qwen_shift).
    if shifted < Float64(1.0) / Float64(1000.0):
        shifted = Float64(1.0) / Float64(1000.0)
    if shifted > Float64(1.0):
        shifted = Float64(1.0)
    return Float32(shifted)


# ─────────────────────────────────────────────────────────────────────────────
# T6.2 — Flow-matching noised input + target.
# ─────────────────────────────────────────────────────────────────────────────
@fieldwise_init
struct FlowMatchOut(Movable):
    """Result of `flow_match_noise_target`: noised input and the v-target."""
    var x_t: Tensor
    var target: Tensor


def _flow_match_kernel(
    x_t: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    target: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    latent: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    noise: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    sigma: Float32, n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var lat = latent[i]
        var noi = noise[i]
        # x_t = (1 - sigma)*latent + sigma*noise
        x_t[i] = rebind[x_t.element_type]((Float32(1.0) - sigma) * lat + sigma * noi)
        # target = noise - latent
        target[i] = rebind[target.element_type](noi - lat)


def flow_match_noise_target(
    latent: Tensor, sigma: Float32, noise: Tensor, ctx: DeviceContext
) raises -> FlowMatchOut:
    """Build the flow-matching noised input and v-target (F32).

      x_t    = (1 - sigma)*latent + sigma*noise
      target = noise - latent

    Matches train_qwenimage.rs:1093-1099 byte-for-byte (F32 affine + add/sub).
    `latent` and `noise` must be F32 and the same numel. Returns two NEW
    tensors with `latent`'s shape (move-only multi-return)."""
    var n = latent.numel()
    if noise.numel() != n:
        raise Error("flow_match_noise_target: latent/noise numel mismatch")

    var x_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var t_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))

    var XT = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x_buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var TG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        t_buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var LAT = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        latent.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var NOI = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        noise.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var grid = (n + _BLK - 1) // _BLK
    ctx.enqueue_function[_flow_match_kernel, _flow_match_kernel](
        XT, TG, LAT, NOI, sigma, n, grid_dim=grid, block_dim=_BLK
    )
    ctx.synchronize()

    # Duplicate latent's shape into two fresh Lists (move-only Tensor needs an
    # owned shape each). `shape()` returns a copied List; bind once then dup.
    var lshape = latent.shape()
    var shape_xt = List[Int]()
    var shape_tg = List[Int]()
    for di in range(len(lshape)):
        shape_xt.append(lshape[di])
        shape_tg.append(lshape[di])
    return FlowMatchOut(
        Tensor(x_buf^, shape_xt^, STDtype.F32),
        Tensor(t_buf^, shape_tg^, STDtype.F32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# T6.3 — EMA update:  shadow = decay*shadow + (1-decay)*live  (in place).
# ─────────────────────────────────────────────────────────────────────────────
def _ema_update_kernel(
    shadow: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    live: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    decay: Float32, n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var s = decay * shadow[i] + (Float32(1.0) - decay) * live[i]
        shadow[i] = rebind[shadow.element_type](s)


def ema_update(mut shadow: Tensor, live: Tensor, decay: Float32, ctx: DeviceContext) raises:
    """In-place EMA: shadow = decay*shadow + (1-decay)*live.

    Matches ema.rs / ParameterEma: decay=0.999, shadow=1.0, live=2.0 -> 1.001
    (the hand-checked single step). `shadow` is mutated; `live` is read-only.
    Both must be F32 and same numel."""
    var n = shadow.numel()
    if live.numel() != n:
        raise Error("ema_update: shadow/live numel mismatch")
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var SH = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        shadow.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var LV = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        live.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var grid = (n + _BLK - 1) // _BLK
    ctx.enqueue_function[_ema_update_kernel, _ema_update_kernel](
        SH, LV, decay, n, grid_dim=grid, block_dim=_BLK
    )
    ctx.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# T6.4 — Gradient accumulation:  acc += new_grad  (in place, micro-batching).
# ─────────────────────────────────────────────────────────────────────────────
def _grad_accum_kernel(
    acc: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    new_grad: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        acc[i] = rebind[acc.element_type](acc[i] + new_grad[i])


def grad_accumulate(mut acc: Tensor, new_grad: Tensor, ctx: DeviceContext) raises:
    """In-place gradient accumulation: acc += new_grad.

    For micro-batch gradient accumulation: zero `acc` once, then call this per
    micro-step; divide by the accumulation count (or pre-scale grads) before the
    optimizer step, per the trainer's accumulation policy. Both tensors F32 and
    same numel; `acc` is mutated, `new_grad` is read-only."""
    var n = acc.numel()
    if new_grad.numel() != n:
        raise Error("grad_accumulate: acc/new_grad numel mismatch")
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var ACC = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        acc.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var NG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        new_grad.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var grid = (n + _BLK - 1) // _BLK
    ctx.enqueue_function[_grad_accum_kernel, _grad_accum_kernel](
        ACC, NG, n, grid_dim=grid, block_dim=_BLK
    )
    ctx.synchronize()
