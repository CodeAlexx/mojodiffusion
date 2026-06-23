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
# Mojo 1.0.0b1, NVIDIA GPU. Tensor storage dtype is preserved at the public
# boundary; kernels load BF16/F16 as scalars, compute in F32, and store back to
# the input storage dtype.

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
# T6.1b — DISCRETE timestep sampling (OneTrainer Qwen path).
#
# Mirrors OneTrainer's ModelSetupNoiseMixin._get_timestep_discrete LOGIT_NORMAL
# branch + ModelSetupFlowMatchingMixin._add_noise_discrete EXACTLY for the qwen
# LoRA 24GB preset defaults:
#   - num_train_timesteps = 1000 (qwen scheduler_config.json).
#   - min_noising_strength=0, max_noising_strength=1 -> min_t=0, num_t=1000.
#   - LOGIT_NORMAL with noising_weight=0 (scale=1.0), noising_bias=0 (bias=0)
#     -> continuous timestep = sigmoid(N(0,1)) * 1000.
#   - shift remap: t = num_t*shift*t / ((shift-1)*t + num_t); with shift=1.0
#     (preset/scheduler default) this is the identity.
#   - .int() truncates -> idx in [0, 999].
# The trainer then derives, per _add_noise_discrete:
#   sigma          = (idx + 1) / 1000        (noise/latent blend coefficient)
#   model_timestep = idx / 1000              (transformer timestep input,
#                                             diffusers scales *1000 internally)
# This struct returns all three so the train loop matches OneTrainer bit-for-bit
# at the timestep level (RNG stream excepted; distribution-level parity gated).
@fieldwise_init
struct DiscreteTimestep(Copyable, Movable):
    var idx: Int           # truncated integer timestep index in [0, num_t-1]
    var sigma: Float32     # (idx+1)/num_train_timesteps — blend coefficient
    var model_t: Float32   # idx/num_train_timesteps — transformer timestep input


def sample_timestep_discrete_qwen(
    seed: UInt64, shift: Float32, num_train_timesteps: Int = 1000
) -> DiscreteTimestep:
    """OneTrainer-faithful discrete timestep for the Qwen LoRA preset.

    idx = int(sigmoid(N(0,1)) * num_t * shift / ((shift-1)*(sigmoid(N)*num_t) + num_t)).
    With shift=1.0 the shift factor is identity, so idx = int(sigmoid(N) * num_t).
    Returns (idx, sigma=(idx+1)/num_t, model_t=idx/num_t)."""
    var ks = _expand_key(seed)
    var d = _standard_normal_at(
        ks[0], ks[1], ks[2], ks[3], ks[4], ks[5], ks[6], ks[7], UInt64(0)
    )
    var num_t = Float64(num_train_timesteps)
    # continuous timestep before shift = sigmoid(N)*num_t (min_t=0, num_timestep=num_t).
    var cont = _sigmoid64(d.z) * num_t
    var shift64 = Float64(shift)
    # OneTrainer shift remap (identity when shift==1):
    #   t = num_t * shift * t / ((shift-1)*t + num_t)
    var shifted = num_t * shift64 * cont / ((shift64 - Float64(1.0)) * cont + num_t)
    # .int() truncates toward zero (sigmoid in (0,1) -> shifted in (0, num_t)).
    var idx = Int(shifted)
    if idx < 0:
        idx = 0
    if idx >= num_train_timesteps:
        idx = num_train_timesteps - 1
    var sigma = Float32(Float64(idx + 1) / num_t)
    var model_t = Float32(Float64(idx) / num_t)
    return DiscreteTimestep(idx, sigma, model_t)


# ─────────────────────────────────────────────────────────────────────────────
# Wave 2A item 2g — selectable Uniform + Sigmoid timestep distributions.
#
# The production default stays sample_timestep_logit_normal (logit-normal +
# qwen-shift). These two additions match EDv2 timestep_dist.rs:
#   Uniform : t ~ U(0,1)   = rand 0.8.5 Standard<f32> = top-24-bits(word)/2^24
#             (timestep_dist.rs:172). One ChaCha word at word_pos 0.
#   Sigmoid : t = sigmoid(noising_weight * (z + noising_bias)), z ~ N(0,1)
#             (timestep_dist.rs:173-180; musubi-style). One Box-Muller draw.
# Both reuse the SAME ChaCha12 stream as the production path so the draw is
# deterministic per seed. Distribution-level (not byte) parity is gated
# statistically (0.999 cos of the histogram), per the task.
#
# Distribution kind enum (matches io reader / TrainConfig.timestep_distribution):
#   TSD_UNIFORM 0 ; TSD_SIGMOID 1 ; TSD_LOGIT_NORMAL 2
comptime TSD_UNIFORM = 0
comptime TSD_SIGMOID = 1
comptime TSD_LOGIT_NORMAL = 2


def sample_timestep_uniform(seed: UInt64) -> Float32:
    """t ~ U(0,1): rand 0.8.5 Standard<f32> = top-24-bits(word)/2^24.

    Mirrors timestep_dist.rs:172 (`rng.r#gen::<f32>()`). One ChaCha word at
    word position 0 of the seed's stream."""
    var ks = _expand_key(seed)
    var w0 = _chacha12_word_from_key(
        ks[0], ks[1], ks[2], ks[3], ks[4], ks[5], ks[6], ks[7], UInt64(0), 0
    )
    return Float32(_standard_f64(w0))


def sample_timestep_sigmoid(seed: UInt64, weight: Float32, bias: Float32) -> Float32:
    """t = sigmoid(weight * (z + bias)), z ~ N(0,1).

    Mirrors timestep_dist.rs:173-180 (musubi-style continuous sigmoid). With
    weight=1.8, bias=0 this is the Z-Image pipeline default. One Box-Muller
    N(0,1) draw from the seed's stream."""
    var ks = _expand_key(seed)
    var d = _standard_normal_at(
        ks[0], ks[1], ks[2], ks[3], ks[4], ks[5], ks[6], ks[7], UInt64(0)
    )
    var arg = Float64(weight) * (d.z + Float64(bias))
    return Float32(_sigmoid64(arg))


# ─────────────────────────────────────────────────────────────────────────────
# T6.2 — Flow-matching noised input + target.
# ─────────────────────────────────────────────────────────────────────────────
@fieldwise_init
struct FlowMatchOut(Movable):
    """Result of `flow_match_noise_target`: noised input and the v-target."""
    var x_t: Tensor
    var target: Tensor


def _flow_match_kernel[dtype: DType](
    x_t: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    target: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    latent: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    noise: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    sigma: Float32, n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var lat = rebind[Scalar[dtype]](latent[i]).cast[DType.float32]()
        var noi = rebind[Scalar[dtype]](noise[i]).cast[DType.float32]()
        # x_t = (1 - sigma)*latent + sigma*noise
        x_t[i] = rebind[x_t.element_type](
            ((Float32(1.0) - sigma) * lat + sigma * noi).cast[dtype]()
        )
        # target = noise - latent
        target[i] = rebind[target.element_type]((noi - lat).cast[dtype]())


def flow_match_noise_target(
    latent: Tensor, sigma: Float32, noise: Tensor, ctx: DeviceContext
) raises -> FlowMatchOut:
    """Build the flow-matching noised input and v-target.

      x_t    = (1 - sigma)*latent + sigma*noise
      target = noise - latent

    Matches train_qwenimage.rs:1093-1099 at the math level. Storage follows
    `latent.dtype()`: BF16/F16 inputs do F32 scalar math inside the kernel and
    write BF16/F16 outputs."""
    var n = latent.numel()
    if noise.numel() != n:
        raise Error("flow_match_noise_target: latent/noise numel mismatch")
    if latent.dtype() != noise.dtype():
        raise Error("flow_match_noise_target: latent/noise dtype mismatch")

    var storage_dtype = latent.dtype()
    var dt = storage_dtype.to_mojo_dtype()
    var x_buf = ctx.enqueue_create_buffer[DType.uint8](n * storage_dtype.byte_size())
    var t_buf = ctx.enqueue_create_buffer[DType.uint8](n * storage_dtype.byte_size())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))

    var grid = (n + _BLK - 1) // _BLK
    if dt == DType.float32:
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
        ctx.enqueue_function[
            _flow_match_kernel[DType.float32], _flow_match_kernel[DType.float32]
        ](XT, TG, LAT, NOI, sigma, n, grid_dim=grid, block_dim=_BLK)
    elif dt == DType.bfloat16:
        var XT = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var TG = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            t_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var LAT = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            latent.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var NOI = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            noise.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[
            _flow_match_kernel[DType.bfloat16], _flow_match_kernel[DType.bfloat16]
        ](XT, TG, LAT, NOI, sigma, n, grid_dim=grid, block_dim=_BLK)
    else:
        var XT = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var TG = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            t_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var LAT = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            latent.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var NOI = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            noise.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[
            _flow_match_kernel[DType.float16], _flow_match_kernel[DType.float16]
        ](XT, TG, LAT, NOI, sigma, n, grid_dim=grid, block_dim=_BLK)
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
        Tensor(x_buf^, shape_xt^, storage_dtype),
        Tensor(t_buf^, shape_tg^, storage_dtype),
    )


# ─────────────────────────────────────────────────────────────────────────────
# T6.3 — EMA update:  shadow = decay*shadow + (1-decay)*live  (in place).
# ─────────────────────────────────────────────────────────────────────────────
def _ema_update_kernel[dtype: DType](
    shadow: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    live: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    decay: Float32, n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var sv = rebind[Scalar[dtype]](shadow[i]).cast[DType.float32]()
        var lv = rebind[Scalar[dtype]](live[i]).cast[DType.float32]()
        var s = decay * sv + (Float32(1.0) - decay) * lv
        shadow[i] = rebind[shadow.element_type](s.cast[dtype]())


def ema_update(mut shadow: Tensor, live: Tensor, decay: Float32, ctx: DeviceContext) raises:
    """In-place EMA: shadow = decay*shadow + (1-decay)*live.

    Matches ema.rs / ParameterEma: decay=0.999, shadow=1.0, live=2.0 -> 1.001
    (the hand-checked single step). `shadow` is mutated; `live` is read-only.
    Both tensors must have the same storage dtype and numel. F32 shadow/master
    EMA remains supported; BF16/F16 shadows no longer route through a full F32
    device copy."""
    var n = shadow.numel()
    if live.numel() != n:
        raise Error("ema_update: shadow/live numel mismatch")
    if shadow.dtype() != live.dtype():
        raise Error("ema_update: shadow/live dtype mismatch")
    var dt = shadow.dtype().to_mojo_dtype()
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLK - 1) // _BLK
    if dt == DType.float32:
        var SH = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            shadow.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var LV = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            live.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[
            _ema_update_kernel[DType.float32], _ema_update_kernel[DType.float32]
        ](SH, LV, decay, n, grid_dim=grid, block_dim=_BLK)
    elif dt == DType.bfloat16:
        var SH = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            shadow.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var LV = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            live.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[
            _ema_update_kernel[DType.bfloat16], _ema_update_kernel[DType.bfloat16]
        ](SH, LV, decay, n, grid_dim=grid, block_dim=_BLK)
    else:
        var SH = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            shadow.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var LV = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            live.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[
            _ema_update_kernel[DType.float16], _ema_update_kernel[DType.float16]
        ](SH, LV, decay, n, grid_dim=grid, block_dim=_BLK)
    ctx.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# T6.4 — Gradient accumulation:  acc += new_grad  (in place, micro-batching).
# ─────────────────────────────────────────────────────────────────────────────
def _grad_accum_kernel[dtype: DType](
    acc: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    new_grad: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var av = rebind[Scalar[dtype]](acc[i]).cast[DType.float32]()
        var gv = rebind[Scalar[dtype]](new_grad[i]).cast[DType.float32]()
        acc[i] = rebind[acc.element_type]((av + gv).cast[dtype]())


def grad_accumulate(mut acc: Tensor, new_grad: Tensor, ctx: DeviceContext) raises:
    """In-place gradient accumulation: acc += new_grad.

    For micro-batch gradient accumulation: zero `acc` once, then call this per
    micro-step; divide by the accumulation count (or pre-scale grads) before the
    optimizer step, per the trainer's accumulation policy. Both tensors must
    have the same storage dtype and numel; `acc` is mutated, `new_grad` is
    read-only."""
    var n = acc.numel()
    if new_grad.numel() != n:
        raise Error("grad_accumulate: acc/new_grad numel mismatch")
    if acc.dtype() != new_grad.dtype():
        raise Error("grad_accumulate: acc/new_grad dtype mismatch")
    var dt = acc.dtype().to_mojo_dtype()
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLK - 1) // _BLK
    if dt == DType.float32:
        var ACC = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            acc.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var NG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            new_grad.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[
            _grad_accum_kernel[DType.float32], _grad_accum_kernel[DType.float32]
        ](ACC, NG, n, grid_dim=grid, block_dim=_BLK)
    elif dt == DType.bfloat16:
        var ACC = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            acc.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var NG = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            new_grad.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[
            _grad_accum_kernel[DType.bfloat16], _grad_accum_kernel[DType.bfloat16]
        ](ACC, NG, n, grid_dim=grid, block_dim=_BLK)
    else:
        var ACC = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            acc.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var NG = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            new_grad.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[
            _grad_accum_kernel[DType.float16], _grad_accum_kernel[DType.float16]
        ](ACC, NG, n, grid_dim=grid, block_dim=_BLK)
    ctx.synchronize()


def sample_timestep_logit_normal_scaled(seed: UInt64, std: Float32) -> Float32:
    """t = sigmoid(std * N(0,1)), clamped to [1/1000, 1].

    The DiffSynth-Studio Ideogram-4 training-time distribution
    (diffsynth/diffusion/flow_match.py set_timesteps_ideogram4: sigma =
    sigmoid(mean + std*z) with mean = mu + 0.5*log(pixels/512^2); at 512px
    mu=0 -> mean=0, std=1.5). Same RNG stream discipline as
    sample_timestep_logit_normal above."""
    var ks = _expand_key(seed)
    var d = _standard_normal_at(
        ks[0], ks[1], ks[2], ks[3], ks[4], ks[5], ks[6], ks[7], UInt64(0)
    )
    var t = _sigmoid64(Float64(std) * d.z)
    if t < Float64(1.0) / Float64(1000.0):
        t = Float64(1.0) / Float64(1000.0)
    if t > Float64(1.0):
        t = Float64(1.0)
    return Float32(t)
