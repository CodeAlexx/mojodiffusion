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
def sample_timestep_logit_normal(
    seed: UInt64,
    shift: Float32,
    min_noising_strength: Float64 = Float64(1.0) / Float64(1000.0),
    max_noising_strength: Float64 = Float64(1.0),
) -> Float32:
    """Sample one training timestep: t = sigmoid(N(0,1)) then qwen-shift remap.

    Matches the EDv2 qwenimage trainer default path EXACTLY:
      - `TimestepDistribution::LogitNormal` with weight=0 (scale=1.0), bias=0
        degenerates to `t = sigmoid(N(0,1))` (timestep_dist.rs:181-189).
      - `apply_qwen_shift(t, shift)` = `shift*t / (1 + (shift-1)*t)`, clamped to
        [min_noising_strength, max_noising_strength] (train_qwenimage.rs:411-414).
    With shift=1.0 (SerenityTrainer qwen preset default) the remap is the identity, so
    the output is exactly sigmoid(N(0,1)) clamped to the strength window.

    The clamp bounds are config-driven (min/max_noising_strength, the reference
    noising-strength window). The DEFAULTS are the exact F64 expressions the clamp
    used when it was hardcoded — `1/1000` and `1` — so a config that omits them (or
    passes the defaults) is byte-identical to the pre-config path (C13).

    `seed` selects the deterministic draw (caller advances it per step). Returns
    sigma in [min_noising_strength, max_noising_strength].
    """
    var ks = _expand_key(seed)
    var d = _standard_normal_at(
        ks[0], ks[1], ks[2], ks[3], ks[4], ks[5], ks[6], ks[7], UInt64(0)
    )
    var t = _sigmoid64(d.z)  # logit-normal, scale=1, bias=0
    var shift64 = Float64(shift)
    var shifted = shift64 * t / (Float64(1.0) + (shift64 - Float64(1.0)) * t)
    # clamp to [min_noising_strength, max_noising_strength]; defaults = the old
    # hardcoded apply_qwen_shift window (train_qwenimage.rs), byte-for-byte.
    if shifted < min_noising_strength:
        shifted = min_noising_strength
    if shifted > max_noising_strength:
        shifted = max_noising_strength
    return Float32(shifted)


def sample_timestep_logit_normal_win(
    seed: UInt64, shift: Float32, min_s: Float32, max_s: Float32
) -> Float32:
    """Config-threaded logit-normal draw: `min_s`/`max_s` are the TrainConfig
    min/max_noising_strength (sentinel -1 = unset). An unset bound falls back to
    the built-in [1/1000, 1] window computed in F64, so an all-unset config is
    byte-identical to the 2-arg path (C13). A set bound overrides it."""
    var lo = Float64(1.0) / Float64(1000.0)
    var hi = Float64(1.0)
    if min_s >= Float32(0.0):
        lo = Float64(min_s)
    if max_s >= Float32(0.0):
        hi = Float64(max_s)
    return sample_timestep_logit_normal(seed, shift, lo, hi)


def flux1_mu_train(image_seq_len: Int) -> Float64:
    """BFL FLUX.1 linear mu: 0.5 @ 256 tokens, 1.15 @ 4096 tokens. Verbatim port
    of sampling/flux1_dev.mojo:flux1_mu (the inference oracle) so dynamic-shift
    training matches inference bit-for-bit. seq_len<=0 => base_shift 0.5."""
    if image_seq_len <= 0:
        return Float64(0.5)
    var x1 = Float64(256.0)
    var y1 = Float64(0.5)
    var x2 = Float64(4096.0)
    var y2 = Float64(1.15)
    var m = (y2 - y1) / (x2 - x1)
    var b = y1 - m * x1
    return m * Float64(image_seq_len) + b


def dynamic_shift_from_seqlen(image_seq_len: Int) -> Float32:
    """Resolution-dependent flow-match shift = exp(flux1_mu(seq_len)). Passed as
    `shift` to sample_timestep_logit_normal it reproduces the BFL flux1_time_shift
    remap exactly (em/(em+1/t-1) == shift*t/(1+(shift-1)*t)), matching
    sampling/flux1_dev.mojo:flux1_dynamic_shift."""
    return Float32(exp(flux1_mu_train(image_seq_len)))


def effective_train_shift(
    dynamic_on: Bool, fixed_shift: Float32, image_seq_len: Int
) -> Float32:
    """Shift a trainer feeds the timestep sampler: resolution-adaptive dynamic
    shift when dynamic_timestep_shifting is on, else the fixed (config) shift —
    so default-off (dynamic_on=False) is byte-identical to the pre-T4 path."""
    if dynamic_on:
        return dynamic_shift_from_seqlen(image_seq_len)
    return fixed_shift


def _uniform_index(seed: UInt64, high_exclusive: Int) -> Int:
    if high_exclusive <= 1:
        return 0
    var ks = _expand_key(seed)
    var w = _chacha12_word_from_key(
        ks[0], ks[1], ks[2], ks[3], ks[4], ks[5], ks[6], ks[7], UInt64(0), 0
    )
    var u = _standard_f64(w)
    var idx = Int(u * Float64(high_exclusive))
    if idx < 0:
        return 0
    if idx >= high_exclusive:
        return high_exclusive - 1
    return idx


def sample_timestep_krea2_aitk_sigmoid_balanced(
    table_seed: UInt64, index_seed: UInt64, num_train_timesteps: Int = 1000
) raises -> Float32:
    """ai-toolkit Krea2 default timestep policy, distribution-level.

    Mirrors CustomFlowMatchEulerDiscreteScheduler.set_train_timesteps("sigmoid")
    plus BaseSDTrainProcess content_or_style="balanced" index sampling:
      timesteps = sort_desc((1 - sigmoid(randn(1000))) * 1000)
      idx       = randint(0, 999) for the default flowmatch min/max window
      sigma     = timesteps[idx] / 1000

    The RNG is Mojo's deterministic ChaCha stream, not PyTorch's CUDA generator,
    so this is oracle-behavior parity rather than same-seed byte replay.
    """
    if num_train_timesteps < 2:
        raise Error("sample_timestep_krea2_aitk_sigmoid_balanced: need >=2 timesteps")
    var ks = _expand_key(table_seed)
    var vals = List[Float64]()
    for i in range(num_train_timesteps):
        var d = _standard_normal_at(
            ks[0], ks[1], ks[2], ks[3], ks[4], ks[5], ks[6], ks[7], UInt64(i)
        )
        vals.append((Float64(1.0) - _sigmoid64(d.z)) * Float64(1000.0))

    # Mojo List has no sort on the deployed toolchain. 1000 values is tiny next
    # to a Krea step, and this keeps the reference's sorted-table/exclude-tail
    # behavior visible instead of collapsing it into a generic logit-normal draw.
    for i in range(1, len(vals)):
        var key = vals[i]
        var j = i - 1
        while j >= 0 and vals[j] < key:
            vals[j + 1] = vals[j]
            j -= 1
        vals[j + 1] = key

    # ai-toolkit's default flowmatch balanced path uses torch.randint(0, 999)
    # when num_train_timesteps=1000, excluding the final sorted entry.
    var idx = _uniform_index(index_seed, num_train_timesteps - 1)
    return Float32(vals[idx] / Float64(1000.0))


def sample_timestep_krea2_aitk_linear_balanced(
    index_seed: UInt64, num_train_timesteps: Int = 1000
) raises -> Float32:
    """ai-toolkit flowmatch timestep_type="linear", content_or_style="balanced".

    CustomFlowMatchEulerDiscreteScheduler builds
      timesteps = linspace(1000, 1, 1000)
    and the balanced flowmatch path samples torch.randint(0, 999), excluding the
    final table entry for the default min/max window.
    """
    if num_train_timesteps < 2:
        raise Error("sample_timestep_krea2_aitk_linear_balanced: need >=2 timesteps")
    var idx = _uniform_index(index_seed, num_train_timesteps - 1)
    var t = Float64(num_train_timesteps) - Float64(idx)
    return Float32(t / Float64(num_train_timesteps))


# ─────────────────────────────────────────────────────────────────────────────
# T6.1b — DISCRETE timestep sampling (SerenityTrainer Qwen path).
#
# Mirrors SerenityTrainer's ModelSetupNoiseMixin._get_timestep_discrete LOGIT_NORMAL
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
# This struct returns all three so the train loop matches SerenityTrainer bit-for-bit
# at the timestep level (RNG stream excepted; distribution-level parity gated).
@fieldwise_init
struct DiscreteTimestep(Copyable, Movable):
    var idx: Int           # truncated integer timestep index in [0, num_t-1]
    var sigma: Float32     # (idx+1)/num_train_timesteps — blend coefficient
    var model_t: Float32   # idx/num_train_timesteps — transformer timestep input


def sample_timestep_discrete_qwen(
    seed: UInt64, shift: Float32, num_train_timesteps: Int = 1000
) -> DiscreteTimestep:
    """SerenityTrainer-faithful discrete timestep for the Qwen LoRA preset.

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
    # SerenityTrainer shift remap (identity when shift==1):
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


# SerenityTrainer discrete-distribution tags (io/train_config_reader.mojo ints 3..5).
# Only INVERTED_PARABOLA is implemented; HEAVY_TAIL/COS_MAP remain policy tags.
comptime TSD_HEAVY_TAIL = 3
comptime TSD_COS_MAP = 4
comptime TSD_INVERTED_PARABOLA = 5


def inverted_parabola_weight_table(
    weight: Float32, bias: Float32, shift: Float32, num_train_timesteps: Int
) -> List[Float64]:
    """SerenityTrainer INVERTED_PARABOLA per-bucket weights (unnormalized PMF).

    Mirrors ModelSetupNoiseMixin._get_timestep_discrete's discrete branch
    (SerenityTrainer 423c3b36) with min/max_noising_strength 0/1:
      lin_i   = i/(N-1)                        # torch.linspace(0, 1, N)
      shifted = lin / (shift - shift*lin + lin)
      deriv   = shift / (shift + lin - lin*shift)^2
      w_i     = clamp(-weight*(shifted - (bias+0.5))^2 + 2, min=0) * deriv
    shift=1.0 makes shifted==lin and deriv==1 (identity correction)."""
    var b = Float64(bias) + 0.5
    var w = Float64(weight)
    var sh = Float64(shift)
    var out = List[Float64]()
    for i in range(num_train_timesteps):
        var lin = Float64(i) / Float64(num_train_timesteps - 1)
        var shifted = lin / (sh - sh * lin + lin)
        var denom = sh + lin - lin * sh
        var deriv = sh / (denom * denom)
        var wi = -w * (shifted - b) * (shifted - b) + Float64(2.0)
        if wi < Float64(0.0):
            wi = Float64(0.0)
        out.append(wi * deriv)
    return out^


def sample_timestep_inverted_parabola_idx(
    seed: UInt64, weight: Float32, bias: Float32, shift: Float32,
    num_train_timesteps: Int,
) -> Int:
    """One SerenityTrainer INVERTED_PARABOLA discrete timestep draw: index in [0, N-1].

    SerenityTrainer draws torch.multinomial(weights, 1, replacement=True); its RNG
    differs from our ChaCha12 stream, so parity is distribution-level
    (weight table exact + histogram-vs-PMF), the same bar as
    sample_timestep_uniform/sigmoid above. One ChaCha word: u ~ U(0,1),
    CDF-inverted over the N-bucket weight table. The drawn index feeds the
    existing conventions sigma=(idx+1)/N and model_t=idx/N
    (ModelSetupFlowMatchingMixin._add_noise_discrete / timestep/1000)."""
    var weights = inverted_parabola_weight_table(
        weight, bias, shift, num_train_timesteps
    )
    return sample_timestep_idx_from_weight_table(seed, weights)


def sample_timestep_idx_from_weight_table(
    seed: UInt64, weights: List[Float64]
) -> Int:
    """CDF-invert one U(0,1) ChaCha draw over an unnormalized weight table."""
    var ks = _expand_key(seed)
    var w0 = _chacha12_word_from_key(
        ks[0], ks[1], ks[2], ks[3], ks[4], ks[5], ks[6], ks[7], UInt64(0), 0
    )
    var u = _standard_f64(w0)

    var total = Float64(0.0)
    for i in range(len(weights)):
        total += weights[i]
    var target = u * total
    var acc = Float64(0.0)
    for i in range(len(weights)):
        acc += weights[i]
        if acc > target:
            return i
    return len(weights) - 1


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


# ─────────────────────────────────────────────────────────────────────────────
# BERNINI-R timestep sampling — logit_normal + mode weighting schemes.
#
# Mirrors bernini/training/data.py::compute_density_for_timestep_sampling + the
# NoiseScheduler.get_noise_sigma path + models/scheduler.py::FlowMatchScheduler
# EXACTLY (verified against /home/alex/Bernini):
#
#   density u:
#     logit_normal : u = sigmoid(normal(logit_mean, logit_std))
#                       = sigmoid(logit_mean + logit_std * N(0,1))     (data.py:84)
#     mode         : raw = U(0,1);
#                    u = 1 - raw - mode_scale*(cos(pi*raw/2)^2 - 1 + raw)
#                                                                       (data.py:86-87)
#   rejection: keep u only if tmin <= u <= tmax (data.py:90), else redraw.
#     tmin/tmax are per-shift boundary fractions of the noise window
#     [noise_tmin, noise_tmax] = [0.875, 1.0] (bernini_renderer_high.yaml).
#
#   sigma lookup (data.py:153-155 + FlowMatchScheduler):
#     idx   = int(u * 1000)                              (timestep_id)
#     sl    = linspace(1, 0, 1001)[:-1][idx] = 1 - idx/1000   (extra_one_step)
#     sigma = shift * sl / (1 + (shift - 1) * sl)         (scheduler.sigmas[idx])
#
# The RNG stream is Mojo's deterministic ChaCha12 (not torch's CPU generator), so
# this is DISTRIBUTION-LEVEL parity (moments + histogram), plus BIT-exact parity
# on the deterministic pieces (mode density(raw), shifted_sigma(idx), window).
# ─────────────────────────────────────────────────────────────────────────────
comptime _PI64 = Float64(3.14159265358979323846264338327950288)


# Deterministic closed forms (bit-exact gate vs the torch oracle) ──────────────
def bernini_mode_density_from_raw(raw: Float64, mode_scale: Float64) -> Float64:
    """u = 1 - raw - mode_scale*(cos(pi*raw/2)^2 - 1 + raw). data.py:86-87."""
    var c = cos(_PI64 * raw / Float64(2.0))
    return Float64(1.0) - raw - mode_scale * (c * c - Float64(1.0) + raw)


def bernini_logit_normal_density_from_z(
    z: Float64, logit_mean: Float64, logit_std: Float64
) -> Float64:
    """u = sigmoid(logit_mean + logit_std*z), z ~ N(0,1). data.py:84."""
    return _sigmoid64(logit_mean + logit_std * z)


def bernini_shifted_sigma(shift: Float64, idx: Int) -> Float64:
    """scheduler.sigmas[idx] for a training FlowMatchScheduler(shift, extra_one_step).

    sl = 1 - idx/1000 (linspace(1,0,1001)[:-1]); sigma = shift*sl/(1+(shift-1)*sl).
    """
    var sl = Float64(1.0) - Float64(idx) / Float64(1000.0)
    return shift * sl / (Float64(1.0) + (shift - Float64(1.0)) * sl)


def bernini_shift2boundary_idx(shift: Float64, target: Float64) -> Int:
    """argmin_i |shifted_sigma(shift, i) - target| over i in [0, 999].

    Replicates shift2boundary(shift) + find_nearest_boundary(sigmas, target)
    (data.py:61-68). num_steps=1000, denoising_strength=1, sigma_min=0."""
    var best_i = 0
    var best_d = Float64(1.0e30)
    for i in range(1000):
        var sh = bernini_shifted_sigma(shift, i)
        var d = sh - target
        if d < Float64(0.0):
            d = -d
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


@fieldwise_init
struct BerniniWindow(Copyable, Movable):
    var tmin: Float64
    var tmax: Float64


def bernini_task_window(
    shift: Float64, noise_tmin: Float64, noise_tmax: Float64
) -> BerniniWindow:
    """Per-shift rejection window (tmin, tmax) on the density u.

    b1 = boundary_idx(noise_tmin)/1000 ; b2 = boundary_idx(noise_tmax)/1000
    tmin = min(b1,b2) ; tmax = max(b1,b2)  (NoiseScheduler.__init__ :120-130)."""
    var b1 = Float64(bernini_shift2boundary_idx(shift, noise_tmin)) / Float64(1000.0)
    var b2 = Float64(bernini_shift2boundary_idx(shift, noise_tmax)) / Float64(1000.0)
    var lo = b1 if b1 < b2 else b2
    var hi = b1 if b1 > b2 else b2
    return BerniniWindow(lo, hi)


def _advance_seed(s: UInt64) -> UInt64:
    # Distinct-stream advance for the rejection loop (not a torch replay).
    return s * UInt64(6364136223846793005) + UInt64(1442695040888963407)


def bernini_density_draw(
    is_mode: Bool, seed: UInt64,
    logit_mean: Float64, logit_std: Float64, mode_scale: Float64,
) -> Float64:
    """One raw density draw u (pre-rejection) for the selected weighting."""
    if is_mode:
        var ks = _expand_key(seed)
        var w0 = _chacha12_word_from_key(
            ks[0], ks[1], ks[2], ks[3], ks[4], ks[5], ks[6], ks[7], UInt64(0), 0
        )
        var raw = _standard_f64(w0)
        return bernini_mode_density_from_raw(raw, mode_scale)
    var ks2 = _expand_key(seed)
    var d = _standard_normal_at(
        ks2[0], ks2[1], ks2[2], ks2[3], ks2[4], ks2[5], ks2[6], ks2[7], UInt64(0)
    )
    return bernini_logit_normal_density_from_z(d.z, logit_mean, logit_std)


def bernini_sample_sigma(
    is_mode: Bool, shift: Float64, seed: UInt64,
    logit_mean: Float64, logit_std: Float64, mode_scale: Float64,
    win: BerniniWindow,
) -> Float32:
    """Full Bernini training-sigma draw: rejection-sample u into [tmin,tmax],
    then idx=int(u*1000) -> sigma=shifted_sigma(shift,idx).

    Returns the flow-match noise coefficient sigma in [noise_tmin, noise_tmax]
    (~[0.875, 1.0] for the high-noise expert)."""
    var s = seed
    var u = Float64(0.5)
    var accepted = False
    for _ in range(100000):
        u = bernini_density_draw(is_mode, s, logit_mean, logit_std, mode_scale)
        s = _advance_seed(s)
        if u >= win.tmin and u <= win.tmax:
            accepted = True
            break
    if not accepted:
        # Degenerate-window fallback: clamp the last draw into the window.
        if u < win.tmin:
            u = win.tmin
        if u > win.tmax:
            u = win.tmax
    var idx = Int(u * Float64(1000.0))
    if idx < 0:
        idx = 0
    if idx > 999:
        idx = 999
    return Float32(bernini_shifted_sigma(shift, idx))
