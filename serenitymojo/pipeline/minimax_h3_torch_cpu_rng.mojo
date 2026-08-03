# serenitymojo/pipeline/minimax_h3_torch_cpu_rng.mojo — PyTorch's CPU RNG,
# reproduced. Host only: no Tensor, no DeviceContext, no GPU.
#
# ── WHY THIS EXISTS ──────────────────────────────────────────────────────────
# MiniMax-H3's keyframe encode samples the VAE posterior off
# `torch.Generator().manual_seed(42)` (encoders.py:297) — a CPU generator, fixed
# at 42 INDEPENDENTLY of the request seed (packing.py:87). `randn_tensor`
# (diffusers/utils/torch_utils.py) honours that: when the generator is a CPU one
# and the target device is not, it draws on the CPU and moves the result. So the
# conditioning latent of every I2VA / FL2VA / L2VA request is a function of
# torch's CPU MT19937 stream, and a port that substitutes any other normal
# generator produces a DIFFERENT conditioning image anchor — not a slightly
# noisier one, a different one.
#
# `ops/random.mojo` is Rust-rand-compatible (ChaCha12) and `ops/random_torch.mojo`
# is torch's CUDA Philox. Neither is torch's CPU stream. Hence this file.
#
# ── WHAT IS REPRODUCED, OP FOR OP ────────────────────────────────────────────
# 1. `at::mt19937_engine` — torch/include/ATen/core/MT19937RNGEngine.h, read in
#    full. Note the pre-decrement: `if (--left_ == 0) next_state();` with
#    `left_` initialized to 1, so the FIRST call already regenerates the state
#    and then reads `state_[0]`. An implementation that seeds `left_ = 0` or
#    post-decrements is off by a whole 624-word block from word one.
# 2. `at::uniform_real_distribution<float>` — `(random() & 0xFFFFFF) * 2^-24`,
#    i.e. 24 mantissa bits, in FLOAT arithmetic
#    (torch/include/ATen/core/TransformationHelper.h `uniform_real`).
# 3. `normal_fill` / `normal_fill_16` — torch/include/ATen/native/cpu/
#    DistributionTemplates.h:138-148 and :108-135. The shape of this is the part
#    a rewrite gets wrong: torch fills the WHOLE buffer with uniforms first,
#    then converts it IN PLACE in blocks of 16, pairing `data[j]` with
#    `data[j+8]` — not consecutive pairs. And when `size % 16 != 0` it draws 16
#    FRESH uniforms over the last 16 slots and re-converts them, so the tail
#    values are NOT the ones the first pass put there and the total number of
#    32-bit draws is `size + 16`, not `size`.
#
# ── THE ONE MEASURED DIVERGENCE (not hidden, quantified) ─────────────────────
# torch compiles an AVX2 variant of this loop (`normal_fill_AVX2`, :88-135) that
# swaps `std::log`/`std::cos` for the polynomial approximations in
# `avx_mathfun.h`, and computes `theta` in float where the scalar path computes
# it in double. On an AVX2 host — this machine — that variant is what runs, so
# NO implementation using exact libm can be bit-identical to it. This file
# reproduces the SCALAR reference path and its probe MEASURES the gap against
# the real torch build rather than asserting a bar it cannot meet.
#
# That gap is also numerically irrelevant WHERE IT IS USED, and this is checkable
# rather than asserted: the sampled latent is immediately rounded to float16
# (encoders.py:300, "~11 bits of every conditioning latent"), whose relative
# resolution is ~4.9e-4 — three orders of magnitude coarser than the divergence
# the probe measures. The probe therefore also reports how many values change
# ACROSS the fp16 rounding, which is the number that actually matters.
#
# Gate: pipeline/parity/minimax_h3_torch_cpu_rng_probe.mojo (host, no GPU).

from std.collections import List
from std.math import log, sqrt, cos, sin

# at::MERSENNE_STATE_N / _M and the twist constants (MT19937RNGEngine.h:19-23).
comptime _MT_N = 624
comptime _MT_M = 397
comptime _MT_MATRIX_A = UInt32(0x9908B0DF)
comptime _MT_UMASK = UInt32(0x80000000)
comptime _MT_LMASK = UInt32(0x7FFFFFFF)

# `uniform_real<float>`: MASK = 2^digits - 1 and DIVISOR = 2^-digits for
# `std::numeric_limits<float>::digits == 24`.
comptime _UNIFORM_MASK = UInt32(0x00FFFFFF)
comptime _UNIFORM_DIVISOR = Float32(5.9604644775390625e-8)  # exactly 2^-24

# `2.0f * c10::pi<double>` — formed in DOUBLE in normal_fill_16 and only then
# multiplied by the float uniform, so it is kept in double here too.
comptime _TWO_PI_F64 = Float64(6.283185307179586476925286766559)


struct MiniMaxH3TorchCpuGenerator(Movable):
    """`at::mt19937_engine`, the engine behind `torch.Generator()` on the CPU."""

    var state: List[UInt32]
    var left: Int
    var next: Int

    def __init__(out self, seed: UInt64):
        """`init_with_uint32` (MT19937RNGEngine.h:157-166). `left_ = 1` and
        `next_ = 0` are part of the seeded state, not an "empty" marker."""
        self.state = List[UInt32]()
        self.state.resize(_MT_N, UInt32(0))
        self.state[0] = UInt32(seed & UInt64(0xFFFFFFFF))
        for j in range(1, _MT_N):
            var prev = self.state[j - 1]
            self.state[j] = UInt32(1812433253) * (prev ^ (prev >> UInt32(30))) + UInt32(j)
        self.left = 1
        self.next = 0

    def _twist(self, u: UInt32, v: UInt32) -> UInt32:
        var mixed = (u & _MT_UMASK) | (v & _MT_LMASK)
        var out = mixed >> UInt32(1)
        if (v & UInt32(1)) != UInt32(0):
            out = out ^ _MT_MATRIX_A
        return out

    def _next_state(mut self):
        """`next_state` (:174-187), written with explicit indices rather than
        the reference's pointer walk — same three loops, same order."""
        self.left = _MT_N
        self.next = 0
        var p = 0
        for _ in range(_MT_N - _MT_M):
            self.state[p] = self.state[p + _MT_M] ^ self._twist(
                self.state[p], self.state[p + 1]
            )
            p += 1
        for _ in range(_MT_M - 1):
            self.state[p] = self.state[p + _MT_M - _MT_N] ^ self._twist(
                self.state[p], self.state[p + 1]
            )
            p += 1
        self.state[p] = self.state[p + _MT_M - _MT_N] ^ self._twist(
            self.state[p], self.state[0]
        )

    def random(mut self) -> UInt32:
        """`operator()` (:141-150): PRE-decrement, then read, then temper."""
        self.left -= 1
        if self.left == 0:
            self._next_state()
        var y = self.state[self.next]
        self.next += 1
        y = y ^ (y >> UInt32(11))
        y = y ^ ((y << UInt32(7)) & UInt32(0x9D2C5680))
        y = y ^ ((y << UInt32(15)) & UInt32(0xEFC60000))
        y = y ^ (y >> UInt32(18))
        return y

    def uniform01(mut self) -> Float32:
        """`uniform_real_distribution<float>(0, 1)`: 24 bits, float arithmetic."""
        return Float32(self.random() & _UNIFORM_MASK) * _UNIFORM_DIVISOR


def minimax_h3_torch_cpu_uniform(numel: Int, seed: UInt64) raises -> List[Float32]:
    """`torch.rand(numel, generator=Generator().manual_seed(seed))` on the CPU.

    Exposed on its own because it is the BIT-EXACT half of this file — no libm
    is involved, so this one either matches torch everywhere or is wrong."""
    if numel < 0:
        raise Error("minimax_h3_torch_cpu_rng: numel must be non-negative")
    var gen = MiniMaxH3TorchCpuGenerator(seed)
    var out = List[Float32](capacity=numel)
    for _ in range(numel):
        out.append(gen.uniform01())
    return out^


def _normal_fill_16(mut data: List[Float32], base: Int):
    """`normal_fill_16` (:138-148). Pairs `j` with `j+8`, NOT `2j` with `2j+1`,
    and takes `u1 = 1 - data[j]` to move the log's argument off zero."""
    for j in range(8):
        var u1 = Float32(1.0) - data[base + j]
        var u2 = data[base + j + 8]
        var radius = sqrt(Float32(-2.0) * log(u1))
        # `2.0f * c10::pi<double> * u2` — the constant is a double and the
        # product is formed in double before landing in a float.
        var theta = Float32(_TWO_PI_F64 * Float64(u2))
        data[base + j] = radius * cos(theta)
        data[base + j + 8] = radius * sin(theta)


def minimax_h3_torch_cpu_randn_from(
    mut gen: MiniMaxH3TorchCpuGenerator, numel: Int
) raises -> List[Float32]:
    """One `torch.randn(...)` draw off an EXISTING generator, advancing it.

    This is the form a real request needs, not the re-seeding one: MiniMax-H3
    draws several tensors from the SAME generator in a fixed order — one
    conditioning noise per keyframe in packed order, then the target video
    noise, then the target audio noise — and "the order is part of what a
    generator reproduces" (packing.py:511-515). Re-seeding per draw would give
    every tensor the same numbers.

    How far each draw advances the stream is not `numel`: `normal_fill` draws
    `numel` uniforms and then, when `numel % 16 != 0`, SIXTEEN MORE for the
    recomputed tail. A caller cannot predict the offset, which is exactly why
    the generator is threaded through rather than re-derived."""
    if numel < 16:
        raise Error(
            String("minimax_h3_torch_cpu_rng: normal_fill needs >= 16 elements,")
            + " got " + String(numel) + " — torch takes its scalar"
            " normal_distribution path below that, which this file does not"
            " reproduce"
        )
    var data = List[Float32](capacity=numel)
    for _ in range(numel):
        data.append(gen.uniform01())

    var i = 0
    while i < numel - 15:
        _normal_fill_16(data, i)
        i += 16

    if numel % 16 != 0:
        # The tail is RE-DRAWN, not carried over: 16 fresh uniforms overwrite
        # the last 16 slots (some of which a full block already converted) and
        # are converted again.
        var base = numel - 16
        for k in range(16):
            data[base + k] = gen.uniform01()
        _normal_fill_16(data, base)
    return data^


def minimax_h3_torch_cpu_randn(numel: Int, seed: UInt64) raises -> List[Float32]:
    """`torch.randn(numel, generator=Generator().manual_seed(seed))` — one draw
    off a FRESH generator. The keyframe posterior sample is exactly this, at the
    fixed seed 42 (encoders.py:297); a request's own noise is not, and must go
    through `minimax_h3_torch_cpu_randn_from` on a threaded generator."""
    var gen = MiniMaxH3TorchCpuGenerator(seed)
    return minimax_h3_torch_cpu_randn_from(gen, numel)
