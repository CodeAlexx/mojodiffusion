# bf16_stochastic_rounding.mojo — 1:1 port of Serenity
# modules/util/bf16_stochastic_rounding.py.
#
# Serenity copies an F32 value into a BF16 tensor using STOCHASTIC ROUNDING
# (copy_stochastic_, py:12-42):
#   result = randint(0, 1<<16)            # uniform 16-bit int per element   (py:24-31)
#   result += source.view(int32)          # add to the int32 bits of the f32 (py:34)
#   result &= 0xFFFF0000                  # mask off low 16 mantissa bits    (py:37)
#   target = result.view(float32)         # keep the high 16 bits = the bf16 (py:40)
#
# EQUIVALENCE (documented): adding a uniform 16-bit integer to the low half of
# the f32 mantissa and then truncating the low 16 bits rounds UP to the next bf16
# (high-16-bit value) with probability exactly equal to the fractional position
# of `source` within the bf16 ULP, and rounds DOWN otherwise. That is precisely
# unbiased stochastic rounding: E[bf16] == source. We reproduce it WITHOUT a
# scalar Float32 bit-reinterpret (unavailable in Mojo 1.0.0b1 kernels — see
# ops/torch_bf16.mojo:30-32) via float math: compute the bf16 ULP at the value,
# the fractional position `frac` within that ULP, and round up iff u < frac for a
# uniform u in [0,1). This is the `_sr_bf16` helper, the single source of truth
# the AdamW / Adam / CAME / Adafactor ports all import from here.
#
# RNG: Serenity seeds a torch.Generator once per step (set_seed, py:6-10) and
# draws `randint(0, 1<<16)` per element. Any unbiased 16-bit uniform source
# preserves the SR guarantee; we use a per-element PCG hash of (seed, index).
# The driver owns the per-step seed (= global_step) and threads it through every
# SR call explicitly (train_step.mojo); the kernels combine it with the element
# index. Mojo 1.0.0b1 has no mutable module-global state and no `global`
# keyword, so there is intentionally no module-level seed store here — the
# torch_Generator side effect of set_seed (py:6-10) is realized by the driver
# passing `seed` per call, matching adam_step/adamw_step/came_step signatures.

from std.builtin.dtype import DType
from std.math import sqrt, floor, log, pow

comptime _LN2 = Float64(0.69314718055994530942)
comptime _U24 = Float32(1.0) / Float32(16777216.0)  # 1/2^24 → uniform [0,1)


# PCG-style hash → uniform UInt32. Wrapping arithmetic (uint32 is modular). This
# is the unbiased 16-bit-uniform source standing in for torch.randint (py:24-31).
def _pcg_hash(x: UInt32) -> UInt32:
    var state = x * UInt32(747796405) + UInt32(2891336453)
    var shift = (state >> UInt32(28)) + UInt32(4)
    var word = ((state >> shift) ^ state) * UInt32(277803737)
    return (word >> UInt32(22)) ^ word


# Per-element uniform [0,1) from (seed, index) — the random 16-bit source behind
# copy_stochastic_ (py:24-37).
def sr_uniform(seed: UInt32, i: Int) -> Float32:
    var rnd = _pcg_hash(seed ^ UInt32(i))
    return Float32(Int(rnd >> UInt32(8))) * _U24


# Stochastic round f32 -> bf16. `u` is uniform [0,1). Rounds UP to the next bf16
# with probability = the fractional position within the bf16 ULP → unbiased
# (E[result] == v). This is the float-math equivalent of copy_stochastic_'s
# random-mantissa add+mask (bf16_stochastic_rounding.py:24-40); see the
# EQUIVALENCE note in this file's header. Mirrors the ULP computation in
# serenitymojo/ops/torch_bf16.mojo::torch_bf16_rne_value.
def _sr_bf16(v: Float32, u: Float32) -> BFloat16:
    if not (v == v):  # NaN
        return v.cast[DType.bfloat16]()
    if v == Float32(0.0):
        return BFloat16(0.0)
    var sign = Float32(1.0)
    var a = v
    if a < Float32(0.0):
        sign = Float32(-1.0)
        a = -a
    if a < Float32(1.0e-38):
        # bf16 subnormal range — the 2^(e-7) ULP math below is invalid there.
        # Values this small are negligible for training; fall back to native cast.
        return v.cast[DType.bfloat16]()
    var av = Float64(a)
    var e = Int(floor(log(av) / _LN2))            # binade
    var step = pow(Float64(2.0), Float64(e - 7))  # bf16 ULP at a (7 mantissa bits)
    var y = av / step
    var kf = floor(y)
    var frac = y - kf
    var k = Int(kf)
    if Float64(u) < frac:                         # round up with probability = frac
        k += 1
    var q = Float32(Float64(k) * step)
    if sign < Float32(0.0):
        q = -q
    return q.cast[DType.bfloat16]()


# copy_stochastic_(target, source): stochastically round one f32 value into bf16
# storage (py:12-42). The optimizer ports drive this per-element inside their GPU
# kernels; this host helper exists for parity and small-tensor / host paths.
def copy_stochastic_value(source: Float32, seed: UInt32, i: Int) -> BFloat16:
    return _sr_bf16(source, sr_uniform(seed, i))


# add_stochastic_(input, other, alpha): input += alpha*other with SR (py:45-57).
#   result = other (f32) ; result += alpha*input ; copy_stochastic_(input, result)
# i.e. new_input_bf16 = SR(input_f32 + alpha*other_f32). Per-element host helper;
# `input_bf16` is the current bf16 value, `other_f32`/`input_f32` are f32 regs.
def add_stochastic_value(
    input_f32: Float32, other_f32: Float32, alpha: Float32, seed: UInt32, i: Int
) -> BFloat16:
    var result = other_f32 + alpha * input_f32
    return _sr_bf16(result, sr_uniform(seed, i))


# addcdiv_stochastic_(input, t1, t2, value): input += value*(t1/t2) with SR
# (py:60-73):  result = input (f32) ; result.addcdiv_(t1,t2,value) ;
#              copy_stochastic_(input, result)
# i.e. new_input_bf16 = SR(input_f32 + value*(t1_f32/t2_f32)). Per-element host
# helper mirroring the in-kernel addcdiv used by adam/adamw.
def addcdiv_stochastic_value(
    input_f32: Float32, t1_f32: Float32, t2_f32: Float32, value: Float32, seed: UInt32, i: Int
) -> BFloat16:
    var result = input_f32 + value * (t1_f32 / t2_f32)
    return _sr_bf16(result, sr_uniform(seed, i))
