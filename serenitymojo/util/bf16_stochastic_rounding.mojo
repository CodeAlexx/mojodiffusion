# bf16_stochastic_rounding.mojo -- OneTrainer BF16 stochastic rounding helper.
#
# OneTrainer copies F32 values into BF16 optimizer tensors with stochastic
# rounding. This helper is the core-side mirror used by OneTrainer-parity
# optimizer paths; production tensor boundaries still preserve checkpoint dtype.

from std.builtin.dtype import DType
from std.math import floor, log, pow

comptime _LN2 = Float64(0.69314718055994530942)
comptime _U24 = Float32(1.0) / Float32(16777216.0)


def _pcg_hash(x: UInt32) -> UInt32:
    var state = x * UInt32(747796405) + UInt32(2891336453)
    var shift = (state >> UInt32(28)) + UInt32(4)
    var word = ((state >> shift) ^ state) * UInt32(277803737)
    return (word >> UInt32(22)) ^ word


def sr_uniform(seed: UInt32, i: Int) -> Float32:
    var rnd = _pcg_hash(seed ^ UInt32(i))
    return Float32(Int(rnd >> UInt32(8))) * _U24


def _sr_bf16(v: Float32, u: Float32) -> BFloat16:
    if not (v == v):
        return v.cast[DType.bfloat16]()
    if v == Float32(0.0):
        return BFloat16(0.0)
    var sign = Float32(1.0)
    var a = v
    if a < Float32(0.0):
        sign = Float32(-1.0)
        a = -a
    if a < Float32(1.0e-38):
        return v.cast[DType.bfloat16]()
    var av = Float64(a)
    var e = Int(floor(log(av) / _LN2))
    var step = pow(Float64(2.0), Float64(e - 7))
    var y = av / step
    var kf = floor(y)
    var frac = y - kf
    var k = Int(kf)
    if Float64(u) < frac:
        k += 1
    var q = Float32(Float64(k) * step)
    if sign < Float32(0.0):
        q = -q
    return q.cast[DType.bfloat16]()


def copy_stochastic_value(source: Float32, seed: UInt32, i: Int) -> BFloat16:
    return _sr_bf16(source, sr_uniform(seed, i))


def add_stochastic_value(
    input_f32: Float32, other_f32: Float32, alpha: Float32, seed: UInt32, i: Int
) -> BFloat16:
    var result = other_f32 + alpha * input_f32
    return _sr_bf16(result, sr_uniform(seed, i))


def addcdiv_stochastic_value(
    input_f32: Float32,
    t1_f32: Float32,
    t2_f32: Float32,
    value: Float32,
    seed: UInt32,
    i: Int,
) -> BFloat16:
    var result = input_f32 + value * (t1_f32 / t2_f32)
    return _sr_bf16(result, sr_uniform(seed, i))
