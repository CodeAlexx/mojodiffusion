# Exact local PyTorch CPU AVX2 F32 randn compatibility profile for H3 caches.
#
# Oracle pin (all digests are checked by the fixture generator):
#   torch 2.12.0+cu130 git 7661cd9c6b841b62b7f411aa52ec51f05457263b
#   ATen/native/cpu/DistributionTemplates.h
#     dce75f8036a0dbeed823f7b282245673b74782e7af182755fd6babee7708331b
#   ATen/native/cpu/avx_mathfun.h
#     e713d6fc64a0e034b13f2cf4b3b9d481fcb4ab0ea2a925154ee1286fe526d07a
#
# This is host-only.  It emits a contiguous List[Float32] in C-order and calls
# no Python.  The narrow public profile refuses N < 16 and unknown profile
# names.  The complete byte contract is x86-64 little-endian; other PyTorch
# releases, compiler contraction patterns, CPU dispatch targets, dtypes, and
# non-contiguous layouts are outside this profile.

from std.collections import List
from std.benchmark import black_box
from std.math import fma, sqrt
from std.memory import bitcast

from serenitymojo.pipeline.minimax_h3_torch_cpu_rng import (
    MiniMaxH3TorchCpuGenerator,
)


comptime TORCH_CPU_RANDN_F32_AVX2_V1 = "torch_cpu_randn_f32_avx2_v1"
comptime _F32_ONE = Float32(1.0)
comptime _F32_HALF = Float32(0.5)
comptime _F32_MINUS_TWO = Float32(-2.0)
comptime _F32_TWO_PI = Float32(6.283185307179586476925286766559)


@always_inline
def _round_f32(value: Float32) -> Float32:
    return black_box(value)


@always_inline
def _add_f32(lhs: Float32, rhs: Float32) -> Float32:
    return _round_f32(lhs + rhs)


@always_inline
def _sub_f32(lhs: Float32, rhs: Float32) -> Float32:
    return _round_f32(lhs - rhs)


@always_inline
def _mul_f32(lhs: Float32, rhs: Float32) -> Float32:
    return _round_f32(lhs * rhs)


@always_inline
def _fma_f32(lhs: Float32, rhs: Float32, acc: Float32) -> Float32:
    return _round_f32(fma(lhs, rhs, acc))


@always_inline
def _f32_bits(value: Float32) -> UInt32:
    return bitcast[DType.uint32, 1](SIMD[DType.float32, 1](value))


@always_inline
def _bits_f32(value: UInt32) -> Float32:
    return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](value))


@always_inline
def _xor_sign(value: Float32, mask: UInt32) -> Float32:
    return _bits_f32(_f32_bits(value) ^ mask)


def _log256_ps_lane(value: Float32) raises -> Float32:
    """One lane of pinned ``log256_ps`` with installed GCC/FMA contraction."""
    if value <= Float32(0.0):
        raise Error("torch_cpu_randn_f32_avx2_v1: log lane must be positive")
    var bits = _f32_bits(value)
    var exponent = Int32(bits >> UInt32(23)) - Int32(0x7F)
    var x = _bits_f32(
        (bits & UInt32(0x807FFFFF)) | _f32_bits(Float32(0.5))
    )
    var e = _add_f32(Float32(exponent), _F32_ONE)
    var below_sqrt_half = x < Float32(0.707106781186547524)
    var extra = x if below_sqrt_half else Float32(0.0)
    x = _sub_f32(x, _F32_ONE)
    if below_sqrt_half:
        e = _sub_f32(e, _F32_ONE)
    x = _add_f32(x, extra)
    var z = _mul_f32(x, x)

    # The installed AVX2 object contracts each polynomial multiply/add into
    # FMA even though avx_mathfun.h spells them as separate intrinsics.
    var y = Float32(7.0376836292e-2)
    y = _fma_f32(y, x, Float32(-1.1514610310e-1))
    y = _fma_f32(y, x, Float32(1.1676998740e-1))
    y = _fma_f32(y, x, Float32(-1.2420140846e-1))
    y = _fma_f32(y, x, Float32(1.4249322787e-1))
    y = _fma_f32(y, x, Float32(-1.6668057665e-1))
    y = _fma_f32(y, x, Float32(2.0000714765e-1))
    y = _fma_f32(y, x, Float32(-2.4999993993e-1))
    y = _fma_f32(y, x, Float32(3.3333331174e-1))
    y = _mul_f32(y, x)
    y = _fma_f32(y, z, _mul_f32(e, Float32(-2.12194440e-4)))
    y = _fma_f32(-z, _F32_HALF, y)
    x = _add_f32(x, y)
    return _fma_f32(e, Float32(0.693359375), x)


@fieldwise_init
struct _SinCosF32(Copyable, Movable):
    var sine: Float32
    var cosine: Float32


def _sincos256_ps_lane(value: Float32) -> _SinCosF32:
    """One lane of pinned ``sincos256_ps`` and its FMA contraction pattern."""
    var source_bits = _f32_bits(value)
    var sign_sine = source_bits & UInt32(0x80000000)
    var x = _bits_f32(source_bits & UInt32(0x7FFFFFFF))
    var scaled = _mul_f32(x, Float32(1.27323954473516))
    var quadrant = Int32(scaled)
    quadrant = (quadrant + Int32(1)) & Int32(-2)
    var qf = Float32(quadrant)
    var swap_sine = UInt32(quadrant & Int32(4)) << UInt32(29)
    sign_sine = sign_sine ^ swap_sine
    var use_sine_poly_for_sine = (quadrant & Int32(2)) == Int32(0)

    x = _fma_f32(qf, Float32(-0.78515625), x)
    x = _fma_f32(qf, Float32(-2.4187564849853515625e-4), x)
    x = _fma_f32(qf, Float32(-3.77489497744594108e-8), x)

    var cosine_quadrant = quadrant - Int32(2)
    var sign_cosine = (
        UInt32((~cosine_quadrant) & Int32(4)) << UInt32(29)
    )
    var z = _mul_f32(x, x)

    var cosine_poly = _fma_f32(
        Float32(2.443315711809948e-5), z,
        Float32(-1.388731625493765e-3),
    )
    cosine_poly = _fma_f32(
        cosine_poly, z, Float32(4.166664568298827e-2)
    )
    cosine_poly = _mul_f32(cosine_poly, z)
    cosine_poly = _fma_f32(
        cosine_poly, z, -_mul_f32(z, _F32_HALF)
    )
    cosine_poly = _add_f32(cosine_poly, _F32_ONE)

    var sine_poly = _fma_f32(
        Float32(-1.9515295891e-4), z, Float32(8.3321608736e-3)
    )
    sine_poly = _fma_f32(
        sine_poly, z, Float32(-1.6666654611e-1)
    )
    sine_poly = _mul_f32(sine_poly, z)
    sine_poly = _fma_f32(sine_poly, x, x)

    var sine = sine_poly if use_sine_poly_for_sine else cosine_poly
    var cosine = cosine_poly if use_sine_poly_for_sine else sine_poly
    return _SinCosF32(
        _xor_sign(sine, sign_sine), _xor_sign(cosine, sign_cosine)
    )


def _normal_fill_16_avx2(mut data: List[Float32], base: Int) raises:
    for lane in range(8):
        var u1 = _sub_f32(_F32_ONE, data[base + lane])
        var u2 = data[base + lane + 8]
        var log_u1 = _log256_ps_lane(u1)
        var radius = _round_f32(sqrt(_mul_f32(_F32_MINUS_TWO, log_u1)))
        var theta = _mul_f32(_F32_TWO_PI, u2)
        var trig = _sincos256_ps_lane(theta)
        var first = _mul_f32(radius, trig.cosine)
        var second = _mul_f32(radius, trig.sine)
        # DistributionTemplates uses fmadd(n, std=1, mean=0).
        data[base + lane] = _fma_f32(first, _F32_ONE, Float32(0.0))
        data[base + lane + 8] = _fma_f32(
            second, _F32_ONE, Float32(0.0)
        )


struct TorchCpuRandnF32Avx2V1Generator(Movable):
    """Full-seed wrapper over the shared exact PyTorch MT19937 engine."""

    var seed: UInt64
    var engine: MiniMaxH3TorchCpuGenerator

    def __init__(out self, seed: UInt64):
        self.seed = seed
        self.engine = MiniMaxH3TorchCpuGenerator(seed)


def _validate_profile(profile: String) raises:
    if profile != String(TORCH_CPU_RANDN_F32_AVX2_V1):
        raise Error(
            String("torch_cpu_randn_f32_avx2_v1: unsupported profile ")
            + profile
        )


def _c_order_numel(shape: List[Int]) raises -> Int:
    if len(shape) == 0:
        raise Error("torch_cpu_randn_f32_avx2_v1: shape must not be empty")
    var total = 1
    for dim in shape:
        if dim <= 0:
            raise Error(
                "torch_cpu_randn_f32_avx2_v1: C-order dimensions must be positive"
            )
        if total > 9223372036854775807 // dim:
            raise Error("torch_cpu_randn_f32_avx2_v1: shape numel overflow")
        total *= dim
    if total < 16:
        raise Error("torch_cpu_randn_f32_avx2_v1: contiguous F32 N must be >= 16")
    return total


def torch_cpu_randn_f32_avx2_v1_from(
    mut generator: TorchCpuRandnF32Avx2V1Generator,
    shape: List[Int],
    profile: String = String(TORCH_CPU_RANDN_F32_AVX2_V1),
) raises -> List[Float32]:
    """Advance one generator and return contiguous F32 values in C-order."""
    _validate_profile(profile)
    var numel = _c_order_numel(shape)
    var data = List[Float32](capacity=numel)
    for _ in range(numel):
        data.append(generator.engine.uniform01())
    var base = 0
    while base < numel - 15:
        _normal_fill_16_avx2(data, base)
        base += 16
    if numel % 16 != 0:
        base = numel - 16
        for lane in range(16):
            data[base + lane] = generator.engine.uniform01()
        _normal_fill_16_avx2(data, base)
    return data^


def torch_cpu_randn_f32_avx2_v1(
    shape: List[Int],
    seed: UInt64,
    profile: String = String(TORCH_CPU_RANDN_F32_AVX2_V1),
) raises -> List[Float32]:
    var generator = TorchCpuRandnF32Avx2V1Generator(seed)
    return torch_cpu_randn_f32_avx2_v1_from(generator, shape.copy(), profile)


def _append_u32_le(mut output: List[UInt8], value: UInt32):
    for index in range(4):
        output.append(UInt8((value >> UInt32(8 * index)) & UInt32(0xFF)))


def _append_u64_le(mut output: List[UInt8], value: UInt64):
    for index in range(8):
        output.append(UInt8((value >> UInt64(8 * index)) & UInt64(0xFF)))


def torch_cpu_randn_f32_avx2_v1_raw_le_bytes(
    values: List[Float32],
) -> List[UInt8]:
    """Serialize admitted F32 output exactly as x86 little-endian raw bytes."""
    var output = List[UInt8](capacity=len(values) * 4)
    for value in values:
        _append_u32_le(output, _f32_bits(value))
    return output^


def torch_cpu_randn_f32_avx2_v1_state_bytes(
    ref generator: TorchCpuRandnF32Avx2V1Generator,
) -> List[UInt8]:
    """PyTorch 2.12 CPU Generator.get_state() bytes for this narrow profile."""
    var output = List[UInt8](capacity=5056)
    _append_u64_le(output, generator.seed)
    _append_u32_le(output, UInt32(generator.engine.left))
    _append_u32_le(output, UInt32(1))  # mt19937_data_pod.seeded_
    _append_u32_le(output, UInt32(generator.engine.next))
    _append_u32_le(output, UInt32(0))  # x86-64 legacy-state padding
    for value in generator.engine.state:
        _append_u32_le(output, value)
        _append_u32_le(output, UInt32(0))
    # next_float_normal_sample and next_double_normal_sample are unused by
    # normal_fill_AVX2 and serialize as zero in this pinned build.
    for _ in range(40):
        output.append(UInt8(0))
    return output^
