# random.mojo - GPU-side deterministic noise helpers.
#
# This is for inference setup, not training RNG. It produces deterministic
# standard-normal samples directly into a Tensor on the device, avoiding the
# previous host-filled latent path in Klein image generation.
#
# The stream mirrors Rust rand 0.8.5 StdRng as used by inference-flame:
# SeedableRng::seed_from_u64 -> PCG32-expanded 32-byte seed -> ChaCha12Rng
# -> Standard f32 -> Box-Muller.

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.math import log, sqrt, cos, sin
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _TWO_PI = Float32(6.2831853071795864769)
comptime _U24_SCALE = Float32(5.9604644775390625e-8)  # 1 / 2^24


@fieldwise_init
struct _PcgOut(Copyable, Movable):
    var state: UInt64
    var word: UInt32


@fieldwise_init
struct _QrOut(Copyable, Movable):
    var a: UInt32
    var b: UInt32
    var c: UInt32
    var d: UInt32


@fieldwise_init
struct _NormalPair(Copyable, Movable):
    var z0: Float32
    var z1: Float32


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


def _standard_f32(word: UInt32) -> Float32:
    return Float32(Int(word >> 8)) * _U24_SCALE


def _std_rng_pair(seed: UInt64, pair: UInt64) -> _NormalPair:
    var p = _pcg32(seed)
    var k0 = p.word
    p = _pcg32(p.state)
    var k1 = p.word
    p = _pcg32(p.state)
    var k2 = p.word
    p = _pcg32(p.state)
    var k3 = p.word
    p = _pcg32(p.state)
    var k4 = p.word
    p = _pcg32(p.state)
    var k5 = p.word
    p = _pcg32(p.state)
    var k6 = p.word
    p = _pcg32(p.state)
    var k7 = p.word

    var word_pos = pair * 2
    var block = word_pos // 16
    var offset = Int(word_pos % 16)
    var w0 = _chacha12_word_from_key(k0, k1, k2, k3, k4, k5, k6, k7, block, offset)
    var w1 = _chacha12_word_from_key(k0, k1, k2, k3, k4, k5, k6, k7, block, offset + 1)
    var u1 = _standard_f32(w0)
    var u2 = _standard_f32(w1)
    if u1 < Float32(1.0e-10):
        u1 = Float32(1.0e-10)
    var r = sqrt(Float32(-2.0) * log(u1))
    var theta = _TWO_PI * u2
    return _NormalPair(r * cos(theta), r * sin(theta))


def _randn_kernel_f32(
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
    seed: UInt64,
):
    var pair_i = Int(global_idx.x)
    var i = pair_i * 2
    if i < n:
        var z = _std_rng_pair(seed, UInt64(pair_i))
        o[i] = rebind[o.element_type](z.z0)
        if i + 1 < n:
            o[i + 1] = rebind[o.element_type](z.z1)


def _randn_kernel_bf16(
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
    seed: UInt64,
):
    var pair_i = Int(global_idx.x)
    var i = pair_i * 2
    if i < n:
        var z = _std_rng_pair(seed, UInt64(pair_i))
        o[i] = rebind[o.element_type](z.z0.cast[DType.bfloat16]())
        if i + 1 < n:
            o[i + 1] = rebind[o.element_type](z.z1.cast[DType.bfloat16]())


def _randn_kernel_f16(
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
    seed: UInt64,
):
    var pair_i = Int(global_idx.x)
    var i = pair_i * 2
    if i < n:
        var z = _std_rng_pair(seed, UInt64(pair_i))
        o[i] = rebind[o.element_type](z.z0.cast[DType.float16]())
        if i + 1 < n:
            o[i + 1] = rebind[o.element_type](z.z1.cast[DType.float16]())


def randn(var shape: List[Int], seed: UInt64, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    """Create a device Tensor filled with deterministic N(0,1) samples.

    The generator mirrors Rust rand 0.8.5 `StdRng::seed_from_u64(seed)` followed
    by `rng.gen::<f32>()` pairs and the same Box-Muller transform used in
    `inference-flame/src/sampling/klein_sampling.rs`.
    """
    var n = 1
    for i in range(len(shape)):
        if shape[i] <= 0:
            raise Error("randn: shape dimensions must be positive")
        n *= shape[i]

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * dtype.byte_size())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var pairs = (n + 1) // 2
    var grid = (pairs + _BLOCK - 1) // _BLOCK
    var dt = dtype.to_mojo_dtype()

    if dt == DType.float32:
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_randn_kernel_f32, _randn_kernel_f32](
            O, n, seed, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_randn_kernel_bf16, _randn_kernel_bf16](
            O, n, seed, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_randn_kernel_f16, _randn_kernel_f16](
            O, n, seed, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, shape^, dtype)
