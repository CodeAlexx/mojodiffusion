# ops/random_torch.mojo — PyTorch-bit-compatible CUDA randn (Philox4x32-10 + curand
# Box-Muller). Faithful Mojo port of flame-core/src/rng/torch_compat.rs
# (flame_randn_torch_f32), which mirrors torch's DistributionTemplates.h normal kernel:
#   curand_init(seed, idx, 0); ctr=(N,0,idx,0); 10 Philox rounds; curand box-muller4;
#   grid-stride scatter to [li, li+stride, li+2*stride, li+3*stride]; block=256,
#   grid=min(SMs*(maxThreadsPerSM/256), ceil(numel/256)).
# CAVEAT (same as torch): the exact bytes depend on the GPU SM count via the grid.
from std.gpu.host import DeviceContext
from std.gpu import global_idx, block_dim, grid_dim
from std.math import log, sqrt, sin, cos
from std.ffi import external_call
from std.memory import alloc
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr

comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256

comptime _W0: UInt32 = 0x9E3779B9
comptime _W1: UInt32 = 0xBB67AE85
comptime _M0: UInt32 = 0xD2511F53
comptime _M1: UInt32 = 0xCD9E8D57
comptime _INV: Float32 = 2.3283064e-10
comptime _INV2PI: Float32 = 2.3283064e-10 * 6.2831855

# CUDA driver attrs (cuda.h): MULTIPROCESSOR_COUNT=16, MAX_THREADS_PER_MULTIPROCESSOR=39.
comptime _CU_ATTR_SM_COUNT: Int32 = 16
comptime _CU_ATTR_MAX_THREADS_PER_SM: Int32 = 39


@fieldwise_init
struct _Quad(ImplicitlyCopyable, Movable):
    var c0: UInt32
    var c1: UInt32
    var c2: UInt32
    var c3: UInt32


fn _mulhi(a: UInt32, b: UInt32) -> UInt32:
    return UInt32((UInt64(a) * UInt64(b)) >> 32)


fn _round(q: _Quad, k0: UInt32, k1: UInt32) -> _Quad:
    var hi0 = _mulhi(_M0, q.c0)
    var lo0 = _M0 * q.c0
    var hi1 = _mulhi(_M1, q.c2)
    var lo1 = _M1 * q.c2
    return _Quad(hi1 ^ q.c1 ^ k0, lo1, hi0 ^ q.c3 ^ k1, lo0)


fn _philox10(q0: _Quad, k0: UInt32, k1: UInt32) -> _Quad:
    var q = q0
    var kx = k0
    var ky = k1
    q = _round(q, kx, ky); kx += _W0; ky += _W1   # 1
    q = _round(q, kx, ky); kx += _W0; ky += _W1   # 2
    q = _round(q, kx, ky); kx += _W0; ky += _W1   # 3
    q = _round(q, kx, ky); kx += _W0; ky += _W1   # 4
    q = _round(q, kx, ky); kx += _W0; ky += _W1   # 5
    q = _round(q, kx, ky); kx += _W0; ky += _W1   # 6
    q = _round(q, kx, ky); kx += _W0; ky += _W1   # 7
    q = _round(q, kx, ky); kx += _W0; ky += _W1   # 8
    q = _round(q, kx, ky); kx += _W0; ky += _W1   # 9
    q = _round(q, kx, ky)                          # 10
    return q


fn _boxmuller(x: UInt32, y: UInt32) -> SIMD[DType.float32, 2]:
    var u = Float32(x) * _INV + (_INV * Float32(0.5))
    var v = Float32(y) * _INV2PI + (_INV2PI * Float32(0.5))
    var s = sqrt(Float32(-2.0) * log(u))
    return SIMD[DType.float32, 2](s * sin(v), s * cos(v))


def _randn_torch_kernel(
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], numel: Int, seed: UInt64
):
    var idx = Int(global_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    var unroll = 4
    var rounded = ((numel - 1) // (stride * unroll) + 1) * stride * unroll
    var k0 = UInt32(seed & 0xFFFFFFFF)
    var k1 = UInt32((seed >> 32) & 0xFFFFFFFF)
    var idx_u = UInt32(idx)
    var calls = 0
    var li = idx
    while li < rounded:
        var q = _philox10(_Quad(UInt32(calls), UInt32(0), idx_u, UInt32(0)), k0, k1)
        var a = _boxmuller(q.c0, q.c1)
        var b = _boxmuller(q.c2, q.c3)
        if li < numel:
            o[li] = a[0]
        if li + stride < numel:
            o[li + stride] = a[1]
        if li + 2 * stride < numel:
            o[li + 2 * stride] = b[0]
        if li + 3 * stride < numel:
            o[li + 3 * stride] = b[1]
        calls += 1
        li += stride * unroll


def _query_grid(numel: Int) raises -> Int:
    # calc_execution_policy: grid = min(SMs*(maxThreadsPerSM/256), ceil(numel/256)).
    var out = alloc[Int32](1)
    _ = external_call["cuInit", Int32](Int32(0))
    var dev = alloc[Int32](1)
    _ = external_call["cuDeviceGet", Int32](
        BytePtr(unsafe_from_address=Int(dev)), Int32(0)
    )
    out[0] = 1
    _ = external_call["cuDeviceGetAttribute", Int32](
        BytePtr(unsafe_from_address=Int(out)), _CU_ATTR_SM_COUNT, dev[0]
    )
    var sms = Int(out[0])
    out[0] = 256
    _ = external_call["cuDeviceGetAttribute", Int32](
        BytePtr(unsafe_from_address=Int(out)), _CU_ATTR_MAX_THREADS_PER_SM, dev[0]
    )
    var mtps = Int(out[0])
    dev.free(); out.free()
    var cap = sms * (mtps // _BLOCK)
    if cap < 1:
        cap = 1
    var ceil = (numel + _BLOCK - 1) // _BLOCK
    var grid = ceil if ceil < cap else cap
    if grid < 1:
        grid = 1
    return grid


def randn_torch(var shape: List[Int], seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    """N(0,1) tensor bit-compatible with torch.randn(shape, generator=Generator(cuda)
    .manual_seed(seed)) on THIS GPU (grid depends on SM count, per torch)."""
    var n = 1
    for i in range(len(shape)):
        if shape[i] <= 0:
            raise Error("randn_torch: dims must be > 0")
        n *= shape[i]
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var grid = _query_grid(n)
    ctx.enqueue_function[_randn_torch_kernel, _randn_torch_kernel](
        O, n, seed, grid_dim=grid, block_dim=_BLOCK
    )
    return Tensor(out_buf^, shape^, STDtype.F32)
