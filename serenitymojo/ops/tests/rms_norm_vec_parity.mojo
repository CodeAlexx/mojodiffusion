# ops/tests/rms_norm_vec_parity.mojo — parity + microbenchmark for the
# RMS_NORM_VEC fast path (Lever B6, AUDIT_FUSION_SPEEDUP_PLAN_2026-05-30.md §B6).
#
# Vectorizing the F32 sum-of-squares load path (1 elem/thread → width-2 SIMD
# pair) regroups the per-thread accumulation by ~1 ULP, so the gate is NOT
# bit-equality — it is cosine ≥ 0.99999 vs the SCALAR kernel, on NON-DEGENERATE
# real-krea2 shapes. Both the scalar and the vec kernel are invoked DIRECTLY in
# this one binary (a same-process comparison), so the RMS_NORM_VEC dispatcher
# flag does not need to be flipped to run this gate.
#
# Shapes (real krea2 RMSNorm sites): rows = L = 4864;
#   cols = 6144 (prenorm/postnorm features), cols = 128 (q/k headdim norm).
# Both forward and the dx-only backward are gated.
#
# Build:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -O2 -Xlinker -lm -Xlinker -lcuda \
#       -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#       serenitymojo/ops/tests/rms_norm_vec_parity.mojo -o /tmp/rms_vec_par
# Run:
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib \
#     /tmp/rms_vec_par

from std.gpu.host import DeviceContext
from std.math import sqrt, sin
from std.time import perf_counter_ns
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.norm import (
    _rms_norm_kernel_bf16,
    _rms_norm_kernel_bf16_vec,
    _rms_norm_kernel_f32,
    _rms_norm_kernel_f32_vec,
)
from serenitymojo.ops.norm_backward import (
    _rms_bwd_dx_kernel,
    _rms_bwd_dx_kernel_vec,
)


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _DYN1 = Layout.row_major(-1)
comptime _TPB = 256


# Non-degenerate input: a bounded sinusoid mixed with a PCG-ish stream, NEVER a
# modular fill (the numeric-parity-testing skill bans modular fills as degenerate).
def _pattern(n: Int, seed: UInt64, amp: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for i in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        var s = sin(Float32(i) * Float32(0.013) + Float32(seed % 17))
        out.append(((u * Float32(2.0) - Float32(1.0)) * Float32(0.5) + s * Float32(0.5)) * amp)
    return out^


def _cosine(a: List[Float32], b: List[Float32]) -> Float64:
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(len(a)):
        var av = Float64(a[i])
        var bv = Float64(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


def _max_abs(a: List[Float32], b: List[Float32]) -> Float64:
    var m: Float64 = 0.0
    for i in range(len(a)):
        var d = abs(Float64(a[i]) - Float64(b[i]))
        if d > m:
            m = d
    return m


# ── BF16 forward parity (scalar kernel vs vec kernel, one binary) ────────────
def _bf16_fwd(
    x: Tensor, g: Tensor, rows: Int, cols: Int, eps: Float32, vec: Bool,
    ctx: DeviceContext,
) raises -> Tensor:
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cols))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](cols))
    var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
    var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        g.buf.unsafe_ptr().bitcast[BFloat16](), g_rl)
    var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
    if vec:
        ctx.enqueue_function[
            _rms_norm_kernel_bf16_vec, _rms_norm_kernel_bf16_vec
        ](X, G, O, cols, eps, grid_dim=rows, block_dim=_TPB)
    else:
        ctx.enqueue_function[_rms_norm_kernel_bf16, _rms_norm_kernel_bf16](
            X, G, O, cols, eps, grid_dim=rows, block_dim=_TPB)
    ctx.synchronize()
    return Tensor(out_buf^, x.shape().copy(), STDtype.BF16)


def _bf16_bwd_dx(
    go: Tensor, x: Tensor, g: Tensor, rows: Int, cols: Int, eps: Float32,
    vec: Bool, ctx: DeviceContext,
) raises -> Tensor:
    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cols))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](cols))
    var GO = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        go.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
    var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
    var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        g.buf.unsafe_ptr().bitcast[BFloat16](), g_rl)
    var DX = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        dx_buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
    if vec:
        ctx.enqueue_function[
            _rms_bwd_dx_kernel_vec[DType.bfloat16],
            _rms_bwd_dx_kernel_vec[DType.bfloat16],
        ](GO, X, G, DX, cols, eps, grid_dim=rows, block_dim=_TPB)
    else:
        ctx.enqueue_function[
            _rms_bwd_dx_kernel[DType.bfloat16], _rms_bwd_dx_kernel[DType.bfloat16]
        ](GO, X, G, DX, cols, eps, grid_dim=rows, block_dim=_TPB)
    ctx.synchronize()
    return Tensor(dx_buf^, x.shape().copy(), STDtype.BF16)


# ── microbenchmark: time scalar vs vec for one kernel, N iters ───────────────
def _bench_fwd(
    x: Tensor, g: Tensor, rows: Int, cols: Int, eps: Float32, vec: Bool,
    iters: Int, ctx: DeviceContext,
) raises -> Float64:
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cols))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](cols))
    var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
    var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        g.buf.unsafe_ptr().bitcast[BFloat16](), g_rl)
    var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
    # warmup
    for _ in range(5):
        if vec:
            ctx.enqueue_function[
                _rms_norm_kernel_bf16_vec, _rms_norm_kernel_bf16_vec
            ](X, G, O, cols, eps, grid_dim=rows, block_dim=_TPB)
        else:
            ctx.enqueue_function[_rms_norm_kernel_bf16, _rms_norm_kernel_bf16](
                X, G, O, cols, eps, grid_dim=rows, block_dim=_TPB)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        if vec:
            ctx.enqueue_function[
                _rms_norm_kernel_bf16_vec, _rms_norm_kernel_bf16_vec
            ](X, G, O, cols, eps, grid_dim=rows, block_dim=_TPB)
        else:
            ctx.enqueue_function[_rms_norm_kernel_bf16, _rms_norm_kernel_bf16](
                X, G, O, cols, eps, grid_dim=rows, block_dim=_TPB)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    _ = out_buf  # keep alive
    return Float64(t1 - t0) / Float64(iters) / 1000.0  # us/iter


def _run_shape(rows: Int, cols: Int, eps: Float32, ctx: DeviceContext) raises:
    print("─── shape rows=", rows, " cols=", cols, " ───", sep="")
    var n = rows * cols
    var xs = _pattern(n, 0x1234 + UInt64(cols), Float32(1.7))
    var gs = _pattern(cols, 0x9ABC + UInt64(cols), Float32(1.1))
    var gos = _pattern(n, 0x55AA + UInt64(cols), Float32(0.9))
    var x = Tensor.from_host(xs, [rows, cols], STDtype.BF16, ctx)
    var g = Tensor.from_host(gs, [cols], STDtype.BF16, ctx)
    var go = Tensor.from_host(gos, [rows, cols], STDtype.BF16, ctx)

    # FORWARD parity
    var y_scalar = _bf16_fwd(x, g, rows, cols, eps, False, ctx)
    var y_vec = _bf16_fwd(x, g, rows, cols, eps, True, ctx)
    var ys = y_scalar.to_host(ctx)
    var yv = y_vec.to_host(ctx)
    var cos_fwd = _cosine(ys, yv)
    var mab_fwd = _max_abs(ys, yv)
    print("  FWD  cos=", cos_fwd, "  max_abs=", mab_fwd, sep="")

    # BACKWARD d_x parity
    var dx_scalar = _bf16_bwd_dx(go, x, g, rows, cols, eps, False, ctx)
    var dx_vec = _bf16_bwd_dx(go, x, g, rows, cols, eps, True, ctx)
    var dxs = dx_scalar.to_host(ctx)
    var dxv = dx_vec.to_host(ctx)
    var cos_bwd = _cosine(dxs, dxv)
    var mab_bwd = _max_abs(dxs, dxv)
    print("  BWD  cos=", cos_bwd, "  max_abs=", mab_bwd, sep="")

    if cos_fwd < 0.99999:
        raise Error("FWD cosine below 0.99999")
    if cos_bwd < 0.99999:
        raise Error("BWD cosine below 0.99999")

    # microbenchmark forward (kernel-only timing)
    var us_scalar = _bench_fwd(x, g, rows, cols, eps, False, 200, ctx)
    var us_vec = _bench_fwd(x, g, rows, cols, eps, True, 200, ctx)
    print("  FWD kernel us/iter  scalar=", us_scalar, "  vec=", us_vec,
          "  speedup=", us_scalar / us_vec, "x", sep="")


def main() raises:
    var ctx = DeviceContext()
    var eps = Float32(1e-6)
    # krea2 prenorm/postnorm: L=4864 rows, 6144 features
    _run_shape(4864, 6144, eps, ctx)
    # krea2 q/k headdim norm: L=4864 rows, 128 headdim
    _run_shape(4864, 128, eps, ctx)
    print("PASS: RMS_NORM_VEC parity cos>=0.99999 (fwd+bwd), both krea2 shapes")
