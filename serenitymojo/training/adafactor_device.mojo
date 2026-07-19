# training/adafactor_device.mojo — DEVICE Adafactor for the krea2 full-FT
# streamed step (FULL_FINETUNE_KREA2_PLAN_2026-07-07 P3).
#
# Same math as training/adafactor.mojo::adafactor_step_2d (the parity-gated
# host port of torch/optim/_adafactor.py:330-416) but the whole per-weight
# update runs ON DEVICE against the streamed bf16 weight slot — grads and
# weights never round-trip to the host (only the factored row/col state,
# ~N+K floats per matrix, stays device-resident; ~20MB for all of krea2).
#
# Differences vs the host oracle (documented, gated):
#   * p is the BF16 device weight (the pinned-host-store image) — the update
#     computes in F32 from the bf16 read and writes back bf16 with STOCHASTIC
#     ROUNDING (reference trainer bf16 full-FT contract, bf16_stochastic_rounding.py: add a
#     random 16-bit value below the mantissa cut, then truncate). An RNE mode
#     (sr_seed == 0) exists for bit-gating vs the host oracle on F32 data.
#   * Scalar bookkeeping (alpha/denom combine) runs in a 1-thread F32 kernel;
#     rho_t / one_minus_beta2_t are host-F64-computed per step and passed in
#     (they depend only on t/lr).
#
# Gate: training/tests (adafactor_device_parity.mojo) — device vs host
# adafactor_step_2d on random data, RNE mode, cos >= 0.99999 on p/row/col.
#
# 2026-07-08 (FULL_SURFACE_PLAN Phase A): + the 1D UNFACTORED path
# (adafactor_step_device_1d, torch _adafactor.py:404-416 grad.dim()<=1
# branch — exp_avg_sq [n] carried in AdafactorDeviceState.row_var with the
# cols==0 sentinel). Gate: adafactor_device_1d_parity.mojo — p BIT-IDENTICAL
# (max_abs 0.0) vs adafactor_step_1d, v exact, SR <=1 ulp. The 2D kernels
# are byte-untouched.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.gpu import global_idx, grid_dim, block_dim, block_idx, thread_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation, ArcPointer
from std.math import sqrt, min, max, floor, log

comptime _AF_LN2 = Float64(0.6931471805599453)
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import zeros_device

comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _PARTS = 1024   # partial-sum blocks for the big reduces

comptime TArc = ArcPointer[Tensor]


# ── K1: block-per-row — row g² mean lerp into row_var + row p² partials ──────
def _af_row_kernel(
    p: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],   # [rows*cols]
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [rows*cols]
    row_var: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [rows]
    row_psq: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [rows] scratch
    cols: Int, rows: Int, w_lerp: Float32,
):
    var r = Int(block_idx.x)
    if r >= rows:
        return
    var tid = Int(thread_idx.x)
    var sh_g = stack_allocation[
        _BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var sh_p = stack_allocation[
        _BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var base = r * cols
    var gs: Float32 = 0.0
    var ps: Float32 = 0.0
    var c = tid
    while c < cols:
        var gv = Float32(rebind[Scalar[DType.float32]](g[base + c]))
        gs += gv * gv
        var pv = Float32(rebind[Scalar[DType.bfloat16]](p[base + c]))
        ps += pv * pv
        c += _BLOCK
    sh_g[tid] = gs
    sh_p[tid] = ps
    barrier()
    var active = _BLOCK // 2
    while active > 0:
        if tid < active:
            sh_g[tid] = sh_g[tid] + sh_g[tid + active]
            sh_p[tid] = sh_p[tid] + sh_p[tid + active]
        barrier()
        active //= 2
    if tid == 0:
        var mean_g2 = Float32(sh_g[0]) / Float32(cols)
        var old = Float32(rebind[Scalar[DType.float32]](row_var[r]))
        row_var[r] = rebind[row_var.element_type](old + (mean_g2 - old) * w_lerp)
        row_psq[r] = rebind[row_psq.element_type](Float32(sh_p[0]))


# ── K2: block-per-col — col g² mean lerp into col_var ────────────────────────
def _af_col_kernel(
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [rows*cols]
    col_var: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [cols]
    cols: Int, rows: Int, w_lerp: Float32,
):
    var c = Int(block_idx.x)
    if c >= cols:
        return
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[
        _BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var gs: Float32 = 0.0
    var r = tid
    while r < rows:
        var gv = Float32(rebind[Scalar[DType.float32]](g[r * cols + c]))
        gs += gv * gv
        r += _BLOCK
    sh[tid] = gs
    barrier()
    var active = _BLOCK // 2
    while active > 0:
        if tid < active:
            sh[tid] = sh[tid] + sh[tid + active]
        barrier()
        active //= 2
    if tid == 0:
        var mean_g2 = Float32(sh[0]) / Float32(rows)
        var old = Float32(rebind[Scalar[DType.float32]](col_var[c]))
        col_var[c] = rebind[col_var.element_type](old + (mean_g2 - old) * w_lerp)


# ── K3: reduce a [n] F32 buffer to scalars[out_idx] (1-block tree) ────────────
def _af_reduce_kernel(
    src: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [n]
    scal: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [8]
    n: Int, out_idx: Int,
):
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[
        _BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var s: Float32 = 0.0
    var i = tid
    while i < n:
        s += Float32(rebind[Scalar[DType.float32]](src[i]))
        i += _BLOCK
    sh[tid] = s
    barrier()
    var active = _BLOCK // 2
    while active > 0:
        if tid < active:
            sh[tid] = sh[tid] + sh[tid + active]
        barrier()
        active //= 2
    if tid == 0:
        scal[out_idx] = rebind[scal.element_type](Float32(sh[0]))


# ── K4: update² partial sums (u recomputed exactly in K6) ────────────────────
def _af_updsq_kernel(
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],       # [n]
    row_var: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    col_var: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    scal: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [8]: [1]=row_var_sum
    parts: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [_PARTS]
    cols: Int, n: Int, rows: Int, eps1: Float32,
):
    var bid = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[
        _BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var row_mean = Float32(rebind[Scalar[DType.float32]](scal[1])) / Float32(rows)
    if row_mean < eps1:
        row_mean = eps1
    var s: Float32 = 0.0
    var i = bid * _BLOCK + tid
    var stride = _PARTS * _BLOCK
    while i < n:
        var r = i // cols
        var c = i % cols
        var ve = Float32(rebind[Scalar[DType.float32]](row_var[r])) * Float32(
            rebind[Scalar[DType.float32]](col_var[c])
        ) / row_mean
        if ve < eps1 * eps1:
            ve = eps1 * eps1
        var u = (1.0 / sqrt(ve)) * Float32(rebind[Scalar[DType.float32]](g[i]))
        s += u * u
        i += stride
    sh[tid] = s
    barrier()
    var active = _BLOCK // 2
    while active > 0:
        if tid < active:
            sh[tid] = sh[tid] + sh[tid + active]
        barrier()
        active //= 2
    if tid == 0:
        parts[bid] = rebind[parts.element_type](Float32(sh[0]))


# ── K5: 1-thread scalar combine → scal[3] = -alpha/denom ─────────────────────
def _af_scalars_kernel(
    scal: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [8]
    n: Int, rho_t: Float32, eps2: Float32, d: Float32,
):
    if Int(global_idx.x) == 0:
        var p_rms = sqrt(Float32(rebind[Scalar[DType.float32]](scal[0])) / Float32(n))
        var alpha = p_rms
        if alpha < eps2:
            alpha = eps2
        alpha = alpha * rho_t
        var upd_rms = sqrt(Float32(rebind[Scalar[DType.float32]](scal[2]))) / (
            sqrt(Float32(n)) * d
        )
        var denom: Float32 = 1.0
        if upd_rms > 1.0:
            denom = upd_rms
        scal[3] = rebind[scal.element_type](-(alpha / denom))


# ── K6: apply — p_bf16 = SR/RNE(f32(p)*wd + scale*u) ─────────────────────────
def _af_apply_kernel(
    p: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],      # [n] IN PLACE
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    row_var: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    col_var: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    scal: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [3]=scale [1]=row_var_sum
    cols: Int, n: Int, rows: Int,
    eps1: Float32, wd_mul: Float32, sr_seed: UInt64,
):
    var idx = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var row_mean = Float32(rebind[Scalar[DType.float32]](scal[1])) / Float32(rows)
    if row_mean < eps1:
        row_mean = eps1
    var scale = Float32(rebind[Scalar[DType.float32]](scal[3]))
    var i = idx
    while i < n:
        var r = i // cols
        var c = i % cols
        var ve = Float32(rebind[Scalar[DType.float32]](row_var[r])) * Float32(
            rebind[Scalar[DType.float32]](col_var[c])
        ) / row_mean
        if ve < eps1 * eps1:
            ve = eps1 * eps1
        var u = (1.0 / sqrt(ve)) * Float32(rebind[Scalar[DType.float32]](g[i]))
        var pv = Float32(rebind[Scalar[DType.bfloat16]](p[i])) * wd_mul
        var out = pv + scale * u
        # bf16 write. Mojo 1.0.0b1 exposes no scalar bit reinterpret in kernels
        # (see ops/torch_bf16.torch_bf16_rne_value), so STOCHASTIC ROUNDING is
        # done MATHEMATICALLY: on the bf16 grid at |out|'s binade (spacing
        # 2^(e-7)), round UP with probability = fractional position — exactly
        # the reference trainer add-rand16-truncate semantics at 16-bit granularity.
        # sr_seed == 0 selects RNE (plain cast — the parity-gate mode).
        if sr_seed != 0 and out == out and out != Float32(0.0):
            var sign = Float32(1.0)
            var a = out
            if a < Float32(0.0):
                sign = Float32(-1.0)
                a = -a
            var av = Float64(a)
            var e = Int(floor(log(av) / _AF_LN2))
            var step = Float64(2.0) ** Float64(e - 7)
            var y = av / step
            var kf = floor(y)
            var frac = y - kf
            var k = Int(kf)
            var h = sr_seed ^ (UInt64(i) * UInt64(0x9E3779B97F4A7C15))
            h = (h ^ (h >> 30)) * UInt64(0xBF58476D1CE4E5B9)
            h = (h ^ (h >> 27)) * UInt64(0x94D049BB133111EB)
            var r = Float64(h & UInt64(0xFFFF)) / Float64(65536.0)
            if r < frac:
                k += 1
            var q = sign * Float32(Float64(k) * step)
            p[i] = rebind[p.element_type](q.cast[DType.bfloat16]())
        else:
            p[i] = rebind[p.element_type](out.cast[DType.bfloat16]())
        i += stride


# ── device state (per weight), device-resident ───────────────────────────────
# factored=True  (2D [rows, cols]): row_var [rows] + col_var [cols] — the
#   original krea2 path, byte-unchanged.
# factored=False (1D [n], torch _adafactor.py:110-113 "variance"): row_var
#   carries the UNFACTORED exp_avg_sq [n] (rows=n), cols == 0 is the sentinel
#   and col_var is a dead [1] zeros tensor (kept allocated so the struct
#   layout and the Copyable synthesis stay uniform). The torch/reference trainer factored
#   threshold is dimensionality-only: torch `p.grad.dim() > 1` (:100/:387),
#   transformers `len(param_shape) >= 2` (optimization.py:1186) — a 1D param
#   is ALWAYS unfactored (a cols=1 factored fake is WRONG: col_var would
#   accumulate over all rows and the outer-product reconstruction is not
#   torch's v).
struct AdafactorDeviceState(Copyable, Movable):
    var row_var: TArc   # F32 [rows] (factored) / exp_avg_sq F32 [n] (unfactored)
    var col_var: TArc   # F32 [cols] (factored) / dead F32 [1] (unfactored)
    var rows: Int       # rows (factored) / n (unfactored)
    var cols: Int       # cols (factored) / 0 sentinel (unfactored)
    var factored: Bool

    def __init__(out self, rows: Int, cols: Int, ctx: DeviceContext) raises:
        self.row_var = TArc(zeros_device([rows], STDtype.F32, ctx))
        self.col_var = TArc(zeros_device([cols], STDtype.F32, ctx))
        self.rows = rows
        self.cols = cols
        self.factored = True

    def __init__(out self, n: Int, ctx: DeviceContext) raises:
        """UNFACTORED 1D state: exp_avg_sq [n] in row_var, cols=0 sentinel."""
        self.row_var = TArc(zeros_device([n], STDtype.F32, ctx))
        self.col_var = TArc(zeros_device([1], STDtype.F32, ctx))
        self.rows = n
        self.cols = 0
        self.factored = False


def _lt_f32(t: Tensor, n: Int) raises -> LayoutTensor[DType.float32, _DYN1, MutAnyOrigin]:
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    return LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        t.buf.unsafe_ptr().bitcast[Float32](), rl
    )


def adafactor_step_device(
    p_bf16: Tensor,           # [rows, cols] BF16 device weight — updated IN PLACE
    g_f32: Tensor,            # [rows, cols] F32 device grad
    state: AdafactorDeviceState,
    t_step: Int,              # 1-based step count (uniform across params)
    lr: Float64,
    beta2_decay: Float64,     # SimpleTuner torch-adafactor: -0.8
    eps1_in: Float64,         # <=0 -> torch F32 default 2^-23
    eps2: Float64,            # 1e-3
    d: Float64,               # 1.0
    weight_decay: Float64,    # 0.0
    sr_seed: UInt64,          # 0 = RNE (gate mode); else stochastic rounding
    ctx: DeviceContext,
) raises:
    """One torch-parity Adafactor step, fully on device (see module header)."""
    if p_bf16.dtype() != STDtype.BF16:
        raise Error("adafactor_step_device: p must be BF16")
    if g_f32.dtype() != STDtype.F32:
        raise Error("adafactor_step_device: g must be F32")
    if not state.factored:
        raise Error(
            "adafactor_step_device: state is UNFACTORED (1D) — use"
            " adafactor_step_device_1d"
        )
    var rows = state.rows
    var cols = state.cols
    var n = rows * cols
    if p_bf16.numel() != n or g_f32.numel() != n:
        raise Error("adafactor_step_device: p/g numel != rows*cols")

    var eps1 = eps1_in
    if eps1 <= 0.0:
        eps1 = Float64(1.1920928955078125e-07)
    var tt = Float64(t_step)
    var one_minus_beta2_t = tt**beta2_decay
    var rho_t = min(lr, 1.0 / sqrt(tt))
    var wd_mul = Float64(1.0)
    if weight_decay != 0.0:
        wd_mul = 1.0 - lr * weight_decay

    # scratch: scalars [8], row p² [rows], update² partials [_PARTS]
    var scal = zeros_device([8], STDtype.F32, ctx)
    var row_psq = zeros_device([rows], STDtype.F32, ctx)
    var parts = zeros_device([_PARTS], STDtype.F32, ctx)

    var nrl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var P = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        p_bf16.buf.unsafe_ptr().bitcast[BFloat16](), nrl
    )
    var G = _lt_f32(g_f32, n)
    var RV = _lt_f32(state.row_var[], rows)
    var CV = _lt_f32(state.col_var[], cols)
    var SC = _lt_f32(scal, 8)
    var RP = _lt_f32(row_psq, rows)
    var PT = _lt_f32(parts, _PARTS)

    # K1/K2: factored second-moment lerps (+ row p² partials, PRE-decay p).
    ctx.enqueue_function[_af_row_kernel, _af_row_kernel](
        P, G, RV, RP, cols, rows, Float32(one_minus_beta2_t),
        grid_dim=rows, block_dim=_BLOCK,
    )
    ctx.enqueue_function[_af_col_kernel, _af_col_kernel](
        G, CV, cols, rows, Float32(one_minus_beta2_t),
        grid_dim=cols, block_dim=_BLOCK,
    )
    # K3: p² total -> scal[0]; row_var sum -> scal[1].
    ctx.enqueue_function[_af_reduce_kernel, _af_reduce_kernel](
        RP, SC, rows, 0, grid_dim=1, block_dim=_BLOCK,
    )
    ctx.enqueue_function[_af_reduce_kernel, _af_reduce_kernel](
        RV, SC, rows, 1, grid_dim=1, block_dim=_BLOCK,
    )
    # K4: update² partials -> parts; reduce -> scal[2].
    ctx.enqueue_function[_af_updsq_kernel, _af_updsq_kernel](
        G, RV, CV, SC, PT, cols, n, rows, Float32(eps1),
        grid_dim=_PARTS, block_dim=_BLOCK,
    )
    ctx.enqueue_function[_af_reduce_kernel, _af_reduce_kernel](
        PT, SC, _PARTS, 2, grid_dim=1, block_dim=_BLOCK,
    )
    # K5: -alpha/denom -> scal[3].
    ctx.enqueue_function[_af_scalars_kernel, _af_scalars_kernel](
        SC, n, Float32(rho_t), Float32(eps2), Float32(d),
        grid_dim=1, block_dim=1,
    )
    # K6: apply + SR/RNE bf16 write IN PLACE.
    var grid = (n + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_af_apply_kernel, _af_apply_kernel](
        P, G, RV, CV, SC, cols, n, rows,
        Float32(eps1), Float32(wd_mul), sr_seed,
        grid_dim=grid, block_dim=_BLOCK,
    )


# ═════ 1D UNFACTORED path (torch _adafactor.py:404-416, the grad.dim() <= 1
# branch) — FULL_SURFACE_PLAN_2026-07-08 Phase A. Same scalar-bookkeeping
# discipline as the 2D path above: everything stays on device (scal[8] +
# partial-sum scratch), zero per-weight readback syncs; the SR/RNE bf16
# write is the SAME mathematical SR as K6 (duplicated verbatim so the 2D
# kernels stay byte-untouched). ═════════════════════════════════════════════


# ── K1d: fused per-element — p² partials (PRE-decay p, :381), v lerp
# (v.lerp_(g*g, w) = v + w*(g²−v), :407-408 — NO eps added to g²: torch
# applies eps1 only at the clamp before rsqrt, :413), and u² partials with
# u = g/sqrt(clamp(v_new, eps1²)) (:413-414). One pass, _PARTS blocks. ───────
def _af1_fused_kernel(
    p: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],   # [n]
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [n]
    v: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [n] exp_avg_sq IN PLACE
    parts_p: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [_PARTS]
    parts_u: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [_PARTS]
    n: Int, w_lerp: Float32, eps1: Float32,
):
    var bid = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var sh_p = stack_allocation[
        _BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var sh_u = stack_allocation[
        _BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var ps: Float32 = 0.0
    var us: Float32 = 0.0
    var i = bid * _BLOCK + tid
    var stride = _PARTS * _BLOCK
    while i < n:
        var pv = Float32(rebind[Scalar[DType.bfloat16]](p[i]))
        ps += pv * pv
        var gv = Float32(rebind[Scalar[DType.float32]](g[i]))
        var g2 = gv * gv
        var old = Float32(rebind[Scalar[DType.float32]](v[i]))
        var vn = old + w_lerp * (g2 - old)
        v[i] = rebind[v.element_type](vn)
        var ve = vn
        if ve < eps1 * eps1:
            ve = eps1 * eps1
        var u = (1.0 / sqrt(ve)) * gv
        us += u * u
        i += stride
    sh_p[tid] = ps
    sh_u[tid] = us
    barrier()
    var active = _BLOCK // 2
    while active > 0:
        if tid < active:
            sh_p[tid] = sh_p[tid] + sh_p[tid + active]
            sh_u[tid] = sh_u[tid] + sh_u[tid + active]
        barrier()
        active //= 2
    if tid == 0:
        parts_p[bid] = rebind[parts_p.element_type](Float32(sh_p[0]))
        parts_u[bid] = rebind[parts_u.element_type](Float32(sh_u[0]))


# ── K2d: apply — p_bf16 = SR/RNE(f32(p)*wd + scale*u), u recomputed exactly
# from the (already-lerped) v; v itself stays UNCLAMPED (:410 clones before
# the clamp). scale = scal[3] = -alpha/denom from the shared K5. ─────────────
def _af1_apply_kernel(
    p: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],      # [n] IN PLACE
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    v: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    scal: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [3]=scale
    n: Int, eps1: Float32, wd_mul: Float32, sr_seed: UInt64,
):
    var idx = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var scale = Float32(rebind[Scalar[DType.float32]](scal[3]))
    var i = idx
    while i < n:
        var ve = Float32(rebind[Scalar[DType.float32]](v[i]))
        if ve < eps1 * eps1:
            ve = eps1 * eps1
        var u = (1.0 / sqrt(ve)) * Float32(rebind[Scalar[DType.float32]](g[i]))
        var pv = Float32(rebind[Scalar[DType.bfloat16]](p[i])) * wd_mul
        var out = pv + scale * u
        # SR/RNE bf16 write — VERBATIM the K6 block (mathematical SR on the
        # bf16 grid; sr_seed == 0 selects RNE, the parity-gate mode).
        if sr_seed != 0 and out == out and out != Float32(0.0):
            var sign = Float32(1.0)
            var a = out
            if a < Float32(0.0):
                sign = Float32(-1.0)
                a = -a
            var av = Float64(a)
            var e = Int(floor(log(av) / _AF_LN2))
            var step = Float64(2.0) ** Float64(e - 7)
            var y = av / step
            var kf = floor(y)
            var frac = y - kf
            var k = Int(kf)
            var h = sr_seed ^ (UInt64(i) * UInt64(0x9E3779B97F4A7C15))
            h = (h ^ (h >> 30)) * UInt64(0xBF58476D1CE4E5B9)
            h = (h ^ (h >> 27)) * UInt64(0x94D049BB133111EB)
            var r = Float64(h & UInt64(0xFFFF)) / Float64(65536.0)
            if r < frac:
                k += 1
            var q = sign * Float32(Float64(k) * step)
            p[i] = rebind[p.element_type](q.cast[DType.bfloat16]())
        else:
            p[i] = rebind[p.element_type](out.cast[DType.bfloat16]())
        i += stride


def adafactor_step_device_1d(
    p_bf16: Tensor,           # [n] BF16 device weight — updated IN PLACE
    g_f32: Tensor,            # [n] F32 device grad
    state: AdafactorDeviceState,   # UNFACTORED (factored == False)
    t_step: Int,              # 1-based step count (uniform across params)
    lr: Float64,
    beta2_decay: Float64,     # -0.8
    eps1_in: Float64,         # <=0 -> torch F32 default 2^-23
    eps2: Float64,            # 1e-3
    d: Float64,               # 1.0
    weight_decay: Float64,    # 0.0 — reference trainer applies wd uniformly (no 1D carve-out)
    sr_seed: UInt64,          # 0 = RNE (gate mode); else stochastic rounding
    ctx: DeviceContext,
) raises:
    """One torch-parity Adafactor step for a 1D param, fully on device — the
    UNFACTORED branch (torch _adafactor.py:404-410 + shared :413-416; host
    oracle training/adafactor.mojo::adafactor_step_1d). Kernel plan: K1d
    fused (p² partials + v lerp + u² partials) → K3 reduce ×2 (p²→scal[0],
    u²→scal[2]) → K5 (UNCHANGED 2D scalar combine: alpha=max(eps2,RMS(p))·
    rho_t :381, denom=max(1,RMS(u)/d) :415 → scal[3]=-alpha/denom) → K2d
    apply with SR/RNE. Same stream/sync discipline as the 2D step: nothing
    reads back to the host."""
    if p_bf16.dtype() != STDtype.BF16:
        raise Error("adafactor_step_device_1d: p must be BF16")
    if g_f32.dtype() != STDtype.F32:
        raise Error("adafactor_step_device_1d: g must be F32")
    if state.factored or state.cols != 0:
        raise Error(
            "adafactor_step_device_1d: state is FACTORED (2D) — use"
            " adafactor_step_device"
        )
    var n = state.rows
    if p_bf16.numel() != n or g_f32.numel() != n:
        raise Error("adafactor_step_device_1d: p/g numel != n")
    if state.row_var[].numel() != n:
        raise Error("adafactor_step_device_1d: exp_avg_sq numel != n")

    var eps1 = eps1_in
    if eps1 <= 0.0:
        eps1 = Float64(1.1920928955078125e-07)
    var tt = Float64(t_step)
    var one_minus_beta2_t = tt**beta2_decay   # :379
    var rho_t = min(lr, 1.0 / sqrt(tt))       # :380
    var wd_mul = Float64(1.0)
    if weight_decay != 0.0:
        wd_mul = 1.0 - lr * weight_decay      # :384-385 stepweight decay

    # scratch: scalars [8], p² partials [_PARTS], update² partials [_PARTS]
    var scal = zeros_device([8], STDtype.F32, ctx)
    var parts_p = zeros_device([_PARTS], STDtype.F32, ctx)
    var parts_u = zeros_device([_PARTS], STDtype.F32, ctx)

    var nrl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var P = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        p_bf16.buf.unsafe_ptr().bitcast[BFloat16](), nrl
    )
    var G = _lt_f32(g_f32, n)
    var V = _lt_f32(state.row_var[], n)
    var SC = _lt_f32(scal, 8)
    var PP = _lt_f32(parts_p, _PARTS)
    var PU = _lt_f32(parts_u, _PARTS)

    # K1d: p² partials + v lerp + u² partials in one pass.
    ctx.enqueue_function[_af1_fused_kernel, _af1_fused_kernel](
        P, G, V, PP, PU, n, Float32(one_minus_beta2_t), Float32(eps1),
        grid_dim=_PARTS, block_dim=_BLOCK,
    )
    # K3: p² total -> scal[0]; update² total -> scal[2] (scal[1] unused here).
    ctx.enqueue_function[_af_reduce_kernel, _af_reduce_kernel](
        PP, SC, _PARTS, 0, grid_dim=1, block_dim=_BLOCK,
    )
    ctx.enqueue_function[_af_reduce_kernel, _af_reduce_kernel](
        PU, SC, _PARTS, 2, grid_dim=1, block_dim=_BLOCK,
    )
    # K5 (shared with 2D — it only reads scal[0]/scal[2]): scal[3]=-alpha/denom.
    ctx.enqueue_function[_af_scalars_kernel, _af_scalars_kernel](
        SC, n, Float32(rho_t), Float32(eps2), Float32(d),
        grid_dim=1, block_dim=1,
    )
    # K2d: apply + SR/RNE bf16 write IN PLACE.
    var grid = (n + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_af1_apply_kernel, _af1_apply_kernel](
        P, G, V, SC, n, Float32(eps1), Float32(wd_mul), sr_seed,
        grid_dim=grid, block_dim=_BLOCK,
    )
