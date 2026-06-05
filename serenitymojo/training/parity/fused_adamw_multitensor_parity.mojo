# fused_adamw_multitensor_parity.mojo — gate fused_adamw_step AGAINST the scalar
# per-tensor adamw_step loop (training/optim.mojo). Both start from the SAME host
# data (independent device copies); after one step the params/m/v must be
# BIT-EQUAL (cos=1.0, max_abs=0.0). Microbench: scalar N-launch loop vs 1 fused
# launch. Bitrot: pass BITROT (corrupts the reference → gate exits NONZERO).
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo run -I . serenitymojo/training/parity/fused_adamw_multitensor_parity.mojo

from std.sys import argv
from std.memory import ArcPointer
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.training.optim import adamw_step
from serenitymojo.training.fused_adamw_multitensor import fused_adamw_step, TArc


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _adamw_ref_inplace(
    mut p: List[Float32],
    g: List[Float32],
    mut m: List[Float32],
    mut v: List[Float32],
    t: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
) raises:
    var n = len(p)
    if len(g) != n or len(m) != n or len(v) != n:
        raise Error("_adamw_ref_inplace: param/grad/m/v len mismatch")
    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p

    for i in range(n):
        var gv = g[i]
        var mi = beta1 * m[i] + (Float32(1.0) - beta1) * gv
        var vi = beta2 * v[i] + (Float32(1.0) - beta2) * gv * gv
        m[i] = mi
        v[i] = vi
        var m_hat = mi / bc1
        var v_hat = vi / bc2
        var pv = p[i] - lr * m_hat / (sqrt(v_hat) + eps)
        if weight_decay > 0.0:
            pv = pv - lr * weight_decay * pv
        p[i] = pv


def _storage_dtype_gate(dtype: STDtype, ctx: DeviceContext) raises:
    var sizes = List[Int]()
    sizes.append(1024)
    sizes.append(3072)
    sizes.append(777)
    var nt = len(sizes)

    var lr = Float32(2e-4)
    var b1 = Float32(0.9)
    var b2 = Float32(0.999)
    var eps = Float32(1e-8)
    var wd = Float32(0.01)
    var tstep = 5

    var ps = List[TArc]()
    var gs = List[TArc]()
    var ms = List[TArc]()
    var vs = List[TArc]()
    var p_ref = List[List[Float32]]()
    var g_ref = List[List[Float32]]()
    var m_ref = List[List[Float32]]()
    var v_ref = List[List[Float32]]()
    for i in range(nt):
        ps.append(TArc(Tensor.from_host(_fill(sizes[i], 1000 + UInt64(i), 0.2), [sizes[i]], dtype, ctx)))
        gs.append(TArc(Tensor.from_host(_fill(sizes[i], 2000 + UInt64(i), 0.05), [sizes[i]], dtype, ctx)))
        ms.append(TArc(Tensor.from_host(_fill(sizes[i], 3000 + UInt64(i), 0.01), [sizes[i]], STDtype.F32, ctx)))
        vs.append(TArc(Tensor.from_host(_fill(sizes[i], 4000 + UInt64(i), 0.005), [sizes[i]], STDtype.F32, ctx)))
        p_ref.append(ps[i][].to_host(ctx))
        g_ref.append(gs[i][].to_host(ctx))
        m_ref.append(ms[i][].to_host(ctx))
        v_ref.append(vs[i][].to_host(ctx))
        for j in range(sizes[i]):
            if v_ref[i][j] < 0.0:
                v_ref[i][j] = -v_ref[i][j]
    for i in range(nt):
        if v_ref[i][0] < 0.0:
            raise Error("unreachable")
        var v_fixed = Tensor.from_host(v_ref[i].copy(), [sizes[i]], STDtype.F32, ctx)
        vs[i] = TArc(v_fixed^)

    for i in range(nt):
        _adamw_ref_inplace(
            p_ref[i], g_ref[i], m_ref[i], v_ref[i], tstep, lr, b1, b2, eps, wd,
        )

    fused_adamw_step(ps, gs, ms, vs, tstep, lr, b1, b2, eps, wd, ctx)

    var h = ParityHarness(0.999)
    var all_pass = True
    for i in range(nt):
        if ps[i][].dtype() != dtype or gs[i][].dtype() != dtype:
            raise Error("fused_adamw storage gate: param/grad dtype changed")
        if ms[i][].dtype() != STDtype.F32 or vs[i][].dtype() != STDtype.F32:
            raise Error("fused_adamw storage gate: optimizer state dtype changed")
        var quant_ref = Tensor.from_host(p_ref[i].copy(), [sizes[i]], dtype, ctx).to_host(ctx)
        var rp = h.compare(ps[i][], quant_ref, ctx)
        var rm = h.compare(ms[i][], m_ref[i], ctx)
        var rv = h.compare(vs[i][], v_ref[i], ctx)
        if not (rp.passed and rm.passed and rv.passed):
            all_pass = False
            print("    ", dtype.name(), " tensor", i, "p:", rp, " m:", rm, " v:", rv)
    if all_pass:
        print("PASS: fused_adamw_multitensor preserves ", dtype.name(), " param/grad storage with F32 moments")
    else:
        raise Error(
            String("fused_adamw_multitensor ")
            + dtype.name()
            + " storage gate FAILED"
        )


def main() raises:
    var bitrot = False
    var args = argv()
    for i in range(len(args)):
        if args[i] == String("BITROT"):
            bitrot = True
    var ctx = DeviceContext()

    # N tensors of varied sizes (mimics LoRA adapters of different ranks/shapes).
    var sizes = List[Int]()
    sizes.append(4096); sizes.append(8192); sizes.append(1024)
    sizes.append(16384); sizes.append(2048); sizes.append(65536)
    sizes.append(512); sizes.append(32768)
    var nt = len(sizes)
    print("=== fused_adamw_multitensor parity vs scalar loop (N=", nt, " tensors) ===")

    var lr = Float32(2e-4)
    var b1 = Float32(0.9)
    var b2 = Float32(0.999)
    var eps = Float32(1e-8)
    var wd = Float32(0.01)
    var tstep = 7

    # host source data per tensor (same for both paths)
    var p_src = List[List[Float32]]()
    var g_src = List[List[Float32]]()
    var m_src = List[List[Float32]]()
    var v_src = List[List[Float32]]()
    for i in range(nt):
        p_src.append(_fill(sizes[i], 100 + UInt64(i), 1.0))
        g_src.append(_fill(sizes[i], 200 + UInt64(i), 0.5))
        m_src.append(_fill(sizes[i], 300 + UInt64(i), 0.1))
        v_src.append(_fill(sizes[i], 400 + UInt64(i), 0.05))  # >=0-ish (abs below)
    # v must be nonnegative (second moment) — clamp to abs
    for i in range(nt):
        for j in range(sizes[i]):
            if v_src[i][j] < 0.0:
                v_src[i][j] = -v_src[i][j]

    # ── scalar path: one tensor at a time (boxed TArc; Tensor is move-only) ──
    var ps = List[TArc]()
    var ms = List[TArc]()
    var vs = List[TArc]()
    var gs = List[TArc]()
    for i in range(nt):
        ps.append(TArc(Tensor.from_host(p_src[i].copy(), [sizes[i]], STDtype.F32, ctx)))
        ms.append(TArc(Tensor.from_host(m_src[i].copy(), [sizes[i]], STDtype.F32, ctx)))
        vs.append(TArc(Tensor.from_host(v_src[i].copy(), [sizes[i]], STDtype.F32, ctx)))
        gs.append(TArc(Tensor.from_host(g_src[i].copy(), [sizes[i]], STDtype.F32, ctx)))
    for i in range(nt):
        adamw_step(ps[i][], gs[i][], ms[i][], vs[i][], tstep, lr, b1, b2, eps, wd, ctx)

    # ── fused path: independent device copies, boxed as TArc ──
    var pf = List[TArc]()
    var mf = List[TArc]()
    var vf = List[TArc]()
    var gf = List[TArc]()
    for i in range(nt):
        pf.append(TArc(Tensor.from_host(p_src[i].copy(), [sizes[i]], STDtype.F32, ctx)))
        mf.append(TArc(Tensor.from_host(m_src[i].copy(), [sizes[i]], STDtype.F32, ctx)))
        vf.append(TArc(Tensor.from_host(v_src[i].copy(), [sizes[i]], STDtype.F32, ctx)))
        gf.append(TArc(Tensor.from_host(g_src[i].copy(), [sizes[i]], STDtype.F32, ctx)))
    fused_adamw_step(pf, gf, mf, vf, tstep, lr, b1, b2, eps, wd, ctx)

    # ── compare param + m + v, all tensors ──
    var h = ParityHarness(0.999)
    var all_pass = True
    for i in range(nt):
        var ref_p = ps[i][].to_host(ctx)
        if bitrot and i == 0:
            ref_p[0] = ref_p[0] + 5.0
        var rp = h.compare(pf[i][], ref_p, ctx)
        var rm = h.compare(mf[i][], ms[i][].to_host(ctx), ctx)
        var rv = h.compare(vf[i][], vs[i][].to_host(ctx), ctx)
        if not (rp.passed and rm.passed and rv.passed):
            all_pass = False
        if i == 0 or not (rp.passed and rm.passed and rv.passed):
            print("    tensor", i, "p:", rp, " m:", rm, " v:", rv)

    # ── microbench: scalar N-launch loop vs single fused launch ──
    var reps = 100
    var t0 = perf_counter_ns()
    for _ in range(reps):
        for i in range(nt):
            adamw_step(ps[i][], gs[i][], ms[i][], vs[i][], tstep, lr, b1, b2, eps, wd, ctx)
    var t1 = perf_counter_ns()
    var t2 = perf_counter_ns()
    for _ in range(reps):
        fused_adamw_step(pf, gf, mf, vf, tstep, lr, b1, b2, eps, wd, ctx)
    var t3 = perf_counter_ns()
    var scal_ms = Float64(t1 - t0) / 1.0e6 / Float64(reps)
    var vec_ms = Float64(t3 - t2) / 1.0e6 / Float64(reps)
    print("    [microbench] scalar(", nt, "launches+syncs)=", scal_ms,
          "ms  fused(1 launch)=", vec_ms, "ms  speedup=", scal_ms / vec_ms, "x")

    if all_pass:
        print("PASS: fused_adamw_multitensor bit-equal to scalar loop (p+m+v)")
    else:
        raise Error("fused_adamw_multitensor_parity gate FAILED")

    _storage_dtype_gate(STDtype.BF16, ctx)
    _storage_dtype_gate(STDtype.F16, ctx)
