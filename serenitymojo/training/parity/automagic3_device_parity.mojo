# automagic3_device_parity.mojo — ORACLE GATE for the GPU automagic3 step.
#
# Drives the HOST automagic3 (already gated == ai-toolkit, see
# AUTOMAGIC3_SKEPTIC_FINDINGS B1/probe) and the DEVICE kernel on byte-identical
# inputs for N steps, then compares the F32 master params, the factored row/col
# 2nd-moment state, and the self-adapted lr trajectory. Device must match host
# within the oracle bar (params/state rel small; lr rel <= 1e-5 — GPU F64
# reductions vs host F64 differ only by summation order).
#
# Build:
#   pixi run mojo build --optimization-level 2 -I . \
#     serenitymojo/training/parity/automagic3_device_parity.mojo -o /tmp/a3dev_gate

from std.math import sqrt, exp, abs
from std.gpu.host import DeviceContext
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.training.automagic3 import (
    Automagic3State, Automagic3Ctl, automagic3_step_2d,
)
from serenitymojo.training.automagic3_device import automagic3_factored_kernel

comptime _DYN1 = Layout.row_major(-1)


def _lcg(mut s: UInt64) -> Float32:
    # small deterministic uniform in (-1,1) for reproducible grads/inits
    s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    var u = Float32(Int((s >> 33) & UInt64(0x7FFFFF))) / Float32(8388608.0)
    return (u * Float32(2.0)) - Float32(1.0)


def _dyn(buf_ptr: UnsafePointer[Float32, MutAnyOrigin], n: Int) -> LayoutTensor[DType.float32, _DYN1, MutAnyOrigin]:
    return LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](buf_ptr, RuntimeLayout[_DYN1].row_major(IndexList[1](n)))


def main() raises:
    var ctx = DeviceContext()

    # two LoRA-shaped matrices: A[8,16], B[16,8]
    comptime R0 = 8
    comptime C0 = 16
    comptime R1 = 16
    comptime C1 = 8
    comptime N0 = R0 * C0
    comptime N1 = R1 * C1
    comptime NP = N0 + N1
    comptime NR = R0 + R1   # row_var total
    comptime NC = C0 + C1   # col_var total
    comptime H = 8
    comptime STEPS = 12

    var beta2 = Float64(0.999)
    var eps = Float64(1.0e-30)
    var clip = Float64(1.0)
    var wd = Float64(1.0e-5)
    var start_lr = Float64(1.0e-4)

    # ── shared init: params (small) ──
    var seed = UInt64(12345)
    var p0 = List[Float32]()
    var p1 = List[Float32]()
    for _ in range(N0):
        p0.append(Float32(0.02) * _lcg(seed))
    for _ in range(N1):
        p1.append(Float32(0.02) * _lcg(seed))

    # ── HOST run ──
    var hp0 = p0.copy()
    var hp1 = p1.copy()
    var hst0 = Automagic3State(R0, C0, H)
    var hst1 = Automagic3State(R1, C1, H)
    var hctl = Automagic3Ctl()
    hctl.init_lr(start_lr)
    var host_lr_traj = List[Float64]()
    var gseed = UInt64(999)
    # capture the per-step grads so DEVICE sees byte-identical grads
    var all_g0 = List[List[Float32]]()
    var all_g1 = List[List[Float32]]()
    for _ in range(STEPS):
        var g0 = List[Float32]()
        var g1 = List[Float32]()
        for _ in range(N0):
            g0.append(Float32(0.01) * _lcg(gseed))
        for _ in range(N1):
            g1.append(Float32(0.01) * _lcg(gseed))
        hctl.reset_accum()
        automagic3_step_2d(hp0, g0, hst0, beta2, eps, clip, wd, hctl)
        automagic3_step_2d(hp1, g1, hst1, beta2, eps, clip, wd, hctl)
        hctl.apply_vote()
        host_lr_traj.append(hctl.lr)
        all_g0.append(g0^)
        all_g1.append(g1^)

    # ── DEVICE run ──
    var p_dev = ctx.enqueue_create_buffer[DType.float32](NP)
    var g_dev = ctx.enqueue_create_buffer[DType.float32](NP)
    var upd_dev = ctx.enqueue_create_buffer[DType.float32](NP)
    var rv_dev = ctx.enqueue_create_buffer[DType.float32](NR)
    var cv_dev = ctx.enqueue_create_buffer[DType.float32](NC)
    var sr_dev = ctx.enqueue_create_buffer[DType.uint8](H * NP)
    var desc_dev = ctx.enqueue_create_buffer[DType.int32](2 * 6)
    var hidx_dev = ctx.enqueue_create_buffer[DType.int32](2)
    var hfill_dev = ctx.enqueue_create_buffer[DType.int32](2)
    var gnum_dev = ctx.enqueue_create_buffer[DType.float64](1)
    var gden_dev = ctx.enqueue_create_buffer[DType.float64](1)
    var pb_dev = ctx.enqueue_create_buffer[DType.bfloat16](NP)

    # upload initial params (p0 ++ p1), zero state, descriptors
    var ph = ctx.enqueue_create_host_buffer[DType.float32](NP)
    for i in range(N0): ph.unsafe_ptr()[i] = p0[i]
    for i in range(N1): ph.unsafe_ptr()[N0 + i] = p1[i]
    ctx.enqueue_copy(dst_buf=p_dev, src_buf=ph)
    rv_dev.enqueue_fill(Float32(0.0))
    cv_dev.enqueue_fill(Float32(0.0))
    sr_dev.enqueue_fill(UInt8(0))
    hidx_dev.enqueue_fill(Int32(0))
    hfill_dev.enqueue_fill(Int32(0))
    # descriptors: rows,cols,goff,roff,coff,soff
    var dh = ctx.enqueue_create_host_buffer[DType.int32](12)
    var dp = dh.unsafe_ptr()
    dp[0]=R0; dp[1]=C0; dp[2]=0;  dp[3]=0;  dp[4]=0;  dp[5]=0
    dp[6]=R1; dp[7]=C1; dp[8]=N0; dp[9]=R0; dp[10]=C0; dp[11]=H*N0
    ctx.enqueue_copy(dst_buf=desc_dev, src_buf=dh)

    var P = _dyn(p_dev.unsafe_ptr(), NP)
    var G = _dyn(g_dev.unsafe_ptr(), NP)
    var U = _dyn(upd_dev.unsafe_ptr(), NP)
    var RV = _dyn(rv_dev.unsafe_ptr(), NR)
    var CV = _dyn(cv_dev.unsafe_ptr(), NC)
    var SR = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](sr_dev.unsafe_ptr(), RuntimeLayout[_DYN1].row_major(IndexList[1](H * NP)))
    var DSC = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](desc_dev.unsafe_ptr(), RuntimeLayout[_DYN1].row_major(IndexList[1](12)))
    var HIDX = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](hidx_dev.unsafe_ptr(), RuntimeLayout[_DYN1].row_major(IndexList[1](2)))
    var HFILL = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](hfill_dev.unsafe_ptr(), RuntimeLayout[_DYN1].row_major(IndexList[1](2)))
    var GNUM = LayoutTensor[DType.float64, _DYN1, MutAnyOrigin](gnum_dev.unsafe_ptr(), RuntimeLayout[_DYN1].row_major(IndexList[1](1)))
    var GDEN = LayoutTensor[DType.float64, _DYN1, MutAnyOrigin](gden_dev.unsafe_ptr(), RuntimeLayout[_DYN1].row_major(IndexList[1](1)))
    var PB = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](pb_dev.unsafe_ptr(), RuntimeLayout[_DYN1].row_major(IndexList[1](NP)))

    var dev_lr = start_lr
    var dev_lr_traj = List[Float64]()
    var gh = ctx.enqueue_create_host_buffer[DType.float32](NP)
    var num_h = ctx.enqueue_create_host_buffer[DType.float64](1)
    var den_h = ctx.enqueue_create_host_buffer[DType.float64](1)
    for s in range(STEPS):
        # upload this step's grads (byte-identical to host)
        for i in range(N0): gh.unsafe_ptr()[i] = all_g0[s][i]
        for i in range(N1): gh.unsafe_ptr()[N0 + i] = all_g1[s][i]
        ctx.enqueue_copy(dst_buf=g_dev, src_buf=gh)
        gnum_dev.enqueue_fill(Float64(0.0))
        gden_dev.enqueue_fill(Float64(0.0))
        ctx.enqueue_function[automagic3_factored_kernel, automagic3_factored_kernel](
            P, G, U, RV, CV, SR, DSC, HIDX, HFILL, GNUM, GDEN, PB,
            Float32(beta2), Float32(1.0 - beta2), Float32(eps), Float32(clip),
            Float32(dev_lr), Float32(wd), Float32(1.0),
            s, UInt64(0x5EED_A3D0),
            grid_dim=2, block_dim=256,
        )
        ctx.enqueue_copy(dst_buf=num_h, src_buf=gnum_dev)
        ctx.enqueue_copy(dst_buf=den_h, src_buf=gden_dev)
        ctx.synchronize()
        # host apply_vote (mirror Automagic3Ctl.apply_vote)
        var gn = num_h.unsafe_ptr()[0]
        var gd = den_h.unsafe_ptr()[0]
        if gd > 0.0:
            var den = gd
            if den < 1.0e-30: den = 1.0e-30
            var sig = gn / den
            if sig > 1.0: sig = 1.0
            elif sig < -1.0: sig = -1.0
            var nl = dev_lr * exp(sig)
            if nl < 1.0e-30: nl = 1.0e-30
            elif nl > 1.0e3: nl = 1.0e3
            dev_lr = nl
        dev_lr_traj.append(dev_lr)

    # download device params + state
    var pres = ctx.enqueue_create_host_buffer[DType.float32](NP)
    var rvres = ctx.enqueue_create_host_buffer[DType.float32](NR)
    var cvres = ctx.enqueue_create_host_buffer[DType.float32](NC)
    ctx.enqueue_copy(dst_buf=pres, src_buf=p_dev)
    ctx.enqueue_copy(dst_buf=rvres, src_buf=rv_dev)
    ctx.enqueue_copy(dst_buf=cvres, src_buf=cv_dev)
    ctx.synchronize()

    # ── compare ──
    fn relmax(a: List[Float32], bp: UnsafePointer[Float32, MutAnyOrigin], off: Int, n: Int) -> Float64:
        var m = Float64(0.0)
        for i in range(n):
            var d = abs(Float64(a[i]) - Float64(bp[off + i]))
            var sc = abs(Float64(a[i])) + 1.0e-12
            var r = d / sc
            if r > m: m = r
        return m

    var p0_rel = relmax(hp0, pres.unsafe_ptr(), 0, N0)
    var p1_rel = relmax(hp1, pres.unsafe_ptr(), N0, N1)
    var rv0_rel = relmax(hst0.row_var, rvres.unsafe_ptr(), 0, R0)
    var rv1_rel = relmax(hst1.row_var, rvres.unsafe_ptr(), R0, R1)
    var cv0_rel = relmax(hst0.col_var, cvres.unsafe_ptr(), 0, C0)
    var cv1_rel = relmax(hst1.col_var, cvres.unsafe_ptr(), C0, C1)
    var lr_rel = Float64(0.0)
    for s in range(STEPS):
        var d = abs(host_lr_traj[s] - dev_lr_traj[s]) / (abs(host_lr_traj[s]) + 1.0e-30)
        if d > lr_rel: lr_rel = d

    print("=== automagic3 DEVICE vs HOST parity (", STEPS, "steps) ===")
    print("param  rel: A", p0_rel, " B", p1_rel)
    print("rowvar rel: A", rv0_rel, " B", rv1_rel)
    print("colvar rel: A", cv0_rel, " B", cv1_rel)
    print("lr-traj max rel:", lr_rel, " host_lr[-1]=", host_lr_traj[STEPS-1], " dev_lr[-1]=", dev_lr_traj[STEPS-1])
    var bar = Float64(1.0e-4)
    var ok = (p0_rel < bar and p1_rel < bar and rv0_rel < bar and rv1_rel < bar
              and cv0_rel < bar and cv1_rel < bar and lr_rel < bar)
    if ok:
        print("DEVICE_PARITY: PASS (all rel <", bar, ")")
    else:
        print("DEVICE_PARITY: FAIL")
