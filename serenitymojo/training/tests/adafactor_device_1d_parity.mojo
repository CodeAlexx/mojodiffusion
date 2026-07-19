# training/tests/adafactor_device_1d_parity.mojo — the 1D UNFACTORED device
# Adafactor (training/adafactor_device.mojo::adafactor_step_device_1d) vs the
# host torch-oracle (training/adafactor.mojo::adafactor_step_1d — the
# torch/optim/_adafactor.py:404-416 grad.dim()<=1 branch, line-by-line).
#
# FULL_SURFACE_PLAN_2026-07-08 Phase A gate. Trajectory (3 steps, fresh
# non-degenerate grads each step) on a [4096] F32 param, host p RNE-rounded to
# bf16 after each step (the device carrier is bf16) so both walk the same
# discretized trajectory. GATES (the 2D gate's achieved standard, hard):
#   * p BIT-IDENTICAL: max_abs(p_dev - p_host) == 0.0
#   * v EXACT:         max_abs(v_dev - v_host) == 0.0 (the lerp is elementwise
#     F32 with identical operand order — no reduction to blur it)
#   * SR mode (sr_seed != 0): moved fraction sane (> 0), ZERO elements more
#     than one bf16 ulp from the RNE result.
# Plus the sidecar v2 mixed-state roundtrip + v1 back-compat smoke (the other
# Phase A deliverable, full_ft_sidecar.mojo).
#
# Build/run (handoff §3 flags):
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 -I . \
#     -I /home/alex/MOJO-libs -Xlinker -lm \
#     serenitymojo/training/tests/adafactor_device_1d_parity.mojo -o /tmp/af_dev_1d_parity
#   LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/af_dev_1d_parity

from std.gpu.host import DeviceContext
from std.math import sqrt, sin
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.training.adafactor import AdafactorState1D, adafactor_step_1d
from serenitymojo.training.adafactor_device import (
    AdafactorDeviceState, adafactor_step_device, adafactor_step_device_1d,
)
from serenitymojo.training.full_ft_sidecar import (
    full_ft_sidecar_save, full_ft_sidecar_load,
)

comptime N = 4096

comptime LR = Float64(3e-4)
comptime B2D = Float64(-0.8)
comptime EPS2 = Float64(1e-3)
comptime DD = Float64(1.0)
comptime WD = Float64(0.0)

comptime SIDE_V2 = "/mnt/disk1/full_ft_resume_gate/afd1d_sidecar_v2_smoke.safetensors"
comptime SIDE_V1 = "/mnt/disk1/full_ft_resume_gate/afd1d_sidecar_v1_smoke.safetensors"


def _bf16_rne(x: Float32) -> Float32:
    return Float32(SIMD[DType.float32, 1](x).cast[DType.bfloat16]().cast[DType.float32]()[0])


def _fill(n: Int, a: Float32, b: Float32, c: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(sin(a * Float32(i) + b) * c)
    return out^


def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var dot: Float64 = 0.0; var na: Float64 = 0.0; var nb: Float64 = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    return dot / (sqrt(na) * sqrt(nb) + 1e-30)


def _maxabs(a: List[Float32], b: List[Float32]) -> Float32:
    var m: Float32 = 0.0
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < 0: d = -d
        if d > m: m = d
    return m


def _read_sidecar_version(path: String) raises -> Int:
    """meta field 4 of __af_meta__ (LE u32), read raw from the file."""
    var st = SafeTensors.open(path)
    var span = st.tensor_bytes(String("__af_meta__"))
    var base = 4 * 4
    var v = UInt32(0)
    for b in range(4):
        v = v | (UInt32(span[base + b]) << UInt32(8 * b))
    return Int(v)


def main() raises:
    var ctx = DeviceContext()
    var allok = True
    print("==== adafactor_device_1d_parity (unfactored device kernels vs host torch-oracle) ====")
    print("N=", N, " steps=3  lr=", LR)

    # bf16-representable start so host(F32) and device(bf16) see the same p0.
    var p_host = List[Float32]()
    var p0 = _fill(N, 0.0007, 0.11, 0.5)
    for i in range(N):
        p_host.append(_bf16_rne(p0[i]))
    var p_dev = Tensor.from_host(p_host.copy(), [N], STDtype.BF16, ctx)

    var hstate = AdafactorState1D(N)
    var dstate = AdafactorDeviceState(N, ctx)   # unfactored ctor
    if dstate.factored or dstate.cols != 0:
        raise Error("unfactored ctor did not set the sentinel")

    for step in range(1, 4):
        var g = _fill(N, 0.0011 * Float32(step), 0.07 + 0.3 * Float32(step), 0.05)
        var g_dev = Tensor.from_host(g.copy(), [N], STDtype.F32, ctx)

        adafactor_step_1d(p_host, g, hstate, LR, B2D, Float64(-1.0), EPS2, DD, WD)
        # bf16 carrier: discretize the host trajectory like the device one.
        for i in range(N):
            p_host[i] = _bf16_rne(p_host[i])

        adafactor_step_device_1d(
            p_dev, g_dev, dstate, step, LR, B2D, Float64(-1.0), EPS2, DD, WD,
            UInt64(0), ctx,   # RNE gate mode
        )
        ctx.synchronize()

    var p_dev_h = p_dev.to_host(ctx)
    var v_h = dstate.row_var[].to_host(ctx)

    var cp = _cos(p_dev_h, p_host)
    var cv = _cos(v_h, hstate.v)
    var map_ = _maxabs(p_dev_h, p_host)
    var mav = _maxabs(v_h, hstate.v)
    print("cos(p) =", cp, "  max_abs(p) =", map_)
    print("cos(v) =", cv, "  max_abs(v) =", mav)
    if map_ != Float32(0.0):
        print("FAIL: p not bit-identical to the host oracle")
        allok = False
    if mav != Float32(0.0):
        print("FAIL: v (exp_avg_sq) not exact vs the host oracle")
        allok = False

    # ── SR bound: sr result within 1 bf16 ulp of the RNE result ─────────────
    var p_a = Tensor.from_host(p_dev_h.copy(), [N], STDtype.BF16, ctx)
    var p_b = Tensor.from_host(p_dev_h.copy(), [N], STDtype.BF16, ctx)
    var g4 = _fill(N, 0.0017, 0.9, 0.05)
    var g4d = Tensor.from_host(g4.copy(), [N], STDtype.F32, ctx)
    var g4d2 = Tensor.from_host(g4.copy(), [N], STDtype.F32, ctx)
    # fresh states (same trajectory) for the pair
    var sa = AdafactorDeviceState(N, ctx)
    var sb = AdafactorDeviceState(N, ctx)
    adafactor_step_device_1d(p_a, g4d, sa, 1, LR, B2D, Float64(-1.0), EPS2, DD, WD, UInt64(0), ctx)
    adafactor_step_device_1d(p_b, g4d2, sb, 1, LR, B2D, Float64(-1.0), EPS2, DD, WD, UInt64(12345), ctx)
    ctx.synchronize()
    var pa = p_a.to_host(ctx)
    var pb = p_b.to_host(ctx)
    var sr_viol = 0
    var sr_moved = 0
    for i in range(N):
        var d = pa[i] - pb[i]
        if d < 0: d = -d
        if d > 0:
            sr_moved += 1
            # one bf16 ulp at |x|: 2^-8 relative (7-bit mantissa) + eps floor
            var mag = pa[i] if pa[i] >= 0 else -pa[i]
            var ulp = mag * Float32(1.0 / 128.0) + Float32(1e-7)
            if d > ulp:
                sr_viol += 1
    print("SR: moved", sr_moved, "/", N, " elements; >1ulp violations:", sr_viol)
    if sr_viol > 0:
        allok = False
    if sr_moved == 0:
        print("SR: WARNING zero elements moved (rounding never hit) — suspicious")
        allok = False

    # ── wrong-path guards fail loud ──────────────────────────────────────────
    var guards_ok = False
    try:
        adafactor_step_device(p_a, g4d, sa, 2, LR, B2D, Float64(-1.0), EPS2, DD, WD, UInt64(0), ctx)
    except:
        try:
            var s2d = AdafactorDeviceState(64, 64, ctx)
            var pf = Tensor.from_host(_fill(4096, 0.001, 0.2, 0.5), [64, 64], STDtype.BF16, ctx)
            var gf = Tensor.from_host(_fill(4096, 0.002, 0.4, 0.05), [64, 64], STDtype.F32, ctx)
            adafactor_step_device_1d(pf, gf, s2d, 1, LR, B2D, Float64(-1.0), EPS2, DD, WD, UInt64(0), ctx)
        except:
            guards_ok = True
    print("guards: 2D-step-on-1D-state and 1D-step-on-2D-state both raise:", guards_ok)
    if not guards_ok:
        allok = False

    # ── sidecar v2: MIXED-state roundtrip (factored + unfactored) ────────────
    # one factored 2D state with non-zero moments + the trajectory's 1D state.
    var s2 = AdafactorDeviceState(64, 64, ctx)
    var p2 = Tensor.from_host(_fill(4096, 0.0009, 0.3, 0.5), [64, 64], STDtype.BF16, ctx)
    var g2 = Tensor.from_host(_fill(4096, 0.0013, 0.6, 0.05), [64, 64], STDtype.F32, ctx)
    adafactor_step_device(p2, g2, s2, 1, LR, B2D, Float64(-1.0), EPS2, DD, WD, UInt64(0), ctx)
    ctx.synchronize()
    var mixed = List[AdafactorDeviceState]()
    mixed.append(s2.copy())
    mixed.append(dstate.copy())
    full_ft_sidecar_save(mixed, 7, UInt64(0xDEADBEEF12345), String(SIDE_V2), ctx)
    var ver2 = _read_sidecar_version(String(SIDE_V2))
    print("sidecar v2 smoke: written version =", ver2)
    if ver2 != 2:
        print("FAIL: mixed-state sidecar did not emit version 2")
        allok = False
    var exp_rows = [64, N]
    var exp_cols = [64, 0]
    var res = full_ft_sidecar_load(String(SIDE_V2), exp_rows, exp_cols, ctx)
    var v2ok = res.t_step == 7 and res.seed_base == UInt64(0xDEADBEEF12345)
    var lr_rv = res.states[0].row_var[].to_host(ctx)
    var lr_cv = res.states[0].col_var[].to_host(ctx)
    var lr_v1d = res.states[1].row_var[].to_host(ctx)
    var or_rv = s2.row_var[].to_host(ctx)
    var or_cv = s2.col_var[].to_host(ctx)
    if _maxabs(lr_rv, or_rv) != 0.0 or _maxabs(lr_cv, or_cv) != 0.0:
        print("FAIL: v2 roundtrip factored state not byte-exact")
        v2ok = False
    if _maxabs(lr_v1d, v_h) != 0.0:
        print("FAIL: v2 roundtrip unfactored state not byte-exact")
        v2ok = False
    if res.states[0].factored != True or res.states[1].factored != False:
        print("FAIL: v2 roundtrip factored flags wrong")
        v2ok = False
    print("sidecar v2 mixed roundtrip ok:", v2ok)
    if not v2ok:
        allok = False

    # ── sidecar v1 back-compat: all-factored save still emits v1; a v1 file
    # cannot satisfy an unfactored expectation (fail-loud) ────────────────────
    var allf = List[AdafactorDeviceState]()
    allf.append(s2.copy())
    full_ft_sidecar_save(allf, 3, UInt64(42), String(SIDE_V1), ctx)
    var ver1 = _read_sidecar_version(String(SIDE_V1))
    print("sidecar v1 smoke: all-factored written version =", ver1)
    if ver1 != 1:
        print("FAIL: all-factored sidecar did not stay version 1")
        allok = False
    var exp_r1 = [64]
    var exp_c1 = [64]
    var res1 = full_ft_sidecar_load(String(SIDE_V1), exp_r1, exp_c1, ctx)
    var v1ok = res1.t_step == 3 and res1.seed_base == UInt64(42)
    var l1_rv = res1.states[0].row_var[].to_host(ctx)
    if _maxabs(l1_rv, or_rv) != 0.0:
        print("FAIL: v1 roundtrip factored state not byte-exact")
        v1ok = False
    print("sidecar v1 roundtrip ok:", v1ok)
    if not v1ok:
        allok = False
    var v1_reject = False
    try:
        var bad_r = [64]
        var bad_c = [0]
        _ = full_ft_sidecar_load(String(SIDE_V1), bad_r, bad_c, ctx)
    except:
        v1_reject = True
    print("v1 file + unfactored expectation rejected:", v1_reject)
    if not v1_reject:
        allok = False

    if allok:
        print("VERDICT: PASS — 1D device Adafactor BIT-IDENTICAL to the host torch-oracle (RNE), v exact, SR bounded, sidecar v2/v1 contracts hold")
    else:
        print("VERDICT: FAIL")
