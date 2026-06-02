# training/lokr_adapter_smoke.mojo — LoKr adapter gates (a)+(b)+(c)+(d).
#
# TENET 4 / parity-bitrot guard: every check ASSERTS and the file `raise`s
# (nonzero exit) on any wrong value. Demonstrated wrong-run: set env
# LOKR_BREAK_FACTOR=1 to zero ONE live factor grad — with FLAME_ASSERT_GRAD_FLOW=1
# the grad-flow gate then ABORTS (exit != 0).
#
# Covers BOTH W2 storage modes (the lokr.rs factorize rule rank<max(out_k,in_n)/2):
#   - FACTORED W2  (rank=1, split (out_k,in_n)=(4,4)): live factors {w1, w2a, w2b}.
#   - FULL W2      (rank=3, same split):               live factors {w1, w2}.
#
# GATE (a) PARITY: the Kronecker delta ΔW = kron(W1,W2)*scale + the forward, vs
#   an INDEPENDENT open-coded recompute (kron index map kron[(l*OK+k),(a*INn+b)]
#   = W1[l,a]*W2[k,b], NOT the lokr_adapter helpers) PLUS a finite-difference of
#   ALL LIVE factor grads. Includes init parity: initial ΔW ~ 0 (zero leg).
# GATE (b) GRAD-FLOW: after one AdamW step (w2b/w2 off zero), fresh fwd+bwd, feed
#   the live factor grads into GradCoverage.measure → coverage_pct==100, dead==0.
#   The LOKR_BREAK_FACTOR demo proves a dead factor is caught.
# GATE (c) SAVE: save → reopen → assert lokr_w1 + (lokr_w2 | lokr_w2_a/b) + alpha
#   keys + shapes + byte-exact values, for BOTH the factored and full configs.
# GATE (d) default-off + trainer AOT-build verified out-of-band (trainer fails
#   loud on adapter_algo==4); see the builder report.
#
# Run (clean PASS, exit 0):
#   rm -f serenitymojo.mojopkg && pixi run mojo run -I . \
#     serenitymojo/training/lokr_adapter_smoke.mojo
# Run (deliberate FAIL, exit != 0):
#   FLAME_ASSERT_GRAD_FLOW=1 LOKR_BREAK_FACTOR=1 pixi run mojo run -I . \
#     serenitymojo/training/lokr_adapter_smoke.mojo

from std.collections import List
from std.math import abs
from std.ffi import external_call
from std.memory import alloc, UnsafePointer, ArcPointer
from std.builtin.type_aliases import MutExternalOrigin
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.lokr_adapter import (
    LoKrAdapter,
    new_lokr_adapter,
    lokr_factorization,
    lokr_resolve_w2,
    lokr_delta_weight,
    lokr_forward,
    lokr_backward,
    lokr_adamw,
)
from serenitymojo.training.lokr_save import (
    NamedLoKr,
    save_lokr_peft,
    read_lokr_module,
)
from serenitymojo.training.grad_coverage import GradCoverage, measure


comptime TArc = ArcPointer[Tensor]
comptime _EnvPtr = UnsafePointer[UInt8, MutExternalOrigin]


def _env_is_set(name: String) -> Bool:
    var n = name.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = name.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var cname = _EnvPtr(unsafe_from_address=Int(buf))
    var ret = external_call["getenv", _EnvPtr](cname)
    buf.free()
    if Int(ret) == 0:
        return False
    return ret[0] == UInt8(49) and ret[1] == UInt8(0)


def _max_abs_diff(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error("max_abs_diff: len mismatch " + String(len(a)) + " != " + String(len(b)))
    var mx = Float32(0.0)
    for i in range(len(a)):
        var d = abs(a[i] - b[i])
        if d > mx:
            mx = d
    return mx


# Relative grad-parity error: max_i |a[i]-b[i]| / (|b[i]| + floor).
# Absolute max|Δ| is VACUOUS for intrinsically-tiny grads — a multiplicative
# analytic-backward bug (e.g. a stray 1.5×) produces an absolute delta below a
# fixed abs tolerance when the grad itself is small, so the gate would silently
# pass (demonstrated: factored-config w1 grad missed a 1.5× bug at abs-tol 2e-2).
# A relative tolerance catches multiplicative bugs regardless of grad scale.
# `b` is the reference (finite-difference) grad. `floor` keeps near-zero
# components from dominating with FD round-off; entries where the reference is
# below `floor` are skipped (their relative error is meaningless and dominated
# by the central-difference truncation error there).
def _max_rel_diff(a: List[Float32], b: List[Float32], floor: Float32) raises -> Float32:
    if len(a) != len(b):
        raise Error("max_rel_diff: len mismatch " + String(len(a)) + " != " + String(len(b)))
    var mx = Float32(0.0)
    for i in range(len(a)):
        if abs(b[i]) < floor:
            continue
        var r = abs(a[i] - b[i]) / abs(b[i])
        if r > mx:
            mx = r
    return mx


# ── INDEPENDENT oracle: ΔW = kron(W1, W2d)*scale via raw kron index map ───────
# W2d is the resolved dense W2:[out_k,in_n] (caller passes lokr_resolve_w2 OR an
# independently-materialized w2a@w2b). kron[(l*OK+k),(a*INn+b)] = W1[l,a]*W2[k,b].
def _oracle_delta(w1: List[Float32], w2d: List[Float32],
                  OL: Int, OK: Int, IM: Int, INn: Int, scale: Float32) -> List[Float32]:
    var OUT = OL * OK
    var IN = IM * INn
    var out = List[Float32]()
    for _ in range(OUT * IN):
        out.append(Float32(0.0))
    for l in range(OL):
        for a in range(IM):
            var w1la = w1[l * IM + a]
            for k in range(OK):
                var row = l * OK + k
                for b in range(INn):
                    var col = a * INn + b
                    out[row * IN + col] = w1la * w2d[k * INn + b] * scale
    return out^


# independent dense W2 from w2a@w2b (factored) — open-coded matmul.
def _oracle_w2_factored(w2a: List[Float32], w2b: List[Float32], OK: Int, R: Int, INn: Int) -> List[Float32]:
    var out = List[Float32]()
    for k in range(OK):
        for b in range(INn):
            var s = Float32(0.0)
            for r in range(R):
                s += w2a[k * R + r] * w2b[r * INn + b]
            out.append(s)
    return out^


# independent dense W1 from w1a@w1b (factored) — open-coded matmul. [out_l,in_m].
def _oracle_w1_factored(w1a: List[Float32], w1b: List[Float32], OL: Int, R: Int, IM: Int) -> List[Float32]:
    var out = List[Float32]()
    for l in range(OL):
        for a in range(IM):
            var s = Float32(0.0)
            for r in range(R):
                s += w1a[l * R + r] * w1b[r * IM + a]
            out.append(s)
    return out^


# Independent forward y = x @ ΔWᵀ.
def _oracle_fwd(x: List[Float32], dw: List[Float32], M: Int, IN: Int, OUT: Int) -> List[Float32]:
    var out = List[Float32]()
    for mm in range(M):
        for o in range(OUT):
            var s = Float32(0.0)
            for i in range(IN):
                s += x[mm * IN + i] * dw[o * IN + i]
            out.append(s)
    return out^


# loss(adapter factors) for FD: sum(x @ ΔWᵀ). Recomputes ΔW from the supplied
# factors (independent of lokr_backward). `w1_factored`/`w2_factored` select the
# W1/W2 storage. Six possible factors carried; only the live ones are used.
def _loss_for_fd(w1: List[Float32], w1a: List[Float32], w1b: List[Float32], w1_factored: Bool,
                 w2: List[Float32], w2a: List[Float32], w2b: List[Float32], w2_factored: Bool,
                 x: List[Float32],
                 OL: Int, OK: Int, IM: Int, INn: Int, R: Int, M: Int, scale: Float32) -> Float32:
    var w1d: List[Float32]
    if w1_factored:
        w1d = _oracle_w1_factored(w1a, w1b, OL, R, IM)
    else:
        w1d = w1.copy()
    var w2d: List[Float32]
    if w2_factored:
        w2d = _oracle_w2_factored(w2a, w2b, OK, R, INn)
    else:
        w2d = w2.copy()
    var dw = _oracle_delta(w1d, w2d, OL, OK, IM, INn, scale)
    var IN = IM * INn
    var OUT = OL * OK
    var y = _oracle_fwd(x, dw, M, IN, OUT)
    var s = Float32(0.0)
    for i in range(len(y)):
        s += y[i]
    return s


# FD grad over one factor (which: 0=w1 1=w2 2=w2a 3=w2b 4=w1a 5=w1b), central diff.
def _fd_grad(which: Int, w1: List[Float32], w1a: List[Float32], w1b: List[Float32], w1_factored: Bool,
             w2: List[Float32], w2a: List[Float32], w2b: List[Float32], w2_factored: Bool,
             x: List[Float32],
             OL: Int, OK: Int, IM: Int, INn: Int, R: Int, M: Int, scale: Float32, h: Float32) raises -> List[Float32]:
    var out = List[Float32]()
    var n: Int
    if which == 0:
        n = len(w1)
    elif which == 1:
        n = len(w2)
    elif which == 2:
        n = len(w2a)
    elif which == 3:
        n = len(w2b)
    elif which == 4:
        n = len(w1a)
    else:
        n = len(w1b)
    for k in range(n):
        var w1p = w1.copy(); var w1m = w1.copy()
        var w1ap = w1a.copy(); var w1am = w1a.copy()
        var w1bp = w1b.copy(); var w1bm = w1b.copy()
        var w2p = w2.copy(); var w2m = w2.copy()
        var w2ap = w2a.copy(); var w2am = w2a.copy()
        var w2bp = w2b.copy(); var w2bm = w2b.copy()
        if which == 0:
            w1p[k] = w1p[k] + h; w1m[k] = w1m[k] - h
        elif which == 1:
            w2p[k] = w2p[k] + h; w2m[k] = w2m[k] - h
        elif which == 2:
            w2ap[k] = w2ap[k] + h; w2am[k] = w2am[k] - h
        elif which == 3:
            w2bp[k] = w2bp[k] + h; w2bm[k] = w2bm[k] - h
        elif which == 4:
            w1ap[k] = w1ap[k] + h; w1am[k] = w1am[k] - h
        else:
            w1bp[k] = w1bp[k] + h; w1bm[k] = w1bm[k] - h
        var lp = _loss_for_fd(w1p, w1ap, w1bp, w1_factored, w2p, w2ap, w2bp, w2_factored, x, OL, OK, IM, INn, R, M, scale)
        var lm = _loss_for_fd(w1m, w1am, w1bm, w1_factored, w2m, w2am, w2bm, w2_factored, x, OL, OK, IM, INn, R, M, scale)
        out.append((lp - lm) / (Float32(2.0) * h))
    return out^


# Run all of gates (a)+(b)+(c) for one config; returns True on PASS.
# `break_factor`/`armed` thread the deliberate-wrong demo through grad-flow.
def _run_config(label: String, in_f: Int, out_f: Int, rank: Int, factor: Int,
                expect_factored: Bool, decompose_both: Bool, expect_w1_factored: Bool,
                break_factor: Bool, break_bwd: Bool, armed: Bool, ctx: DeviceContext) raises -> Bool:
    var ok = True
    print("=== config", label, "(in=", in_f, " out=", out_f, " rank=", rank, " factor=", factor,
          " decompose_both=", decompose_both, ") ===")

    var alpha = Float32(Float32(rank))   # scale = alpha/rank = 1.0
    var lo = new_lokr_adapter(in_f, out_f, rank, alpha, factor, 7, decompose_both)
    if lo.w2_factored != expect_factored:
        print("FAIL: expected w2_factored=", expect_factored, " got", lo.w2_factored); ok = False
    if lo.w1_factored != expect_w1_factored:
        print("FAIL: expected w1_factored=", expect_w1_factored, " got", lo.w1_factored); ok = False
    if lo.scale != Float32(1.0):
        print("FAIL: expected scale 1.0, got", lo.scale); ok = False

    var OL = lo.out_l; var OK = lo.out_k; var IM = lo.in_m; var INn = lo.in_n; var R = lo.rank
    var IN = lo.in_f; var OUT = lo.out_f; var M = 3

    # ── (a) init parity: ΔW ~ 0 at init (zero leg) ──
    var dw0 = lokr_delta_weight(lo)
    var init_mx = Float32(0.0)
    for i in range(len(dw0)):
        if abs(dw0[i]) > init_mx:
            init_mx = abs(dw0[i])
    if init_mx != Float32(0.0):
        print("FAIL (a-init): initial ΔW not exactly 0, max|ΔW|=", init_mx); ok = False
    else:
        print("PASS (a-init): initial ΔW == 0 (zero leg)")

    # Drive the zero leg off zero so all live factors are exercised. The
    # W1-factored config uses a LARGER drive + factor amplitude so the (degree-4
    # multilinear) factor grads sit well above the FD rel_floor — otherwise the
    # w1a/w1b grads are intrinsically ~1e-8 and the relative FD check is VACUOUS
    # (every reference component below floor → skipped). Amplifying lifts them
    # into the meaningful regime so a wrong-but-nonzero backward is caught.
    var drive = Float32(1.0) if lo.w1_factored else Float32(0.04)
    if lo.w2_factored:
        for i in range(len(lo.w2b)):
            lo.w2b[i] = drive * Float32((i % 5) + 1)
    else:
        for i in range(len(lo.w2)):
            lo.w2[i] = drive * Float32((i % 5) + 1)
    # Amplify the (kaiming-init) W1 factor legs in the factored config too.
    if lo.w1_factored:
        for i in range(len(lo.w1a)):
            lo.w1a[i] = lo.w1a[i] * Float32(10.0)
        for i in range(len(lo.w1b)):
            lo.w1b[i] = lo.w1b[i] * Float32(10.0)

    # deterministic input x [M,IN]
    var xscale = Float32(1.0) if lo.w1_factored else Float32(0.1)
    var x = List[Float32]()
    for i in range(M * IN):
        x.append(xscale * Float32(((i * 3) % 11) - 5))

    # ── (a) delta + forward parity vs independent oracle ──
    var w2d = lokr_resolve_w2(lo)
    var w2d_oracle: List[Float32]
    if lo.w2_factored:
        w2d_oracle = _oracle_w2_factored(lo.w2a, lo.w2b, OK, R, INn)
    else:
        w2d_oracle = lo.w2.copy()
    var w1d_oracle: List[Float32]
    if lo.w1_factored:
        w1d_oracle = _oracle_w1_factored(lo.w1a, lo.w1b, OL, R, IM)
    else:
        w1d_oracle = lo.w1.copy()
    var dw_impl = lokr_delta_weight(lo)
    var dw_oracle = _oracle_delta(w1d_oracle, w2d_oracle, OL, OK, IM, INn, lo.scale)
    var dw_mx = _max_abs_diff(dw_impl, dw_oracle)
    if dw_mx > Float32(1.0e-5):
        print("FAIL (a-delta): ΔW vs oracle max|Δ|=", dw_mx); ok = False
    else:
        print("PASS (a-delta): ΔW matches Kronecker oracle, max|Δ|=", dw_mx)

    var y_impl = lokr_forward(x, lo, M)
    var y_oracle = _oracle_fwd(x, dw_oracle, M, IN, OUT)
    var y_mx = _max_abs_diff(y_impl, y_oracle)
    if y_mx > Float32(1.0e-5):
        print("FAIL (a-fwd): y vs oracle max|Δ|=", y_mx); ok = False
    else:
        print("PASS (a-fwd): forward y matches oracle, max|Δ|=", y_mx)

    # ── (a) backward: analytic live-factor grads vs finite-difference ──
    var d_y = List[Float32]()
    for _ in range(M * OUT):
        d_y.append(Float32(1.0))
    var g = lokr_backward(d_y, x, lo, M)
    var h = Float32(1.0e-3)
    var tol_fd = Float32(2.0e-2)
    # Relative tolerance + a small floor so intrinsically-tiny grads (e.g. the
    # factored-config w1 grad) are not vacuous: a multiplicative analytic-backward
    # bug is caught by the relative check regardless of grad magnitude. Floor is
    # ~2 orders above the central-difference truncation noise (h²~1e-6 with the
    # 1e-3 step) so near-zero FD components don't trip a false positive.
    var tol_rel = Float32(2.0e-2)
    var rel_floor = Float32(1.0e-4)

    if lo.w1_factored:
        # PARITY-BITROT DEMO: corrupt analytic d_w1a (×1.5) — the FD parity gate
        # must catch this wrong-but-nonzero backward (grad-flow alone wouldn't).
        if break_bwd:
            for _bz in range(len(g.d_w1a)):
                g.d_w1a[_bz] = g.d_w1a[_bz] * Float32(1.5)
            print("INFO: LOKR_BREAK_BWD set — scaling analytic d_w1a by 1.5 to prove the FD gate catches it")
        var fd_w1a = _fd_grad(4, lo.w1, lo.w1a, lo.w1b, True, lo.w2, lo.w2a, lo.w2b, lo.w2_factored, x, OL, OK, IM, INn, R, M, lo.scale, h)
        var fd_w1b = _fd_grad(5, lo.w1, lo.w1a, lo.w1b, True, lo.w2, lo.w2a, lo.w2b, lo.w2_factored, x, OL, OK, IM, INn, R, M, lo.scale, h)
        var e_w1a = _max_abs_diff(g.d_w1a, fd_w1a)
        var er_w1a = _max_rel_diff(g.d_w1a, fd_w1a, rel_floor)
        var e_w1b = _max_abs_diff(g.d_w1b, fd_w1b)
        var er_w1b = _max_rel_diff(g.d_w1b, fd_w1b, rel_floor)
        if e_w1a > tol_fd or er_w1a > tol_rel:
            print("FAIL (a-bwd w1a): analytic vs FD abs|Δ|=", e_w1a, " rel=", er_w1a); ok = False
        else:
            print("PASS (a-bwd w1a): max|Δ| vs FD=", e_w1a, " rel=", er_w1a)
        if e_w1b > tol_fd or er_w1b > tol_rel:
            print("FAIL (a-bwd w1b): analytic vs FD abs|Δ|=", e_w1b, " rel=", er_w1b); ok = False
        else:
            print("PASS (a-bwd w1b): max|Δ| vs FD=", e_w1b, " rel=", er_w1b)
    else:
        var fd_w1 = _fd_grad(0, lo.w1, lo.w1a, lo.w1b, False, lo.w2, lo.w2a, lo.w2b, lo.w2_factored, x, OL, OK, IM, INn, R, M, lo.scale, h)
        var e_w1 = _max_abs_diff(g.d_w1, fd_w1)
        var er_w1 = _max_rel_diff(g.d_w1, fd_w1, rel_floor)
        if e_w1 > tol_fd or er_w1 > tol_rel:
            print("FAIL (a-bwd w1): analytic vs FD abs|Δ|=", e_w1, " rel=", er_w1); ok = False
        else:
            print("PASS (a-bwd w1): max|Δ| vs FD=", e_w1, " rel=", er_w1)

    if lo.w2_factored:
        var fd_w2a = _fd_grad(2, lo.w1, lo.w1a, lo.w1b, lo.w1_factored, lo.w2, lo.w2a, lo.w2b, True, x, OL, OK, IM, INn, R, M, lo.scale, h)
        var fd_w2b = _fd_grad(3, lo.w1, lo.w1a, lo.w1b, lo.w1_factored, lo.w2, lo.w2a, lo.w2b, True, x, OL, OK, IM, INn, R, M, lo.scale, h)
        var e_w2a = _max_abs_diff(g.d_w2a, fd_w2a)
        var er_w2a = _max_rel_diff(g.d_w2a, fd_w2a, rel_floor)
        var e_w2b = _max_abs_diff(g.d_w2b, fd_w2b)
        var er_w2b = _max_rel_diff(g.d_w2b, fd_w2b, rel_floor)
        if e_w2a > tol_fd or er_w2a > tol_rel:
            print("FAIL (a-bwd w2a): analytic vs FD abs|Δ|=", e_w2a, " rel=", er_w2a); ok = False
        else:
            print("PASS (a-bwd w2a): max|Δ| vs FD=", e_w2a, " rel=", er_w2a)
        if e_w2b > tol_fd or er_w2b > tol_rel:
            print("FAIL (a-bwd w2b): analytic vs FD abs|Δ|=", e_w2b, " rel=", er_w2b); ok = False
        else:
            print("PASS (a-bwd w2b): max|Δ| vs FD=", e_w2b, " rel=", er_w2b)
    else:
        var fd_w2 = _fd_grad(1, lo.w1, lo.w1a, lo.w1b, lo.w1_factored, lo.w2, lo.w2a, lo.w2b, False, x, OL, OK, IM, INn, R, M, lo.scale, h)
        var e_w2 = _max_abs_diff(g.d_w2, fd_w2)
        var er_w2 = _max_rel_diff(g.d_w2, fd_w2, rel_floor)
        if e_w2 > tol_fd or er_w2 > tol_rel:
            print("FAIL (a-bwd w2): analytic vs FD abs|Δ|=", e_w2, " rel=", er_w2); ok = False
        else:
            print("PASS (a-bwd w2): max|Δ| vs FD=", e_w2, " rel=", er_w2)

    # ── (b) GRAD-FLOW over all LIVE factors ──
    var lo_b = lo.copy()
    lokr_adamw(lo_b, g, 1, Float32(1.0e-3))
    var g2 = lokr_backward(d_y, x, lo_b, M)

    var names = List[String]()
    var grads = List[TArc]()
    if lo.w1_factored:
        names.append(String("toy.lokr_w1_a"))
        grads.append(TArc(Tensor.from_host(g2.d_w1a.copy(), [OL, R], STDtype.F32, ctx)))
        names.append(String("toy.lokr_w1_b"))
        grads.append(TArc(Tensor.from_host(g2.d_w1b.copy(), [R, IM], STDtype.F32, ctx)))
    else:
        names.append(String("toy.lokr_w1"))
        grads.append(TArc(Tensor.from_host(g2.d_w1.copy(), [OL, IM], STDtype.F32, ctx)))
    if lo.w2_factored:
        names.append(String("toy.lokr_w2_a"))
        grads.append(TArc(Tensor.from_host(g2.d_w2a.copy(), [OK, R], STDtype.F32, ctx)))
        # DELIBERATE-WRONG: zero w2b grad if armed.
        var w2b_maybe = g2.d_w2b.copy()
        if break_factor:
            for i in range(len(w2b_maybe)):
                w2b_maybe[i] = Float32(0.0)
            print("INFO: LOKR_BREAK_FACTOR set — zeroing lokr_w2_b grad to prove the gate catches it")
        names.append(String("toy.lokr_w2_b"))
        grads.append(TArc(Tensor.from_host(w2b_maybe^, [R, INn], STDtype.F32, ctx)))
    else:
        # DELIBERATE-WRONG: zero w2 grad if armed.
        var w2_maybe = g2.d_w2.copy()
        if break_factor:
            for i in range(len(w2_maybe)):
                w2_maybe[i] = Float32(0.0)
            print("INFO: LOKR_BREAK_FACTOR set — zeroing lokr_w2 grad to prove the gate catches it")
        names.append(String("toy.lokr_w2"))
        grads.append(TArc(Tensor.from_host(w2_maybe^, [OK, INn], STDtype.F32, ctx)))

    var rep = measure(names, grads, ctx)
    print("[grad-flow] total=", rep.total, " nonzero=", rep.nonzero,
          " dead=", rep.dead, " coverage=", rep.coverage_pct(), "%")

    if break_factor:
        if rep.dead == 0:
            print("FAIL (b-demo): broken factor NOT detected (dead==0)"); ok = False
        else:
            print("PASS (b-demo): dead factor detected, dead=", rep.dead)
        if armed:
            raise Error(
                String("[grad-flow] LoKr FAILED: ") + String(rep.dead)
                + " of " + String(rep.total) + " live factors DEAD — gate aborts (exit != 0)"
            )
    else:
        if rep.dead != 0:
            print("FAIL (b): a LoKr factor grad is DEAD (dead=", rep.dead, ")"); ok = False
        else:
            print("PASS (b): all live LoKr factor grads nonzero, dead=0")
        if rep.coverage_pct() != Float64(100.0):
            print("FAIL (b): coverage_pct != 100 (got", rep.coverage_pct(), ")"); ok = False
        else:
            print("PASS (b): coverage_pct == 100 (all live factors)")

    # ── (c) SAVE round-trip ──
    var named = List[NamedLoKr]()
    named.append(NamedLoKr(String("double_blocks.0.img_attn.to_q"), lo.copy()))
    var path = String("/tmp/lokr_smoke_") + label + String(".safetensors")
    var n_written = save_lokr_peft(named, path, ctx)
    if n_written != 1:
        print("FAIL (c): save returned", n_written, "adapters, expected 1"); ok = False
    else:
        print("PASS (c): saved 1 LoKr adapter to", path)

    var rb = read_lokr_module(String("double_blocks.0.img_attn.to_q"), path, ctx)
    if rb.w2_factored != lo.w2_factored:
        print("FAIL (c): w2_factored mismatch got", rb.w2_factored, "expected", lo.w2_factored); ok = False
    if rb.w1_factored != lo.w1_factored:
        print("FAIL (c): w1_factored mismatch got", rb.w1_factored, "expected", lo.w1_factored); ok = False
    if rb.out_l != OL or rb.in_m != IM or rb.out_k != OK or rb.in_n != INn:
        print("FAIL (c): shape mismatch got", rb.out_l, rb.out_k, rb.in_m, rb.in_n,
              "expected", OL, OK, IM, INn); ok = False
    else:
        print("PASS (c): shapes round-trip out_l=", rb.out_l, " out_k=", rb.out_k, " in_m=", rb.in_m, " in_n=", rb.in_n)
    if abs(rb.alpha - alpha) > Float32(1.0e-6):
        print("FAIL (c): alpha mismatch got", rb.alpha, "expected", alpha); ok = False
    else:
        print("PASS (c): alpha round-trip =", rb.alpha)
    # W1 byte-exact (factored: w1a/w1b ; full: w1).
    if lo.w1_factored:
        var a1_mx = _max_abs_diff(rb.w1a, lo.w1a)
        var b1_mx = _max_abs_diff(rb.w1b, lo.w1b)
        if a1_mx > Float32(1.0e-6) or b1_mx > Float32(1.0e-6):
            print("FAIL (c): w1a/w1b not byte-exact, w1a Δ=", a1_mx, " w1b Δ=", b1_mx); ok = False
        else:
            print("PASS (c): lokr_w1_a/w1_b values round-trip byte-exact")
    else:
        var w1_mx = _max_abs_diff(rb.w1, lo.w1)
        if w1_mx > Float32(1.0e-6):
            print("FAIL (c): w1 not byte-exact, Δ=", w1_mx); ok = False
        else:
            print("PASS (c): lokr_w1 values round-trip byte-exact")
    if lo.w2_factored:
        var a_mx = _max_abs_diff(rb.w2a, lo.w2a)
        var b_mx = _max_abs_diff(rb.w2b, lo.w2b)
        if a_mx > Float32(1.0e-6) or b_mx > Float32(1.0e-6):
            print("FAIL (c): w2a/w2b not byte-exact, w2a Δ=", a_mx, " w2b Δ=", b_mx); ok = False
        else:
            print("PASS (c): lokr_w1/w2_a/w2_b values round-trip byte-exact")
    else:
        var w2_mx = _max_abs_diff(rb.w2, lo.w2)
        if w2_mx > Float32(1.0e-6):
            print("FAIL (c): w2 not byte-exact, Δ=", w2_mx); ok = False
        else:
            print("PASS (c): lokr_w1/w2 values round-trip byte-exact")

    return ok


def main() raises:
    var ctx = DeviceContext()
    var armed = _env_is_set(String("FLAME_ASSERT_GRAD_FLOW"))
    var break_factor = _env_is_set(String("LOKR_BREAK_FACTOR"))
    var break_bwd = _env_is_set(String("LOKR_BREAK_BWD"))

    # FACTORED W2, W1 FULL: in=out=8, factor=2 → (out_l,out_k)=(2,4),(in_m,in_n)=(2,4);
    #   max(out_k,in_n)=4, rank=1 < 2 → W2 factored. decompose_both=false → W1 full.
    #   Live: {w1, w2a, w2b}. The LOKR_BREAK_FACTOR demo arms here (zeroes w2b).
    var ok1 = _run_config(String("factored"), 8, 8, 1, 2, True, False, False, break_factor, False, armed, ctx)

    # FULL W2, W1 FULL: same split but rank=3 → 3 < 2 false → W2 full. Live: {w1, w2}.
    var ok2 = _run_config(String("full"), 8, 8, 3, 2, False, False, False, False, False, armed, ctx)

    # W1-FACTORED + W2-FACTORED (decompose_both=true): in=out=64, factor=8 →
    #   factorization(64,8)=(8,8) so out_l=out_k=in_m=in_n=8; max(.,.)=8, rank=1<4
    #   → BOTH W1 and W2 factored. FOUR live factors {w1a, w1b, w2a, w2b}.
    #   (NO regression: ok1/ok2 still exercise the W1-full path.)
    var ok3 = _run_config(String("w1factored"), 64, 64, 1, 8, True, True, True, False, break_bwd, armed, ctx)

    if not (ok1 and ok2 and ok3):
        raise Error("lokr_adapter_smoke FAILED")
    print("lokr_adapter_smoke ALL GATES PASS (W1-full factored + W1-full full + W1-factored)")
