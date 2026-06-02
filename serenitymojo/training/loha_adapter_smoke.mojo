# training/loha_adapter_smoke.mojo — LoHa adapter gates (a)+(b)+(c).
#
# TENET 4 / parity-bitrot guard: every check ASSERTS and the file `raise`s
# (nonzero exit) on any wrong value. Demonstrated wrong-run: set env
# LOHA_BREAK_FACTOR=1 to zero one factor grad — with FLAME_ASSERT_GRAD_FLOW=1
# the grad-flow gate then ABORTS (exit != 0), proving it catches the historical
# "1 of 4 hada factors dead" bug.
#
# GATE (a) PARITY: forward delta + the 4 backward grads vs an INDEPENDENT
#   recompute of the loha.py HadaWeight math (open-coded triple loops, NOT the
#   loha_adapter helpers) PLUS a finite-difference check of all 4 factor grads.
#   Includes init parity: initial ΔW ~ 0 (w2a==0 leaf).
# GATE (b) GRAD-FLOW: run fwd+bwd on a toy layer (after one AdamW step so w2a is
#   off zero), feed the 4 factor grads into GradCoverage.measure, ASSERT
#   coverage_pct==100 and dead==0. The LOHA_BREAK_FACTOR demo proves a dead
#   factor is caught.
# GATE (c) SAVE: save a LoHa adapter, reopen, assert the 4 keys + alpha + shapes.
#
# Run (clean PASS, exit 0):
#   rm -f serenitymojo.mojopkg && pixi run mojo run -I . \
#     serenitymojo/training/loha_adapter_smoke.mojo
# Run (deliberate FAIL, exit != 0 — proves the guard works):
#   FLAME_ASSERT_GRAD_FLOW=1 LOHA_BREAK_FACTOR=1 pixi run mojo run -I . \
#     serenitymojo/training/loha_adapter_smoke.mojo

from std.collections import List
from std.math import sqrt, abs
from std.ffi import external_call
from std.memory import alloc, UnsafePointer, ArcPointer
from std.builtin.type_aliases import MutExternalOrigin
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.loha_adapter import (
    LoHaAdapter,
    new_loha_adapter,
    loha_diff_weight,
    loha_forward,
    loha_backward,
    loha_adamw,
)
from serenitymojo.training.loha_save import (
    NamedLoHa,
    save_loha_peft,
    read_loha_module,
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


# ── INDEPENDENT oracle: ΔW = (w1a@w1b) ⊙ (w2a@w2b) * scale via raw triple loops ─
def _oracle_diff(lo: LoHaAdapter) -> List[Float32]:
    var IN = lo.in_f
    var OUT = lo.out_f
    var R = lo.rank
    var out = List[Float32]()
    for i in range(IN):
        for j in range(OUT):
            var s1 = Float32(0.0)
            var s2 = Float32(0.0)
            for r in range(R):
                s1 += lo.w1a[i * R + r] * lo.w1b[r * OUT + j]
                s2 += lo.w2a[i * R + r] * lo.w2b[r * OUT + j]
            out.append(s1 * s2 * lo.scale)
    return out^


# Independent forward y = x @ ΔW.
def _oracle_fwd(x: List[Float32], diff: List[Float32], M: Int, IN: Int, OUT: Int) -> List[Float32]:
    var out = List[Float32]()
    for m in range(M):
        for j in range(OUT):
            var s = Float32(0.0)
            for i in range(IN):
                s += x[m * IN + i] * diff[i * OUT + j]
            out.append(s)
    return out^


# Scalar loss = sum(y) → d_y is all-ones (so d_y feeds every output).
def _max_abs_diff(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error("max_abs_diff: len mismatch " + String(len(a)) + " != " + String(len(b)))
    var mx = Float32(0.0)
    for i in range(len(a)):
        var d = abs(a[i] - b[i])
        if d > mx:
            mx = d
    return mx


# loss(lo) for finite-difference: sum over (x @ ΔW(lo)). Uses oracle diff so the
# FD check is independent of loha_backward's d_diff path.
def _loss_for_fd(lo: LoHaAdapter, x: List[Float32], M: Int) -> Float32:
    var diff = _oracle_diff(lo)
    var y = _oracle_fwd(x, diff, M, lo.in_f, lo.out_f)
    var s = Float32(0.0)
    for i in range(len(y)):
        s += y[i]
    return s


# ── module-level FD grad over one factor (which: 0=w1a 1=w1b 2=w2a 3=w2b) ────
# Central difference on _loss_for_fd. Lifted to module level so it captures
# nothing (Mojo 1.0.0b1 can't infer captures for nested defs — MOJO_CONVENTIONS).
def _fd_grad(lo_in: LoHaAdapter, which: Int, x: List[Float32], M: Int, h: Float32) raises -> List[Float32]:
    var out = List[Float32]()
    var n: Int
    if which == 0:
        n = len(lo_in.w1a)
    elif which == 1:
        n = len(lo_in.w1b)
    elif which == 2:
        n = len(lo_in.w2a)
    else:
        n = len(lo_in.w2b)
    for k in range(n):
        var lp = lo_in.copy()
        var lm = lo_in.copy()
        if which == 0:
            lp.w1a[k] = lp.w1a[k] + h; lm.w1a[k] = lm.w1a[k] - h
        elif which == 1:
            lp.w1b[k] = lp.w1b[k] + h; lm.w1b[k] = lm.w1b[k] - h
        elif which == 2:
            lp.w2a[k] = lp.w2a[k] + h; lm.w2a[k] = lm.w2a[k] - h
        else:
            lp.w2b[k] = lp.w2b[k] + h; lm.w2b[k] = lm.w2b[k] - h
        out.append((_loss_for_fd(lp, x, M) - _loss_for_fd(lm, x, M)) / (Float32(2.0) * h))
    return out^


def main() raises:
    var ctx = DeviceContext()
    var ok = True

    var IN = 5
    var OUT = 4
    var R = 2
    var M = 3
    var alpha = Float32(2.0)   # scale = alpha/rank = 1.0

    # ── GATE (a) init parity: ΔW ~ 0 at init (w2a==0) ────────────────────────
    var lo0 = new_loha_adapter(IN, OUT, R, alpha, 42)
    var diff0 = loha_diff_weight(lo0)
    var init_mx = Float32(0.0)
    for i in range(len(diff0)):
        if abs(diff0[i]) > init_mx:
            init_mx = abs(diff0[i])
    if init_mx != Float32(0.0):
        print("FAIL (a-init): initial ΔW not exactly 0, max|ΔW|=", init_mx); ok = False
    else:
        print("PASS (a-init): initial ΔW == 0 (w2a zero leaf)")

    # Drive w2a off zero so all factors are live for the parity / grad-flow gates.
    # Use a non-degenerate adapter: hand-set w2a to nonzero deterministic values.
    var lo = new_loha_adapter(IN, OUT, R, alpha, 7)
    for i in range(len(lo.w2a)):
        lo.w2a[i] = Float32(0.05) * Float32((i % 7) + 1)
    if lo.scale != Float32(1.0):
        print("FAIL: expected scale 1.0, got", lo.scale); ok = False

    # deterministic input x [M,IN]
    var x = List[Float32]()
    for i in range(M * IN):
        x.append(Float32(0.1) * Float32(((i * 3) % 11) - 5))

    # ── GATE (a) forward parity: loha_forward vs independent oracle ──────────
    var diff_impl = loha_diff_weight(lo)
    var diff_oracle = _oracle_diff(lo)
    var diff_mx = _max_abs_diff(diff_impl, diff_oracle)
    if diff_mx > Float32(1.0e-5):
        print("FAIL (a-diff): ΔW vs oracle max|Δ|=", diff_mx); ok = False
    else:
        print("PASS (a-diff): ΔW matches oracle, max|Δ|=", diff_mx)

    var y_impl = loha_forward(x, lo, M)
    var y_oracle = _oracle_fwd(x, diff_oracle, M, IN, OUT)
    var y_mx = _max_abs_diff(y_impl, y_oracle)
    if y_mx > Float32(1.0e-5):
        print("FAIL (a-fwd): y vs oracle max|Δ|=", y_mx); ok = False
    else:
        print("PASS (a-fwd): forward y matches oracle, max|Δ|=", y_mx)

    # ── GATE (a) backward: analytic 4 grads vs finite-difference ─────────────
    # d_y = ones [M,OUT] (loss = sum(y)). loha_backward gives the 4 factor grads.
    var d_y = List[Float32]()
    for _ in range(M * OUT):
        d_y.append(Float32(1.0))
    var g = loha_backward(d_y, x, lo, M)

    # FD oracle: perturb each factor element, recompute _loss_for_fd.
    var h = Float32(1.0e-3)
    var fd1a = _fd_grad(lo, 0, x, M, h)
    var fd1b = _fd_grad(lo, 1, x, M, h)
    var fd2a = _fd_grad(lo, 2, x, M, h)
    var fd2b = _fd_grad(lo, 3, x, M, h)

    var tol_fd = Float32(2.0e-2)   # central-difference tolerance at h=1e-3
    var e1a = _max_abs_diff(g.d_w1a, fd1a)
    var e1b = _max_abs_diff(g.d_w1b, fd1b)
    var e2a = _max_abs_diff(g.d_w2a, fd2a)
    var e2b = _max_abs_diff(g.d_w2b, fd2b)
    if e1a > tol_fd:
        print("FAIL (a-bwd w1a): analytic vs FD max|Δ|=", e1a); ok = False
    else:
        print("PASS (a-bwd w1a): max|Δ| vs FD=", e1a)
    if e1b > tol_fd:
        print("FAIL (a-bwd w1b): analytic vs FD max|Δ|=", e1b); ok = False
    else:
        print("PASS (a-bwd w1b): max|Δ| vs FD=", e1b)
    if e2a > tol_fd:
        print("FAIL (a-bwd w2a): analytic vs FD max|Δ|=", e2a); ok = False
    else:
        print("PASS (a-bwd w2a): max|Δ| vs FD=", e2a)
    if e2b > tol_fd:
        print("FAIL (a-bwd w2b): analytic vs FD max|Δ|=", e2b); ok = False
    else:
        print("PASS (a-bwd w2b): max|Δ| vs FD=", e2b)

    # ── GATE (b) GRAD-FLOW: all 4 factor grads must be nonzero (the critical one)─
    # Take one AdamW step first (w2a was already off-zero; this exercises the
    # optimizer path too), then a fresh fwd+bwd, then GradCoverage over 4 grads.
    var lo_b = lo.copy()
    loha_adamw(lo_b, g, 1, Float32(1.0e-3))
    var g2 = loha_backward(d_y, x, lo_b, M)

    var names = List[String]()
    names.append(String("toy.hada_w1_a"))
    names.append(String("toy.hada_w1_b"))
    names.append(String("toy.hada_w2_a"))
    names.append(String("toy.hada_w2_b"))

    # DELIBERATE-WRONG demo: zero ONE factor grad (the historical bug) if armed.
    var break_factor = _env_is_set(String("LOHA_BREAK_FACTOR"))
    var d2a_maybe = g2.d_w2a.copy()
    if break_factor:
        for i in range(len(d2a_maybe)):
            d2a_maybe[i] = Float32(0.0)
        print("INFO: LOHA_BREAK_FACTOR set — zeroing hada_w2_a grad to prove the gate catches it")

    var grads = List[TArc]()
    grads.append(TArc(Tensor.from_host(g2.d_w1a.copy(), [IN, R], STDtype.F32, ctx)))
    grads.append(TArc(Tensor.from_host(g2.d_w1b.copy(), [R, OUT], STDtype.F32, ctx)))
    grads.append(TArc(Tensor.from_host(d2a_maybe^, [IN, R], STDtype.F32, ctx)))
    grads.append(TArc(Tensor.from_host(g2.d_w2b.copy(), [R, OUT], STDtype.F32, ctx)))

    var rep = measure(names, grads, ctx)
    print("[grad-flow] total=", rep.total, " nonzero=", rep.nonzero,
          " dead=", rep.dead, " coverage=", rep.coverage_pct(), "%")

    var armed = _env_is_set(String("FLAME_ASSERT_GRAD_FLOW"))
    if break_factor:
        # The broken path: gate MUST see dead>0. If armed, abort (nonzero exit).
        if rep.dead == 0:
            print("FAIL (b-demo): broken factor NOT detected (dead==0)"); ok = False
        else:
            print("PASS (b-demo): dead factor detected, dead=", rep.dead)
        if armed:
            raise Error(
                String("[grad-flow] LoHa FAILED: ") + String(rep.dead)
                + " of " + String(rep.total) + " factors DEAD (zeroed hada_w2_a) — "
                + "this is the historical LoHa bug; gate aborts (exit != 0)"
            )
    else:
        # The healthy path: ALL FOUR factors nonzero.
        if rep.dead != 0:
            print("FAIL (b): a LoHa factor grad is DEAD (dead=", rep.dead, ")"); ok = False
        else:
            print("PASS (b): all 4 LoHa factor grads nonzero, dead=0")
        if rep.coverage_pct() != Float64(100.0):
            print("FAIL (b): coverage_pct != 100 (got", rep.coverage_pct(), ")"); ok = False
        else:
            print("PASS (b): coverage_pct == 100 (all 4 factors live)")

    # ── GATE (c) SAVE: save → reopen → assert 4 keys + alpha + shapes ────────
    var named = List[NamedLoHa]()
    named.append(NamedLoHa(String("double_blocks.0.img_attn.to_q"), lo.copy()))
    var path = String("/tmp/loha_smoke.safetensors")
    var n_written = save_loha_peft(named, path, ctx)
    if n_written != 1:
        print("FAIL (c): save returned", n_written, "adapters, expected 1"); ok = False
    else:
        print("PASS (c): saved 1 LoHa adapter to", path)

    var rb = read_loha_module(String("double_blocks.0.img_attn.to_q"), path, ctx)
    if rb.in_f != IN or rb.out_f != OUT or rb.rank != R:
        print("FAIL (c): shape mismatch in/out/rank got",
              rb.in_f, rb.out_f, rb.rank, "expected", IN, OUT, R); ok = False
    else:
        print("PASS (c): shapes round-trip in=", rb.in_f, " out=", rb.out_f, " rank=", rb.rank)
    if abs(rb.alpha - alpha) > Float32(1.0e-6):
        print("FAIL (c): alpha mismatch got", rb.alpha, "expected", alpha); ok = False
    else:
        print("PASS (c): alpha round-trip =", rb.alpha)
    var w1a_mx = _max_abs_diff(rb.w1a, lo.w1a)
    var w2a_mx = _max_abs_diff(rb.w2a, lo.w2a)
    if w1a_mx > Float32(1.0e-6) or w2a_mx > Float32(1.0e-6):
        print("FAIL (c): factor values not byte-exact, w1a Δ=", w1a_mx, " w2a Δ=", w2a_mx); ok = False
    else:
        print("PASS (c): factor values round-trip byte-exact")

    if not ok:
        raise Error("loha_adapter_smoke FAILED")
    print("loha_adapter_smoke ALL GATES PASS")
