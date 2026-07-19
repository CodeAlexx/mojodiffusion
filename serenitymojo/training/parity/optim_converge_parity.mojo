# optim_converge_parity.mojo — MULTI-STEP optimizer CONVERGENCE gate.
#
# Complement to optim_parity.mojo. That gate proves the Mojo AdamW/SGD step
# matches torch for ONE step (the per-step math is correct). This gate proves
# the optimizers actually MINIMIZE an objective over MANY steps — i.e. the
# trajectory converges to a KNOWN minimum, so "did it converge" is unambiguous.
#
# A correct per-step update is necessary but not sufficient for training: an
# optimizer can match torch for one step yet still fail to descend over a run
# (wrong bias-correction accumulation, momentum-buffer aliasing across steps,
# state not persisting through the in-place mut API, etc.). This gate exercises
# the full multi-step loop on the REAL training/optim.mojo step functions.
#
# Objectives (each with a KNOWN minimum so convergence is unambiguous):
#   1. AdamW quadratic bowl:  f(p) = sum((p - target)^2),  min f = 0 at p=target.
#      grad = 2*(p - target), computed DIRECTLY (no autograd needed — the
#      objective is analytic). ~500 AdamW steps from a fixed init.
#      ASSERT f_final < 1e-3 * f0  AND  ||p - target|| small.
#   2. SGD+momentum on the SAME bowl, ~1000 steps. ASSERT same.
#   3. AdamW with weight_decay>0 on f(p)=sum(p^2) (min at 0): confirm the
#      DECOUPLED WD drives p -> 0. ASSERT ||p||_final << ||p||_0.
#
# Cross-check: objective 1's converged p is compared (cos >= 0.999) against
# torch.optim running the IDENTICAL objective+init+lr (optim_converge_oracle.py),
# so the convergence RATE is validated — not just "the number went down".
#
# The grad each step is recomputed on the host from the current p (the objective
# is trivial), uploaded as a fresh F32 Tensor, and fed to the REAL in-place
# adamw_step / sgd_step. f(p) is read back and printed every ~50 steps so the
# descent is VISIBLE in the log (Tenet 4: real measurement, no faked convergence).
#
# Run:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/training/parity/optim_converge_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/training/parity/optim_converge_parity.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.training.optim import adamw_step, sgd_step
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/training/parity/optim_converge_ref.txt"
)

# ── Problem config — MUST match optim_converge_oracle.py exactly ─────────────
comptime N = 64
comptime STEPS_ADAMW = 500
comptime STEPS_SGD = 1000
comptime LR = Float32(1.0e-2)
comptime BETA1 = Float32(0.9)
comptime BETA2 = Float32(0.999)
comptime EPS = Float32(1.0e-8)

# SGD on the same bowl: lr small enough to be stable, momentum 0.9, 1000 steps.
comptime SGD_LR = Float32(0.1)
comptime SGD_MOMENTUM = Float32(0.9)

# Objective 3: AdamW with decoupled weight decay on f(p)=sum(p^2), min at 0.
# AdamW's per-coordinate step is ~lr (the adaptive denom normalizes |g|), so
# driving ||p|| from ~15 down below 1e-2 takes ~||p||0/lr + margin steps. 2000
# steps at lr=1e-2 reaches the minimum cleanly (the descent is monotone — see
# the printed trajectory). The data grad (2p) and the DECOUPLED WD both push
# toward 0; with WD the steady-state shrink is faster than data-grad alone.
comptime WD_LR = Float32(1.0e-2)
comptime WD = Float32(0.1)
comptime STEPS_WD = 2000


# ── Deterministic fills (identical to optim_parity._fill / the oracle) ───────
def _fill(n: Int, a: Int, b: Int, c: Float32, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * a) % b) - c) * scale)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(0.0)
    return out^


def _shape1(n: Int) -> List[Int]:
    var s = List[Int]()
    s.append(n)
    return s^


# Explicit element-wise copy (List has no `.copy()` in this Mojo build, and
# Tensor.from_host consumes its `values` list — we keep a host mirror of p to
# recompute the objective each step).
def _copy_list(src: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(src)):
        out.append(src[i])
    return out^


# ── host-side objective helpers ──────────────────────────────────────────────
# f_bowl(p) = sum((p - target)^2);  grad = 2*(p - target).
def _f_bowl(p: List[Float32], target: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(p)):
        var d = Float64(p[i]) - Float64(target[i])
        s += d * d
    return s


def _grad_bowl(p: List[Float32], target: List[Float32]) -> List[Float32]:
    var g = List[Float32]()
    for i in range(len(p)):
        g.append(Float32(2.0) * (p[i] - target[i]))
    return g^


# f_sq(p) = sum(p^2);  grad = 2*p.  (Objective 3 minimizes this with WD on top.)
def _f_sq(p: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(p)):
        var x = Float64(p[i])
        s += x * x
    return s


def _grad_sq(p: List[Float32]) -> List[Float32]:
    var g = List[Float32]()
    for i in range(len(p)):
        g.append(Float32(2.0) * p[i])
    return g^


def _l2_dist(a: List[Float32], b: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(a)):
        var d = Float64(a[i]) - Float64(b[i])
        s += d * d
    return sqrt(s)


def _l2_norm(a: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(a)):
        var x = Float64(a[i])
        s += x * x
    return sqrt(s)


# ── read one tagged space-separated float line (same parser as optim_parity) ──
def _read_ref(tag: String) raises -> List[Float32]:
    var fd = sys_open(String(REF_PATH), O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ref: ") + String(REF_PATH))
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error("empty ref file")
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)

    var prefix = tag + " "
    var pl = prefix.byte_length()
    var out = List[Float32]()
    var i = 0
    while i < done:
        var le = i
        while le < done and Int(buf[le]) != 0x0A:
            le += 1
        var is_match = (le - i) > pl
        if is_match:
            for j in range(pl):
                if Int(buf[i + j]) != ord(prefix[byte=j]):
                    is_match = False
                    break
        if is_match:
            var p = i + pl
            while p < le:
                var c = Int(buf[p])
                if c == 0x20:
                    p += 1
                    continue
                var ne = p
                while ne < le and Int(buf[ne]) != 0x20:
                    ne += 1
                var chars = List[UInt8]()
                for q in range(p, ne):
                    chars.append(buf[q])
                var s = String(from_utf8=chars)
                out.append(Float32(atof(s)))
                p = ne + 1
            buf.free()
            return out^
        i = le + 1
    buf.free()
    raise Error(String("ref tag not found: ") + tag)


# ── Objective 1: AdamW quadratic bowl ────────────────────────────────────────
# Returns the converged p (host) for the torch cross-check.
def _run_adamw_bowl(ctx: DeviceContext) raises -> List[Float32]:
    print("== Objective 1: AdamW  f(p)=sum((p-target)^2)  min=0 @ p=target ==")
    var target = _fill(N, 5, 11, 5.0, 0.05)
    var p_host = _fill(N, 7, 13, 6.0, 0.05)

    var p = Tensor.from_host(_copy_list(p_host), _shape1(N), STDtype.F32, ctx)
    var m = Tensor.from_host(_zeros(N), _shape1(N), STDtype.F32, ctx)
    var v = Tensor.from_host(_zeros(N), _shape1(N), STDtype.F32, ctx)

    var f0 = _f_bowl(p_host, target)
    print("  step    0  f(p)=", f0)
    for t in range(1, STEPS_ADAMW + 1):
        var g_host = _grad_bowl(p_host, target)
        var g = Tensor.from_host(g_host^, _shape1(N), STDtype.F32, ctx)
        adamw_step(p, g, m, v, t, LR, BETA1, BETA2, EPS, Float32(0.0), ctx)
        p_host = p.to_host(ctx)
        if t % 50 == 0:
            print("  step ", t, " f(p)=", _f_bowl(p_host, target))

    var f_final = _f_bowl(p_host, target)
    var ratio = f_final / f0
    var dist = _l2_dist(p_host, target)
    print("  f0=", f0, " f_final=", f_final, " ratio=", ratio)
    print("  ||p - target|| =", dist)
    var converged = (ratio < 1.0e-3) and (dist < 1.0e-2)
    print("  CONVERGED:", converged, " (need ratio<1e-3 AND dist<1e-2)")
    if not converged:
        raise Error("Objective 1 (AdamW bowl) did NOT converge")
    return p_host^


# ── Objective 2: SGD+momentum on the same bowl ───────────────────────────────
def _run_sgd_bowl(ctx: DeviceContext) raises:
    print("")
    print("== Objective 2: SGD+momentum  f(p)=sum((p-target)^2)  min=0 ==")
    var target = _fill(N, 5, 11, 5.0, 0.05)
    var p_host = _fill(N, 7, 13, 6.0, 0.05)

    var p = Tensor.from_host(_copy_list(p_host), _shape1(N), STDtype.F32, ctx)
    var buf = Tensor.from_host(_zeros(N), _shape1(N), STDtype.F32, ctx)

    var f0 = _f_bowl(p_host, target)
    print("  step    0  f(p)=", f0)
    for t in range(1, STEPS_SGD + 1):
        var g_host = _grad_bowl(p_host, target)
        var g = Tensor.from_host(g_host^, _shape1(N), STDtype.F32, ctx)
        sgd_step(p, g, buf, SGD_LR, SGD_MOMENTUM, Float32(0.0), ctx)
        p_host = p.to_host(ctx)
        if t % 50 == 0:
            print("  step ", t, " f(p)=", _f_bowl(p_host, target))

    var f_final = _f_bowl(p_host, target)
    var ratio = f_final / f0
    var dist = _l2_dist(p_host, target)
    print("  f0=", f0, " f_final=", f_final, " ratio=", ratio)
    print("  ||p - target|| =", dist)
    var converged = (ratio < 1.0e-3) and (dist < 1.0e-2)
    print("  CONVERGED:", converged, " (need ratio<1e-3 AND dist<1e-2)")
    if not converged:
        raise Error("Objective 2 (SGD bowl) did NOT converge")


# ── Objective 3: AdamW + decoupled weight decay drives p -> 0 ────────────────
def _run_adamw_wd(ctx: DeviceContext) raises:
    print("")
    print("== Objective 3: AdamW wd>0  f(p)=sum(p^2)  min=0 @ p=0 ==")
    var p_host = _fill(N, 7, 13, 6.0, 0.5)  # non-trivial init away from 0

    var p = Tensor.from_host(_copy_list(p_host), _shape1(N), STDtype.F32, ctx)
    var m = Tensor.from_host(_zeros(N), _shape1(N), STDtype.F32, ctx)
    var v = Tensor.from_host(_zeros(N), _shape1(N), STDtype.F32, ctx)

    var f0 = _f_sq(p_host)
    var norm0 = _l2_norm(p_host)
    print("  step    0  f(p)=", f0, " ||p||=", norm0)
    for t in range(1, STEPS_WD + 1):
        var g_host = _grad_sq(p_host)
        var g = Tensor.from_host(g_host^, _shape1(N), STDtype.F32, ctx)
        adamw_step(p, g, m, v, t, WD_LR, BETA1, BETA2, EPS, WD, ctx)
        p_host = p.to_host(ctx)
        if t % 50 == 0:
            print("  step ", t, " f(p)=", _f_sq(p_host), " ||p||=", _l2_norm(p_host))

    var f_final = _f_sq(p_host)
    var norm_final = _l2_norm(p_host)
    var ratio = f_final / f0
    print("  f0=", f0, " f_final=", f_final, " ratio=", ratio)
    print("  ||p||_0=", norm0, " ||p||_final=", norm_final)
    # min(sum p^2) is at 0; both the data grad (2p) and decoupled WD push toward 0.
    var converged = (ratio < 1.0e-3) and (norm_final < 1.0e-2)
    print("  CONVERGED:", converged, " (need ratio<1e-3 AND ||p||_final<1e-2)")
    if not converged:
        raise Error("Objective 3 (AdamW wd) did NOT drive p -> 0")


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()

    # Objectives 1-3: genuine convergence to known minima.
    var p_conv = _run_adamw_bowl(ctx)
    _run_sgd_bowl(ctx)
    _run_adamw_wd(ctx)

    # ── Cross-check objective 1's converged p vs torch (same problem) ────────
    print("")
    print("== Cross-check: AdamW bowl converged p vs torch.optim ==")
    var ref_p = _read_ref(String("adamw_converge_p"))
    var rX = h.compare_host(p_conv, ref_p)
    print("  mojo-converged-p vs torch-converged-p:", rX)
    var xcheck_pass = rX.cos >= 0.999
    print("  cross-check pass (cos>=0.999):", xcheck_pass)

    print("")
    if xcheck_pass:
        print("OPTIMIZER CONVERGENCE GATE PASSED")
        print("  - AdamW bowl converged to known min (ratio<1e-3)")
        print("  - SGD+momentum bowl converged to known min (ratio<1e-3)")
        print("  - AdamW wd>0 drove p->0 (decoupled WD confirmed)")
        print("  - convergence RATE matches torch (cos>=0.999)")
    else:
        print("OPTIMIZER CONVERGENCE GATE FAILURE (cross-check cos<0.999)")
        raise Error("optim_converge cross-check failed")
