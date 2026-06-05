# training/dora_adapter_smoke.mojo — DoRA adapter gates (a)+(b)+(c)+(d).
#
# TENET 4 / parity-bitrot guard: every check ASSERTS and the file `raise`s
# (nonzero exit) on any wrong value. Demonstrated wrong-run: set env
# DORA_BREAK_GRAD=1 to zero ONE trainable grad (magnitude) — with
# FLAME_ASSERT_GRAD_FLOW=1 the grad-flow gate then ABORTS (exit != 0).
#
# GATE (a) PARITY: the DoRA-normalized effective weight WP_dora + forward, and
#   the 3 grads (lora_down A, lora_up B, magnitude m), checked against:
#     - an INDEPENDENT open-coded recompute of apply_weight_decompose (NOT the
#       dora_adapter helpers) for the forward, AND
#     - a finite-difference of all 3 grads. The FD holds the L2-norm DENOMINATOR
#       FIXED at its base-point value (matching the DETACHED-norm contract,
#       paper §4.3 / dora.rs:124-139) so FD and the analytic detached grad agree.
#   Includes init parity: WP_dora == W_orig at init (ΔW=0 zero-leaf B).
# GATE (b) GRAD-FLOW: after one AdamW step (B off zero), fresh fwd+bwd, feed the
#   3 grads into GradCoverage.measure → ASSERT coverage_pct==100 and dead==0.
#   The DORA_BREAK_GRAD demo proves a zeroed magnitude grad is caught.
# GATE (c) SAVE: save a DoRA adapter, reopen, assert lora_A/lora_B/dora_scale/alpha
#   keys + shapes + byte-exact values round-trip.
# GATE (d) default-off + trainer AOT-build is verified by the builder out-of-band
#   (the trainer fails loud on adapter_algo==3); see the report.
#
# Run (clean PASS, exit 0):
#   rm -f serenitymojo.mojopkg && pixi run mojo run -I . \
#     serenitymojo/training/dora_adapter_smoke.mojo
# Run (deliberate FAIL, exit != 0):
#   FLAME_ASSERT_GRAD_FLOW=1 DORA_BREAK_GRAD=1 pixi run mojo run -I . \
#     serenitymojo/training/dora_adapter_smoke.mojo

from std.collections import List
from std.math import sqrt, abs
from std.ffi import external_call
from std.memory import alloc, UnsafePointer, ArcPointer
from std.builtin.type_aliases import MutExternalOrigin
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.dora_adapter import (
    DoRAAdapter,
    new_dora_adapter,
    dora_delta_weight,
    dora_effective_weight,
    dora_forward,
    dora_backward,
    dora_adamw,
)
from serenitymojo.training.dora_save import (
    NamedDoRA,
    save_dora_peft,
    read_dora_module,
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


def _bf16_to_f32_list(v: List[BFloat16]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(v)):
        out.append(v[i].cast[DType.float32]())
    return out^


def _max_abs_diff_bf16(a: List[BFloat16], b: List[BFloat16]) raises -> Float32:
    if len(a) != len(b):
        raise Error("max_abs_diff_bf16: len mismatch " + String(len(a)) + " != " + String(len(b)))
    var mx = Float32(0.0)
    for i in range(len(a)):
        var d = abs(a[i].cast[DType.float32]() - b[i].cast[DType.float32]())
        if d > mx:
            mx = d
    return mx


# ── INDEPENDENT oracle: WP_dora = m * (WP / (‖WP‖₂+eps)) via raw triple loops ─
# Open-coded, NOT using dora_adapter helpers. ΔW = (B@A)*scale, WP = W_orig+ΔW,
# norm along input axis (wd_on_out=true). Returns wp_dora:[out,in].
def _oracle_wp_dora(w_orig: List[Float32], a: List[Float32], b: List[Float32], m: List[Float32],
                    OUT: Int, IN: Int, R: Int, scale: Float32, eps: Float32) -> List[Float32]:
    # ΔW = (B@A)*scale  [out,in]
    var wp = List[Float32]()
    for o in range(OUT):
        for i in range(IN):
            var dw = Float32(0.0)
            for r in range(R):
                dw += b[o * R + r] * a[r * IN + i]
            wp.append(w_orig[o * IN + i] + dw * scale)
    # WP_dora
    var out = List[Float32]()
    for o in range(OUT):
        var s = Float32(0.0)
        for i in range(IN):
            var v = wp[o * IN + i]
            s += v * v
        var den = sqrt(s) + eps
        for i in range(IN):
            out.append(m[o] * wp[o * IN + i] / den)
    return out^


# Independent forward y = x @ WP_doraᵀ.
def _oracle_fwd(x: List[Float32], wp_dora: List[Float32], M: Int, IN: Int, OUT: Int) -> List[Float32]:
    var out = List[Float32]()
    for mm in range(M):
        for o in range(OUT):
            var s = Float32(0.0)
            for i in range(IN):
                s += x[mm * IN + i] * wp_dora[o * IN + i]
            out.append(s)
    return out^


# ── DETACHED-norm loss for FD: den HELD FIXED at the base-point value ────────
# The analytic grad treats den as constant (detach). So the FD oracle must too:
# we recompute WP at the perturbed parameters but normalize by the FIXED den_base
# (computed once at the unperturbed point). loss = sum(x @ WP_doraᵀ).
def _loss_detached(w_orig: List[Float32], a: List[Float32], b: List[Float32], m: List[Float32],
                   den_base: List[Float32], x: List[Float32],
                   OUT: Int, IN: Int, R: Int, M: Int, scale: Float32) -> Float32:
    # WP = W_orig + (B@A)*scale ; WP_dora = m * WP / den_base (den FIXED).
    var s = Float32(0.0)
    for mm in range(M):
        for o in range(OUT):
            var deno = den_base[o]
            var mo = m[o]
            var acc = Float32(0.0)
            for i in range(IN):
                var dw = Float32(0.0)
                for r in range(R):
                    dw += b[o * R + r] * a[r * IN + i]
                var wp = w_orig[o * IN + i] + dw * scale
                var wpd = mo * wp / deno
                acc += x[mm * IN + i] * wpd
            s += acc
    return s


# den at the base (unperturbed) point — the detached denominator.
def _den_base(w_orig: List[Float32], a: List[Float32], b: List[Float32],
              OUT: Int, IN: Int, R: Int, scale: Float32, eps: Float32) -> List[Float32]:
    var den = List[Float32]()
    for o in range(OUT):
        var s = Float32(0.0)
        for i in range(IN):
            var dw = Float32(0.0)
            for r in range(R):
                dw += b[o * R + r] * a[r * IN + i]
            var wp = w_orig[o * IN + i] + dw * scale
            s += wp * wp
        den.append(sqrt(s) + eps)
    return den^


# FD grad over one trainable tensor (which: 0=A 1=B 2=m), central difference on
# _loss_detached (den held fixed). Module level so it captures nothing.
def _fd_grad(which: Int, w_orig: List[Float32], a: List[Float32], b: List[Float32], m: List[Float32],
             den_base: List[Float32], x: List[Float32],
             OUT: Int, IN: Int, R: Int, M: Int, scale: Float32, h: Float32) raises -> List[Float32]:
    var out = List[Float32]()
    var n: Int
    if which == 0:
        n = len(a)
    elif which == 1:
        n = len(b)
    else:
        n = len(m)
    for k in range(n):
        var ap = a.copy(); var am = a.copy()
        var bp = b.copy(); var bm = b.copy()
        var mp = m.copy(); var mmn = m.copy()
        if which == 0:
            ap[k] = ap[k] + h; am[k] = am[k] - h
        elif which == 1:
            bp[k] = bp[k] + h; bm[k] = bm[k] - h
        else:
            mp[k] = mp[k] + h; mmn[k] = mmn[k] - h
        var lp = _loss_detached(w_orig, ap, bp, mp, den_base, x, OUT, IN, R, M, scale)
        var lm = _loss_detached(w_orig, am, bm, mmn, den_base, x, OUT, IN, R, M, scale)
        out.append((lp - lm) / (Float32(2.0) * h))
    return out^


def main() raises:
    var ctx = DeviceContext()
    var ok = True

    var IN = 5
    var OUT = 4
    var R = 2
    var M = 3
    var alpha = Float32(2.0)   # scale = alpha/rank = 1.0
    var eps = Float32(1.0e-7)

    # representative FROZEN base weight W_orig:[out,in] (deterministic, nonzero).
    var w_orig = List[Float32]()
    for i in range(OUT * IN):
        w_orig.append(Float32(0.1) * Float32(((i * 5) % 13) - 6) + Float32(0.3))

    # ── GATE (a) init parity: WP_dora == W_orig at init (ΔW=0 zero-leaf B) ────
    var d0 = new_dora_adapter(w_orig, IN, OUT, R, alpha, 42, eps)
    var eff0 = dora_effective_weight(w_orig, d0)
    var init_mx = _max_abs_diff(eff0.wp_dora, w_orig)
    if init_mx > Float32(1.0e-5):
        print("FAIL (a-init): WP_dora != W_orig at init, max|Δ|=", init_mx); ok = False
    else:
        print("PASS (a-init): WP_dora == W_orig at init (B zero leaf), max|Δ|=", init_mx)

    # Drive B off zero so all 3 trainables are live for parity / grad-flow.
    var d = new_dora_adapter(w_orig, IN, OUT, R, alpha, 7, eps)
    for i in range(len(d.b)):
        d.b[i] = BFloat16(Float32(0.03) * Float32((i % 5) + 1))
    if d.scale != Float32(1.0):
        print("FAIL: expected scale 1.0, got", d.scale); ok = False

    # deterministic input x [M,IN]
    var x = List[Float32]()
    for i in range(M * IN):
        x.append(Float32(0.1) * Float32(((i * 3) % 11) - 5))

    # ── GATE (a) effective-weight + forward parity vs independent oracle ─────
    var eff = dora_effective_weight(w_orig, d)
    var d_a_h = _bf16_to_f32_list(d.a)
    var d_b_h = _bf16_to_f32_list(d.b)
    var d_m_h = _bf16_to_f32_list(d.m)
    var wp_oracle = _oracle_wp_dora(w_orig, d_a_h, d_b_h, d_m_h, OUT, IN, R, d.scale, eps)
    var wp_mx = _max_abs_diff(eff.wp_dora, wp_oracle)
    if wp_mx > Float32(1.0e-5):
        print("FAIL (a-wp): WP_dora vs oracle max|Δ|=", wp_mx); ok = False
    else:
        print("PASS (a-wp): WP_dora matches oracle, max|Δ|=", wp_mx)

    var y_impl = dora_forward(x, w_orig, d, M)
    var y_oracle = _oracle_fwd(x, wp_oracle, M, IN, OUT)
    var y_mx = _max_abs_diff(y_impl, y_oracle)
    if y_mx > Float32(1.0e-5):
        print("FAIL (a-fwd): y vs oracle max|Δ|=", y_mx); ok = False
    else:
        print("PASS (a-fwd): forward y matches oracle, max|Δ|=", y_mx)

    # ── GATE (a) backward: analytic 3 grads vs finite-difference (detached den) ─
    var d_y = List[Float32]()
    for _ in range(M * OUT):
        d_y.append(Float32(1.0))               # loss = sum(y) → d_y all ones
    var g = dora_backward(d_y, x, w_orig, d, M)

    var h = Float32(1.0e-3)
    var den_base = _den_base(w_orig, d_a_h, d_b_h, OUT, IN, R, d.scale, eps)
    var fda = _fd_grad(0, w_orig, d_a_h, d_b_h, d_m_h, den_base, x, OUT, IN, R, M, d.scale, h)
    var fdb = _fd_grad(1, w_orig, d_a_h, d_b_h, d_m_h, den_base, x, OUT, IN, R, M, d.scale, h)
    var fdm = _fd_grad(2, w_orig, d_a_h, d_b_h, d_m_h, den_base, x, OUT, IN, R, M, d.scale, h)

    var tol_fd = Float32(2.0e-2)
    var ea = _max_abs_diff(g.d_a, fda)
    var eb = _max_abs_diff(g.d_b, fdb)
    var em = _max_abs_diff(g.d_m, fdm)
    if ea > tol_fd:
        print("FAIL (a-bwd lora_A): analytic vs FD max|Δ|=", ea); ok = False
    else:
        print("PASS (a-bwd lora_A): max|Δ| vs FD=", ea)
    if eb > tol_fd:
        print("FAIL (a-bwd lora_B): analytic vs FD max|Δ|=", eb); ok = False
    else:
        print("PASS (a-bwd lora_B): max|Δ| vs FD=", eb)
    if em > tol_fd:
        print("FAIL (a-bwd magnitude): analytic vs FD max|Δ|=", em); ok = False
    else:
        print("PASS (a-bwd magnitude): max|Δ| vs FD=", em)

    # ── GATE (b) GRAD-FLOW: A, B, m all nonzero ──────────────────────────────
    var d_b2 = d.copy()
    dora_adamw(d_b2, g, 1, Float32(1.0e-3))
    var g2 = dora_backward(d_y, x, w_orig, d_b2, M)

    var names = List[String]()
    names.append(String("toy.lora_A"))
    names.append(String("toy.lora_B"))
    names.append(String("toy.dora_scale"))

    # DELIBERATE-WRONG demo: zero the magnitude grad if armed.
    var break_grad = _env_is_set(String("DORA_BREAK_GRAD"))
    var dm_maybe = g2.d_m.copy()
    if break_grad:
        for i in range(len(dm_maybe)):
            dm_maybe[i] = Float32(0.0)
        print("INFO: DORA_BREAK_GRAD set — zeroing magnitude grad to prove the gate catches it")

    var grads = List[TArc]()
    grads.append(TArc(Tensor.from_host(g2.d_a.copy(), [R, IN], STDtype.F32, ctx)))
    grads.append(TArc(Tensor.from_host(g2.d_b.copy(), [OUT, R], STDtype.F32, ctx)))
    grads.append(TArc(Tensor.from_host(dm_maybe^, [OUT, 1], STDtype.F32, ctx)))

    var rep = measure(names, grads, ctx)
    print("[grad-flow] total=", rep.total, " nonzero=", rep.nonzero,
          " dead=", rep.dead, " coverage=", rep.coverage_pct(), "%")

    var armed = _env_is_set(String("FLAME_ASSERT_GRAD_FLOW"))
    if break_grad:
        if rep.dead == 0:
            print("FAIL (b-demo): broken grad NOT detected (dead==0)"); ok = False
        else:
            print("PASS (b-demo): dead grad detected, dead=", rep.dead)
        if armed:
            raise Error(
                String("[grad-flow] DoRA FAILED: ") + String(rep.dead)
                + " of " + String(rep.total) + " trainables DEAD (zeroed magnitude) — "
                + "gate aborts (exit != 0)"
            )
    else:
        if rep.dead != 0:
            print("FAIL (b): a DoRA trainable grad is DEAD (dead=", rep.dead, ")"); ok = False
        else:
            print("PASS (b): all 3 DoRA grads nonzero, dead=0")
        if rep.coverage_pct() != Float64(100.0):
            print("FAIL (b): coverage_pct != 100 (got", rep.coverage_pct(), ")"); ok = False
        else:
            print("PASS (b): coverage_pct == 100 (A, B, magnitude all live)")

    # ── GATE (c) SAVE: save → reopen → assert keys + shapes + alpha + values ──
    var named = List[NamedDoRA]()
    named.append(NamedDoRA(String("double_blocks.0.img_attn.to_q"), d.copy()))
    var path = String("/tmp/dora_smoke.safetensors")
    var n_written = save_dora_peft(named, path, ctx)
    if n_written != 1:
        print("FAIL (c): save returned", n_written, "adapters, expected 1"); ok = False
    else:
        print("PASS (c): saved 1 DoRA adapter to", path)

    var rb = read_dora_module(String("double_blocks.0.img_attn.to_q"), path, ctx)
    if rb.in_f != IN or rb.out_f != OUT or rb.rank != R:
        print("FAIL (c): shape mismatch in/out/rank got",
              rb.in_f, rb.out_f, rb.rank, "expected", IN, OUT, R); ok = False
    else:
        print("PASS (c): shapes round-trip in=", rb.in_f, " out=", rb.out_f, " rank=", rb.rank)
    if abs(rb.alpha - alpha) > Float32(1.0e-6):
        print("FAIL (c): alpha mismatch got", rb.alpha, "expected", alpha); ok = False
    else:
        print("PASS (c): alpha round-trip =", rb.alpha)
    var a_mx = _max_abs_diff_bf16(rb.a, d.a)
    var b_mx = _max_abs_diff_bf16(rb.b, d.b)
    var m_mx = _max_abs_diff_bf16(rb.m, d.m)
    if a_mx > Float32(1.0e-6) or b_mx > Float32(1.0e-6) or m_mx > Float32(1.0e-6):
        print("FAIL (c): values not byte-exact, A Δ=", a_mx, " B Δ=", b_mx, " m Δ=", m_mx); ok = False
    else:
        print("PASS (c): lora_A/lora_B/dora_scale values round-trip byte-exact")

    if not ok:
        raise Error("dora_adapter_smoke FAILED")
    print("dora_adapter_smoke ALL GATES PASS")
