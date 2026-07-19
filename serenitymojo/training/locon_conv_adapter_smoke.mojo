# training/locon_conv_adapter_smoke.mojo — LoCon conv adapter gates (a)+(b)+(c)+(d).
#
# TENET 4 / parity-bitrot guard: every check ASSERTS and the file `raise`s
# (nonzero exit) on any wrong value. Two bitrot demonstrations:
#   LOCON_BREAK_FACTOR=1 (+ FLAME_ASSERT_GRAD_FLOW=1) zeroes ONE live factor grad
#     → the grad-flow gate ABORTS (exit != 0).
#   LOCON_BREAK_BWD=1 injects a WRONG-BUT-NONZERO d_down (×1.5) → the FD parity
#     gate catches it (exit 1) even though grad-flow stays green — proving the FD
#     gate is not vacuous (grad-flow alone is insufficient).
#
# GATE (a) PARITY: forward delta Δy = up(down(x))*scale vs an INDEPENDENT
#   open-coded recompute (direct conv cross-correlation, NOT the adapter helpers)
#   PLUS finite-difference of d_down and d_up. Includes identity-at-init
#   (up=0 → Δy == 0).
# GATE (b) GRAD-FLOW: after one AdamW step (up off zero), fresh fwd+bwd, feed the
#   {down, up} grads into GradCoverage.measure → coverage_pct==100, dead==0.
# GATE (c) SAVE: save → reopen → assert lora_down/lora_up/alpha keys + shapes +
#   byte-exact values (round-trips through the Kohya OIHW <-> Flame RSCF permute).
# GATE (d) default-off + trainer AOT-build verified out-of-band (trainer fails
#   loud on adapter_algo==7); see the builder report.
#
# Run (clean PASS, exit 0):
#   rm -f serenitymojo.mojopkg && pixi run mojo run -I . \
#     serenitymojo/training/locon_conv_adapter_smoke.mojo
# Run (FD bitrot FAIL, exit != 0):
#   LOCON_BREAK_BWD=1 pixi run mojo run -I . \
#     serenitymojo/training/locon_conv_adapter_smoke.mojo
# Run (grad-flow FAIL, exit != 0):
#   FLAME_ASSERT_GRAD_FLOW=1 LOCON_BREAK_FACTOR=1 pixi run mojo run -I . \
#     serenitymojo/training/locon_conv_adapter_smoke.mojo

from std.collections import List
from std.math import abs
from std.ffi import external_call
from std.memory import alloc, UnsafePointer, ArcPointer
from std.builtin.type_aliases import MutExternalOrigin
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.locon_conv_adapter import (
    LoConConvAdapter,
    new_locon_conv_adapter,
    locon_out_h,
    locon_out_w,
    locon_forward,
    locon_backward,
    locon_adamw,
)
from serenitymojo.training.locon_save import (
    NamedLoCon,
    save_locon_peft,
    read_locon_module,
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


# ── INDEPENDENT oracle forward: direct conv cross-correlation (down then 1x1 up,
# scaled), open-coded — NOT the adapter helpers. ──────────────────────────────
def _oracle_forward(
    x: List[Float32], down: List[Float32], up: List[Float32], scale: Float32,
    N: Int, Hi: Int, Wi: Int, Cin: Int, Kh: Int, Kw: Int, R: Int, Cout: Int,
    sh: Int, sw: Int, ph: Int, pw: Int,
) -> List[Float32]:
    var Ho = (Hi + 2 * ph - Kh) // sh + 1
    var Wo = (Wi + 2 * pw - Kw) // sw + 1
    var rows = N * Ho * Wo
    var y = List[Float32]()
    for _ in range(rows * Cout):
        y.append(Float32(0.0))
    for n in range(N):
        for oh in range(Ho):
            for ow in range(Wo):
                # h[r] = Σ x * down
                var hr = List[Float32]()
                for _ in range(R):
                    hr.append(Float32(0.0))
                for kh in range(Kh):
                    var ih = oh * sh - ph + kh
                    if ih < 0 or ih >= Hi:
                        continue
                    for kw in range(Kw):
                        var iw = ow * sw - pw + kw
                        if iw < 0 or iw >= Wi:
                            continue
                        for ci in range(Cin):
                            var xv = x[(((n * Hi + ih) * Wi + iw) * Cin) + ci]
                            for r in range(R):
                                var wv = down[(((kh * Kw + kw) * Cin) + ci) * R + r]
                                hr[r] = hr[r] + xv * wv
                # y[co] = scale * Σ_r h[r] * up[r,co]
                var ybase = (((n * Ho + oh) * Wo + ow) * Cout)
                for co in range(Cout):
                    var s = Float32(0.0)
                    for r in range(R):
                        s += hr[r] * up[r * Cout + co]
                    y[ybase + co] = s * scale
    return y^


# loss(down, up) for FD = sum(oracle forward). Independent of locon_backward.
def _loss_for_fd(
    down: List[Float32], up: List[Float32], scale: Float32, x: List[Float32],
    N: Int, Hi: Int, Wi: Int, Cin: Int, Kh: Int, Kw: Int, R: Int, Cout: Int,
    sh: Int, sw: Int, ph: Int, pw: Int,
) -> Float32:
    var y = _oracle_forward(x, down, up, scale, N, Hi, Wi, Cin, Kh, Kw, R, Cout, sh, sw, ph, pw)
    var s = Float32(0.0)
    for i in range(len(y)):
        s += y[i]
    return s


# FD grad over one factor (which: 0=down 1=up), central difference.
def _fd_grad(
    which: Int, down: List[Float32], up: List[Float32], scale: Float32, x: List[Float32],
    N: Int, Hi: Int, Wi: Int, Cin: Int, Kh: Int, Kw: Int, R: Int, Cout: Int,
    sh: Int, sw: Int, ph: Int, pw: Int, h: Float32,
) raises -> List[Float32]:
    var out = List[Float32]()
    var n: Int
    if which == 0:
        n = len(down)
    else:
        n = len(up)
    for k in range(n):
        var dp = down.copy(); var dm = down.copy()
        var upp = up.copy(); var upm = up.copy()
        if which == 0:
            dp[k] = dp[k] + h; dm[k] = dm[k] - h
        else:
            upp[k] = upp[k] + h; upm[k] = upm[k] - h
        var lp = _loss_for_fd(dp, upp, scale, x, N, Hi, Wi, Cin, Kh, Kw, R, Cout, sh, sw, ph, pw)
        var lm = _loss_for_fd(dm, upm, scale, x, N, Hi, Wi, Cin, Kh, Kw, R, Cout, sh, sw, ph, pw)
        out.append((lp - lm) / (Float32(2.0) * h))
    return out^


def _run_config(
    label: String, N: Int, Hi: Int, Wi: Int, Cin: Int, Cout: Int,
    Kh: Int, Kw: Int, rank: Int, sh: Int, sw: Int, ph: Int, pw: Int,
    break_factor: Bool, break_bwd: Bool, armed: Bool, ctx: DeviceContext,
) raises -> Bool:
    var ok = True
    print("=== config", label, "(N=", N, " HxW=", Hi, "x", Wi, " Cin=", Cin,
          " Cout=", Cout, " K=", Kh, "x", Kw, " rank=", rank,
          " stride=", sh, "x", sw, " pad=", ph, "x", pw, ") ===")

    var alpha = Float32(Float32(rank))   # scale = alpha/rank = 1.0
    var lo = new_locon_conv_adapter(Cin, Cout, Kh, Kw, rank, alpha, sh, sw, ph, pw, 13)
    if lo.scale != Float32(1.0):
        print("FAIL: expected scale 1.0, got", lo.scale); ok = False

    var Ho = locon_out_h(Hi, Kh, sh, ph)
    var Wo = locon_out_w(Wi, Kw, sw, pw)

    # deterministic input x [N,Hi,Wi,Cin]
    var x = List[Float32]()
    for i in range(N * Hi * Wi * Cin):
        x.append(Float32(0.1) * Float32(((i * 3) % 11) - 5))

    # ── (a-init) Δy == 0 at init (zero up leg) ──
    var y0 = locon_forward(x, lo, N, Hi, Wi)
    var init_mx = Float32(0.0)
    for i in range(len(y0)):
        if abs(y0[i]) > init_mx:
            init_mx = abs(y0[i])
    if init_mx != Float32(0.0):
        print("FAIL (a-init): initial Δy not exactly 0, max|Δy|=", init_mx); ok = False
    else:
        print("PASS (a-init): initial Δy == 0 (zero up leg)")

    # Drive up off zero so both factors are exercised.
    for i in range(len(lo.up)):
        lo.up[i] = Float32(0.05) * Float32((i % 7) + 1)

    # ── (a-fwd) forward parity vs independent oracle ──
    var y_impl = locon_forward(x, lo, N, Hi, Wi)
    var y_oracle = _oracle_forward(x, lo.down, lo.up, lo.scale, N, Hi, Wi, Cin, Kh, Kw, rank, Cout, sh, sw, ph, pw)
    var y_mx = _max_abs_diff(y_impl, y_oracle)
    if y_mx > Float32(1.0e-4):
        print("FAIL (a-fwd): Δy vs oracle max|Δ|=", y_mx); ok = False
    else:
        print("PASS (a-fwd): forward Δy matches conv oracle, max|Δ|=", y_mx)

    # ── (a-bwd) analytic grads vs finite-difference ──
    var d_y = List[Float32]()
    for _ in range(N * Ho * Wo * Cout):
        d_y.append(Float32(1.0))
    var g = locon_backward(d_y, x, lo, N, Hi, Wi)

    # BITROT DEMO: corrupt analytic d_down by ×1.5 (wrong-but-nonzero).
    if break_bwd:
        for i in range(len(g.d_down)):
            g.d_down[i] = g.d_down[i] * Float32(1.5)
        print("INFO: LOCON_BREAK_BWD set — scaling analytic d_down by 1.5 to prove the FD gate catches it")

    var hfd = Float32(1.0e-3)
    var tol_fd = Float32(2.0e-2)
    var tol_rel = Float32(2.0e-2)
    var rel_floor = Float32(1.0e-4)

    var fd_down = _fd_grad(0, lo.down, lo.up, lo.scale, x, N, Hi, Wi, Cin, Kh, Kw, rank, Cout, sh, sw, ph, pw, hfd)
    var e_down = _max_abs_diff(g.d_down, fd_down)
    var er_down = _max_rel_diff(g.d_down, fd_down, rel_floor)
    if e_down > tol_fd or er_down > tol_rel:
        print("FAIL (a-bwd down): analytic vs FD abs|Δ|=", e_down, " rel=", er_down); ok = False
    else:
        print("PASS (a-bwd down): max|Δ| vs FD=", e_down, " rel=", er_down)

    var fd_up = _fd_grad(1, lo.down, lo.up, lo.scale, x, N, Hi, Wi, Cin, Kh, Kw, rank, Cout, sh, sw, ph, pw, hfd)
    var e_up = _max_abs_diff(g.d_up, fd_up)
    var er_up = _max_rel_diff(g.d_up, fd_up, rel_floor)
    if e_up > tol_fd or er_up > tol_rel:
        print("FAIL (a-bwd up): analytic vs FD abs|Δ|=", e_up, " rel=", er_up); ok = False
    else:
        print("PASS (a-bwd up): max|Δ| vs FD=", e_up, " rel=", er_up)

    # ── (b) GRAD-FLOW over {down, up} ──
    var lo_b = lo.copy()
    var g_clean = locon_backward(d_y, x, lo_b, N, Hi, Wi)
    locon_adamw(lo_b, g_clean, 1, Float32(1.0e-3))
    var g2 = locon_backward(d_y, x, lo_b, N, Hi, Wi)

    var names = List[String]()
    var grads = List[TArc]()
    names.append(String("toy.lora_down"))
    grads.append(TArc(Tensor.from_host(g2.d_down.copy(), [Kh, Kw, Cin, rank], STDtype.F32, ctx)))
    var up_maybe = g2.d_up.copy()
    if break_factor:
        for i in range(len(up_maybe)):
            up_maybe[i] = Float32(0.0)
        print("INFO: LOCON_BREAK_FACTOR set — zeroing lora_up grad to prove the gate catches it")
    names.append(String("toy.lora_up"))
    grads.append(TArc(Tensor.from_host(up_maybe^, [rank, Cout], STDtype.F32, ctx)))

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
                String("[grad-flow] LoCon FAILED: ") + String(rep.dead)
                + " of " + String(rep.total) + " live factors DEAD — gate aborts (exit != 0)"
            )
    else:
        if rep.dead != 0:
            print("FAIL (b): a LoCon factor grad is DEAD (dead=", rep.dead, ")"); ok = False
        else:
            print("PASS (b): all live LoCon factor grads nonzero, dead=0")
        if rep.coverage_pct() != Float64(100.0):
            print("FAIL (b): coverage_pct != 100 (got", rep.coverage_pct(), ")"); ok = False
        else:
            print("PASS (b): coverage_pct == 100 (down + up)")

    # ── (c) SAVE round-trip ──
    var named = List[NamedLoCon]()
    named.append(NamedLoCon(String("down_blocks.0.resnets.0.conv1"), lo.copy()))
    var path = String("/tmp/locon_smoke_") + label + String(".safetensors")
    var n_written = save_locon_peft(named, path, ctx)
    if n_written != 1:
        print("FAIL (c): save returned", n_written, "adapters, expected 1"); ok = False
    else:
        print("PASS (c): saved 1 LoCon adapter to", path)

    var rb = read_locon_module(String("down_blocks.0.resnets.0.conv1"), path, ctx)
    if rb.cin != Cin or rb.cout != Cout or rb.kh != Kh or rb.kw != Kw or rb.rank != rank:
        print("FAIL (c): shape mismatch got cin=", rb.cin, " cout=", rb.cout, " kh=", rb.kh,
              " kw=", rb.kw, " rank=", rb.rank); ok = False
    else:
        print("PASS (c): shapes round-trip cin=", rb.cin, " cout=", rb.cout, " kh=", rb.kh, " kw=", rb.kw, " rank=", rb.rank)
    if abs(rb.alpha - alpha) > Float32(1.0e-6):
        print("FAIL (c): alpha mismatch got", rb.alpha, "expected", alpha); ok = False
    else:
        print("PASS (c): alpha round-trip =", rb.alpha)
    var down_mx = _max_abs_diff(rb.down, lo.down)
    var up_mx = _max_abs_diff(rb.up, lo.up)
    if down_mx > Float32(1.0e-6) or up_mx > Float32(1.0e-6):
        print("FAIL (c): down/up not byte-exact, down Δ=", down_mx, " up Δ=", up_mx); ok = False
    else:
        print("PASS (c): lora_down/lora_up round-trip byte-exact (OIHW<->RSCF permute)")

    return ok


def main() raises:
    var ctx = DeviceContext()
    var armed = _env_is_set(String("FLAME_ASSERT_GRAD_FLOW"))
    var break_factor = _env_is_set(String("LOCON_BREAK_FACTOR"))
    var break_bwd = _env_is_set(String("LOCON_BREAK_BWD"))

    # 3x3 stride 1 pad 1 (same-size conv): the canonical resnet conv.
    var ok1 = _run_config(String("k3s1p1"), 2, 5, 4, 3, 4, 3, 3, 2, 1, 1, 1, 1,
                           break_factor, break_bwd, armed, ctx)
    # 3x3 stride 2 pad 1 (downsampling conv): exercises stride/pad in the bwd map.
    var ok2 = _run_config(String("k3s2p1"), 1, 6, 6, 2, 3, 3, 3, 1, 2, 2, 1, 1,
                           False, break_bwd, armed, ctx)

    if not (ok1 and ok2):
        raise Error("locon_conv_adapter_smoke FAILED")
    print("locon_conv_adapter_smoke ALL GATES PASS (k3s1p1 + k3s2p1)")
