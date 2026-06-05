# training/oft_adapter_smoke.mojo — Diag-OFT adapter gates (a)+(b)+(c)+(d).
#
# TENET 4 / parity-bitrot guard: every check ASSERTS and the file `raise`s
# (nonzero exit) on any wrong value. Demonstrated wrong-run: set env
# OFT_BREAK_FACTOR=1 to zero the S grad — with FLAME_ASSERT_GRAD_FLOW=1 the
# grad-flow gate then ABORTS (exit != 0).
#
# GATE (a) PARITY:
#   - identity-at-init: S=0 → R=I → W_eff==W (max|Δ|==0).
#   - orthogonality: after driving S off zero, RᵀR ≈ I per block (max|Δ| tiny).
#   - forward parity: y = x @ W_effᵀ vs an INDEPENDENT open-coded recompute
#     (skew → Cayley via an independent 2×2 / general inverse → W_eff → y),
#     NOT the oft_adapter helpers.
#   - FD parity: analytic d_S vs central finite-difference of loss=sum(y) over
#     EVERY element of S (the exact-Cayley analytic backward is proven here).
# GATE (b) GRAD-FLOW: after one AdamW step (S off zero), fresh fwd+bwd, feed
#   d_S into GradCoverage.measure → coverage_pct==100, dead==0. The
#   OFT_BREAK_FACTOR demo proves a dead grad is caught.
# GATE (c) SAVE: save → reopen → assert oft_blocks [num_blocks,b,b] + alpha +
#   byte-exact S values.
# GATE (d) default-off + trainer AOT-build verified out-of-band (trainer fails
#   loud on adapter_algo==5); see the builder report.
#
# Run (clean PASS, exit 0):
#   rm -f serenitymojo.mojopkg && pixi run mojo run -I . \
#     serenitymojo/training/oft_adapter_smoke.mojo
# Run (deliberate FAIL, exit != 0):
#   FLAME_ASSERT_GRAD_FLOW=1 OFT_BREAK_FACTOR=1 pixi run mojo run -I . \
#     serenitymojo/training/oft_adapter_smoke.mojo

from std.collections import List
from std.math import abs
from std.ffi import external_call
from std.memory import alloc, UnsafePointer, ArcPointer
from std.builtin.type_aliases import MutExternalOrigin
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.oft_adapter import (
    OFTAdapter,
    new_oft_adapter,
    oft_rotation_full,
    oft_effective_weight,
    oft_forward,
    oft_backward,
    oft_adamw,
)
from serenitymojo.training.oft_save import (
    NamedOFT,
    save_oft_peft,
    read_oft_module,
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


# Relative grad-parity error (same rationale as lokr_adapter_smoke): a
# multiplicative analytic-backward bug is below a fixed abs tolerance when the
# grad is intrinsically small, so the gate would silently pass. `b` is the FD
# reference; entries below `floor` are skipped (FD truncation noise dominates).
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


# ── INDEPENDENT oracle: skew → Cayley → block-diag W_eff = R@W, open-coded ────
# General b×b inverse (independent Gauss-Jordan, distinct code from the adapter).
def _oracle_inverse(a: List[Float32], n: Int) raises -> List[Float32]:
    var aug = List[Float32]()
    for i in range(n):
        for j in range(n):
            aug.append(a[i * n + j])
        for j in range(n):
            aug.append(Float32(1.0) if i == j else Float32(0.0))
    var w = 2 * n
    for col in range(n):
        var piv = col
        var babs = abs(aug[col * w + col])
        for r in range(col + 1, n):
            var va = abs(aug[r * w + col])
            if va > babs:
                babs = va; piv = r
        if babs == Float32(0.0):
            raise Error("oracle inverse singular")
        if piv != col:
            for k in range(w):
                var tmp = aug[col * w + k]
                aug[col * w + k] = aug[piv * w + k]
                aug[piv * w + k] = tmp
        var inv_d = Float32(1.0) / aug[col * w + col]
        for k in range(w):
            aug[col * w + k] = aug[col * w + k] * inv_d
        for r in range(n):
            if r == col:
                continue
            var f = aug[r * w + col]
            for k in range(w):
                aug[r * w + k] = aug[r * w + k] - f * aug[col * w + k]
    var out = List[Float32]()
    for i in range(n):
        for j in range(n):
            out.append(aug[i * w + n + j])
    return out^


def _oracle_block_r(sblk: List[Float32], b: Int) raises -> List[Float32]:
    # Q = 0.5(S - Sᵀ); M = I+Q; A = I-Q; R = M⁻¹ A.
    var mplus = List[Float32]()
    var aminus = List[Float32]()
    for _ in range(b * b):
        mplus.append(Float32(0.0)); aminus.append(Float32(0.0))
    for i in range(b):
        for j in range(b):
            var qij = Float32(0.5) * (sblk[i * b + j] - sblk[j * b + i])
            var id = Float32(1.0) if i == j else Float32(0.0)
            mplus[i * b + j] = id + qij
            aminus[i * b + j] = id - qij
    var minv = _oracle_inverse(mplus, b)
    # R = minv @ aminus
    var R = List[Float32]()
    for i in range(b):
        for j in range(b):
            var acc = Float32(0.0)
            for k in range(b):
                acc += minv[i * b + k] * aminus[k * b + j]
            R.append(acc)
    return R^


def _oracle_weff(s: List[Float32], w_base: List[Float32], NB: Int, B: Int, IN: Int) raises -> List[Float32]:
    var w_eff = w_base.copy()
    for g in range(NB):
        var sblk = List[Float32]()
        var base_s = g * B * B
        for i in range(B * B):
            sblk.append(s[base_s + i])
        var rg = _oracle_block_r(sblk, B)
        var base = g * B
        for i in range(B):
            for c in range(IN):
                var acc = Float32(0.0)
                for j in range(B):
                    acc += rg[i * B + j] * w_base[(base + j) * IN + c]
                w_eff[(base + i) * IN + c] = acc
    return w_eff^


def _oracle_fwd(x: List[Float32], w_eff: List[Float32], M: Int, IN: Int, OUT: Int) -> List[Float32]:
    var out = List[Float32]()
    for mm in range(M):
        for o in range(OUT):
            var s = Float32(0.0)
            for i in range(IN):
                s += x[mm * IN + i] * w_eff[o * IN + i]
            out.append(s)
    return out^


# loss(S) for FD: sum(x @ W_effᵀ). Recomputes W_eff from S independently.
def _loss_for_fd(s: List[Float32], w_base: List[Float32], x: List[Float32],
                 NB: Int, B: Int, IN: Int, OUT: Int, M: Int) raises -> Float32:
    var w_eff = _oracle_weff(s, w_base, NB, B, IN)
    var y = _oracle_fwd(x, w_eff, M, IN, OUT)
    var s_sum = Float32(0.0)
    for i in range(len(y)):
        s_sum += y[i]
    return s_sum


# FD grad over S, central difference.
def _fd_grad_s(s: List[Float32], w_base: List[Float32], x: List[Float32],
               NB: Int, B: Int, IN: Int, OUT: Int, M: Int, h: Float32) raises -> List[Float32]:
    var out = List[Float32]()
    for k in range(len(s)):
        var sp = s.copy(); var sm = s.copy()
        sp[k] = sp[k] + h; sm[k] = sm[k] - h
        var lp = _loss_for_fd(sp, w_base, x, NB, B, IN, OUT, M)
        var lm = _loss_for_fd(sm, w_base, x, NB, B, IN, OUT, M)
        out.append((lp - lm) / (Float32(2.0) * h))
    return out^


# RᵀR ≈ I check over the full block-diagonal R [out,out].
def _max_rtr_minus_i(R: List[Float32], OUT: Int) raises -> Float32:
    var rt = List[Float32]()
    for _ in range(OUT * OUT):
        rt.append(Float32(0.0))
    for i in range(OUT):
        for j in range(OUT):
            rt[j * OUT + i] = R[i * OUT + j]
    var mx = Float32(0.0)
    for i in range(OUT):
        for j in range(OUT):
            var acc = Float32(0.0)
            for k in range(OUT):
                acc += rt[i * OUT + k] * R[k * OUT + j]
            var id = Float32(1.0) if i == j else Float32(0.0)
            var d = abs(acc - id)
            if d > mx:
                mx = d
    return mx


def _run_config(label: String, in_f: Int, out_f: Int, block_size: Int,
                break_factor: Bool, armed: Bool, ctx: DeviceContext) raises -> Bool:
    var ok = True
    print("=== config", label, "(in=", in_f, " out=", out_f, " block_size=", block_size, ") ===")

    var alpha = Float32(1.0)
    # deterministic frozen base weight [out,in]
    var w_base = List[Float32]()
    for i in range(out_f * in_f):
        w_base.append(Float32(0.05) * Float32(((i * 7) % 13) - 6) + Float32(0.3))

    var lo = new_oft_adapter(in_f, out_f, block_size, alpha, w_base.copy())
    var NB = lo.num_blocks; var B = lo.block_size
    var IN = lo.in_f; var OUT = lo.out_f; var M = 3

    # ── (a) identity-at-init: S=0 → R=I → W_eff == W ──
    var weff0 = oft_effective_weight(lo)
    var init_mx = _max_abs_diff(weff0, lo.w_base)
    if init_mx != Float32(0.0):
        print("FAIL (a-init): W_eff != W_base at init, max|Δ|=", init_mx); ok = False
    else:
        print("PASS (a-init): W_eff == W_base at init (R=I, zero S)")

    var R0 = oft_rotation_full(lo)
    var rtr0 = _max_rtr_minus_i(R0, OUT)
    if rtr0 > Float32(1.0e-5):
        print("FAIL (a-init): RᵀR != I at init, max|Δ|=", rtr0); ok = False
    else:
        print("PASS (a-init): RᵀR == I at init, max|Δ|=", rtr0)

    # Drive S off zero so the rotation is non-trivial and all S entries exercised.
    for i in range(len(lo.s)):
        lo.s[i] = BFloat16(Float32(0.07) * Float32((i % 7) - 3) + Float32(0.05))
    var s_h = _bf16_to_f32_list(lo.s)

    # ── (a) orthogonality of the trained R: RᵀR ≈ I ──
    var R = oft_rotation_full(lo)
    var rtr = _max_rtr_minus_i(R, OUT)
    if rtr > Float32(2.0e-5):
        print("FAIL (a-orth): RᵀR != I, max|RᵀR - I|=", rtr); ok = False
    else:
        print("PASS (a-orth): RᵀR == I (exact Cayley), max|RᵀR - I|=", rtr)

    # deterministic input x [M,IN]
    var x = List[Float32]()
    for i in range(M * IN):
        x.append(Float32(0.1) * Float32(((i * 3) % 11) - 5))

    # ── (a) W_eff + forward parity vs independent oracle ──
    var weff_impl = oft_effective_weight(lo)
    var weff_oracle = _oracle_weff(s_h, lo.w_base, NB, B, IN)
    var weff_mx = _max_abs_diff(weff_impl, weff_oracle)
    if weff_mx > Float32(1.0e-5):
        print("FAIL (a-weff): W_eff vs oracle max|Δ|=", weff_mx); ok = False
    else:
        print("PASS (a-weff): W_eff matches Cayley oracle, max|Δ|=", weff_mx)

    var y_impl = oft_forward(x, lo, M)
    var y_oracle = _oracle_fwd(x, weff_oracle, M, IN, OUT)
    var y_mx = _max_abs_diff(y_impl, y_oracle)
    if y_mx > Float32(1.0e-5):
        print("FAIL (a-fwd): y vs oracle max|Δ|=", y_mx); ok = False
    else:
        print("PASS (a-fwd): forward y matches oracle, max|Δ|=", y_mx)

    # ── (a) backward: analytic d_S vs finite-difference ──
    var d_y = List[Float32]()
    for _ in range(M * OUT):
        d_y.append(Float32(1.0))
    var g = oft_backward(d_y, x, lo, M)
    var h = Float32(1.0e-3)
    var tol_fd = Float32(2.0e-2)
    var tol_rel = Float32(2.0e-2)
    var rel_floor = Float32(1.0e-4)

    var fd_s = _fd_grad_s(s_h, lo.w_base, x, NB, B, IN, OUT, M, h)
    var e_s = _max_abs_diff(g.d_s, fd_s)
    var er_s = _max_rel_diff(g.d_s, fd_s, rel_floor)
    if e_s > tol_fd or er_s > tol_rel:
        print("FAIL (a-bwd S): analytic vs FD abs|Δ|=", e_s, " rel=", er_s); ok = False
    else:
        print("PASS (a-bwd S): max|Δ| vs FD=", e_s, " rel=", er_s)

    # ── (b) GRAD-FLOW over S ──
    var lo_b = lo.copy()
    oft_adamw(lo_b, g, 1, Float32(1.0e-3))
    var g2 = oft_backward(d_y, x, lo_b, M)

    var names = List[String]()
    var grads = List[TArc]()
    var s_maybe = g2.d_s.copy()
    if break_factor:
        for i in range(len(s_maybe)):
            s_maybe[i] = Float32(0.0)
        print("INFO: OFT_BREAK_FACTOR set — zeroing oft_blocks(S) grad to prove the gate catches it")
    names.append(String("toy.oft_blocks"))
    grads.append(TArc(Tensor.from_host(s_maybe^, [NB, B * B], STDtype.F32, ctx)))

    var rep = measure(names, grads, ctx)
    print("[grad-flow] total=", rep.total, " nonzero=", rep.nonzero,
          " dead=", rep.dead, " coverage=", rep.coverage_pct(), "%")

    if break_factor:
        if rep.dead == 0:
            print("FAIL (b-demo): broken S grad NOT detected (dead==0)"); ok = False
        else:
            print("PASS (b-demo): dead S grad detected, dead=", rep.dead)
        if armed:
            raise Error(
                String("[grad-flow] OFT FAILED: ") + String(rep.dead)
                + " of " + String(rep.total) + " trainable params DEAD — gate aborts (exit != 0)"
            )
    else:
        if rep.dead != 0:
            print("FAIL (b): OFT S grad is DEAD (dead=", rep.dead, ")"); ok = False
        else:
            print("PASS (b): OFT S grad nonzero, dead=0")
        if rep.coverage_pct() != Float64(100.0):
            print("FAIL (b): coverage_pct != 100 (got", rep.coverage_pct(), ")"); ok = False
        else:
            print("PASS (b): coverage_pct == 100")

    # ── (c) SAVE round-trip ──
    var named = List[NamedOFT]()
    named.append(NamedOFT(String("double_blocks.0.img_attn.to_q"), lo.copy()))
    var path = String("/tmp/oft_smoke_") + label + String(".safetensors")
    var n_written = save_oft_peft(named, path, ctx)
    if n_written != 1:
        print("FAIL (c): save returned", n_written, "adapters, expected 1"); ok = False
    else:
        print("PASS (c): saved 1 OFT adapter to", path)

    var rb = read_oft_module(String("double_blocks.0.img_attn.to_q"), path, ctx)
    if rb.num_blocks != NB or rb.block_size != B:
        print("FAIL (c): shape mismatch got nb=", rb.num_blocks, " b=", rb.block_size,
              " expected nb=", NB, " b=", B); ok = False
    else:
        print("PASS (c): shapes round-trip num_blocks=", rb.num_blocks, " block_size=", rb.block_size)
    if abs(rb.alpha - alpha) > Float32(1.0e-6):
        print("FAIL (c): alpha mismatch got", rb.alpha, "expected", alpha); ok = False
    else:
        print("PASS (c): alpha round-trip =", rb.alpha)
    var s_mx = _max_abs_diff_bf16(rb.s, lo.s)
    if s_mx > Float32(1.0e-6):
        print("FAIL (c): oft_blocks(S) not byte-exact, Δ=", s_mx); ok = False
    else:
        print("PASS (c): oft_blocks(S) values round-trip byte-exact, Δ=", s_mx)

    return ok


def main() raises:
    var ctx = DeviceContext()
    var armed = _env_is_set(String("FLAME_ASSERT_GRAD_FLOW"))
    var break_factor = _env_is_set(String("OFT_BREAK_FACTOR"))

    # block_size=2: out=6 → num_blocks=3 blocks of 2×2 (cheap exact inverse).
    var ok1 = _run_config(String("b2"), 5, 6, 2, break_factor, armed, ctx)
    # block_size=4: out=8 → num_blocks=2 blocks of 4×4 (general inverse path).
    # break only armed-aborts in the first config; this one runs clean.
    var ok2 = _run_config(String("b4"), 7, 8, 4, False, armed, ctx)

    if not (ok1 and ok2):
        raise Error("oft_adapter_smoke FAILED")
    print("oft_adapter_smoke ALL GATES PASS (b2 + b4)")
