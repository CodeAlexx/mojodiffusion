# training/boft_adapter_smoke.mojo — BOFT adapter gates (a)+(b)+(c)+(d).
#
# TENET 4 / parity-bitrot guard: every check ASSERTS and the file `raise`s
# (nonzero exit) on any wrong value. Demonstrated wrong-run: set env
# BOFT_BREAK_FACTOR=1 to zero ONE stage's S grad — with FLAME_ASSERT_GRAD_FLOW=1
# the grad-flow gate then ABORTS (exit != 0).
#
# GATE (a) PARITY:
#   - identity-at-init: S=0 → every R_i=I → T=I → W_eff==W (max|Δ|==0).
#   - orthogonality: after driving S off zero, the OVERALL transform TᵀT ≈ I.
#   - forward parity: y = x @ W_effᵀ vs an INDEPENDENT open-coded recompute of
#     the whole butterfly product (skew → Cayley via independent inverse →
#     block-diag R → Pᵀ R P → product → T@W → y), NOT the boft_adapter helpers.
#   - FD parity: analytic d_S vs central finite-difference of loss=sum(y) over
#     EVERY element of S across ALL stages (proves the reverse-order analytic
#     backward through the butterfly product is correct for every factor param).
# GATE (b) GRAD-FLOW: after one AdamW step (S off zero), fresh fwd+bwd, feed
#   EACH STAGE's d_S in as a separate named grad → coverage_pct==100, dead==0.
#   The BOFT_BREAK_FACTOR demo proves a dead stage grad is caught.
# GATE (c) SAVE: save → reopen → assert oft_blocks 4D [boft_m,nb,b,b] + alpha +
#   byte-exact S values.
# GATE (d) default-off + trainer AOT-build verified out-of-band (trainer fails
#   loud on adapter_algo==6); see the builder report.
#
# Run (clean PASS, exit 0):
#   rm -f serenitymojo.mojopkg && pixi run mojo run -I . \
#     serenitymojo/training/boft_adapter_smoke.mojo
# Run (deliberate FAIL, exit != 0):
#   FLAME_ASSERT_GRAD_FLOW=1 BOFT_BREAK_FACTOR=1 pixi run mojo run -I . \
#     serenitymojo/training/boft_adapter_smoke.mojo

from std.collections import List
from std.math import abs
from std.ffi import external_call
from std.memory import alloc, UnsafePointer, ArcPointer
from std.builtin.type_aliases import MutExternalOrigin
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.boft_adapter import (
    BOFTAdapter,
    new_boft_adapter,
    boft_stage_count,
    boft_transform,
    boft_effective_weight,
    boft_forward,
    boft_backward,
    boft_adamw,
)
from serenitymojo.training.boft_save import (
    NamedBOFT,
    save_boft_peft,
    read_boft_module,
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


# ── INDEPENDENT oracle: full butterfly product, open-coded (distinct code) ────
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


def _o_matmul(a: List[Float32], n: Int, b: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        for j in range(n):
            var acc = Float32(0.0)
            for k in range(n):
                acc += a[i * n + k] * b[k * n + j]
            out.append(acc)
    return out^


def _o_transpose(a: List[Float32], n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n * n):
        out.append(Float32(0.0))
    for i in range(n):
        for j in range(n):
            out[j * n + i] = a[i * n + j]
    return out^


def _o_perm(out_f: Int, b: Int, stage: Int) -> List[Float32]:
    var g = 2
    var k = (1 << stage) * (b // 2)
    var gk = g * k
    var P = List[Float32]()
    for _ in range(out_f * out_f):
        P.append(Float32(0.0))
    if gk == 0 or out_f % gk != 0:
        for i in range(out_f):
            P[i * out_f + i] = Float32(1.0)
        return P^
    var c = out_f // gk
    for cc in range(c):
        for gg in range(g):
            for kk in range(k):
                var old_idx = (cc * g + gg) * k + kk
                var new_idx = (cc * k + kk) * g + gg
                P[new_idx * out_f + old_idx] = Float32(1.0)
    return P^


def _o_block_r(sblk: List[Float32], b: Int) raises -> List[Float32]:
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
    var R = List[Float32]()
    for i in range(b):
        for j in range(b):
            var acc = Float32(0.0)
            for k in range(b):
                acc += minv[i * b + k] * aminus[k * b + j]
            R.append(acc)
    return R^


# Independent total transform T from S, distinct code path from the adapter.
def _oracle_transform(s: List[Float32], MM: Int, NB: Int, B: Int, OUT: Int) raises -> List[Float32]:
    var T = List[Float32]()
    for i in range(OUT * OUT):
        T.append(Float32(1.0) if (i // OUT) == (i % OUT) else Float32(0.0))  # I
    for stage in range(MM):
        # block-diagonal R_stage
        var R = List[Float32]()
        for _ in range(OUT * OUT):
            R.append(Float32(0.0))
        var stage_base = stage * NB * B * B
        for grp in range(NB):
            var sblk = List[Float32]()
            var sb = stage_base + grp * B * B
            for q in range(B * B):
                sblk.append(s[sb + q])
            var rg = _o_block_r(sblk, B)
            var base = grp * B
            for ii in range(B):
                for jj in range(B):
                    R[(base + ii) * OUT + (base + jj)] = rg[ii * B + jj]
        var P = _o_perm(OUT, B, stage)
        var Pt = _o_transpose(P, OUT)
        var PtR = _o_matmul(Pt, OUT, R)
        var Bmat = _o_matmul(PtR, OUT, P)        # Pᵀ R P
        T = _o_matmul(Bmat, OUT, T)              # T ← B_stage @ T
    return T^


def _oracle_weff(s: List[Float32], w_base: List[Float32], MM: Int, NB: Int, B: Int, IN: Int, OUT: Int) raises -> List[Float32]:
    var T = _oracle_transform(s, MM, NB, B, OUT)
    # W_eff = T @ W
    var out = List[Float32]()
    for i in range(OUT):
        for c in range(IN):
            var acc = Float32(0.0)
            for k in range(OUT):
                acc += T[i * OUT + k] * w_base[k * IN + c]
            out.append(acc)
    return out^


def _oracle_fwd(x: List[Float32], w_eff: List[Float32], M: Int, IN: Int, OUT: Int) -> List[Float32]:
    var out = List[Float32]()
    for mm in range(M):
        for o in range(OUT):
            var s = Float32(0.0)
            for i in range(IN):
                s += x[mm * IN + i] * w_eff[o * IN + i]
            out.append(s)
    return out^


def _loss_for_fd(s: List[Float32], w_base: List[Float32], x: List[Float32],
                 MM: Int, NB: Int, B: Int, IN: Int, OUT: Int, M: Int) raises -> Float32:
    var w_eff = _oracle_weff(s, w_base, MM, NB, B, IN, OUT)
    var y = _oracle_fwd(x, w_eff, M, IN, OUT)
    var s_sum = Float32(0.0)
    for i in range(len(y)):
        s_sum += y[i]
    return s_sum


def _fd_grad_s(s: List[Float32], w_base: List[Float32], x: List[Float32],
               MM: Int, NB: Int, B: Int, IN: Int, OUT: Int, M: Int, h: Float32) raises -> List[Float32]:
    var out = List[Float32]()
    for k in range(len(s)):
        var sp = s.copy(); var sm = s.copy()
        sp[k] = sp[k] + h; sm[k] = sm[k] - h
        var lp = _loss_for_fd(sp, w_base, x, MM, NB, B, IN, OUT, M)
        var lm = _loss_for_fd(sm, w_base, x, MM, NB, B, IN, OUT, M)
        out.append((lp - lm) / (Float32(2.0) * h))
    return out^


def _max_ttt_minus_i(T: List[Float32], OUT: Int) raises -> Float32:
    var tt = _o_transpose(T, OUT)
    var mx = Float32(0.0)
    for i in range(OUT):
        for j in range(OUT):
            var acc = Float32(0.0)
            for k in range(OUT):
                acc += tt[i * OUT + k] * T[k * OUT + j]
            var id = Float32(1.0) if i == j else Float32(0.0)
            var d = abs(acc - id)
            if d > mx:
                mx = d
    return mx


def _run_config(label: String, in_f: Int, out_f: Int, block_size: Int,
                break_factor: Bool, armed: Bool, ctx: DeviceContext) raises -> Bool:
    var ok = True
    var stages = boft_stage_count(out_f // block_size)
    print("=== config", label, "(in=", in_f, " out=", out_f, " block_size=", block_size,
          " → boft_m=", stages, ") ===")

    var alpha = Float32(1.0)
    var w_base = List[Float32]()
    for i in range(out_f * in_f):
        w_base.append(Float32(0.05) * Float32(((i * 7) % 13) - 6) + Float32(0.3))

    var lo = new_boft_adapter(in_f, out_f, block_size, alpha, w_base.copy())
    var MM = lo.boft_m; var NB = lo.num_blocks; var B = lo.block_size
    var IN = lo.in_f; var OUT = lo.out_f; var M = 3

    if MM < 2:
        print("FAIL (cfg): boft_m must be >= 2 to exercise the butterfly product, got", MM); ok = False
    else:
        print("PASS (cfg): boft_m=", MM, " stages (num_blocks=", NB, ")")

    # ── (a) identity-at-init: S=0 → T=I → W_eff == W ──
    var weff0 = boft_effective_weight(lo)
    var init_mx = _max_abs_diff(weff0, lo.w_base)
    if init_mx != Float32(0.0):
        print("FAIL (a-init): W_eff != W_base at init, max|Δ|=", init_mx); ok = False
    else:
        print("PASS (a-init): W_eff == W_base at init (T=I, zero S)")

    var T0 = boft_transform(lo)
    var ttt0 = _max_ttt_minus_i(T0, OUT)
    if ttt0 > Float32(1.0e-5):
        print("FAIL (a-init): TᵀT != I at init, max|Δ|=", ttt0); ok = False
    else:
        print("PASS (a-init): TᵀT == I at init, max|Δ|=", ttt0)

    # Drive S off zero across all stages so every factor is exercised.
    for i in range(len(lo.s)):
        lo.s[i] = Float32(0.06) * Float32((i % 7) - 3) + Float32(0.04)

    # ── (a) orthogonality of overall transform: TᵀT ≈ I ──
    var T = boft_transform(lo)
    var ttt = _max_ttt_minus_i(T, OUT)
    if ttt > Float32(5.0e-5):
        print("FAIL (a-orth): TᵀT != I, max|TᵀT - I|=", ttt); ok = False
    else:
        print("PASS (a-orth): TᵀT == I (product of orthogonals), max|TᵀT - I|=", ttt)

    # independent-oracle transform cross-check
    var T_oracle = _oracle_transform(lo.s, MM, NB, B, OUT)
    var T_mx = _max_abs_diff(T, T_oracle)
    if T_mx > Float32(1.0e-5):
        print("FAIL (a-T): transform vs oracle max|Δ|=", T_mx); ok = False
    else:
        print("PASS (a-T): transform matches butterfly oracle, max|Δ|=", T_mx)

    var x = List[Float32]()
    for i in range(M * IN):
        x.append(Float32(0.1) * Float32(((i * 3) % 11) - 5))

    # ── (a) W_eff + forward parity vs independent oracle ──
    var weff_impl = boft_effective_weight(lo)
    var weff_oracle = _oracle_weff(lo.s, lo.w_base, MM, NB, B, IN, OUT)
    var weff_mx = _max_abs_diff(weff_impl, weff_oracle)
    if weff_mx > Float32(1.0e-5):
        print("FAIL (a-weff): W_eff vs oracle max|Δ|=", weff_mx); ok = False
    else:
        print("PASS (a-weff): W_eff matches oracle, max|Δ|=", weff_mx)

    var y_impl = boft_forward(x, lo, M)
    var y_oracle = _oracle_fwd(x, weff_oracle, M, IN, OUT)
    var y_mx = _max_abs_diff(y_impl, y_oracle)
    if y_mx > Float32(1.0e-5):
        print("FAIL (a-fwd): y vs oracle max|Δ|=", y_mx); ok = False
    else:
        print("PASS (a-fwd): forward y matches oracle, max|Δ|=", y_mx)

    # ── (a) backward: analytic d_S vs finite-difference (ALL stages) ──
    var d_y = List[Float32]()
    for _ in range(M * OUT):
        d_y.append(Float32(1.0))
    var g = boft_backward(d_y, x, lo, M)
    var h = Float32(1.0e-3)
    var tol_fd = Float32(2.0e-2)
    var tol_rel = Float32(2.0e-2)
    var rel_floor = Float32(1.0e-4)

    var fd_s = _fd_grad_s(lo.s, lo.w_base, x, MM, NB, B, IN, OUT, M, h)
    var e_s = _max_abs_diff(g.d_s, fd_s)
    var er_s = _max_rel_diff(g.d_s, fd_s, rel_floor)
    if e_s > tol_fd or er_s > tol_rel:
        print("FAIL (a-bwd S): analytic vs FD abs|Δ|=", e_s, " rel=", er_s); ok = False
    else:
        print("PASS (a-bwd S): max|Δ| vs FD=", e_s, " rel=", er_s, " (all", MM, "stages)")

    # ── (b) GRAD-FLOW: one named grad PER STAGE ──
    var lo_b = lo.copy()
    boft_adamw(lo_b, g, 1, Float32(1.0e-3))
    var g2 = boft_backward(d_y, x, lo_b, M)

    var names = List[String]()
    var grads = List[TArc]()
    var per_stage = NB * B * B
    for stage in range(MM):
        var sg = List[Float32]()
        var base = stage * per_stage
        for i in range(per_stage):
            sg.append(g2.d_s[base + i])
        # DELIBERATE-WRONG: zero stage 0's grad if armed.
        if break_factor and stage == 0:
            for i in range(len(sg)):
                sg[i] = Float32(0.0)
            print("INFO: BOFT_BREAK_FACTOR set — zeroing stage-0 oft_blocks grad to prove the gate catches it")
        names.append(String("toy.oft_blocks.stage") + String(stage))
        grads.append(TArc(Tensor.from_host(sg^, [NB, B * B], STDtype.F32, ctx)))

    var rep = measure(names, grads, ctx)
    print("[grad-flow] total=", rep.total, " nonzero=", rep.nonzero,
          " dead=", rep.dead, " coverage=", rep.coverage_pct(), "%")

    if break_factor:
        if rep.dead == 0:
            print("FAIL (b-demo): broken stage grad NOT detected (dead==0)"); ok = False
        else:
            print("PASS (b-demo): dead stage grad detected, dead=", rep.dead)
        if armed:
            raise Error(
                String("[grad-flow] BOFT FAILED: ") + String(rep.dead)
                + " of " + String(rep.total) + " stage grads DEAD — gate aborts (exit != 0)"
            )
    else:
        if rep.dead != 0:
            print("FAIL (b): a BOFT stage grad is DEAD (dead=", rep.dead, ")"); ok = False
        else:
            print("PASS (b): all", MM, "BOFT stage grads nonzero, dead=0")
        if rep.coverage_pct() != Float64(100.0):
            print("FAIL (b): coverage_pct != 100 (got", rep.coverage_pct(), ")"); ok = False
        else:
            print("PASS (b): coverage_pct == 100 (all stages)")

    # ── (c) SAVE round-trip ──
    var named = List[NamedBOFT]()
    named.append(NamedBOFT(String("double_blocks.0.img_attn.to_q"), lo.copy()))
    var path = String("/tmp/boft_smoke_") + label + String(".safetensors")
    var n_written = save_boft_peft(named, path, ctx)
    if n_written != 1:
        print("FAIL (c): save returned", n_written, "adapters, expected 1"); ok = False
    else:
        print("PASS (c): saved 1 BOFT adapter to", path)

    var rb = read_boft_module(String("double_blocks.0.img_attn.to_q"), path, ctx)
    if rb.boft_m != MM or rb.num_blocks != NB or rb.block_size != B:
        print("FAIL (c): shape mismatch got m=", rb.boft_m, " nb=", rb.num_blocks, " b=", rb.block_size,
              " expected m=", MM, " nb=", NB, " b=", B); ok = False
    else:
        print("PASS (c): shapes round-trip boft_m=", rb.boft_m, " num_blocks=", rb.num_blocks, " block_size=", rb.block_size)
    if abs(rb.alpha - alpha) > Float32(1.0e-6):
        print("FAIL (c): alpha mismatch got", rb.alpha, "expected", alpha); ok = False
    else:
        print("PASS (c): alpha round-trip =", rb.alpha)
    var s_mx = _max_abs_diff(rb.s, lo.s)
    if s_mx > Float32(1.0e-6):
        print("FAIL (c): oft_blocks(S) not byte-exact, Δ=", s_mx); ok = False
    else:
        print("PASS (c): oft_blocks(S) values round-trip byte-exact, Δ=", s_mx)

    return ok


def main() raises:
    var ctx = DeviceContext()
    var armed = _env_is_set(String("FLAME_ASSERT_GRAD_FLOW"))
    var break_factor = _env_is_set(String("BOFT_BREAK_FACTOR"))

    # block_size=2, out=8 → num_blocks=4 → boft_m=popcount(3)+1=3 stages.
    var ok1 = _run_config(String("b2x4"), 5, 8, 2, break_factor, armed, ctx)
    # block_size=2, out=4 → num_blocks=2 → boft_m=popcount(1)+1=2 stages.
    var ok2 = _run_config(String("b2x2"), 6, 4, 2, False, armed, ctx)

    if not (ok1 and ok2):
        raise Error("boft_adapter_smoke FAILED")
    print("boft_adapter_smoke ALL GATES PASS (b2x4 + b2x2)")
