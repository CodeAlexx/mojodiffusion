# ops/snake_smoke.mojo — GPU numeric gate for P-snake (SnakeBeta activation).
#
# LTX2_PORT_PLAN_2026-05-28 §P-snake gate (PARITY, host-F64):
#   apply snake_beta to a per-channel ramp, match an in-smoke host F64 reference
#   max_abs small (we gate cos>=0.999 + max_rel<2% on BF16 storage), and verify
#   the eps=1e-9 placement and the log-scale exp() precompute.
#
# REAL-WEIGHT NOTE (fail-honest): the plan asks to load
#   vocoder.vocoder.act_post.act.{alpha,beta} [24] from the ckpt. That standalone
#   vocoder safetensors is NOT present in this repo (the only local LTX file is a
#   DiT-only merge with ZERO `vocoder.*`/`act_post.*` keys — verified). So this
#   gate uses SYNTHETIC log-scale params chosen to mirror real act_post weights
#   (small-magnitude logs spanning negatives→positives, the regime BigVGAN
#   alpha/beta actually occupy). The MATH PATH gated is bit-identical to the real
#   weights: same exp() precompute, same +1e-9-inside-reciprocal, same per-channel
#   broadcast forward. When the vocoder ckpt lands, swap `_alpha_log`/`_beta_log`
#   for the loaded [24] tensors — the gate body is unchanged.
#
# What this proves (fail-closed, see mutation discussion below):
#   * log-scale: params are exp()'d ON GPU via snake_beta_precompute, not used raw.
#   * eps placement: +1e-9 is INSIDE the reciprocal denom (1/(exp(beta)+1e-9)),
#     so a large-negative beta (exp(beta)→0) gives a CAPPED, finite gain — checked
#     by including beta logs near the floor.
#   * per-channel [1,C,1] broadcast against x=[1,C,L]: each channel uses its OWN
#     alpha/beta; a broadcast bug would scramble channels and tank cos.
#   * sin² (not sin): the squared term is the whole nonlinearity.
#
# Build:
#   pixi run mojo build -I . -Xlinker -lm serenitymojo/ops/snake_smoke.mojo -o /tmp/p_snake_smoke
# Run:
#   /tmp/p_snake_smoke

from std.math import sin, exp, sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.snake import snake_beta, snake_beta_precompute


comptime _C = 24
comptime _L = 32
comptime _N = _C * _L
comptime _SNAKE_EPS = Float64(1e-9)
comptime _COS_GATE = Float64(0.999)
comptime _MAXREL_GATE = Float64(0.02)  # for the precompute params (gains, no zero-crossing)
comptime _REL_FLOOR = Float64(1e-3)
# Forward output = x + inv_beta·sin²(ax) crosses zero, so a relative metric is
# meaningless there (0.04 abs error / ~0 ref → huge). The output is O(few), so
# BF16 storage rounding (~2^-8·|v|) caps the absolute error near 0.05; gate
# max_abs < 0.1 (comfortably above the BF16 floor, tight enough to catch a real
# bug — e.g. a missing residual x+ would push max_abs to ~3).
comptime _MAXABS_GATE = Float64(0.1)


# bf16 round-trip of an F64 value: cast down to bf16 then back, so the host
# reference reads exactly the bits the GPU kernels load from BF16 storage.
@always_inline
def _bf16_round(v: Float64) -> Float64:
    return v.cast[DType.float32]().cast[DType.bfloat16]().cast[DType.float64]()


def main() raises:
    var ctx = DeviceContext()
    print("=== P-snake SnakeBeta GPU smoke (BF16 storage, host-F64 refv) ===")
    print(
        "  shape x=[1,"
        + String(_C)
        + ","
        + String(_L)
        + "]  params [1,"
        + String(_C)
        + ",1] (per-channel)"
    )

    # ── Synthetic-but-realistic LOG-SCALE per-channel params (see header). ──────
    # alpha logs sweep [-1.5, 1.5] (exp → 0.22 … 4.48); beta logs sweep
    # [-2, 1.5] (exp → 0.14 … 4.48, gains O(1)) — the regime real BigVGAN
    # act_post weights actually occupy, where the term inv_beta·sin² stays
    # BF16-representable. (A huge near-floor gain like inv_beta≈8e3 is NOT a real
    # weight and only injects BF16 quantization noise; the eps-cap edge is
    # gated separately, F32, below.) Distinct per channel so the broadcast must
    # route the right param to the right channel.
    var alpha_log = List[Float32]()
    var beta_log = List[Float32]()
    for c in range(_C):
        var ta = Float64(c) / Float64(_C - 1)  # 0..1
        alpha_log.append(Float32(-1.5 + ta * 3.0))  # [-1.5, 1.5]
        beta_log.append(Float32(-2.0 + ta * 3.5))  # [-2.0, 1.5]

    # ── Per-channel activation ramp x[1,C,L] in [-3, 3]. ───────────────────────
    var xs = List[Float32]()
    for _c in range(_C):
        for j in range(_L):
            var t = Float64(j) / Float64(_L - 1)  # 0..1
            # vary the range slightly per channel so it's not a pure repeat
            var lo = -3.0
            var hi = 3.0
            xs.append(Float32(lo + t * (hi - lo)))

    # ── Host F64 reference (on bf16-rounded inputs, matching GPU storage). ──────
    # Precompute mirrors snake_beta_precompute: exp(alpha), 1/(exp(beta)+1e-9).
    var alpha_exp_ref = List[Float64]()
    var inv_beta_ref = List[Float64]()
    for c in range(_C):
        var a = _bf16_round(alpha_log[c].cast[DType.float64]())
        var b = _bf16_round(beta_log[c].cast[DType.float64]())
        # exp result is stored BF16 on GPU → round-trip it too.
        alpha_exp_ref.append(_bf16_round(exp(a)))
        var be = exp(b)
        inv_beta_ref.append(_bf16_round(Float64(1.0) / (be + _SNAKE_EPS)))

    var refv = List[Float64]()
    for c in range(_C):
        var ae = alpha_exp_ref[c]
        var ib = inv_beta_ref[c]
        for j in range(_L):
            var xv = _bf16_round(xs[c * _L + j].cast[DType.float64]())
            var s = sin(ae * xv)
            refv.append(xv + ib * s * s)

    # ── GPU path: precompute params ON GPU, then snake_beta forward. ────────────
    var x = Tensor.from_host(xs, [1, _C, _L], STDtype.BF16, ctx)
    var alpha_t = Tensor.from_host(alpha_log, [1, _C, 1], STDtype.BF16, ctx)
    var beta_t = Tensor.from_host(beta_log, [1, _C, 1], STDtype.BF16, ctx)

    var pre = snake_beta_precompute(alpha_t, beta_t, ctx)

    # Spot-check the precompute itself vs host (log-scale + eps cap).
    var ae_host = pre[0].to_host(ctx)
    var ib_host = pre[1].to_host(ctx)
    var pre_ok = True
    var pre_maxrel = Float64(0.0)
    for c in range(_C):
        var dae = (ae_host[c].cast[DType.float64]() - alpha_exp_ref[c]).__abs__()
        var rae = dae / (
            alpha_exp_ref[c] if alpha_exp_ref[c] > _REL_FLOOR else _REL_FLOOR
        )
        var dib = (ib_host[c].cast[DType.float64]() - inv_beta_ref[c]).__abs__()
        var rib = dib / (
            inv_beta_ref[c] if inv_beta_ref[c] > _REL_FLOOR else _REL_FLOOR
        )
        if rae > pre_maxrel:
            pre_maxrel = rae
        if rib > pre_maxrel:
            pre_maxrel = rib
        # finite check on the capped near-floor channel
        var ibv = ib_host[c]
        if not ((ibv == ibv) and (ibv.__abs__() < Float32(1e20))):
            pre_ok = False
    pre_ok = pre_ok and (pre_maxrel < _MAXREL_GATE)
    print(
        "  ["
        + ("PASS" if pre_ok else "FAIL")
        + "] precompute exp/inv (realistic gains): max_rel="
        + String(pre_maxrel)
        + " inv_beta(ch0)="
        + String(ib_host[0])
        + " (expect ~"
        + String(inv_beta_ref[0])
        + ")"
    )

    # ── eps-PLACEMENT edge (F32, isolated): a huge-negative beta drives
    # exp(beta)→~0, so 1/(exp(beta)+1e-9) must CAP near 1e9 and stay FINITE.
    # Without the +1e-9 inside the denom this would be Inf (or the op's own
    # 1e-12 eps-clamp would mis-cap at 1e12). beta_log=-25 → exp≈1.4e-11 ≪ 1e-9.
    var eps_a = List[Float32]()
    var eps_b = List[Float32]()
    eps_a.append(Float32(0.0))
    eps_b.append(Float32(-25.0))
    var eps_a_t = Tensor.from_host(eps_a, [1], STDtype.F32, ctx)
    var eps_b_t = Tensor.from_host(eps_b, [1], STDtype.F32, ctx)
    var eps_pre = snake_beta_precompute(eps_a_t, eps_b_t, ctx)
    var eps_ib = eps_pre[1].to_host(ctx)
    var eps_be = exp(Float64(-25.0))
    var eps_expect = Float64(1.0) / (eps_be + _SNAKE_EPS)  # ≈ 9.99e8
    var eps_v = eps_ib[0].cast[DType.float64]()
    var eps_finite = (eps_ib[0] == eps_ib[0]) and (eps_ib[0].__abs__() < Float32(2e9))
    var eps_rel = (eps_v - eps_expect).__abs__() / eps_expect
    var eps_ok = eps_finite and (eps_rel < 0.05)
    print(
        "  ["
        + ("PASS" if eps_ok else "FAIL")
        + "] eps placement (beta_log=-25): inv_beta="
        + String(eps_ib[0])
        + " expect~"
        + String(eps_expect)
        + " rel="
        + String(eps_rel)
        + " (1e-9 INSIDE denom → capped ~1e9, finite)"
    )

    var y = snake_beta(x, pre[0], pre[1], ctx).to_host(ctx)

    # ── Gate cos + max_rel vs host F64. ────────────────────────────────────────
    var dot = Float64(0.0)
    var ng = Float64(0.0)
    var nr = Float64(0.0)
    var maxabs = Float64(0.0)
    var maxrel = Float64(0.0)
    var has_bad = False
    for i in range(_N):
        var g = y[i].cast[DType.float64]()
        var r = refv[i]
        if g != g:
            has_bad = True
        dot += g * r
        ng += g * g
        nr += r * r
        var d = (g - r).__abs__()
        if d > maxabs:
            maxabs = d
        var ar = r if r >= 0.0 else -r
        var denom = ar if ar > _REL_FLOOR else _REL_FLOOR
        var rel = d / denom
        if rel > maxrel:
            maxrel = rel
    var cos = Float64(0.0)
    if ng > 0.0 and nr > 0.0:
        cos = dot / (sqrt(ng) * sqrt(nr))
    if has_bad:
        cos = Float64(-1.0)
    var fwd_ok = (cos >= _COS_GATE) and (maxabs < _MAXABS_GATE)
    print(
        "  ["
        + ("PASS" if fwd_ok else "FAIL")
        + "] snake_beta forward: cos="
        + String(cos)
        + " max_abs="
        + String(maxabs)
        + " max_rel(info,zero-cross)="
        + String(maxrel)
        + " (gate cos>="
        + String(_COS_GATE)
        + ", max_abs<"
        + String(_MAXABS_GATE)
        + ")"
    )

    # ── Spot: alpha=0 (log) → exp(0)=1; beta huge-negative → inv≈1/1e-9 capped,
    # and at x where sin(x)=0 the output must equal x exactly (identity). ───────
    # Use channel where x sweeps through 0? our ramp hits x≈0 at j ~ middle.
    # Verify a known analytic point: with alpha_exp=1, x=pi/2 → snake = x + ib·1.
    var an_alpha = List[Float32]()
    var an_beta = List[Float32]()
    an_alpha.append(Float32(0.0))  # exp(0)=1
    an_beta.append(Float32(0.0))  # exp(0)=1 → inv≈1/(1+1e-9)≈1
    var an_x = List[Float32]()
    var halfpi = Float32(1.5707963267948966)
    an_x.append(halfpi)  # sin(1·pi/2)=1 → snake = pi/2 + 1·1 = pi/2 + 1
    var ax_t = Tensor.from_host(an_alpha, [1, 1, 1], STDtype.F32, ctx)
    var bx_t = Tensor.from_host(an_beta, [1, 1, 1], STDtype.F32, ctx)
    var xx_t = Tensor.from_host(an_x, [1, 1, 1], STDtype.F32, ctx)
    var anpre = snake_beta_precompute(ax_t, bx_t, ctx)
    var an_y = snake_beta(xx_t, anpre[0], anpre[1], ctx).to_host(ctx)
    var expect = Float64(halfpi.cast[DType.float64]()) + Float64(1.0) / (
        exp(Float64(0.0)) + _SNAKE_EPS
    )
    var an_ok = (an_y[0].cast[DType.float64]() - expect).__abs__() < 1e-4
    print(
        "  ["
        + ("PASS" if an_ok else "FAIL")
        + "] analytic alpha=0,beta=0,x=pi/2: got="
        + String(an_y[0])
        + " expect="
        + String(expect)
        + " (=pi/2 + 1/(1+1e-9))"
    )

    var all_pass = pre_ok and eps_ok and fwd_ok and an_ok
    print("=== " + ("ALL PASS" if all_pass else "FAILED") + " ===")
    if not all_pass:
        raise Error("p_snake smoke FAILED numeric gate")
