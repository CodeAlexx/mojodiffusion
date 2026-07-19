# training/parity/schedule_parity.mojo — training-loop policy parity gate (T6).
#
# Gates the numeric policy primitives in training/schedule.mojo against host F64
# references (the host computation IS the oracle — same discipline as
# optim_parity.mojo), plus a statistical check on the RNG-driven timestep.
#
# Gates:
#   flow_match_noise_target : x_t & target each cos >= 0.999 vs host F64
#   ema_update              : cos >= 0.999 vs decay*shadow+(1-decay)*live (F64)
#   grad_accumulate         : cos >= 0.999 vs host sum
#   sample_timestep_logit_normal : STATISTICAL — draw N=100000, assert mean/std
#                            in-range for sigmoid(N(0,1)) (NOT a cos gate).
#
# Run:
#   cd /home/alex/mojodiffusion && (rm -f serenitymojo.mojopkg) && \
#     pixi run mojo run -I . serenitymojo/training/parity/schedule_parity.mojo
#
# Mojo 1.0.0b1. Loud-fail: prints FAIL + raises on any arm under threshold.

from std.gpu.host import DeviceContext
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.schedule import (
    flow_match_noise_target,
    ema_update,
    grad_accumulate,
    sample_timestep_logit_normal,
)


def _cos(a: List[Float64], b: List[Float64]) -> Float64:
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    return dot / (sqrt(na) * sqrt(nb))


def _to_f64(t: Tensor, ctx: DeviceContext) raises -> List[Float64]:
    var host = t.to_host(ctx)
    var out = List[Float64]()
    for i in range(len(host)):
        out.append(Float64(host[i]))
    return out^


def _f32_tensor(vals: List[Float64], ctx: DeviceContext) raises -> Tensor:
    var host = List[Float32]()
    for i in range(len(vals)):
        host.append(Float32(vals[i]))
    var sh = List[Int]()
    sh.append(len(vals))
    return Tensor.from_host(host, sh^, STDtype.F32, ctx)


def main() raises:
    var ctx = DeviceContext()
    print("=== schedule parity (T6) ===")

    # Fixed (latent, noise) tuple — deterministic, spread across sign/scale.
    var n = 256
    var latent_vals = List[Float64]()
    var noise_vals = List[Float64]()
    for i in range(n):
        latent_vals.append(Float64(i) * 0.013 - 1.6)
        noise_vals.append(Float64(n - i) * 0.009 - 1.1)

    # ── Arm 1: flow_match_noise_target ──────────────────────────────────────
    var sigma = Float32(0.3741)
    var latent = _f32_tensor(latent_vals, ctx)
    var noise = _f32_tensor(noise_vals, ctx)
    var fm = flow_match_noise_target(latent, sigma, noise, ctx)
    var got_xt = _to_f64(fm.x_t, ctx)
    var got_tg = _to_f64(fm.target, ctx)

    var want_xt = List[Float64]()
    var want_tg = List[Float64]()
    var s64 = Float64(sigma)
    for i in range(n):
        want_xt.append((Float64(1.0) - s64) * latent_vals[i] + s64 * noise_vals[i])
        want_tg.append(noise_vals[i] - latent_vals[i])

    var c_xt = _cos(got_xt, want_xt)
    var c_tg = _cos(got_tg, want_tg)
    print("flow_match x_t cos    =", c_xt)
    print("flow_match target cos =", c_tg)
    if c_xt < Float64(0.999):
        print("  FAIL flow_match x_t <0.999")
        raise Error("flow_match x_t parity fail")
    if c_tg < Float64(0.999):
        print("  FAIL flow_match target <0.999")
        raise Error("flow_match target parity fail")

    # ── Arm 2: ema_update ───────────────────────────────────────────────────
    var decay = Float32(0.999)
    var shadow_vals = List[Float64]()
    var live_vals = List[Float64]()
    for i in range(n):
        shadow_vals.append(Float64(i) * 0.02 - 2.0)
        live_vals.append(Float64(i) * 0.017 + 0.5)
    var shadow = _f32_tensor(shadow_vals, ctx)
    var live = _f32_tensor(live_vals, ctx)
    ema_update(shadow, live, decay, ctx)
    var got_ema = _to_f64(shadow, ctx)
    var want_ema = List[Float64]()
    var d64 = Float64(decay)
    for i in range(n):
        want_ema.append(d64 * shadow_vals[i] + (Float64(1.0) - d64) * live_vals[i])
    var c_ema = _cos(got_ema, want_ema)
    print("ema_update cos        =", c_ema)
    if c_ema < Float64(0.999):
        print("  FAIL ema_update <0.999")
        raise Error("ema_update parity fail")

    # Hand-check the documented single step: decay=0.999, shadow=1, live=2 ->1.001
    var sh1 = _f32_tensor([Float64(1.0)], ctx)
    var lv1 = _f32_tensor([Float64(2.0)], ctx)
    ema_update(sh1, lv1, Float32(0.999), ctx)
    var got1 = _to_f64(sh1, ctx)
    print("ema hand-check (want 1.001) =", got1[0])
    if abs(got1[0] - Float64(1.001)) > Float64(1.0e-4):
        print("  FAIL ema hand-check")
        raise Error("ema hand-check fail")

    # ── Arm 3: grad_accumulate ──────────────────────────────────────────────
    var acc_vals = List[Float64]()
    var new_vals = List[Float64]()
    for i in range(n):
        acc_vals.append(Float64(i) * 0.005 - 0.6)
        new_vals.append(Float64(i) * 0.003 + 0.1)
    var acc = _f32_tensor(acc_vals, ctx)
    var newg = _f32_tensor(new_vals, ctx)
    grad_accumulate(acc, newg, ctx)
    var got_acc = _to_f64(acc, ctx)
    var want_acc = List[Float64]()
    for i in range(n):
        want_acc.append(acc_vals[i] + new_vals[i])
    var c_acc = _cos(got_acc, want_acc)
    print("grad_accumulate cos   =", c_acc)
    if c_acc < Float64(0.999):
        print("  FAIL grad_accumulate <0.999")
        raise Error("grad_accumulate parity fail")

    # ── Arm 4: sample_timestep_logit_normal (STATISTICAL) ───────────────────
    # t = sigmoid(N(0,1)) then qwen-shift remap. With shift=1.0 the remap is the
    # identity, so the draw is sigmoid(N(0,1)) clamped to [1/1000, 1].
    #
    # Expected range (the logistic-normal distribution, scale=1, loc=0):
    #   mean = 0.5 EXACTLY by symmetry of sigmoid(-z)=1-sigmoid(z) over N(0,1).
    #   std  ≈ 0.2078 (numerically; logistic-normal has no closed form). We use
    #   a generous band [0.18, 0.24] for std to absorb the [1/1000,1] clamp and
    #   finite-sample noise at N=100000. Mean band [0.49, 0.51].
    # This is NOT a cos gate — it asserts the distribution shape, per the task.
    var N = 100000
    var sum = Float64(0.0)
    var sumsq = Float64(0.0)
    var vmin = Float64(2.0)
    var vmax = Float64(-1.0)
    for i in range(N):
        var t = Float64(sample_timestep_logit_normal(UInt64(i), Float32(1.0)))
        sum += t
        sumsq += t * t
        if t < vmin:
            vmin = t
        if t > vmax:
            vmax = t
    var mean = sum / Float64(N)
    var var_ = sumsq / Float64(N) - mean * mean
    var std = sqrt(var_)
    print("timestep N            =", N)
    print("timestep mean (want ~0.5)   =", mean)
    print("timestep std  (want ~0.208) =", std)
    print("timestep min          =", vmin)
    print("timestep max          =", vmax)
    if mean < Float64(0.49) or mean > Float64(0.51):
        print("  FAIL timestep mean out of [0.49, 0.51]")
        raise Error("timestep mean out of range")
    if std < Float64(0.18) or std > Float64(0.24):
        print("  FAIL timestep std out of [0.18, 0.24]")
        raise Error("timestep std out of range")
    if vmin < Float64(1.0) / Float64(1000.0) - Float64(1.0e-6):
        print("  FAIL timestep min below clamp 1/1000")
        raise Error("timestep min below clamp")
    if vmax > Float64(1.0):
        print("  FAIL timestep max above 1.0")
        raise Error("timestep max above 1.0")

    # Shift sanity: shift>1 pushes mass toward 1 (mean rises); shift=1 identity.
    var sum_sh = Float64(0.0)
    var Nsh = 20000
    for i in range(Nsh):
        sum_sh += Float64(sample_timestep_logit_normal(UInt64(i), Float32(3.0)))
    var mean_sh = sum_sh / Float64(Nsh)
    print("timestep mean shift=3 (want >0.5) =", mean_sh)
    if mean_sh <= Float64(0.5):
        print("  FAIL shift=3 did not push mean up")
        raise Error("shift remap not pushing mean up")

    print("PASS schedule parity")
