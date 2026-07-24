# seedvr2_sampler_parity.mojo — parity probe for the SeedVR2 sampler math.
#
# Gates (F32, should be ~1.0):
#   * cfg_pred  cos  — cfg(pred_pos, pred_neg, 7.5)                 vs cfg_pred
#   * x_s       cos  — euler_vlerp_step(pred_pos, x_t, ts10, s10)   vs x_s
#   * x_final   cos  — euler_vlerp_step(pred_pos, x_t, ts49, 0)     vs x_final (clamp)
#   * schedule       — sampler_timesteps(1000, 50, 3.0) vs timesteps / s_all (max abs diff)
#
# PASS iff the 3 cos >= 0.999 AND both schedule max-diffs < 1e-2.
#
# Oracle: /home/alex/models/seedvr2-3b/sampler_oracle.safetensors (F32)
#   x_t / pred_pos / pred_neg [1024,16]; cfg_pred / x_s / x_final [1024,16];
#   timesteps / s_all [50].
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/dit/tests/seedvr2_sampler_parity.mojo

from std.gpu.host import DeviceContext
from math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.models.dit.seedvr2_sampler import (
    cfg,
    euler_vlerp_step,
    sampler_timesteps,
)

comptime ORACLE = "/home/alex/models/seedvr2-3b/sampler_oracle.safetensors"


def _cos(a: List[Float32], b: List[Float32]) -> Float32:
    var n = len(a)
    if len(b) < n:
        n = len(b)
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(n):
        var x = Float64(a[i])
        var y = Float64(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    var denom = sqrt(na) * sqrt(nb)
    if denom > 0.0:
        return Float32(dot / denom)
    return Float32(0.0)


def _maxabs(a: List[Float32], b: List[Float32]) -> Float32:
    var n = len(a)
    if len(b) < n:
        n = len(b)
    var maxd: Float32 = 0.0
    for i in range(n):
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > maxd:
            maxd = d
    return maxd


def main() raises:
    var ctx = DeviceContext()
    print("ctx ok", Int(ctx.id()))

    var orc = SafeTensors.open(ORACLE)

    # ── inputs (F32) ─────────────────────────────────────────────────────────
    var x_t = load_bias(orc, "x_t", ctx)          # [1024,16]
    var pred_pos = load_bias(orc, "pred_pos", ctx)  # [1024,16]
    var pred_neg = load_bias(orc, "pred_neg", ctx)  # [1024,16]

    # ── schedule (oracle refs) ───────────────────────────────────────────────
    var ts_ref = load_bias(orc, "timesteps", ctx).to_host(ctx)  # [50]
    var s_ref = load_bias(orc, "s_all", ctx).to_host(ctx)       # [50]

    # ── 1. CFG (scale 7.5) ───────────────────────────────────────────────────
    var cfg_out = cfg(pred_pos, pred_neg, Float32(7.5), ctx)
    var r_cfg = _cos(cfg_out.to_host(ctx), load_bias(orc, "cfg_pred", ctx).to_host(ctx))

    # The euler step is driven by the CFG-combined prediction (real pipeline:
    # cfg -> step), which is what the oracle x_s / x_final were produced from.

    # ── 2. euler step at t=timesteps[10], s=s_all[10] ────────────────────────
    var t10 = ts_ref[10]
    var s10 = s_ref[10]
    var xs_out = euler_vlerp_step(cfg_out, x_t, t10, s10, Float32(1000.0), ctx)
    var r_xs = _cos(xs_out.to_host(ctx), load_bias(orc, "x_s", ctx).to_host(ctx))

    # ── 3. final euler step at t=timesteps[49], s=0 (clamp branch) ────────────
    var t49 = ts_ref[49]
    var xf_out = euler_vlerp_step(cfg_out, x_t, t49, Float32(0.0), Float32(1000.0), ctx)
    var r_xf = _cos(xf_out.to_host(ctx), load_bias(orc, "x_final", ctx).to_host(ctx))

    # ── 4. schedule vs oracle ────────────────────────────────────────────────
    var sched = sampler_timesteps(Float32(1000.0), 50, Float32(3.0))
    var ts_max = _maxabs(sched[0], ts_ref)
    var s_max = _maxabs(sched[1], s_ref)

    print("cfg_pred cos =", r_cfg)
    print("x_s      cos =", r_xs, " (t=", t10, "s=", s10, ")")
    print("x_final  cos =", r_xf, " (t=", t49, "s= 0.0 clamp )")
    print("timesteps max_abs_diff =", ts_max)
    print("s_all     max_abs_diff =", s_max)

    var cos_ok = r_cfg >= 0.999 and r_xs >= 0.999 and r_xf >= 0.999
    var sched_ok = ts_max < 1e-2 and s_max < 1e-2
    if cos_ok and sched_ok:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
