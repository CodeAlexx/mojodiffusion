# Step-parity gate for the ACE-Step rectified-flow (Euler ODE) sampler.
#
# Loads acestep_sampler_fixture.safetensors (canonical schedule + per-step
# velocities + per-step xt from modeling_acestep_v15_base.generate_audio's ODE
# loop). Verifies:
#   (1) build_acestep_schedule(N=8, shift=3.0) matches the canonical `sched`.
#   (2) each acestep_euler_step output matches the canonical xt_i (cos>=0.999).
#
# Velocities are supplied by the fixture (deterministic) so the gate isolates
# the SAMPLER math (schedule + Euler update) from the DiT.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.sampling.acestep_flow_match import (
    build_acestep_schedule, acestep_euler_step,
)

comptime FIX = "/home/alex/mojodiffusion/serenitymojo/sampling/acestep_sampler_fixture.safetensors"
comptime N = 8
comptime SHIFT: Float32 = 3.0


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(FIX)

    # (1) schedule parity (host F32).
    var sched_ref = Tensor.from_view(st.tensor_view("sched"), ctx).to_host(ctx)
    var sched = build_acestep_schedule(N, SHIFT)
    var max_sched_err: Float32 = 0.0
    for i in range(N + 1):
        var d = sched[i] - sched_ref[i]
        if d < 0.0:
            d = -d
        if d > max_sched_err:
            max_sched_err = d
    print("schedule max abs err:", max_sched_err)

    # (2) per-step Euler parity.
    var xt = Tensor.from_view(st.tensor_view("x0"), ctx)  # F32
    var min_cos: Float64 = 1.0
    for i in range(N):
        var vt = Tensor.from_view(st.tensor_view(String("vel_") + String(i)), ctx)
        xt = acestep_euler_step(xt, vt, sched[i], sched[i + 1], ctx)
        var ref_xt = Tensor.from_view(
            st.tensor_view(String("xt_") + String(i)), ctx
        ).to_host(ctx)
        var ph = ParityHarness(0.999)
        var res = ph.compare(xt, ref_xt, ctx)
        print("step", i, "parity:", res)
        if res.cos < min_cos:
            min_cos = res.cos

    print("sampler min step cos:", min_cos)
    var sched_ok = max_sched_err < 1e-5
    if min_cos >= 0.999 and sched_ok:
        print("GATE: PASS")
    else:
        print("GATE: FAIL")
