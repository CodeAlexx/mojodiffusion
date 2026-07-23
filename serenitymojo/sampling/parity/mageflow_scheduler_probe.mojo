# mageflow_scheduler_probe.mojo — Mage-Flow rectified-flow sigma-schedule parity gate.
#
# Verifies serenitymojo's flow_match.build_sigma_schedule(N, shift=6.0) reproduces
# Mage-Flow's (microsoft/Mage-Flow-Edit-Turbo) FlowMatchEulerDiscreteScheduler
# sigma schedule (static_shift=6.0, use_dynamic_shifting=false).
#
# Reference math (mage_flow/pipeline.py:37-50):
#   base_sigmas = torch.linspace(1.0, 1.0/N, N)  (N values, no trailing 0)
#   scheduler.set_timesteps(sigmas=base_sigmas) -> static shift + append 0 -> N+1
#     sigma = shift*s / (1 + (shift-1)*s)
# serenitymojo build_sigma_schedule(N, shift):
#   t_i = 1 - i/N for i in 0..=N  (N+1 values incl trailing t=0 -> sigma 0)
#   sigma_i = shift*t_i / (1 + (shift-1)*t_i)
# linspace(1, 1/N, N) == [1 - i/N for i in 0..N-1] (step 1/N), so the two agree
# element-wise; the trailing 0 matches (t=0 -> sigma 0 == diffusers' appended 0).
#
# Oracle values from:
#   pixi run python \
#     serenitymojo/sampling/parity/mageflow_scheduler_oracle.py
# Build+run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo run -I . serenitymojo/sampling/parity/mageflow_scheduler_probe.mojo
from std.collections import List
from serenitymojo.sampling.flow_match import build_sigma_schedule

comptime SHIFT: Float32 = 6.0
comptime TOL: Float32 = 1.0e-5


def _oracle_n4() -> List[Float32]:
    var out = List[Float32]()
    out.append(1.0)
    out.append(0.9473684430122375)
    out.append(0.8571428656578064)
    out.append(0.6666666865348816)
    out.append(0.0)
    return out^


def _oracle_n20() -> List[Float32]:
    var out = List[Float32]()
    out.append(1.0)
    out.append(0.991304337978363)
    out.append(0.9818181395530701)
    out.append(0.971428632736206)
    out.append(0.9600000381469727)
    out.append(0.9473684430122375)
    out.append(0.9333332777023315)
    out.append(0.9176470041275024)
    out.append(0.9000000357627869)
    out.append(0.8800000548362732)
    out.append(0.8571428656578064)
    out.append(0.8307692408561707)
    out.append(0.800000011920929)
    out.append(0.7636363506317139)
    out.append(0.7200000286102295)
    out.append(0.6666666865348816)
    out.append(0.6000000238418579)
    out.append(0.5142857432365417)
    out.append(0.4000000059604645)
    out.append(0.24000000953674316)
    out.append(0.0)
    return out^


def _compare(name: String, mine: List[Float32], reference: List[Float32]) raises -> Float32:
    print("===", name, "===")
    if len(mine) != len(reference):
        raise Error(
            String("len mismatch: mine=") + String(len(mine))
            + " ref=" + String(len(reference))
        )
    var maxabs = Float32(0.0)
    for i in range(len(reference)):
        var d = mine[i] - reference[i]
        if d < 0.0:
            d = -d
        if d > maxabs:
            maxabs = d
        print("  i=", i, " mine=", mine[i], " ref=", reference[i], " |d|=", d)
    print("  len =", len(mine), "  max-abs-diff =", maxabs)
    return maxabs


def main() raises:
    print("=== Mage-Flow rectified-flow sigma schedule parity (shift=6.0) ===")
    # Turbo = 4 steps, base/edit = 20 steps. build_sigma_schedule(N, 6.0).
    var mine4 = build_sigma_schedule(4, SHIFT)
    var mine20 = build_sigma_schedule(20, SHIFT)
    var ref4 = _oracle_n4()
    var ref20 = _oracle_n20()

    var m4 = _compare("N=4 (turbo)", mine4, ref4)
    var m20 = _compare("N=20 (base/edit)", mine20, ref20)

    print("---")
    print("N=4  max-abs =", m4)
    print("N=20 max-abs =", m20)
    var worst = m4 if m4 > m20 else m20
    if worst > TOL:
        raise Error(
            String("FAIL: worst max-abs ") + String(worst) + " > tol " + String(TOL)
        )
    print("VERDICT: PASS — build_sigma_schedule(shift=6.0) matches Mage-Flow (max-abs <= 1e-5)")
