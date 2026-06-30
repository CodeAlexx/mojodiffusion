# automagic3_sr_parity_gate.mojo — oracle gate for the bf16 STOCHASTIC-ROUNDING
# writeback (sr_truncate_f32_to_bits). SR is random, so parity vs ai-toolkit's
# _sr_truncate is DISTRIBUTIONAL: over many draws at a fixed fractional position
# between two bf16 grid points, the Mojo port must (a) round UP with probability
# == that fraction and (b) be UNBIASED (mean == x). Compare the printed numbers
# against gen_sr_oracle.py (ai-toolkit) — both must match the analytic fraction.
#
# Build (-O2 to avoid the -O3 compile OOM):
#   pixi run mojo build --optimization-level 2 -I . \
#     serenitymojo/training/parity/automagic3_sr_parity_gate.mojo -o /tmp/sr_gate
from serenitymojo.training.automagic3 import sr_truncate_f32_to_bits, Automagic3Rng


def main():
    var n = 2000000
    var ulp = Float32(0.0078125)   # bf16 ULP at 1.0 = 2^-7
    var v_lo = Float32(1.0)
    var fracs = List[Float32]()
    fracs.append(0.1)
    fracs.append(0.25)
    fracs.append(0.5)
    fracs.append(0.75)
    fracs.append(0.9)

    print("# automagic3 SR Mojo gate (sr_truncate_f32_to_bits), N=", n, " ULP@1=", ulp)
    print("frac   analytic_p_up   mojo_p_up      mojo_mean      target_mean    |bias|")
    var rng = Automagic3Rng(42)
    var all_ok = True
    var tol = Float32(0.003)
    for fi in range(len(fracs)):
        var f = fracs[fi]
        var x = v_lo + f * ulp
        var up = 0
        var s = Float64(0.0)
        for _ in range(n):
            var r = sr_truncate_f32_to_bits(x, 16, rng.next_u32())
            if r > v_lo:
                up += 1
            s += Float64(r)
        var p_up = Float32(up) / Float32(n)
        var mean = s / Float64(n)
        var bias = abs(Float64(x) - mean)
        print(f, "  ", f, "      ", p_up, "    ", mean, "   ", x, "   ", bias)
        if abs(p_up - f) >= tol:
            all_ok = False
    if all_ok:
        print("\nMOJO_VERDICT: PASS (Mojo P(up) matches analytic frac within", tol, ")")
    else:
        print("\nMOJO_VERDICT: FAIL")
