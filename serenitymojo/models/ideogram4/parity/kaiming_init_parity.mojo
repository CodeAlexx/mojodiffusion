# Parity gate: Mojo rand_uniform LoRA-A init vs torch nn.init.kaiming_uniform_(a=sqrt(5)).
#
# RNG streams differ across runtimes (torch Generator vs our ChaCha), so this is
# FORMULA parity (bound = 1/sqrt(fan_in), exact) + DISTRIBUTION stats (mean/std/
# range close, NOT bit-match) — the numeric-parity-testing skill's RNG protocol.
#
# Reference values are torch's ACTUAL runtime output (kaiming_init_parity.py):
#   [64,4608]  bound 0.01473139  std 0.008508
#   [16,4608]  bound 0.01473139  std 0.008481
#   [64,512]   bound 0.04419417  std 0.025468
#   [16,12288] bound 0.00902110  std 0.005207
# torch mean ~0, min/max = +/-bound in every case.
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/ideogram4/parity/kaiming_init_parity.mojo
from max.gpu.host import DeviceContext
from std.math import sqrt
from serenitymojo.ops.random import rand_uniform
from serenitymojo.io.dtype import STDtype


def _stats(rank: Int, in_features: Int, torch_bound: Float32, torch_std: Float32,
           ctx: DeviceContext) raises -> Bool:
    var bound = Float32(1.0) / sqrt(Float32(in_features))
    var sh = List[Int](); sh.append(rank); sh.append(in_features)
    var t = rand_uniform(sh^, UInt64(0), STDtype.BF16, -bound, bound, ctx)
    var h = t.to_host(ctx)
    var n = rank * in_features
    var s = Float64(0.0); var ss = Float64(0.0)
    var mn = Float64(1.0e30); var mx = Float64(-1.0e30)
    for i in range(n):
        var v = Float64(h[i])
        s += v; ss += v * v
        if v < mn: mn = v
        if v > mx: mx = v
    var mean = s / Float64(n)
    var var_ = ss / Float64(n) - mean * mean
    var std = sqrt(Float32(var_)) if var_ > 0 else Float32(0.0)

    # formula parity: Mojo's bound == torch's bound (both 1/sqrt(in)), EXACT.
    var bound_ok = abs(bound - torch_bound) < Float32(1.0e-6)
    # distribution: std within 3% of torch's measured std.
    var std_rel = abs(std - torch_std) / torch_std
    var std_ok = std_rel < Float32(0.03)
    # mean ~0 (within 2% of bound); support reaches near +/-bound and stays in it
    # (bf16 rounding may nudge a hair past; allow 1 bf16 ulp = 2^-8 slack).
    var mean_ok = abs(Float32(mean)) < Float32(0.02) * bound
    var slack = bound * Float32(1.0 + 0.0039)
    var range_ok = (Float32(mn) >= -slack and Float32(mx) <= slack
                    and Float32(-mn) > Float32(0.9) * bound
                    and Float32(mx) > Float32(0.9) * bound)
    var ok = bound_ok and std_ok and mean_ok and range_ok
    print("  [", rank, ",", in_features, "] bound=", bound, "(torch", torch_bound,
          ") std=", std, "(torch", torch_std, " rel", std_rel,
          ") mean=", Float32(mean), " min=", Float32(mn), " max=", Float32(mx),
          " ->", "PASS" if ok else "FAIL")
    return ok


def main() raises:
    var ctx = DeviceContext()
    print("==== kaiming_uniform init parity: Mojo rand_uniform vs torch kaiming_uniform_(a=sqrt5) ====")
    var all_ok = True
    all_ok = _stats(64, 4608, Float32(0.01473139), Float32(0.008508), ctx) and all_ok
    all_ok = _stats(16, 4608, Float32(0.01473139), Float32(0.008481), ctx) and all_ok
    all_ok = _stats(64, 512, Float32(0.04419417), Float32(0.025468), ctx) and all_ok
    all_ok = _stats(16, 12288, Float32(0.00902110), Float32(0.005207), ctx) and all_ok
    if all_ok:
        print("VERDICT: PASS — bound EXACT (formula parity), std within 3% + mean~0 + range in [-b,b] (distribution parity vs torch)")
    else:
        print("VERDICT: FAIL")
