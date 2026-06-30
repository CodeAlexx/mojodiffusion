#!/usr/bin/env python3
# Oracle for the automagic3 bf16 STOCHASTIC-ROUNDING writeback.
# Runs ai-toolkit's ACTUAL Automagic3._sr_truncate over many draws at fixed
# fractional positions and measures (a) P(round up) and (b) the unbiased mean.
# SR is random, so parity vs the Mojo port is DISTRIBUTIONAL, not bit-exact:
# both must match the analytic P(up)=frac, mean=x within Monte-Carlo error.
import sys, struct
import torch
sys.path.insert(0, "/home/alex/ai-toolkit")
from toolkit.optimizers.automagic3 import Automagic3

N = 2_000_000
# bf16 ULP at 1.0 is 2^-7 (7 mantissa bits). x = 1.0 + frac*ULP lands strictly
# between the bf16 grid points 1.0 and 1.0+ULP, so SR must pick one of the two.
ULP = 2.0 ** -7
V_LO = 1.0
V_HI = 1.0 + ULP
fracs = [0.1, 0.25, 0.5, 0.75, 0.9]

print(f"# automagic3 SR oracle (ai-toolkit _sr_truncate), N={N}, bf16 ULP@1={ULP}")
print(f"# V_LO={V_LO}  V_HI={V_HI}")
print("frac     analytic_p_up   oracle_p_up   oracle_mean   target_mean   |bias|")
torch.manual_seed(42)
out_lines = []
for f in fracs:
    x = V_LO + f * ULP
    t = torch.full((N,), x, dtype=torch.float32)
    rounded = Automagic3._sr_truncate(t.clone(), 16).to(torch.bfloat16).float()
    # only V_LO / V_HI may appear
    uniq = torch.unique(rounded)
    assert all(abs(u.item() - V_LO) < 1e-9 or abs(u.item() - V_HI) < 1e-9 for u in uniq), \
        f"SR produced an off-grid value: {uniq.tolist()}"
    p_up = (rounded > V_LO).float().mean().item()
    mean = rounded.mean().item()
    bias = abs(mean - x)
    print(f"{f:<8} {f:<15.6f} {p_up:<13.6f} {mean:<13.7f} {x:<13.7f} {bias:.2e}")
    out_lines.append((f, p_up, mean, x))

# verdict: oracle must match analytic within MC tolerance (~3/sqrt(N) ~ 0.002)
tol = 0.003
ok = all(abs(p_up - f) < tol for (f, p_up, _, _) in out_lines)
print(f"\nORACLE_VERDICT: {'PASS' if ok else 'FAIL'} (oracle P(up) matches analytic frac within {tol})")
