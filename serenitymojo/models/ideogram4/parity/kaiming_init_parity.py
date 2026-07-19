#!/usr/bin/env python3
# Reference generator for the LoRA-A kaiming_uniform init parity gate.
# Runs torch.nn.init.kaiming_uniform_(a=sqrt(5)) — the PEFT/torch reference the
# Mojo make_lora_adapter must match DISTRIBUTIONALLY (RNG streams differ across
# runtimes, so this is formula-parity + distribution-stats, NOT bit parity;
# see the numeric-parity-testing skill's RNG row).
#
# Dumps, per shape [rank, in_features]:
#   fan_in, bound (= torch's computed uniform bound), and the empirical
#   mean/std/min/max of a real kaiming_uniform_ draw. The Mojo gate reads this
#   JSON, draws rand_uniform on the SAME shape+bound, and compares.
import json, math
import torch
import torch.nn as nn

def ref(rank: int, in_features: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    w = torch.empty(rank, in_features, dtype=torch.float32)
    # torch's kaiming_uniform_ doesn't take a generator arg pre-2.x cleanly;
    # seed the global RNG for reproducibility of the REFERENCE stats.
    torch.manual_seed(seed)
    nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    # torch's own computed bound (what kaiming_uniform_ used internally):
    fan_in = in_features  # _calculate_fan_in for [out,in] weight
    gain = math.sqrt(2.0 / (1.0 + 5.0))       # a=sqrt(5) -> 1/sqrt(3)
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std               # = 1/sqrt(fan_in)
    return {
        "rank": rank, "in_features": in_features, "fan_in": fan_in,
        "bound": bound,                         # THE formula-parity value
        "theo_std": bound / math.sqrt(3.0),     # uniform(-b,b) std
        "mean": float(w.mean()), "std": float(w.std()),
        "min": float(w.min()), "max": float(w.max()),
        "n": rank * in_features,
    }

shapes = [(64, 4608), (16, 4608), (64, 512), (16, 12288)]
out = {"cases": [ref(r, i) for (r, i) in shapes]}
p = "/home/alex/mojodiffusion/serenitymojo/models/ideogram4/parity/kaiming_init_ref.json"
json.dump(out, open(p, "w"), indent=1)
print("wrote", p)
for c in out["cases"]:
    print(f"  [{c['rank']},{c['in_features']}] bound={c['bound']:.8f} "
          f"mean={c['mean']:+.6f} std={c['std']:.6f} "
          f"min={c['min']:+.6f} max={c['max']:+.6f} theo_std={c['theo_std']:.6f}")
