"""Krea-2 sampler schedule oracle — the reference `timesteps()` output.

Regenerates the pinned values in sampling/krea2_sampler_smoke.mojo. The body of
`timesteps()` is copied VERBATIM from ai-toolkit krea2 src/pipeline.py:138-160
(the module can't be imported directly — its relative `from .mmdit import ...`
needs the package context — so the pure schedule fn is inlined here; it depends
only on `math` + `torch.linspace`).

Defaults are the krea2 model_kwargs (krea2.py:206-211 / pipeline.py:142-145):
  y1=0.5, y2=1.15, min_res=256, max_res=1280, sigma=1.0, align=8*2=16,
  x1=(256//16)^2=256, x2=(1280//16)^2=6400.

Run: /home/alex/serenityflow-v2/.venv/bin/python \
        serenitymojo/sampling/parity/gen_krea2_timesteps_oracle.py
"""

import json
import math
import os

import torch


def timesteps(seq_len, steps, x1, x2, y1=0.5, y2=1.15, sigma=1.0, mu=None):
    # VERBATIM ai-toolkit krea2 src/pipeline.py:138-160.
    ts = torch.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()


def main():
    align = 16
    minres, maxres = 256, 1280
    y1, y2 = 0.5, 1.15
    x1 = (minres // align) ** 2  # 256
    x2 = (maxres // align) ** 2  # 6400

    cases = []
    for (H, W, steps) in [(256, 256, 4), (1024, 1024, 28), (512, 768, 10)]:
        gh, gw = H // align, W // align
        seq = gh * gw
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq + (y1 - slope * x1)
        ts = timesteps(seq, steps, x1, x2, y1=y1, y2=y2)
        cases.append({"H": H, "W": W, "steps": steps, "seq": seq, "mu": mu, "ts": ts})
        print(f"H={H} W={W} steps={steps} seq={seq} mu={mu:.6f}")
        print("  ts =", [round(v, 8) for v in ts])

    out = os.path.join(os.path.dirname(__file__), "krea2_timesteps_oracle.json")
    with open(out, "w") as f:
        json.dump(cases, f, indent=2)
    print("WROTE", out)


if __name__ == "__main__":
    main()
