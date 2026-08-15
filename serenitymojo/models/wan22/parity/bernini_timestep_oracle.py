#!/usr/bin/env python
# BERNINI-R timestep-sampling ORACLE (G1).
#
# Emits the reference distribution of TRAINING sigmas for the Bernini renderer,
# per (weighting_scheme, shift), using:
#   * the REAL FlowMatchScheduler loaded directly from
#     /home/alex/Bernini/bernini/models/scheduler.py (torch-only; loaded by file
#     path to skip the package __init__ that needs veomni), and
#   * compute_density_for_timestep_sampling + the NoiseScheduler window logic
#     copied VERBATIM from /home/alex/Bernini/bernini/training/data.py (:71-130).
#
# For each case it prints: the per-shift rejection window (tmin,tmax); N-sample
# sigma mean/std/min/max; a histogram over [0.86,1.0]; and DETERMINISTIC anchors
# (mode density(raw), logit density(z), shifted_sigma(idx)) for the bit-exact
# gate against the Mojo closed forms.
#
# Run: /home/alex/torchref-image/venv/bin/python \
#        serenitymojo/models/wan22/parity/bernini_timestep_oracle.py

import importlib.util
import math
import sys

import torch

_SCHED = "/home/alex/Bernini/bernini/models/scheduler.py"
_spec = importlib.util.spec_from_file_location("bernini_scheduler_standalone", _SCHED)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FlowMatchScheduler = _mod.FlowMatchScheduler


# ── verbatim from bernini/training/data.py ────────────────────────────────────
def shift2boundary(shift, sigma_min=0, sigma_max=1, denoising_strength=1.0, num_steps=1000):
    sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
    sigmas = torch.linspace(sigma_start, sigma_min, num_steps + 1)[:-1]
    return shift * sigmas / (1 + (shift - 1) * sigmas)


def find_nearest_boundary(sigmas, sigma_value):
    return int(torch.argmin((sigmas - sigma_value).abs()).item())


def compute_density_for_timestep_sampling(weighting_scheme, batch_size, logit_mean=0.5,
                                          logit_std=1.0, mode_scale=1.29, min_t=0.0, max_t=1.0):
    samples = []
    for _ in range(batch_size):
        while True:
            if weighting_scheme == "logit_normal":
                u = torch.sigmoid(torch.normal(mean=logit_mean, std=logit_std, size=(1,), device="cpu"))
            elif weighting_scheme == "mode":
                raw = torch.rand(size=(1,), device="cpu")
                u = 1 - raw - mode_scale * (torch.cos(math.pi * raw / 2) ** 2 - 1 + raw)
            else:
                u = torch.rand(size=(1,), device="cpu") * (max_t - min_t) + min_t
            if min_t <= float(u.item()) <= max_t:
                samples.append(u)
                break
    return torch.cat(samples, dim=0)


def build_window(shift, noise_tmin=0.875, noise_tmax=1.0):
    sigmas = shift2boundary(shift)
    b1 = find_nearest_boundary(sigmas, noise_tmin) / 1000
    b2 = find_nearest_boundary(sigmas, noise_tmax) / 1000
    return min(b1, b2), max(b1, b2)


def sample_sigmas(weighting, shift, n, tmin, tmax):
    sched = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    sched.set_timesteps(1000, training=True, device="cpu", dtype=torch.float64)
    out = []
    for _ in range(n):
        while True:
            if weighting == "logit_normal":
                uu = torch.sigmoid(torch.normal(mean=0.5, std=1.0, size=(1,)))
            else:
                raw = torch.rand(size=(1,))
                uu = 1 - raw - 1.29 * (torch.cos(math.pi * raw / 2) ** 2 - 1 + raw)
            if tmin <= float(uu.item()) <= tmax:
                break
        idx = int((uu * sched.num_train_timesteps).long().item())
        idx = max(0, min(sched.num_train_timesteps - 1, idx))
        sigma = float(sched.sigmas[idx].item())
        out.append(sigma)
    return out


def histo(vals, lo=0.86, hi=1.0, nb=14):
    counts = [0] * nb
    for v in vals:
        b = int((v - lo) / (hi - lo) * nb)
        b = max(0, min(nb - 1, b))
        counts[b] += 1
    return counts


def moments(vals):
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    return m, var ** 0.5, min(vals), max(vals)


def main():
    torch.manual_seed(1234)
    N = 200000
    cases = [
        ("logit_normal", 3.0, "t2i"),
        ("logit_normal", 4.0, "i2i"),
        ("mode", 3.0, "t2v"),
        ("mode", 4.0, "r2v"),
        ("mode", 5.0, "i2v/v2v"),
    ]
    print("=== BERNINI-R timestep oracle (N=%d per case) ===" % N)
    for weighting, shift, tasks in cases:
        tmin, tmax = build_window(shift)
        vals = sample_sigmas(weighting, shift, N, tmin, tmax)
        m, s, lo, hi = moments(vals)
        h = histo(vals)
        print("\n[%s shift=%.1f tasks=%s]" % (weighting, shift, tasks))
        print("  window u in [tmin=%.4f, tmax=%.4f]" % (tmin, tmax))
        print("  sigma mean=%.6f std=%.6f min=%.6f max=%.6f" % (m, s, lo, hi))
        print("  hist[0.86..1.0 /14]:", h)

    # deterministic anchors (bit-exact gate vs Mojo closed forms)
    print("\n=== deterministic anchors ===")
    for raw in (0.1, 0.5, 0.9):
        u = 1 - raw - 1.29 * (math.cos(math.pi * raw / 2) ** 2 - 1 + raw)
        print("  mode_density(raw=%.1f) = %.10f" % (raw, u))
    for z in (-1.0, 0.0, 1.0):
        u = 1.0 / (1.0 + math.exp(-(0.5 + 1.0 * z)))
        print("  logit_density(z=%+.1f) = %.10f" % (z, u))
    for shift in (3.0, 4.0, 5.0):
        for idx in (0, 50, 125):
            sl = 1 - idx / 1000
            sig = shift * sl / (1 + (shift - 1) * sl)
            print("  shifted_sigma(shift=%.1f, idx=%3d) = %.10f" % (shift, idx, sig))


if __name__ == "__main__":
    main()
