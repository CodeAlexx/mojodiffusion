#!/usr/bin/env python
"""OFFLINE parity oracle for Mage-Flow's rectified-flow sigma schedule.

This is a PARITY ORACLE ONLY — never imported by shipped Mojo. It reproduces the
EXACT sigma schedule that `microsoft/Mage-Flow-Edit-Turbo` builds, so we can
verify serenitymojo's `sampling/flow_match.mojo::build_sigma_schedule` matches.

Reference (verbatim math from /home/alex/Mage/mage_flow/pipeline.py:37-50):

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=6.0, use_dynamic_shifting=False)
    base_sigmas = torch.linspace(1.0, 1.0 / num_steps, num_steps).tolist()
    scheduler.set_timesteps(sigmas=base_sigmas, device=device)

scheduler_config.json (microsoft/Mage-Flow-Edit-Turbo): shift=6.0,
use_dynamic_shifting=false, num_train_timesteps=1000. There is NO use_time_shift
field (static shift only).

Run with a venv that has torch + diffusers, e.g.:
    pixi run python \
        serenitymojo/sampling/parity/mageflow_scheduler_oracle.py
"""
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

STATIC_SHIFT = 6.0


def mageflow_sigmas(num_steps: int, shift: float = STATIC_SHIFT):
    """Build the scheduler EXACTLY as mage_flow/pipeline.py:build_scheduler does
    and return (sigmas, timesteps) as python float lists."""
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
    base_sigmas = torch.linspace(1.0, 1.0 / num_steps, num_steps).tolist()
    scheduler.set_timesteps(sigmas=base_sigmas, device="cpu")
    sigmas = [float(x) for x in scheduler.sigmas.tolist()]
    timesteps = [float(x) for x in scheduler.timesteps.tolist()]
    return sigmas, timesteps


def main():
    for n in (4, 20):
        sigmas, timesteps = mageflow_sigmas(n, STATIC_SHIFT)
        print(f"=== num_inference_steps={n}, static_shift={STATIC_SHIFT} ===")
        print(f"len(sigmas)={len(sigmas)}  len(timesteps)={len(timesteps)}")
        print("SIGMAS_N%d=%s" % (n, ",".join(repr(x) for x in sigmas)))
        print("TIMESTEPS_N%d=%s" % (n, ",".join(repr(x) for x in timesteps)))
        print()


if __name__ == "__main__":
    main()
