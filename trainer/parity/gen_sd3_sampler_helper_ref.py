#!/usr/bin/env python3
"""Generate the SD3 sampler helper reference artifact.

This is intentionally helper-only. It mirrors deterministic values from
Serenity's StableDiffusion3Sampler path and the copied model scheduler:

* BaseModelSampler.quantize_resolution for image sizing.
* StableDiffusion3Sampler.__sample_base shape/CFG/decode contracts from source.
* diffusers.FlowMatchEulerDiscreteScheduler.set_timesteps for scheduler helpers.

It does not load SD3 weights, encode text, run transformer inference, sample
latents, decode with the VAE, or claim end-to-end sampler parity.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

from diffusers import FlowMatchEulerDiscreteScheduler


ROOT = Path(__file__).resolve().parents[1]
SERENITY_ROOT = Path("/home/alex/Serenity")
DEFAULT_OUT = ROOT / "parity" / "sd3_sampler_helper_ref.json"

if str(SERENITY_ROOT) not in sys.path:
    sys.path.insert(0, str(SERENITY_ROOT))

from modules.modelSampler.BaseModelSampler import BaseModelSampler  # noqa: E402


def _f32_list(tensor) -> list[float]:
    return [float(x) for x in tensor.detach().cpu().to(dtype=tensor.dtype).tolist()]


def build_ref(diffusion_steps: int, scheduler_shift: float) -> dict:
    height_in = 1025
    width_in = 1048
    height = BaseModelSampler.quantize_resolution(height_in, 16)
    width = BaseModelSampler.quantize_resolution(width_in, 16)
    latent_h = height // 8
    latent_w = width // 8

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=scheduler_shift,
    )
    scheduler.set_timesteps(diffusion_steps)
    sigmas = _f32_list(scheduler.sigmas)
    timesteps = _f32_list(scheduler.timesteps)

    cfg_negative = 1.0
    cfg_positive = 3.0
    cfg_scale = 4.0
    sample = 0.25
    model_output = 0.5

    return {
        "artifact": "sd3_sampler_helper_ref",
        "scope": "helper-only; not end-to-end sampler parity",
        "source": {
            "serenity_sampler": str(SERENITY_ROOT / "modules/modelSampler/StableDiffusion3Sampler.py"),
            "serenity_quantize": "BaseModelSampler.quantize_resolution",
            "scheduler": "diffusers.FlowMatchEulerDiscreteScheduler",
            "scheduler_set_timesteps": "StableDiffusion3Sampler.__sample_base copies model.noise_scheduler and calls set_timesteps(diffusion_steps)",
            "diffusers_version": __import__("diffusers").__version__,
        },
        "inputs": {
            "height": height_in,
            "width": width_in,
            "resolution_quantization": 16,
            "vae_scale_factor": 8,
            "latent_channels": 16,
            "latent_batch_size": 1,
            "contract_height": 1024,
            "contract_width": 512,
            "contract_batch_size": 3,
            "diffusion_steps": diffusion_steps,
            "num_train_timesteps": int(scheduler.config.num_train_timesteps),
            "scheduler_shift": float(scheduler.config.shift),
            "cfg_negative": cfg_negative,
            "cfg_positive": cfg_positive,
            "cfg_scale": cfg_scale,
            "euler_sample": sample,
            "euler_model_output": model_output,
        },
        "quantize_1025_16": BaseModelSampler.quantize_resolution(1025, 16),
        "quantize_1032_16": BaseModelSampler.quantize_resolution(1032, 16),
        "quantize_1048_16": BaseModelSampler.quantize_resolution(1048, 16),
        "plan_height": height,
        "plan_width": width,
        "plan_latent_h": latent_h,
        "plan_latent_w": latent_w,
        "plan_latent_channels": 16,
        "plan_cfg_batch": 2,
        "always_uses_negative_prompt": True,
        "scales_latents_before_transformer": False,
        "contract_batch": 3,
        "contract_latent_h": 1024 // 8,
        "contract_latent_w": 512 // 8,
        "contract_cfg_batch": 6,
        "cfg_batch_scalar": 4,
        "cfg_combine": cfg_negative + cfg_scale * (cfg_positive - cfg_negative),
        "flow_shift_sigma": scheduler_shift * 0.5 / (1.0 + (scheduler_shift - 1.0) * 0.5),
        "schedule_timesteps_len": len(timesteps),
        "schedule_sigmas_len": len(sigmas),
        "schedule_timesteps": timesteps,
        "schedule_sigmas": sigmas,
        "euler_update": sample + (sigmas[1] - sigmas[0]) * model_output,
        "step_accepts_generator": "generator" in inspect.signature(scheduler.step).parameters,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--diffusion-steps", type=int, default=4)
    parser.add_argument("--scheduler-shift", type=float, default=3.0)
    args = parser.parse_args()

    ref = build_ref(args.diffusion_steps, args.scheduler_shift)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
