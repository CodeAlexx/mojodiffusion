#!/usr/bin/env python3
"""Generate the Anima sampler helper reference artifact.

This is intentionally helper-only. It mirrors deterministic values from
Serenity-anima-ref's AnimaSampler path:

* BaseModelSampler.quantize_resolution for image sizing.
* AnimaSampler.__sample_base shape/CFG/decode contracts from source.
* FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=...) with explicit
  scheduler fixture config values.

It does not load Anima weights, encode text, run transformer inference, sample
latents, decode with the VAE, postprocess an image, or claim end-to-end sampler
parity.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler


ROOT = Path(__file__).resolve().parents[1]
ANIMA_REF_ROOT = Path("/home/alex/Serenity-anima-ref")
DEFAULT_OUT = ROOT / "parity" / "anima_sampler_helper_ref.json"

if str(ANIMA_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(ANIMA_REF_ROOT))

from modules.modelSampler.BaseModelSampler import BaseModelSampler  # noqa: E402


def _f32_list(tensor) -> list[float]:
    return [float(x) for x in tensor.detach().cpu().to(dtype=tensor.dtype).tolist()]


def _cfg_combine(positive: float, negative: float, cfg_scale: float) -> float:
    return negative + cfg_scale * (positive - negative)


def build_ref(
    diffusion_steps: int,
    num_train_timesteps: int,
    scheduler_shift: float,
    scheduler_use_dynamic_shifting: bool,
) -> dict:
    height_in = 1025
    width_in = 1120
    quantization = 64
    vae_scale_factor = 8
    latent_channels = 16
    latent_frames = 1

    height = BaseModelSampler.quantize_resolution(height_in, quantization)
    width = BaseModelSampler.quantize_resolution(width_in, quantization)
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=num_train_timesteps,
        shift=scheduler_shift,
        use_dynamic_shifting=scheduler_use_dynamic_shifting,
    )
    schedule_input_sigmas_np = np.linspace(1.0, 1.0 / diffusion_steps, diffusion_steps)
    scheduler.set_timesteps(sigmas=schedule_input_sigmas_np)
    schedule_input_sigmas = [float(x) for x in schedule_input_sigmas_np.tolist()]
    sigmas = _f32_list(scheduler.sigmas)
    timesteps = _f32_list(scheduler.timesteps)
    model_timestep_values = [float(x) / float(num_train_timesteps) for x in timesteps]

    cfg_negative = 1.0
    cfg_positive = 3.0
    cfg_scale = 4.0
    sample = 0.25
    model_output = 0.5

    return {
        "artifact": "anima_sampler_helper_ref",
        "scope": "helper-only; not end-to-end sampler parity",
        "source": {
            "serenity_sampler": str(ANIMA_REF_ROOT / "modules/modelSampler/AnimaSampler.py"),
            "serenity_model": str(ANIMA_REF_ROOT / "modules/model/AnimaModel.py"),
            "serenity_quantize": "BaseModelSampler.quantize_resolution",
            "scheduler": "diffusers.FlowMatchEulerDiscreteScheduler",
            "scheduler_set_timesteps": "AnimaSampler.__sample_base copies model.noise_scheduler and calls set_timesteps(sigmas=np.linspace(1.0, 1.0 / diffusion_steps, diffusion_steps))",
            "scheduler_fixture": "explicit helper inputs; Serenity loads these fields from model.noise_scheduler.config",
            "diffusers_version": __import__("diffusers").__version__,
        },
        "inputs": {
            "height": height_in,
            "width": width_in,
            "resolution_quantization": quantization,
            "vae_scale_factor": vae_scale_factor,
            "latent_channels": latent_channels,
            "latent_frames": latent_frames,
            "latent_batch_size": 1,
            "diffusion_steps": diffusion_steps,
            "num_train_timesteps": int(scheduler.config.num_train_timesteps),
            "scheduler_shift": float(scheduler.config.shift),
            "scheduler_use_dynamic_shifting": bool(scheduler.config.use_dynamic_shifting),
            "prompt_max_length": 512,
            "cfg_negative": cfg_negative,
            "cfg_positive": cfg_positive,
            "cfg_scale": cfg_scale,
            "euler_sample": sample,
            "euler_model_output": model_output,
        },
        "quantize_1025_64": BaseModelSampler.quantize_resolution(1025, quantization),
        "quantize_1056_64": BaseModelSampler.quantize_resolution(1056, quantization),
        "quantize_1120_64": BaseModelSampler.quantize_resolution(1120, quantization),
        "plan_height": height,
        "plan_width": width,
        "plan_latent_h": latent_h,
        "plan_latent_w": latent_w,
        "plan_latent_channels": latent_channels,
        "plan_latent_frames": latent_frames,
        "plan_latent_batch": 1,
        "plan_cfg_batch": 2,
        "padding_mask_batch": 1,
        "padding_mask_channels": 1,
        "padding_mask_h": height,
        "padding_mask_w": width,
        "use_cfg_at_1": False,
        "use_cfg_above_1": True,
        "uses_negative_prompt_when_cfg": True,
        "uses_negative_prompt_without_cfg": False,
        "scales_latents_before_transformer": False,
        "unscales_latents_before_vae_decode": True,
        "decoded_frame_index": 0,
        "cfg_combine": _cfg_combine(cfg_positive, cfg_negative, cfg_scale),
        "schedule_input_sigmas_len": len(schedule_input_sigmas),
        "schedule_timesteps_len": len(timesteps),
        "schedule_sigmas_len": len(sigmas),
        "schedule_input_sigmas": schedule_input_sigmas,
        "schedule_timesteps": timesteps,
        "schedule_sigmas": sigmas,
        "model_timestep_values": model_timestep_values,
        "euler_update": sample + (sigmas[1] - sigmas[0]) * model_output,
        "step_accepts_generator": "generator" in inspect.signature(scheduler.step).parameters,
        "initial_noise_dtype": "F32",
        "transformer_input_dtype": "transformer.dtype",
        "padding_mask_dtype": "transformer.dtype",
        "prompt_embedding_dtype": "transformer.dtype",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--diffusion-steps", type=int, default=4)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--scheduler-shift", type=float, default=3.0)
    parser.add_argument("--scheduler-use-dynamic-shifting", action="store_true")
    args = parser.parse_args()

    ref = build_ref(
        args.diffusion_steps,
        args.num_train_timesteps,
        args.scheduler_shift,
        args.scheduler_use_dynamic_shifting,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
