#!/usr/bin/env python3
"""Generate the SDXL sampler helper-contract reference artifact.

This is intentionally helper/contract-only. It mirrors deterministic values from
Serenity's StableDiffusionXLSampler path:

* BaseModelSampler.quantize_resolution for 64px image sizing.
* StableDiffusionXLSampler.__sample_base helper shape/CFG/decode metadata.
* StableDiffusionXLSampler.__sample_inpainting conditioning/mask helper contracts.
* create.create_noise_scheduler selection metadata as written in Serenity.

It does not load SDXL weights, encode text, run UNet inference, sample latents,
decode with the VAE, postprocess an image, save output, or claim denoise,
decode, image, or end-to-end sampler parity.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    UniPCMultistepScheduler,
)


ROOT = Path(__file__).resolve().parents[1]
SERENITY_ROOT = Path("/home/alex/Serenity")
DEFAULT_OUT = ROOT / "parity" / "sdxl_sampler_helper_ref.json"

if str(SERENITY_ROOT) not in sys.path:
    sys.path.insert(0, str(SERENITY_ROOT))

from modules.modelSampler.BaseModelSampler import BaseModelSampler  # noqa: E402
from modules.util.config.SampleConfig import SampleConfig  # noqa: E402
from modules.util.enum.ModelType import ModelType  # noqa: E402
from modules.util.enum.NoiseScheduler import NoiseScheduler  # noqa: E402


NOISE_SCHEDULER_INDEX = {
    scheduler: index for index, scheduler in enumerate(NoiseScheduler)
}


def _scheduler_metadata(noise_scheduler: NoiseScheduler) -> dict:
    """Mirror Serenity modules/util/create.py:create_noise_scheduler."""

    common = dict(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        trained_betas=None,
        prediction_type="epsilon",
    )
    steps_offset = 1
    use_karras_sigmas = False
    algorithm_type = ""

    if noise_scheduler == NoiseScheduler.DDIM:
        scheduler = DDIMScheduler(
            **common,
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=steps_offset,
        )
    elif noise_scheduler == NoiseScheduler.EULER:
        scheduler = EulerDiscreteScheduler(
            **common,
            steps_offset=steps_offset,
            use_karras_sigmas=False,
        )
    elif noise_scheduler == NoiseScheduler.EULER_A:
        scheduler = EulerAncestralDiscreteScheduler(
            **common,
            steps_offset=steps_offset,
        )
    elif noise_scheduler == NoiseScheduler.DPMPP:
        steps_offset = 0
        algorithm_type = "dpmsolver++"
        scheduler = DPMSolverMultistepScheduler(
            **common,
            steps_offset=steps_offset,
            use_karras_sigmas=False,
            algorithm_type=algorithm_type,
        )
    elif noise_scheduler == NoiseScheduler.DPMPP_SDE:
        steps_offset = 0
        algorithm_type = "sde-dpmsolver++"
        scheduler = DPMSolverMultistepScheduler(
            **common,
            steps_offset=steps_offset,
            use_karras_sigmas=False,
            algorithm_type=algorithm_type,
        )
    elif noise_scheduler == NoiseScheduler.UNIPC:
        scheduler = UniPCMultistepScheduler(
            **common,
            steps_offset=steps_offset,
            use_karras_sigmas=False,
        )
    elif noise_scheduler == NoiseScheduler.EULER_KARRAS:
        use_karras_sigmas = True
        scheduler = EulerDiscreteScheduler(
            **common,
            steps_offset=steps_offset,
            use_karras_sigmas=True,
        )
    elif noise_scheduler == NoiseScheduler.DPMPP_KARRAS:
        use_karras_sigmas = True
        algorithm_type = "dpmsolver++"
        scheduler = DPMSolverMultistepScheduler(
            **common,
            steps_offset=steps_offset,
            use_karras_sigmas=True,
            algorithm_type=algorithm_type,
        )
    elif noise_scheduler == NoiseScheduler.DPMPP_SDE_KARRAS:
        use_karras_sigmas = True
        algorithm_type = "sde-dpmsolver++"
        scheduler = DPMSolverMultistepScheduler(
            **common,
            steps_offset=steps_offset,
            use_karras_sigmas=True,
            algorithm_type=algorithm_type,
        )
    elif noise_scheduler == NoiseScheduler.UNIPC_KARRAS:
        use_karras_sigmas = True
        scheduler = UniPCMultistepScheduler(
            **common,
            steps_offset=steps_offset,
            use_karras_sigmas=True,
        )
    else:
        raise ValueError(noise_scheduler)

    return {
        "name": noise_scheduler.name,
        "index": NOISE_SCHEDULER_INDEX[noise_scheduler],
        "class_name": scheduler.__class__.__name__,
        "steps_offset": steps_offset,
        "use_karras_sigmas": use_karras_sigmas,
        "algorithm_type": algorithm_type,
        "step_accepts_generator": "generator" in inspect.signature(scheduler.step).parameters,
    }


def _cfg_rescale_fixture() -> dict:
    negative = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
    positive = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])
    cfg_scale = 4.0
    cfg_rescale = 0.7
    pred = negative + cfg_scale * (positive - negative)
    std_positive = positive.std(dim=list(range(1, positive.ndim)), keepdim=True)
    std_pred = pred.std(dim=list(range(1, pred.ndim)), keepdim=True)
    pred_rescaled = pred * (std_positive / std_pred)
    final = cfg_rescale * pred_rescaled + (1.0 - cfg_rescale) * pred
    return {
        "cfg_rescale_noise_pred_sample": float(pred.flatten()[0]),
        "cfg_rescale_std_positive": float(std_positive.flatten()[0]),
        "cfg_rescale_std_pred": float(std_pred.flatten()[0]),
        "cfg_rescale": cfg_rescale,
        "cfg_rescale_value": float(final.flatten()[0]),
    }


def _timestep_counts(
    diffusion_steps: int,
    *,
    force_last_timestep: bool,
    inpainting_model_type: bool,
    sample_inpainting: bool,
) -> tuple[int, int]:
    min_count = diffusion_steps
    max_count = diffusion_steps + (1 if force_last_timestep else 0)
    if inpainting_model_type and sample_inpainting:
        min_count -= 1
        max_count -= 1
    return min_count, max_count


def build_ref(diffusion_steps: int) -> dict:
    height_in = 1025
    width_in = 1120
    quantization = 64
    vae_scale_factor = 8
    latent_channels = 4
    latent_batch = 1
    cfg_batch = latent_batch * 2

    height = BaseModelSampler.quantize_resolution(height_in, quantization)
    width = BaseModelSampler.quantize_resolution(width_in, quantization)
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor

    cfg_negative = 1.0
    cfg_positive = 3.0
    cfg_scale = 4.0
    erode_kernel_radius = 2
    erode_kernel_size = erode_kernel_radius * 2 + 1

    default_config = SampleConfig.default_values(ModelType.STABLE_DIFFUSION_XL_10_BASE)
    default_scheduler = default_config.noise_scheduler
    scheduler_cases = {
        scheduler.name: _scheduler_metadata(scheduler) for scheduler in NoiseScheduler
    }
    default_scheduler_meta = scheduler_cases[default_scheduler.name]
    base_force_min, base_force_max = _timestep_counts(
        diffusion_steps,
        force_last_timestep=True,
        inpainting_model_type=False,
        sample_inpainting=False,
    )
    inpaint_force_min, inpaint_force_max = _timestep_counts(
        diffusion_steps,
        force_last_timestep=True,
        inpainting_model_type=True,
        sample_inpainting=True,
    )
    inpaint_no_force_min, inpaint_no_force_max = _timestep_counts(
        diffusion_steps,
        force_last_timestep=False,
        inpainting_model_type=True,
        sample_inpainting=True,
    )

    result = {
        "artifact": "sdxl_sampler_helper_ref",
        "scope": "helper/contract only; not denoise/decode/image parity; not end-to-end sampler parity",
        "source": {
            "serenity_sampler": str(SERENITY_ROOT / "modules/modelSampler/StableDiffusionXLSampler.py"),
            "serenity_quantize": "BaseModelSampler.quantize_resolution",
            "noise_scheduler_factory": str(SERENITY_ROOT / "modules/util/create.py:create_noise_scheduler"),
            "sample_config_defaults": str(SERENITY_ROOT / "modules/util/config/SampleConfig.py"),
            "diffusers_version": __import__("diffusers").__version__,
            "torch_version": torch.__version__,
        },
        "inputs": {
            "height": height_in,
            "width": width_in,
            "resolution_quantization": quantization,
            "vae_scale_factor": vae_scale_factor,
            "latent_channels": latent_channels,
            "latent_batch_size": latent_batch,
            "diffusion_steps": diffusion_steps,
            "cfg_negative": cfg_negative,
            "cfg_positive": cfg_positive,
            "cfg_scale": cfg_scale,
            "default_noise_scheduler": default_scheduler.name,
            "default_noise_scheduler_index": NOISE_SCHEDULER_INDEX[default_scheduler],
        },
        "quantize_1025_64": BaseModelSampler.quantize_resolution(1025, quantization),
        "quantize_1056_64": BaseModelSampler.quantize_resolution(1056, quantization),
        "quantize_1120_64": BaseModelSampler.quantize_resolution(1120, quantization),
        "plan_height": height,
        "plan_width": width,
        "plan_latent_h": latent_h,
        "plan_latent_w": latent_w,
        "plan_latent_channels": latent_channels,
        "plan_latent_batch": latent_batch,
        "plan_cfg_batch": cfg_batch,
        "base_unet_input_channels": latent_channels,
        "base_unet_input_batch": cfg_batch,
        "inpaint_unet_input_channels": latent_channels * 2 + 1,
        "inpaint_unet_input_batch": cfg_batch,
        "latent_mask_channels": 1,
        "latent_conditioning_channels": latent_channels,
        "conditioning_image_channels": 3,
        "conditioning_image_h": height,
        "conditioning_image_w": width,
        "add_time_ids": [height, width, 0, 0, height, width],
        "add_time_ids_rows": cfg_batch,
        "add_time_ids_cols": 6,
        "uses_negative_prompt": True,
        "pooled_text_embedding_used": True,
        "scheduler_scales_model_input": True,
        "extra_step_kwargs_may_include_generator": True,
        "cfg_combine": cfg_negative + cfg_scale * (cfg_positive - cfg_negative),
        "cfg_rescale_force_last": 0.7,
        "cfg_rescale_no_force": 0.0,
        "base_force_timestep_min": base_force_min,
        "base_force_timestep_max": base_force_max,
        "inpaint_force_timestep_min": inpaint_force_min,
        "inpaint_force_timestep_max": inpaint_force_max,
        "inpaint_no_force_timestep_min": inpaint_no_force_min,
        "inpaint_no_force_timestep_max": inpaint_no_force_max,
        "sample_inpainting_drops_first_timestep": True,
        "erode_kernel_radius": erode_kernel_radius,
        "erode_kernel_size": erode_kernel_size,
        "erode_kernel_weight_count": erode_kernel_size * erode_kernel_size,
        "erode_kernel_uniform_weight": 1.0 / float(erode_kernel_size * erode_kernel_size),
        "decode_formula": "vae.decode(latent_image / vae.config.scaling_factor)",
        "postprocess_output_type": "pil",
        "initial_noise_dtype": "model.train_dtype.torch_dtype()",
        "decode_input_dtype": "model.vae_train_dtype.torch_dtype()",
        "default_scheduler_class": default_scheduler_meta["class_name"],
        "default_scheduler_steps_offset": default_scheduler_meta["steps_offset"],
        "default_scheduler_step_accepts_generator": default_scheduler_meta["step_accepts_generator"],
        "dpmpp_steps_offset": scheduler_cases["DPMPP"]["steps_offset"],
        "dpmpp_algorithm_type": scheduler_cases["DPMPP"]["algorithm_type"],
        "euler_karras_uses_karras_sigmas": scheduler_cases["EULER_KARRAS"]["use_karras_sigmas"],
        "unipc_step_accepts_generator": scheduler_cases["UNIPC"]["step_accepts_generator"],
        "scheduler_cases": scheduler_cases,
    }
    result.update(_cfg_rescale_fixture())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--diffusion-steps", type=int, default=4)
    args = parser.parse_args()

    if args.diffusion_steps <= 0:
        raise ValueError("--diffusion-steps must be positive")
    ref = build_ref(args.diffusion_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
