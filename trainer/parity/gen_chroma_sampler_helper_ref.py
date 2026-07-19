#!/usr/bin/env python3
"""Generate the Chroma sampler helper reference artifact.

This is intentionally helper-only. It mirrors deterministic values from
Serenity's Chroma sampler/model/setup path:

* ChromaSampler.sample quantizes image size by 64 before __sample_base.
* ChromaSampler.__sample_base fixes VAE scale 8, 16 latent channels, CFG batch
  2, packed latent ids, FlowMatch scheduler set_timesteps, timestep / 1000,
  plain CFG combine, and VAE decode scaling.
* ChromaModel.encode_text, prepare_latent_image_ids, pack_latents, and
  unpack_latents define the text-mask and packing contracts.
* BaseChromaSetup.predict defines training-side latent scaling, flow noising,
  target, and attention-mask conventions used by this bounded gate.

It does not tokenize prompts, run the T5 encoder, run transformer inference,
sample random noise, execute the scheduler tensor step, decode with the VAE,
postprocess an image, save output, or claim end-to-end sampler parity.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SERENITY_ROOT = Path("/home/alex/Serenity")
DEFAULT_OUT = ROOT / "parity" / "chroma_sampler_helper_ref.json"
DEFAULT_CHROMA_MODEL = Path(
    "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/"
    "snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0"
)

SOURCE_FILES = [
    str(SERENITY_ROOT / "modules/modelSampler/ChromaSampler.py"),
    str(SERENITY_ROOT / "modules/model/ChromaModel.py"),
    str(SERENITY_ROOT / "modules/modelSetup/BaseChromaSetup.py"),
]
INHERITED_QUANTIZE_SOURCE = str(
    SERENITY_ROOT / "modules/modelSampler/BaseModelSampler.py"
)
CALLED_SETUP_HELPER_SOURCES = [
    str(SERENITY_ROOT / "modules/modelSetup/mixin/ModelSetupFlowMatchingMixin.py"),
    str(SERENITY_ROOT / "modules/modelSetup/mixin/ModelSetupNoiseMixin.py"),
]

if str(SERENITY_ROOT) not in sys.path:
    sys.path.insert(0, str(SERENITY_ROOT))

try:
    from modules.modelSampler.BaseModelSampler import BaseModelSampler
except Exception as exc:  # pragma: no cover - blocker path depends on env
    BaseModelSampler = None  # type: ignore[assignment]
    BASE_MODEL_SAMPLER_IMPORT_ERROR = exc
else:
    BASE_MODEL_SAMPLER_IMPORT_ERROR = None


def _add_blocker(blockers: list[str], message: str) -> None:
    if message not in blockers:
        blockers.append(message)


def _quantize_resolution(resolution: int, quantization: int, blockers: list[str]) -> int:
    if BaseModelSampler is not None:
        return int(BaseModelSampler.quantize_resolution(resolution, quantization))

    _add_blocker(
        blockers,
        "could not import Serenity BaseModelSampler.quantize_resolution "
        f"from {INHERITED_QUANTIZE_SOURCE}: {BASE_MODEL_SAMPLER_IMPORT_ERROR!r}; "
        "using Python round(resolution / quantization) * quantization fallback",
    )
    return round(resolution / quantization) * quantization


def _try_import_diffusers(blockers: list[str]):
    try:
        import diffusers
        from diffusers import FlowMatchEulerDiscreteScheduler
    except Exception as exc:  # pragma: no cover - blocker path depends on env
        _add_blocker(blockers, f"could not import diffusers FlowMatchEulerDiscreteScheduler: {exc!r}")
        return None, None
    return diffusers, FlowMatchEulerDiscreteScheduler


def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return config[key]
    except Exception:
        return default


def _flow_shift_sigma(sigma: float, shift: float) -> float:
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def _fallback_schedule(diffusion_steps: int, num_train_timesteps: int, shift: float) -> dict:
    n_train = float(num_train_timesteps)
    sigma_min = _flow_shift_sigma(1.0 / n_train, shift)
    t_start = n_train
    t_end = sigma_min * n_train
    sigmas: list[float] = []
    timesteps: list[float] = []
    for i in range(diffusion_steps):
        if diffusion_steps == 1:
            timestep = t_start
        else:
            frac = float(i) / float(diffusion_steps - 1)
            timestep = t_start + frac * (t_end - t_start)
        sigma = _flow_shift_sigma(timestep / n_train, shift)
        sigmas.append(float(sigma))
        timesteps.append(float(sigma * n_train))
    sigmas.append(0.0)
    return {
        "scheduler_class": "fallback static FlowMatch formula",
        "scheduler_step_accepts_generator": False,
        "schedule_sigmas": sigmas,
        "schedule_timesteps": timesteps,
    }


def _fallback_scheduler_instance(scheduler_cls: Any, config: dict):
    try:
        return scheduler_cls(**config)
    except TypeError:
        supported = set(inspect.signature(scheduler_cls.__init__).parameters)
        filtered = {key: value for key, value in config.items() if key in supported}
        return scheduler_cls(**filtered)


def _load_schedule(
    model_dir: Path | None,
    diffusion_steps: int,
    blockers: list[str],
) -> tuple[dict, dict, str, str | None]:
    diffusers, scheduler_cls = _try_import_diffusers(blockers)
    fallback_config = {
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "use_dynamic_shifting": False,
        "invert_sigmas": False,
        "stochastic_sampling": False,
    }

    if scheduler_cls is None:
        schedule = _fallback_schedule(
            diffusion_steps,
            fallback_config["num_train_timesteps"],
            fallback_config["shift"],
        )
        return (
            fallback_config,
            schedule,
            "fallback static FlowMatch formula because diffusers is unavailable",
            None,
        )

    scheduler_source: str
    if model_dir is not None and (model_dir / "scheduler" / "scheduler_config.json").is_file():
        try:
            scheduler = scheduler_cls.from_pretrained(model_dir, subfolder="scheduler")
            scheduler_source = str(model_dir / "scheduler" / "scheduler_config.json")
        except Exception as exc:  # pragma: no cover - depends on local cache
            _add_blocker(
                blockers,
                f"could not load Chroma scheduler from {model_dir / 'scheduler'}: {exc!r}; "
                "using explicit fallback FlowMatch scheduler config",
            )
            scheduler = _fallback_scheduler_instance(scheduler_cls, fallback_config)
            scheduler_source = "explicit fallback FlowMatchEulerDiscreteScheduler config"
    else:
        _add_blocker(
            blockers,
            f"missing Chroma scheduler config at {model_dir / 'scheduler' / 'scheduler_config.json' if model_dir else '<none>'}; "
            "using explicit fallback FlowMatch scheduler config",
        )
        scheduler = _fallback_scheduler_instance(scheduler_cls, fallback_config)
        scheduler_source = "explicit fallback FlowMatchEulerDiscreteScheduler config"

    scheduler = scheduler_cls.from_config(scheduler.config)
    scheduler.set_timesteps(diffusion_steps)
    sigmas = [float(x) for x in scheduler.sigmas.detach().cpu().tolist()]
    timesteps = [float(x) for x in scheduler.timesteps.detach().cpu().tolist()]
    config = {
        "num_train_timesteps": int(_config_get(scheduler.config, "num_train_timesteps", 1000)),
        "shift": float(_config_get(scheduler.config, "shift", 3.0)),
        "use_dynamic_shifting": bool(_config_get(scheduler.config, "use_dynamic_shifting", False)),
        "invert_sigmas": bool(_config_get(scheduler.config, "invert_sigmas", False)),
        "stochastic_sampling": bool(_config_get(scheduler.config, "stochastic_sampling", False)),
    }
    schedule = {
        "scheduler_class": scheduler.__class__.__name__,
        "scheduler_step_accepts_generator": "generator" in inspect.signature(scheduler.step).parameters,
        "schedule_sigmas": sigmas,
        "schedule_timesteps": timesteps,
    }
    return config, schedule, scheduler_source, getattr(diffusers, "__version__", None)


def _vae_config(model_dir: Path | None, blockers: list[str]) -> tuple[float, float, str]:
    fallback_scaling = 0.3611
    fallback_shift = 0.1159
    config_path = model_dir / "vae" / "config.json" if model_dir is not None else None
    if config_path is None or not config_path.is_file():
        _add_blocker(
            blockers,
            f"missing Chroma VAE config at {config_path if config_path else '<none>'}; "
            f"using fallback scaling_factor={fallback_scaling}, shift_factor={fallback_shift}",
        )
        return fallback_scaling, fallback_shift, "fallback Chroma VAE constants"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return float(data["scaling_factor"]), float(data["shift_factor"]), str(config_path)
    except Exception as exc:  # pragma: no cover - depends on local cache
        _add_blocker(
            blockers,
            f"could not read Chroma VAE config {config_path}: {exc!r}; "
            f"using fallback scaling_factor={fallback_scaling}, shift_factor={fallback_shift}",
        )
        return fallback_scaling, fallback_shift, "fallback Chroma VAE constants"


def _text_mask_contract(pos_tokens: int, neg_tokens: int, token_capacity: int) -> dict:
    # ChromaModel.encode_text tokenizer branch:
    # seq_lengths = tokens_mask.sum(dim=1)
    # bool_attention_mask = (mask_indices <= seq_lengths.unsqueeze(1))
    pos_bool = min(pos_tokens + 1, token_capacity)
    neg_bool = min(neg_tokens + 1, token_capacity)
    max_seq_length = max(pos_bool, neg_bool)
    pads = max_seq_length % 16 > 0 and (pos_bool != max_seq_length or neg_bool != max_seq_length)
    if pads:
        max_seq_length += 16 - max_seq_length % 16
    return {
        "positive_input_tokens": pos_tokens,
        "negative_input_tokens": neg_tokens,
        "token_capacity": token_capacity,
        "positive_bool_tokens": pos_bool,
        "negative_bool_tokens": neg_bool,
        "text_max_seq_length": max_seq_length,
        "text_pads_to_16_because_lengths_differ": pads,
        "text_ids_rows": max_seq_length,
        "text_ids_cols": 3,
        "sampler_tokenized_prompt_unmasks_one_token": True,
        "cached_tokens_mask_keeps_exact_mask": True,
    }


def build_ref(model_dir: Path | None, diffusion_steps: int) -> dict:
    blockers: list[str] = []

    height_in = 1025
    width_in = 1120
    quantization = 64
    vae_scale_factor = 8
    latent_channels = 16
    pack_size = 2
    latent_batch = 1

    height = _quantize_resolution(height_in, quantization, blockers)
    width = _quantize_resolution(width_in, quantization, blockers)
    quantize_1025 = _quantize_resolution(1025, quantization, blockers)
    quantize_1056 = _quantize_resolution(1056, quantization, blockers)
    quantize_1120 = _quantize_resolution(1120, quantization, blockers)
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor
    packed_seq_len = (latent_h // pack_size) * (latent_w // pack_size)
    packed_channels = latent_channels * 4

    scheduler_config, schedule, scheduler_source, diffusers_version = _load_schedule(
        model_dir, diffusion_steps, blockers
    )
    vae_scaling_factor, vae_shift_factor, vae_source = _vae_config(model_dir, blockers)
    text_contract = _text_mask_contract(7, 23, 512)

    cfg_positive = 3.0
    cfg_negative = 1.0
    cfg_scale = 3.5
    cfg_combine = cfg_negative + cfg_scale * (cfg_positive - cfg_negative)

    euler_sample = 0.25
    euler_model_output = 0.5
    sigmas = schedule["schedule_sigmas"]
    timesteps = schedule["schedule_timesteps"]
    euler_value = euler_sample + (sigmas[1] - sigmas[0]) * euler_model_output

    decode_latent_sample = 0.5
    decode_input_sample = decode_latent_sample / vae_scaling_factor + vae_shift_factor
    train_latent_sample = 0.5
    train_noise_sample = 0.25
    train_scaled_latent_sample = (train_latent_sample - vae_shift_factor) * vae_scaling_factor
    deterministic_timestep = int(scheduler_config["num_train_timesteps"] * 0.5) - 1
    flow_sigma = (deterministic_timestep + 1) / scheduler_config["num_train_timesteps"]
    train_noisy_sample = train_noise_sample * flow_sigma + train_scaled_latent_sample * (1.0 - flow_sigma)
    train_target_sample = train_noise_sample - train_scaled_latent_sample
    train_predicted_scaled_latent_sample = train_noisy_sample - 0.125 * flow_sigma
    shifted_timestep_sample = (
        scheduler_config["num_train_timesteps"]
        * scheduler_config["shift"]
        * 500.0
        / ((scheduler_config["shift"] - 1.0) * 500.0 + scheduler_config["num_train_timesteps"])
    )

    sample_channel = 3
    sample_latent_y = 5
    sample_latent_x = 7
    sample_seq = (sample_latent_y // pack_size) * (latent_w // pack_size) + (sample_latent_x // pack_size)
    sample_packed_channel = sample_channel * 4 + (sample_latent_y % pack_size) * 2 + (sample_latent_x % pack_size)

    image_id_sample_tile_y = 2
    image_id_sample_tile_x = 3
    image_id_sample_row = image_id_sample_tile_y * (latent_w // pack_size) + image_id_sample_tile_x

    return {
        "artifact": "chroma_sampler_helper_ref",
        "scope": "helper-only; not end-to-end sampler parity",
        "runtime_reference_complete": len(blockers) == 0,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "blockers_text": "; ".join(blockers),
        "source": {
            "serenity_files": SOURCE_FILES,
            "inherited_quantize_source": INHERITED_QUANTIZE_SOURCE,
            "called_setup_helper_sources": CALLED_SETUP_HELPER_SOURCES,
            "scheduler": "diffusers.FlowMatchEulerDiscreteScheduler",
            "scheduler_source": scheduler_source,
            "scheduler_set_timesteps": "ChromaSampler.__sample_base copies model.noise_scheduler and calls set_timesteps(diffusion_steps)",
            "diffusers_version": diffusers_version,
            "vae_source": vae_source,
        },
        "inputs": {
            "height": height_in,
            "width": width_in,
            "resolution_quantization": quantization,
            "vae_scale_factor": vae_scale_factor,
            "latent_channels": latent_channels,
            "latent_batch_size": latent_batch,
            "diffusion_steps": diffusion_steps,
            "num_train_timesteps": scheduler_config["num_train_timesteps"],
            "scheduler_shift": scheduler_config["shift"],
            "scheduler_use_dynamic_shifting": scheduler_config["use_dynamic_shifting"],
            "scheduler_invert_sigmas": scheduler_config["invert_sigmas"],
            "scheduler_stochastic_sampling": scheduler_config["stochastic_sampling"],
            "cfg_positive": cfg_positive,
            "cfg_negative": cfg_negative,
            "cfg_scale": cfg_scale,
            "euler_sample": euler_sample,
            "euler_model_output": euler_model_output,
            "train_latent_sample": train_latent_sample,
            "train_noise_sample": train_noise_sample,
            "train_predicted_flow_sample": 0.125,
        },
        "quantize_1025_64": quantize_1025,
        "quantize_1056_64": quantize_1056,
        "quantize_1120_64": quantize_1120,
        "plan_height": height,
        "plan_width": width,
        "plan_latent_h": latent_h,
        "plan_latent_w": latent_w,
        "plan_latent_channels": latent_channels,
        "plan_latent_batch": latent_batch,
        "plan_cfg_batch": 2,
        "packed_seq_len": packed_seq_len,
        "packed_channels": packed_channels,
        "image_ids_rows": packed_seq_len,
        "image_ids_cols": 3,
        "image_ids_first_0": 0,
        "image_ids_first_1": 0,
        "image_ids_first_2": 0,
        "image_ids_last_0": 0,
        "image_ids_last_1": latent_h // pack_size - 1,
        "image_ids_last_2": latent_w // pack_size - 1,
        "image_id_sample_row": image_id_sample_row,
        "image_id_sample_1": image_id_sample_tile_y,
        "image_id_sample_2": image_id_sample_tile_x,
        "pack_sample_channel": sample_channel,
        "pack_sample_latent_y": sample_latent_y,
        "pack_sample_latent_x": sample_latent_x,
        "pack_sample_sequence_index": sample_seq,
        "pack_sample_packed_channel": sample_packed_channel,
        **text_contract,
        "attention_mask_rows": 2,
        "attention_mask_cols": text_contract["text_max_seq_length"] + packed_seq_len,
        "image_attention_mask_all_true": True,
        "sampler_always_passes_attention_mask": True,
        "training_passes_attention_mask_when_text_not_all_true": True,
        "training_omits_attention_mask_when_text_all_true": True,
        "text_ids_all_zero": True,
        "uses_negative_prompt": True,
        "has_cfg_rescale": False,
        "cfg_combine": cfg_combine,
        "scheduler_class": schedule["scheduler_class"],
        "scheduler_num_train_timesteps": scheduler_config["num_train_timesteps"],
        "scheduler_shift": scheduler_config["shift"],
        "scheduler_use_dynamic_shifting": scheduler_config["use_dynamic_shifting"],
        "scheduler_invert_sigmas": scheduler_config["invert_sigmas"],
        "scheduler_stochastic_sampling": scheduler_config["stochastic_sampling"],
        "extra_step_kwargs_may_include_generator": schedule["scheduler_step_accepts_generator"],
        "flow_shift_sigma_0_5": _flow_shift_sigma(0.5, scheduler_config["shift"]),
        "schedule_timesteps_len": len(timesteps),
        "schedule_sigmas_len": len(sigmas),
        "schedule_timesteps": timesteps,
        "schedule_sigmas": sigmas,
        "transformer_timestep_divisor": 1000.0,
        "model_timestep_1": timesteps[1] / 1000.0,
        "euler_value": euler_value,
        "vae_scaling_factor": vae_scaling_factor,
        "vae_shift_factor": vae_shift_factor,
        "decode_latent_sample": decode_latent_sample,
        "decode_input_sample": decode_input_sample,
        "decode_formula": "(latent_image / vae.config.scaling_factor) + vae.config.shift_factor",
        "train_scaled_latent_sample": train_scaled_latent_sample,
        "deterministic_timestep_index": deterministic_timestep,
        "shifted_timestep_sample": shifted_timestep_sample,
        "flow_sigma_sample": flow_sigma,
        "flow_one_minus_sigma_sample": 1.0 - flow_sigma,
        "train_noisy_sample": train_noisy_sample,
        "train_target_sample": train_target_sample,
        "train_predicted_scaled_latent_sample": train_predicted_scaled_latent_sample,
        "postprocess_output_type": "pil",
        "output_file_type": "IMAGE",
        "initial_noise_dtype": "torch.float32",
        "transformer_input_dtype": "model.train_dtype.torch_dtype()",
        "prompt_embedding_input_dtype": "model.train_dtype.torch_dtype()",
        "id_input_dtype": "model.train_dtype.torch_dtype()",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_CHROMA_MODEL)
    parser.add_argument("--diffusion-steps", type=int, default=4)
    args = parser.parse_args()

    model_dir = args.model_dir if args.model_dir else None
    ref = build_ref(model_dir, args.diffusion_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ref, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(
        "plan",
        f"{ref['plan_height']}x{ref['plan_width']}",
        "latent",
        f"{ref['plan_latent_h']}x{ref['plan_latent_w']}",
        "packed",
        f"{ref['packed_seq_len']}x{ref['packed_channels']}",
    )
    print(
        "schedule",
        f"shift={ref['scheduler_shift']}",
        f"sigma1={ref['schedule_sigmas'][1]}",
        f"timestep1={ref['schedule_timesteps'][1]}",
    )
    if ref["blocker_count"]:
        print(f"blockers: {ref['blockers_text']}")


if __name__ == "__main__":
    main()
