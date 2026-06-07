#!/usr/bin/env python3
"""Produce OneTrainer Klein/Flux2 sampler artifacts for parity infrastructure.

This is a development/parity helper, not product runtime. It follows
OneTrainer's Flux2Sampler/Flux2Model path and writes the OneTrainer-side
artifacts needed before a Mojo numeric comparison can be accepted.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ONETRAINER = Path("/home/alex/OneTrainer")
DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-9B"
DEFAULT_OUT_DIR = Path("/home/alex/onetrainer-mojo/parity/klein_sampler_parity")
DEFAULT_MANIFEST = DEFAULT_OUT_DIR / "manifest_fragment.json"
DEFAULT_CHECKER_MANIFEST = Path("/home/alex/onetrainer-mojo/parity/klein_sampler_parity_manifest.json")

DEFAULT_PROMPT_ID = "alina_garden_fast512"
DEFAULT_PROMPT = (
    "alverone , a high-resolution photograph featuring a young caucasian woman "
    "with long blonde hair, wearing a casual white sundress, standing in a "
    "sunlit garden, soft natural lighting, professional photography"
)
DEFAULT_NEGATIVE = "low quality, blurry, distorted anatomy, text, watermark"
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_STEPS = 1
DEFAULT_CFG_SCALE = 1.0
DEFAULT_SEED = 42


def _install_onetrainer_imports() -> None:
    scripts_dir = ONETRAINER / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from util.import_util import script_imports

    os.chdir(ONETRAINER)
    script_imports()


def _load_sample_prompt(path: Path, prompt_id: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        defaults: dict[str, Any] = {}
        prompts = data
    elif isinstance(data, dict):
        defaults = data.get("defaults", {})
        prompts = data.get("prompts", [])
    else:
        raise ValueError(f"invalid sample prompt file: {path}")
    if not isinstance(defaults, dict) or not isinstance(prompts, list):
        raise ValueError(f"invalid sample prompt file: {path}")

    selected: dict[str, Any] | None = None
    for index, prompt in enumerate(prompts):
        label = f"p{index + 1}"
        if isinstance(prompt, dict) and (
            prompt.get("id") == prompt_id
            or prompt.get("label") == prompt_id
            or label == prompt_id
            or str(index) == prompt_id
        ):
            selected = prompt
            break
    if selected is None:
        if prompt_id:
            raise ValueError(f"prompt id {prompt_id!r} not found in {path}")
        selected = prompts[0] if prompts else {}
    if not isinstance(selected, dict):
        raise ValueError(f"invalid prompt entry in {path}")

    merged = dict(defaults)
    merged.update(selected)
    return {
        "id": str(merged.get("id", prompt_id or "sample")),
        "prompt": str(merged.get("prompt", "")),
        "negative": str(merged.get("negative", merged.get("negative_prompt", ""))),
        "width": int(merged.get("width", DEFAULT_WIDTH)),
        "height": int(merged.get("height", DEFAULT_HEIGHT)),
        "steps": int(merged.get("steps", merged.get("diffusion_steps", DEFAULT_STEPS))),
        "cfg_scale": float(merged.get("cfg", merged.get("cfg_scale", merged.get("guidance", DEFAULT_CFG_SCALE)))),
        "seed": int(merged.get("seed", DEFAULT_SEED)),
    }


def _quantize_resolution(resolution: int, quantization: int = 64) -> int:
    return round(resolution / quantization) * quantization


def _resolve_sample(args: argparse.Namespace) -> dict[str, Any]:
    sample = {
        "id": DEFAULT_PROMPT_ID,
        "prompt": DEFAULT_PROMPT,
        "negative": DEFAULT_NEGATIVE,
        "width": DEFAULT_WIDTH,
        "height": DEFAULT_HEIGHT,
        "steps": DEFAULT_STEPS,
        "cfg_scale": DEFAULT_CFG_SCALE,
        "seed": DEFAULT_SEED,
    }
    if args.sample_file is not None:
        sample.update(_load_sample_prompt(args.sample_file, args.sample_id))

    prompt = args.prompt or sample["prompt"]
    negative = sample["negative"] if args.negative is None else args.negative
    if not str(negative).strip():
        if args.negative is not None:
            raise ValueError("negative prompt must be non-empty")
        negative = DEFAULT_NEGATIVE

    requested_width = int(sample["width"] if args.width is None else args.width)
    requested_height = int(sample["height"] if args.height is None else args.height)
    width = _quantize_resolution(requested_width)
    height = _quantize_resolution(requested_height)
    steps = int(sample["steps"] if args.steps is None else args.steps)
    cfg_scale = float(sample["cfg_scale"] if args.cfg_scale is None else args.cfg_scale)
    seed = int(sample["seed"] if args.seed is None else args.seed)

    if not str(prompt).strip():
        raise ValueError("prompt must be non-empty")
    if width <= 0 or height <= 0 or steps <= 0:
        raise ValueError("width/height/steps must be positive after OneTrainer-style quantization")
    if not math.isfinite(cfg_scale) or cfg_scale <= 0.0:
        raise ValueError("cfg scale must be positive finite")

    return {
        "id": args.sample_id or sample["id"],
        "positive": str(prompt),
        "negative": str(negative),
        "seed": seed,
        "width": width,
        "height": height,
        "steps": steps,
        "random_seed": bool(args.random_seed),
        "cfg_scale": cfg_scale,
        "requested_width": requested_width,
        "requested_height": requested_height,
    }


def _gpu_memory_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    values: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(line.split()[0]))
        except ValueError:
            pass
    return max(values) if values else None


class VramPoller:
    def __init__(self, interval: float) -> None:
        self.interval = max(interval, 0.05)
        self.baseline_mib: int | None = None
        self.peak_mib: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        used = _gpu_memory_used_mib()
        if used is None:
            return
        if self.baseline_mib is None:
            self.baseline_mib = used
        self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(self.interval):
            self._sample()
        self._sample()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def _sync_cuda(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


@contextmanager
def _timed_stage(torch_module: Any) -> Any:
    _sync_cuda(torch_module)
    start = time.perf_counter()
    bucket: dict[str, float] = {}
    try:
        yield bucket
    finally:
        _sync_cuda(torch_module)
        bucket["seconds"] = time.perf_counter() - start


def _dtype_to_onetrainer(dtype_name: str) -> Any:
    from modules.util.enum.DataType import DataType

    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16", "bfloat_16"}:
        return DataType.BFLOAT_16
    if normalized in {"f32", "fp32", "float32", "float_32"}:
        return DataType.FLOAT_32
    raise ValueError("dtype must be bf16/bfloat16 or f32/float32")


def _build_weight_dtypes(dtype_name: str) -> Any:
    from modules.util.ModelWeightDtypes import ModelWeightDtypes
    from modules.util.enum.DataType import DataType

    train_dtype = _dtype_to_onetrainer(dtype_name)
    return ModelWeightDtypes(
        train_dtype=train_dtype,
        fallback_train_dtype=train_dtype,
        unet=train_dtype,
        prior=train_dtype,
        transformer=train_dtype,
        text_encoder=train_dtype,
        text_encoder_2=train_dtype,
        text_encoder_3=train_dtype,
        text_encoder_4=train_dtype,
        vae=DataType.FLOAT_32,
        effnet_encoder=train_dtype,
        decoder=train_dtype,
        decoder_text_encoder=train_dtype,
        decoder_vqgan=train_dtype,
        lora=train_dtype,
        embedding=train_dtype,
    )


def _preflight_model_args(args: argparse.Namespace) -> None:
    base_path = Path(args.base_model)
    if base_path.exists() and base_path.is_file():
        raise ValueError(
            "OneTrainer Flux2ModelLoader does not load a single-file Flux2 base model. "
            "Use a diffusers directory or HF repo id for --base-model, and optionally "
            "pass a transformer override with --transformer-model."
        )


def _load_model(args: argparse.Namespace, device: Any, temp_device: Any) -> Any:
    from modules.util import create
    from modules.util.ModelNames import ModelNames
    from modules.util.config.TrainConfig import QuantizationConfig, TrainConfig
    from modules.util.enum.ModelType import ModelType
    from modules.util.enum.TrainingMethod import TrainingMethod

    _preflight_model_args(args)
    training_method = TrainingMethod.LORA if args.lora else TrainingMethod.FINE_TUNE
    model_loader = create.create_model_loader(ModelType.FLUX_2, training_method=training_method)
    model_setup = create.create_model_setup(
        ModelType.FLUX_2,
        device,
        temp_device,
        training_method=training_method,
    )
    if model_loader is None or model_setup is None:
        raise RuntimeError("could not create OneTrainer Flux2 loader/setup")

    dtype = _dtype_to_onetrainer(args.dtype)
    train_config = TrainConfig.default_values().from_dict(
        {
            "__version": 10,
            "model_type": "FLUX_2",
            "training_method": "LORA" if args.lora else "FINE_TUNE",
            "base_model_name": args.base_model,
            "train_dtype": str(dtype),
            "fallback_train_dtype": str(dtype),
            "weight_dtype": str(dtype),
            "output_dtype": "BFLOAT_16",
            "gradient_checkpointing": "OFF",
            "compile": False,
            "layer_offload_fraction": 0.0,
            "latent_caching": True,
            "text_encoder": {"train": False, "weight_dtype": str(dtype)},
            "transformer": {"train": False, "weight_dtype": str(dtype)},
            "vae": {"weight_dtype": "FLOAT_32"},
        }
    )
    train_config.quantization = QuantizationConfig.default_values()

    model_names = ModelNames(
        base_model=args.base_model,
        transformer_model=args.transformer_model or "",
        vae_model=args.vae_model or "",
        lora=str(args.lora or ""),
    )
    model = model_loader.load(
        model_type=ModelType.FLUX_2,
        model_names=model_names,
        weight_dtypes=_build_weight_dtypes(args.dtype),
        quantization=train_config.quantization,
    )
    model.train_config = train_config
    model.train_dtype = dtype
    model.text_encoder_train_dtype = dtype

    if args.lora:
        model_setup.setup_model(model, train_config)
    else:
        model_setup.setup_optimizations(model, train_config)
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)
    model.eval()

    if not model.is_klein():
        raise RuntimeError(
            "loaded FLUX_2 model is Flux2-dev, not Klein; transformer.config.num_attention_heads == 48"
        )
    return model


def _save_tensor(path: Path, tensor: Any, *, note: str = "") -> dict[str, Any]:
    from safetensors.torch import save_file
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    cpu = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    save_file({"tensor": cpu}, str(path))
    artifact: dict[str, Any] = {
        "path": str(path),
        "dtype": "F32",
        "shape": [int(dim) for dim in cpu.shape],
        "format": "safetensors",
        "tensor_key": "tensor",
        "byte_size": path.stat().st_size,
    }
    if note:
        artifact["note"] = note
    return artifact


def _png_artifact(path: Path, width: int, height: int) -> dict[str, Any]:
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "format": "png",
        "byte_size": path.stat().st_size if path.exists() else 0,
    }


def _to_float_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _scheduler_metadata(noise_scheduler: Any, steps: int, mu: float, input_sigmas: Any) -> dict[str, Any]:
    sigmas = _to_float_list(noise_scheduler.sigmas)
    timesteps = _to_float_list(noise_scheduler.timesteps)
    step_trace = []
    for index in range(steps):
        sigma = sigmas[index] if index < len(sigmas) else float("nan")
        sigma_next = sigmas[index + 1] if index + 1 < len(sigmas) else float("nan")
        timestep = timesteps[index] if index < len(timesteps) else float("nan")
        step_trace.append(
            {
                "index": index,
                "sigma": sigma,
                "sigma_next": sigma_next,
                "dt": sigma_next - sigma,
                "timestep": timestep,
            }
        )
    return {
        "name": "FlowMatchEuler",
        "mu": float(mu),
        "input_sigmas": _to_float_list(input_sigmas),
        "sigmas": sigmas,
        "timesteps": timesteps,
        "step_trace": step_trace,
    }


def _artifact_groups() -> dict[str, Any]:
    return {
        "onetrainer_seed_replay_inputs": [
            "onetrainer_initial_noise_raw_nchw",
            "onetrainer_initial_noise_post_patch_nchw",
            "onetrainer_initial_noise_post_pack",
        ],
        "onetrainer_latent_trajectory": [
            "onetrainer_latent_trajectory",
            "onetrainer_final_packed_latent",
            "onetrainer_final_unpacked_latent",
            "onetrainer_final_unscaled_unpatchified_latent",
        ],
        "onetrainer_decode_and_png": [
            "onetrainer_vae_decoded_tensor",
            "onetrainer_png",
        ],
        "missing_for_sampler_parity": [
            "mojo_latent_trajectory",
            "mojo_final_packed_latent",
            "mojo_final_unpacked_latent",
            "mojo_final_unscaled_unpatchified_latent",
            "mojo_vae_decoded_tensor",
            "mojo_png",
            "numeric_comparisons",
        ],
    }


def _install_repo_scripts_imports() -> None:
    scripts_dir = REPO / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def _checker_static_sections() -> dict[str, Any]:
    _install_repo_scripts_imports()
    try:
        from check_klein_sampler_artifact_manifest import (
            current_split_validation_evidence,
            required_artifacts,
            required_manifest_fields,
            required_paired_sampler_fields,
        )

        return {
            "current_split_validation_evidence": current_split_validation_evidence(),
            "required_manifest_fields": required_manifest_fields(),
            "required_paired_sampler_fields": required_paired_sampler_fields(),
            "required_paired_sampler_artifacts": required_artifacts(),
        }
    except Exception as exc:  # noqa: BLE001 - this helper should still write the OT fragment.
        return {
            "current_split_validation_evidence": {
                "status": "support_only_not_sampler_parity",
                "sampler_parity_accepted": False,
                "smoke_images_accepted_parity": False,
                "note": f"could not import checker static sections: {exc}",
            }
        }


def _comparison_stubs() -> dict[str, Any]:
    return {
        key: {
            "accepted": False,
            "max_abs": None,
            "tolerance": None,
            "note": "No Mojo numeric comparison has been run; do not accept sampler parity.",
        }
        for key in ("trajectory", "final_latent", "vae_png")
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ot-klein-sampler] wrote manifest: {path}")


def _write_checker_template(path: Path, args: argparse.Namespace, sample: dict[str, Any]) -> None:
    _install_repo_scripts_imports()
    from check_klein_sampler_artifact_manifest import build_template_manifest

    manifest = build_template_manifest(
        target_manifest=args.checker_manifest,
        artifact_root=str(args.out_dir),
        width=int(sample["width"]),
        height=int(sample["height"]),
        steps=int(sample["steps"]),
        prompt_id=str(sample["id"]),
        positive=str(sample["positive"]),
        negative=str(sample["negative"]),
        seed=int(sample["seed"]),
        random_seed=bool(sample["random_seed"]),
        cfg_scale=float(sample["cfg_scale"]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ot-klein-sampler] wrote checker template: {path} parity_claimed=false")


def _write_checker_readiness(path: Path, manifest: Path) -> None:
    _install_repo_scripts_imports()
    from check_klein_sampler_artifact_manifest import Check, inspect_manifest, write_readiness_report

    try:
        checks = inspect_manifest(manifest)
    except Exception as exc:  # noqa: BLE001 - keep this no-CUDA path report-first.
        checks = [Check(False, "manifest", str(exc))]
    write_readiness_report(path, manifest, checks)


def _cuda_peak_allocated_mib(torch_module: Any, device: Any) -> float | None:
    if device.type != "cuda" or not torch_module.cuda.is_available():
        return None
    return float(torch_module.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)


def _run_onetrainer_sample(
    model: Any,
    sample: dict[str, Any],
    *,
    out_dir: Path,
    device: Any,
    temp_device: Any,
    text_encoder_sequence_length: int | None,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu
    from modules.util.torch_util import torch_gc

    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    timing: dict[str, float] = {}
    prompt = sample["positive"]
    negative = sample["negative"]
    width = int(sample["width"])
    height = int(sample["height"])
    steps = int(sample["steps"])
    cfg_scale = float(sample["cfg_scale"])

    with torch.no_grad(), model.autocast_context:
        generator = torch.Generator(device=device)
        if sample["random_seed"]:
            actual_seed = int(generator.seed())
        else:
            actual_seed = int(sample["seed"])
            generator.manual_seed(actual_seed)

        noise_scheduler = copy.deepcopy(model.noise_scheduler)
        pipeline = model.create_pipeline()
        image_processor = pipeline.image_processor
        transformer = pipeline.transformer
        vae = pipeline.vae

        vae_scale_factor = 8
        num_latent_channels = 32
        patch_size = 2

        batch_size = 2 if cfg_scale > 1.0 and not transformer.config.guidance_embeds else 1
        with _timed_stage(torch) as text_timer:
            model.text_encoder_to(device)
            prompt_embedding = model.encode_text(
                text=[prompt, negative] if batch_size == 2 else prompt,
                train_device=device,
                text_encoder_sequence_length=text_encoder_sequence_length,
            )
            model.text_encoder_to(temp_device)
            torch_gc()
        timing["text_encode_seconds"] = text_timer["seconds"]

        latent_raw = torch.randn(
            size=(1, num_latent_channels, height // vae_scale_factor, width // vae_scale_factor),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        artifacts["onetrainer_initial_noise_raw_nchw"] = _save_tensor(
            out_dir / "onetrainer_initial_noise_raw_nchw.safetensors",
            latent_raw,
            note="OneTrainer Flux2Sampler raw PyTorch F32 noise before patchify_latents.",
        )

        latent_patch = model.patchify_latents(latent_raw)
        artifacts["onetrainer_initial_noise_post_patch_nchw"] = _save_tensor(
            out_dir / "onetrainer_initial_noise_post_patch_nchw.safetensors",
            latent_patch,
            note="OneTrainer Flux2Model.patchify_latents output.",
        )
        image_ids = model.prepare_latent_image_ids(latent_patch)

        latent_packed = model.pack_latents(latent_patch)
        artifacts["onetrainer_initial_noise_post_pack"] = _save_tensor(
            out_dir / "onetrainer_initial_noise_post_pack.safetensors",
            latent_packed.squeeze(0),
            note="OneTrainer Flux2Model.pack_latents output, squeezed to [tokens, 128] for the manifest guard.",
        )

        image_seq_len = int(latent_packed.shape[1])
        mu = float(compute_empirical_mu(image_seq_len, steps))
        input_sigmas = np.linspace(1.0, 1 / steps, steps)
        noise_scheduler.set_timesteps(steps, device=device, mu=mu, sigmas=input_sigmas)
        timesteps = noise_scheduler.timesteps
        scheduler = _scheduler_metadata(noise_scheduler, steps, mu, input_sigmas)

        extra_step_kwargs = {}
        if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
            extra_step_kwargs["generator"] = generator

        text_ids = model.prepare_text_ids(prompt_embedding)
        trajectory_tensors = [latent_packed.detach()]

        model.transformer_to(device)
        guidance = (
            torch.tensor([cfg_scale], device=device, dtype=model.train_dtype.torch_dtype())
            if transformer.config.guidance_embeds
            else None
        )
        with _timed_stage(torch) as denoise_timer:
            for index, timestep in enumerate(timesteps):
                latent_model_input = torch.cat([latent_packed] * batch_size)
                expanded_timestep = timestep.expand(latent_model_input.shape[0])

                noise_pred = transformer(
                    hidden_states=latent_model_input.to(dtype=model.train_dtype.torch_dtype()),
                    timestep=expanded_timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embedding.to(dtype=model.train_dtype.torch_dtype()),
                    txt_ids=text_ids,
                    img_ids=image_ids,
                    joint_attention_kwargs=None,
                    return_dict=True,
                ).sample

                if batch_size == 2:
                    noise_pred_positive, noise_pred_negative = noise_pred.chunk(2)
                    noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)

                latent_packed = noise_scheduler.step(
                    noise_pred,
                    timestep,
                    latent_packed,
                    return_dict=False,
                    **extra_step_kwargs,
                )[0]
                trajectory_tensors.append(latent_packed.detach())
                print(f"[ot-klein-sampler] denoise step={index + 1}/{len(timesteps)}")
        timing["denoise_seconds"] = denoise_timer["seconds"]
        timing["denoise_seconds_per_step"] = denoise_timer["seconds"] / float(max(steps, 1))
        model.transformer_to(temp_device)
        torch_gc()

        trajectory = torch.stack([item.squeeze(0).to(device="cpu", dtype=torch.float32) for item in trajectory_tensors], dim=0)
        artifacts["onetrainer_latent_trajectory"] = _save_tensor(
            out_dir / "onetrainer_latent_trajectory.safetensors",
            trajectory,
            note="OneTrainer packed latent trajectory including step 0, shape [steps + 1, tokens, 128].",
        )
        artifacts["onetrainer_final_packed_latent"] = _save_tensor(
            out_dir / "onetrainer_final_packed_latent.safetensors",
            latent_packed.squeeze(0),
            note="OneTrainer final packed latent, squeezed to [tokens, 128] for the manifest guard.",
        )

        with _timed_stage(torch) as vae_timer:
            model.vae_to(device)
            final_unpacked = model.unpack_latents(
                latent_packed,
                height // vae_scale_factor // patch_size,
                width // vae_scale_factor // patch_size,
            )
            final_unscaled = model.unscale_latents(final_unpacked)
            final_unscaled_unpatchified = model.unpatchify_latents(final_unscaled)
            decoded = vae.decode(final_unscaled_unpatchified, return_dict=False)[0]
            image = image_processor.postprocess(decoded, output_type="pil")
            model.vae_to(temp_device)
            torch_gc()
        timing["vae_decode_seconds"] = vae_timer["seconds"]

        artifacts["onetrainer_final_unpacked_latent"] = _save_tensor(
            out_dir / "onetrainer_final_unpacked_latent.safetensors",
            final_unpacked,
            note="OneTrainer Flux2Model.unpack_latents output before VAE batch-norm unscale.",
        )
        artifacts["onetrainer_final_unscaled_unpatchified_latent"] = _save_tensor(
            out_dir / "onetrainer_final_unscaled_unpatchified_latent.safetensors",
            final_unscaled_unpatchified,
            note="OneTrainer unscale_latents then unpatchify_latents output passed to the VAE decoder.",
        )
        artifacts["onetrainer_vae_decoded_tensor"] = _save_tensor(
            out_dir / "onetrainer_vae_decoded_tensor.safetensors",
            decoded,
            note="Raw OneTrainer VAE decoded tensor before image_processor.postprocess.",
        )
        png_path = out_dir / "onetrainer_png.png"
        image[0].save(png_path)
        artifacts["onetrainer_png"] = _png_artifact(png_path, width, height)

    sample = dict(sample)
    sample["seed"] = actual_seed
    return {
        "artifacts": artifacts,
        "scheduler": scheduler,
        "timing": timing,
        "prompt": sample,
    }


def _build_manifest(
    *,
    args: argparse.Namespace,
    prompt: dict[str, Any],
    run: dict[str, Any],
    elapsed_seconds: float,
    poller: VramPoller,
    cuda_peak_allocated_mib: float | None,
    command: list[str],
) -> dict[str, Any]:
    timing = dict(run["timing"])
    peak_vram_mib = poller.peak_mib if poller.peak_mib is not None else cuda_peak_allocated_mib
    onetrainer_metrics = {
        "denoise_seconds_per_step": float(timing.get("denoise_seconds_per_step", 0.0)),
        "vae_decode_seconds": float(timing.get("vae_decode_seconds", 0.0)),
        "peak_vram_mib": float(peak_vram_mib or 0.0),
    }

    manifest = {
        "schema_version": 1,
        "producer": "scripts/run_klein_onetrainer_sampler_parity.py",
        "scope": "OneTrainer-only Klein/Flux2 sampler artifact producer; not product runtime and not sampler parity.",
        "model_type": "FLUX_2",
        "variant": "klein",
        "accepted": False,
        "parity_claimed": False,
        "sampler_parity_accepted": False,
        "mojo_comparison_present": False,
        "parity_note": (
            "This manifest records OneTrainer-side artifacts only. Accepted/parity flags stay false "
            "until a paired Mojo run and numeric comparisons are present."
        ),
        "prompt": {
            "id": str(prompt["id"]),
            "positive": prompt["positive"],
            "negative": prompt["negative"],
            "seed": int(prompt["seed"]),
            "width": int(prompt["width"]),
            "height": int(prompt["height"]),
            "steps": int(prompt["steps"]),
            "random_seed": bool(prompt["random_seed"]),
            "cfg_scale": float(prompt["cfg_scale"]),
            "requested_width": int(prompt["requested_width"]),
            "requested_height": int(prompt["requested_height"]),
            "resolution_quantization": 64,
        },
        "scheduler": run["scheduler"],
        "artifact_groups": _artifact_groups(),
        "artifacts": run["artifacts"],
        "metrics": {
            "onetrainer": onetrainer_metrics,
            "mojo": {
                "denoise_seconds_per_step": 0.0,
                "vae_decode_seconds": 0.0,
                "peak_vram_mib": 0.0,
            },
        },
        "comparisons": _comparison_stubs(),
        "timing": {
            **timing,
            "elapsed_seconds": float(elapsed_seconds),
        },
        "memory": {
            "gpu_memory_baseline_mib": poller.baseline_mib,
            "peak_vram_mib": peak_vram_mib,
            "cuda_peak_allocated_mib": cuda_peak_allocated_mib,
        },
        "model": {
            "base_model": args.base_model,
            "transformer_model": args.transformer_model or "",
            "vae_model": args.vae_model or "",
            "lora": str(args.lora or ""),
            "dtype": args.dtype,
            "device": args.device,
            "temp_device": args.temp_device,
        },
        "measurement": {
            "source": "OneTrainer Flux2Sampler/Flux2Model via scripts/run_klein_onetrainer_sampler_parity.py",
            "command": command,
            "capture_instrumented": True,
        },
    }
    manifest.update(_checker_static_sections())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce OneTrainer-only Klein/Flux2 sampler artifacts for the manifest guard."
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--transformer-model", default="")
    parser.add_argument("--vae-model", default="")
    parser.add_argument("--lora", type=Path, default=None)
    parser.add_argument("--sample-file", type=Path, default=None)
    parser.add_argument("--sample-id", default=DEFAULT_PROMPT_ID)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative", default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--random-seed", action="store_true")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temp-device", default="cpu")
    parser.add_argument("--text-encoder-sequence-length", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--also-write-checker-manifest",
        action="store_true",
        help=f"Also write the fragment to {DEFAULT_CHECKER_MANIFEST} for checker invocation.",
    )
    parser.add_argument("--checker-manifest", type=Path, default=DEFAULT_CHECKER_MANIFEST)
    parser.add_argument(
        "--write-checker-template",
        type=Path,
        default=None,
        help="Write a paired-manifest template matching this sample without loading OneTrainer/CUDA.",
    )
    parser.add_argument(
        "--write-checker-readiness",
        type=Path,
        default=None,
        help="Write the checker readiness report for --manifest; with --no-run this is no-CUDA.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Only write requested checker template/readiness outputs; do not import OneTrainer or load CUDA.",
    )
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample = _resolve_sample(args)

    if args.write_checker_template is not None:
        _write_checker_template(args.write_checker_template, args, sample)
    if args.no_run:
        if args.write_checker_template is None and args.write_checker_readiness is None:
            print("[ot-klein-sampler] ERROR --no-run requires --write-checker-template or --write-checker-readiness")
            return 2
        if args.write_checker_readiness is not None:
            _write_checker_readiness(args.write_checker_readiness, args.manifest)
        print("[ot-klein-sampler] INFO no-run complete; no OneTrainer/CUDA execution; parity_claimed=false")
        return 0

    _install_onetrainer_imports()
    import torch

    device = torch.device(args.device)
    temp_device = torch.device(args.temp_device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    poller = VramPoller(args.poll_seconds)
    command = [sys.executable, *sys.argv]
    start = time.perf_counter()
    poller.start()
    try:
        model = _load_model(args, device, temp_device)
        run = _run_onetrainer_sample(
            model,
            sample,
            out_dir=args.out_dir,
            device=device,
            temp_device=temp_device,
            text_encoder_sequence_length=args.text_encoder_sequence_length,
        )
    finally:
        poller.stop()

    elapsed = time.perf_counter() - start
    cuda_peak = _cuda_peak_allocated_mib(torch, device)
    manifest = _build_manifest(
        args=args,
        prompt=run["prompt"],
        run=run,
        elapsed_seconds=elapsed,
        poller=poller,
        cuda_peak_allocated_mib=cuda_peak,
        command=command,
    )
    _write_manifest(args.manifest, manifest)
    if args.also_write_checker_manifest:
        _write_manifest(args.checker_manifest, manifest)
    if args.write_checker_readiness is not None:
        _write_checker_readiness(args.write_checker_readiness, args.manifest)

    print(
        "[ot-klein-sampler] PASS "
        f"artifacts={args.out_dir} manifest={args.manifest} "
        f"steps={sample['steps']} denoise_s_per_step={manifest['metrics']['onetrainer']['denoise_seconds_per_step']:.6f} "
        f"vae_s={manifest['metrics']['onetrainer']['vae_decode_seconds']:.6f} "
        f"peak_vram_mib={manifest['metrics']['onetrainer']['peak_vram_mib']:.1f} "
        "parity_claimed=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
