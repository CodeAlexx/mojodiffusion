#!/usr/bin/env python3
"""Run a timed OneTrainer Z-Image sampler reference.

This is a parity-evidence helper. It imports OneTrainer and executes the same
ZImageSampler logic, but writes explicit timing/VRAM metadata for the strict
Mojo gate. It does not modify OneTrainer and does not claim speed parity.
"""

from __future__ import annotations

import argparse
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
DEFAULT_BASE_MODEL = Path("/home/alex/.serenity/models/zimage_base")
DEFAULT_SAMPLE_FILE = Path("/home/alex/OneTrainer/training_samples/eri2_5prompts_1024.json")
DEFAULT_PROMPT_ID = "p1"
DEFAULT_OUT = Path("/home/alex/onetrainer-mojo/parity/zi_OT_timed_1024.png")
DEFAULT_SPEED_JSON = Path("/home/alex/onetrainer-mojo/parity/zimage_sampler_speed.json")


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
    selected = None
    for idx, prompt in enumerate(prompts):
        label = f"p{idx + 1}"
        if isinstance(prompt, dict) and (
            prompt.get("id") == prompt_id
            or prompt.get("label") == prompt_id
            or label == prompt_id
            or str(idx) == prompt_id
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
        "prompt": str(merged.get("prompt", "")),
        "negative": str(merged.get("negative", merged.get("negative_prompt", ""))),
        "width": int(merged.get("width", 1024)),
        "height": int(merged.get("height", 1024)),
        "steps": int(merged.get("steps", merged.get("diffusion_steps", 28))),
        "guidance": float(merged.get("cfg", merged.get("cfg_scale", merged.get("guidance", 4.0)))),
        "seed": int(merged.get("seed", 42)),
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


def _build_weight_dtypes(dtype_name: str) -> Any:
    from modules.util.ModelWeightDtypes import ModelWeightDtypes
    from modules.util.enum.DataType import DataType

    if dtype_name.lower() not in {"bf16", "bfloat16"}:
        raise ValueError("Z-Image reference wrapper only accepts bf16/bfloat16 dtype")
    return ModelWeightDtypes(
        train_dtype=DataType.BFLOAT_16,
        fallback_train_dtype=DataType.BFLOAT_16,
        unet=DataType.BFLOAT_16,
        prior=DataType.BFLOAT_16,
        transformer=DataType.BFLOAT_16,
        text_encoder=DataType.BFLOAT_16,
        text_encoder_2=DataType.BFLOAT_16,
        text_encoder_3=DataType.BFLOAT_16,
        text_encoder_4=DataType.BFLOAT_16,
        vae=DataType.FLOAT_32,
        effnet_encoder=DataType.BFLOAT_16,
        decoder=DataType.BFLOAT_16,
        decoder_text_encoder=DataType.BFLOAT_16,
        decoder_vqgan=DataType.BFLOAT_16,
        lora=DataType.BFLOAT_16,
        embedding=DataType.BFLOAT_16,
    )


def _load_model(args: argparse.Namespace) -> Any:
    import torch
    from modules.util import create
    from modules.util.ModelNames import ModelNames
    from modules.util.config.TrainConfig import QuantizationConfig, TrainConfig
    from modules.util.enum.DataType import DataType
    from modules.util.enum.ModelType import ModelType
    from modules.util.enum.TrainingMethod import TrainingMethod

    device = torch.device(args.device)
    temp_device = torch.device(args.temp_device)
    training_method = TrainingMethod.LORA if args.lora else TrainingMethod.FINE_TUNE
    model_loader = create.create_model_loader(ModelType.Z_IMAGE, training_method=training_method)
    model_setup = create.create_model_setup(
        ModelType.Z_IMAGE,
        device,
        temp_device,
        training_method=training_method,
    )
    if model_loader is None or model_setup is None:
        raise RuntimeError("could not create OneTrainer Z-Image loader/setup")

    train_config = TrainConfig.default_values().from_dict(
        {
            "__version": 10,
            "model_type": "Z_IMAGE",
            "training_method": "LORA" if args.lora else "FINE_TUNE",
            "base_model_name": str(args.base_model),
            "train_dtype": "BFLOAT_16",
            "fallback_train_dtype": "BFLOAT_16",
            "weight_dtype": "BFLOAT_16",
            "output_dtype": "BFLOAT_16",
            "gradient_checkpointing": "OFF",
            "layer_offload_fraction": 0.0,
            "latent_caching": True,
            "text_encoder": {"train": False, "weight_dtype": "BFLOAT_16"},
            "transformer": {"train": False, "weight_dtype": "BFLOAT_16"},
            "vae": {"weight_dtype": "FLOAT_32"},
        }
    )
    train_config.quantization = QuantizationConfig.default_values()

    model_names = ModelNames(base_model=str(args.base_model), lora=str(args.lora or ""))
    model = model_loader.load(
        model_type=ModelType.Z_IMAGE,
        model_names=model_names,
        weight_dtypes=_build_weight_dtypes(args.dtype),
        quantization=train_config.quantization,
    )
    model.train_config = train_config
    model.train_dtype = DataType.BFLOAT_16
    model.text_encoder_train_dtype = DataType.BFLOAT_16
    if args.lora:
        model_setup.setup_model(model, train_config)
    else:
        model_setup.setup_optimizations(model, train_config)
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)
    model.eval()
    return model


def _run_timed_onetrainer_sample(
    model: Any,
    *,
    prompt: str,
    negative: str,
    height: int,
    width: int,
    seed: int,
    steps: int,
    guidance: float,
    output_png: Path,
    device: Any,
    temp_device: Any,
) -> dict[str, Any]:
    import copy
    import inspect
    import torch
    from modules.util.torch_util import torch_gc

    with torch.no_grad(), model.autocast_context:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        noise_scheduler = copy.deepcopy(model.noise_scheduler)
        image_processor = model.create_pipeline().image_processor
        transformer = model.transformer
        vae = model.vae
        vae_scale_factor = 8
        num_latent_channels = transformer.in_channels
        batch_size = 2 if guidance > 1.0 else 1

        with _timed_stage(torch) as text_timer:
            model.text_encoder_to(device)
            prompt_embedding = model.encode_text(
                text=[prompt, negative] if guidance > 1.0 else prompt,
                batch_size=batch_size,
                train_device=device,
            )
            model.text_encoder_to(temp_device)
            torch_gc()

        latent_image = torch.randn(
            size=(1, num_latent_channels, height // vae_scale_factor, width // vae_scale_factor),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        noise_scheduler.set_timesteps(steps, device=device)
        timesteps = noise_scheduler.timesteps
        extra_step_kwargs = {}
        if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
            extra_step_kwargs["generator"] = generator

        model.transformer_to(device)
        with _timed_stage(torch) as denoise_timer:
            for i, timestep in enumerate(timesteps):
                latent_model_input = latent_image.unsqueeze(2).to(dtype=model.train_dtype.torch_dtype())
                latent_model_input = torch.cat([latent_model_input] * batch_size)
                latent_model_input_list = list(latent_model_input.unbind(dim=0))
                timestep_model_input = timestep.unsqueeze(0)
                output_list = transformer(
                    latent_model_input_list,
                    (1000 - timestep_model_input) / 1000,
                    prompt_embedding,
                    return_dict=True,
                ).sample

                noise_pred = -torch.stack(output_list, dim=0).squeeze(dim=2)
                if guidance > 1.0:
                    noise_pred_positive, noise_pred_negative = noise_pred.chunk(2)
                    noise_pred = noise_pred_negative + guidance * (
                        noise_pred_positive - noise_pred_negative
                    )

                latent_image = noise_scheduler.step(
                    noise_pred,
                    timestep,
                    latent_image,
                    return_dict=False,
                    **extra_step_kwargs,
                )[0]
                print(f"[ot-zimage-sample] denoise step={i + 1}/{len(timesteps)}")
        model.transformer_to(temp_device)
        torch_gc()

        with _timed_stage(torch) as vae_timer:
            model.vae_to(device)
            latents = model.unscale_latents(latent_image)
            image = vae.decode(latents, return_dict=False)[0]
            image = image_processor.postprocess(image, output_type="pil")
            output_png.parent.mkdir(parents=True, exist_ok=True)
            image[0].save(output_png)
        model.vae_to(temp_device)
        torch_gc()

    denoise_seconds = denoise_timer["seconds"]
    return {
        "prompt": prompt,
        "seed": seed,
        "resolution": {"width": width, "height": height},
        "steps": steps,
        "guidance": guidance,
        "dtype": "bf16",
        "text_encode_seconds": text_timer["seconds"],
        "denoise_seconds": denoise_seconds,
        "denoise_seconds_per_step": denoise_seconds / float(max(steps, 1)),
        "vae_decode_seconds": vae_timer["seconds"],
        "artifact_paths": [str(output_png)],
    }


def _write_speed_json(
    path: Path,
    *,
    onetrainer: dict[str, Any],
    peak_vram_mib: int | None,
    baseline_vram_mib: int | None,
    elapsed_seconds: float,
    command: list[str],
) -> None:
    data = {
        "schema": "serenity.zimage.onetrainer_sampler_speed.v1",
        "accepted_speed_parity": False,
        "speed_parity_claim": "not claimed",
        "onetrainer": dict(onetrainer),
        "measurement": {
            "source": "OneTrainer ZImageSampler logic via scripts/run_zimage_onetrainer_sample_speed.py",
            "elapsed_seconds": elapsed_seconds,
            "gpu_memory_baseline_mib": baseline_vram_mib,
            "peak_vram_mib": peak_vram_mib,
            "command": command,
        },
    }
    data["onetrainer"]["peak_vram_mib"] = peak_vram_mib if peak_vram_mib is not None else 0
    data["onetrainer"]["artifact_paths"] = list(onetrainer.get("artifact_paths", [])) + [str(path)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ot-zimage-sample] wrote speed JSON: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Timed OneTrainer Z-Image sampler evidence helper.")
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora", type=Path, default=None)
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--sample-id", default=DEFAULT_PROMPT_ID)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative", default=None)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temp-device", default="cpu")
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--speed-json", type=Path, default=DEFAULT_SPEED_JSON)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps < 0 or args.width < 0 or args.height < 0:
        raise ValueError("steps/width/height must be non-negative")

    sample = _load_sample_prompt(args.sample_file, args.sample_id)
    prompt = args.prompt or sample["prompt"]
    negative = sample["negative"] if args.negative is None else args.negative
    width = args.width or sample["width"]
    height = args.height or sample["height"]
    steps = args.steps or sample["steps"]
    guidance = args.guidance or sample["guidance"]
    seed = sample["seed"] if args.seed < 0 else args.seed
    if not prompt:
        raise ValueError("prompt must be non-empty")
    if steps <= 0 or width <= 0 or height <= 0:
        raise ValueError("steps/width/height must be positive")
    if not math.isfinite(guidance) or guidance <= 0.0:
        raise ValueError("guidance must be positive finite")

    _install_onetrainer_imports()
    import torch

    device = torch.device(args.device)
    temp_device = torch.device(args.temp_device)
    poller = VramPoller(args.poll_seconds)
    command = [sys.executable, *sys.argv]
    start = time.perf_counter()
    poller.start()
    try:
        model = _load_model(args)
        onetrainer = _run_timed_onetrainer_sample(
            model,
            prompt=prompt,
            negative=negative,
            height=height,
            width=width,
            seed=seed,
            steps=steps,
            guidance=guidance,
            output_png=args.output_png,
            device=device,
            temp_device=temp_device,
        )
    finally:
        poller.stop()
    elapsed = time.perf_counter() - start
    _write_speed_json(
        args.speed_json,
        onetrainer=onetrainer,
        peak_vram_mib=poller.peak_mib,
        baseline_vram_mib=poller.baseline_mib,
        elapsed_seconds=elapsed,
        command=command,
    )
    print(
        "[ot-zimage-sample] PASS "
        f"steps={steps} denoise_s_per_step={onetrainer['denoise_seconds_per_step']:.6f} "
        f"vae_s={onetrainer['vae_decode_seconds']:.6f} peak_vram_mib={poller.peak_mib}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
