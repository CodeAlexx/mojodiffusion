#!/usr/bin/env python3
"""
Dump bounded Serenity Ernie train-step reference data for Mojo parity gates.

This intentionally reuses the proven Qwen dump harness for the generic
Serenity train loop, optimizer, adapter dump, and safetensors/json writing.
Only the model validation, dry-run evidence, and Ernie-specific trace hooks are
overridden here.

Run with the Serenity venv, for example:
  /home/alex/Serenity/venv/bin/python scripts/ernie_dump_train_ref.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import qwen_dump_train_ref as base


DEFAULT_SERENITY = Path("/home/alex/Serenity")
DEFAULT_CONFIG = DEFAULT_SERENITY / "configs" / "ernie_eri2_100step_baseline.json"
DEFAULT_OUT_DIR = Path("/home/alex/serenity-trainer/parity")
DEFAULT_PREFIX = "ernie_train_ref"
PRODUCER = "scripts/ernie_dump_train_ref.py"
BLOCKER_SCHEMA_VERSION = 1

REFERENCE_SOURCES = [
    "/home/alex/Serenity/modules/model/ErnieModel.py",
    "/home/alex/Serenity/modules/dataLoader/ErnieBaseDataLoader.py",
    "/home/alex/Serenity/modules/modelSetup/BaseErnieSetup.py",
    "/home/alex/Serenity/modules/modelSetup/ErnieLoRASetup.py",
    "/home/alex/Serenity/modules/modelSetup/ErnieFineTuneSetup.py",
    "/home/alex/Serenity/modules/modelSampler/ErnieSampler.py",
    "/home/alex/Serenity/modules/modelLoader/ErnieModelLoader.py",
    "/home/alex/Serenity/modules/modelSaver/ErnieLoRAModelSaver.py",
    "/home/alex/Serenity/modules/modelSaver/ErnieFineTuneModelSaver.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump one bounded Serenity Ernie train-step parity ref."
    )
    parser.add_argument("--serenity", type=Path, default=DEFAULT_SERENITY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--secrets", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--train-device", default=None)
    parser.add_argument("--temp-device", default=None)
    parser.add_argument(
        "--adapter-dump",
        choices=("none", "initial", "step", "step-with-grads"),
        default="step",
    )
    parser.add_argument("--allow-cache-build", action="store_true")
    parser.add_argument("--torch-seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    return parser.parse_args()


def validate_config(config: Any, args: argparse.Namespace, *, check_cache: bool = True) -> None:
    from modules.util.enum.ModelType import ModelType
    from modules.util.enum.TrainingMethod import TrainingMethod

    if config.model_type != ModelType.ERNIE:
        raise ValueError(f"Expected ERNIE config, got {config.model_type}")
    if config.training_method != TrainingMethod.LORA:
        raise ValueError(f"Expected LORA training method, got {config.training_method}")
    if args.max_steps != 1:
        raise ValueError("--max-steps must be exactly 1 for this bounded Ernie dump")
    if config.optimizer.fused_back_pass:
        raise NotImplementedError("fused_back_pass is not supported by this bounded dump loop")
    if config.optimizer.optimizer.is_schedule_free:
        raise NotImplementedError("schedule-free optimizer mode is not supported by this dump loop")
    if not args.dry_run:
        if str(config.train_device).split(":", 1)[0] != "cuda":
            raise RuntimeError(
                "CPU PyTorch is not numeric parity evidence. Non-dry-run "
                "Serenity reference dumps must use --train-device cuda."
            )
        if not base.torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Run --dry-run for structural checks only; "
                "do not produce CPU numeric reference dumps."
            )
    if check_cache and config.latent_caching and not args.allow_cache_build and not base.cache_present(config):
        raise FileNotFoundError(
            "Required cache dirs are missing or empty. Reuse an existing cache or pass "
            "--allow-cache-build to let Serenity build it."
        )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.json_safe(data), indent=2) + "\n", encoding="utf-8")


def run_dry_run(args: argparse.Namespace, config: Any) -> None:
    info = {
        "producer": PRODUCER,
        "serenity": str(args.serenity.resolve()),
        "config": str(args.config.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "prefix": args.prefix,
        "max_steps": args.max_steps,
        "torch": base.torch.__version__,
        "cuda_available": base.torch.cuda.is_available(),
        "cuda_device_count": base.torch.cuda.device_count(),
        "train_device": config.train_device,
        "temp_device": config.temp_device,
        "model_type": base.enum_value(config.model_type),
        "training_method": base.enum_value(config.training_method),
        "cache_dir": config.cache_dir,
        "cache_present": base.cache_present(config),
        "allow_cache_build": args.allow_cache_build,
        "base_model_name": config.base_model_name,
        "train_dtype": base.enum_value(config.train_dtype),
        "lora_weight_dtype": base.enum_value(config.lora_weight_dtype),
        "layer_filter": config.layer_filter,
        "layer_filter_preset": config.layer_filter_preset,
    }
    blockers = []
    if not args.config.is_file():
        blockers.append({"kind": "missing_config", "path": str(args.config)})
    if config.latent_caching and not base.cache_present(config) and not args.allow_cache_build:
        blockers.append({"kind": "missing_cache", "path": config.cache_dir})
    if not Path(str(config.base_model_name)).exists():
        blockers.append({"kind": "missing_base_model", "path": str(config.base_model_name)})

    contract = {
        "producer": PRODUCER,
        "schema_version": BLOCKER_SCHEMA_VERSION,
        "reference_sources": REFERENCE_SOURCES,
        "required_tensors": [
            "output.predicted",
            "output.target",
            "output.loss_pre_scale",
            "output.loss_for_backward",
            "trace.scaled_latent_image",
            "trace.latent_noise",
            "trace.scaled_noisy_latent_image",
            "trace.sigma",
            "trace.transformer_hidden_states",
            "trace.transformer_timestep",
            "trace.encoder_hidden_states",
            "trace.packed_predicted_flow",
            "trace.flow",
        ],
    }
    blocker_report = {
        "producer": PRODUCER,
        "schema_version": BLOCKER_SCHEMA_VERSION,
        "blocked": bool(blockers),
        "structured_blockers": blockers,
        "dry_run": info,
    }

    _write_json(args.out_dir / f"{args.prefix}_contract.json", contract)
    _write_json(args.out_dir / f"{args.prefix}_blockers.json", blocker_report)
    print(json.dumps(base.json_safe(blocker_report), indent=2))


class ErnieStepTraceHooks:
    def __init__(self, setup: Any, model: Any, trace: dict[str, Any]):
        self.setup = setup
        self.model = model
        self.trace = trace
        self.orig_create_noise = None
        self.orig_get_timestep = None
        self.orig_add_noise = None
        self.orig_encode_text = None
        self.orig_scale_latents = None
        self.orig_patchify_latents = None
        self.orig_transformer_forward = None

    def __enter__(self) -> "ErnieStepTraceHooks":
        self.orig_create_noise = self.setup._create_noise
        self.orig_get_timestep = self.setup._get_timestep_discrete
        self.orig_add_noise = self.setup._add_noise_discrete
        self.orig_encode_text = self.model.encode_text
        self.orig_scale_latents = self.model.scale_latents
        self.orig_patchify_latents = self.model.patchify_latents
        self.orig_transformer_forward = self.model.transformer.forward

        def encode_text(*args, **kwargs):
            self.trace["encode_text.tokens"] = kwargs.get("tokens")
            self.trace["encode_text.tokens_mask"] = kwargs.get("tokens_mask")
            self.trace["encode_text.cached_hidden_state"] = kwargs.get("text_encoder_output")
            out, text_lens = self.orig_encode_text(*args, **kwargs)
            self.trace["text_encoder_output"] = out
            self.trace["text_lens"] = text_lens
            return out, text_lens

        def patchify_latents(latent_image):
            self.trace["latent_image_unpatchified"] = latent_image
            out = self.orig_patchify_latents(latent_image)
            self.trace["latent_image_before_scale"] = out
            return out

        def scale_latents(latent_image):
            out = self.orig_scale_latents(latent_image)
            self.trace["scaled_latent_image"] = out
            return out

        def create_noise(source_tensor, config, generator, timestep=None, betas=None):
            self.trace["noise_source_tensor"] = source_tensor
            out = self.orig_create_noise(source_tensor, config, generator, timestep, betas)
            self.trace["latent_noise"] = out
            return out

        def get_timestep(num_train_timesteps, deterministic, generator, batch_size, config, shift=None):
            out = self.orig_get_timestep(
                num_train_timesteps, deterministic, generator, batch_size, config, shift
            )
            self.trace["timestep_shift"] = shift if shift is not None else config.timestep_shift
            self.trace["num_train_timesteps"] = int(num_train_timesteps)
            self.trace["batch_seed"] = int(generator.initial_seed())
            return out

        def add_noise(scaled_latent_image, latent_noise, timestep, timesteps):
            scaled_noisy, sigma = self.orig_add_noise(
                scaled_latent_image, latent_noise, timestep, timesteps
            )
            self.trace["scaled_noisy_latent_image"] = scaled_noisy
            self.trace["sigma"] = sigma
            return scaled_noisy, sigma

        def transformer_forward(*args, **kwargs):
            self.trace["transformer_hidden_states"] = kwargs.get("hidden_states")
            self.trace["transformer_timestep"] = kwargs.get("timestep")
            self.trace["encoder_hidden_states"] = kwargs.get("text_bth")
            self.trace["text_lens"] = kwargs.get("text_lens")
            out = self.orig_transformer_forward(*args, **kwargs)
            if isinstance(out, (tuple, list)) and out:
                self.trace["predicted_flow"] = out[0]
            else:
                self.trace["predicted_flow"] = getattr(out, "sample", None)
            return out

        self.model.encode_text = encode_text
        self.model.patchify_latents = patchify_latents
        self.model.scale_latents = scale_latents
        self.setup._create_noise = create_noise
        self.setup._get_timestep_discrete = get_timestep
        self.setup._add_noise_discrete = add_noise
        self.model.transformer.forward = transformer_forward
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.setup._create_noise = self.orig_create_noise
        self.setup._get_timestep_discrete = self.orig_get_timestep
        self.setup._add_noise_discrete = self.orig_add_noise
        self.model.encode_text = self.orig_encode_text
        self.model.patchify_latents = self.orig_patchify_latents
        self.model.scale_latents = self.orig_scale_latents
        self.model.transformer.forward = self.orig_transformer_forward


def enrich_trace(model: Any, trace: dict[str, Any]) -> None:
    latent_noise = trace.get("latent_noise")
    scaled_latent = trace.get("scaled_latent_image")
    predicted_flow = trace.get("predicted_flow")
    if latent_noise is not None and scaled_latent is not None:
        trace["flow"] = latent_noise - scaled_latent
    if predicted_flow is not None:
        trace["packed_predicted_flow"] = predicted_flow


def main() -> None:
    base.DEFAULT_SERENITY = DEFAULT_SERENITY
    base.DEFAULT_CONFIG = DEFAULT_CONFIG
    base.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    base.DEFAULT_PREFIX = DEFAULT_PREFIX
    base.PRODUCER = PRODUCER
    base.parse_args = parse_args
    base.validate_config = validate_config
    base.run_dry_run = run_dry_run
    base.QwenStepTraceHooks = ErnieStepTraceHooks
    base.enrich_trace = enrich_trace
    base.main()


if __name__ == "__main__":
    main()
