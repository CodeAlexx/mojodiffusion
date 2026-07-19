#!/usr/bin/env python3
"""
Dump real Serenity Klein/Flux2 train-step reference data for Mojo parity gates.

This script imports Serenity by path, uses its real model loader, setup,
cached-data loader, predict, loss, backward, clipping, optimizer, and LR scheduler
paths, and writes bounded one/few-step references to safetensors/json.

Run with the Serenity venv, for example:
  /home/alex/Serenity/venv/bin/python scripts/klein_dump_train_ref.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn


DEFAULT_SERENITY = Path("/home/alex/Serenity")
DEFAULT_CONFIG = DEFAULT_SERENITY / "configs" / "AB50_klein9b_serenity.json"
DEFAULT_OUT_DIR = Path("/home/alex/serenity-trainer/parity")
DEFAULT_PREFIX = "klein_train_ref"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump bounded Serenity Klein/Flux2 train-step parity refs."
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
        help=(
            "Adapter tensor dump mode. 'step' writes trainable LoRA params before "
            "and after each optimizer update; 'step-with-grads' also writes grads."
        ),
    )
    parser.add_argument(
        "--allow-cache-build",
        action="store_true",
        help="Allow Serenity to build missing latent/text caches during start_next_epoch().",
    )
    parser.add_argument(
        "--torch-seed",
        type=int,
        default=None,
        help="Optional process RNG seed before Serenity setup. Default preserves Serenity's process RNG behavior.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check imports, config, output path, CUDA, and cache presence without loading the model.",
    )
    return parser.parse_args()


def add_serenity_to_path(root: Path) -> None:
    root = root.resolve()
    if not (root / "modules").is_dir():
        raise FileNotFoundError(f"Serenity root does not contain modules/: {root}")
    sys.path.insert(0, str(root))


def enter_serenity_root(args: argparse.Namespace) -> None:
    """Serenity factory.import_dir uses cwd-relative module paths."""
    args.serenity = args.serenity.resolve()
    args.config = args.config.resolve()
    if args.secrets is not None:
        args.secrets = args.secrets.resolve()
    args.out_dir = args.out_dir.resolve()
    os.chdir(args.serenity)


def enum_value(value: Any) -> Any:
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    return str(value)


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    stats: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).replace("torch.", ""),
        "device": str(detached.device),
        "numel": int(detached.numel()),
    }
    if detached.numel() == 0:
        return stats

    if detached.dtype == torch.bool:
        stats["true_count"] = int(detached.to(torch.int64).sum().item())
        return stats

    if detached.is_floating_point() or detached.is_complex():
        values = detached.detach().to(dtype=torch.float32, device="cpu").flatten()
        finite = values[torch.isfinite(values)]
        stats["nonfinite"] = int(values.numel() - finite.numel())
        if finite.numel() > 0:
            stats.update({
                "mean": float(finite.mean().item()),
                "std": float(finite.std(unbiased=False).item()),
                "min": float(finite.min().item()),
                "max": float(finite.max().item()),
                "max_abs": float(finite.abs().max().item()),
            })
        return stats

    values = detached.detach().to(device="cpu").flatten()
    stats.update({
        "min": int(values.min().item()),
        "max": int(values.max().item()),
    })
    return stats


def cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous()


def add_tensor(
    tensor_map: dict[str, torch.Tensor],
    meta: dict[str, Any],
    name: str,
    tensor: torch.Tensor | None,
    *,
    stats: bool = True,
) -> None:
    if tensor is None:
        return
    tensor_map[name] = cpu_tensor(tensor)
    if stats:
        meta[name] = tensor_stats(tensor)


def add_batch(
    tensor_map: dict[str, torch.Tensor],
    meta: dict[str, Any],
    batch: dict[str, Any],
) -> dict[str, Any]:
    non_tensor: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            add_tensor(tensor_map, meta, f"batch.{key}", value)
        else:
            non_tensor[key] = json_safe(value)
    return non_tensor


def optimizer_meta(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        group_meta = {
            "index": index,
            "name": group.get("name"),
            "lr": float(group.get("lr", 0.0)),
            "initial_lr": float(group.get("initial_lr", group.get("lr", 0.0))),
            "weight_decay": group.get("weight_decay"),
            "betas": list(group["betas"]) if "betas" in group else None,
            "eps": group.get("eps"),
            "foreach": group.get("foreach"),
            "fused": group.get("fused"),
            "param_count": len(group.get("params", [])),
            "param_numel": int(sum(p.numel() for p in group.get("params", []))),
        }
        groups.append(json_safe(group_meta))

    state_tensor_count = 0
    state_numel = 0
    state_names: set[str] = set()
    scalar_steps = []
    for state in optimizer.state.values():
        for key, value in state.items():
            state_names.add(str(key))
            if torch.is_tensor(value):
                state_tensor_count += 1
                state_numel += int(value.numel())
                if key == "step" and value.numel() == 1 and len(scalar_steps) < 16:
                    scalar_steps.append(float(value.detach().cpu().item()))

    return {
        "class": optimizer.__class__.__name__,
        "param_groups": groups,
        "state": {
            "parameter_entries": len(optimizer.state),
            "tensor_count": state_tensor_count,
            "tensor_numel": state_numel,
            "keys": sorted(state_names),
            "sample_steps": scalar_steps,
        },
    }


def named_trainable_parameters(model: Any, parameters: list[torch.nn.Parameter]) -> list[tuple[str, torch.nn.Parameter]]:
    id_to_name: dict[int, str] = {}
    lora_wrapper = getattr(model, "transformer_lora", None)
    lora_modules = getattr(lora_wrapper, "lora_modules", None)
    if isinstance(lora_modules, dict):
        for prefix, module in lora_modules.items():
            if not hasattr(module, "named_parameters"):
                continue
            for name, param in module.named_parameters():
                id_to_name[id(param)] = f"{prefix}.{name}"

    named = []
    fallback_index = 0
    for param in parameters:
        if not param.requires_grad:
            continue
        name = id_to_name.get(id(param))
        if name is None:
            name = f"unknown_{fallback_index:04d}_{id(param):x}"
            fallback_index += 1
        named.append((name, param))
    return named


def add_adapter_tensors(
    tensor_map: dict[str, torch.Tensor],
    named_params: list[tuple[str, torch.nn.Parameter]],
    prefix: str,
    *,
    include_grad: bool = False,
) -> None:
    for name, param in named_params:
        add_tensor(tensor_map, {}, f"{prefix}.{name}", param, stats=False)
        if include_grad and param.grad is not None:
            add_tensor(tensor_map, {}, f"{prefix}_grad.{name}", param.grad, stats=False)


def global_grad_norm(parameters: list[torch.nn.Parameter]) -> torch.Tensor | None:
    norms = [
        p.grad.detach().to(torch.float32).norm(2)
        for p in parameters
        if p.grad is not None
    ]
    if not norms:
        return None
    return torch.stack(norms).norm(2)


class Flux2StepTraceHooks:
    def __init__(self, setup: Any, model: Any, trace: dict[str, Any]):
        self.setup = setup
        self.model = model
        self.trace = trace
        self.orig_create_noise = None
        self.orig_get_timestep = None
        self.orig_add_noise = None
        self.orig_transformer_forward = None

    def __enter__(self) -> "Flux2StepTraceHooks":
        self.orig_create_noise = self.setup._create_noise
        self.orig_get_timestep = self.setup._get_timestep_discrete
        self.orig_add_noise = self.setup._add_noise_discrete
        self.orig_transformer_forward = self.model.transformer.forward

        def create_noise(source_tensor, config, generator, timestep=None, betas=None):
            self.trace["scaled_latent_image"] = source_tensor
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
            self.trace["packed_latent_input"] = kwargs.get("hidden_states")
            self.trace["transformer_timestep"] = kwargs.get("timestep")
            self.trace["guidance"] = kwargs.get("guidance")
            self.trace["encoder_hidden_states"] = kwargs.get("encoder_hidden_states")
            self.trace["text_ids"] = kwargs.get("txt_ids")
            self.trace["image_ids"] = kwargs.get("img_ids")
            out = self.orig_transformer_forward(*args, **kwargs)
            self.trace["packed_predicted_flow"] = getattr(out, "sample", None)
            return out

        self.setup._create_noise = create_noise
        self.setup._get_timestep_discrete = get_timestep
        self.setup._add_noise_discrete = add_noise
        self.model.transformer.forward = transformer_forward
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.setup._create_noise = self.orig_create_noise
        self.setup._get_timestep_discrete = self.orig_get_timestep
        self.setup._add_noise_discrete = self.orig_add_noise
        self.model.transformer.forward = self.orig_transformer_forward


def cache_present(config: Any) -> bool:
    if not config.latent_caching:
        return True
    cache_dir = Path(config.cache_dir)
    required = [cache_dir / "image", cache_dir / "text"]
    return all(path.is_dir() and any(path.rglob("*")) for path in required)


def load_train_config(args: argparse.Namespace) -> Any:
    from modules.util.config.SecretsConfig import SecretsConfig
    from modules.util.config.TrainConfig import TrainConfig

    config = TrainConfig.default_values()
    with args.config.open("r", encoding="utf-8") as handle:
        config.from_dict(json.load(handle))

    if args.secrets is not None:
        with args.secrets.open("r", encoding="utf-8") as handle:
            config.secrets = SecretsConfig.default_values().from_dict(json.load(handle))

    if args.train_device is not None:
        config.train_device = args.train_device
    if args.temp_device is not None:
        config.temp_device = args.temp_device

    # Runtime bounds for this reference tool. These avoid samples/backups/saves
    # and preserve existing caches unless the caller explicitly allows a rebuild.
    config.validation = False
    config.samples = []
    config.tensorboard = False
    config.tensorboard_always_on = False
    config.samples_to_tensorboard = False
    config.backup_before_save = False
    config.save_every = 0
    config.only_cache = False
    config.clear_cache_before_training = False
    config.compile = False
    config.multi_gpu = False
    return config


def validate_config(config: Any, args: argparse.Namespace, *, check_cache: bool = True) -> None:
    from modules.util.enum.ModelType import ModelType
    from modules.util.enum.TrainingMethod import TrainingMethod

    if config.model_type != ModelType.FLUX_2:
        raise ValueError(f"Expected FLUX_2 config, got {config.model_type}")
    if config.training_method != TrainingMethod.LORA:
        raise ValueError(f"Expected LORA training method, got {config.training_method}")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
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
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Run --dry-run for structural checks only; "
                "do not produce CPU numeric reference dumps."
            )
    if check_cache and config.latent_caching and not args.allow_cache_build and not cache_present(config):
        raise FileNotFoundError(
            "Required cache dirs are missing or empty. Reuse an existing cache or pass "
            "--allow-cache-build to let Serenity build it."
        )


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)


def run_dry_run(args: argparse.Namespace, config: Any) -> None:
    info = {
        "serenity": str(args.serenity.resolve()),
        "config": str(args.config.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "prefix": args.prefix,
        "max_steps": args.max_steps,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "train_device": config.train_device,
        "temp_device": config.temp_device,
        "model_type": enum_value(config.model_type),
        "training_method": enum_value(config.training_method),
        "cache_dir": config.cache_dir,
        "cache_present": cache_present(config),
        "allow_cache_build": args.allow_cache_build,
    }
    print(json.dumps(json_safe(info), indent=2))


def build_model_and_loader(config: Any) -> tuple[Any, Any, Any]:
    from modules.util import create
    from modules.util.torch_util import torch_gc

    if config.quantization.cache_dir is None:
        config.quantization.cache_dir = config.cache_dir + "/quantization"
    os.makedirs(config.quantization.cache_dir, exist_ok=True)

    model_loader = create.create_model_loader(config.model_type, config.training_method)
    model_setup = create.create_model_setup(
        config.model_type,
        torch.device(config.train_device),
        torch.device(config.temp_device),
        config.training_method,
        config.debug_mode,
    )

    model = model_loader.load(
        model_type=config.model_type,
        model_names=config.model_names(),
        weight_dtypes=config.weight_dtypes(),
        quantization=config.quantization,
    )
    model.train_config = config

    model_setup.setup_optimizations(model, config)
    model_setup.setup_train_device(model, config)
    model_setup.setup_model(model, config)
    model.to(torch.device(config.temp_device))
    model.eval()
    torch_gc()

    data_loader = create.create_data_loader(
        torch.device(config.train_device),
        torch.device(config.temp_device),
        model,
        config.model_type,
        model_setup,
        config.training_method,
        config,
        model.train_progress,
        False,
    )

    return model, model_setup, data_loader


def create_scheduler(config: Any, model: Any, data_loader: Any) -> Any:
    from modules.util import create

    return create.create_lr_scheduler(
        config=config,
        optimizer=model.optimizer,
        learning_rate_scheduler=config.learning_rate_scheduler,
        warmup_steps=config.learning_rate_warmup_steps,
        num_cycles=config.learning_rate_cycles,
        min_factor=config.learning_rate_min_factor,
        num_epochs=config.epochs,
        approximate_epoch_length=data_loader.get_data_set().approximate_length(),
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        global_step=model.train_progress.global_step,
    )


def enrich_trace(model: Any, trace: dict[str, Any]) -> None:
    scaled_noisy = trace.get("scaled_noisy_latent_image")
    packed_pred = trace.get("packed_predicted_flow")
    if scaled_noisy is not None and packed_pred is not None:
        trace["predicted_flow"] = model.unpack_latents(
            packed_pred, scaled_noisy.shape[2], scaled_noisy.shape[3]
        )

    latent_noise = trace.get("latent_noise")
    scaled_latent = trace.get("scaled_latent_image")
    if latent_noise is not None and scaled_latent is not None:
        trace["flow"] = latent_noise - scaled_latent


def save_step_dump(
    args: argparse.Namespace,
    step_index: int,
    train_progress: Any,
    batch: dict[str, Any],
    trace: dict[str, Any],
    model_output_data: dict[str, Any],
    loss: torch.Tensor,
    loss_for_backward: torch.Tensor,
    lr_before: list[float],
    lr_after: list[float],
    grad_norm_pre_clip: float | None,
    grad_norm_no_clip: float | None,
    optimizer_before: dict[str, Any],
    optimizer_after: dict[str, Any],
    adapter_tensors: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    step_tag = f"step{step_index:03d}"
    tensor_path = args.out_dir / f"{args.prefix}_{step_tag}.safetensors"
    adapter_path = args.out_dir / f"{args.prefix}_{step_tag}_adapters.safetensors"

    tensors: dict[str, torch.Tensor] = {}
    tensor_meta: dict[str, Any] = {}
    non_tensor_batch = add_batch(tensors, tensor_meta, batch)

    for key in [
        "scaled_latent_image",
        "latent_noise",
        "scaled_noisy_latent_image",
        "sigma",
        "packed_latent_input",
        "transformer_timestep",
        "guidance",
        "encoder_hidden_states",
        "text_ids",
        "image_ids",
        "packed_predicted_flow",
        "predicted_flow",
        "flow",
    ]:
        add_tensor(tensors, tensor_meta, f"trace.{key}", trace.get(key))

    for key in ["timestep", "predicted", "target", "prior_target"]:
        value = model_output_data.get(key)
        if torch.is_tensor(value):
            add_tensor(tensors, tensor_meta, f"output.{key}", value)

    add_tensor(tensors, tensor_meta, "output.loss_pre_scale", loss.reshape(()))
    add_tensor(tensors, tensor_meta, "output.loss_for_backward", loss_for_backward.reshape(()))

    write_safetensors(
        tensor_path,
        tensors,
        metadata={
            "producer": "scripts/klein_dump_train_ref.py",
            "step_index": str(step_index),
            "global_step": str(train_progress.global_step),
        },
    )

    adapter_file = None
    if adapter_tensors:
        write_safetensors(
            adapter_path,
            adapter_tensors,
            metadata={
                "producer": "scripts/klein_dump_train_ref.py",
                "step_index": str(step_index),
                "global_step": str(train_progress.global_step),
                "adapter_dump": args.adapter_dump,
            },
        )
        adapter_file = str(adapter_path)

    return {
        "step_index": step_index,
        "global_step": train_progress.global_step,
        "epoch": train_progress.epoch,
        "epoch_step": train_progress.epoch_step,
        "batch_seed": trace.get("batch_seed"),
        "timestep_shift": trace.get("timestep_shift"),
        "num_train_timesteps": trace.get("num_train_timesteps"),
        "loss_pre_scale": float(loss.detach().cpu().item()),
        "loss_for_backward": float(loss_for_backward.detach().cpu().item()),
        "grad_norm_pre_clip": grad_norm_pre_clip,
        "grad_norm_no_clip": grad_norm_no_clip,
        "lr_before": [float(x) for x in lr_before],
        "lr_after": [float(x) for x in lr_after],
        "optimizer_before": optimizer_before,
        "optimizer_after": optimizer_after,
        "safetensors": str(tensor_path),
        "adapter_safetensors": adapter_file,
        "batch_non_tensor": non_tensor_batch,
        "tensor_meta": tensor_meta,
    }


def train_and_dump(args: argparse.Namespace, config: Any) -> dict[str, Any]:
    from modules.modelSetup.BaseModelSetup import BaseModelSetup
    from modules.util import multi_gpu_util as multi
    from modules.util import torch_util
    from modules.util.bf16_stochastic_rounding import set_seed as bf16_stochastic_rounding_set_seed
    from modules.util.dtype_util import create_grad_scaler, enable_grad_scaling
    from modules.util.enum.ConceptType import ConceptType
    from modules.util.enum.TrainingMethod import TrainingMethod

    if multi.is_enabled():
        raise NotImplementedError("This dump script is single-process only")

    start_time = time.monotonic()
    model, model_setup, data_loader = build_model_and_loader(config)
    parameters = model.parameters.parameters()
    named_params = named_trainable_parameters(model, parameters)

    initial_adapter_path = None
    if args.adapter_dump == "initial":
        initial_tensors: dict[str, torch.Tensor] = {}
        add_adapter_tensors(initial_tensors, named_params, "adapter_initial")
        initial_adapter_path = args.out_dir / f"{args.prefix}_initial_adapters.safetensors"
        write_safetensors(
            initial_adapter_path,
            initial_tensors,
            metadata={"producer": "scripts/klein_dump_train_ref.py", "adapter_dump": "initial"},
        )

    if config.latent_caching:
        data_loader.get_data_set().start_next_epoch()
        model_setup.setup_train_device(model, config)
    else:
        model_setup.setup_train_device(model, config)
        data_loader.get_data_set().start_next_epoch()

    lr_scheduler = create_scheduler(config, model, data_loader)
    scaler = create_grad_scaler() if enable_grad_scaling(config.train_dtype, parameters) else None
    update_every = int(config.gradient_accumulation_steps)
    accumulated_steps = 0
    step_results = []

    for batch in data_loader.get_data_loader():
        if accumulated_steps >= args.max_steps:
            break

        step_index = accumulated_steps
        train_progress = model.train_progress
        bf16_stochastic_rounding_set_seed(train_progress.global_step, torch.device(config.train_device))

        adapter_tensors: dict[str, torch.Tensor] | None = None
        if args.adapter_dump in ("step", "step-with-grads"):
            adapter_tensors = {}
            add_adapter_tensors(adapter_tensors, named_params, "adapter_before")

        trace: dict[str, Any] = {}
        prior_pred_indices = [
            i for i in range(config.batch_size)
            if ConceptType(batch["concept_type"][i]) == ConceptType.PRIOR_PREDICTION
        ]
        needs_prior = (
            len(prior_pred_indices) > 0
            or (
                config.masked_training
                and config.masked_prior_preservation_weight > 0
                and config.training_method == TrainingMethod.LORA
            )
        )

        if needs_prior:
            prior_trace: dict[str, Any] = {}
            with model_setup.prior_model(model, config), torch.no_grad():
                with Flux2StepTraceHooks(model_setup, model, prior_trace):
                    prior_model_output_data = model_setup.predict(model, batch, config, train_progress)
            with Flux2StepTraceHooks(model_setup, model, trace):
                model_output_data = model_setup.predict(model, batch, config, train_progress)
            prior_model_prediction = prior_model_output_data["predicted"].to(
                dtype=model_output_data["target"].dtype
            )
            model_output_data["target"][prior_pred_indices] = prior_model_prediction[prior_pred_indices]
            model_output_data["prior_target"] = prior_model_prediction
            trace["prior_trace"] = prior_trace
        else:
            with Flux2StepTraceHooks(model_setup, model, trace):
                model_output_data = model_setup.predict(model, batch, config, train_progress)

        enrich_trace(model, trace)

        loss = model_setup.calculate_loss(model, batch, model_output_data, config)
        loss_for_backward = loss / config.gradient_accumulation_steps

        if scaler:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        if args.adapter_dump == "step-with-grads" and adapter_tensors is not None:
            add_adapter_tensors(adapter_tensors, named_params, "adapter_pre_clip", include_grad=True)

        multi.reduce_grads_mean(parameters, config.gradient_reduce_precision)

        grad_norm_pre_clip = None
        grad_norm_no_clip_tensor = global_grad_norm(parameters)
        grad_norm_no_clip = (
            float(grad_norm_no_clip_tensor.detach().cpu().item())
            if grad_norm_no_clip_tensor is not None else None
        )

        lr_before = lr_scheduler.get_last_lr()
        optimizer_before = optimizer_meta(model.optimizer)

        should_update = (train_progress.global_step + 1) % update_every == 0
        if should_update:
            if scaler:
                scaler.unscale_(model.optimizer)
                if config.clip_grad_norm is not None:
                    grad_norm = nn.utils.clip_grad_norm_(parameters, config.clip_grad_norm)
                    grad_norm_pre_clip = float(grad_norm.detach().cpu().item())
                scaler.step(model.optimizer)
                scaler.update()
            else:
                if config.clip_grad_norm is not None:
                    grad_norm = nn.utils.clip_grad_norm_(parameters, config.clip_grad_norm)
                    grad_norm_pre_clip = float(grad_norm.detach().cpu().item())
                model.optimizer.step()

            if args.adapter_dump == "step-with-grads" and adapter_tensors is not None:
                add_adapter_tensors(adapter_tensors, named_params, "adapter_post_clip", include_grad=True)

            lr_scheduler.step()
            model.optimizer.zero_grad(set_to_none=True)
            model_setup.after_optimizer_step(model, config, train_progress)

        if args.adapter_dump in ("step", "step-with-grads") and adapter_tensors is not None:
            add_adapter_tensors(adapter_tensors, named_params, "adapter_after")

        lr_after = lr_scheduler.get_last_lr()
        optimizer_after = optimizer_meta(model.optimizer)

        step_results.append(save_step_dump(
            args=args,
            step_index=step_index,
            train_progress=train_progress,
            batch=batch,
            trace=trace,
            model_output_data=model_output_data,
            loss=loss,
            loss_for_backward=loss_for_backward,
            lr_before=lr_before,
            lr_after=lr_after,
            grad_norm_pre_clip=grad_norm_pre_clip,
            grad_norm_no_clip=grad_norm_no_clip,
            optimizer_before=optimizer_before,
            optimizer_after=optimizer_after,
            adapter_tensors=adapter_tensors,
        ))

        train_progress.next_step(config.batch_size)
        accumulated_steps += 1

    if accumulated_steps == 0:
        raise RuntimeError("No batches were produced by the Serenity data loader")

    torch_util.torch_gc()

    return {
        "producer": "scripts/klein_dump_train_ref.py",
        "created_unix": time.time(),
        "elapsed_seconds": time.monotonic() - start_time,
        "serenity": str(args.serenity.resolve()),
        "config_path": str(args.config.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "prefix": args.prefix,
        "max_steps": args.max_steps,
        "adapter_dump": args.adapter_dump,
        "initial_adapter_safetensors": str(initial_adapter_path) if initial_adapter_path else None,
        "torch": torch.__version__,
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "runtime_config": {
            "model_type": enum_value(config.model_type),
            "training_method": enum_value(config.training_method),
            "train_device": config.train_device,
            "temp_device": config.temp_device,
            "train_dtype": enum_value(config.train_dtype),
            "base_model_name": config.base_model_name,
            "cache_dir": config.cache_dir,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "learning_rate_scheduler": enum_value(config.learning_rate_scheduler),
            "optimizer": enum_value(config.optimizer.optimizer),
            "clip_grad_norm": config.clip_grad_norm,
            "timestep_distribution": enum_value(config.timestep_distribution),
            "dynamic_timestep_shifting": config.dynamic_timestep_shifting,
            "text_encoder_sequence_length": config.text_encoder_sequence_length,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_weight_dtype": enum_value(config.lora_weight_dtype),
            "layer_filter": config.layer_filter,
            "layer_filter_preset": config.layer_filter_preset,
        },
        "trainable_parameters": {
            "count": len(named_params),
            "numel": int(sum(param.numel() for _, param in named_params)),
            "names": [name for name, _ in named_params],
        },
        "steps": step_results,
    }


def main() -> None:
    args = parse_args()
    enter_serenity_root(args)
    add_serenity_to_path(args.serenity)
    config = load_train_config(args)
    validate_config(config, args, check_cache=not args.dry_run)

    if args.dry_run:
        run_dry_run(args, config)
        return

    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.torch_seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = train_and_dump(args, config)
    meta_path = args.out_dir / f"{args.prefix}_meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2)
    print(json.dumps({"meta": str(meta_path), "steps": len(summary["steps"])}, indent=2))


if __name__ == "__main__":
    main()
