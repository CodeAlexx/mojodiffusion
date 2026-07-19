#!/usr/bin/env python3
"""
Dump bounded Serenity Anima train-step reference data for parity gates.

The dry-run mode is intentionally filesystem/registration oriented: it checks
the local Serenity Anima config, 100-step baseline artifacts, cached image/text
records, local diffusers snapshot, and factory registrations without loading the
model or running training.

Run with the Serenity-anima-ref venv when executing the real one-step dump:
  /home/alex/Serenity-anima-ref/venv/bin/python scripts/anima_dump_train_ref.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEFAULT_SERENITY = Path("/home/alex/Serenity-anima-ref")
DEFAULT_CONFIG = DEFAULT_SERENITY / "configs" / "anima_100step_baseline.json"
DEFAULT_PREFIX = "anima_train_ref"
PRODUCER = "scripts/anima_dump_train_ref.py"
BLOCKER_SCHEMA_VERSION = 1


def default_out_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "parity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump one bounded Serenity Anima train-step parity reference."
    )
    parser.add_argument("--serenity", type=Path, default=DEFAULT_SERENITY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--secrets", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=default_out_dir())
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--train-device", default=None)
    parser.add_argument("--temp-device", default=None)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--require-baseline-steps", type=int, default=100)
    parser.add_argument(
        "--adapter-dump",
        choices=("none", "initial", "step", "step-with-grads"),
        default="step",
        help=(
            "Adapter tensor dump mode. 'step' writes trainable LoRA params at "
            "before/pre/post/after phases; 'step-with-grads' also writes grads."
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
        help="Optional process RNG seed before Serenity setup. Default preserves Serenity behavior.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write structured blockers/evidence without loading the model or training.",
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
    if args.baseline_dir is not None:
        args.baseline_dir = args.baseline_dir.resolve()
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
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    return str(value)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_file_count(path: Path, limit: int = 100000) -> int:
    if not path.is_dir():
        return 0
    count = 0
    for child in path.rglob("*"):
        if child.is_file():
            count += 1
            if count >= limit:
                return count
    return count


def sample_files(path: Path, limit: int = 8) -> list[str]:
    if not path.is_dir():
        return []
    samples: list[str] = []
    for child in path.rglob("*"):
        if child.is_file():
            samples.append(str(child))
            if len(samples) >= limit:
                break
    return samples


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
    stats.update({"min": int(values.min().item()), "max": int(values.max().item())})
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
        groups.append(json_safe({
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
        }))

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


def global_grad_norm(parameters: list[torch.nn.Parameter]) -> torch.Tensor | None:
    total: torch.Tensor | None = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().to(dtype=torch.float32)
        contribution = torch.sum(grad * grad)
        total = contribution if total is None else total + contribution
    if total is None:
        return None
    return torch.sqrt(total)


def cache_present(config: Any) -> bool:
    if not config.latent_caching:
        return True
    cache_dir = Path(config.cache_dir)
    required = [cache_dir / "image", cache_dir / "text"]
    return all(path.is_dir() and any(path.rglob("*")) for path in required)


def cache_status(config: Any) -> dict[str, Any]:
    cache_dir = Path(config.cache_dir)
    image_dir = cache_dir / "image"
    text_dir = cache_dir / "text"
    return {
        "enabled": bool(config.latent_caching),
        "cache_dir": str(cache_dir),
        "present": cache_present(config),
        "image": {
            "path": str(image_dir),
            "exists": image_dir.is_dir(),
            "file_count_bounded": bounded_file_count(image_dir),
            "sample_files": sample_files(image_dir),
        },
        "text": {
            "path": str(text_dir),
            "exists": text_dir.is_dir(),
            "file_count_bounded": bounded_file_count(text_dir),
            "sample_files": sample_files(text_dir),
        },
    }


def safe_safetensors_summary(path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except Exception as exc:
        return {"inspectable": False, "error": f"safetensors import failed: {exc}"}

    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            dtype_counts: dict[str, int] = {}
            sample_shapes: dict[str, list[int]] = {}
            for key in keys:
                tensor = handle.get_tensor(key)
                dtype = str(tensor.dtype).replace("torch.", "")
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                if len(sample_shapes) < 12:
                    sample_shapes[key] = list(tensor.shape)
    except Exception as exc:
        return {"inspectable": False, "error": f"safetensors open failed: {exc}"}

    lower_keys = [key.lower() for key in keys]

    def count_contains(*needles: str) -> int:
        return sum(1 for key in lower_keys if any(needle in key for needle in needles))

    return {
        "inspectable": True,
        "key_count": len(keys),
        "dtype_counts": dtype_counts,
        "transformer_lora_key_count": count_contains("transformer.", "transformer_blocks"),
        "sample_keys": keys[:16],
        "sample_shapes": sample_shapes,
    }


def file_status(path: Path, *, safetensors: bool = False) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
    }
    if path.is_file():
        status["size_bytes"] = path.stat().st_size
        status["sha256"] = sha256_file(path)
        if safetensors:
            status["safetensors"] = safe_safetensors_summary(path)
    return status


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metrics_summary(path: Path) -> dict[str, Any]:
    data = _load_json_if_present(path)
    if data is None:
        return {"path": str(path), "exists": False}

    progress_events = data.get("progress_events") or []
    summary = {
        "path": str(path),
        "exists": True,
        "status": data.get("status"),
        "requested_steps": data.get("requested_steps"),
        "global_steps_seen": data.get("global_steps_seen"),
        "started_at": data.get("started_at"),
        "completed_at": data.get("completed_at"),
        "progress_event_count": len(progress_events),
        "first_progress_event": progress_events[0] if progress_events else None,
        "last_progress_event": progress_events[-1] if progress_events else None,
        "torch_cuda_max_allocated_mib": data.get("torch_cuda_max_allocated_mib"),
        "torch_cuda_max_reserved_mib": data.get("torch_cuda_max_reserved_mib"),
    }
    for key in (
        "loss_count",
        "grad_norm_count",
        "last_loss",
        "last_smooth_loss",
        "last_grad_norm",
        "overall_wall_seconds",
        "train_wall_seconds",
        "mean_step_seconds_excluding_first",
        "min_step_seconds_excluding_first",
        "max_step_seconds_excluding_first",
        "mean_gpu_util_percent_sampled",
        "peak_gpu_memory_used_mib_sampled",
        "cache_dir",
        "workspace_dir",
        "output_model_destination",
    ):
        if key in data:
            summary[key] = data[key]
    return summary


def baseline_dir(config: Any, args: argparse.Namespace) -> Path:
    if args.baseline_dir is not None:
        return args.baseline_dir
    destination = Path(config.output_model_destination)
    if destination:
        return destination.parent
    return args.serenity / "output" / "anima_100step_baseline"


def baseline_status(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    root = baseline_dir(config, args)
    metrics = root / "metrics.json"
    metrics_with_grad = root / "metrics_with_grad.json"
    lora = root / "lora.safetensors"
    return {
        "required_steps": args.require_baseline_steps,
        "baseline_dir": str(root),
        "metrics": {
            **file_status(metrics),
            "summary": _metrics_summary(metrics),
        },
        "metrics_with_grad": {
            **file_status(metrics_with_grad),
            "summary": _metrics_summary(metrics_with_grad),
        },
        "lora": file_status(lora, safetensors=True),
    }


def _is_local_model_ref(model_name: str) -> bool:
    if not model_name:
        return False
    expanded = os.path.expanduser(model_name)
    path = Path(expanded)
    return (
        path.is_absolute()
        or model_name.startswith(".")
        or model_name.startswith("~")
        or path.suffix in (".safetensors", ".ckpt", ".pt", ".bin")
    )


def local_model_status(config: Any) -> dict[str, Any]:
    names = config.model_names()
    base_model = getattr(names, "base_model", "") or config.base_model_name
    transformer_model = getattr(names, "transformer_model", "") or ""
    vae_model = getattr(names, "vae_model", "") or ""
    lora_model = getattr(names, "lora", "") or ""
    base = Path(os.path.expanduser(base_model)) if base_model else Path("")

    status: dict[str, Any] = {
        "base_model": base_model,
        "transformer_model": transformer_model,
        "vae_model": vae_model,
        "lora_model": lora_model,
        "base_exists": base.exists() if base_model else False,
    }

    if base_model and base.is_dir():
        subfolders = {
            "tokenizer": (base / "tokenizer").is_dir(),
            "t5_tokenizer": (base / "t5_tokenizer").is_dir(),
            "text_encoder": (base / "text_encoder").is_dir(),
            "text_conditioner": (base / "text_conditioner").is_dir(),
            "transformer": (base / "transformer").is_dir(),
            "vae": (base / "vae").is_dir(),
            "scheduler": (base / "scheduler").is_dir(),
        }
        status.update({
            "kind": "diffusers_dir",
            "has_modular_model_index": (base / "modular_model_index.json").is_file(),
            "subfolders": subfolders,
            "file_count_bounded": bounded_file_count(base),
            "sample_files": sample_files(base),
        })
    elif base_model and base.is_file():
        status.update({
            "kind": "single_file",
            "suffix": base.suffix,
            "file": file_status(base, safetensors=base.suffix == ".safetensors"),
        })
    elif base_model:
        status["kind"] = "remote_or_missing"
    else:
        status["kind"] = "unset"

    override_status: dict[str, Any] = {}
    for name, model_ref in (("transformer_model", transformer_model), ("vae_model", vae_model)):
        if not model_ref:
            override_status[name] = {"configured": False}
            continue
        path = Path(os.path.expanduser(model_ref))
        override_status[name] = {
            "configured": True,
            "path": model_ref,
            "exists": path.exists(),
            "is_local_ref": _is_local_model_ref(model_ref),
            "kind": "dir" if path.is_dir() else "file" if path.is_file() else "missing",
        }
    status["overrides"] = override_status
    return status


def concept_status(config: Any) -> dict[str, Any]:
    concept_file = Path(config.concept_file_name) if config.concept_file_name else None
    concepts = []
    for concept in config.concepts or []:
        path = Path(concept.path) if getattr(concept, "path", None) else None
        concepts.append({
            "name": getattr(concept, "name", None),
            "enabled": bool(getattr(concept, "enabled", False)),
            "path": str(path) if path else None,
            "path_exists": path.exists() if path else False,
            "file_count_bounded": bounded_file_count(path) if path and path.exists() else 0,
        })
    concept_file_error = None
    if not concepts and concept_file is not None and concept_file.is_file():
        try:
            with concept_file.open("r", encoding="utf-8") as handle:
                for entry in json.load(handle):
                    path_value = entry.get("path")
                    path = Path(path_value) if path_value else None
                    concepts.append({
                        "name": entry.get("name"),
                        "enabled": bool(entry.get("enabled", False)),
                        "path": str(path) if path else None,
                        "path_exists": path.exists() if path else False,
                        "file_count_bounded": bounded_file_count(path) if path and path.exists() else 0,
                        "source": "concept_file_name",
                    })
        except Exception as exc:
            concept_file_error = repr(exc)
    return {
        "concept_file_name": str(concept_file) if concept_file else "",
        "concept_file_exists": concept_file.is_file() if concept_file else False,
        "concept_file_error": concept_file_error,
        "concept_count": len(concepts),
        "concepts": concepts,
    }


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


def structured_reference_blockers(config: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []

    try:
        from modules.dataLoader.BaseDataLoader import BaseDataLoader
        from modules.util import create, factory
    except Exception as exc:
        blockers.append({
            "id": "serenity_registration_import_failed",
            "category": "registration",
            "message": f"Could not import Serenity create/factory modules: {exc}",
            "details": {"error": repr(exc)},
        })
        create = None
        factory = None
        BaseDataLoader = None

    if create is not None and create.create_model_loader(config.model_type, config.training_method) is None:
        blockers.append({
            "id": "missing_model_loader_registration",
            "category": "registration",
            "message": (
                f"No Serenity model loader registered for "
                f"{enum_value(config.model_type)}/{enum_value(config.training_method)}."
            ),
            "details": {
                "model_type": enum_value(config.model_type),
                "training_method": enum_value(config.training_method),
            },
        })
    if create is not None and create.create_model_setup(
        config.model_type,
        torch.device(config.train_device),
        torch.device(config.temp_device),
        config.training_method,
        config.debug_mode,
    ) is None:
        blockers.append({
            "id": "missing_model_setup_registration",
            "category": "registration",
            "message": (
                f"No Serenity model setup registered for "
                f"{enum_value(config.model_type)}/{enum_value(config.training_method)}."
            ),
            "details": {
                "model_type": enum_value(config.model_type),
                "training_method": enum_value(config.training_method),
                "train_device": config.train_device,
                "temp_device": config.temp_device,
            },
        })
    if factory is not None and BaseDataLoader is not None:
        data_loader_cls = factory.get(BaseDataLoader, config.model_type, config.training_method)
        fallback_cls = factory.get(BaseDataLoader, config.model_type)
        if data_loader_cls is None and fallback_cls is None:
            blockers.append({
                "id": "missing_data_loader_registration",
                "category": "registration",
                "message": f"No Serenity data loader registered for {enum_value(config.model_type)}.",
                "details": {
                    "model_type": enum_value(config.model_type),
                    "training_method": enum_value(config.training_method),
                },
            })

    baseline = baseline_status(config, args)
    metrics_summary = baseline["metrics"]["summary"]
    metrics_grad_summary = baseline["metrics_with_grad"]["summary"]
    required_steps = int(args.require_baseline_steps)

    if not baseline["metrics"]["exists"]:
        blockers.append({
            "id": "missing_100_step_metrics",
            "category": "baseline",
            "message": "Required Anima 100-step baseline metrics.json is missing.",
            "details": baseline["metrics"],
        })
    elif metrics_summary.get("status") != "completed" or int(metrics_summary.get("global_steps_seen") or 0) < required_steps:
        blockers.append({
            "id": "incomplete_100_step_metrics",
            "category": "baseline",
            "message": (
                "Anima baseline metrics.json is not a completed "
                f"{required_steps}-step run."
            ),
            "details": metrics_summary,
        })

    if not baseline["metrics_with_grad"]["exists"]:
        blockers.append({
            "id": "missing_100_step_metrics_with_grad",
            "category": "baseline",
            "message": "Required Anima 100-step metrics_with_grad.json is missing.",
            "details": baseline["metrics_with_grad"],
        })
    elif metrics_grad_summary.get("status") != "completed" or int(metrics_grad_summary.get("global_steps_seen") or 0) < required_steps:
        blockers.append({
            "id": "incomplete_100_step_metrics_with_grad",
            "category": "baseline",
            "message": (
                "Anima metrics_with_grad.json is not a completed "
                f"{required_steps}-step run."
            ),
            "details": metrics_grad_summary,
        })

    if not baseline["lora"]["exists"]:
        blockers.append({
            "id": "missing_100_step_lora",
            "category": "baseline",
            "message": "Required Anima 100-step LoRA output is missing.",
            "details": baseline["lora"],
        })

    cache = cache_status(config)
    if config.latent_caching and not cache["present"] and not args.allow_cache_build:
        blockers.append({
            "id": "missing_required_cache",
            "category": "cache",
            "message": (
                "Required latent/text cache is missing or empty "
                f"(image files={cache['image']['file_count_bounded']}, "
                f"text files={cache['text']['file_count_bounded']})."
            ),
            "details": cache,
        })

    model = local_model_status(config)
    base_model = model.get("base_model") or ""
    if base_model and model.get("kind") == "remote_or_missing":
        blocker_id = "remote_base_model_ref" if not _is_local_model_ref(base_model) else "missing_base_model_path"
        blockers.append({
            "id": blocker_id,
            "category": "weights",
            "message": (
                "Anima base model is not available as a local diffusers snapshot: "
                f"{base_model}"
            ),
            "details": model,
        })
    elif model.get("kind") == "single_file":
        blockers.append({
            "id": "single_file_base_model_not_supported",
            "category": "weights",
            "message": "Anima train reference requires a local diffusers directory base model.",
            "details": model,
        })
    elif model.get("kind") == "diffusers_dir":
        subfolders = model.get("subfolders", {})
        missing = [name for name, exists in subfolders.items() if not exists]
        if not model.get("has_modular_model_index") or missing:
            blockers.append({
                "id": "incomplete_diffusers_base_model",
                "category": "weights",
                "message": (
                    "Anima local diffusers snapshot is missing required files/folders: "
                    + ", ".join(["modular_model_index.json"] if not model.get("has_modular_model_index") else [] + missing)
                ),
                "details": model,
            })

    for name, status in model.get("overrides", {}).items():
        if status.get("configured") and status.get("is_local_ref") and not status.get("exists"):
            blockers.append({
                "id": f"missing_{name}",
                "category": "weights",
                "message": f"Configured Anima {name} path does not exist: {status.get('path')}",
                "details": status,
            })

    concepts = concept_status(config)
    if concepts["concept_file_name"] and not concepts["concept_file_exists"]:
        blockers.append({
            "id": "missing_concept_file",
            "category": "data",
            "message": f"Configured concept file does not exist: {concepts['concept_file_name']}",
            "details": concepts,
        })
    missing_concepts = [
        concept for concept in concepts["concepts"]
        if concept["enabled"] and not concept["path_exists"]
    ]
    if missing_concepts:
        blockers.append({
            "id": "missing_concept_data_path",
            "category": "data",
            "message": "One or more enabled Anima concept data paths are missing.",
            "details": {"missing_concepts": missing_concepts, "concepts": concepts},
        })

    return blockers


def reference_blockers(config: Any, args: argparse.Namespace) -> list[str]:
    return [str(blocker["message"]) for blocker in structured_reference_blockers(config, args)]


def dry_run_report(args: argparse.Namespace, config: Any) -> dict[str, Any]:
    names = config.model_names()
    structured_blockers = structured_reference_blockers(config, args)
    return {
        "schema_version": BLOCKER_SCHEMA_VERSION,
        "producer": PRODUCER,
        "created_unix": time.time(),
        "serenity": str(args.serenity.resolve()),
        "config": str(args.config.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "prefix": args.prefix,
        "max_steps": args.max_steps,
        "require_baseline_steps": args.require_baseline_steps,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "train_device": config.train_device,
        "temp_device": config.temp_device,
        "model_type": enum_value(config.model_type),
        "training_method": enum_value(config.training_method),
        "model_names": {
            "base_model": getattr(names, "base_model", ""),
            "transformer_model": getattr(names, "transformer_model", ""),
            "vae_model": getattr(names, "vae_model", ""),
            "lora_model": getattr(names, "lora", ""),
        },
        "baseline": baseline_status(config, args),
        "cache": cache_status(config),
        "allow_cache_build": args.allow_cache_build,
        "local_model": local_model_status(config),
        "concepts": concept_status(config),
        "blockers": [blocker["message"] for blocker in structured_blockers],
        "structured_blockers": structured_blockers,
        "blocked": bool(structured_blockers),
        "train_dtype": enum_value(config.train_dtype),
        "fallback_train_dtype": enum_value(config.fallback_train_dtype),
        "weight_dtypes": json_safe(config.weight_dtypes().__dict__),
        "lora_weight_dtype": enum_value(config.lora_weight_dtype),
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "layer_filter": config.layer_filter,
        "layer_filter_preset": config.layer_filter_preset,
        "timestep_distribution": enum_value(config.timestep_distribution),
        "dynamic_timestep_shifting": config.dynamic_timestep_shifting,
    }


def write_dry_run_blocker_artifact(args: argparse.Namespace, info: dict[str, Any]) -> Path:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blocker_path = args.out_dir / f"{args.prefix}_blockers.json"
    info["blocker_artifact"] = str(blocker_path)
    with blocker_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(info), handle, indent=2)
    return blocker_path


def validate_config(config: Any, args: argparse.Namespace, *, check_runtime_blockers: bool = True) -> None:
    from modules.util.enum.ModelType import ModelType
    from modules.util.enum.TrainingMethod import TrainingMethod

    if config.model_type != ModelType.ANIMA:
        raise ValueError(f"Expected ANIMA config, got {config.model_type}")
    if config.training_method != TrainingMethod.LORA:
        raise ValueError(f"Expected LORA training method, got {config.training_method}")
    if args.max_steps != 1:
        raise ValueError("--max-steps must be exactly 1 for this bounded Anima dump")
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
    if check_runtime_blockers:
        blockers = reference_blockers(config, args)
        if blockers:
            raise RuntimeError("Anima reference dump blockers:\n- " + "\n- ".join(blockers))


def run_dry_run(args: argparse.Namespace, config: Any) -> None:
    info = dry_run_report(args, config)
    write_dry_run_blocker_artifact(args, info)
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
    if model_loader is None:
        raise RuntimeError(f"No model loader registered for {config.model_type}/{config.training_method}")
    if model_setup is None:
        raise RuntimeError(f"No model setup registered for {config.model_type}/{config.training_method}")

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
    if data_loader is None:
        raise RuntimeError(f"No data loader registered for {config.model_type}/{config.training_method}")
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


def named_trainable_parameters(model: Any, parameters: list[torch.nn.Parameter]) -> list[tuple[str, torch.nn.Parameter]]:
    id_to_name: dict[int, str] = {}
    for wrapper_name in ("text_encoder_lora", "transformer_lora"):
        lora_wrapper = getattr(model, wrapper_name, None)
        lora_modules = getattr(lora_wrapper, "lora_modules", None)
        if not isinstance(lora_modules, dict):
            continue
        for module_key, module in lora_modules.items():
            if not hasattr(module, "named_parameters"):
                continue
            module_prefix = getattr(module, "prefix", f"{wrapper_name}.{module_key}.")
            module_prefix = str(module_prefix).removesuffix(".")
            for name, param in module.named_parameters():
                id_to_name[id(param)] = f"{module_prefix}.{name}"

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


def adapter_stats(named_params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, Any]:
    return {name: tensor_stats(param) for name, param in named_params}


class AnimaStepTraceHooks:
    def __init__(self, setup: Any, model: Any, trace: dict[str, Any]):
        self.setup = setup
        self.model = model
        self.trace = trace
        self.orig_create_noise = None
        self.orig_get_timestep = None
        self.orig_add_noise = None
        self.orig_encode_text = None
        self.orig_scale_latents = None
        self.orig_transformer_forward = None

    def __enter__(self) -> "AnimaStepTraceHooks":
        self.orig_create_noise = self.setup._create_noise
        self.orig_get_timestep = self.setup._get_timestep_discrete
        self.orig_add_noise = self.setup._add_noise_discrete
        self.orig_encode_text = self.model.encode_text
        self.orig_scale_latents = self.model.scale_latents
        self.orig_transformer_forward = self.model.transformer.forward

        def encode_text(*args, **kwargs):
            self.trace["encode_text.tokens"] = kwargs.get("tokens")
            self.trace["encode_text.tokens_mask"] = kwargs.get("tokens_mask")
            self.trace["encode_text.cached_hidden_state"] = kwargs.get("text_encoder_output")
            out = self.orig_encode_text(*args, **kwargs)
            self.trace["text_encoder_output"] = out
            return out

        def scale_latents(latent_image):
            self.trace["latent_image_before_scale"] = latent_image
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
            self.trace["encoder_hidden_states"] = kwargs.get("encoder_hidden_states")
            self.trace["padding_mask"] = kwargs.get("padding_mask")
            out = self.orig_transformer_forward(*args, **kwargs)
            self.trace["predicted_flow"] = out[0] if isinstance(out, tuple) else getattr(out, "sample", None)
            return out

        self.model.encode_text = encode_text
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
        self.model.scale_latents = self.orig_scale_latents
        self.model.transformer.forward = self.orig_transformer_forward


def enrich_trace(trace: dict[str, Any]) -> None:
    latent_noise = trace.get("latent_noise")
    scaled_latent = trace.get("scaled_latent_image")
    if latent_noise is not None and scaled_latent is not None:
        trace["flow"] = latent_noise - scaled_latent


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)


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
        "encode_text.tokens",
        "encode_text.tokens_mask",
        "encode_text.cached_hidden_state",
        "text_encoder_output",
        "latent_image_before_scale",
        "scaled_latent_image",
        "noise_source_tensor",
        "latent_noise",
        "scaled_noisy_latent_image",
        "sigma",
        "transformer_hidden_states",
        "transformer_timestep",
        "encoder_hidden_states",
        "padding_mask",
        "predicted_flow",
        "flow",
    ]:
        value = trace.get(key)
        if torch.is_tensor(value):
            add_tensor(tensors, tensor_meta, f"trace.{key}", value)

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
            "producer": PRODUCER,
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
                "producer": PRODUCER,
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
        "timestep_shift": json_safe(trace.get("timestep_shift")),
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
    if args.adapter_dump == "initial" and named_params:
        initial_tensors: dict[str, torch.Tensor] = {}
        add_adapter_tensors(initial_tensors, named_params, "adapter_initial")
        initial_adapter_path = args.out_dir / f"{args.prefix}_initial_adapters.safetensors"
        write_safetensors(
            initial_adapter_path,
            initial_tensors,
            metadata={"producer": PRODUCER, "adapter_dump": "initial"},
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
        if args.adapter_dump in ("step", "step-with-grads") and named_params:
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
                with AnimaStepTraceHooks(model_setup, model, prior_trace):
                    prior_model_output_data = model_setup.predict(model, batch, config, train_progress)
            with AnimaStepTraceHooks(model_setup, model, trace):
                model_output_data = model_setup.predict(model, batch, config, train_progress)
            prior_model_prediction = prior_model_output_data["predicted"].to(
                dtype=model_output_data["target"].dtype
            )
            model_output_data["target"][prior_pred_indices] = prior_model_prediction[prior_pred_indices]
            model_output_data["prior_target"] = prior_model_prediction
            trace["prior_trace"] = prior_trace
        else:
            with AnimaStepTraceHooks(model_setup, model, trace):
                model_output_data = model_setup.predict(model, batch, config, train_progress)

        enrich_trace(trace)

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

        if adapter_tensors is not None:
            add_adapter_tensors(adapter_tensors, named_params, "adapter_pre")

        lr_before = lr_scheduler.get_last_lr()
        optimizer_before = optimizer_meta(model.optimizer)
        should_update = (train_progress.global_step + 1) % update_every == 0
        if should_update:
            if scaler:
                scaler.unscale_(model.optimizer)
                if config.clip_grad_norm is not None:
                    grad_norm = nn.utils.clip_grad_norm_(parameters, config.clip_grad_norm)
                    grad_norm_pre_clip = float(grad_norm.detach().cpu().item())
                if adapter_tensors is not None:
                    add_adapter_tensors(adapter_tensors, named_params, "adapter_post")
                if args.adapter_dump == "step-with-grads" and adapter_tensors is not None:
                    add_adapter_tensors(adapter_tensors, named_params, "adapter_post_clip", include_grad=True)
                scaler.step(model.optimizer)
                scaler.update()
            else:
                if config.clip_grad_norm is not None:
                    grad_norm = nn.utils.clip_grad_norm_(parameters, config.clip_grad_norm)
                    grad_norm_pre_clip = float(grad_norm.detach().cpu().item())
                if adapter_tensors is not None:
                    add_adapter_tensors(adapter_tensors, named_params, "adapter_post")
                if args.adapter_dump == "step-with-grads" and adapter_tensors is not None:
                    add_adapter_tensors(adapter_tensors, named_params, "adapter_post_clip", include_grad=True)
                model.optimizer.step()

            lr_scheduler.step()
            model.optimizer.zero_grad(set_to_none=True)
            model_setup.after_optimizer_step(model, config, train_progress)
        elif adapter_tensors is not None:
            add_adapter_tensors(adapter_tensors, named_params, "adapter_post")

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
        "producer": PRODUCER,
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
            "fallback_train_dtype": enum_value(config.fallback_train_dtype),
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
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_weight_dtype": enum_value(config.lora_weight_dtype),
            "layer_filter": config.layer_filter,
            "layer_filter_preset": config.layer_filter_preset,
        },
        "baseline": baseline_status(config, args),
        "local_model": local_model_status(config),
        "cache": cache_status(config),
        "trainable_parameters": {
            "count": len(named_params),
            "numel": int(sum(param.numel() for _, param in named_params)),
            "names": [name for name, _ in named_params],
            "stats": adapter_stats(named_params),
        },
        "steps": step_results,
    }


def main() -> None:
    args = parse_args()
    enter_serenity_root(args)
    add_serenity_to_path(args.serenity)
    config = load_train_config(args)
    validate_config(config, args, check_runtime_blockers=not args.dry_run)

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
