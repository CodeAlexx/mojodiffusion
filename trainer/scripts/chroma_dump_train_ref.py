#!/usr/bin/env python3
"""
Dump bounded Serenity Chroma train-step reference data for Mojo parity gates.

Dry-run mode is blocker-aware and does not load model weights. It writes:
  parity/chroma_train_ref_contract.json
  parity/chroma_train_ref_blockers.json
  parity/chroma_train_ref_meta.json

When a local Chroma 100-step baseline config/cache/weights are available, run:
  /home/alex/Serenity/venv/bin/python scripts/chroma_dump_train_ref.py --max-steps=1
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
from safetensors.torch import save_file
from torch import nn


DEFAULT_SERENITY = Path("/home/alex/Serenity")
DEFAULT_CONFIG = DEFAULT_SERENITY / "configs" / "chroma_100step_baseline.json"
DEFAULT_OUT_DIR = Path("/home/alex/serenity-trainer/parity")
DEFAULT_PREFIX = "chroma_train_ref"
DEFAULT_BASELINE_DIR = DEFAULT_SERENITY / "output" / "chroma_100step_baseline"
DEFAULT_CACHE_DIR = DEFAULT_SERENITY / "workspace-cache" / "chroma_100step_baseline"
DEFAULT_REMOTE_MODEL = "lodestones/Chroma1-HD"
PRODUCER = "scripts/chroma_dump_train_ref.py"
BLOCKER_SCHEMA_VERSION = 1

REFERENCE_SOURCES = [
    "/home/alex/Serenity/modules/modelSetup/ChromaLoRASetup.py",
    "/home/alex/Serenity/modules/modelSetup/ChromaFineTuneSetup.py",
    "/home/alex/Serenity/modules/modelSetup/BaseChromaSetup.py",
    "/home/alex/Serenity/modules/dataLoader/ChromaBaseDataLoader.py",
    "/home/alex/Serenity/modules/model/ChromaModel.py",
    "/home/alex/Serenity/modules/modelSampler/ChromaSampler.py",
]

PRESET_CANDIDATES = [
    DEFAULT_SERENITY / "training_presets" / "#chroma LoRA 8GB.json",
    DEFAULT_SERENITY / "training_presets" / "#chroma LoRA 16GB.json",
    DEFAULT_SERENITY / "training_presets" / "#chroma LoRA 24GB.json",
    DEFAULT_SERENITY / "training_presets" / "#chroma Finetune 8GB.json",
    DEFAULT_SERENITY / "training_presets" / "#chroma Finetune 16GB.json",
    DEFAULT_SERENITY / "training_presets" / "#chroma Finetune 24GB.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump one bounded Serenity Chroma train-step parity reference."
    )
    parser.add_argument("--serenity", type=Path, default=DEFAULT_SERENITY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--secrets", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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
            "Adapter tensor dump mode. 'step' writes trainable Chroma LoRA params "
            "at before/pre/post/after phases; 'step-with-grads' also writes grads."
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
        help="Write structured blockers/evidence without loading Chroma model weights.",
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


def file_status(path: Path, *, safetensors: bool = False) -> dict[str, Any]:
    status: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        status["size_bytes"] = path.stat().st_size
        status["sha256"] = sha256_file(path)
        if safetensors:
            status["safetensors"] = safe_safetensors_summary(path)
    return status


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
    return {
        "inspectable": True,
        "key_count": len(keys),
        "dtype_counts": dtype_counts,
        "transformer_lora_key_count": sum(
            1 for key in lower_keys
            if "transformer" in key and ("lora_down" in key or "lora_up" in key)
        ),
        "text_encoder_lora_key_count": sum(
            1 for key in lower_keys
            if ("lora_te" in key or "text_encoder" in key) and ("lora_down" in key or "lora_up" in key)
        ),
        "sample_keys": keys[:16],
        "sample_shapes": sample_shapes,
    }


def load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metrics_summary(path: Path) -> dict[str, Any]:
    data = load_json_if_present(path)
    if data is None:
        return {"path": str(path), "exists": False}

    progress_events = data.get("progress_events") or []
    summary = {
        "path": str(path),
        "exists": True,
        "status": data.get("status"),
        "requested_steps": data.get("requested_steps"),
        "global_steps_seen": data.get("global_steps_seen"),
        "progress_event_count": len(progress_events),
        "first_progress_event": progress_events[0] if progress_events else None,
        "last_progress_event": progress_events[-1] if progress_events else None,
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
        "torch_cuda_max_allocated_mib",
        "torch_cuda_max_reserved_mib",
        "cache_dir",
        "workspace_dir",
        "output_model_destination",
    ):
        if key in data:
            summary[key] = data[key]
    return summary


def resolve_maybe_relative(path: str | Path | None, base: Path) -> Path:
    if path is None or str(path) == "":
        return Path("")
    value = Path(os.path.expanduser(str(path)))
    return value if value.is_absolute() else base / value


def path_is_set(path: Path) -> bool:
    return str(path) not in ("", ".")


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


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)


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


def baseline_dir(config: Any | None, args: argparse.Namespace) -> Path:
    if args.baseline_dir is not None:
        return args.baseline_dir
    if config is not None:
        destination = resolve_maybe_relative(
            getattr(config, "output_model_destination", ""), args.serenity
        )
        if path_is_set(destination):
            return destination.parent
    return DEFAULT_BASELINE_DIR


def baseline_status(config: Any | None, args: argparse.Namespace) -> dict[str, Any]:
    root = baseline_dir(config, args)
    configured_destination = None
    if config is not None:
        configured_destination = resolve_maybe_relative(
            getattr(config, "output_model_destination", ""), args.serenity
        )
    candidates = []
    if configured_destination and path_is_set(configured_destination):
        candidates.append(configured_destination)
    candidates += [
        root / "lora.safetensors",
        root / "lora_last.safetensors",
        root / "model.safetensors",
    ]

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)

    metrics = root / "metrics.json"
    metrics_with_grad = root / "metrics_with_grad.json"
    artifact_statuses = [file_status(path, safetensors=path.suffix == ".safetensors") for path in deduped]
    metrics_info = metrics_summary(metrics)
    required_steps = int(args.require_baseline_steps)
    complete_metrics = (
        metrics_info.get("status") == "completed"
        and int(metrics_info.get("global_steps_seen") or 0) >= required_steps
    )
    return {
        "required_steps": required_steps,
        "baseline_dir": str(root),
        "metrics": {**file_status(metrics), "summary": metrics_info},
        "metrics_with_grad": {
            **file_status(metrics_with_grad),
            "summary": metrics_summary(metrics_with_grad),
        },
        "model_artifacts": artifact_statuses,
        "complete": bool(complete_metrics and any(item["exists"] for item in artifact_statuses)),
    }


def text_cache_required(config: Any | None) -> bool:
    if config is None:
        return True
    try:
        return not bool(config.train_text_encoder_or_embedding())
    except Exception:
        return True


def cache_status(config: Any | None, args: argparse.Namespace) -> dict[str, Any]:
    enabled = bool(getattr(config, "latent_caching", True)) if config is not None else True
    cache_dir = resolve_maybe_relative(
        getattr(config, "cache_dir", str(DEFAULT_CACHE_DIR)) if config is not None else DEFAULT_CACHE_DIR,
        args.serenity,
    )
    image_dir = cache_dir / "image"
    text_dir = cache_dir / "text"
    image_files = bounded_file_count(image_dir)
    text_files = bounded_file_count(text_dir)
    text_required = text_cache_required(config)
    present = True
    if enabled:
        present = image_files > 0 and (text_files > 0 if text_required else True)
    return {
        "enabled": enabled,
        "cache_dir": str(cache_dir),
        "present": present,
        "text_cache_required": text_required,
        "image": {
            "path": str(image_dir),
            "exists": image_dir.is_dir(),
            "file_count_bounded": image_files,
            "sample_files": sample_files(image_dir),
        },
        "text": {
            "path": str(text_dir),
            "exists": text_dir.is_dir(),
            "file_count_bounded": text_files,
            "sample_files": sample_files(text_dir),
        },
    }


def is_local_model_ref(model_name: str) -> bool:
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


def hf_cache_status(model_id: str) -> dict[str, Any]:
    if "/" not in model_id or is_local_model_ref(model_id):
        return {"model_id": model_id, "applicable": False}
    root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
    snapshots = root / "snapshots"
    snapshot_dirs = [str(path) for path in sorted(snapshots.iterdir())] if snapshots.is_dir() else []
    return {
        "model_id": model_id,
        "applicable": True,
        "expected_cache_root": str(root),
        "cache_root_exists": root.is_dir(),
        "snapshots_dir": str(snapshots),
        "snapshots_dir_exists": snapshots.is_dir(),
        "snapshot_dirs": snapshot_dirs,
    }


def model_names_summary(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {
            "base_model": DEFAULT_REMOTE_MODEL,
            "vae_model": "",
            "lora_model": "",
            "source": "default_when_config_missing",
        }
    names = config.model_names()
    return {
        "base_model": getattr(names, "base_model", "") or getattr(config, "base_model_name", ""),
        "vae_model": getattr(names, "vae_model", ""),
        "lora_model": getattr(names, "lora", ""),
        "include_text_encoder": getattr(names, "include_text_encoder", None),
    }


def local_model_status(config: Any | None, args: argparse.Namespace) -> dict[str, Any]:
    names = model_names_summary(config)
    base_model = names.get("base_model") or DEFAULT_REMOTE_MODEL
    base_path = Path(os.path.expanduser(base_model))
    status: dict[str, Any] = {
        "model_names": names,
        "base_model": base_model,
        "is_local_ref": is_local_model_ref(base_model),
        "hf_cache": hf_cache_status(base_model),
    }

    if base_model and base_path.is_dir():
        required = ["tokenizer", "text_encoder", "transformer", "vae", "scheduler"]
        required_paths = {name: str(base_path / name) for name in required}
        subfolders = {name: (base_path / name).is_dir() for name in required}
        status.update({
            "kind": "diffusers_dir",
            "base_exists": True,
            "has_model_index_json": (base_path / "model_index.json").is_file(),
            "has_modular_model_index_json": (base_path / "modular_model_index.json").is_file(),
            "required_subfolders": required,
            "required_paths": required_paths,
            "subfolders": subfolders,
            "file_count_bounded": bounded_file_count(base_path),
            "sample_files": sample_files(base_path),
        })
    elif base_model and base_path.is_file():
        status.update({
            "kind": "single_file",
            "base_exists": True,
            "suffix": base_path.suffix,
            "file": file_status(base_path, safetensors=base_path.suffix == ".safetensors"),
        })
    elif base_model:
        status.update({
            "kind": "remote_or_missing",
            "base_exists": False,
            "path_if_local": str(base_path),
        })
    else:
        status.update({"kind": "unset", "base_exists": False})

    overrides: dict[str, Any] = {}
    for key in ("vae_model", "lora_model"):
        value = names.get(key) or ""
        if not value:
            overrides[key] = {"configured": False}
            continue
        path = Path(os.path.expanduser(value))
        overrides[key] = {
            "configured": True,
            "path": value,
            "is_local_ref": is_local_model_ref(value),
            "exists": path.exists(),
            "kind": "dir" if path.is_dir() else "file" if path.is_file() else "missing",
        }
    status["overrides"] = overrides
    return status


def concept_status(config: Any | None, args: argparse.Namespace) -> dict[str, Any]:
    if config is None:
        return {"available": False, "reason": "config_missing"}
    concept_file_value = getattr(config, "concept_file_name", "")
    concept_file = (
        resolve_maybe_relative(concept_file_value, args.serenity)
        if concept_file_value else Path("")
    )
    concepts = []
    for concept in getattr(config, "concepts", []) or []:
        path_value = getattr(concept, "path", None)
        path = resolve_maybe_relative(path_value, args.serenity) if path_value else Path("")
        concepts.append({
            "name": getattr(concept, "name", None),
            "enabled": bool(getattr(concept, "enabled", False)),
            "path": str(path) if str(path) else "",
            "path_exists": path.exists() if str(path) else False,
            "file_count_bounded": bounded_file_count(path) if path.exists() else 0,
        })

    concept_file_error = None
    if not concepts and path_is_set(concept_file) and concept_file.is_file():
        try:
            with concept_file.open("r", encoding="utf-8") as handle:
                for entry in json.load(handle):
                    path_value = entry.get("path")
                    path = resolve_maybe_relative(path_value, args.serenity) if path_value else Path("")
                    concepts.append({
                        "name": entry.get("name"),
                        "enabled": bool(entry.get("enabled", False)),
                        "path": str(path) if str(path) else "",
                        "path_exists": path.exists() if str(path) else False,
                        "file_count_bounded": bounded_file_count(path) if path.exists() else 0,
                        "source": "concept_file_name",
                    })
        except Exception as exc:
            concept_file_error = repr(exc)

    return {
        "available": True,
        "concept_file_name": str(concept_file) if path_is_set(concept_file) else "",
        "concept_file_exists": concept_file.is_file() if path_is_set(concept_file) else False,
        "concept_file_error": concept_file_error,
        "concept_count": len(concepts),
        "concepts": concepts,
    }


def load_train_config(args: argparse.Namespace) -> Any:
    from modules.util.config.SecretsConfig import SecretsConfig
    from modules.util.config.TrainConfig import TrainConfig

    if not args.config.is_file():
        raise FileNotFoundError(f"Chroma train reference config not found: {args.config}")

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


def preset_candidates_status() -> list[dict[str, Any]]:
    candidates = []
    for path in PRESET_CANDIDATES:
        info: dict[str, Any] = file_status(path)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                info["model_type"] = data.get("model_type")
                info["training_method"] = data.get("training_method")
                info["base_model_name"] = data.get("base_model_name")
            except Exception as exc:
                info["json_error"] = repr(exc)
        candidates.append(info)
    return candidates


def structured_reference_blockers(
    config: Any | None,
    args: argparse.Namespace,
    config_error: str | None = None,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []

    if config is None:
        blockers.append({
            "id": "missing_chroma_reference_config",
            "category": "config",
            "message": f"Required Chroma 100-step baseline config is missing: {args.config}",
            "details": {
                "config": str(args.config),
                "error": config_error,
                "preset_candidates": preset_candidates_status(),
            },
        })
    else:
        from modules.util.enum.ModelType import ModelType
        from modules.util.enum.TrainingMethod import TrainingMethod

        if config.model_type != ModelType.CHROMA_1:
            blockers.append({
                "id": "wrong_model_type",
                "category": "config",
                "message": f"Expected CHROMA_1 config, got {enum_value(config.model_type)}.",
                "details": {"model_type": enum_value(config.model_type)},
            })
        if config.training_method not in (TrainingMethod.LORA, TrainingMethod.FINE_TUNE):
            blockers.append({
                "id": "unsupported_training_method",
                "category": "config",
                "message": (
                    "Chroma train reference supports LORA or FINE_TUNE configs, "
                    f"got {enum_value(config.training_method)}."
                ),
                "details": {"training_method": enum_value(config.training_method)},
            })
        if args.max_steps != 1:
            blockers.append({
                "id": "unsupported_max_steps",
                "category": "config",
                "message": "--max-steps must be exactly 1 for this bounded Chroma dump.",
                "details": {"max_steps": args.max_steps},
            })
        if args.require_baseline_steps <= 0:
            blockers.append({
                "id": "invalid_required_baseline_steps",
                "category": "config",
                "message": "--require-baseline-steps must be positive.",
                "details": {"require_baseline_steps": args.require_baseline_steps},
            })
        if getattr(config.optimizer, "fused_back_pass", False):
            blockers.append({
                "id": "unsupported_fused_back_pass",
                "category": "runtime",
                "message": "fused_back_pass is not supported by this bounded dump loop.",
                "details": json_safe(getattr(config.optimizer, "__dict__", {})),
            })
        optimizer = getattr(config.optimizer, "optimizer", None)
        if optimizer is not None and getattr(optimizer, "is_schedule_free", False):
            blockers.append({
                "id": "unsupported_schedule_free_optimizer",
                "category": "runtime",
                "message": "schedule-free optimizer mode is not supported by this dump loop.",
                "details": {"optimizer": enum_value(optimizer)},
            })

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
    metrics_info = baseline["metrics"]["summary"]
    if not baseline["metrics"]["exists"]:
        blockers.append({
            "id": "missing_100_step_metrics",
            "category": "baseline",
            "message": "Required Chroma 100-step baseline metrics.json is missing.",
            "details": baseline["metrics"],
        })
    elif metrics_info.get("status") != "completed" or int(metrics_info.get("global_steps_seen") or 0) < args.require_baseline_steps:
        blockers.append({
            "id": "incomplete_100_step_metrics",
            "category": "baseline",
            "message": (
                "Chroma baseline metrics.json is not a completed "
                f"{args.require_baseline_steps}-step run."
            ),
            "details": metrics_info,
        })
    if not any(item["exists"] for item in baseline["model_artifacts"]):
        blockers.append({
            "id": "missing_100_step_model_artifact",
            "category": "baseline",
            "message": "Required Chroma 100-step LoRA/model output artifact is missing.",
            "details": {
                "baseline_dir": baseline["baseline_dir"],
                "model_artifacts": baseline["model_artifacts"],
            },
        })

    cache = cache_status(config, args)
    if cache["enabled"] and not cache["present"] and not args.allow_cache_build:
        blockers.append({
            "id": "missing_required_cache",
            "category": "cache",
            "message": (
                "Required Chroma latent/text cache is missing or empty "
                f"(image files={cache['image']['file_count_bounded']}, "
                f"text files={cache['text']['file_count_bounded']}, "
                f"text required={cache['text_cache_required']})."
            ),
            "details": cache,
        })

    model = local_model_status(config, args)
    base_model = model.get("base_model") or ""
    if base_model and model.get("kind") == "remote_or_missing":
        blocker_id = "remote_base_model_ref" if not is_local_model_ref(base_model) else "missing_base_model_path"
        blockers.append({
            "id": blocker_id,
            "category": "weights",
            "message": f"Chroma base model is not available as a local diffusers snapshot: {base_model}",
            "details": model,
        })
    elif model.get("kind") == "single_file":
        blockers.append({
            "id": "single_file_base_model_not_supported",
            "category": "weights",
            "message": "Chroma train reference requires a local diffusers directory base model.",
            "details": model,
        })
    elif model.get("kind") == "diffusers_dir":
        subfolders = model.get("subfolders", {})
        missing = [name for name, exists in subfolders.items() if not exists]
        if missing:
            blockers.append({
                "id": "missing_diffusers_chroma_folders",
                "category": "weights",
                "message": (
                    "Local Chroma diffusers snapshot is missing required folders: "
                    + ", ".join(missing)
                ),
                "details": {
                    "base_model": base_model,
                    "missing_folders": missing,
                    "required_paths": model.get("required_paths"),
                    "subfolders": subfolders,
                },
            })

    for name, status in model.get("overrides", {}).items():
        if status.get("configured") and status.get("is_local_ref") and not status.get("exists"):
            blockers.append({
                "id": f"missing_{name}",
                "category": "weights",
                "message": f"Configured Chroma {name} path does not exist: {status.get('path')}",
                "details": status,
            })

    concepts = concept_status(config, args)
    if concepts.get("concept_file_name") and not concepts.get("concept_file_exists"):
        blockers.append({
            "id": "missing_concept_file",
            "category": "data",
            "message": f"Configured concept file does not exist: {concepts.get('concept_file_name')}",
            "details": concepts,
        })
    for concept in concepts.get("concepts", []):
        if concept.get("enabled") and not concept.get("path_exists"):
            blockers.append({
                "id": "missing_enabled_concept_path",
                "category": "data",
                "message": f"Enabled concept path does not exist: {concept.get('path')}",
                "details": concept,
            })

    return blockers


def reference_blockers(config: Any, args: argparse.Namespace) -> list[str]:
    return [str(blocker["message"]) for blocker in structured_reference_blockers(config, args)]


def contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": args.prefix,
        "purpose": (
            "Serenity Chroma cached-data train parity dump contract for "
            "predict -> loss -> backward_lora_or_trainable -> optimizer gates."
        ),
        "producer_script": "/home/alex/serenity-trainer/scripts/chroma_dump_train_ref.py",
        "reference_policy": {
            "only_reference": "/home/alex/Serenity",
            "external_references_used": False,
        },
        "reference_files_inspected": REFERENCE_SOURCES,
        "source_line_anchors": {
            "BaseChromaSetup.predict": "/home/alex/Serenity/modules/modelSetup/BaseChromaSetup.py:158",
            "ChromaBaseDataLoader.cache_output_names": "/home/alex/Serenity/modules/dataLoader/ChromaBaseDataLoader.py:58",
            "ChromaModel.encode_text": "/home/alex/Serenity/modules/model/ChromaModel.py:156",
            "ChromaModel.prepare_pack_unpack": "/home/alex/Serenity/modules/model/ChromaModel.py:232",
            "ChromaSampler.no_grad_sampling": "/home/alex/Serenity/modules/modelSampler/ChromaSampler.py:36",
        },
        "default_config": str(DEFAULT_CONFIG),
        "default_reference_baseline": {
            "required_steps": args.require_baseline_steps,
            "baseline_dir": str(DEFAULT_BASELINE_DIR),
            "cache_dir": str(DEFAULT_CACHE_DIR),
            "remote_preset_base_model": DEFAULT_REMOTE_MODEL,
        },
        "preset_candidates": [str(path) for path in PRESET_CANDIDATES],
        "default_outputs": {
            "run_metadata": f"parity/{args.prefix}_meta.json",
            "step_tensors": f"parity/{args.prefix}_step000.safetensors",
            "adapter_tensors": f"parity/{args.prefix}_step000_adapters.safetensors",
            "dry_run_blockers": f"parity/{args.prefix}_blockers.json",
            "contract": f"parity/{args.prefix}_contract.json",
        },
        "serenity_reference_contract": {
            "setup": (
                "BaseChromaSetup.predict derives batch_seed from train_progress, "
                "encodes or consumes cached T5 hidden states, applies VAE "
                "(latent_image - shift_factor) * scaling_factor, samples noise and "
                "discrete timestep, adds noise returning sigma, prepares txt/img ids "
                "and optional combined attention_mask, packs latents, calls "
                "ChromaTransformer2DModel.forward, unpacks predicted flow, and targets "
                "latent_noise - scaled_latent_image."
            ),
            "loss": (
                "BaseChromaSetup.calculate_loss delegates to Serenity "
                "_flow_matching_losses(..., sigmas=model.noise_scheduler.sigmas).mean()."
            ),
            "data_loader": (
                "ChromaBaseDataLoader caches latent_image plus image metadata; when "
                "the text encoder is frozen it also caches tokens, tokens_mask, and "
                "text_encoder_hidden_state, then pads masked tokens on output."
            ),
            "sampler_boundary": (
                "ChromaSampler samples under torch.no_grad(); this reference producer "
                "is training-only and does not claim sampler parity."
            ),
        },
        "step_safetensors_keys": {
            "cached_batch": (
                "batch.* tensors exactly as Serenity data_loader yields them, "
                "including latent_image, tokens, tokens_mask, optional "
                "text_encoder_hidden_state, loss_weight, concept_type, image metadata, "
                "and optional latent_mask."
            ),
            "trace": (
                "trace.* tensors captured from BaseChromaSetup.predict, "
                "ChromaModel.encode_text, prepare_latent_image_ids, pack_latents, "
                "unpack_latents, and ChromaTransformer2DModel.forward. Expected keys "
                "include text_encoder_output, text_attention_mask, scaled_latent_image, "
                "latent_noise, scaled_noisy_latent_image, sigma, latent_input, "
                "packed_latent_input, image_ids, text_ids, transformer_hidden_states, "
                "transformer_timestep, encoder_hidden_states, attention_mask, "
                "packed_predicted_flow, predicted_flow, and flow."
            ),
            "outputs": (
                "output.timestep, output.predicted, output.target, optional "
                "output.prior_target, output.loss_pre_scale, and "
                "output.loss_for_backward."
            ),
        },
        "adapter_safetensors_keys": {
            "initial": "adapter_initial.* trainable Chroma LoRA tensors when --adapter-dump initial is used.",
            "step": "adapter_before.*, adapter_pre.*, adapter_post.*, and adapter_after.* trainable Chroma LoRA tensors.",
            "step-with-grads": (
                "adapter_before.*, adapter_pre.*, adapter_pre_clip.*, "
                "adapter_pre_clip_grad.*, adapter_post.*, adapter_post_clip.*, "
                "adapter_post_clip_grad.*, and adapter_after.* trainable Chroma "
                "LoRA tensors/gradients."
            ),
        },
        "dry_run_contract": {
            "required_reports": [
                "config path and config load error if missing",
                "model_type and training_method when config loads",
                "train_device and temp_device",
                "100-step baseline metrics and output artifact status",
                "cache directory, image cache status, and text cache status",
                "local Chroma diffusers snapshot status and exact missing folders",
                "configured concept file and enabled concept paths",
                "structured_blockers with stable id, category, message, and details",
            ],
            "required_blockers_when_missing": [
                "missing_chroma_reference_config",
                "missing_100_step_metrics",
                "missing_100_step_model_artifact",
                "missing_required_cache",
                "remote_base_model_ref or missing_base_model_path",
                "missing_diffusers_chroma_folders",
            ],
        },
        "bounded_run_notes": [
            "The script is single-process only and enforces --max-steps 1.",
            "Dry-run does not load Chroma weights.",
            "By default it requires a completed local 100-step baseline, existing image/text caches, and local diffusers weights.",
            "Pass --allow-cache-build only when local data and local text encoder weights are present.",
            "It imports Serenity from /home/alex/Serenity and chdirs there before importing modules.util.create because factory.import_dir uses cwd-relative module paths.",
            "It preserves tensor storage dtype in safetensors dumps; scalar stats in JSON are computed in float32 where needed.",
            "It does not call GenericTrainer.end(), so it does not perform Serenity final model saves.",
            "It does not edit Serenity source or Mojo runtime files.",
        ],
    }


def write_contract_artifact(args: argparse.Namespace) -> Path:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.prefix}_contract.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(contract(args)), handle, indent=2)
    return path


def dry_run_report(
    args: argparse.Namespace,
    config: Any | None,
    config_error: str | None,
) -> dict[str, Any]:
    structured_blockers = structured_reference_blockers(config, args, config_error)
    names = model_names_summary(config)
    return {
        "schema_version": BLOCKER_SCHEMA_VERSION,
        "producer": PRODUCER,
        "created_unix": time.time(),
        "dry_run": True,
        "one_step_dump_produced": False,
        "serenity": str(args.serenity.resolve()),
        "config": str(args.config.resolve()),
        "config_exists": args.config.is_file(),
        "config_error": config_error,
        "out_dir": str(args.out_dir.resolve()),
        "prefix": args.prefix,
        "max_steps": args.max_steps,
        "required_baseline_steps": args.require_baseline_steps,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "train_device": getattr(config, "train_device", args.train_device),
        "temp_device": getattr(config, "temp_device", args.temp_device),
        "model_type": enum_value(getattr(config, "model_type", "CHROMA_1")),
        "training_method": enum_value(getattr(config, "training_method", None)),
        "model_names": names,
        "baseline": baseline_status(config, args),
        "cache": cache_status(config, args),
        "allow_cache_build": args.allow_cache_build,
        "local_model": local_model_status(config, args),
        "data": concept_status(config, args),
        "preset_candidates": preset_candidates_status(),
        "blockers": [blocker["message"] for blocker in structured_blockers],
        "structured_blockers": structured_blockers,
        "blocked": bool(structured_blockers),
        "reference_files_inspected": REFERENCE_SOURCES,
        "train_dtype": enum_value(getattr(config, "train_dtype", None)),
        "fallback_train_dtype": enum_value(getattr(config, "fallback_train_dtype", None)),
        "weight_dtypes": json_safe(config.weight_dtypes().__dict__) if config is not None else None,
        "lora_weight_dtype": enum_value(getattr(config, "lora_weight_dtype", None)),
        "lora_rank": getattr(config, "lora_rank", None),
        "lora_alpha": getattr(config, "lora_alpha", None),
        "layer_filter": getattr(config, "layer_filter", None),
        "layer_filter_preset": getattr(config, "layer_filter_preset", None),
    }


def write_json_artifact(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, indent=2)
    return path


def write_dry_run_artifacts(
    args: argparse.Namespace,
    config: Any | None,
    config_error: str | None,
) -> dict[str, str]:
    write_contract_artifact(args)
    info = dry_run_report(args, config, config_error)
    blocker_path = args.out_dir / f"{args.prefix}_blockers.json"
    meta_path = args.out_dir / f"{args.prefix}_meta.json"
    info["blocker_artifact"] = str(blocker_path)
    info["meta_artifact"] = str(meta_path)
    write_json_artifact(blocker_path, info)
    write_json_artifact(meta_path, {
        "producer": PRODUCER,
        "created_unix": time.time(),
        "dry_run": True,
        "one_step_dump_produced": False,
        "blocked": bool(info["structured_blockers"]),
        "blocker_artifact": str(blocker_path),
        "contract_artifact": str(args.out_dir / f"{args.prefix}_contract.json"),
        "structured_blockers": info["structured_blockers"],
        "runtime_config": {
            "model_type": info["model_type"],
            "training_method": info["training_method"],
            "train_device": info["train_device"],
            "temp_device": info["temp_device"],
            "cache_dir": info["cache"]["cache_dir"],
        },
        "steps": [],
    })
    return {"blockers": str(blocker_path), "meta": str(meta_path)}


def validate_config(config: Any, args: argparse.Namespace, *, check_runtime_blockers: bool = True) -> None:
    from modules.util.enum.ModelType import ModelType
    from modules.util.enum.TrainingMethod import TrainingMethod

    if config.model_type != ModelType.CHROMA_1:
        raise ValueError(f"Expected CHROMA_1 config, got {config.model_type}")
    if config.training_method not in (TrainingMethod.LORA, TrainingMethod.FINE_TUNE):
        raise ValueError(f"Expected LORA or FINE_TUNE training method, got {config.training_method}")
    if args.max_steps != 1:
        raise ValueError("--max-steps must be exactly 1 for this bounded Chroma dump")
    if args.require_baseline_steps <= 0:
        raise ValueError("--require-baseline-steps must be positive")
    if getattr(config.optimizer, "fused_back_pass", False):
        raise NotImplementedError("fused_back_pass is not supported by this bounded dump loop")
    optimizer = getattr(config.optimizer, "optimizer", None)
    if optimizer is not None and getattr(optimizer, "is_schedule_free", False):
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
            raise RuntimeError("Chroma reference dump blockers:\n- " + "\n- ".join(blockers))


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


class ChromaStepTraceHooks:
    def __init__(self, setup: Any, model: Any, trace: dict[str, Any]):
        self.setup = setup
        self.model = model
        self.trace = trace
        self.orig_create_noise = None
        self.orig_get_timestep = None
        self.orig_add_noise = None
        self.orig_encode_text = None
        self.orig_prepare_latent_image_ids = None
        self.orig_pack_latents = None
        self.orig_unpack_latents = None
        self.orig_transformer_forward = None

    def __enter__(self) -> "ChromaStepTraceHooks":
        self.orig_create_noise = self.setup._create_noise
        self.orig_get_timestep = self.setup._get_timestep_discrete
        self.orig_add_noise = self.setup._add_noise_discrete
        self.orig_encode_text = self.model.encode_text
        self.orig_prepare_latent_image_ids = self.model.prepare_latent_image_ids
        self.orig_pack_latents = self.model.pack_latents
        self.orig_unpack_latents = self.model.unpack_latents
        self.orig_transformer_forward = self.model.transformer.forward

        def encode_text(*args, **kwargs):
            self.trace["encode_text.tokens"] = kwargs.get("tokens")
            self.trace["encode_text.tokens_mask"] = kwargs.get("tokens_mask")
            self.trace["encode_text.cached_hidden_state"] = kwargs.get("text_encoder_output")
            self.trace["encode_text.dropout_probability"] = kwargs.get("text_encoder_dropout_probability")
            out, mask = self.orig_encode_text(*args, **kwargs)
            self.trace["text_encoder_output"] = out
            self.trace["text_attention_mask"] = mask
            return out, mask

        def create_noise(source_tensor, config, generator, timestep=None, betas=None):
            self.trace["noise_source_tensor"] = source_tensor
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

        def prepare_latent_image_ids(height, width, device, dtype):
            out = self.orig_prepare_latent_image_ids(height, width, device, dtype)
            self.trace["image_ids_height"] = int(height)
            self.trace["image_ids_width"] = int(width)
            self.trace["image_ids"] = out
            return out

        def pack_latents(latent_input):
            self.trace["latent_input"] = latent_input
            out = self.orig_pack_latents(latent_input)
            self.trace["packed_latent_input"] = out
            return out

        def unpack_latents(latents, height: int, width: int):
            self.trace["unpack_latents.input"] = latents
            out = self.orig_unpack_latents(latents, height, width)
            self.trace["predicted_flow"] = out
            return out

        def transformer_forward(*args, **kwargs):
            self.trace["transformer_hidden_states"] = kwargs.get("hidden_states")
            self.trace["transformer_timestep"] = kwargs.get("timestep")
            self.trace["encoder_hidden_states"] = kwargs.get("encoder_hidden_states")
            self.trace["text_ids"] = kwargs.get("txt_ids")
            self.trace["image_ids_forward"] = kwargs.get("img_ids")
            self.trace["attention_mask"] = kwargs.get("attention_mask")
            out = self.orig_transformer_forward(*args, **kwargs)
            self.trace["packed_predicted_flow"] = getattr(out, "sample", None)
            return out

        self.model.encode_text = encode_text
        self.setup._create_noise = create_noise
        self.setup._get_timestep_discrete = get_timestep
        self.setup._add_noise_discrete = add_noise
        self.model.prepare_latent_image_ids = prepare_latent_image_ids
        self.model.pack_latents = pack_latents
        self.model.unpack_latents = unpack_latents
        self.model.transformer.forward = transformer_forward
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.setup._create_noise = self.orig_create_noise
        self.setup._get_timestep_discrete = self.orig_get_timestep
        self.setup._add_noise_discrete = self.orig_add_noise
        self.model.encode_text = self.orig_encode_text
        self.model.prepare_latent_image_ids = self.orig_prepare_latent_image_ids
        self.model.pack_latents = self.orig_pack_latents
        self.model.unpack_latents = self.orig_unpack_latents
        self.model.transformer.forward = self.orig_transformer_forward


def enrich_trace(model: Any, batch: dict[str, Any], trace: dict[str, Any]) -> None:
    latent = batch.get("latent_image")
    if torch.is_tensor(latent):
        trace["latent_image_before_scale"] = latent
    vae_config = getattr(getattr(model, "vae", None), "config", {})
    trace["vae_scaling_factor"] = vae_config.get("scaling_factor")
    trace["vae_shift_factor"] = vae_config.get("shift_factor")

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
        "encode_text.tokens",
        "encode_text.tokens_mask",
        "encode_text.cached_hidden_state",
        "text_encoder_output",
        "text_attention_mask",
        "latent_image_before_scale",
        "scaled_latent_image",
        "noise_source_tensor",
        "latent_noise",
        "scaled_noisy_latent_image",
        "sigma",
        "latent_input",
        "packed_latent_input",
        "image_ids",
        "image_ids_forward",
        "text_ids",
        "transformer_hidden_states",
        "transformer_timestep",
        "encoder_hidden_states",
        "attention_mask",
        "packed_predicted_flow",
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
        "vae_scaling_factor": json_safe(trace.get("vae_scaling_factor")),
        "vae_shift_factor": json_safe(trace.get("vae_shift_factor")),
        "image_ids_height": trace.get("image_ids_height"),
        "image_ids_width": trace.get("image_ids_width"),
        "loss_type": model_output_data.get("loss_type"),
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
                with ChromaStepTraceHooks(model_setup, model, prior_trace):
                    prior_model_output_data = model_setup.predict(model, batch, config, train_progress)
            with ChromaStepTraceHooks(model_setup, model, trace):
                model_output_data = model_setup.predict(model, batch, config, train_progress)
            prior_model_prediction = prior_model_output_data["predicted"].to(
                dtype=model_output_data["target"].dtype
            )
            model_output_data["target"][prior_pred_indices] = prior_model_prediction[prior_pred_indices]
            model_output_data["prior_target"] = prior_model_prediction
            trace["prior_trace"] = prior_trace
        else:
            with ChromaStepTraceHooks(model_setup, model, trace):
                model_output_data = model_setup.predict(model, batch, config, train_progress)

        enrich_trace(model, batch, trace)

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

        if adapter_tensors is not None:
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
        "dry_run": False,
        "one_step_dump_produced": True,
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
            "stats": adapter_stats(named_params),
        },
        "reference_files_inspected": REFERENCE_SOURCES,
        "steps": step_results,
    }


def main() -> None:
    args = parse_args()
    enter_serenity_root(args)
    add_serenity_to_path(args.serenity)
    write_contract_artifact(args)

    config = None
    config_error = None
    try:
        config = load_train_config(args)
    except Exception as exc:
        config_error = repr(exc)

    if config is None:
        artifacts = write_dry_run_artifacts(args, config, config_error)
        print(json.dumps({"blocked": True, **artifacts}, indent=2))
        if not args.dry_run:
            raise SystemExit(2)
        return

    if args.dry_run:
        artifacts = write_dry_run_artifacts(args, config, config_error)
        print(json.dumps({"blocked": True, **artifacts}, indent=2))
        return

    validate_config(config, args, check_runtime_blockers=True)

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
