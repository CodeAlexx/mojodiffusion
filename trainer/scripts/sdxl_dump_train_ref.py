#!/usr/bin/env python3
"""
Dump real Serenity SDXL train-step reference data for Mojo parity gates.

This script imports Serenity by path and, outside dry-run mode, uses its real
SDXL model loader, setup, cached-data loader, predict, loss, backward, clipping,
optimizer, and LR scheduler paths for one bounded train step.

Run with the Serenity venv, for example:
  /home/alex/Serenity/venv/bin/python scripts/sdxl_dump_train_ref.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

from qwen_dump_train_ref import (
    add_batch,
    add_tensor,
    create_scheduler,
    enum_value,
    global_grad_norm,
    json_safe,
    optimizer_meta,
    tensor_stats,
    write_safetensors,
)


DEFAULT_SERENITY = Path("/home/alex/Serenity")
DEFAULT_CONFIG = DEFAULT_SERENITY / "configs" / "sdxl_100step_baseline.json"
DEFAULT_OUT_DIR = Path("/home/alex/serenity-trainer/parity")
DEFAULT_PREFIX = "sdxl_train_ref"
PRODUCER = "scripts/sdxl_dump_train_ref.py"
BLOCKER_SCHEMA_VERSION = 1
REFERENCE_POLICY = {
    "reference": "Serenity only",
    "dry_run_numeric_parity": False,
    "dry_run_note": (
        "Dry-run/blocker artifacts are structural evidence only; they do not "
        "load SDXL weights, run a train step, or claim numeric parity."
    ),
    "non_dry_run_device": "cuda",
    "cpu_pytorch_numeric_parity": False,
}
TRAINING_METHOD_SCOPE = {
    "LORA": "in scope when Serenity registers loader/setup/data-loader paths",
    "FINE_TUNE": "in scope when Serenity registers loader/setup/data-loader paths",
    "EMBEDDING": "not in scope for this SDXL train-reference scaffold",
    "FINE_TUNE_VAE": "not in scope for this SDXL train-reference scaffold",
}
FULL_FINETUNE_GATE = {
    "scope": "separate full-weight gate",
    "adapter_safetensors": "LoRA-only; never full-finetune weights",
    "note": (
        "Serenity SDXL FINE_TUNE registrations are tracked here as train-reference "
        "scope, but full-weight save/load/resume parity is a separate gate."
    ),
}
REFERENCE_SOURCES = [
    "/home/alex/Serenity/modules/modelLoader/StableDiffusionXLLoRAModelLoader.py",
    "/home/alex/Serenity/modules/modelLoader/StableDiffusionXLFineTuneModelLoader.py",
    "/home/alex/Serenity/modules/modelLoader/GenericFineTuneModelLoader.py",
    "/home/alex/Serenity/modules/modelSetup/BaseStableDiffusionXLSetup.py",
    "/home/alex/Serenity/modules/modelSetup/StableDiffusionXLLoRASetup.py",
    "/home/alex/Serenity/modules/modelSetup/StableDiffusionXLFineTuneSetup.py",
    "/home/alex/Serenity/modules/dataLoader/StableDiffusionXLBaseDataLoader.py",
    "/home/alex/Serenity/modules/modelSaver/StableDiffusionXLLoRAModelSaver.py",
    "/home/alex/Serenity/modules/modelSaver/StableDiffusionXLFineTuneModelSaver.py",
    "/home/alex/Serenity/modules/modelSaver/GenericFineTuneModelSaver.py",
    "/home/alex/Serenity/training_presets/#sdxl 1.0 LoRA.json",
    "/home/alex/Serenity/training_presets/#sdxl 1.0.json",
    "/home/alex/Serenity/configs/sdxl_100step_baseline.json",
]


def cuda_probe_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "available": False,
        "device_count": 0,
        "error": None,
        "warnings": [],
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            status["available"] = bool(torch.cuda.is_available())
        except Exception as exc:
            status["error"] = f"torch.cuda.is_available failed: {exc}"

        try:
            status["device_count"] = int(torch.cuda.device_count())
        except Exception as exc:
            if status["error"] is None:
                status["error"] = f"torch.cuda.device_count failed: {exc}"

    status["warnings"] = sorted({str(warning.message) for warning in caught})

    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump one bounded Serenity SDXL train-step parity ref."
    )
    parser.add_argument("--serenity", type=Path, default=DEFAULT_SERENITY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--secrets", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--baseline-steps", type=int, default=100)
    parser.add_argument("--train-device", default=None)
    parser.add_argument("--temp-device", default=None)
    parser.add_argument(
        "--adapter-dump",
        choices=("none", "initial", "step", "step-with-grads"),
        default="step",
        help=(
            "Adapter tensor dump mode. 'step' writes trainable SDXL LoRA params "
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
        help="Optional process RNG seed before Serenity setup. Default preserves Serenity's process RNG behavior.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check imports, config, baseline, CUDA, cache, and local model blockers without loading the model.",
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


def _bounded_file_count(path: Path, limit: int = 1000) -> int:
    if not path.is_dir():
        return 0
    count = 0
    for child in path.rglob("*"):
        if child.is_file():
            count += 1
            if count >= limit:
                return count
    return count


def _text_cache_required(config: Any) -> bool:
    return (
        not config.train_text_encoder_or_embedding()
        or not config.train_text_encoder_2_or_embedding()
    )


def cache_status(config: Any) -> dict[str, Any]:
    cache_dir = Path(config.cache_dir)
    image_dir = cache_dir / "image"
    text_dir = cache_dir / "text"
    image_files = _bounded_file_count(image_dir)
    text_files = _bounded_file_count(text_dir)
    text_required = _text_cache_required(config)
    present = True
    if config.latent_caching:
        present = image_files > 0 and (text_files > 0 if text_required else True)

    return {
        "enabled": bool(config.latent_caching),
        "cache_dir": str(cache_dir),
        "present": present,
        "text_cache_required": text_required,
        "image": {
            "path": str(image_dir),
            "exists": image_dir.is_dir(),
            "file_count_bounded": image_files,
        },
        "text": {
            "path": str(text_dir),
            "exists": text_dir.is_dir(),
            "file_count_bounded": text_files,
        },
    }


def baseline_status(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    destination = Path(config.output_model_destination or "")
    if destination and not destination.is_absolute():
        destination = Path.cwd() / destination

    output_dir = destination.parent if str(destination) else Path("")
    metrics_path = output_dir / "metrics.json" if str(output_dir) else Path("")
    metrics: dict[str, Any] | None = None
    metrics_error = None

    if metrics_path and metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            metrics_error = str(exc)

    requested_steps = int(metrics.get("requested_steps", 0)) if isinstance(metrics, dict) else 0
    global_steps_seen = int(metrics.get("global_steps_seen", 0)) if isinstance(metrics, dict) else 0
    status = metrics.get("status") if isinstance(metrics, dict) else None
    model_exists = destination.is_file() if str(destination) else False
    metrics_complete = (
        isinstance(metrics, dict)
        and status == "completed"
        and requested_steps >= args.baseline_steps
        and global_steps_seen >= args.baseline_steps
    )

    return {
        "required_steps": args.baseline_steps,
        "output_model_destination": str(destination),
        "output_model_exists": model_exists,
        "output_model_size_bytes": destination.stat().st_size if model_exists else 0,
        "metrics_path": str(metrics_path),
        "metrics_exists": metrics_path.is_file() if str(metrics_path) else False,
        "metrics_error": metrics_error,
        "metrics_status": status,
        "requested_steps": requested_steps,
        "global_steps_seen": global_steps_seen,
        "complete": bool(model_exists and metrics_complete),
    }


def _safe_safetensors_key_summary(path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except Exception as exc:
        return {"inspectable": False, "error": f"safetensors import failed: {exc}"}

    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    except Exception as exc:
        return {"inspectable": False, "error": f"safetensors open failed: {exc}"}

    lower_keys = [key.lower() for key in keys]

    def count_if(predicate) -> int:
        return sum(1 for key in lower_keys if predicate(key))

    unet_count = count_if(
        lambda key: "model.diffusion_model" in key or key.startswith("unet.")
    )
    vae_count = count_if(
        lambda key: "first_stage_model" in key or key.startswith("vae.")
    )
    text_1_count = count_if(
        lambda key: (
            "conditioner.embedders.0" in key
            or key.startswith("text_encoder.")
            or "clip_l" in key
        )
    )
    text_2_count = count_if(
        lambda key: (
            "conditioner.embedders.1" in key
            or key.startswith("text_encoder_2.")
            or "clip_g" in key
        )
    )

    return {
        "inspectable": True,
        "key_count": len(keys),
        "unet_key_count": unet_count,
        "vae_key_count": vae_count,
        "text_encoder_1_key_count": text_1_count,
        "text_encoder_2_key_count": text_2_count,
        "text_encoder_key_count": text_1_count + text_2_count,
        "sample_keys": keys[:16],
    }


def local_model_status(config: Any) -> dict[str, Any]:
    names = config.model_names()
    base = Path(os.path.expanduser(names.base_model)) if names.base_model else Path("")
    include = {
        "text_encoder_1": bool(names.include_text_encoder),
        "text_encoder_2": bool(names.include_text_encoder_2),
    }

    status: dict[str, Any] = {
        "base_model": names.base_model,
        "vae_model": names.vae_model,
        "lora_model": names.lora,
        "include_text_encoders": include,
        "exists": base.exists() if names.base_model else False,
    }

    if names.base_model and base.is_file():
        status["kind"] = "single_file"
        status["suffix"] = base.suffix
        status["size_bytes"] = base.stat().st_size
        if base.suffix == ".safetensors":
            status["safetensors"] = _safe_safetensors_key_summary(base)
    elif names.base_model and base.is_dir():
        status["kind"] = "diffusers_dir"
        status["has_model_index_json"] = (base / "model_index.json").is_file()
        status["subfolders"] = {
            "tokenizer": (base / "tokenizer").is_dir(),
            "tokenizer_2": (base / "tokenizer_2").is_dir(),
            "text_encoder": (base / "text_encoder").is_dir(),
            "text_encoder_2": (base / "text_encoder_2").is_dir(),
            "unet": (base / "unet").is_dir(),
            "vae": (base / "vae").is_dir(),
        }
    else:
        status["kind"] = "remote_or_missing"

    return status


def serenity_registration_status(config: Any) -> dict[str, Any]:
    from modules.dataLoader.BaseDataLoader import BaseDataLoader
    from modules.modelSaver.BaseModelSaver import BaseModelSaver
    from modules.util import create, factory
    from modules.util.enum.TrainingMethod import TrainingMethod

    sources = {
        "LORA": {
            "model_loader": [
                "/home/alex/Serenity/modules/modelLoader/StableDiffusionXLLoRAModelLoader.py",
            ],
            "model_setup": [
                "/home/alex/Serenity/modules/modelSetup/StableDiffusionXLLoRASetup.py",
            ],
            "data_loader": [
                "/home/alex/Serenity/modules/dataLoader/StableDiffusionXLBaseDataLoader.py",
            ],
            "model_saver": [
                "/home/alex/Serenity/modules/modelSaver/StableDiffusionXLLoRAModelSaver.py",
            ],
        },
        "FINE_TUNE": {
            "model_loader": [
                "/home/alex/Serenity/modules/modelLoader/StableDiffusionXLFineTuneModelLoader.py",
                "/home/alex/Serenity/modules/modelLoader/GenericFineTuneModelLoader.py",
            ],
            "model_setup": [
                "/home/alex/Serenity/modules/modelSetup/StableDiffusionXLFineTuneSetup.py",
            ],
            "data_loader": [
                "/home/alex/Serenity/modules/dataLoader/StableDiffusionXLBaseDataLoader.py",
            ],
            "model_saver": [
                "/home/alex/Serenity/modules/modelSaver/StableDiffusionXLFineTuneModelSaver.py",
                "/home/alex/Serenity/modules/modelSaver/GenericFineTuneModelSaver.py",
            ],
        },
    }

    status: dict[str, Any] = {}
    for method in (TrainingMethod.LORA, TrainingMethod.FINE_TUNE):
        method_name = enum_value(method)
        setup = create.create_model_setup(
            config.model_type,
            torch.device(config.train_device),
            torch.device(config.temp_device),
            method,
            config.debug_mode,
        )
        data_loader_method = factory.get(BaseDataLoader, config.model_type, method)
        data_loader_model = factory.get(BaseDataLoader, config.model_type)
        model_saver = factory.get(BaseModelSaver, config.model_type, method)
        status[method_name] = {
            "in_scope": True,
            "model_type": enum_value(config.model_type),
            "model_loader_registered": create.create_model_loader(config.model_type, method) is not None,
            "model_setup_registered": setup is not None,
            "data_loader_registered": data_loader_method is not None or data_loader_model is not None,
            "data_loader_registration": (
                "model_and_method" if data_loader_method is not None
                else "model_only" if data_loader_model is not None
                else None
            ),
            "model_saver_registered": model_saver is not None,
            "source_paths": sources[method_name],
        }
        if method == TrainingMethod.FINE_TUNE:
            status[method_name]["full_weight_gate"] = FULL_FINETUNE_GATE

    return status


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


def structured_reference_blockers(config: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    from modules.dataLoader.BaseDataLoader import BaseDataLoader
    from modules.util import create, factory

    cuda_status = cuda_probe_status()
    if str(config.train_device).split(":", 1)[0] != "cuda":
        blockers.append({
            "id": "cuda_train_device_not_configured",
            "category": "cuda_reference",
            "message": "Numeric SDXL references must use Serenity's CUDA path; CPU dry-runs are structural only.",
            "details": {
                "train_device": config.train_device,
                "temp_device": config.temp_device,
                "required": "--train-device cuda --temp-device cpu or an equivalent CUDA config",
            },
        })
    if not cuda_status["available"]:
        blockers.append({
            "id": "cuda_unavailable_in_current_process",
            "category": "cuda_reference",
            "message": "Current process does not report CUDA; SDXL dry-run remains structural only.",
            "details": cuda_status,
        })

    if create.create_model_loader(config.model_type, config.training_method) is None:
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
    if create.create_model_setup(
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
    if factory.get(BaseDataLoader, config.model_type, config.training_method) is None \
            and factory.get(BaseDataLoader, config.model_type) is None:
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
    if not baseline["complete"]:
        blockers.append({
            "id": "missing_100step_baseline",
            "category": "baseline",
            "message": (
                f"Required {args.baseline_steps}-step SDXL baseline output is missing "
                "or incomplete."
            ),
            "details": baseline,
        })

    cache = cache_status(config)
    if config.latent_caching and not cache["present"]:
        blockers.append({
            "id": "missing_required_cache",
            "category": "cache",
            "message": (
                "Required SDXL latent/text cache is missing or empty "
                f"(image files={cache['image']['file_count_bounded']}, "
                f"text files={cache['text']['file_count_bounded']}, "
                f"text required={cache['text_cache_required']})."
            ),
            "details": cache,
        })

    model = local_model_status(config)
    base_model = model.get("base_model") or ""
    if base_model and not model.get("exists") and _is_local_model_ref(base_model):
        blockers.append({
            "id": "missing_base_model_path",
            "category": "weights",
            "message": f"Base model path does not exist: {base_model}",
            "details": {
                "base_model": base_model,
                "kind": model.get("kind"),
            },
        })

    if model.get("kind") == "single_file":
        st = model.get("safetensors", {})
        unet_keys = int(st.get("unet_key_count", 0)) if st.get("inspectable") else 0
        vae_keys = int(st.get("vae_key_count", 0)) if st.get("inspectable") else 0
        text_1_keys = int(st.get("text_encoder_1_key_count", 0)) if st.get("inspectable") else 0
        text_2_keys = int(st.get("text_encoder_2_key_count", 0)) if st.get("inspectable") else 0
        missing_parts = []
        if unet_keys == 0:
            missing_parts.append("unet")
        if vae_keys == 0:
            missing_parts.append("vae")
        if model["include_text_encoders"]["text_encoder_1"] and text_1_keys == 0:
            missing_parts.append("text_encoder_1")
        if model["include_text_encoders"]["text_encoder_2"] and text_2_keys == 0:
            missing_parts.append("text_encoder_2")
        if missing_parts:
            blockers.append({
                "id": "missing_single_file_sdxl_keys",
                "category": "weights",
                "message": (
                    "Single-file SDXL checkpoint is missing inspectable model component keys: "
                    + ", ".join(missing_parts)
                ),
                "details": {
                    "base_model": base_model,
                    "missing_parts": missing_parts,
                    "safetensors": st,
                    "include_text_encoders": model.get("include_text_encoders"),
                },
            })
        if args.allow_cache_build and cache["text_cache_required"] and (text_1_keys == 0 or text_2_keys == 0):
            blockers.append({
                "id": "cache_build_missing_text_encoder_weights",
                "category": "cache",
                "message": (
                    "--allow-cache-build cannot create a valid SDXL text cache from "
                    "this checkpoint because local text encoder weights are absent."
                ),
                "details": {
                    "base_model": base_model,
                    "allow_cache_build": args.allow_cache_build,
                    "safetensors": st,
                },
            })
    elif model.get("kind") == "diffusers_dir":
        subfolders = model.get("subfolders", {})
        required = ["unet", "vae"]
        if model["include_text_encoders"]["text_encoder_1"]:
            required += ["tokenizer", "text_encoder"]
        if model["include_text_encoders"]["text_encoder_2"]:
            required += ["tokenizer_2", "text_encoder_2"]
        missing = [name for name in required if not subfolders.get(name, False)]
        if missing:
            blockers.append({
                "id": "missing_diffusers_sdxl_folders",
                "category": "weights",
                "message": (
                    "Full local SDXL diffusers snapshot is missing required folders: "
                    + ", ".join(missing)
                ),
                "details": {
                    "base_model": base_model,
                    "missing_folders": missing,
                    "subfolders": subfolders,
                    "text_cache": cache["text"],
                },
            })

    return blockers


def reference_blockers(config: Any, args: argparse.Namespace) -> list[str]:
    return [str(blocker["message"]) for blocker in structured_reference_blockers(config, args)]


def dry_run_report(args: argparse.Namespace, config: Any) -> dict[str, Any]:
    names = config.model_names()
    structured_blockers = structured_reference_blockers(config, args)
    cuda_status = cuda_probe_status()
    return {
        "schema_version": BLOCKER_SCHEMA_VERSION,
        "producer": PRODUCER,
        "created_unix": time.time(),
        "reference_policy": REFERENCE_POLICY,
        "dry_run": True,
        "one_step_dump_produced": False,
        "numeric_parity_status": "none",
        "numeric_parity_claimed": False,
        "dry_run_checks_are_structural_only": True,
        "numeric_parity_note": (
            "CPU PyTorch and dry-run artifacts are structural only. SDXL loss/grad/speed "
            "parity requires Serenity CUDA reference tensors."
        ),
        "training_method_scope": TRAINING_METHOD_SCOPE,
        "full_finetune_gate": FULL_FINETUNE_GATE,
        "serenity_registration_status": serenity_registration_status(config),
        "serenity": str(args.serenity.resolve()),
        "config": str(args.config.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "prefix": args.prefix,
        "max_steps": args.max_steps,
        "baseline_steps": args.baseline_steps,
        "torch": torch.__version__,
        "cuda_available": cuda_status["available"],
        "cuda_device_count": cuda_status["device_count"],
        "cuda_probe": cuda_status,
        "train_device": config.train_device,
        "temp_device": config.temp_device,
        "model_type": enum_value(config.model_type),
        "training_method": enum_value(config.training_method),
        "model_names": {
            "base_model": names.base_model,
            "vae_model": names.vae_model,
            "lora_model": names.lora,
            "include_text_encoder": names.include_text_encoder,
            "include_text_encoder_2": names.include_text_encoder_2,
        },
        "baseline": baseline_status(config, args),
        "cache": cache_status(config),
        "allow_cache_build": args.allow_cache_build,
        "local_model": local_model_status(config),
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
        "reference_files_inspected": REFERENCE_SOURCES,
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

    if config.model_type not in (
        ModelType.STABLE_DIFFUSION_XL_10_BASE,
        ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING,
    ):
        raise ValueError(f"Expected SDXL config, got {config.model_type}")
    if config.training_method not in (TrainingMethod.LORA, TrainingMethod.FINE_TUNE):
        raise ValueError(f"Expected LORA or FINE_TUNE training method, got {config.training_method}")
    if args.max_steps != 1:
        raise ValueError("--max-steps must be exactly 1 for this bounded SDXL dump")
    if args.baseline_steps <= 0:
        raise ValueError("--baseline-steps must be positive")
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
        cuda_status = cuda_probe_status()
        if not cuda_status["available"]:
            raise RuntimeError(
                "CUDA is not available. Run --dry-run for structural checks only; "
                "do not produce CPU numeric reference dumps. "
                f"CUDA probe: {cuda_status}"
            )

    if check_runtime_blockers:
        blockers = reference_blockers(config, args)
        if blockers:
            raise RuntimeError("SDXL reference dump blockers:\n- " + "\n- ".join(blockers))


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


def named_trainable_parameters(model: Any, parameters: list[torch.nn.Parameter]) -> list[tuple[str, torch.nn.Parameter]]:
    id_to_name: dict[int, str] = {}
    for wrapper_name in ("text_encoder_1_lora", "text_encoder_2_lora", "unet_lora"):
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


class SDXLStepTraceHooks:
    def __init__(self, setup: Any, model: Any, trace: dict[str, Any]):
        self.setup = setup
        self.model = model
        self.trace = trace
        self.orig_create_noise = None
        self.orig_get_timestep = None
        self.orig_add_noise = None
        self.orig_encode_text = None
        self.orig_combine_text = None
        self.orig_unet_forward = None

    def __enter__(self) -> "SDXLStepTraceHooks":
        self.orig_create_noise = self.setup._create_noise
        self.orig_get_timestep = self.setup._get_timestep_discrete
        self.orig_add_noise = self.setup._add_noise_discrete
        self.orig_encode_text = self.model.encode_text
        self.orig_combine_text = self.model.combine_text_encoder_output
        self.orig_unet_forward = self.model.unet.forward

        def encode_text(*args, **kwargs):
            for key in (
                "tokens_1",
                "tokens_2",
                "text_encoder_1_output",
                "text_encoder_2_output",
                "pooled_text_encoder_2_output",
            ):
                self.trace[f"encode_text.{key}"] = kwargs.get(key)
            out = self.orig_encode_text(*args, **kwargs)
            (
                self.trace["text_encoder_1_output"],
                self.trace["text_encoder_2_output"],
                self.trace["pooled_text_encoder_2_output"],
            ) = out
            return out

        def combine_text_encoder_output(*args, **kwargs):
            out = self.orig_combine_text(*args, **kwargs)
            self.trace["combined_text_encoder_output"] = out[0]
            self.trace["combined_pooled_text_encoder_2_output"] = out[1]
            return out

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

        def add_noise(scaled_latent_image, latent_noise, timestep, betas):
            out = self.orig_add_noise(scaled_latent_image, latent_noise, timestep, betas)
            self.trace["scaled_noisy_latent_image"] = out
            return out

        def unet_forward(*args, **kwargs):
            added_cond = kwargs.get("added_cond_kwargs") or {}
            self.trace["latent_input"] = kwargs.get("sample", args[0] if args else None)
            self.trace["unet_timestep"] = kwargs.get("timestep")
            self.trace["encoder_hidden_states"] = kwargs.get("encoder_hidden_states")
            self.trace["added_cond_text_embeds"] = added_cond.get("text_embeds")
            self.trace["added_cond_time_ids"] = added_cond.get("time_ids")
            out = self.orig_unet_forward(*args, **kwargs)
            self.trace["predicted_latent_noise"] = getattr(out, "sample", None)
            return out

        self.model.encode_text = encode_text
        self.model.combine_text_encoder_output = combine_text_encoder_output
        self.setup._create_noise = create_noise
        self.setup._get_timestep_discrete = get_timestep
        self.setup._add_noise_discrete = add_noise
        self.model.unet.forward = unet_forward
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.setup._create_noise = self.orig_create_noise
        self.setup._get_timestep_discrete = self.orig_get_timestep
        self.setup._add_noise_discrete = self.orig_add_noise
        self.model.encode_text = self.orig_encode_text
        self.model.combine_text_encoder_output = self.orig_combine_text
        self.model.unet.forward = self.orig_unet_forward


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
        "encode_text.tokens_1",
        "encode_text.tokens_2",
        "encode_text.text_encoder_1_output",
        "encode_text.text_encoder_2_output",
        "encode_text.pooled_text_encoder_2_output",
        "text_encoder_1_output",
        "text_encoder_2_output",
        "pooled_text_encoder_2_output",
        "combined_text_encoder_output",
        "combined_pooled_text_encoder_2_output",
        "scaled_latent_image",
        "latent_noise",
        "scaled_noisy_latent_image",
        "latent_input",
        "unet_timestep",
        "encoder_hidden_states",
        "added_cond_text_embeds",
        "added_cond_time_ids",
        "predicted_latent_noise",
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
        "prediction_type": model_output_data.get("prediction_type"),
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
    is_lora_training = config.training_method == TrainingMethod.LORA
    named_params = named_trainable_parameters(model, parameters) if is_lora_training else []

    initial_adapter_path = None
    if is_lora_training and args.adapter_dump == "initial" and named_params:
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
        if is_lora_training and args.adapter_dump in ("step", "step-with-grads") and named_params:
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
                with SDXLStepTraceHooks(model_setup, model, prior_trace):
                    prior_model_output_data = model_setup.predict(model, batch, config, train_progress)
            with SDXLStepTraceHooks(model_setup, model, trace):
                model_output_data = model_setup.predict(model, batch, config, train_progress)
            prior_model_prediction = prior_model_output_data["predicted"].to(
                dtype=model_output_data["target"].dtype
            )
            model_output_data["target"][prior_pred_indices] = prior_model_prediction[prior_pred_indices]
            model_output_data["prior_target"] = prior_model_prediction
            trace["prior_trace"] = prior_trace
        else:
            with SDXLStepTraceHooks(model_setup, model, trace):
                model_output_data = model_setup.predict(model, batch, config, train_progress)

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
        "baseline": baseline_status(config, args),
        "adapter_dump": args.adapter_dump,
        "adapter_dump_policy": FULL_FINETUNE_GATE["adapter_safetensors"],
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
            "prediction_type": None,
            "timestep_distribution": enum_value(config.timestep_distribution),
            "dynamic_timestep_shifting": config.dynamic_timestep_shifting,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_weight_dtype": enum_value(config.lora_weight_dtype),
            "layer_filter": config.layer_filter,
            "layer_filter_preset": config.layer_filter_preset,
        },
        "local_model": local_model_status(config),
        "cache": cache_status(config),
        "trainable_parameters": {
            "count": len(parameters),
            "numel": int(sum(param.numel() for param in parameters)),
            "adapter_dump_enabled": bool(is_lora_training and args.adapter_dump != "none"),
            "names": [name for name, _ in named_params],
            "adapter_dump_count": len(named_params),
            "adapter_dump_numel": int(sum(param.numel() for _, param in named_params)),
            "stats": adapter_stats(named_params),
            "full_finetune_gate_required": not is_lora_training,
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
