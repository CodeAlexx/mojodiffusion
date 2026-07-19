#!/usr/bin/env python3
"""
Flux2 dev train-reference scaffold for Serenity-to-Mojo parity work.

Dry-run mode writes a contract and blocker report only. It does not load model
weights, run a CPU train step, or produce numeric parity evidence. Numeric parity
for Flux2 dev starts only after Serenity CUDA reference tensors exist:

  parity/flux2_dev_train_ref_meta.json
  parity/flux2_dev_train_ref_step000.safetensors
  parity/flux2_dev_train_ref_step000_adapters.safetensors

The reference source for model behavior is only /home/alex/Serenity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

import klein_dump_train_ref as klein


DEFAULT_SERENITY = Path("/home/alex/Serenity")
DEFAULT_CONFIG = DEFAULT_SERENITY / "configs" / "flux2_dev_100step_baseline.json"
DEFAULT_OUT_DIR = Path("/home/alex/serenity-trainer/parity")
DEFAULT_PREFIX = "flux2_dev_train_ref"
DEFAULT_DEV_TRANSFORMER_CONFIG = Path(
    "/home/alex/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/"
    "snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c/transformer/config.json"
)
PRODUCER = "scripts/flux2_dev_dump_train_ref.py"
BLOCKER_SCHEMA_VERSION = 1

NO_NUMERIC_PARITY_NOTE = (
    "Dry-run/config checks are structural only. There is no Flux2 dev numeric "
    "parity until Serenity CUDA reference tensors exist."
)

REFERENCE_SOURCES = [
    "/home/alex/Serenity/modules/model/Flux2Model.py",
    "/home/alex/Serenity/modules/modelLoader/Flux2ModelLoader.py",
    "/home/alex/Serenity/modules/dataLoader/Flux2BaseDataLoader.py",
    "/home/alex/Serenity/modules/modelSetup/BaseFlux2Setup.py",
    "/home/alex/Serenity/modules/modelSetup/Flux2LoRASetup.py",
    "/home/alex/Serenity/modules/modelSetup/Flux2FineTuneSetup.py",
    "/home/alex/Serenity/modules/modelSampler/Flux2Sampler.py",
    "/home/alex/Serenity/modules/modelSaver/Flux2LoRAModelSaver.py",
    "/home/alex/Serenity/modules/modelSaver/Flux2FineTuneModelSaver.py",
]

PRESET_CANDIDATES = [
    "/home/alex/Serenity/training_presets/#flux2 LoRA 8GB.json",
    "/home/alex/Serenity/training_presets/#flux2 LoRA 16GB.json",
    "/home/alex/Serenity/training_presets/#flux2 Finetune 16GB.json",
    "/home/alex/Serenity/training_presets/#flux2 Finetune 24GB.json",
]

REQUIRED_TENSOR_FILES = [
    "flux2_dev_train_ref_meta.json",
    "flux2_dev_train_ref_step000.safetensors",
    "flux2_dev_train_ref_step000_adapters.safetensors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write Flux2 dev train-reference contract/blocker evidence."
    )
    parser.add_argument("--serenity", type=Path, default=DEFAULT_SERENITY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--secrets", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--train-device", default=None)
    parser.add_argument("--temp-device", default=None)
    parser.add_argument("--allow-cache-build", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write structural contract/blocker JSON without loading Flux2 dev weights.",
    )
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> None:
    args.serenity = args.serenity.resolve()
    args.config = args.config.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.secrets is not None:
        args.secrets = args.secrets.resolve()


def add_serenity_to_path(root: Path) -> None:
    root = root.resolve()
    if not (root / "modules").is_dir():
        raise FileNotFoundError(f"Serenity root does not contain modules/: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(klein.json_safe(data), indent=2) + "\n", encoding="utf-8")


def read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bounded_file_count(path: Path, limit: int = 1000) -> int:
    if not path.is_dir():
        return 0
    count = 0
    for child in path.rglob("*"):
        if child.is_file():
            count += 1
            if count >= limit:
                return count
    return count


def cache_status(config: Any | None) -> dict[str, Any] | None:
    if config is None:
        return None

    cache_dir = Path(config.cache_dir)
    image_dir = cache_dir / "image"
    text_dir = cache_dir / "text"
    present = klein.cache_present(config)
    return {
        "enabled": bool(config.latent_caching),
        "cache_dir": str(cache_dir),
        "present": present,
        "image": {
            "path": str(image_dir),
            "exists": image_dir.is_dir(),
            "file_count_bounded": bounded_file_count(image_dir),
        },
        "text": {
            "path": str(text_dir),
            "exists": text_dir.is_dir(),
            "file_count_bounded": bounded_file_count(text_dir),
        },
    }


def file_status(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        status["size_bytes"] = path.stat().st_size
    return status


def reference_tensor_status(args: argparse.Namespace) -> dict[str, Any]:
    files = {
        name: file_status(args.out_dir / name)
        for name in REQUIRED_TENSOR_FILES
    }
    return {
        "required_for_numeric_parity": REQUIRED_TENSOR_FILES,
        "files": files,
        "complete": all(item["exists"] for item in files.values()),
        "numeric_parity_status": (
            "reference_tensors_present_not_validated_by_this_dry_run"
            if all(item["exists"] for item in files.values())
            else "none"
        ),
        "note": NO_NUMERIC_PARITY_NOTE,
    }


def transformer_config_status(base_model_name: str, transformer_model_name: str | None) -> dict[str, Any]:
    status: dict[str, Any] = {
        "base_model_name": base_model_name,
        "transformer_model_name": transformer_model_name or "",
        "source": None,
        "config_exists": False,
        "num_attention_heads": None,
        "structural_dev_hint": False,
    }

    candidates: list[Path] = []
    if transformer_model_name:
        transformer_path = Path(os.path.expanduser(transformer_model_name))
        if transformer_path.is_dir():
            candidates.append(transformer_path / "config.json")
    base_path = Path(os.path.expanduser(base_model_name))
    if base_path.is_dir():
        candidates.append(base_path / "transformer" / "config.json")

    for candidate in candidates:
        data = read_json_if_present(candidate)
        if data is None:
            continue
        status["source"] = str(candidate)
        status["config_exists"] = True
        status["num_attention_heads"] = data.get("num_attention_heads")
        status["structural_dev_hint"] = data.get("num_attention_heads") == 48
        break

    return status


def local_dev_transformer_config_status() -> dict[str, Any]:
    data = read_json_if_present(DEFAULT_DEV_TRANSFORMER_CONFIG)
    return {
        "path": str(DEFAULT_DEV_TRANSFORMER_CONFIG),
        "exists": data is not None,
        "num_attention_heads": None if data is None else data.get("num_attention_heads"),
        "num_layers": None if data is None else data.get("num_layers"),
        "num_single_layers": None if data is None else data.get("num_single_layers"),
        "joint_attention_dim": None if data is None else data.get("joint_attention_dim"),
        "serenity_trainer_is_dev_condition": "transformer.config.num_attention_heads == 48",
        "structural_dev_hint": False if data is None else data.get("num_attention_heads") == 48,
        "scope": (
            "partial transformer config only; no model_index, scheduler, VAE, "
            "tokenizer, text_encoder, cache, train config, baseline, or numeric parity"
        ),
    }


def model_status(config: Any | None) -> dict[str, Any] | None:
    if config is None:
        return None

    names = config.model_names()
    base_model_name = names.base_model or config.base_model_name
    transformer_model_name = names.transformer_model
    vae_model_name = names.vae_model
    base_path = Path(os.path.expanduser(base_model_name)) if base_model_name else Path("")

    status: dict[str, Any] = {
        "model_names": {
            "base_model": base_model_name,
            "transformer_model": transformer_model_name,
            "vae_model": vae_model_name,
            "lora_model": names.lora,
            "include_text_encoder": names.include_text_encoder,
        },
        "base_model_exists": base_path.exists() if base_model_name else False,
        "base_model_is_local_path": (
            base_path.is_absolute()
            or str(base_model_name).startswith(".")
            or str(base_model_name).startswith("~")
        ) if base_model_name else False,
        "base_model_name_mentions_klein": "klein" in str(base_model_name).lower(),
        "transformer_config": transformer_config_status(base_model_name, transformer_model_name),
    }
    if base_path.is_dir():
        status["diffusers_subfolders"] = {
            "tokenizer": (base_path / "tokenizer").is_dir(),
            "text_encoder": (base_path / "text_encoder").is_dir(),
            "transformer": (base_path / "transformer").is_dir(),
            "scheduler": (base_path / "scheduler").is_dir(),
            "vae": (base_path / "vae").is_dir(),
            "meta_json": (base_path / "meta.json").is_file(),
        }
    return status


def load_config_for_dry_run(args: argparse.Namespace) -> tuple[Any | None, str | None]:
    if not args.config.is_file():
        return None, None

    try:
        add_serenity_to_path(args.serenity)
        os.chdir(args.serenity)
        return klein.load_train_config(args), None
    except Exception as exc:
        return None, f"{exc.__class__.__name__}: {exc}"


def validate_config_structural(config: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    from modules.util.enum.ModelType import ModelType
    from modules.util.enum.TrainingMethod import TrainingMethod

    if config.model_type != ModelType.FLUX_2:
        blockers.append({
            "id": "wrong_model_type",
            "category": "config",
            "message": "Flux2 dev references must use Serenity ModelType.FLUX_2.",
            "details": {"actual": klein.enum_value(config.model_type)},
        })
    if config.training_method != TrainingMethod.LORA:
        blockers.append({
            "id": "wrong_training_method",
            "category": "config",
            "message": "This scaffold is for Flux2 dev LoRA train-reference tensors.",
            "details": {"actual": klein.enum_value(config.training_method)},
        })
    if args.max_steps != 1:
        blockers.append({
            "id": "unbounded_step_count",
            "category": "config",
            "message": "--max-steps must be exactly 1 for the bounded reference contract.",
            "details": {"max_steps": args.max_steps},
        })
    if config.optimizer.fused_back_pass:
        blockers.append({
            "id": "unsupported_fused_back_pass",
            "category": "optimizer",
            "message": "fused_back_pass is not supported by the bounded reference loop.",
            "details": {},
        })
    if config.optimizer.optimizer.is_schedule_free:
        blockers.append({
            "id": "unsupported_schedule_free_optimizer",
            "category": "optimizer",
            "message": "Schedule-free optimizer mode is not supported by the dump loop.",
            "details": {"optimizer": klein.enum_value(config.optimizer.optimizer)},
        })
    return blockers


def structured_blockers(
    args: argparse.Namespace,
    config: Any | None,
    config_error: str | None,
    local_model: dict[str, Any] | None,
    cache: dict[str, Any] | None,
    ref_tensors: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []

    if not args.config.is_file():
        blockers.append({
            "id": "missing_flux2_dev_reference_config",
            "category": "config",
            "message": "No Flux2 dev Serenity train config exists at the default path.",
            "details": {
                "path": str(args.config),
                "expected_action": "Create a real Serenity Flux2 dev CUDA config before numeric parity.",
            },
        })
    if config_error is not None:
        blockers.append({
            "id": "config_load_error",
            "category": "config",
            "message": "The Serenity config could not be loaded.",
            "details": {"error": config_error},
        })
    if config is not None:
        blockers.extend(validate_config_structural(config, args))

        if cache is not None and cache["enabled"] and not cache["present"] and not args.allow_cache_build:
            blockers.append({
                "id": "missing_required_cache",
                "category": "cache",
                "message": "Configured cached latent/text data is missing or empty.",
                "details": cache,
            })

        train_device = str(config.train_device)
        if train_device != "cuda":
            blockers.append({
                "id": "cuda_train_device_not_configured",
                "category": "cuda_reference",
                "message": "Numeric tensors must come from the Serenity CUDA path, not CPU.",
                "details": {
                    "train_device": train_device,
                    "override": "--train-device cuda --temp-device cpu",
                },
            })

    if local_model is not None:
        transformer_config = local_model["transformer_config"]
        if local_model["base_model_name_mentions_klein"]:
            blockers.append({
                "id": "flux2_klein_config_not_dev",
                "category": "variant",
                "message": "The configured base model name points at Flux2 Klein, not Flux2 dev.",
                "details": {"base_model": local_model["model_names"]["base_model"]},
            })
        elif not transformer_config["structural_dev_hint"]:
            blockers.append({
                "id": "flux2_dev_variant_unverified",
                "category": "variant",
                "message": (
                    "Dry-run did not prove Flux2 dev. Serenity defines dev as "
                    "Flux2Model.is_dev() == True, i.e. transformer.config.num_attention_heads == 48."
                ),
                "details": transformer_config,
            })

        if not local_model["base_model_exists"] and local_model["base_model_is_local_path"]:
            blockers.append({
                "id": "missing_base_model_path",
                "category": "model",
                "message": "Configured local base model path does not exist.",
                "details": {"base_model": local_model["model_names"]["base_model"]},
            })
        if not local_model["base_model_is_local_path"]:
            blockers.append({
                "id": "remote_base_model_reference",
                "category": "model",
                "message": "Dry-run will not download remote Flux2 dev weights.",
                "details": {"base_model": local_model["model_names"]["base_model"]},
            })

    if not torch.cuda.is_available():
        blockers.append({
            "id": "cuda_unavailable_in_current_process",
            "category": "cuda_reference",
            "message": "Current process does not report CUDA; dry-run remains structural only.",
            "details": {
                "cuda_available": False,
                "cuda_device_count": torch.cuda.device_count(),
            },
        })

    if not ref_tensors["complete"]:
        blockers.append({
            "id": "missing_serenity_cuda_reference_tensors",
            "category": "numeric_parity",
            "message": NO_NUMERIC_PARITY_NOTE,
            "details": ref_tensors,
        })

    return blockers


def contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": BLOCKER_SCHEMA_VERSION,
        "name": DEFAULT_PREFIX,
        "purpose": (
            "Serenity Flux2 dev cached-data train-reference scaffold for "
            "predict -> loss -> backward_lora -> optimizer parity gates."
        ),
        "producer_script": str(Path(__file__).resolve()),
        "reference_policy": {
            "only_reference": str(args.serenity),
            "external_references_used": False,
            "cpu_pytorch_dry_runs": "structural/import/config/cache checks only; never numeric parity",
        },
        "numeric_parity_policy": {
            "status": "none_until_cuda_reference_tensors_exist",
            "note": NO_NUMERIC_PARITY_NOTE,
            "required_files": REQUIRED_TENSOR_FILES,
            "forbidden_claims": [
                "CPU PyTorch numeric parity",
                "dry-run numeric parity",
                "Flux2 Klein tensors standing in for Flux2 dev tensors",
            ],
        },
        "reference_files_inspected": REFERENCE_SOURCES,
        "source_line_anchors": {
            "Flux2Model.is_dev": "/home/alex/Serenity/modules/model/Flux2Model.py:232",
            "Flux2Model.encode_text_dev_branch": "/home/alex/Serenity/modules/model/Flux2Model.py:179",
            "Flux2ModelLoader.dev_branch": "/home/alex/Serenity/modules/modelLoader/Flux2ModelLoader.py:100",
            "Flux2BaseDataLoader.dev_branch": "/home/alex/Serenity/modules/dataLoader/Flux2BaseDataLoader.py:42",
            "BaseFlux2Setup.predict": "/home/alex/Serenity/modules/modelSetup/BaseFlux2Setup.py:82",
            "Flux2LoRASetup.setup_model": "/home/alex/Serenity/modules/modelSetup/Flux2LoRASetup.py:52",
            "Flux2Sampler.__sample_base": "/home/alex/Serenity/modules/modelSampler/Flux2Sampler.py:39",
            "Flux2Sampler.sample": "/home/alex/Serenity/modules/modelSampler/Flux2Sampler.py:163",
        },
        "variant_contract": {
            "model_type": "FLUX_2",
            "training_method": "LORA",
            "dev_detection": (
                "Serenity Flux2Model.is_dev() returns True iff "
                "model.transformer.config.num_attention_heads == 48."
            ),
            "dev_text_path": "PixtralProcessor tokenizer plus Mistral3 hidden states.",
            "klein_text_path": "Qwen tokenizer/Qwen3 hidden states; not valid for Flux2 dev parity.",
        },
        "default_config": str(DEFAULT_CONFIG),
        "local_partial_flux2_dev_transformer_config": local_dev_transformer_config_status(),
        "preset_candidates": PRESET_CANDIDATES,
        "default_outputs": {
            "run_metadata": f"parity/{DEFAULT_PREFIX}_meta.json",
            "step_tensors": f"parity/{DEFAULT_PREFIX}_step000.safetensors",
            "adapter_tensors": f"parity/{DEFAULT_PREFIX}_step000_adapters.safetensors",
            "dry_run_blockers": f"parity/{DEFAULT_PREFIX}_blockers.json",
            "contract": f"parity/{DEFAULT_PREFIX}_contract.json",
        },
        "step_safetensors_keys": {
            "cached_batch": (
                "batch.* tensors exactly as Serenity Flux2BaseDataLoader yields them, "
                "including latent_image, tokens, tokens_mask, text_encoder_hidden_state, "
                "loss_weight, concept_type, and image metadata."
            ),
            "trace": (
                "trace.* tensors from BaseFlux2Setup.predict and Flux2Model helpers: "
                "scaled_latent_image, latent_noise, scaled_noisy_latent_image, sigma, "
                "packed_latent_input, transformer_timestep, guidance, encoder_hidden_states, "
                "text_ids, image_ids, packed_predicted_flow, predicted_flow, and flow."
            ),
            "outputs": (
                "output.timestep, output.predicted, output.target, optional prior_target, "
                "output.loss_pre_scale, and output.loss_for_backward."
            ),
        },
        "bounded_run_notes": [
            "The current scaffold is dry-run/contract first and does not claim tensor parity.",
            "A future tensor-producing run must use Serenity from /home/alex/Serenity on CUDA.",
            "The runtime model must satisfy model.is_dev() after Serenity loads the transformer.",
            "Existing Klein train-reference tensors are not acceptable Flux2 dev substitutes.",
            "Dry-run writes blockers and contract JSON without loading Flux2 dev model weights.",
        ],
    }


def run_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_error = load_config_for_dry_run(args)
    cache = cache_status(config)
    local_model = model_status(config)
    local_dev_config = local_dev_transformer_config_status()
    ref_tensors = reference_tensor_status(args)
    blockers = structured_blockers(args, config, config_error, local_model, cache, ref_tensors)

    report = {
        "schema_version": BLOCKER_SCHEMA_VERSION,
        "producer": PRODUCER,
        "created_unix": time.time(),
        "dry_run": True,
        "one_step_dump_produced": False,
        "numeric_parity_status": "none",
        "numeric_parity_note": NO_NUMERIC_PARITY_NOTE,
        "dry_run_checks_are_structural_only": True,
        "serenity": str(args.serenity),
        "config": str(args.config),
        "config_exists": args.config.is_file(),
        "config_error": config_error,
        "out_dir": str(args.out_dir),
        "prefix": args.prefix,
        "max_steps": args.max_steps,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "train_device": str(config.train_device) if config is not None else args.train_device,
        "temp_device": str(config.temp_device) if config is not None else args.temp_device,
        "model_type": klein.enum_value(config.model_type) if config is not None else None,
        "training_method": klein.enum_value(config.training_method) if config is not None else None,
        "cache": cache,
        "local_model": local_model,
        "local_partial_flux2_dev_transformer_config": local_dev_config,
        "reference_tensors": ref_tensors,
        "structured_blockers": blockers,
        "blocked": bool(blockers),
    }

    write_json(args.out_dir / f"{args.prefix}_contract.json", contract(args))
    write_json(args.out_dir / f"{args.prefix}_blockers.json", report)
    return report


def main() -> None:
    args = parse_args()
    resolve_args(args)

    if not args.dry_run:
        raise SystemExit(
            "Flux2 dev train-reference tensor production is not enabled in this scaffold. "
            "Run --dry-run for structural blockers. Numeric parity remains absent until "
            "a Serenity CUDA path produces flux2_dev_train_ref_step000*.safetensors."
        )

    report = run_dry_run(args)
    print(json.dumps(klein.json_safe(report), indent=2))


if __name__ == "__main__":
    main()
