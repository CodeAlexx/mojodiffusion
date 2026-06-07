#!/usr/bin/env python3
"""Flux.1-dev sampler artifact manifest guard.

This is a no-CUDA metadata/header gate for Flux.1-dev sampler parity evidence.
It does not run text encoders, the transformer, VAE, or image comparison. It
defines the concrete OneTrainer/Mojo artifact bundle required before a Flux.1
"sampler parity" or "sampler speed parity" claim can be accepted.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("/home/alex/onetrainer-mojo/parity/flux1_sampler_parity_manifest.json")
EXPECTED_MODEL_TYPE = "FLUX_DEV_1"
EXPECTED_SCHEDULER = "FlowMatchEulerDiscreteScheduler"
EXPECTED_LATENT_CHANNELS = 16
EXPECTED_PACKED_CHANNELS = 64
TEMPLATE_PROMPT_ID = "flux1_dev_sampler_parity_todo"
TEMPLATE_PROMPT_POSITIVE = "TODO_FILL_EXACT_PROMPT"
TEMPLATE_PROMPT_NEGATIVE = ""
TEMPLATE_SEED = 42
TEMPLATE_WIDTH = 1024
TEMPLATE_HEIGHT = 1024
TEMPLATE_STEPS = 20
TEMPLATE_TEXT_TOKENS = 512
TEMPLATE_CFG_SCALE = 3.5
REQUIRED_PROMPT_KEYS = (
    "id",
    "positive",
    "negative",
    "seed",
    "width",
    "height",
    "steps",
    "text_tokens",
    "random_seed",
    "cfg_scale",
)
REQUIRED_SCHEDULER_KEYS = (
    "name",
    "sigmas",
    "timesteps",
    "mu",
    "dynamic_shift",
    "step_trace",
)
REQUIRED_ARTIFACT_KEYS = (
    "onetrainer_initial_noise_raw_nchw",
    "onetrainer_initial_noise_packed",
    "onetrainer_latent_trajectory",
    "mojo_latent_trajectory",
    "onetrainer_prompt_embedding",
    "onetrainer_pooled_prompt_embedding",
    "onetrainer_text_ids",
    "onetrainer_image_ids",
    "onetrainer_final_packed_latent",
    "onetrainer_final_unpacked_latent",
    "onetrainer_vae_input_latent",
    "onetrainer_vae_decoded_tensor",
    "mojo_prompt_embedding",
    "mojo_pooled_prompt_embedding",
    "mojo_text_ids",
    "mojo_image_ids",
    "mojo_final_packed_latent",
    "mojo_final_unpacked_latent",
    "mojo_vae_input_latent",
    "mojo_vae_decoded_tensor",
    "onetrainer_png",
    "mojo_png",
)
REQUIRED_PNG_ARTIFACT_KEYS = ("onetrainer_png", "mojo_png")
REQUIRED_TENSOR_ARTIFACT_KEYS = tuple(
    key for key in REQUIRED_ARTIFACT_KEYS if key not in REQUIRED_PNG_ARTIFACT_KEYS
)
REQUIRED_METRIC_KEYS = (
    "text_seconds",
    "denoise_seconds_per_step",
    "vae_decode_seconds",
    "postprocess_save_seconds",
    "peak_vram_mib",
)
REQUIRED_COMPARISON_KEYS = (
    "text_conditioning",
    "trajectory",
    "final_latent",
    "vae_tensor",
    "vae_png",
)


@dataclass(frozen=True)
class Check:
    ok: bool
    label: str
    detail: str


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return data


def as_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def as_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def as_path(value: Any, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key}.path must be a non-empty string")
    return Path(value)


def numel(shape: list[int]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def read_safetensors_header(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"safetensors file too small: {path}")
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = raw[8 : 8 + header_len]
    decoded = json.loads(header.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    return decoded


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"not a PNG IHDR header: {path}")
    return struct.unpack(">II", header[16:24])


def path_check(path: Path, label: str) -> Check:
    if not path.exists():
        return Check(False, label, f"missing file: {path}")
    if path.stat().st_size <= 0:
        return Check(False, label, f"empty file: {path}")
    return Check(True, label, f"present bytes={path.stat().st_size} path={path}")


def artifact_path(artifacts: dict[str, Any], name: str) -> Path:
    value = artifacts.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"artifacts.{name} must be an object")
    return as_path(value.get("path"), f"artifacts.{name}")


def check_tensor_artifact(
    artifacts: dict[str, Any],
    name: str,
    *,
    expected_dtype: str | None = None,
    expected_rank: int | None = None,
    allowed_shapes: tuple[tuple[int, ...], ...] = (),
) -> list[Check]:
    value = artifacts.get(name)
    if not isinstance(value, dict):
        return [Check(False, f"artifact {name}", f"artifacts.{name} must be an object")]
    path = as_path(value.get("path"), f"artifacts.{name}")
    checks = [path_check(path, f"artifact {name}")]
    if not checks[-1].ok:
        return checks

    dtype = value.get("dtype")
    if expected_dtype is not None and dtype != expected_dtype:
        checks.append(Check(False, f"artifact {name} dtype", f"got {dtype!r}, expected {expected_dtype!r}"))
    elif not isinstance(dtype, str) or not dtype:
        checks.append(Check(False, f"artifact {name} dtype", "dtype must be recorded"))
    else:
        checks.append(Check(True, f"artifact {name} dtype", f"dtype={dtype!r}"))

    shape_raw = value.get("shape")
    if not isinstance(shape_raw, list) or not all(isinstance(dim, int) for dim in shape_raw):
        checks.append(Check(False, f"artifact {name} shape", "shape must be a list of ints"))
        return checks
    shape = [int(dim) for dim in shape_raw]
    if expected_rank is not None and len(shape) != expected_rank:
        checks.append(Check(False, f"artifact {name} rank", f"got shape={shape}, expected rank={expected_rank}"))
    elif allowed_shapes and tuple(shape) not in allowed_shapes:
        checks.append(Check(False, f"artifact {name} shape", f"got shape={shape}, allowed={list(allowed_shapes)}"))
    else:
        checks.append(Check(True, f"artifact {name} shape", f"shape={shape}"))

    if path.suffix == ".safetensors":
        try:
            header = read_safetensors_header(path)
            keys = [key for key in header if key != "__metadata__"]
            checks.append(Check(bool(keys), f"artifact {name} safetensors", f"tensors={len(keys)}"))
        except Exception as exc:  # noqa: BLE001 - report damaged local artifact.
            checks.append(Check(False, f"artifact {name} safetensors", str(exc)))
    elif path.suffix == ".bin":
        expected_bytes = value.get("byte_size")
        if isinstance(expected_bytes, int):
            actual = path.stat().st_size
            checks.append(
                Check(
                    actual == expected_bytes,
                    f"artifact {name} byte_size",
                    f"actual={actual} expected={expected_bytes}",
                )
            )
        else:
            checks.append(Check(True, f"artifact {name} bin", f"declared_numel={numel(shape)}"))
    return checks


def check_numeric_pair(data: dict[str, Any], key: str) -> Check:
    value = data.get(key)
    if not isinstance(value, dict):
        return Check(False, f"comparison {key}", f"comparisons.{key} must be an object")
    accepted = value.get("accepted")
    max_abs = value.get("max_abs")
    tolerance = value.get("tolerance")
    if accepted is not True:
        return Check(False, f"comparison {key}", "accepted must be true")
    if not isinstance(max_abs, (int, float)) or not isinstance(tolerance, (int, float)):
        return Check(False, f"comparison {key}", "max_abs and tolerance must be numeric")
    return Check(float(max_abs) <= float(tolerance), f"comparison {key}", f"max_abs={max_abs} tolerance={tolerance}")


def check_metric_pair(metrics: dict[str, Any], key: str) -> Check:
    ot = as_dict(metrics, "onetrainer")
    mojo = as_dict(metrics, "mojo")
    ot_value = ot.get(key)
    mojo_value = mojo.get(key)
    if not isinstance(ot_value, (int, float)) or not isinstance(mojo_value, (int, float)):
        return Check(False, f"metric {key}", "onetrainer and mojo values must be numeric")
    if float(ot_value) <= 0.0 or float(mojo_value) <= 0.0:
        return Check(False, f"metric {key}", f"values must be positive: ot={ot_value} mojo={mojo_value}")
    return Check(True, f"metric {key}", f"ot={ot_value} mojo={mojo_value}")


def check_step_trace(scheduler: dict[str, Any], steps: int) -> list[Check]:
    trace = scheduler.get("step_trace")
    if not isinstance(trace, list):
        return [Check(False, "scheduler.step_trace", "must be a list")]
    checks = [Check(len(trace) == steps, "scheduler.step_trace", f"len={len(trace)} expected={steps}")]
    for index, item in enumerate(trace[:steps]):
        if not isinstance(item, dict):
            checks.append(Check(False, f"scheduler.step_trace[{index}]", "must be an object"))
            continue
        checks.append(Check(item.get("index") == index, f"scheduler.step_trace[{index}].index", f"got={item.get('index')!r}"))
        for key in ("sigma", "sigma_next", "dt", "timestep", "model_timestep"):
            checks.append(
                Check(
                    isinstance(item.get(key), (int, float)),
                    f"scheduler.step_trace[{index}].{key}",
                    str(item.get(key)),
                )
            )
    return checks


def check_schema_markers(data: dict[str, Any]) -> list[Check]:
    schema_version = data.get("schema_version")
    template = data.get("template")
    return [
        Check(schema_version == 1, "schema_version", f"got={schema_version!r} expected=1"),
        Check(template is not True, "template", "must not be true for strict evidence manifests"),
    ]


def _todo_series(prefix: str, count: int) -> list[str]:
    return [f"TODO_{prefix}_{index}" for index in range(count)]


def _template_step_trace(steps: int) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "sigma": f"TODO_SIGMA_{index}",
            "sigma_next": f"TODO_SIGMA_{index + 1}",
            "dt": f"TODO_DT_{index}",
            "timestep": f"TODO_TIMESTEP_{index}",
            "model_timestep": f"TODO_MODEL_TIMESTEP_{index}",
        }
        for index in range(steps)
    ]


def _template_tensor_artifact(
    artifact_root: str,
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: str,
) -> dict[str, Any]:
    return {
        "path": f"{artifact_root}/{name}.safetensors",
        "dtype": dtype,
        "shape": list(shape),
    }


def _template_png_artifact(artifact_root: str, name: str, width: int, height: int) -> dict[str, Any]:
    return {
        "path": f"{artifact_root}/{name}.png",
        "width": width,
        "height": height,
    }


def _template_artifact_groups() -> dict[str, Any]:
    return {
        "one_trainer_seed_replay_inputs": [
            "onetrainer_initial_noise_raw_nchw",
            "onetrainer_initial_noise_packed",
        ],
        "paired_text_conditioning": [
            {"onetrainer": "onetrainer_prompt_embedding", "mojo": "mojo_prompt_embedding"},
            {
                "onetrainer": "onetrainer_pooled_prompt_embedding",
                "mojo": "mojo_pooled_prompt_embedding",
            },
            {"onetrainer": "onetrainer_text_ids", "mojo": "mojo_text_ids"},
            {"onetrainer": "onetrainer_image_ids", "mojo": "mojo_image_ids"},
        ],
        "paired_latent_trajectory": [
            {"onetrainer": "onetrainer_latent_trajectory", "mojo": "mojo_latent_trajectory"},
            {"onetrainer": "onetrainer_final_packed_latent", "mojo": "mojo_final_packed_latent"},
            {"onetrainer": "onetrainer_final_unpacked_latent", "mojo": "mojo_final_unpacked_latent"},
            {"onetrainer": "onetrainer_vae_input_latent", "mojo": "mojo_vae_input_latent"},
        ],
        "paired_decode_and_png": [
            {"onetrainer": "onetrainer_vae_decoded_tensor", "mojo": "mojo_vae_decoded_tensor"},
            {"onetrainer": "onetrainer_png", "mojo": "mojo_png"},
        ],
    }


def required_manifest_fields() -> dict[str, Any]:
    return {
        "top_level": [
            "schema_version",
            "model_type",
            "prompt",
            "scheduler",
            "artifacts",
            "metrics",
            "comparisons",
        ],
        "prompt": list(REQUIRED_PROMPT_KEYS),
        "scheduler": list(REQUIRED_SCHEDULER_KEYS),
        "tensor_artifacts": [
            f"{key}.path/dtype/shape" for key in REQUIRED_TENSOR_ARTIFACT_KEYS
        ],
        "png_artifacts": [f"{key}.path" for key in REQUIRED_PNG_ARTIFACT_KEYS],
        "metrics": [f"onetrainer.{key}" for key in REQUIRED_METRIC_KEYS]
        + [f"mojo.{key}" for key in REQUIRED_METRIC_KEYS],
        "comparisons": [
            f"{key}.accepted/max_abs/tolerance" for key in REQUIRED_COMPARISON_KEYS
        ],
    }


def required_paired_sampler_fields(target_manifest: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    return {
        "one_trainer_reference": "/home/alex/OneTrainer FluxSampler.py, same prompt/seed/resolution/steps/cfg/dtype",
        "mojo_reference": "serenitymojo Flux.1 sampler/product path, same prompt/seed/resolution/steps/cfg/dtype",
        "strict_check_command": (
            "python3 scripts/check_flux1_sampler_artifact_manifest.py "
            f"--manifest {target_manifest} --strict"
        ),
        "template_command": (
            "python3 scripts/check_flux1_sampler_artifact_manifest.py "
            f"--manifest {target_manifest} "
            "--write-template /tmp/flux1_sampler_parity_manifest.template.json"
        ),
        "readiness_command": (
            "python3 scripts/check_flux1_sampler_artifact_manifest.py "
            f"--manifest {target_manifest} "
            "--write-readiness /tmp/flux1_sampler_manifest_readiness.json"
        ),
        "parity_acceptance_note": (
            "All tensor paths must point to real paired OneTrainer/Mojo artifacts; "
            "all timings and peak VRAM must be positive; all comparisons must set "
            "accepted=true with max_abs <= tolerance."
        ),
    }


def required_artifacts() -> list[str]:
    return list(REQUIRED_ARTIFACT_KEYS)


def build_template_manifest(
    *,
    target_manifest: Path = DEFAULT_MANIFEST,
    artifact_root: str = "/tmp/flux1_sampler_parity_artifacts",
    width: int = TEMPLATE_WIDTH,
    height: int = TEMPLATE_HEIGHT,
    steps: int = TEMPLATE_STEPS,
    text_tokens: int = TEMPLATE_TEXT_TOKENS,
) -> dict[str, Any]:
    latent_h = height // 8
    latent_w = width // 8
    packed_h = height // 16
    packed_w = width // 16
    n_img = packed_h * packed_w
    raw_shape = (1, EXPECTED_LATENT_CHANNELS, latent_h, latent_w)
    packed_shape = (1, n_img, EXPECTED_PACKED_CHANNELS)
    trajectory_shape = (steps + 1, 1, n_img, EXPECTED_PACKED_CHANNELS)
    prompt_embedding_shape = (1, text_tokens, 4096)
    pooled_embedding_shape = (1, 768)
    text_ids_shape = (text_tokens, 3)
    image_ids_shape = (n_img, 3)
    decoded_shape = (1, 3, height, width)

    artifacts: dict[str, Any] = {
        "onetrainer_initial_noise_raw_nchw": _template_tensor_artifact(
            artifact_root, "onetrainer_initial_noise_raw_nchw", raw_shape, dtype="F32"
        ),
        "onetrainer_initial_noise_packed": _template_tensor_artifact(
            artifact_root, "onetrainer_initial_noise_packed", packed_shape, dtype="F32"
        ),
        "onetrainer_latent_trajectory": _template_tensor_artifact(
            artifact_root, "onetrainer_latent_trajectory", trajectory_shape, dtype="F32"
        ),
        "mojo_latent_trajectory": _template_tensor_artifact(
            artifact_root, "mojo_latent_trajectory", trajectory_shape, dtype="TODO_FILL_STORAGE_DTYPE"
        ),
    }
    for prefix in ("onetrainer", "mojo"):
        dtype = "TODO_FILL_STORAGE_DTYPE"
        artifacts[f"{prefix}_prompt_embedding"] = _template_tensor_artifact(
            artifact_root, f"{prefix}_prompt_embedding", prompt_embedding_shape, dtype=dtype
        )
        artifacts[f"{prefix}_pooled_prompt_embedding"] = _template_tensor_artifact(
            artifact_root, f"{prefix}_pooled_prompt_embedding", pooled_embedding_shape, dtype=dtype
        )
        artifacts[f"{prefix}_text_ids"] = _template_tensor_artifact(
            artifact_root, f"{prefix}_text_ids", text_ids_shape, dtype=dtype
        )
        artifacts[f"{prefix}_image_ids"] = _template_tensor_artifact(
            artifact_root, f"{prefix}_image_ids", image_ids_shape, dtype=dtype
        )
        artifacts[f"{prefix}_final_packed_latent"] = _template_tensor_artifact(
            artifact_root, f"{prefix}_final_packed_latent", packed_shape, dtype=dtype
        )
        artifacts[f"{prefix}_final_unpacked_latent"] = _template_tensor_artifact(
            artifact_root, f"{prefix}_final_unpacked_latent", raw_shape, dtype=dtype
        )
        artifacts[f"{prefix}_vae_input_latent"] = _template_tensor_artifact(
            artifact_root, f"{prefix}_vae_input_latent", raw_shape, dtype=dtype
        )
        artifacts[f"{prefix}_vae_decoded_tensor"] = _template_tensor_artifact(
            artifact_root, f"{prefix}_vae_decoded_tensor", decoded_shape, dtype=dtype
        )
    artifacts["onetrainer_png"] = _template_png_artifact(artifact_root, "onetrainer_png", width, height)
    artifacts["mojo_png"] = _template_png_artifact(artifact_root, "mojo_png", width, height)

    return {
        "schema_version": 1,
        "producer": "scripts/check_flux1_sampler_artifact_manifest.py --write-template",
        "template": True,
        "template_note": "Fill every TODO path/value with real paired OneTrainer/Mojo evidence before strict use.",
        "target_manifest": str(target_manifest),
        "model_type": EXPECTED_MODEL_TYPE,
        "parity_claimed": False,
        "parity_note": "Template only; this is not sampler parity evidence.",
        "prompt": {
            "id": TEMPLATE_PROMPT_ID,
            "positive": TEMPLATE_PROMPT_POSITIVE,
            "negative": TEMPLATE_PROMPT_NEGATIVE,
            "seed": TEMPLATE_SEED,
            "width": width,
            "height": height,
            "steps": steps,
            "text_tokens": text_tokens,
            "random_seed": False,
            "cfg_scale": TEMPLATE_CFG_SCALE,
        },
        "scheduler": {
            "name": EXPECTED_SCHEDULER,
            "sigmas": _todo_series("SIGMA", steps + 1),
            "timesteps": _todo_series("TIMESTEP", steps),
            "mu": "TODO_NUMERIC_MU",
            "dynamic_shift": "TODO_NUMERIC_DYNAMIC_SHIFT",
            "step_trace": _template_step_trace(steps),
        },
        "artifact_groups": _template_artifact_groups(),
        "artifacts": artifacts,
        "metrics": {
            "onetrainer": {key: 0.0 for key in REQUIRED_METRIC_KEYS},
            "mojo": {key: 0.0 for key in REQUIRED_METRIC_KEYS},
        },
        "comparisons": {
            key: {
                "accepted": False,
                "max_abs": None,
                "tolerance": None,
                "note": "Fill with real numeric comparison and set accepted true only when max_abs <= tolerance.",
            }
            for key in REQUIRED_COMPARISON_KEYS
        },
        "required_manifest_fields": required_manifest_fields(),
        "required_paired_sampler_fields": required_paired_sampler_fields(target_manifest),
        "required_paired_sampler_artifacts": required_artifacts(),
    }


def write_template_manifest(path: Path, target_manifest: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_template_manifest(target_manifest=target_manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[flux1-sampler-manifest] INFO wrote template manifest: {path}")


def inspect_manifest(path: Path) -> list[Check]:
    if not path.exists():
        return [
            Check(
                False,
                "manifest",
                f"missing {path}; expected paired OT/Mojo Flux.1 sampler evidence manifest",
            )
        ]

    data = load_json(path)
    checks: list[Check] = [Check(True, "manifest", f"loaded {path}")]
    checks.extend(check_schema_markers(data))
    model_type = data.get("model_type")
    checks.append(Check(model_type == EXPECTED_MODEL_TYPE, "model_type", f"got={model_type!r} expected={EXPECTED_MODEL_TYPE!r}"))

    prompt = as_dict(data, "prompt")
    for key in ("id", "positive"):
        checks.append(Check(isinstance(prompt.get(key), str) and bool(prompt.get(key)), f"prompt.{key}", str(prompt.get(key))))
    checks.append(Check(isinstance(prompt.get("negative"), str), "prompt.negative", str(prompt.get("negative"))))
    for key in ("seed", "width", "height", "steps", "text_tokens"):
        checks.append(Check(isinstance(prompt.get(key), int) and int(prompt.get(key)) > 0, f"prompt.{key}", str(prompt.get(key))))
    checks.append(Check(prompt.get("random_seed") is False, "prompt.random_seed", "must be false for deterministic OT replay"))
    checks.append(Check(isinstance(prompt.get("cfg_scale"), (int, float)), "prompt.cfg_scale", str(prompt.get("cfg_scale"))))

    width = int(prompt.get("width", 0)) if isinstance(prompt.get("width"), int) else 0
    height = int(prompt.get("height", 0)) if isinstance(prompt.get("height"), int) else 0
    steps = int(prompt.get("steps", 0)) if isinstance(prompt.get("steps"), int) else 0
    text_tokens = int(prompt.get("text_tokens", 0)) if isinstance(prompt.get("text_tokens"), int) else 0
    checks.append(Check(width % 16 == 0 and height % 16 == 0, "prompt.resolution_packable", f"{width}x{height}"))

    latent_h = height // 8 if height > 0 else 0
    latent_w = width // 8 if width > 0 else 0
    packed_h = height // 16 if height > 0 else 0
    packed_w = width // 16 if width > 0 else 0
    n_img = packed_h * packed_w
    raw_shape = (1, EXPECTED_LATENT_CHANNELS, latent_h, latent_w)
    packed_shape = (1, n_img, EXPECTED_PACKED_CHANNELS)
    packed_shape_no_batch = (n_img, EXPECTED_PACKED_CHANNELS)
    trajectory_shapes = (
        (steps + 1, 1, n_img, EXPECTED_PACKED_CHANNELS),
        (steps + 1, n_img, EXPECTED_PACKED_CHANNELS),
    )
    prompt_embedding_shape = (1, text_tokens, 4096)
    pooled_embedding_shape = (1, 768)
    text_ids_shape = (text_tokens, 3)
    image_ids_shape = (n_img, 3)

    scheduler = as_dict(data, "scheduler")
    scheduler_name = scheduler.get("name")
    checks.append(Check(scheduler_name == EXPECTED_SCHEDULER, "scheduler.name", f"got={scheduler_name!r}"))
    sigmas = as_list(scheduler, "sigmas")
    checks.append(Check(len(sigmas) == steps + 1, "scheduler.sigmas", f"len={len(sigmas)} expected={steps + 1}"))
    checks.append(Check(all(isinstance(value, (int, float)) for value in sigmas), "scheduler.sigmas.numeric", "all numeric"))
    timesteps = as_list(scheduler, "timesteps")
    checks.append(Check(len(timesteps) == steps, "scheduler.timesteps", f"len={len(timesteps)} expected={steps}"))
    checks.append(Check(all(isinstance(value, (int, float)) for value in timesteps), "scheduler.timesteps.numeric", "all numeric"))
    checks.append(Check(isinstance(scheduler.get("mu"), (int, float)), "scheduler.mu", str(scheduler.get("mu"))))
    checks.append(Check(isinstance(scheduler.get("dynamic_shift"), (int, float)), "scheduler.dynamic_shift", str(scheduler.get("dynamic_shift"))))
    checks.extend(check_step_trace(scheduler, steps))

    artifacts = as_dict(data, "artifacts")
    checks.extend(
        check_tensor_artifact(
            artifacts,
            "onetrainer_initial_noise_raw_nchw",
            expected_dtype="F32",
            allowed_shapes=(raw_shape,),
        )
    )
    checks.extend(
        check_tensor_artifact(
            artifacts,
            "onetrainer_initial_noise_packed",
            expected_dtype="F32",
            allowed_shapes=(packed_shape, packed_shape_no_batch),
        )
    )
    checks.extend(
        check_tensor_artifact(
            artifacts,
            "onetrainer_latent_trajectory",
            expected_dtype="F32",
            allowed_shapes=trajectory_shapes,
        )
    )
    checks.extend(
        check_tensor_artifact(
            artifacts,
            "mojo_latent_trajectory",
            allowed_shapes=trajectory_shapes,
        )
    )
    for prefix in ("onetrainer", "mojo"):
        checks.extend(
            check_tensor_artifact(
                artifacts,
                f"{prefix}_prompt_embedding",
                allowed_shapes=(prompt_embedding_shape,),
            )
        )
        checks.extend(
            check_tensor_artifact(
                artifacts,
                f"{prefix}_pooled_prompt_embedding",
                allowed_shapes=(pooled_embedding_shape,),
            )
        )
        checks.extend(
            check_tensor_artifact(
                artifacts,
                f"{prefix}_text_ids",
                allowed_shapes=(text_ids_shape,),
            )
        )
        checks.extend(
            check_tensor_artifact(
                artifacts,
                f"{prefix}_image_ids",
                allowed_shapes=(image_ids_shape,),
            )
        )
        checks.extend(
            check_tensor_artifact(
                artifacts,
                f"{prefix}_final_packed_latent",
                allowed_shapes=(packed_shape, packed_shape_no_batch),
            )
        )
        checks.extend(
            check_tensor_artifact(
                artifacts,
                f"{prefix}_final_unpacked_latent",
                allowed_shapes=(raw_shape,),
            )
        )
        checks.extend(
            check_tensor_artifact(
                artifacts,
                f"{prefix}_vae_input_latent",
                allowed_shapes=(raw_shape,),
            )
        )
        checks.extend(
            check_tensor_artifact(
                artifacts,
                f"{prefix}_vae_decoded_tensor",
                allowed_shapes=((1, 3, height, width),),
            )
        )
    for image_key in ("onetrainer_png", "mojo_png"):
        try:
            image_path = artifact_path(artifacts, image_key)
            present = path_check(image_path, f"artifact {image_key}")
            checks.append(present)
            if present.ok:
                png_w, png_h = png_size(image_path)
                checks.append(
                    Check(png_w == width and png_h == height, f"artifact {image_key} dimensions", f"{png_w}x{png_h}")
                )
        except Exception as exc:  # noqa: BLE001 - report damaged local artifact.
            checks.append(Check(False, f"artifact {image_key}", str(exc)))

    metrics = as_dict(data, "metrics")
    for key in (
        "text_seconds",
        "denoise_seconds_per_step",
        "vae_decode_seconds",
        "postprocess_save_seconds",
        "peak_vram_mib",
    ):
        checks.append(check_metric_pair(metrics, key))

    comparisons = as_dict(data, "comparisons")
    for key in ("text_conditioning", "trajectory", "final_latent", "vae_tensor", "vae_png"):
        checks.append(check_numeric_pair(comparisons, key))
    return checks


def print_report(path: Path, checks: list[Check]) -> None:
    print("[flux1-sampler-manifest] scope=no-CUDA metadata/header gate")
    print(f"[flux1-sampler-manifest] manifest={path}")
    for check in checks:
        status = "PASS" if check.ok else "BLOCKED"
        print(f"[flux1-sampler-manifest] {status} {check.label}: {check.detail}")
    blockers = [check for check in checks if not check.ok]
    print(f"[flux1-sampler-manifest] blockers={len(blockers)}")


def check_as_json(check: Check) -> dict[str, Any]:
    return {"ok": check.ok, "label": check.label, "detail": check.detail}


def build_readiness_report(path: Path, checks: list[Check]) -> dict[str, Any]:
    blockers = [check_as_json(check) for check in checks if not check.ok]
    return {
        "schema_version": 1,
        "producer": "scripts/check_flux1_sampler_artifact_manifest.py",
        "scope": "Flux.1-dev sampler manifest readiness; no CUDA, no denoise, no VAE run, no image comparison",
        "manifest": str(path),
        "strict_manifest_ready": not blockers,
        "parity_claimed": False,
        "parity_note": "This readiness report only lists manifest evidence and blockers; it does not accept sampler parity.",
        "required_manifest_fields": required_manifest_fields(),
        "required_paired_sampler_fields": required_paired_sampler_fields(path),
        "required_paired_sampler_artifacts": required_artifacts(),
        "present_evidence": [check_as_json(check) for check in checks if check.ok],
        "current_blockers": blockers,
        "blockers": blockers,
        "checks": [check_as_json(check) for check in checks],
    }


def write_readiness_report(path: Path, manifest: Path, checks: list[Check]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_readiness_report(manifest, checks), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[flux1-sampler-manifest] INFO wrote readiness report: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--write-readiness",
        type=Path,
        help="Write a no-CUDA JSON report listing required paired sampler evidence and current blockers.",
    )
    parser.add_argument(
        "--write-template",
        type=Path,
        help="Write a fill-in skeleton for the paired Flux.1-dev sampler parity manifest.",
    )
    args = parser.parse_args()

    try:
        checks = inspect_manifest(args.manifest)
    except Exception as exc:  # noqa: BLE001 - keep this a guard, not a traceback.
        checks = [Check(False, "manifest", str(exc))]

    if args.write_template is not None:
        write_template_manifest(args.write_template, args.manifest)

    if args.write_readiness is not None:
        write_readiness_report(args.write_readiness, args.manifest, checks)

    if args.json:
        print(
            json.dumps(
                {
                    "producer": "scripts/check_flux1_sampler_artifact_manifest.py",
                    "manifest": str(args.manifest),
                    "scope": "no-CUDA metadata/header gate",
                    "ok": all(check.ok for check in checks),
                    "checks": [check.__dict__ for check in checks],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print_report(args.manifest, checks)

    if args.strict and not all(check.ok for check in checks):
        return 2
    return 0 if checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
