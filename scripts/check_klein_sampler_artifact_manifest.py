#!/usr/bin/env python3
"""Klein/Flux2 sampler artifact manifest guard.

This is a no-CUDA metadata/header gate for the sampler parity evidence that is
still missing. It does not run the DiT, VAE, or image comparison. It only makes
the required OneTrainer/Mojo sampler artifact bundle explicit enough that a
later parity run cannot hide behind vague "sample worked" output.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("/home/alex/onetrainer-mojo/parity/klein_sampler_parity_manifest.json")
ONETRAINER_PRODUCER = "scripts/run_klein_onetrainer_sampler_parity.py"
EXPECTED_MODEL_TYPE = "FLUX_2"
EXPECTED_SCHEDULER = "FlowMatchEuler"
TEMPLATE_PROMPT_ID = "alina_garden"
TEMPLATE_PROMPT_POSITIVE = (
    "alverone , a high-resolution photograph featuring a young caucasian woman "
    "with long blonde hair, wearing a casual white sundress, standing in a "
    "sunlit garden, soft natural lighting, professional photography"
)
TEMPLATE_PROMPT_NEGATIVE = "TODO_FILL_NEGATIVE_PROMPT"
TEMPLATE_SEED = 42
TEMPLATE_WIDTH = 1024
TEMPLATE_HEIGHT = 1024
TEMPLATE_STEPS = 20
TEMPLATE_CFG_SCALE = 4.0
SUPPORT_EVIDENCE_STATUS = "support_only_not_sampler_parity"
SUPPORT_EVIDENCE_RECORD_IDS = (
    "resume10_lora_identity",
    "resume20_lora_identity",
    "resume10_fast512_cfg1_smoke",
    "resume10_guided512_cfg4_smoke",
    "resume20_fast512_cfg1_smoke",
)
EXPECTED_LORA_TENSOR_COUNT = 432
EXPECTED_STATE_ADAPTER_TENSOR_COUNT = 288
EXPECTED_STATE_MOMENT_TENSOR_COUNT = 576
METRIC_KEYS = ("denoise_seconds_per_step", "vae_decode_seconds", "peak_vram_mib")
COMPARISON_KEYS = ("trajectory", "final_latent", "vae_png")
ONETRAINER_PRODUCER_ARTIFACT_KEYS = (
    "onetrainer_initial_noise_raw_nchw",
    "onetrainer_initial_noise_post_patch_nchw",
    "onetrainer_initial_noise_post_pack",
    "onetrainer_latent_trajectory",
    "onetrainer_final_packed_latent",
    "onetrainer_final_unpacked_latent",
    "onetrainer_final_unscaled_unpatchified_latent",
    "onetrainer_vae_decoded_tensor",
    "onetrainer_png",
)
MOJO_REQUIRED_ARTIFACT_KEYS = (
    "mojo_latent_trajectory",
    "mojo_final_packed_latent",
    "mojo_final_unpacked_latent",
    "mojo_final_unscaled_unpatchified_latent",
    "mojo_vae_decoded_tensor",
    "mojo_png",
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


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


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
    shape_raw = value.get("shape")
    if expected_dtype is not None and dtype != expected_dtype:
        checks.append(Check(False, f"artifact {name} dtype", f"got {dtype!r}, expected {expected_dtype!r}"))
    elif not isinstance(dtype, str) or not dtype:
        checks.append(Check(False, f"artifact {name} dtype", "dtype must be recorded"))
    else:
        checks.append(Check(True, f"artifact {name} dtype", f"dtype={dtype!r}"))
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
    if not finite_number(max_abs) or not finite_number(tolerance):
        return Check(False, f"comparison {key}", "max_abs and tolerance must be finite numeric values")
    return Check(float(max_abs) <= float(tolerance), f"comparison {key}", f"max_abs={max_abs} tolerance={tolerance}")


def check_metric_pair(metrics: dict[str, Any], key: str) -> Check:
    ot = as_dict(metrics, "onetrainer")
    mojo = as_dict(metrics, "mojo")
    ot_value = ot.get(key)
    mojo_value = mojo.get(key)
    if not finite_number(ot_value) or not finite_number(mojo_value):
        return Check(False, f"metric {key}", "onetrainer and mojo values must be finite numeric values")
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
        index_ok = item.get("index") == index
        checks.append(Check(index_ok, f"scheduler.step_trace[{index}].index", f"got={item.get('index')!r}"))
        for key in ("sigma", "sigma_next", "dt", "timestep"):
            checks.append(
                Check(
                    finite_number(item.get(key)),
                    f"scheduler.step_trace[{index}].{key}",
                    str(item.get(key)),
                )
            )
    return checks


def required_manifest_fields() -> list[dict[str, Any]]:
    return [
        {
            "section": "model",
            "fields": [
                {"field": "model_type", "required": EXPECTED_MODEL_TYPE},
            ],
        },
        {
            "section": "prompt",
            "fields": [
                {"field": "prompt.id", "required": "non-empty string"},
                {"field": "prompt.positive", "required": "non-empty string"},
                {"field": "prompt.negative", "required": "non-empty string"},
                {"field": "prompt.seed", "required": "positive integer"},
                {"field": "prompt.width", "required": "positive integer"},
                {"field": "prompt.height", "required": "positive integer"},
                {"field": "prompt.steps", "required": "positive integer"},
                {"field": "prompt.random_seed", "required": False},
                {"field": "prompt.cfg_scale", "required": "numeric"},
            ],
        },
        {
            "section": "scheduler",
            "fields": [
                {"field": "scheduler.name", "required": EXPECTED_SCHEDULER},
                {"field": "scheduler.sigmas", "required": "numeric list, len prompt.steps + 1"},
                {"field": "scheduler.timesteps", "required": "numeric list, len prompt.steps"},
                {"field": "scheduler.mu", "required": "numeric"},
                {
                    "field": "scheduler.step_trace",
                    "required": "prompt.steps entries with index, sigma, sigma_next, dt, timestep",
                },
            ],
        },
        {
            "section": "paired_runtime_metrics",
            "fields": [
                {
                    "field": f"metrics.onetrainer.{key}",
                    "required": "positive numeric",
                }
                for key in METRIC_KEYS
            ]
            + [
                {
                    "field": f"metrics.mojo.{key}",
                    "required": "positive numeric",
                }
                for key in METRIC_KEYS
            ],
        },
        {
            "section": "numeric_comparisons",
            "fields": [
                {
                    "field": f"comparisons.{key}",
                    "required": "accepted=true with numeric max_abs <= tolerance",
                }
                for key in COMPARISON_KEYS
            ],
        },
        {
            "section": "current_split_validation_evidence",
            "fields": [
                {
                    "field": "current_split_validation_evidence.status",
                    "required": SUPPORT_EVIDENCE_STATUS,
                },
                {
                    "field": "current_split_validation_evidence.sampler_parity_accepted",
                    "required": False,
                },
                {
                    "field": "current_split_validation_evidence.smoke_images_accepted_parity",
                    "required": False,
                },
                {
                    "field": "current_split_validation_evidence.records",
                    "required": (
                        "support-only records for fast cfg=1 512 smoke, guided cfg=4 smoke, "
                        "resume10/resume20 LoRA artifact identity, and explicit non-parity scope"
                    ),
                },
            ],
        },
    ]


def required_artifacts() -> list[dict[str, Any]]:
    return [
        {
            "group": "onetrainer_seed_replay_inputs",
            "artifacts": [
                {
                    "name": "onetrainer_initial_noise_raw_nchw",
                    "dtype": "F32",
                    "shape": "[1, 32, prompt.height / 8, prompt.width / 8]",
                },
                {
                    "name": "onetrainer_initial_noise_post_patch_nchw",
                    "dtype": "F32",
                    "shape": "[1, 128, prompt.height / 16, prompt.width / 16]",
                },
                {
                    "name": "onetrainer_initial_noise_post_pack",
                    "dtype": "F32",
                    "shape": "[1, 128, prompt.height / 16, prompt.width / 16] or [tokens, 128]",
                },
            ],
        },
        {
            "group": "paired_latent_trajectory",
            "pairs": [
                {
                    "onetrainer": "onetrainer_latent_trajectory",
                    "mojo": "mojo_latent_trajectory",
                    "shape": "[prompt.steps + 1, tokens, 128] or [prompt.steps + 1, 1, 128, h16, w16]",
                },
                {
                    "onetrainer": "onetrainer_final_packed_latent",
                    "mojo": "mojo_final_packed_latent",
                    "shape": "[1, 128, h16, w16] or [tokens, 128]",
                },
                {
                    "onetrainer": "onetrainer_final_unpacked_latent",
                    "mojo": "mojo_final_unpacked_latent",
                    "shape": "[1, 128, h16, w16]",
                },
                {
                    "onetrainer": "onetrainer_final_unscaled_unpatchified_latent",
                    "mojo": "mojo_final_unscaled_unpatchified_latent",
                    "shape": "[1, 32, h8, w8]",
                },
            ],
        },
        {
            "group": "paired_decode_and_png",
            "pairs": [
                {
                    "onetrainer": "onetrainer_vae_decoded_tensor",
                    "mojo": "mojo_vae_decoded_tensor",
                    "shape": "[1, 3, prompt.height, prompt.width]",
                },
                {
                    "onetrainer": "onetrainer_png",
                    "mojo": "mojo_png",
                    "shape": "PNG dimensions prompt.width x prompt.height",
                },
            ],
        },
    ]


def required_paired_sampler_fields() -> list[dict[str, Any]]:
    return [
        {
            "group": "shared_run_identity",
            "required": "same OneTrainer/Mojo sample configuration",
            "fields": [
                "model_type",
                "prompt.id",
                "prompt.positive",
                "prompt.negative",
                "prompt.seed",
                "prompt.width",
                "prompt.height",
                "prompt.steps",
                "prompt.random_seed",
                "prompt.cfg_scale",
                "scheduler.name",
                "scheduler.sigmas",
                "scheduler.timesteps",
                "scheduler.mu",
                "scheduler.step_trace",
            ],
        },
        {
            "group": "paired_runtime_metrics",
            "required": "positive numeric OneTrainer and Mojo values",
            "fields": [
                "denoise_seconds_per_step",
                "vae_decode_seconds",
                "peak_vram_mib",
            ],
        },
        {
            "group": "paired_numeric_comparisons",
            "required": "accepted=true with numeric max_abs <= tolerance",
            "fields": [
                "trajectory",
                "final_latent",
                "vae_png",
            ],
        },
    ]


def current_split_validation_evidence() -> dict[str, Any]:
    """Document current split evidence without promoting it to sampler parity."""
    return {
        "status": SUPPORT_EVIDENCE_STATUS,
        "sampler_parity_accepted": False,
        "smoke_images_accepted_parity": False,
        "note": (
            "These are current product smoke and artifact-identity records. "
            "They do not replace the paired OneTrainer/Mojo sampler manifest, "
            "trajectory comparison, VAE/PNG numeric comparison, or speed/VRAM parity."
        ),
        "records": [
            {
                "id": "resume10_lora_identity",
                "kind": "lora_artifact_identity",
                "evidence_scope": "artifact_identity_not_sampler_parity",
                "parity_accepted": False,
                "config": "serenitymojo/configs/klein9b_cpu_offloaded_resume10_smoke.json",
                "lora_artifact": "/tmp/klein9b_cpu_offloaded_resume10_smoke.safetensors",
                "state_artifact": "/tmp/klein9b_cpu_offloaded_resume10_smoke.safetensors.state.safetensors",
                "resume_from": "/tmp/klein9b_cpu_offloaded_5step_smoke.safetensors",
                "global_steps": [6, 7, 8, 9, 10],
                "lora_tensor_count": EXPECTED_LORA_TENSOR_COUNT,
                "lora_dtype": "BF16",
                "state_adapter_tensor_count": EXPECTED_STATE_ADAPTER_TENSOR_COUNT,
                "state_adapter_dtype": "BF16",
                "state_moment_tensor_count": EXPECTED_STATE_MOMENT_TENSOR_COUNT,
                "state_moment_dtype": "F32",
                "guard": "scripts/check_klein_product_smoke_artifacts.py",
            },
            {
                "id": "resume20_lora_identity",
                "kind": "lora_artifact_identity",
                "evidence_scope": "artifact_identity_not_sampler_parity",
                "parity_accepted": False,
                "config": "serenitymojo/configs/klein9b_cpu_offloaded_resume20_smoke.json",
                "lora_artifact": "/tmp/klein9b_cpu_offloaded_resume20_smoke.safetensors",
                "state_artifact": "/tmp/klein9b_cpu_offloaded_resume20_smoke.safetensors.state.safetensors",
                "resume_from": "/tmp/klein9b_cpu_offloaded_resume10_smoke.safetensors",
                "global_steps": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "lora_tensor_count": EXPECTED_LORA_TENSOR_COUNT,
                "lora_dtype": "BF16",
                "state_adapter_tensor_count": EXPECTED_STATE_ADAPTER_TENSOR_COUNT,
                "state_adapter_dtype": "BF16",
                "state_moment_tensor_count": EXPECTED_STATE_MOMENT_TENSOR_COUNT,
                "state_moment_dtype": "F32",
                "guard": "scripts/check_klein_product_smoke_artifacts.py",
            },
            {
                "id": "resume10_fast512_cfg1_smoke",
                "kind": "sampler_smoke_image",
                "evidence_scope": "fast cfg=1 512 smoke; not accepted parity",
                "parity_accepted": False,
                "quality_accepted": False,
                "speed_parity_accepted": False,
                "config": "serenitymojo/configs/klein9b_alina_samples_fast512.json",
                "lora_artifact": "/tmp/klein9b_cpu_offloaded_resume10_smoke.safetensors",
                "image_path": "output/alina_train/klein_lora_resume10_fast512_cfg1.png",
                "width": 512,
                "height": 512,
                "steps": 1,
                "cfg_scale": 1.0,
                "denoise_seconds_per_step": 3.1,
            },
            {
                "id": "resume10_guided512_cfg4_smoke",
                "kind": "sampler_smoke_image",
                "evidence_scope": "guided cfg=4 smoke; not accepted parity",
                "parity_accepted": False,
                "quality_accepted": False,
                "speed_parity_accepted": False,
                "config": "serenitymojo/configs/klein9b_alina_samples_fast512.json",
                "lora_artifact": "/tmp/klein9b_cpu_offloaded_resume10_smoke.safetensors",
                "image_path": "output/alina_train/klein_lora_resume10_fast512.png",
                "width": 512,
                "height": 512,
                "steps": 1,
                "cfg_scale": 4.0,
                "denoise_seconds_per_step": 24.5,
            },
            {
                "id": "resume20_fast512_cfg1_smoke",
                "kind": "sampler_smoke_image",
                "evidence_scope": "fast cfg=1 512 smoke; not accepted parity",
                "parity_accepted": False,
                "quality_accepted": False,
                "speed_parity_accepted": False,
                "config": "serenitymojo/configs/klein9b_alina_samples_fast512.json",
                "lora_artifact": "/tmp/klein9b_cpu_offloaded_resume20_smoke.safetensors",
                "image_path": "output/alina_train/klein_lora_resume20_fast512_cfg1.png",
                "width": 512,
                "height": 512,
                "steps": 1,
                "cfg_scale": 1.0,
                "denoise_seconds_per_step": 3.1,
            },
        ],
        "remaining_sampler_parity_requirements": [
            "real paired OneTrainer/Mojo sampler manifest",
            "shared prompt, seed, resolution, steps, cfg, scheduler, and dtype identity",
            "OneTrainer-equivalent raw/post-patch/post-pack initial-noise artifacts",
            "paired latent trajectory and final latent numeric comparisons",
            "paired VAE tensor and final PNG numeric comparisons",
            "matched denoise/VAE timing and peak VRAM evidence",
        ],
    }


def _records_by_id(records: list[Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = record.get("id")
        if isinstance(record_id, str):
            out[record_id] = record
    return out


def check_current_split_validation_evidence(data: dict[str, Any]) -> list[Check]:
    expected = _records_by_id(current_split_validation_evidence()["records"])
    value = data.get("current_split_validation_evidence")
    if not isinstance(value, dict):
        return [
            Check(
                False,
                "current_split_validation_evidence",
                "must be an object documenting support-only smoke/artifact evidence",
            )
        ]

    checks = [
        Check(
            value.get("status") == SUPPORT_EVIDENCE_STATUS,
            "current_split_validation_evidence.status",
            f"got={value.get('status')!r} expected={SUPPORT_EVIDENCE_STATUS!r}",
        ),
        Check(
            value.get("sampler_parity_accepted") is False,
            "current_split_validation_evidence.sampler_parity_accepted",
            "must be false; smoke/artifact identity is support evidence only",
        ),
        Check(
            value.get("smoke_images_accepted_parity") is False,
            "current_split_validation_evidence.smoke_images_accepted_parity",
            "must be false; smoke images are not accepted parity",
        ),
    ]

    records_raw = value.get("records")
    if not isinstance(records_raw, list):
        checks.append(Check(False, "current_split_validation_evidence.records", "must be a list"))
        return checks
    records = _records_by_id(records_raw)
    missing = [record_id for record_id in SUPPORT_EVIDENCE_RECORD_IDS if record_id not in records]
    checks.append(
        Check(
            not missing,
            "current_split_validation_evidence.records.ids",
            f"missing={missing}",
        )
    )

    for record_id in SUPPORT_EVIDENCE_RECORD_IDS:
        record = records.get(record_id)
        expected_record = expected[record_id]
        if record is None:
            continue
        checks.append(
            Check(
                record.get("kind") == expected_record["kind"],
                f"current_split_validation_evidence.{record_id}.kind",
                f"got={record.get('kind')!r}",
            )
        )
        checks.append(
            Check(
                record.get("parity_accepted") is False,
                f"current_split_validation_evidence.{record_id}.parity_accepted",
                "must be false",
            )
        )
        scope = record.get("evidence_scope")
        checks.append(
            Check(
                isinstance(scope, str) and "not" in scope and "parity" in scope,
                f"current_split_validation_evidence.{record_id}.scope",
                str(scope),
            )
        )

        if expected_record["kind"] == "sampler_smoke_image":
            for key in ("width", "height", "steps"):
                checks.append(
                    Check(
                        record.get(key) == expected_record[key],
                        f"current_split_validation_evidence.{record_id}.{key}",
                        f"got={record.get(key)!r} expected={expected_record[key]!r}",
                    )
                )
            cfg_scale = record.get("cfg_scale")
            checks.append(
                Check(
                    finite_number(cfg_scale) and abs(float(cfg_scale) - float(expected_record["cfg_scale"])) < 1.0e-9,
                    f"current_split_validation_evidence.{record_id}.cfg_scale",
                    f"got={cfg_scale!r} expected={expected_record['cfg_scale']!r}",
                )
            )
            checks.append(
                Check(
                    record.get("quality_accepted") is False,
                    f"current_split_validation_evidence.{record_id}.quality_accepted",
                    "must be false",
                )
            )
            checks.append(
                Check(
                    record.get("speed_parity_accepted") is False,
                    f"current_split_validation_evidence.{record_id}.speed_parity_accepted",
                    "must be false",
                )
            )
            for key in ("config", "lora_artifact", "image_path"):
                checks.append(
                    Check(
                        isinstance(record.get(key), str) and bool(record.get(key)),
                        f"current_split_validation_evidence.{record_id}.{key}",
                        str(record.get(key)),
                    )
                )
        else:
            for key in (
                "lora_tensor_count",
                "state_adapter_tensor_count",
                "state_moment_tensor_count",
            ):
                checks.append(
                    Check(
                        record.get(key) == expected_record[key],
                        f"current_split_validation_evidence.{record_id}.{key}",
                        f"got={record.get(key)!r} expected={expected_record[key]!r}",
                    )
                )
            for key in ("lora_dtype", "state_adapter_dtype", "state_moment_dtype"):
                checks.append(
                    Check(
                        record.get(key) == expected_record[key],
                        f"current_split_validation_evidence.{record_id}.{key}",
                        f"got={record.get(key)!r} expected={expected_record[key]!r}",
                    )
                )
            for key in ("config", "lora_artifact", "state_artifact", "guard"):
                checks.append(
                    Check(
                        isinstance(record.get(key), str) and bool(record.get(key)),
                        f"current_split_validation_evidence.{record_id}.{key}",
                        str(record.get(key)),
                    )
                )

    requirements = value.get("remaining_sampler_parity_requirements")
    checks.append(
        Check(
            isinstance(requirements, list)
            and any("paired OneTrainer/Mojo sampler manifest" in str(item) for item in requirements),
            "current_split_validation_evidence.remaining_sampler_parity_requirements",
            "must retain real paired OT/Mojo manifest blocker",
        )
    )
    return checks


def is_onetrainer_producer_manifest(data: dict[str, Any]) -> bool:
    producer = data.get("producer")
    scope = data.get("scope")
    return producer == ONETRAINER_PRODUCER or (
        isinstance(scope, str) and "OneTrainer-only Klein/Flux2 sampler artifact producer" in scope
    )


def check_onetrainer_producer_fragment(data: dict[str, Any]) -> list[Check]:
    """Validate the OT-only fragment without promoting it to paired parity."""
    if not is_onetrainer_producer_manifest(data):
        return []

    checks: list[Check] = [
        Check(
            True,
            "onetrainer producer fragment",
            "OneTrainer-only producer output detected; validating fragment readiness and non-parity flags.",
        )
    ]

    for key in ("accepted", "parity_claimed", "sampler_parity_accepted", "mojo_comparison_present"):
        checks.append(
            Check(
                data.get(key) is False,
                f"onetrainer producer flag {key}",
                f"got={data.get(key)!r}; producer fragments must not claim parity",
            )
        )

    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        checks.append(Check(False, "onetrainer producer artifacts", "artifacts must be an object"))
    else:
        missing = [key for key in ONETRAINER_PRODUCER_ARTIFACT_KEYS if key not in artifacts]
        checks.append(
            Check(
                not missing,
                "onetrainer producer artifacts.present",
                f"missing={missing}",
            )
        )
        mojo_present = [key for key in MOJO_REQUIRED_ARTIFACT_KEYS if key in artifacts]
        checks.append(
            Check(
                not mojo_present,
                "onetrainer producer artifacts.mojo_absent",
                f"mojo_artifacts_present={mojo_present}; producer output must stay OT-only",
            )
        )
        for key in ONETRAINER_PRODUCER_ARTIFACT_KEYS:
            value = artifacts.get(key)
            if not isinstance(value, dict):
                checks.append(Check(False, f"onetrainer producer artifact {key}", "must be an object"))
                continue
            path = value.get("path")
            checks.append(
                Check(
                    isinstance(path, str) and bool(path),
                    f"onetrainer producer artifact {key}.path",
                    str(path),
                )
            )
            if key.endswith("_png"):
                checks.append(
                    Check(
                        isinstance(value.get("width"), int) and isinstance(value.get("height"), int),
                        f"onetrainer producer artifact {key}.dimensions",
                        f"{value.get('width')}x{value.get('height')}",
                    )
                )
            else:
                checks.append(
                    Check(
                        value.get("dtype") == "F32",
                        f"onetrainer producer artifact {key}.dtype",
                        f"got={value.get('dtype')!r} expected='F32'",
                    )
                )
                shape = value.get("shape")
                checks.append(
                    Check(
                        isinstance(shape, list) and all(isinstance(dim, int) and dim > 0 for dim in shape),
                        f"onetrainer producer artifact {key}.shape",
                        str(shape),
                    )
                )

    groups = data.get("artifact_groups")
    if not isinstance(groups, dict):
        checks.append(Check(False, "onetrainer producer artifact_groups", "artifact_groups must be an object"))
    else:
        missing_for_parity = groups.get("missing_for_sampler_parity")
        required_missing = (*MOJO_REQUIRED_ARTIFACT_KEYS, "numeric_comparisons")
        missing = [
            key
            for key in required_missing
            if not isinstance(missing_for_parity, list) or key not in missing_for_parity
        ]
        checks.append(
            Check(
                not missing,
                "onetrainer producer artifact_groups.missing_for_sampler_parity",
                f"missing={missing}",
            )
        )

    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        checks.append(Check(False, "onetrainer producer metrics", "metrics must be an object"))
    else:
        onetrainer = metrics.get("onetrainer")
        mojo = metrics.get("mojo")
        for namespace, values in (("onetrainer", onetrainer), ("mojo", mojo)):
            if not isinstance(values, dict):
                checks.append(Check(False, f"onetrainer producer metrics.{namespace}", "must be an object"))
                continue
            for key in METRIC_KEYS:
                value = values.get(key)
                if namespace == "onetrainer":
                    checks.append(
                        Check(
                            finite_number(value) and float(value) > 0.0,
                            f"onetrainer producer metric {namespace}.{key}",
                            f"value={value!r}",
                        )
                    )
                else:
                    checks.append(
                        Check(
                            finite_number(value) and float(value) == 0.0,
                            f"onetrainer producer metric {namespace}.{key}",
                            f"value={value!r}; must remain zero until Mojo run",
                        )
                    )

    comparisons = data.get("comparisons")
    if not isinstance(comparisons, dict):
        checks.append(Check(False, "onetrainer producer comparisons", "comparisons must be an object"))
    else:
        for key in COMPARISON_KEYS:
            value = comparisons.get(key)
            checks.append(
                Check(
                    isinstance(value, dict)
                    and value.get("accepted") is False
                    and value.get("max_abs") is None
                    and value.get("tolerance") is None,
                    f"onetrainer producer comparison {key}",
                    str(value),
                )
            )

    return checks


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
        "onetrainer_seed_replay_inputs": [
            "onetrainer_initial_noise_raw_nchw",
            "onetrainer_initial_noise_post_patch_nchw",
            "onetrainer_initial_noise_post_pack",
        ],
        "paired_latent_trajectory": [
            {
                "onetrainer": "onetrainer_latent_trajectory",
                "mojo": "mojo_latent_trajectory",
            },
            {
                "onetrainer": "onetrainer_final_packed_latent",
                "mojo": "mojo_final_packed_latent",
            },
            {
                "onetrainer": "onetrainer_final_unpacked_latent",
                "mojo": "mojo_final_unpacked_latent",
            },
            {
                "onetrainer": "onetrainer_final_unscaled_unpatchified_latent",
                "mojo": "mojo_final_unscaled_unpatchified_latent",
            },
        ],
        "paired_decode_and_png": [
            {
                "onetrainer": "onetrainer_vae_decoded_tensor",
                "mojo": "mojo_vae_decoded_tensor",
            },
            {
                "onetrainer": "onetrainer_png",
                "mojo": "mojo_png",
            },
        ],
    }


def build_template_manifest(
    *,
    target_manifest: Path = DEFAULT_MANIFEST,
    artifact_root: str = "/tmp/klein_sampler_parity_artifacts",
    width: int = TEMPLATE_WIDTH,
    height: int = TEMPLATE_HEIGHT,
    steps: int = TEMPLATE_STEPS,
    prompt_id: str = TEMPLATE_PROMPT_ID,
    positive: str = TEMPLATE_PROMPT_POSITIVE,
    negative: str = TEMPLATE_PROMPT_NEGATIVE,
    seed: int = TEMPLATE_SEED,
    random_seed: bool = False,
    cfg_scale: float = TEMPLATE_CFG_SCALE,
) -> dict[str, Any]:
    if width <= 0 or height <= 0 or steps <= 0:
        raise ValueError("template width/height/steps must be positive")
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError("template width/height must be divisible by 16 for Klein packing")

    raw_h = height // 8
    raw_w = width // 8
    latent_h = height // 16
    latent_w = width // 16
    n_img = latent_h * latent_w
    trajectory_shape = (steps + 1, n_img, 128)
    packed_shape = (1, 128, latent_h, latent_w)
    unpacked_shape = (1, 128, latent_h, latent_w)
    unscaled_shape = (1, 32, raw_h, raw_w)
    decoded_shape = (1, 3, height, width)
    artifacts = {
        "onetrainer_initial_noise_raw_nchw": _template_tensor_artifact(
            artifact_root,
            "onetrainer_initial_noise_raw_nchw",
            (1, 32, raw_h, raw_w),
            dtype="F32",
        ),
        "onetrainer_initial_noise_post_patch_nchw": _template_tensor_artifact(
            artifact_root,
            "onetrainer_initial_noise_post_patch_nchw",
            unpacked_shape,
            dtype="F32",
        ),
        "onetrainer_initial_noise_post_pack": _template_tensor_artifact(
            artifact_root,
            "onetrainer_initial_noise_post_pack",
            packed_shape,
            dtype="F32",
        ),
        "onetrainer_latent_trajectory": _template_tensor_artifact(
            artifact_root,
            "onetrainer_latent_trajectory",
            trajectory_shape,
            dtype="F32",
        ),
        "mojo_latent_trajectory": _template_tensor_artifact(
            artifact_root,
            "mojo_latent_trajectory",
            trajectory_shape,
            dtype="TODO_FILL_STORAGE_DTYPE",
        ),
        "onetrainer_final_packed_latent": _template_tensor_artifact(
            artifact_root,
            "onetrainer_final_packed_latent",
            packed_shape,
            dtype="F32",
        ),
        "mojo_final_packed_latent": _template_tensor_artifact(
            artifact_root,
            "mojo_final_packed_latent",
            packed_shape,
            dtype="TODO_FILL_STORAGE_DTYPE",
        ),
        "onetrainer_final_unpacked_latent": _template_tensor_artifact(
            artifact_root,
            "onetrainer_final_unpacked_latent",
            unpacked_shape,
            dtype="F32",
        ),
        "mojo_final_unpacked_latent": _template_tensor_artifact(
            artifact_root,
            "mojo_final_unpacked_latent",
            unpacked_shape,
            dtype="TODO_FILL_STORAGE_DTYPE",
        ),
        "onetrainer_final_unscaled_unpatchified_latent": _template_tensor_artifact(
            artifact_root,
            "onetrainer_final_unscaled_unpatchified_latent",
            unscaled_shape,
            dtype="F32",
        ),
        "mojo_final_unscaled_unpatchified_latent": _template_tensor_artifact(
            artifact_root,
            "mojo_final_unscaled_unpatchified_latent",
            unscaled_shape,
            dtype="TODO_FILL_STORAGE_DTYPE",
        ),
        "onetrainer_vae_decoded_tensor": _template_tensor_artifact(
            artifact_root,
            "onetrainer_vae_decoded_tensor",
            decoded_shape,
            dtype="F32",
        ),
        "mojo_vae_decoded_tensor": _template_tensor_artifact(
            artifact_root,
            "mojo_vae_decoded_tensor",
            decoded_shape,
            dtype="TODO_FILL_STORAGE_DTYPE",
        ),
        "onetrainer_png": _template_png_artifact(artifact_root, "onetrainer_png", width, height),
        "mojo_png": _template_png_artifact(artifact_root, "mojo_png", width, height),
    }

    return {
        "schema_version": 1,
        "producer": "scripts/check_klein_sampler_artifact_manifest.py --write-template",
        "template": True,
        "template_note": "Fill every TODO path/value with real paired OneTrainer/Mojo evidence before strict use.",
        "target_manifest": str(target_manifest),
        "model_type": EXPECTED_MODEL_TYPE,
        "parity_claimed": False,
        "parity_note": "Template only; this is not sampler parity evidence.",
        "prompt": {
            "id": prompt_id,
            "positive": positive,
            "negative": negative,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "random_seed": random_seed,
            "cfg_scale": cfg_scale,
        },
        "scheduler": {
            "name": EXPECTED_SCHEDULER,
            "sigmas": _todo_series("SIGMA", steps + 1),
            "timesteps": _todo_series("TIMESTEP", steps),
            "mu": "TODO_NUMERIC_MU",
            "step_trace": _template_step_trace(steps),
        },
        "artifact_groups": _template_artifact_groups(),
        "artifacts": artifacts,
        "metrics": {
            "onetrainer": {key: 0.0 for key in METRIC_KEYS},
            "mojo": {key: 0.0 for key in METRIC_KEYS},
        },
        "comparisons": {
            key: {
                "accepted": False,
                "max_abs": None,
                "tolerance": None,
                "note": "Fill with real numeric comparison and set accepted true only when max_abs <= tolerance.",
            }
            for key in COMPARISON_KEYS
        },
        "current_split_validation_evidence": current_split_validation_evidence(),
        "required_manifest_fields": required_manifest_fields(),
        "required_paired_sampler_fields": required_paired_sampler_fields(),
        "required_paired_sampler_artifacts": required_artifacts(),
    }


def write_template_manifest(
    path: Path,
    target_manifest: Path,
    *,
    artifact_root: str,
    width: int,
    height: int,
    steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            build_template_manifest(
                target_manifest=target_manifest,
                artifact_root=artifact_root,
                width=width,
                height=height,
                steps=steps,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[klein-sampler-manifest] INFO wrote template manifest: {path}")


def inspect_manifest(path: Path) -> list[Check]:
    if not path.exists():
        return [
            Check(
                False,
                "manifest",
                f"missing {path}; expected paired OT/Mojo sampler evidence manifest",
            )
        ]

    data = load_json(path)
    checks: list[Check] = [Check(True, "manifest", f"loaded {path}")]
    model_type = data.get("model_type")
    checks.append(Check(model_type == EXPECTED_MODEL_TYPE, "model_type", f"got={model_type!r} expected={EXPECTED_MODEL_TYPE!r}"))
    checks.extend(check_onetrainer_producer_fragment(data))

    prompt = as_dict(data, "prompt")
    for key in ("id", "positive", "negative"):
        checks.append(Check(isinstance(prompt.get(key), str) and bool(prompt.get(key)), f"prompt.{key}", str(prompt.get(key))))
    for key in ("seed", "width", "height", "steps"):
        checks.append(Check(isinstance(prompt.get(key), int) and int(prompt.get(key)) > 0, f"prompt.{key}", str(prompt.get(key))))
    checks.append(Check(prompt.get("random_seed") is False, "prompt.random_seed", "must be false for deterministic OT replay"))
    checks.append(Check(finite_number(prompt.get("cfg_scale")), "prompt.cfg_scale", str(prompt.get("cfg_scale"))))

    width = int(prompt.get("width", 0)) if isinstance(prompt.get("width"), int) else 0
    height = int(prompt.get("height", 0)) if isinstance(prompt.get("height"), int) else 0
    steps = int(prompt.get("steps", 0)) if isinstance(prompt.get("steps"), int) else 0
    checks.append(Check(width % 16 == 0 and height % 16 == 0, "prompt.resolution_packable", f"{width}x{height}"))
    latent_h = height // 16 if height > 0 else 0
    latent_w = width // 16 if width > 0 else 0
    raw_h = height // 8 if height > 0 else 0
    raw_w = width // 8 if width > 0 else 0
    n_img = latent_h * latent_w
    trajectory_shapes = (
        (steps + 1, n_img, 128),
        (steps + 1, 1, 128, latent_h, latent_w),
    )

    scheduler = as_dict(data, "scheduler")
    scheduler_name = scheduler.get("name")
    checks.append(Check(scheduler_name == EXPECTED_SCHEDULER, "scheduler.name", f"got={scheduler_name!r}"))
    sigmas = as_list(scheduler, "sigmas")
    checks.append(Check(len(sigmas) == steps + 1, "scheduler.sigmas", f"len={len(sigmas)} expected={steps + 1}"))
    checks.append(Check(all(finite_number(value) for value in sigmas), "scheduler.sigmas.numeric", "all finite numeric"))
    timesteps = as_list(scheduler, "timesteps")
    checks.append(Check(len(timesteps) == steps, "scheduler.timesteps", f"len={len(timesteps)} expected={steps}"))
    checks.append(
        Check(all(finite_number(value) for value in timesteps), "scheduler.timesteps.numeric", "all finite numeric")
    )
    checks.append(Check(finite_number(scheduler.get("mu")), "scheduler.mu", str(scheduler.get("mu"))))
    checks.extend(check_step_trace(scheduler, steps))

    artifacts = as_dict(data, "artifacts")
    checks.extend(
        check_tensor_artifact(
            artifacts,
            "onetrainer_initial_noise_raw_nchw",
            expected_dtype="F32",
            allowed_shapes=((1, 32, raw_h, raw_w),),
        )
    )
    checks.extend(
        check_tensor_artifact(
            artifacts,
            "onetrainer_initial_noise_post_patch_nchw",
            expected_dtype="F32",
            allowed_shapes=((1, 128, latent_h, latent_w),),
        )
    )
    checks.extend(
        check_tensor_artifact(
            artifacts,
            "onetrainer_initial_noise_post_pack",
            expected_dtype="F32",
            allowed_shapes=((1, 128, latent_h, latent_w), (n_img, 128)),
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
    checks.extend(
        check_tensor_artifact(
            artifacts,
            "onetrainer_final_packed_latent",
            expected_dtype="F32",
            allowed_shapes=((1, 128, latent_h, latent_w), (n_img, 128)),
        )
    )
    checks.extend(
        check_tensor_artifact(
            artifacts,
            "mojo_final_packed_latent",
            allowed_shapes=((1, 128, latent_h, latent_w), (n_img, 128)),
        )
    )
    for prefix in ("onetrainer", "mojo"):
        checks.extend(
            check_tensor_artifact(
                artifacts,
                f"{prefix}_final_unpacked_latent",
                allowed_shapes=((1, 128, latent_h, latent_w),),
            )
        )
        checks.extend(
            check_tensor_artifact(
                artifacts,
                f"{prefix}_final_unscaled_unpatchified_latent",
                allowed_shapes=((1, 32, raw_h, raw_w),),
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
    for key in METRIC_KEYS:
        checks.append(check_metric_pair(metrics, key))

    comparisons = as_dict(data, "comparisons")
    for key in COMPARISON_KEYS:
        checks.append(check_numeric_pair(comparisons, key))
    checks.extend(check_current_split_validation_evidence(data))
    return checks


def print_report(path: Path, checks: list[Check]) -> None:
    print("[klein-sampler-manifest] scope=no-CUDA metadata/header gate")
    print(f"[klein-sampler-manifest] manifest={path}")
    for check in checks:
        status = "PASS" if check.ok else "BLOCKED"
        print(f"[klein-sampler-manifest] {status} {check.label}: {check.detail}")
    blockers = [check for check in checks if not check.ok]
    print(f"[klein-sampler-manifest] blockers={len(blockers)}")


def check_as_json(check: Check) -> dict[str, Any]:
    return {"ok": check.ok, "label": check.label, "detail": check.detail}


def is_producer_fragment_check(check: Check) -> bool:
    return check.label.startswith("onetrainer producer")


def build_readiness_report(path: Path, checks: list[Check]) -> dict[str, Any]:
    blockers = [check_as_json(check) for check in checks if not check.ok]
    producer_checks = [check for check in checks if is_producer_fragment_check(check)]
    return {
        "schema_version": 1,
        "producer": "scripts/check_klein_sampler_artifact_manifest.py",
        "scope": "Klein/Flux2 sampler manifest readiness; no CUDA, no denoise, no VAE run, no image comparison",
        "manifest": str(path),
        "strict_manifest_ready": not blockers,
        "onetrainer_producer_fragment_ready": bool(producer_checks) and all(check.ok for check in producer_checks),
        "parity_claimed": False,
        "parity_note": "This readiness report only lists manifest evidence and blockers; it does not accept sampler parity.",
        "current_split_validation_evidence": current_split_validation_evidence(),
        "required_manifest_fields": required_manifest_fields(),
        "required_paired_sampler_fields": required_paired_sampler_fields(),
        "required_paired_sampler_artifacts": required_artifacts(),
        "onetrainer_producer_fragment_checks": [check_as_json(check) for check in producer_checks],
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
    print(f"[klein-sampler-manifest] INFO wrote readiness report: {path}")


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
        help="Write a fill-in skeleton for the paired Klein sampler parity manifest.",
    )
    parser.add_argument("--template-artifact-root", default="/tmp/klein_sampler_parity_artifacts")
    parser.add_argument("--template-width", type=int, default=TEMPLATE_WIDTH)
    parser.add_argument("--template-height", type=int, default=TEMPLATE_HEIGHT)
    parser.add_argument("--template-steps", type=int, default=TEMPLATE_STEPS)
    args = parser.parse_args()

    try:
        checks = inspect_manifest(args.manifest)
    except Exception as exc:  # noqa: BLE001 - keep this a guard, not a traceback.
        checks = [Check(False, "manifest", str(exc))]

    if args.write_template is not None:
        write_template_manifest(
            args.write_template,
            args.manifest,
            artifact_root=args.template_artifact_root,
            width=args.template_width,
            height=args.template_height,
            steps=args.template_steps,
        )

    if args.write_readiness is not None:
        write_readiness_report(args.write_readiness, args.manifest, checks)

    if args.json:
        print(
            json.dumps(
                {
                    "producer": "scripts/check_klein_sampler_artifact_manifest.py",
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
