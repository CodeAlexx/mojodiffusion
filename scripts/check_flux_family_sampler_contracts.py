#!/usr/bin/env python3
"""Static guard for Flux-family OneTrainer sampler contracts.

This is a report-first source guard. It verifies that the local OneTrainer
sampler/setup files still expose the contracts mirrored by the Mojo sampler
helpers for Flux.1-dev, Flux2 dev, Flux2/Klein, and Chroma1-HD.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ONETRAINER = Path("/home/alex/OneTrainer")

OT_SAMPLERS = ONETRAINER / "modules/modelSampler"
OT_SETUPS = ONETRAINER / "modules/modelSetup"
OT_MODELS = ONETRAINER / "modules/model"
OT_CONFIGS = ONETRAINER / "configs"

MOJO_SAMPLING = REPO / "serenitymojo/sampling"
MOJO_TRAINING = REPO / "serenitymojo/training"
MOJO_PIPELINE = REPO / "serenitymojo/pipeline"
MOJO_DOCS = REPO / "serenitymojo/docs"
PORT_DOC = REPO / "OT_MOJO_PORT_REMAINING.md"

PARITY_DIR = Path("/home/alex/onetrainer-mojo/parity")
FLUX2_DEV_CONTRACT_JSON = PARITY_DIR / "flux2_dev_train_ref_contract.json"
FLUX2_DEV_BLOCKERS_JSON = PARITY_DIR / "flux2_dev_train_ref_blockers.json"
KLEIN_SAMPLER_MANIFEST = PARITY_DIR / "klein_sampler_parity_manifest.json"
FLUX1_SAMPLER_MANIFEST = PARITY_DIR / "flux1_sampler_parity_manifest.json"
CHROMA_SAMPLER_MANIFEST = PARITY_DIR / "chroma_sampler_parity_manifest.json"

FLUX2_DEV_REQUIRED_REFERENCE_TENSORS = (
    "flux2_dev_train_ref_meta.json",
    "flux2_dev_train_ref_step000.safetensors",
    "flux2_dev_train_ref_step000_adapters.safetensors",
)

FLUX1_CLAIM_ALIASES = (
    "flux1",
    "flux.1",
    "flux_1",
    "flux1_dev",
    "flux1-dev",
    "flux dev 1",
    "flux_dev_1",
    "flux.1-dev",
)
FLUX2_KLEIN_CLAIM_ALIASES = (
    "flux2",
    "flux 2",
    "flux.2",
    "flux_2",
    "flux2_klein",
    "flux2 klein",
    "flux.2 klein",
    "flux_2_klein",
    "klein",
)

SPEED_PARITY_CLAIM_MARKERS = (
    "speed_parity_accepted",
    "sampler speed parity accepted",
    "accepted sampler speed parity",
    "speed parity accepted",
    "sampler_speed_parity_accepted",
    "speed parity: accepted",
)

NEGATIVE_CLAIM_MARKERS = (
    "not accepted",
    "not a speed-parity claim",
    "no model has accepted",
    "cannot be accepted",
    "must not",
    "not image or speed parity",
    "not image or speed-parity",
    "not an image sampler",
    "scaffold",
    "blocked",
)

FLUX1_SPEED_EVIDENCE_GROUPS = (
    (
        "OneTrainer seconds/step",
        ("onetrainer seconds/step", "ot_baseline_seconds_per_step", "ot_seconds_per_step"),
    ),
    ("Mojo seconds/step", ("mojo seconds/step", "mojo_seconds_per_step")),
    ("OneTrainer peak VRAM", ("onetrainer peak vram", "ot_peak_vram_mib")),
    ("Mojo peak VRAM", ("mojo peak vram", "mojo_peak_vram_mib")),
    ("prompt", ("prompt",)),
    ("seed", ("seed",)),
    ("resolution", ("resolution", "width", "height")),
    ("steps", ("steps", "diffusion_steps", "sample_steps")),
    ("cfg", ("cfg", "cfg_scale", "guidance_scale")),
    ("dtype", ("dtype", "train_dtype")),
    (
        "denoise trajectory",
        ("denoise trajectory", "denoise_trajectory", "latent trajectory", "trajectory"),
    ),
)
FLUX2_KLEIN_SPEED_EVIDENCE_GROUPS = FLUX1_SPEED_EVIDENCE_GROUPS

SAMPLER_ARTIFACT_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "Klein/Flux2": {
        "manifest": KLEIN_SAMPLER_MANIFEST,
        "nearby_non_sampler_files": (
            PARITY_DIR / "klein_train_ref_meta.json",
            PARITY_DIR / "klein_train_ref_step000.safetensors",
            PARITY_DIR / "klein_train_ref_step000_adapters.safetensors",
            PARITY_DIR / "klein_fwd_meta.json",
            PARITY_DIR / "klein_fwd.safetensors",
        ),
        "prompt_keys": (
            "id",
            "positive",
            "negative",
            "seed",
            "width",
            "height",
            "steps",
            "random_seed",
            "cfg_scale",
        ),
        "scheduler_keys": ("name", "sigmas", "timesteps", "mu", "step_trace"),
        "artifact_keys": (
            "onetrainer_initial_noise_raw_nchw",
            "onetrainer_initial_noise_post_patch_nchw",
            "onetrainer_initial_noise_post_pack",
            "onetrainer_latent_trajectory",
            "mojo_latent_trajectory",
            "onetrainer_final_packed_latent",
            "mojo_final_packed_latent",
            "onetrainer_final_unpacked_latent",
            "onetrainer_final_unscaled_unpatchified_latent",
            "onetrainer_vae_decoded_tensor",
            "mojo_final_unpacked_latent",
            "mojo_final_unscaled_unpatchified_latent",
            "mojo_vae_decoded_tensor",
            "onetrainer_png",
            "mojo_png",
        ),
        "metric_keys": (
            "denoise_seconds_per_step",
            "vae_decode_seconds",
            "peak_vram_mib",
        ),
        "comparison_keys": ("trajectory", "final_latent", "vae_png"),
    },
    "Flux.1-dev": {
        "manifest": FLUX1_SAMPLER_MANIFEST,
        "nearby_non_sampler_files": (),
        "prompt_keys": (
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
        ),
        "scheduler_keys": (
            "name",
            "sigmas",
            "timesteps",
            "mu",
            "dynamic_shift",
            "step_trace",
        ),
        "artifact_keys": (
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
        ),
        "metric_keys": (
            "text_seconds",
            "denoise_seconds_per_step",
            "vae_decode_seconds",
            "postprocess_save_seconds",
            "peak_vram_mib",
        ),
        "comparison_keys": (
            "text_conditioning",
            "trajectory",
            "final_latent",
            "vae_tensor",
            "vae_png",
        ),
    },
    "Chroma1-HD": {
        "manifest": CHROMA_SAMPLER_MANIFEST,
        "nearby_non_sampler_files": (
            PARITY_DIR / "chroma_sampler_helper_ref.json",
            PARITY_DIR / "chroma_train_ref_meta.json",
            PARITY_DIR / "chroma_train_ref_step000.safetensors",
            PARITY_DIR / "chroma_train_ref_step000_adapters.safetensors",
        ),
        "prompt_keys": (
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
            "cfg_batch_size",
        ),
        "scheduler_keys": (
            "name",
            "sigmas",
            "timesteps",
            "shift",
            "dynamic_shift",
            "step_trace",
        ),
        "artifact_keys": (
            "onetrainer_initial_noise_raw_nchw",
            "onetrainer_initial_noise_packed",
            "onetrainer_latent_trajectory",
            "mojo_latent_trajectory",
            "onetrainer_cfg_prediction_trajectory",
            "mojo_cfg_prediction_trajectory",
            "onetrainer_prompt_embedding",
            "onetrainer_text_attention_mask",
            "onetrainer_attention_mask",
            "onetrainer_text_ids",
            "onetrainer_image_ids",
            "onetrainer_final_packed_latent",
            "onetrainer_final_unpacked_latent",
            "onetrainer_vae_input_latent",
            "onetrainer_vae_decoded_tensor",
            "mojo_prompt_embedding",
            "mojo_text_attention_mask",
            "mojo_attention_mask",
            "mojo_text_ids",
            "mojo_image_ids",
            "mojo_final_packed_latent",
            "mojo_final_unpacked_latent",
            "mojo_vae_input_latent",
            "mojo_vae_decoded_tensor",
            "onetrainer_png",
            "mojo_png",
        ),
        "metric_keys": (
            "text_seconds",
            "denoise_seconds_per_step",
            "vae_decode_seconds",
            "postprocess_save_seconds",
            "peak_vram_mib",
        ),
        "comparison_keys": (
            "text_conditioning",
            "text_attention_mask",
            "attention_mask",
            "trajectory",
            "cfg_prediction",
            "final_latent",
            "vae_tensor",
            "vae_png",
        ),
    },
}

CLAIM_SCAN_FILES = [
    MOJO_SAMPLING / "flux1_dev.mojo",
    MOJO_SAMPLING / "flux1_dev_smoke.mojo",
    MOJO_SAMPLING / "flux2_klein.mojo",
    MOJO_SAMPLING / "flux2_klein_smoke.mojo",
    MOJO_SAMPLING / "onetrainer_sampler_contract.mojo",
    MOJO_SAMPLING / "onetrainer_sampler_contract_smoke.mojo",
    MOJO_SAMPLING / "product_sampler_harness.mojo",
    MOJO_SAMPLING / "product_sampler_harness_smoke.mojo",
    MOJO_PIPELINE / "flux1_pipeline_smoke.mojo",
    MOJO_PIPELINE / "flux1_pipeline_cached_smoke.mojo",
    MOJO_DOCS / "SAMPLER_PRODUCT_HARNESS_2026-06-05.md",
    PORT_DOC,
]


def read(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"[flux-family-sampler] missing file: {path}")
    return path.read_text(encoding="utf-8")


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    text = read(path)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit(f"[flux-family-sampler] {label} must be a JSON object: {path}")
    print(f"[flux-family-sampler] PASS {label}: {path}")
    return data


def require(path: Path, label: str, needles: list[str]) -> None:
    text = read(path)
    missing = [needle for needle in needles if needle not in text]
    if missing:
        print(f"[flux-family-sampler] FAIL {label}: {path}")
        for needle in missing:
            print(f"  missing: {needle}")
        raise SystemExit(1)
    print(f"[flux-family-sampler] PASS {label}")


def require_no_tensor_upcast(path: Path, label: str) -> None:
    text = read(path)
    forbidden = [
        ".to_host(",
        "STDtype.F32",
        "DType.float32",
        "dtype=Float32",
    ]
    found = [needle for needle in forbidden if needle in text]
    if found:
        print(f"[flux-family-sampler] FAIL dtype boundary {label}: {path}")
        for needle in found:
            print(f"  forbidden tensor-storage pattern: {needle}")
        raise SystemExit(1)
    if "tensor_algebra preserves tensor storage" not in text:
        print(f"[flux-family-sampler] FAIL dtype comment {label}: {path}")
        print("  missing explicit tensor_algebra storage-dtype comment")
        raise SystemExit(1)
    print(f"[flux-family-sampler] PASS dtype boundary {label}")


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _claim_context(lines: list[str], index: int, radius: int = 20) -> str:
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    return "\n".join(lines[start:end]).lower()


def require_flux1_speed_claim_evidence() -> None:
    """Require hard evidence before any bounded Flux.1 speed-parity claim.

    This intentionally does not accept a generic product-harness measurement as
    Flux.1-dev parity. The claim must be near a Flux.1-dev marker and carry the
    run identity plus OneTrainer/Mojo speed, VRAM, dtype, and denoise-trajectory
    evidence markers in the same local context.
    """
    found_claims = 0
    for path in CLAIM_SCAN_FILES:
        text = read(path)
        lines = text.splitlines()
        for index, line in enumerate(lines):
            lowered = line.lower()
            if not _has_any(lowered, SPEED_PARITY_CLAIM_MARKERS):
                continue
            context = _claim_context(lines, index)
            if not _has_any(context, FLUX1_CLAIM_ALIASES):
                continue
            if _has_any(context, NEGATIVE_CLAIM_MARKERS):
                continue

            found_claims += 1
            missing = [
                label
                for label, markers in FLUX1_SPEED_EVIDENCE_GROUPS
                if not _has_any(context, markers)
            ]
            if missing:
                print(
                    "[flux-family-sampler] FAIL Flux.1 speed-parity claim "
                    f"evidence: {path}:{index + 1}"
                )
                for label in missing:
                    print(f"  missing evidence marker near claim: {label}")
                raise SystemExit(1)

    if found_claims:
        print(f"[flux-family-sampler] PASS Flux.1 speed-parity evidence claims={found_claims}")
    else:
        print("[flux-family-sampler] PASS Flux.1 speed-parity evidence gate: no accepted claim")


def require_flux2_klein_speed_claim_evidence() -> None:
    """Require hard evidence before any bounded Flux2/Klein speed-parity claim."""
    found_claims = 0
    for path in CLAIM_SCAN_FILES:
        text = read(path)
        lines = text.splitlines()
        for index, line in enumerate(lines):
            lowered = line.lower()
            if not _has_any(lowered, SPEED_PARITY_CLAIM_MARKERS):
                continue
            context = _claim_context(lines, index)
            if not _has_any(context, FLUX2_KLEIN_CLAIM_ALIASES):
                continue
            if _has_any(context, NEGATIVE_CLAIM_MARKERS):
                continue

            found_claims += 1
            missing = [
                label
                for label, markers in FLUX2_KLEIN_SPEED_EVIDENCE_GROUPS
                if not _has_any(context, markers)
            ]
            if missing:
                print(
                    "[flux-family-sampler] FAIL Flux2/Klein speed-parity claim "
                    f"evidence: {path}:{index + 1}"
                )
                for label in missing:
                    print(f"  missing evidence marker near claim: {label}")
                raise SystemExit(1)

    if found_claims:
        print(
            "[flux-family-sampler] PASS Flux2/Klein speed-parity evidence "
            f"claims={found_claims}"
        )
    else:
        print("[flux-family-sampler] PASS Flux2/Klein speed-parity evidence gate: no accepted claim")


def report_flux1_sampler_artifact_manifest_current_state() -> None:
    """Report Flux.1 sampler artifact readiness without accepting parity."""
    from check_flux1_sampler_artifact_manifest import Check, inspect_manifest

    try:
        checks = inspect_manifest(FLUX1_SAMPLER_MANIFEST)
    except Exception as exc:  # noqa: BLE001 - report damaged local artifact.
        checks = [Check(False, "manifest", str(exc))]
    blockers = [check for check in checks if not check.ok]
    if blockers:
        print(
            "[flux-family-sampler] BLOCKED Flux.1 sampler artifact manifest: "
            f"blockers={len(blockers)} manifest={FLUX1_SAMPLER_MANIFEST}"
        )
        for check in blockers[:8]:
            print(f"  {check.label}: {check.detail}")
        if len(blockers) > 8:
            print(f"  ... {len(blockers) - 8} more blockers")
    else:
        print(
            "[flux-family-sampler] PASS Flux.1 sampler artifact manifest: "
            f"{FLUX1_SAMPLER_MANIFEST}"
        )


def report_chroma_sampler_artifact_manifest_current_state() -> None:
    """Report Chroma sampler artifact readiness without accepting parity."""
    from check_chroma_sampler_artifact_manifest import Check, inspect_manifest

    try:
        checks = inspect_manifest(CHROMA_SAMPLER_MANIFEST)
    except Exception as exc:  # noqa: BLE001 - report damaged local artifact.
        checks = [Check(False, "manifest", str(exc))]
    blockers = [check for check in checks if not check.ok]
    if blockers:
        print(
            "[flux-family-sampler] BLOCKED Chroma sampler artifact manifest: "
            f"blockers={len(blockers)} manifest={CHROMA_SAMPLER_MANIFEST}"
        )
        for check in blockers[:8]:
            print(f"  {check.label}: {check.detail}")
        if len(blockers) > 8:
            print(f"  ... {len(blockers) - 8} more blockers")
    else:
        print(
            "[flux-family-sampler] PASS Chroma sampler artifact manifest: "
            f"{CHROMA_SAMPLER_MANIFEST}"
        )


def _print_inventory_keys(label: str, keys: tuple[str, ...]) -> None:
    print(f"  {label} ({len(keys)}):")
    for key in keys:
        print(f"    {key}")


def _missing_section_keys(data: dict[str, Any], section: str, keys: tuple[str, ...]) -> list[str]:
    value = data.get(section)
    if not isinstance(value, dict):
        return [f"{section}.{key}" for key in keys]
    return [f"{section}.{key}" for key in keys if key not in value]


def _missing_metric_keys(data: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        return [f"metrics.onetrainer.{key}" for key in keys] + [f"metrics.mojo.{key}" for key in keys]

    missing: list[str] = []
    for prefix in ("onetrainer", "mojo"):
        values = metrics.get(prefix)
        if not isinstance(values, dict):
            missing.extend(f"metrics.{prefix}.{key}" for key in keys)
            continue
        missing.extend(f"metrics.{prefix}.{key}" for key in keys if key not in values)
    return missing


def _missing_comparison_keys(data: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    comparisons = data.get("comparisons")
    required_fields = ("accepted", "max_abs", "tolerance")
    if not isinstance(comparisons, dict):
        return [f"comparisons.{key}.{field}" for key in keys for field in required_fields]

    missing: list[str] = []
    for key in keys:
        value = comparisons.get(key)
        if not isinstance(value, dict):
            missing.extend(f"comparisons.{key}.{field}" for field in required_fields)
            continue
        missing.extend(f"comparisons.{key}.{field}" for field in required_fields if field not in value)
    return missing


def _inspect_declared_artifact_paths(
    data: dict[str, Any], artifact_keys: tuple[str, ...]
) -> tuple[list[str], list[str], int]:
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        return [f"artifacts.{key}.path" for key in artifact_keys], [], 0

    missing_manifest_keys: list[str] = []
    missing_files: list[str] = []
    present_files = 0
    for key in artifact_keys:
        value = artifacts.get(key)
        if not isinstance(value, dict):
            missing_manifest_keys.append(f"artifacts.{key}.path")
            continue
        path_value = value.get("path")
        if not isinstance(path_value, str) or not path_value:
            missing_manifest_keys.append(f"artifacts.{key}.path")
            continue

        path = Path(path_value)
        if not path.exists():
            missing_files.append(f"artifacts.{key}.path -> {path}")
        elif path.stat().st_size <= 0:
            missing_files.append(f"artifacts.{key}.path -> empty file: {path}")
        else:
            present_files += 1

    return missing_manifest_keys, missing_files, present_files


def report_sampler_artifact_inventory_current_state() -> None:
    """Print all Flux-family sampler manifest evidence gaps in one place."""
    print("[flux-family-sampler] sampler artifact manifest inventory")
    for family, spec in SAMPLER_ARTIFACT_REQUIREMENTS.items():
        manifest = spec["manifest"]
        if not isinstance(manifest, Path):
            raise SystemExit(f"[flux-family-sampler] malformed inventory spec: {family}")

        prompt_keys = tuple(str(key) for key in spec["prompt_keys"])
        scheduler_keys = tuple(str(key) for key in spec["scheduler_keys"])
        artifact_keys = tuple(str(key) for key in spec["artifact_keys"])
        metric_keys = tuple(str(key) for key in spec["metric_keys"])
        comparison_keys = tuple(str(key) for key in spec["comparison_keys"])
        nearby_files = tuple(path for path in spec["nearby_non_sampler_files"] if isinstance(path, Path) and path.exists())

        print(f"[flux-family-sampler] INVENTORY {family}: manifest={manifest}")
        for path in nearby_files:
            print(f"  nearby non-sampler artifact present; not wired as parity evidence: {path}")

        if not manifest.exists():
            print(f"  missing file: {manifest}")
            _print_inventory_keys("required prompt keys", tuple(f"prompt.{key}" for key in prompt_keys))
            _print_inventory_keys("required scheduler keys", tuple(f"scheduler.{key}" for key in scheduler_keys))
            _print_inventory_keys(
                "required artifact path keys",
                tuple(f"artifacts.{key}.path" for key in artifact_keys),
            )
            _print_inventory_keys(
                "required metric keys",
                tuple(f"metrics.onetrainer.{key}" for key in metric_keys)
                + tuple(f"metrics.mojo.{key}" for key in metric_keys),
            )
            _print_inventory_keys(
                "required comparison keys",
                tuple(
                    f"comparisons.{key}.{field}"
                    for key in comparison_keys
                    for field in ("accepted", "max_abs", "tolerance")
                ),
            )
            continue

        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - report damaged local artifact.
            print(f"  damaged manifest: {exc}")
            continue
        if not isinstance(data, dict):
            print("  damaged manifest: JSON root is not an object")
            continue

        missing_keys = []
        missing_keys.extend(_missing_section_keys(data, "prompt", prompt_keys))
        missing_keys.extend(_missing_section_keys(data, "scheduler", scheduler_keys))
        artifact_missing_keys, missing_files, present_files = _inspect_declared_artifact_paths(data, artifact_keys)
        missing_keys.extend(artifact_missing_keys)
        missing_keys.extend(_missing_metric_keys(data, metric_keys))
        missing_keys.extend(_missing_comparison_keys(data, comparison_keys))

        print(f"  manifest file present bytes={manifest.stat().st_size}")
        print(f"  declared artifact files present={present_files}/{len(artifact_keys)}")
        if missing_keys:
            _print_inventory_keys("missing manifest keys", tuple(missing_keys))
        if missing_files:
            _print_inventory_keys("missing declared files", tuple(missing_files))
        if not missing_keys and not missing_files:
            print("  no inventory-level missing keys or files")


def _json_at(data: dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def require_flux2_dev_current_state() -> None:
    """Report the current Flux2-dev state without accepting train parity.

    Flux2-dev is OneTrainer ModelType.FLUX_2 only after the loaded transformer
    proves `num_attention_heads == 48`; Klein is the other Flux2 branch. The
    current Mojo product path must fail loud for dev instead of silently routing
    to the Klein runner, and the reference JSON must continue to state that no
    CUDA one-step tensors or train parity exist yet.
    """
    contract = load_json_object(FLUX2_DEV_CONTRACT_JSON, "Flux2-dev contract JSON present")
    blockers = load_json_object(FLUX2_DEV_BLOCKERS_JSON, "Flux2-dev blockers JSON present")

    failures: list[str] = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    expect(contract.get("name") == "flux2_dev_train_ref", "contract name is not flux2_dev_train_ref")
    expect(
        _json_at(contract, "reference_policy.only_reference") == "/home/alex/OneTrainer",
        "contract does not pin /home/alex/OneTrainer as the only reference",
    )
    expect(
        _json_at(contract, "numeric_parity_policy.status")
        == "none_until_cuda_reference_tensors_exist",
        "contract does not block numeric parity until CUDA reference tensors exist",
    )
    forbidden_claims = _json_at(contract, "numeric_parity_policy.forbidden_claims")
    if not isinstance(forbidden_claims, list):
        forbidden_claims = []
    expect(
        "Flux2 Klein tensors standing in for Flux2 dev tensors"
        in forbidden_claims,
        "contract does not forbid Klein tensors as Flux2-dev substitutes",
    )
    expect(
        _json_at(contract, "variant_contract.model_type") == "FLUX_2",
        "contract does not record OneTrainer ModelType.FLUX_2",
    )
    expect(
        "num_attention_heads == 48"
        in str(_json_at(contract, "variant_contract.dev_detection")),
        "contract does not record Flux2-dev num_attention_heads==48 detection",
    )
    expect(
        "Mistral" in str(_json_at(contract, "variant_contract.dev_text_path")),
        "contract does not record the Flux2-dev Mistral text path",
    )
    expect(
        "not valid" in str(_json_at(contract, "variant_contract.klein_text_path")),
        "contract does not reject the Klein text path for Flux2-dev parity",
    )
    expect(
        _json_at(contract, "local_partial_flux2_dev_transformer_config.num_attention_heads") == 48,
        "contract does not record the local partial Flux2-dev transformer heads",
    )
    expect(
        list(_json_at(contract, "numeric_parity_policy.required_files") or [])
        == list(FLUX2_DEV_REQUIRED_REFERENCE_TENSORS),
        "contract required reference tensor list drifted",
    )

    expect(blockers.get("dry_run") is True, "blockers JSON is not marked dry_run=true")
    expect(
        blockers.get("one_step_dump_produced") is False,
        "blockers JSON does not report one_step_dump_produced=false",
    )
    expect(
        blockers.get("numeric_parity_status") == "none",
        "blockers JSON does not report numeric_parity_status=none",
    )
    expect(
        blockers.get("dry_run_checks_are_structural_only") is True,
        "blockers JSON does not report structural-only dry-run checks",
    )
    expect(blockers.get("blocked") is True, "blockers JSON is not blocked=true")
    expect(
        _json_at(blockers, "reference_tensors.complete") is False,
        "blockers JSON does not report incomplete Flux2-dev reference tensors",
    )
    files = _json_at(blockers, "reference_tensors.files")
    if not isinstance(files, dict):
        failures.append("blockers JSON reference_tensors.files is missing")
    else:
        for name in FLUX2_DEV_REQUIRED_REFERENCE_TENSORS:
            info = files.get(name)
            if not isinstance(info, dict):
                failures.append(f"blockers JSON missing reference tensor entry: {name}")
            else:
                expect(
                    info.get("exists") is False,
                    f"blockers JSON no longer reports missing reference tensor: {name}",
                )

    for claim in (
        "CPU PyTorch numeric parity",
        "dry-run numeric parity",
        "Flux2 Klein tensors standing in for Flux2 dev tensors",
    ):
        expect(
            claim in forbidden_claims,
            f"contract missing forbidden claim: {claim}",
        )

    if failures:
        print("[flux-family-sampler] FAIL Flux2-dev contract evidence")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)

    print(
        "[flux-family-sampler] PASS Flux2-dev contract evidence: "
        "contract/blockers present, one-step dump absent, train parity not accepted"
    )

    require(
        MOJO_TRAINING / "onetrainer_lifecycle.mojo",
        "Mojo Flux2-dev lifecycle fail-loud",
        [
            "def _is_flux2_dev_alias(model_type: String) -> Bool:",
            'return model_type == String("flux2_dev") or model_type == String("FLUX_2_DEV")',
            "if _is_flux2_model_type(cfg.name) and cfg.n_heads == 48:",
            'return String("FLUX_2_DEV")',
            "if sampler_plan.family == OT_SAMPLER_FLUX2_DEV:",
            "Flux2 dev has no real Mojo product runner",
            "must not be dispatched to train_klein_real",
        ],
    )
    require(
        MOJO_TRAINING / "onetrainer_lifecycle_smoke.mojo",
        "Mojo Flux2-dev lifecycle smoke fail-loud coverage",
        [
            'blocked_cfg.name = String("FLUX_2_DEV")',
            "Flux2 dev alias product runner",
            "must not be dispatched to train_klein_real",
            'blocked_cfg.name = String("FLUX_2")',
            "blocked_cfg.n_heads = 48",
            "Flux2 dev num_attention_heads product runner",
            "num_attention_heads == 48",
        ],
    )
    require(
        MOJO_TRAINING / "onetrainer_product_run_smoke.mojo",
        "Mojo Flux2-dev product smoke fail-loud coverage",
        [
            '_write_min_product_config(flux2_dev_alias_path, String("FLUX_2_DEV"), concept_path)',
            "Flux2 dev has no product runner",
            "flux2_dev_heads_path, String(\"FLUX_2\"), concept_path, 48",
            "FLUX_2 num_attention_heads==48 has no product runner",
        ],
    )

    catalog = read(MOJO_TRAINING / "onetrainer_preset_catalog.mojo")
    for alias in ("flux2_dev", "FLUX_2_DEV", "flux2_dev_lora"):
        marker = f'String("{alias}")'
        index = catalog.find(marker)
        if index < 0:
            continue
        context = catalog[index : index + 600]
        if "_entry(" in context and "klein" in context.lower():
            print("[flux-family-sampler] FAIL Flux2-dev catalog alias routes to Klein")
            print(f"  alias: {alias}")
            raise SystemExit(1)
    print("[flux-family-sampler] PASS Flux2-dev catalog does not route dev aliases to Klein")


def scheduler_config_from_train_config(config_path: Path) -> Path | None:
    text = read(config_path)
    match = re.search(r'"base_model_name"\s*:\s*"([^"]+)"', text)
    if not match:
        print(f"[flux-family-sampler] WARN no base_model_name in {config_path}")
        return None
    base = Path(match.group(1))
    if not base.is_absolute():
        print(
            "[flux-family-sampler] WARN non-local base_model_name in "
            f"{config_path}: {match.group(1)}"
        )
        return None
    return base / "scheduler/scheduler_config.json"


def require_scheduler_config(
    label: str, train_config_path: Path, expected: dict[str, object]
) -> None:
    scheduler_path = scheduler_config_from_train_config(train_config_path)
    if scheduler_path is None:
        return
    data = json.loads(read(scheduler_path))
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual = data.get(key)
        if actual != expected_value:
            mismatches.append(f"{key}: got {actual!r}, expected {expected_value!r}")
    if mismatches:
        print(f"[flux-family-sampler] FAIL scheduler config {label}: {scheduler_path}")
        for mismatch in mismatches:
            print(f"  {mismatch}")
        raise SystemExit(1)
    print(f"[flux-family-sampler] PASS scheduler config {label}: {scheduler_path}")


def main() -> int:
    require(
        OT_SAMPLERS / "FluxSampler.py",
        "OneTrainer Flux.1 sampler",
        [
            "shift = self.model.calculate_timestep_shift",
            "noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device, mu=math.log(shift))",
            "timestep=expanded_timestep / 1000",
            "guidance = torch.tensor([cfg_scale]",
            "noise_scheduler.step(",
        ],
    )
    require(
        OT_MODELS / "FluxModel.py",
        "OneTrainer Flux.1 model shift",
        [
            "def calculate_timestep_shift(self, latent_height: int, latent_width: int):",
            "base_image_seq_len",
            "max_image_seq_len",
            "return math.exp(mu)",
        ],
    )
    require(
        OT_SETUPS / "BaseFluxSetup.py",
        "OneTrainer Flux.1 setup",
        [
            "shift = model.calculate_timestep_shift",
            "shift = shift if config.dynamic_timestep_shifting else config.timestep_shift",
            "scaled_noisy_latent_image, sigma = self._add_noise_discrete",
            "timestep=timestep / 1000",
            "flow = latent_noise - scaled_latent_image",
        ],
    )

    require(
        OT_SAMPLERS / "Flux2Sampler.py",
        "OneTrainer shared Flux2 sampler",
        [
            "from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu",
            "batch_size = 2 if cfg_scale > 1.0 and not transformer.config.guidance_embeds else 1",
            "mu = compute_empirical_mu(image_seq_len, diffusion_steps)",
            "sigmas = np.linspace(1.0, 1 / diffusion_steps, diffusion_steps)",
            "noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device, mu=mu, sigmas=sigmas)",
            "timestep=expanded_timestep / 1000",
            "guidance = (torch.tensor([cfg_scale]",
            "if transformer.config.guidance_embeds else None)",
            "guidance=guidance",
            "noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)",
        ],
    )
    require(
        OT_MODELS / "Flux2Model.py",
        "OneTrainer Flux2 dev/Klein discriminator",
        [
            "klass = Flux2Pipeline if self.is_dev() else Flux2KleinPipeline",
            "MISTRAL_HIDDEN_STATES_LAYERS = [10, 20, 30]",
            "QWEN3_HIDDEN_STATES_LAYERS = [9, 18, 27]",
            "if self.is_dev():",
            "messages = mistral_format_input(prompts=text, system_message=MISTRAL_SYSTEM_MESSAGE)",
            "else: #Flux2.Klein",
            "enable_thinking=False",
            "def is_dev(self) -> bool:",
            "return self.transformer.config.num_attention_heads == 48",
        ],
    )
    require(
        OT_SETUPS / "BaseFlux2Setup.py",
        "OneTrainer shared Flux2 setup",
        [
            "shift = model.calculate_timestep_shift",
            "shift = shift if config.dynamic_timestep_shifting else config.timestep_shift",
            "scaled_noisy_latent_image, sigma = self._add_noise_discrete",
            "timestep=timestep / 1000",
            "flow = latent_noise - scaled_latent_image",
        ],
    )

    require(
        OT_SAMPLERS / "ChromaSampler.py",
        "OneTrainer Chroma sampler",
        [
            "text=[prompt, negative_prompt]",
            "batch_size = 2",
            "noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device)",
            "timestep=expanded_timestep / 1000",
            "noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)",
            "attention_mask=attention_mask",
        ],
    )
    require(
        OT_SETUPS / "BaseChromaSetup.py",
        "OneTrainer Chroma setup",
        [
            "timestep = self._get_timestep_discrete",
            "scaled_noisy_latent_image, sigma = self._add_noise_discrete",
            "timestep=timestep / 1000",
            "flow = latent_noise - scaled_latent_image",
        ],
    )

    require_scheduler_config(
        "Flux.1-dev",
        OT_CONFIGS / "flux1_100step_baseline.json",
        {
            "shift": 3.0,
            "base_shift": 0.5,
            "max_shift": 1.15,
            "use_dynamic_shifting": True,
        },
    )
    require_scheduler_config(
        "Chroma1-HD",
        OT_CONFIGS / "chroma_100step_baseline.json",
        {
            "shift": 3.0,
            "base_shift": 0.5,
            "max_shift": 1.15,
            "use_dynamic_shifting": False,
            "time_shift_type": "exponential",
        },
    )

    require(
        MOJO_SAMPLING / "flux1_dev.mojo",
        "Mojo Flux.1 sampler helper",
        [
            "flux1_dynamic_shift",
            "flux1_model_timestep_from_scheduler_timestep",
            "flux1_guidance_embed_value",
            "flux1_euler_update_value",
            "Flux1DevScheduler",
        ],
    )
    require(
        MOJO_SAMPLING / "product_sampler_harness.mojo",
        "Mojo Flux.1 product measurement readiness",
        [
            "ProductSamplerRunContract",
            "SamplePrompt",
            "run.prompt.enabled",
            "run.prompt.prompt",
            "prompt.copy()",
            "run.plan.width",
            "run.plan.height",
            "run.plan.diffusion_steps",
            "run.plan.cfg_scale",
            "SamplerProductMeasurements",
            "ot_baseline_seconds_per_step",
            "mojo_seconds_per_step",
            "ot_peak_vram_mib",
            "mojo_peak_vram_mib",
            "transformer_denoise_ready",
            "measurement scaffold only, speed parity not accepted",
        ],
    )
    require(
        MOJO_SAMPLING / "product_sampler_harness_smoke.mojo",
        "Mojo product harness does not accept Flux.1 speed parity",
        [
            "empty measurement blocked as expected",
            "validate_sampler_speed_parity",
            "product_sampler_harness_smoke PASS",
        ],
    )
    require(
        MOJO_SAMPLING / "onetrainer_sampler_contract.mojo",
        "Mojo Flux2 dev/Klein sampler contract split",
        [
            "OT_SAMPLER_FLUX2_DEV",
            'model_type == String("flux2_dev")',
            'return String("flux2_dev")',
            "OT_SAMPLER_FLUX2_KLEIN",
            "OT_SCHEDULER_FLUX2_MU_CUSTOM_SIGMAS",
            "OT_CFG_GUIDANCE_EMBED_OR_TEXTBOOK",
        ],
    )
    require(
        MOJO_SAMPLING / "onetrainer_sampler_contract_smoke.mojo",
        "Mojo Flux2 dev sampler contract smoke",
        [
            'String("FLUX_2_DEV")',
            "OT_SAMPLER_FLUX2_DEV",
            "Flux2 dev alias",
        ],
    )
    require(
        MOJO_SAMPLING / "flux2_klein.mojo",
        "Mojo shared Flux2 sampler helper",
        [
            "compute_empirical_mu",
            "flux2_cfg_batch_size",
            "flux2_guidance_embed_value",
            "flux2_model_timestep_from_scheduler_timestep",
            "flux2_cfg_value",
            "flux2_euler_update_value",
            "Flux2KleinScheduler",
        ],
    )
    require(
        MOJO_SAMPLING / "chroma1_hd.mojo",
        "Mojo Chroma1-HD sampler helper",
        [
            "CHROMA1_HD_DEFAULT_SHIFT",
            "build_chroma1_hd_sigma_schedule",
            "chroma1_hd_cfg_batch_size",
            "chroma1_hd_model_timestep_from_scheduler_timestep",
            "chroma1_hd_euler_update_value",
            "Chroma1HDScheduler",
        ],
    )

    for label, path in [
        ("Flux.1", MOJO_SAMPLING / "flux1_dev.mojo"),
        ("Flux2 dev/Klein", MOJO_SAMPLING / "flux2_klein.mojo"),
        ("Chroma1-HD", MOJO_SAMPLING / "chroma1_hd.mojo"),
    ]:
        require_no_tensor_upcast(path, label)

    require_flux2_dev_current_state()
    report_sampler_artifact_inventory_current_state()
    report_flux1_sampler_artifact_manifest_current_state()
    report_chroma_sampler_artifact_manifest_current_state()
    require_flux1_speed_claim_evidence()
    require_flux2_klein_speed_claim_evidence()

    print("[flux-family-sampler] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
