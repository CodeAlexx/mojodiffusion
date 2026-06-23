#!/usr/bin/env python3
"""Static guard for SD3.5/SD35 OneTrainer parity blockers.

This is intentionally conservative. Passing report mode does not prove parity;
strict mode is the gate that should become green before SD35 is treated as an
official production trainer path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ONETRAINER = Path("/home/alex/OneTrainer")
ONETRAINER_MOJO = Path("/home/alex/onetrainer-mojo")

TARGET_MODEL_TYPE = "STABLE_DIFFUSION_35"
PLAIN_SD3_MODEL_TYPE = "STABLE_DIFFUSION_3"

CONFIGS = [
    ONETRAINER / "configs/sd35m_100step_baseline.json",
    ONETRAINER / "configs/sd35m_benchmark.json",
]
SD35_BLOCK = REPO / "serenitymojo/models/sd35/sd35_block.mojo"
SD35_STACK = REPO / "serenitymojo/models/sd35/sd35_stack_lora.mojo"
SD35_WEIGHTS = REPO / "serenitymojo/models/sd35/weights.mojo"
SD35_TRAIN = REPO / "../serenity-trainer/src/serenity_trainer/trainer/train_sd35_real.mojo"
SD35_CACHE = REPO / "serenitymojo/training/prepare_sd35_cache.py"
SD35_METRICS = ONETRAINER / "output/sd35m_100step_baseline/metrics.json"
SD35_BASELINE_LORA = ONETRAINER / "output/sd35m_100step_baseline/lora_last.safetensors"
SD3_TRAIN_REF_CONTRACT = ONETRAINER_MOJO / "parity/sd3_train_ref_contract.json"
SD3_TRAIN_REF_BLOCKERS = ONETRAINER_MOJO / "parity/sd3_train_ref_blockers.json"
SD3_SAMPLER_HELPER_REF = ONETRAINER_MOJO / "parity/sd3_sampler_helper_ref.json"
OT_MODEL_TYPE = ONETRAINER / "modules/util/enum/ModelType.py"
OT_SD3_DATALOADER = ONETRAINER / "modules/dataLoader/StableDiffusion3BaseDataLoader.py"


def read(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def has_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def resolve_onetrainer_mojo_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return ONETRAINER_MOJO / path


def contains_plain_sd3_alias(value: Any) -> bool:
    if isinstance(value, str):
        return PLAIN_SD3_MODEL_TYPE in value and TARGET_MODEL_TYPE not in value
    if isinstance(value, dict):
        return any(contains_plain_sd3_alias(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_plain_sd3_alias(item) for item in value)
    return False


def add_model_type_blocker(
    blockers: list[str],
    path: Path,
    model_type: Any,
    context: str,
) -> None:
    if model_type == TARGET_MODEL_TYPE:
        return
    if model_type == PLAIN_SD3_MODEL_TYPE:
        blockers.append(
            f"{path}: {context} uses plain SD3 model_type {PLAIN_SD3_MODEL_TYPE}; "
            f"SD3.5 target requires {TARGET_MODEL_TYPE}"
        )
    else:
        blockers.append(
            f"{path}: {context} has model_type={model_type!r}; "
            f"SD3.5 target requires {TARGET_MODEL_TYPE}"
        )


def sd35_lora_slots_incomplete(stack: str) -> bool:
    slots = [
        "SLOT_CTX_QKV",
        "SLOT_CTX_PROJ",
        "SLOT_CTX_FC1",
        "SLOT_CTX_FC2",
        "SLOT_X_QKV",
        "SLOT_X_PROJ",
        "SLOT_X_FC1",
        "SLOT_X_FC2",
    ]
    if has_any(
        stack,
        [
            "TODO: add proj/fc1/fc2",
            "TODO: add ctx_d_qkv",
            "if s != SLOT_X_QKV",
            "ctx_qkv_delta",
            "LoRA on proj outputs is handled in the backward",
        ],
    ):
        return True
    for slot in slots:
        if stack.count(f"lora.ad[base_lora + {slot}]") < 2:
            return True
        if f"d_a_flat[base_lora + {slot}]" not in stack:
            return True
        if f"d_b_flat[base_lora + {slot}]" not in stack:
            return True
    return False


def check_onetrainer_sd35_target(blockers: list[str]) -> None:
    enum_source = read(OT_MODEL_TYPE)
    dataloader_source = read(OT_SD3_DATALOADER)

    if f"{TARGET_MODEL_TYPE} = '{TARGET_MODEL_TYPE}'" not in enum_source:
        blockers.append(f"{OT_MODEL_TYPE}: missing SD3.5 enum {TARGET_MODEL_TYPE}")
    if "def is_stable_diffusion_3_5" not in enum_source:
        blockers.append(f"{OT_MODEL_TYPE}: missing explicit SD3.5 model-type helper")
    if f"return self == ModelType.{TARGET_MODEL_TYPE}" not in enum_source:
        blockers.append(f"{OT_MODEL_TYPE}: SD3.5 helper does not target {TARGET_MODEL_TYPE}")

    target_registration = (
        "factory.register(BaseDataLoader, StableDiffusion3BaseDataLoader, "
        f"ModelType.{TARGET_MODEL_TYPE})"
    )
    plain_registration = (
        "factory.register(BaseDataLoader, StableDiffusion3BaseDataLoader, "
        f"ModelType.{PLAIN_SD3_MODEL_TYPE})"
    )
    if target_registration not in dataloader_source:
        blockers.append(f"{OT_SD3_DATALOADER}: SD3.5 data loader is not registered")
    if plain_registration in dataloader_source:
        blockers.append(
            f"{OT_SD3_DATALOADER}: plain SD3 data loader is registered; "
            "SD3.5 contract must not accept plain SD3 aliases"
        )


def check_config_targets(blockers: list[str]) -> None:
    for config in CONFIGS:
        data = load_json(config)
        add_model_type_blocker(blockers, config, data.get("model_type"), "config")
        if data.get("training_method") != "LORA":
            blockers.append(
                f"{config}: reference baseline must be LoRA for the local "
                f"SD3.5 LoRA artifact gate (training_method={data.get('training_method')!r})"
            )

    baseline = load_json(CONFIGS[0])
    destination = baseline.get("output_model_destination")
    if destination != str(SD35_BASELINE_LORA):
        blockers.append(
            f"{CONFIGS[0]}: output_model_destination={destination!r}; "
            f"expected {SD35_BASELINE_LORA}"
        )


def check_reference_artifacts(blockers: list[str]) -> None:
    metrics = load_json(SD35_METRICS)
    config_path = metrics.get("config_path")
    if isinstance(config_path, str) and config_path:
        resolved_config = Path(config_path)
        if not resolved_config.is_absolute():
            resolved_config = ONETRAINER / resolved_config
        if resolved_config != CONFIGS[0]:
            blockers.append(
                f"{SD35_METRICS}: metrics came from {resolved_config}; "
                f"expected SD3.5 baseline config {CONFIGS[0]}"
            )
    else:
        blockers.append(f"{SD35_METRICS}: metrics do not record the source config path")

    status = metrics.get("status")
    requested_steps = as_int(metrics.get("requested_steps"))
    steps = as_int(metrics.get("global_steps_seen"))
    if status != "completed":
        blockers.append(
            f"{SD35_METRICS}: local OneTrainer baseline is not usable "
            f"(status={status!r}, steps={steps!r}, error={metrics.get('error')!r})"
        )
    elif steps <= 0:
        blockers.append(f"{SD35_METRICS}: local OneTrainer baseline completed with no training steps")
    elif requested_steps > 0 and steps < requested_steps:
        blockers.append(
            f"{SD35_METRICS}: local OneTrainer baseline stopped early "
            f"(steps={steps}, requested_steps={requested_steps})"
        )

    if as_int(metrics.get("loss_count")) <= 0:
        blockers.append(f"{SD35_METRICS}: no recorded training loss scalars")
    if metrics.get("last_loss") is None:
        blockers.append(f"{SD35_METRICS}: last_loss is missing")
    if as_int(metrics.get("grad_norm_count")) <= 0:
        blockers.append(f"{SD35_METRICS}: no recorded grad_norm scalars")
    if metrics.get("last_grad_norm") is None:
        blockers.append(f"{SD35_METRICS}: last_grad_norm is missing")

    output_destination = metrics.get("output_model_destination")
    if output_destination != str(SD35_BASELINE_LORA):
        blockers.append(
            f"{SD35_METRICS}: output_model_destination={output_destination!r}; "
            f"expected {SD35_BASELINE_LORA}"
        )
    if not SD35_BASELINE_LORA.exists():
        blockers.append(f"{SD35_BASELINE_LORA}: missing local OneTrainer SD3.5 LoRA baseline artifact")
    elif SD35_BASELINE_LORA.stat().st_size <= 0:
        blockers.append(f"{SD35_BASELINE_LORA}: local OneTrainer SD3.5 LoRA baseline artifact is empty")


def check_train_ref_contract(blockers: list[str]) -> None:
    contract = load_json(SD3_TRAIN_REF_CONTRACT)
    policy = contract.get("numeric_parity_policy", {})
    if policy.get("claimed_by_dry_run") is not False:
        blockers.append(f"{SD3_TRAIN_REF_CONTRACT}: dry-run numeric parity claim is not explicitly false")
    if policy.get("status") != "none_until_cuda_reference_tensors_exist":
        blockers.append(
            f"{SD3_TRAIN_REF_CONTRACT}: numeric_parity_policy.status={policy.get('status')!r}; "
            "expected none_until_cuda_reference_tensors_exist"
        )
    if policy.get("required_device") != "OneTrainer CUDA path":
        blockers.append(
            f"{SD3_TRAIN_REF_CONTRACT}: required_device={policy.get('required_device')!r}; "
            "expected OneTrainer CUDA path"
        )

    default_config = contract.get("default_config")
    if default_config != str(CONFIGS[0]):
        blockers.append(
            f"{SD3_TRAIN_REF_CONTRACT}: default_config={default_config!r}; "
            f"expected {CONFIGS[0]}"
        )

    dry_run_contract = contract.get("dry_run_contract", {})
    known_blockers = dry_run_contract.get("current_known_blockers_for_default_config", [])
    if contains_plain_sd3_alias(known_blockers):
        blockers.append(
            f"{SD3_TRAIN_REF_CONTRACT}: default-config blockers still describe plain "
            f"{PLAIN_SD3_MODEL_TYPE}; regenerate against {TARGET_MODEL_TYPE} before "
            "accepting SD3.5 reference evidence"
        )

    outputs = contract.get("default_outputs", {})
    for field, label in [
        ("step_tensors", "one-step train tensor"),
        ("adapter_tensors", "one-step LoRA adapter"),
    ]:
        output_path = resolve_onetrainer_mojo_path(outputs.get(field))
        if output_path is None:
            blockers.append(f"{SD3_TRAIN_REF_CONTRACT}: missing default_outputs.{field}")
        elif not output_path.exists():
            blockers.append(f"{output_path}: missing SD3.5 OneTrainer CUDA {label} dump")
        elif output_path.stat().st_size <= 0:
            blockers.append(f"{output_path}: SD3.5 OneTrainer CUDA {label} dump is empty")


def check_train_ref_blockers(blockers: list[str]) -> None:
    artifact = load_json(SD3_TRAIN_REF_BLOCKERS)
    add_model_type_blocker(
        blockers,
        SD3_TRAIN_REF_BLOCKERS,
        artifact.get("model_type"),
        "train-reference blocker artifact",
    )

    config = artifact.get("config")
    if config != str(CONFIGS[0]):
        blockers.append(
            f"{SD3_TRAIN_REF_BLOCKERS}: config={config!r}; expected SD3.5 baseline config {CONFIGS[0]}"
        )

    if artifact.get("dry_run") is True:
        blockers.append(
            f"{SD3_TRAIN_REF_BLOCKERS}: dry-run artifact is structural only; "
            "it is not SD3.5 baseline or train-dump parity"
        )
    elif artifact.get("dry_run") is not False:
        blockers.append(f"{SD3_TRAIN_REF_BLOCKERS}: dry_run marker is missing or invalid")

    if artifact.get("one_step_dump_produced") is not True:
        blockers.append(f"{SD3_TRAIN_REF_BLOCKERS}: no OneTrainer CUDA one-step train dump was produced")
    if artifact.get("numeric_parity_claimed") is not False:
        blockers.append(f"{SD3_TRAIN_REF_BLOCKERS}: numeric_parity_claimed is not explicitly false")
    if artifact.get("numeric_parity_status") != "none":
        blockers.append(
            f"{SD3_TRAIN_REF_BLOCKERS}: numeric_parity_status={artifact.get('numeric_parity_status')!r}; "
            "SD3.5 train parity must not be claimed here"
        )
    if artifact.get("dry_run_checks_are_structural_only") is not True:
        blockers.append(f"{SD3_TRAIN_REF_BLOCKERS}: structural-only dry-run marker is missing")
    if artifact.get("blocked") is not True:
        blockers.append(f"{SD3_TRAIN_REF_BLOCKERS}: blocker artifact is not marked blocked")

    structured_blockers = artifact.get("structured_blockers", [])
    blocker_ids = {
        item.get("id")
        for item in structured_blockers
        if isinstance(item, dict)
    }
    if artifact.get("model_type") == PLAIN_SD3_MODEL_TYPE and "missing_data_loader_registration" not in blocker_ids:
        blockers.append(
            f"{SD3_TRAIN_REF_BLOCKERS}: plain SD3 blocker artifact lacks "
            "missing_data_loader_registration; plain SD3 must fail loud"
        )


def check_sampler_helper_ref(blockers: list[str]) -> None:
    helper = load_json(SD3_SAMPLER_HELPER_REF)
    scope = helper.get("scope")
    if not isinstance(scope, str) or "helper-only" not in scope or "not end-to-end sampler parity" not in scope:
        blockers.append(
            f"{SD3_SAMPLER_HELPER_REF}: sampler helper scope must remain helper-only "
            "and must not claim end-to-end sampler parity"
        )


def collect_blockers() -> list[str]:
    blockers: list[str] = []

    check_onetrainer_sd35_target(blockers)
    check_config_targets(blockers)

    block = read(SD35_BLOCK)
    stack = read(SD35_STACK)
    weights = read(SD35_WEIGHTS)
    train = read(SD35_TRAIN)
    cache = read(SD35_CACHE)

    if "host List[Float32] API boundary" in block or "List[Float32]" in block:
        blockers.append(f"{SD35_BLOCK}: host-F32 block API is still present")
    if "STDtype.F32" in stack and "def _t(" in stack:
        blockers.append(f"{SD35_STACK}: stack helper still materializes F32 tensors")
    if has_any(weights, ["to_host(ctx)", "STDtype.F32"]):
        blockers.append(f"{SD35_WEIGHTS}: loader still has host/F32 tensor boundaries")
    if has_any(stack, ["_block_host_f32", "cast_tensor(block[key][], STDtype.F32"]):
        blockers.append(f"{SD35_STACK}: streamed block path still widens to host/F32")
    if "BlockSaved" in stack and "blocks.append(BlockSaved" in stack:
        blockers.append(f"{SD35_STACK}: forward still saves full block activations")
    if sd35_lora_slots_incomplete(stack):
        blockers.append(f"{SD35_STACK}: LoRA target/grad coverage is still incomplete")
    if '"latent_image"' not in train and "latent" in train:
        blockers.append(f"{SD35_TRAIN}: train path does not consume OneTrainer latent_image cache")
    if has_any(cache, ["dist.sample()", "lat = (lat -", "latent_scaled"]):
        blockers.append(f"{SD35_CACHE}: cache writer does not match OneTrainer latent mean/scale timing")
    check_reference_artifacts(blockers)
    check_train_ref_contract(blockers)
    check_train_ref_blockers(blockers)
    check_sampler_helper_ref(blockers)
    return blockers


def classify_blocker(blocker: str) -> str:
    if "model_type" in blocker or "data loader" in blocker or "config" in blocker:
        return "onetrainer_config"
    if "baseline" in blocker or "metrics" in blocker or "lora_last" in blocker:
        return "onetrainer_baseline"
    if "one-step" in blocker or "train dump" in blocker or "train_ref" in blocker:
        return "train_ref_artifact"
    if "host-F32" in blocker or "F32" in blocker or "full block activations" in blocker:
        return "mojo_dtype_or_memory"
    if "sampler helper" in blocker:
        return "sampler_helper"
    return "other"


def build_readiness_report(blockers: list[str]) -> dict[str, Any]:
    category_counts: dict[str, int] = {}
    for blocker in blockers:
        category = classify_blocker(blocker)
        category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "schema": "serenity.sd35.contract_readiness.v1",
        "target_model_type": TARGET_MODEL_TYPE,
        "plain_sd3_model_type": PLAIN_SD3_MODEL_TYPE,
        "ready": not blockers,
        "accepted_training_parity": False,
        "accepted_sampler_parity": False,
        "blocker_count": len(blockers),
        "category_counts": category_counts,
        "blockers": [
            {"category": classify_blocker(blocker), "message": blocker}
            for blocker in blockers
        ],
        "required_onetrainer_inputs": {
            "baseline_configs": [str(path) for path in CONFIGS],
            "baseline_metrics": str(SD35_METRICS),
            "baseline_lora": str(SD35_BASELINE_LORA),
            "train_ref_contract": str(SD3_TRAIN_REF_CONTRACT),
            "train_ref_blockers": str(SD3_TRAIN_REF_BLOCKERS),
            "sampler_helper_ref": str(SD3_SAMPLER_HELPER_REF),
        },
        "required_next_evidence": [
            "Regenerate the OneTrainer SD3.5 baseline configs with model_type STABLE_DIFFUSION_35.",
            "Run a completed OneTrainer 100-step SD3.5 LoRA baseline with loss, grad norm, speed, and saved LoRA artifact.",
            "Dump OneTrainer CUDA one-step SD3.5 train tensors plus LoRA adapter tensors for Mojo replay.",
            "Remove or replace the remaining Mojo host-F32/full-activation SD3.5 training boundaries before dtype/memory parity.",
            "Keep sampler helper evidence helper-only until paired sampler trajectory, image, speed, and VRAM artifacts exist.",
        ],
        "strict_command": "python3 scripts/check_sd35_contract.py --strict",
    }


def write_readiness(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_text_report(blockers: list[str]) -> None:
    report = build_readiness_report(blockers)

    print("[sd35-contract] target_model_type:", TARGET_MODEL_TYPE)
    print("[sd35-contract] blockers:", len(blockers))
    if blockers:
        print("[sd35-contract] categories:", json.dumps(report["category_counts"], sort_keys=True))
    for blocker in blockers:
        print("[sd35-contract] WARN:", blocker)
    if not blockers:
        print("[sd35-contract] PASS")
    else:
        print("[sd35-contract] report-only BLOCKED; use --strict for a nonzero production gate")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail while any known SD35 non-production blocker is present",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable readiness JSON instead of the text report",
    )
    parser.add_argument(
        "--write-readiness",
        type=Path,
        help="write the machine-readable readiness report to this path",
    )
    args = parser.parse_args()

    blockers = collect_blockers()
    report = build_readiness_report(blockers)
    if args.write_readiness is not None:
        write_readiness(args.write_readiness, report)
        if not args.json:
            print(f"[sd35-contract] wrote readiness report: {args.write_readiness}")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(blockers)

    if args.strict and blockers:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
