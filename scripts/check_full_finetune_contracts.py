#!/usr/bin/env python3
"""Inventory OneTrainer full-finetune contracts and Mojo readiness blockers.

This is a report-first guard. Default mode prints the OneTrainer registrations
and separates shared save/resume scaffolding from real product-loop full-finetune
parity evidence. Strict mode fails while any registered OneTrainer full-finetune
target still has blockers, or when product-loop support is claimed without a
real dispatch/smoke/parity proof.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ONETRAINER = Path("/home/alex/OneTrainer")
ONETRAINER_ANIMA_REF = Path("/home/alex/OneTrainer-anima-ref")
OT_SETUP_DIR = ONETRAINER / "modules/modelSetup"

TRAIN_CONFIG = REPO / "serenitymojo/training/train_config.mojo"
TRAIN_CONFIG_READER = REPO / "serenitymojo/io/train_config_reader.mojo"
TRAINING_DIR = REPO / "serenitymojo/training"
TRAIN_LOOP = TRAINING_DIR / "loop.mojo"
FULL_FT_CONTRACT = TRAINING_DIR / "full_finetune_contract.mojo"
FULL_FT_CONTRACT_SMOKE = TRAINING_DIR / "full_finetune_contract_smoke.mojo"
FULL_FT_SAVE = TRAINING_DIR / "full_finetune_save.mojo"
FULL_FT_SAVE_SMOKE = TRAINING_DIR / "full_finetune_save_smoke.mojo"
RESUME_CONTRACT = TRAINING_DIR / "onetrainer_resume_contract.mojo"
PRODUCT_RUN = TRAINING_DIR / "onetrainer_product_run.mojo"
PRODUCT_RUN_SMOKE = TRAINING_DIR / "onetrainer_product_run_smoke.mojo"
PARITY_DIR = Path("/home/alex/onetrainer-mojo/parity")


@dataclass(frozen=True)
class Target:
    key: str
    label: str
    aliases: tuple[str, ...]
    setup_file: str
    model_types: tuple[str, ...]
    train_loops: tuple[Path, ...]
    reference_root: Path = ONETRAINER


@dataclass(frozen=True)
class SetupInfo:
    path: Path
    text: str
    registrations: tuple[str, ...]
    setup_class: str


@dataclass(frozen=True)
class BehaviorSummary:
    requires_grad: str
    parameters: str


@dataclass(frozen=True)
class ModelProof:
    source_files: tuple[Path, ...]
    inventory_files: tuple[Path, ...]
    full_save_files: tuple[Path, ...]
    full_load_files: tuple[Path, ...]
    manifest_files: tuple[Path, ...]
    optimizer_sidecar_files: tuple[Path, ...]
    resume_files: tuple[Path, ...]
    parity_artifacts: tuple[Path, ...]
    product_resume_full_support: bool
    product_resume_fail_loud: bool
    product_loop_status: str


@dataclass(frozen=True)
class ProductLoopProof:
    support_claimed: bool
    fail_loud_smoke: bool
    lifecycle_rejects_finetune: bool
    lifecycle_lora_only_validator: bool
    full_finetune_blocker_preflight: bool
    full_runner_dispatch: bool
    positive_full_finetune_smoke: bool
    real_full_finetune_loop: bool
    status: str
    detail: str


TARGETS: tuple[Target, ...] = (
    Target(
        key="qwen",
        label="Qwen",
        aliases=("qwen", "qwenimage", "qwen-image", "qwen_image", "qwen image"),
        setup_file="QwenFineTuneSetup.py",
        model_types=("QWEN",),
        train_loops=(TRAINING_DIR / "train_qwenimage_real.mojo",),
    ),
    Target(
        key="flux1",
        label="Flux.1 dev/fill",
        aliases=("flux", "flux1", "flux.1", "flux-dev", "flux-fill", "flux fill"),
        setup_file="FluxFineTuneSetup.py",
        model_types=("FLUX_DEV_1", "FLUX_FILL_DEV_1"),
        train_loops=(TRAINING_DIR / "train_flux_real.mojo",),
    ),
    Target(
        key="flux2",
        label="Flux2/Klein",
        aliases=("flux2", "flux-2", "flux_2", "klein", "klein9b", "klein-9b"),
        setup_file="Flux2FineTuneSetup.py",
        model_types=("FLUX_2",),
        train_loops=(TRAINING_DIR / "train_klein_real.mojo",),
    ),
    Target(
        key="zimage",
        label="Z-Image",
        aliases=("zimage", "z-image", "z_image"),
        setup_file="ZImageFineTuneSetup.py",
        model_types=("Z_IMAGE",),
        train_loops=(TRAINING_DIR / "train_zimage_real.mojo",),
    ),
    Target(
        key="chroma",
        label="Chroma",
        aliases=("chroma", "chroma1", "chroma-1", "chroma_1"),
        setup_file="ChromaFineTuneSetup.py",
        model_types=("CHROMA_1",),
        train_loops=(TRAINING_DIR / "train_chroma_real.mojo",),
    ),
    Target(
        key="sdxl",
        label="SDXL",
        aliases=("sdxl", "stable-diffusion-xl", "stable_diffusion_xl"),
        setup_file="StableDiffusionXLFineTuneSetup.py",
        model_types=(
            "STABLE_DIFFUSION_XL_10_BASE",
            "STABLE_DIFFUSION_XL_10_BASE_INPAINTING",
        ),
        train_loops=(TRAINING_DIR / "train_sdxl_real.mojo",),
    ),
    Target(
        key="sd35",
        label="SD3.5",
        aliases=("sd35", "sd3.5", "sd3-5", "stable-diffusion-3.5"),
        setup_file="StableDiffusion3FineTuneSetup.py",
        model_types=("STABLE_DIFFUSION_35",),
        train_loops=(TRAINING_DIR / "train_sd35_real.mojo",),
    ),
    Target(
        key="ernie",
        label="Ernie",
        aliases=("ernie", "ernie-image", "ernie_image"),
        setup_file="ErnieFineTuneSetup.py",
        model_types=("ERNIE",),
        train_loops=(TRAINING_DIR / "train_ernie_real.mojo",),
    ),
    Target(
        key="anima",
        label="Anima",
        aliases=("anima",),
        setup_file="AnimaFineTuneSetup.py",
        model_types=("ANIMA",),
        train_loops=(TRAINING_DIR / "train_anima_real.mojo",),
        reference_root=ONETRAINER_ANIMA_REF,
    ),
)


MODEL_SOURCE_DIRS: dict[str, tuple[Path, ...]] = {
    "qwen": (REPO / "serenitymojo/models/qwenimage",),
    "flux1": (REPO / "serenitymojo/models/flux",),
    "flux2": (REPO / "serenitymojo/models/klein",),
    "zimage": (REPO / "serenitymojo/models/zimage",),
    "chroma": (REPO / "serenitymojo/models/chroma",),
    "sdxl": (REPO / "serenitymojo/models/sdxl",),
    "sd35": (REPO / "serenitymojo/models/sd35",),
    "ernie": (REPO / "serenitymojo/models/ernie",),
    "anima": (REPO / "serenitymojo/models/anima",),
}


REGISTER_RE = re.compile(
    r"factory\.register\(\s*BaseModelSetup\s*,\s*(?P<class>\w+)\s*,\s*"
    r"ModelType\.(?P<model_type>[A-Z0-9_]+)\s*,\s*"
    r"TrainingMethod\.(?P<method>[A-Z0-9_]+)\s*\)",
    re.DOTALL,
)


def read_optional(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def strip_mojo_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def norm_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _target_source_files(target: Target) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in target.train_loops:
        if path.exists():
            files.append(path)
    for root in MODEL_SOURCE_DIRS.get(target.key, ()):
        if root.exists():
            files.extend(sorted(root.rglob("*.mojo")))
    return tuple(dict.fromkeys(files))


def _files_with_markers(
    paths: tuple[Path, ...], markers: tuple[str, ...], require_all: bool = False
) -> tuple[Path, ...]:
    hits: list[Path] = []
    for path in paths:
        text = strip_mojo_comments(read_optional(path))
        if not text:
            continue
        if require_all:
            ok = all(marker in text for marker in markers)
        else:
            ok = any(marker in text for marker in markers)
        if ok:
            hits.append(path)
    return tuple(hits)


def _parity_artifacts(target: Target) -> tuple[Path, ...]:
    if not PARITY_DIR.exists():
        return ()
    tokens = {target.key}
    for alias in target.aliases:
        tokens.add(norm_token(alias))
    for model_type in target.model_types:
        tokens.add(norm_token(model_type))
    if target.key == "flux2":
        tokens.add("klein")
    if target.key == "sd35":
        tokens.add("sd3")

    hits: list[Path] = []
    for path in sorted(PARITY_DIR.iterdir()):
        if not path.is_file():
            continue
        name = norm_token(path.name)
        if not any(token and token in name for token in tokens):
            continue
        if any(
            word in name
            for word in (
                "full",
                "finetune",
                "finetun",
                "resume",
                "backup",
                "checkpoint",
            )
        ):
            hits.append(path)
    return tuple(hits)


def target_alias_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for target in TARGETS:
        keys = (target.key, target.label, *target.aliases, *target.model_types)
        for key in keys:
            out[norm_token(key)] = target.key
    return out


def select_targets(filters: list[str] | None, parser: argparse.ArgumentParser) -> list[Target]:
    if not filters:
        return list(TARGETS)

    aliases = target_alias_map()
    selected_keys: list[str] = []
    for raw in filters:
        key = aliases.get(norm_token(raw))
        if key is None:
            parser.error(
                "unknown --target-model "
                + repr(raw)
                + "; expected one of "
                + ", ".join(target.key for target in TARGETS)
            )
        if key not in selected_keys:
            selected_keys.append(key)
    by_key = {target.key: target for target in TARGETS}
    return [by_key[key] for key in selected_keys]


def target_setup_path(target: Target) -> Path:
    return target.reference_root / "modules/modelSetup" / target.setup_file


def scan_onetrainer_setups(targets: list[Target]) -> dict[str, SetupInfo]:
    by_key: dict[str, SetupInfo] = {}
    for target in targets:
        path = target_setup_path(target)
        text = read_optional(path)
        registrations = tuple(
            match.group("model_type")
            for match in REGISTER_RE.finditer(text)
            if match.group("method") == "FINE_TUNE"
        )
        class_match = re.search(r"^\s*class\s+(\w+)\b", text, flags=re.MULTILINE)
        by_key[target.key] = SetupInfo(
            path=path,
            text=text,
            registrations=registrations,
            setup_class=class_match.group(1) if class_match else path.stem,
        )
    return by_key


def unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def summarize_setup_behavior(info: SetupInfo | None) -> BehaviorSummary:
    if info is None:
        return BehaviorSummary("n/a", "n/a")

    text = info.text
    requires_parts = unique_strings(
        re.findall(r"_setup_model_part_requires_grad\(\s*\"([^\"]+)\"", text)
    )
    frozen_parts = unique_strings(
        re.findall(r"model\.([A-Za-z0-9_]+)\.requires_grad_\(False\)", text)
    )
    require_bits: list[str] = []
    if requires_parts:
        require_bits.append("config model parts: " + ", ".join(requires_parts))
    if "_setup_embeddings_requires_grad" in text:
        require_bits.append("embedding requires_grad setup")
    if frozen_parts:
        require_bits.append("forced frozen: " + ", ".join(frozen_parts))

    param_parts = unique_strings(
        re.findall(r"_create_model_part_parameters\([^,]+,\s*\"([^\"]+)\"", text)
    )
    param_bits: list[str] = []
    if param_parts:
        param_bits.append("parameter groups: " + ", ".join(param_parts))
    if "_add_embedding_param_groups" in text:
        param_bits.append("embedding parameter groups")
    if "ModuleFilter.create(config)" in text:
        param_bits.append("ModuleFilter freeze filter")
    if "init_model_parameters" in text:
        param_bits.append("init_model_parameters")

    return BehaviorSummary(
        requires_grad="; ".join(require_bits) if require_bits else "not detected",
        parameters="; ".join(param_bits) if param_bits else "not detected",
    )


def has_training_method_finetune_mode() -> tuple[bool, str]:
    config = strip_mojo_comments(read_optional(TRAIN_CONFIG))
    reader = strip_mojo_comments(read_optional(TRAIN_CONFIG_READER))
    config_has_field = re.search(r"\bvar\s+training_method\b", config) is not None
    config_has_finetune = (
        "TRAINING_METHOD_FINE_TUNE" in config
        or "TRAINING_METHOD_FULL_FINETUNE" in config
        or "TRAINING_METHOD_FULL_FINE_TUNE" in config
        or re.search(r"\bfine_tune\b", config, flags=re.IGNORECASE) is not None
    )
    reader_reads_key = '"training_method"' in reader
    reader_has_finetune = (
        "TRAINING_METHOD_FINE_TUNE" in reader
        or "TRAINING_METHOD_FULL_FINETUNE" in reader
        or "TRAINING_METHOD_FULL_FINE_TUNE" in reader
        or re.search(r"\bfine_tune\b", reader, flags=re.IGNORECASE) is not None
    )
    ok = config_has_field and config_has_finetune and reader_reads_key and reader_has_finetune
    detail = (
        f"TrainConfig.training_method={config_has_field}, "
        f"TrainConfig.fine_tune_mode={config_has_finetune}, "
        f"reader.training_method_key={reader_reads_key}, "
        f"reader.fine_tune_mapping={reader_has_finetune}"
    )
    return ok, detail


def full_finetune_save_scaffolding() -> tuple[bool, list[Path]]:
    if not TRAINING_DIR.exists():
        return False, []

    hits: list[Path] = []
    patterns = (
        re.compile(r"\b(save|load|resume)_[a-z0-9_]*(full_finetune|full_fine_tune|fine_tune)", re.I),
        re.compile(r"\b(full_finetune|full_fine_tune|fine_tune)_[a-z0-9_]*(save|load|resume)", re.I),
        re.compile(r"\bfull_weight[a-z0-9_]*(save|checkpoint|resume)", re.I),
    )
    for path in sorted(TRAINING_DIR.rglob("*.mojo")):
        text = strip_mojo_comments(read_optional(path))
        if "full_adapter" in path.name:
            continue
        if any(pattern.search(text) for pattern in patterns):
            hits.append(path)
    return bool(hits), hits


def full_finetune_manifest_scaffolding() -> tuple[bool, str]:
    path = TRAINING_DIR / "full_finetune_save.mojo"
    text = strip_mojo_comments(read_optional(path))
    required = (
        "FULL_FINETUNE_NAME_MANIFEST_KEY",
        "def save_full_finetune_name_manifest",
        "def load_full_finetune_name_manifest",
        "def assert_full_finetune_name_manifest_matches",
        "STDtype.U8",
    )
    missing = [item for item in required if item not in text]
    if missing:
        return False, "missing " + ", ".join(missing)
    return True, str(path.relative_to(REPO))


def full_finetune_mojo_contract() -> tuple[bool, str]:
    contract = strip_mojo_comments(read_optional(FULL_FT_CONTRACT))
    smoke = strip_mojo_comments(read_optional(FULL_FT_CONTRACT_SMOKE))
    required = (
        "def full_finetune_targets",
        "def full_finetune_target_for_model_type",
        "def full_finetune_target_for_product_model_type",
        "struct OTFullFinetuneProductRunPlan",
        "def create_full_finetune_product_run_plan",
        "def validate_full_finetune_product_run_plan",
        "def full_finetune_product_run_blocker",
        "def validate_full_finetune_contract",
        "OT_FULL_FT_LOOP_UNSUPPORTED",
        "FULL_FINETUNE_TENSOR_NAME_MANIFEST_KEY",
        "FULL_FINETUNE_PARAM_MASTER_PREFIX",
        "FULL_FINETUNE_ADAM_M_PREFIX",
        "FULL_FINETUNE_ADAM_V_PREFIX",
        "FULL_FINETUNE_META_KEY",
        "FULL_FINETUNE_META_T_STEP_INDEX",
        "FULL_FINETUNE_META_ACCUM_COUNT_INDEX",
        "FULL_FINETUNE_META_FIELD_COUNT",
        "FULL_FINETUNE_OPTIMIZER_DTYPE",
        "def validate_full_finetune_resume_sidecar_spec",
    )
    missing = [item for item in required if item not in contract]
    if "std.gpu" in contract or "DeviceContext" in contract:
        missing.append("contract must not import GPU/DeviceContext")
    if not smoke:
        missing.append(str(FULL_FT_CONTRACT_SMOKE.relative_to(REPO)))
    elif "std.gpu" in smoke or "DeviceContext" in smoke:
        missing.append("smoke must not import GPU/DeviceContext")

    for target in TARGETS:
        if f'String("{target.key}")' not in contract:
            missing.append(f"target key {target.key}")
        if str(target.reference_root) not in contract:
            missing.append(f"reference root for {target.key}")
        for model_type in target.model_types:
            if model_type not in contract:
                missing.append(f"ModelType.{model_type}")

    if missing:
        return False, "missing " + ", ".join(missing)
    return True, str(FULL_FT_CONTRACT.relative_to(REPO))


def optimizer_state_scaffolding() -> tuple[bool, str]:
    text = strip_mojo_comments(read_optional(TRAIN_LOOP))
    required = (
        "struct TrainState",
        "def save_checkpoint",
        "def load_checkpoint",
        '"param."',
        '"adam_m."',
        '"adam_v."',
        '"__meta__"',
    )
    missing = [item for item in required if item not in text]
    if missing:
        return False, "missing " + ", ".join(missing)
    return True, str(TRAIN_LOOP.relative_to(REPO))


def shared_mojo_blockers() -> list[str]:
    blockers: list[str] = []

    mode_ok, mode_detail = has_training_method_finetune_mode()
    if not mode_ok:
        blockers.append(
            "TrainConfig/readers do not expose a OneTrainer-style "
            f"training_method=FINE_TUNE/full-finetune mode ({mode_detail})"
        )

    save_ok, save_hits = full_finetune_save_scaffolding()
    if not save_ok:
        blockers.append("full-finetune full-weight save/load/resume scaffolding not detected")
    else:
        rel_hits = [str(path.relative_to(REPO)) for path in save_hits[:4]]
        if len(save_hits) > 4:
            rel_hits.append(f"... +{len(save_hits) - 4} more")
        print("[full-ft] shared save scaffolding: " + ", ".join(rel_hits))

    manifest_ok, manifest_detail = full_finetune_manifest_scaffolding()
    if not manifest_ok:
        blockers.append(
            "full-finetune tensor-name/order manifest scaffolding not detected "
            f"({manifest_detail})"
        )
    else:
        print("[full-ft] shared tensor-name manifest scaffold: " + manifest_detail)

    opt_ok, opt_detail = optimizer_state_scaffolding()
    if not opt_ok:
        blockers.append(
            "full-finetune optimizer/master-state checkpoint scaffolding not detected "
            f"({opt_detail})"
        )
    else:
        print("[full-ft] shared optimizer state scaffold: " + opt_detail)

    contract_ok, contract_detail = full_finetune_mojo_contract()
    if not contract_ok:
        blockers.append(
            "full-finetune no-CUDA target/sidecar contract not detected "
            f"({contract_detail})"
        )
    else:
        print("[full-ft] no-CUDA target/sidecar contract: " + contract_detail)

    return blockers


def product_loop_proof() -> ProductLoopProof:
    product = strip_mojo_comments(read_optional(PRODUCT_RUN))
    smoke = strip_mojo_comments(read_optional(PRODUCT_RUN_SMOKE))
    lifecycle = strip_mojo_comments(read_optional(REPO / "serenitymojo/training/onetrainer_lifecycle.mojo"))
    support_claimed = (
        "product_loop_supports_full_finetune = True" in product
        or re.search(
            r"OTProductResumePreflight\([^)]*TRAINING_METHOD_FINE_TUNE[^)]*True",
            product,
            flags=re.DOTALL,
        )
        is not None
    )
    fail_loud_smoke = (
        'cfg.training_method = TRAINING_METHOD_FINE_TUNE' in smoke
        and "full finetune product loop" in smoke
        and "_expect_plan_raises" in smoke
    )
    lifecycle_rejects_finetune = "full-finetune product loop is not wired" in lifecycle
    lifecycle_lora_only_validator = "only LORA product runners are wired" in lifecycle
    full_finetune_blocker_preflight = all(
        marker in (product + "\n" + lifecycle)
        for marker in (
            "create_full_finetune_product_run_plan",
            "validate_full_finetune_product_run_plan",
            "full_finetune_product_run_blocker",
        )
    )
    full_runner_dispatch = (
        re.search(
            r"OTProductRunPlan\([^)]*OT_PRODUCT_RUNNER_FULL_FINETUNE_REAL",
            lifecycle,
            flags=re.DOTALL,
        )
        is not None
    )
    positive_full_finetune_smoke = (
        "full finetune product loop PASS" in smoke
        or "expect_full_finetune_product_loop_ready" in smoke
        or re.search(
            r"cfg\.training_method\s*=\s*TRAINING_METHOD_FINE_TUNE"
            r"(?:(?!_expect_plan_raises).){0,700}"
            r"(validate_onetrainer_train_entrypoint_plan|create_onetrainer_train_entrypoint_plan)",
            smoke,
            flags=re.DOTALL,
        )
        is not None
    )
    real_full_finetune_loop = (
        support_claimed
        and full_runner_dispatch
        and positive_full_finetune_smoke
        and not lifecycle_rejects_finetune
        and not lifecycle_lora_only_validator
    )
    if real_full_finetune_loop:
        status = "real_product_loop_full_finetune_parity_evidence_present"
    elif support_claimed or full_runner_dispatch:
        status = "claimed_without_real_product_loop_parity"
    elif fail_loud_smoke and lifecycle_rejects_finetune:
        status = "unsupported_fail_loud_scaffold_only"
    else:
        status = "unsupported_silent_scaffold_only"
    detail = (
        rel(PRODUCT_RUN)
        + " support_claimed="
        + str(support_claimed)
        + "; "
        + rel(PRODUCT_RUN_SMOKE)
        + " fail_loud_smoke="
        + str(fail_loud_smoke)
        + "; "
        + rel(REPO / "serenitymojo/training/onetrainer_lifecycle.mojo")
        + " full_runner_dispatch="
        + str(full_runner_dispatch)
        + " positive_full_finetune_smoke="
        + str(positive_full_finetune_smoke)
        + " lifecycle_rejects_finetune="
        + str(lifecycle_rejects_finetune)
        + " lifecycle_lora_only_validator="
        + str(lifecycle_lora_only_validator)
        + " full_finetune_blocker_preflight="
        + str(full_finetune_blocker_preflight)
    )
    return ProductLoopProof(
        support_claimed,
        fail_loud_smoke,
        lifecycle_rejects_finetune,
        lifecycle_lora_only_validator,
        full_finetune_blocker_preflight,
        full_runner_dispatch,
        positive_full_finetune_smoke,
        real_full_finetune_loop,
        status,
        detail,
    )


def product_full_resume_status() -> tuple[bool, bool, str]:
    proof = product_loop_proof()
    return proof.real_full_finetune_loop, proof.fail_loud_smoke, proof.detail


def product_loop_blockers(proof: ProductLoopProof) -> list[str]:
    if proof.real_full_finetune_loop:
        return []

    blockers: list[str] = []
    if proof.support_claimed or proof.full_runner_dispatch:
        blockers.append(
            "product loop advertises full-finetune support without real "
            "product-loop parity proof (needs full runner dispatch, positive "
            "no-CUDA product smoke, and no LoRA-only validator/rejection)"
        )
    if not proof.fail_loud_smoke or not proof.lifecycle_rejects_finetune:
        blockers.append(
            "unsupported full-finetune product path is silent; strict mode "
            "requires an explicit fail-loud product smoke and lifecycle rejection"
        )
    if not proof.full_finetune_blocker_preflight:
        blockers.append(
            "unsupported full-finetune product path lacks a per-target blocker "
            "preflight with save/load/manifest/resume requirements"
        )
    return blockers


def collect_model_proof(target: Target, product_proof: ProductLoopProof | None = None) -> ModelProof:
    source_files = _target_source_files(target)
    product_proof = product_proof if product_proof is not None else product_loop_proof()
    return ModelProof(
        source_files=source_files,
        inventory_files=_files_with_markers(
            source_files,
            (
                "FullFinetuneTensor",
                "full_finetune_tensor_names",
                "full_finetune_name_manifest",
                "full_weight_tensor_names",
                "full_weight_inventory",
            ),
        ),
        full_save_files=_files_with_markers(
            source_files,
            (
                "save_full_finetune_model_tensors",
                "FullFinetuneTensor",
            ),
        ),
        full_load_files=_files_with_markers(
            source_files,
            (
                "load_full_finetune_model_tensors",
                "assert_full_finetune_name_manifest_matches",
            ),
        ),
        manifest_files=_files_with_markers(
            source_files,
            (
                "save_full_finetune_name_manifest",
                "load_full_finetune_name_manifest",
                "assert_full_finetune_name_manifest_matches",
            ),
        ),
        optimizer_sidecar_files=_files_with_markers(
            source_files,
            (
                "full_finetune",
                "TrainState",
                "param.",
                "adam_m.",
                "adam_v.",
            ),
            require_all=True,
        ),
        resume_files=_files_with_markers(
            source_files,
            (
                "validate_ot_resume_manifest",
                "OT_RESUME_SURFACE_FULL_FINETUNE_INTERNAL",
                "product_loop_supports_full_finetune",
                "has_full_model_payload",
            ),
        ),
        parity_artifacts=_parity_artifacts(target),
        product_resume_full_support=product_proof.real_full_finetune_loop,
        product_resume_fail_loud=product_proof.fail_loud_smoke,
        product_loop_status=product_proof.status,
    )


def _file_list(paths: tuple[Path, ...], limit: int = 3) -> str:
    if not paths:
        return "none"
    names = [rel(path) for path in paths[:limit]]
    if len(paths) > limit:
        names.append(f"... +{len(paths) - limit} more")
    return ", ".join(names)


def full_weight_evidence_blockers(target: Target, proof: ModelProof) -> list[str]:
    blockers: list[str] = []
    if not proof.inventory_files:
        blockers.append(
            "missing model-specific full-weight tensor inventory with exact "
            "OneTrainer checkpoint keys and deterministic trainable-tensor order"
        )
    if not proof.full_save_files:
        blockers.append(
            "missing model-specific full-weight save hook using "
            "save_full_finetune_model_tensors / FullFinetuneTensor"
        )
    if not proof.full_load_files:
        blockers.append(
            "missing model-specific full-weight load/rebind proof using "
            "load_full_finetune_model_tensors and manifest assertion"
        )
    if not proof.manifest_files:
        blockers.append(
            "missing ordered full-weight tensor-name manifest bound to "
            "OneTrainer checkpoint keys"
        )
    if not proof.optimizer_sidecar_files:
        blockers.append(
            "missing full-weight TrainState optimizer/master sidecar binding "
            "(param.N, adam_m.N, adam_v.N order)"
        )
    if not proof.resume_files or not proof.product_resume_full_support:
        if proof.product_resume_fail_loud:
            blockers.append(
                "missing real product-loop full-finetune parity; current "
                f"product status is {proof.product_loop_status}"
            )
        else:
            blockers.append(
                "missing real product-loop full-finetune parity and fail-loud "
                f"coverage; current product status is {proof.product_loop_status}"
            )
    if not proof.parity_artifacts:
        blockers.append(
            "missing full-finetune parity artifact/manifest under "
            "/home/alex/onetrainer-mojo/parity"
        )
    return blockers


def full_weight_readiness_status(
    found: tuple[str, ...],
    shared_count: int,
    loop_count: int,
    evidence_count: int,
    product_proof: ProductLoopProof,
) -> str:
    if not found:
        return "onetrainer_not_registered"
    if not product_proof.real_full_finetune_loop and shared_count == 0:
        return "onetrainer_registered_scaffold_only_product_loop_parity_missing"
    if shared_count or loop_count or evidence_count:
        return "onetrainer_registered_product_loop_parity_incomplete"
    return "full_weight_product_loop_parity_evidence_present"


def loop_status(path: Path) -> tuple[bool, str]:
    text = strip_mojo_comments(read_optional(path))
    if not text:
        return False, f"{path.relative_to(REPO)} missing"

    lower = text.lower()
    lora_markers = (
        "_lora_set" in lower
        or "lora_adamw_step" in lower
        or "save_" in lower and "_lora" in lower
    )
    method_marker = "training_method" in lower or "trainingmethod" in lower
    fine_marker = "fine_tune" in lower or "full_finetune" in lower or "full_fine_tune" in lower
    full_update_marker = (
        "full_finetune" in lower
        or "full_fine_tune" in lower
        or "full_weight" in lower
        or "base_weight_adamw" in lower
    )
    shared_lora_only_policy_marker = (
        "validate_ot_lora_adamw_loop_policy" in text
        or "validate_ot_train_math_policy" in text
        or "validate_ot_lora_only_or_fail_full_finetune" in text
    )
    fail_loud_marker = method_marker and fine_marker and "raise Error" in text

    rel = path.relative_to(REPO)
    if full_update_marker and method_marker and fine_marker:
        return True, f"{rel} has training_method/fine-tune full-weight markers"
    if shared_lora_only_policy_marker:
        return (
            False,
            f"{rel} delegates FINE_TUNE/full-finetune rejection to the shared "
            "LoRA-only policy but still lacks a full-weight update path",
        )
    if fail_loud_marker:
        return False, f"{rel} detects fine-tune mode but still lacks a full-weight update path"
    if lora_markers:
        return False, f"{rel} is LoRA-only and has no training_method/fine-tune dispatch"
    return False, f"{rel} has no detected full-finetune update path"


def loop_blockers(target: Target) -> list[str]:
    blockers: list[str] = []
    for path in target.train_loops:
        ok, message = loop_status(path)
        if not ok:
            blockers.append(message)
    return blockers


def target_registration_status(target: Target, info: SetupInfo | None) -> tuple[str, tuple[str, ...]]:
    if info is None:
        return "unsupported/not registered", ()
    found = tuple(model_type for model_type in target.model_types if model_type in info.registrations)
    if not found:
        return "unsupported/not registered", ()
    missing = tuple(model_type for model_type in target.model_types if model_type not in found)
    if missing:
        return "partially registered; missing " + ", ".join(missing), found
    return "registered", found


def print_target_line(
    target: Target,
    info: SetupInfo | None,
    shared_count: int,
    loop_count: int,
    proof: ModelProof | None,
    evidence_count: int,
    product_proof: ProductLoopProof,
) -> None:
    status, found = target_registration_status(target, info)
    behavior = summarize_setup_behavior(info)
    setup = str(info.path) if info is not None else str(target_setup_path(target))
    regs = ", ".join(found) if found else "none"
    readiness = full_weight_readiness_status(
        found, shared_count, loop_count, evidence_count, product_proof
    )
    if status == "registered" or status.startswith("partially"):
        blocker_status = (
            f"{shared_count + loop_count + evidence_count} blocker(s) "
            f"(shared={shared_count}, loop={loop_count}, evidence={evidence_count})"
        )
    else:
        blocker_status = "unsupported/not registered (not counted)"
    shared_surface = (
        "shared save/load+resume sidecar available"
        if shared_count == 0
        else "shared save/load+resume sidecar blocked"
    )
    if not found:
        loop_surface = "real full-finetune loop not evaluated"
    elif loop_count:
        loop_surface = "real full-finetune loop unsupported"
    else:
        loop_surface = "real full-finetune loop markers detected"
    product_surface = "product loop parity=" + product_proof.status

    print(
        "[full-ft] "
        + target.label
        + " | OneTrainer setup file="
        + setup
        + " | ModelType registrations="
        + regs
        + " | registration status="
        + status
        + " | readiness="
        + readiness
        + " | Mojo surfaces="
        + shared_surface
        + "; "
        + loop_surface
        + "; "
        + product_surface
        + " | requires_grad="
        + behavior.requires_grad
        + " | optimizer/parameter setup="
        + behavior.parameters
        + " | Mojo blocker status="
        + blocker_status
    )
    if proof is not None and found:
        print(
            "[full-ft]   evidence: source files scanned="
            + str(len(proof.source_files))
            + " | full inventory="
            + _file_list(proof.inventory_files)
            + " | full save hooks="
            + _file_list(proof.full_save_files)
            + " | full load hooks="
            + _file_list(proof.full_load_files)
            + " | manifest binding="
            + _file_list(proof.manifest_files)
            + " | optimizer sidecar binding="
            + _file_list(proof.optimizer_sidecar_files)
        )
        print(
            "[full-ft]   resume evidence: product_full_resume_support="
            + str(proof.product_resume_full_support)
            + " | product_loop_status="
            + proof.product_loop_status
            + " | product_fail_loud="
            + str(proof.product_resume_fail_loud)
            + " | resume source hooks="
            + _file_list(proof.resume_files)
            + " | parity artifacts="
            + _file_list(proof.parity_artifacts)
        )


def _json_paths(paths: tuple[Path, ...]) -> list[str]:
    return [rel(path) for path in paths]


def _evidence_bucket(
    present: bool, files: tuple[Path, ...], missing_requirement: str
) -> dict[str, object]:
    return {
        "present": present,
        "files": _json_paths(files),
        "missing_requirement": None if present else missing_requirement,
    }


def model_specific_evidence_report(proof: ModelProof) -> dict[str, object]:
    return {
        "full_weight_inventory": _evidence_bucket(
            bool(proof.inventory_files),
            proof.inventory_files,
            (
                "model-specific deterministic inventory of full-weight tensors "
                "using exact OneTrainer checkpoint keys and the trainable tensor "
                "order used for save/resume sidecars"
            ),
        ),
        "full_weight_save_hook": _evidence_bucket(
            bool(proof.full_save_files),
            proof.full_save_files,
            (
                "model-specific hook that builds FullFinetuneTensor entries and "
                "calls save_full_finetune_model_tensors for the full-weight payload"
            ),
        ),
        "full_weight_load_rebind": _evidence_bucket(
            bool(proof.full_load_files),
            proof.full_load_files,
            (
                "model-specific load_full_finetune_model_tensors path that "
                "asserts the saved manifest and rebinds loaded tensors to the "
                "model struct used by the product loop"
            ),
        ),
        "tensor_name_manifest_binding": _evidence_bucket(
            bool(proof.manifest_files),
            proof.manifest_files,
            (
                "ordered tensor-name manifest bound to the OneTrainer checkpoint "
                "keys and reused for save, TrainState param.N order, and resume"
            ),
        ),
        "optimizer_master_sidecar_binding": _evidence_bucket(
            bool(proof.optimizer_sidecar_files),
            proof.optimizer_sidecar_files,
            (
                "full-weight TrainState sidecar mapping for param.N, adam_m.N, "
                "adam_v.N, and __meta__ using the same tensor-name manifest order"
            ),
        ),
        "resume_mapping": _evidence_bucket(
            bool(proof.resume_files) and proof.product_resume_full_support,
            proof.resume_files,
            (
                "product-loop full-finetune resume manifest/load mapping with "
                "product_loop_supports_full_finetune=true only after a real "
                "full-weight runner is wired"
            ),
        ),
        "parity_artifact": _evidence_bucket(
            bool(proof.parity_artifacts),
            proof.parity_artifacts,
            (
                "full-finetune parity artifact or manifest under "
                "/home/alex/onetrainer-mojo/parity proving inventory/save/load/"
                "resume behavior against OneTrainer"
            ),
        ),
    }


TEMPLATE_SECTION_SPECS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "inventory",
        "full_weight_inventory",
        "Model-specific full-weight tensor inventory",
        (
            "List every trainable full-weight tensor in deterministic update order.",
            "Use exact OneTrainer checkpoint keys, not Mojo-only aliases.",
            "Record shape, dtype, owning model part, and trainable/frozen reason.",
        ),
    ),
    (
        "save",
        "full_weight_save_hook",
        "Model-specific full-weight save hook",
        (
            "Build FullFinetuneTensor entries from the inventory order.",
            "Call save_full_finetune_model_tensors for the full-weight payload.",
            "Keep BF16/F16 storage dtype at tensor boundaries unless the checkpoint requires otherwise.",
        ),
    ),
    (
        "load",
        "full_weight_load_rebind",
        "Model-specific full-weight load/rebind path",
        (
            "Call load_full_finetune_model_tensors for the payload.",
            "Assert the saved manifest before rebinding tensors to the runtime model.",
            "Document every key that is intentionally skipped or frozen.",
        ),
    ),
    (
        "manifest",
        "tensor_name_manifest_binding",
        "Ordered tensor-name manifest binding",
        (
            "Save the exact inventory name order with save_full_finetune_name_manifest.",
            "Reuse the same order for save, optimizer sidecars, and resume.",
            "Assert manifest equality on load before accepting any tensor payload.",
        ),
    ),
    (
        "optimizer",
        "optimizer_master_sidecar_binding",
        "Optimizer/master-state sidecar binding",
        (
            "Map TrainState param.N, adam_m.N, adam_v.N, and __meta__ to manifest order.",
            "Preserve master-state dtype contract and step/accumulation counters.",
            "Prove resume can restore optimizer state without reordering tensors.",
        ),
    ),
    (
        "resume",
        "resume_mapping",
        "Product-loop resume mapping",
        (
            "Wire resume manifest validation to the model-specific load path.",
            "Keep product_loop_supports_full_finetune false until a real runner is wired.",
            "Fail loudly for unsupported full-finetune resume requests.",
        ),
    ),
    (
        "parity",
        "parity_artifact",
        "OneTrainer parity artifact",
        (
            "Write a parity manifest under /home/alex/onetrainer-mojo/parity.",
            "Prove inventory order, save payload, load/rebind behavior, and resume sidecar mapping.",
            "Record artifact paths, command lines, tensor counts, dtypes, and accepted tolerances.",
        ),
    ),
)


TEMPLATE_SECTION_ARTIFACT_KEYS: dict[str, str] = {
    "inventory": "trainable_tensor_inventory",
    "save": "save_load_rebind_hooks",
    "load": "save_load_rebind_hooks",
    "manifest": "ordered_checkpoint_key_manifest",
    "optimizer": "optimizer_master_state_sidecar_order",
    "resume": "product_resume_proof",
    "parity": "parity_artifact_paths",
}


TARGET_TEMPLATE_ARTIFACTS: dict[str, dict[str, dict[str, object]]] = {
    "flux2": {
        "trainable_tensor_inventory": {
            "status": "missing_next_artifact",
            "scope": (
                "OneTrainer Flux2/Klein FINE_TUNE trains model.transformer "
                "only; model.vae and model.text_encoder are frozen. Apply "
                "ModuleFilter.create(config) before accepting a tensor as trainable."
            ),
            "implementation_files": [
                (
                    "serenitymojo/models/klein/weights.mojo::add "
                    "klein_full_finetune_trainable_tensors(...)"
                ),
                (
                    "serenitymojo/models/klein/klein_stack.mojo::bind "
                    "KleinStackBase, DoubleBlockWeights, and SingleBlockWeights "
                    "fields to inventory entries"
                ),
                (
                    "serenitymojo/training/train_klein_real.mojo::build the "
                    "full-weight TrainState param list from the inventory order"
                ),
            ],
            "onetrainer_reference_files": [
                (
                    "/home/alex/OneTrainer/modules/modelSetup/"
                    "Flux2FineTuneSetup.py::create_parameters and "
                    "__setup_requires_grad"
                ),
                (
                    "/home/alex/OneTrainer/modules/model/Flux2Model.py::"
                    "diffusers_checkpoint_to_original"
                ),
                (
                    "/home/alex/OneTrainer/modules/modelSaver/flux2/"
                    "Flux2ModelSaver.py::__save_safetensors"
                ),
            ],
            "checkpoint_key_families": [
                "img_in.*",
                "txt_in.*",
                "time_in.{in_layer,out_layer}.*",
                "guidance_in.{in_layer,out_layer}.*",
                "double_stream_modulation_{img,txt}.lin.*",
                "single_stream_modulation.lin.*",
                "final_layer.linear.*",
                "final_layer.adaLN_modulation.1.*",
                (
                    "double_blocks.{i}.{img,txt}_attn."
                    "{qkv,proj,norm.query_norm,norm.key_norm}.*"
                ),
                "double_blocks.{i}.{img,txt}_mlp.{0,2}.*",
                "single_blocks.{i}.{linear1,linear2,norm.query_norm,norm.key_norm}.*",
            ],
            "acceptance_evidence": [
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_tensor_inventory.json",
            ],
        },
        "save_load_rebind_hooks": {
            "status": "missing_next_artifact",
            "implementation_files": [
                (
                    "serenitymojo/models/klein/weights.mojo::add "
                    "save_klein_full_finetune_model_tensors(...)"
                ),
                (
                    "serenitymojo/models/klein/weights.mojo::add "
                    "load_klein_full_finetune_model_tensors(...)"
                ),
                (
                    "serenitymojo/models/klein/weights.mojo::add "
                    "rebind_klein_full_finetune_model_tensors(...)"
                ),
                (
                    "serenitymojo/training/train_klein_real.mojo::dispatch "
                    "FINE_TUNE to these hooks instead of the LoRA-only path"
                ),
            ],
            "shared_scaffold_files": [
                "serenitymojo/training/full_finetune_save.mojo::FullFinetuneTensor",
                (
                    "serenitymojo/training/full_finetune_save.mojo::"
                    "save_full_finetune_model_tensors"
                ),
                (
                    "serenitymojo/training/full_finetune_save.mojo::"
                    "load_full_finetune_model_tensors"
                ),
            ],
            "onetrainer_reference_files": [
                "/home/alex/OneTrainer/modules/modelSaver/Flux2FineTuneModelSaver.py",
                "/home/alex/OneTrainer/modules/modelLoader/Flux2ModelLoader.py",
                "/home/alex/OneTrainer/modules/modelSaver/flux2/Flux2ModelSaver.py",
            ],
            "acceptance_evidence": [
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_step000_model.safetensors",
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_rebind_roundtrip.json",
            ],
        },
        "ordered_checkpoint_key_manifest": {
            "status": "missing_next_artifact",
            "implementation_files": [
                (
                    "serenitymojo/models/klein/weights.mojo::add "
                    "klein_full_finetune_checkpoint_key_manifest(...)"
                ),
                (
                    "serenitymojo/training/train_klein_real.mojo::save and load "
                    "the manifest next to each full-finetune model payload"
                ),
            ],
            "shared_scaffold_files": [
                (
                    "serenitymojo/training/full_finetune_save.mojo::"
                    "save_full_finetune_name_manifest"
                ),
                (
                    "serenitymojo/training/full_finetune_save.mojo::"
                    "assert_full_finetune_name_manifest_matches"
                ),
            ],
            "acceptance_evidence": [
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_checkpoint_key_manifest.json",
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_name_manifest.safetensors",
            ],
        },
        "optimizer_master_state_sidecar_order": {
            "status": "missing_next_artifact",
            "implementation_files": [
                (
                    "serenitymojo/training/train_klein_real.mojo::construct "
                    "TrainState params/m/v in the exact manifest order"
                ),
                (
                    "serenitymojo/training/loop.mojo::save_checkpoint/load_checkpoint "
                    "param.N, adam_m.N, adam_v.N, __meta__ sidecars"
                ),
                (
                    "serenitymojo/training/full_finetune_contract.mojo::keep "
                    "FULL_FINETUNE_PARAM_MASTER_PREFIX/FULL_FINETUNE_ADAM_* "
                    "bound to the manifest order"
                ),
            ],
            "acceptance_evidence": [
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_optimizer_sidecar_order.json",
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_step000_train_state.safetensors",
            ],
        },
        "product_resume_proof": {
            "status": "missing_next_artifact",
            "implementation_files": [
                (
                    "serenitymojo/training/onetrainer_lifecycle.mojo::replace "
                    "LoRA-only FINE_TUNE rejection with "
                    "OT_PRODUCT_RUNNER_FULL_FINETUNE_REAL only after model hooks exist"
                ),
                (
                    "serenitymojo/training/onetrainer_product_run.mojo::set "
                    "product_loop_supports_full_finetune=true only after "
                    "resume preflight validates model payload, manifest, and TrainState"
                ),
                (
                    "serenitymojo/training/train_klein_real.mojo::prove resumed "
                    "model tensors and optimizer sidecars produce the same next update"
                ),
            ],
            "acceptance_evidence": [
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_product_resume_proof.json",
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_resume_roundtrip.safetensors",
            ],
        },
        "parity_artifact_paths": {
            "status": "missing_next_artifact",
            "required_paths": [
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_tensor_inventory.json",
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_checkpoint_key_manifest.json",
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_step000_model.safetensors",
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_name_manifest.safetensors",
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_step000_train_state.safetensors",
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_optimizer_sidecar_order.json",
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_product_resume_proof.json",
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_parity_report.json",
            ],
            "mojo_consumers_to_add": [
                "serenitymojo/models/klein/parity/klein_full_finetune_artifact_smoke.mojo",
            ],
            "acceptance_evidence": [
                "/home/alex/onetrainer-mojo/parity/klein_full_finetune_parity_report.json",
            ],
        },
    },
    "zimage": {
        "trainable_tensor_inventory": {
            "status": "missing_next_artifact",
            "scope": (
                "OneTrainer Z-Image FINE_TUNE trains model.transformer only; "
                "model.vae and model.text_encoder are frozen. Apply "
                "ModuleFilter.create(config) before accepting a tensor as trainable."
            ),
            "implementation_files": [
                (
                    "serenitymojo/models/zimage/weights.mojo::add "
                    "zimage_full_finetune_block_trainable_tensors(...)"
                ),
                (
                    "serenitymojo/models/zimage/real_weights.mojo::add "
                    "zimage_full_finetune_aux_trainable_tensors(...)"
                ),
                (
                    "serenitymojo/training/train_zimage_real.mojo::build the "
                    "full-weight TrainState param list from the inventory order"
                ),
            ],
            "onetrainer_reference_files": [
                (
                    "/home/alex/OneTrainer/modules/modelSetup/"
                    "ZImageFineTuneSetup.py::create_parameters and "
                    "__setup_requires_grad"
                ),
                (
                    "/home/alex/OneTrainer/modules/modelSaver/zImage/"
                    "ZImageModelSaver.py::__save_safetensors"
                ),
                "/home/alex/OneTrainer/modules/modelLoader/ZImageModelLoader.py",
            ],
            "checkpoint_key_families": [
                "t_embedder.mlp.{0,2}.{weight,bias}",
                "cap_embedder.{0.weight,1.weight,1.bias}",
                "all_x_embedder.2-1.{weight,bias}",
                "x_pad_token",
                "cap_pad_token",
                (
                    "noise_refiner.{i}.{attention_norm1,attention_norm2,"
                    "ffn_norm1,ffn_norm2}.weight"
                ),
                (
                    "noise_refiner.{i}.attention."
                    "{to_q,to_k,to_v,to_out.0,norm_q,norm_k}.weight"
                ),
                "noise_refiner.{i}.feed_forward.{w1,w2,w3}.weight",
                "noise_refiner.{i}.adaLN_modulation.0.{weight,bias}",
                "context_refiner.{i}.*",
                "layers.{i}.*",
                "all_final_layer.2-1.adaLN_modulation.1.{weight,bias}",
                "all_final_layer.2-1.linear.{weight,bias}",
            ],
            "acceptance_evidence": [
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_tensor_inventory.json",
            ],
        },
        "save_load_rebind_hooks": {
            "status": "missing_next_artifact",
            "implementation_files": [
                (
                    "serenitymojo/models/zimage/weights.mojo::add "
                    "save_zimage_full_finetune_model_tensors(...)"
                ),
                (
                    "serenitymojo/models/zimage/weights.mojo::add "
                    "load_zimage_full_finetune_model_tensors(...)"
                ),
                (
                    "serenitymojo/models/zimage/real_weights.mojo::add "
                    "rebind_zimage_full_finetune_aux_tensors(...)"
                ),
                (
                    "serenitymojo/training/train_zimage_real.mojo::dispatch "
                    "FINE_TUNE to these hooks instead of the LoRA-only path"
                ),
            ],
            "shared_scaffold_files": [
                "serenitymojo/training/full_finetune_save.mojo::FullFinetuneTensor",
                (
                    "serenitymojo/training/full_finetune_save.mojo::"
                    "save_full_finetune_model_tensors"
                ),
                (
                    "serenitymojo/training/full_finetune_save.mojo::"
                    "load_full_finetune_model_tensors"
                ),
            ],
            "onetrainer_reference_files": [
                "/home/alex/OneTrainer/modules/modelSaver/ZImageFineTuneModelSaver.py",
                "/home/alex/OneTrainer/modules/modelLoader/ZImageModelLoader.py",
                "/home/alex/OneTrainer/modules/modelSaver/zImage/ZImageModelSaver.py",
            ],
            "acceptance_evidence": [
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_step000_model.safetensors",
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_rebind_roundtrip.json",
            ],
        },
        "ordered_checkpoint_key_manifest": {
            "status": "missing_next_artifact",
            "implementation_files": [
                (
                    "serenitymojo/models/zimage/weights.mojo::add "
                    "zimage_full_finetune_checkpoint_key_manifest(...)"
                ),
                (
                    "serenitymojo/training/train_zimage_real.mojo::save and load "
                    "the manifest next to each full-finetune model payload"
                ),
            ],
            "shared_scaffold_files": [
                (
                    "serenitymojo/training/full_finetune_save.mojo::"
                    "save_full_finetune_name_manifest"
                ),
                (
                    "serenitymojo/training/full_finetune_save.mojo::"
                    "assert_full_finetune_name_manifest_matches"
                ),
            ],
            "acceptance_evidence": [
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_checkpoint_key_manifest.json",
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_name_manifest.safetensors",
            ],
        },
        "optimizer_master_state_sidecar_order": {
            "status": "missing_next_artifact",
            "implementation_files": [
                (
                    "serenitymojo/training/train_zimage_real.mojo::construct "
                    "TrainState params/m/v in the exact manifest order"
                ),
                (
                    "serenitymojo/training/loop.mojo::save_checkpoint/load_checkpoint "
                    "param.N, adam_m.N, adam_v.N, __meta__ sidecars"
                ),
                (
                    "serenitymojo/training/full_finetune_contract.mojo::keep "
                    "FULL_FINETUNE_PARAM_MASTER_PREFIX/FULL_FINETUNE_ADAM_* "
                    "bound to the manifest order"
                ),
            ],
            "acceptance_evidence": [
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_optimizer_sidecar_order.json",
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_step000_train_state.safetensors",
            ],
        },
        "product_resume_proof": {
            "status": "missing_next_artifact",
            "implementation_files": [
                (
                    "serenitymojo/training/onetrainer_lifecycle.mojo::replace "
                    "LoRA-only FINE_TUNE rejection with "
                    "OT_PRODUCT_RUNNER_FULL_FINETUNE_REAL only after model hooks exist"
                ),
                (
                    "serenitymojo/training/onetrainer_product_run.mojo::set "
                    "product_loop_supports_full_finetune=true only after "
                    "resume preflight validates model payload, manifest, and TrainState"
                ),
                (
                    "serenitymojo/training/train_zimage_real.mojo::prove resumed "
                    "model tensors and optimizer sidecars produce the same next update"
                ),
            ],
            "acceptance_evidence": [
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_product_resume_proof.json",
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_resume_roundtrip.safetensors",
            ],
        },
        "parity_artifact_paths": {
            "status": "missing_next_artifact",
            "required_paths": [
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_tensor_inventory.json",
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_checkpoint_key_manifest.json",
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_step000_model.safetensors",
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_name_manifest.safetensors",
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_step000_train_state.safetensors",
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_optimizer_sidecar_order.json",
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_product_resume_proof.json",
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_parity_report.json",
            ],
            "mojo_consumers_to_add": [
                "serenitymojo/models/zimage/parity/zimage_full_finetune_artifact_smoke.mojo",
            ],
            "acceptance_evidence": [
                "/home/alex/onetrainer-mojo/parity/zimage_full_finetune_parity_report.json",
            ],
        },
    },
}


def _target_template_slug(target: Target) -> str:
    if target.key == "flux2":
        return "klein"
    return target.key


def _target_template_artifacts(target: Target) -> dict[str, dict[str, object]]:
    return TARGET_TEMPLATE_ARTIFACTS.get(target.key, {})


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _template_section(
    target: Target,
    record: dict[str, object],
    section_name: str,
    evidence_key: str,
    title: str,
    required_work: tuple[str, ...],
) -> dict[str, object]:
    evidence = record.get("model_specific_evidence", {})
    bucket = evidence.get(evidence_key, {}) if isinstance(evidence, dict) else {}
    present = bool(bucket.get("present", False)) if isinstance(bucket, dict) else False
    files = bucket.get("files", []) if isinstance(bucket, dict) else []
    missing = bucket.get("missing_requirement") if isinstance(bucket, dict) else None
    artifact_key = TEMPLATE_SECTION_ARTIFACT_KEYS.get(section_name, "")
    artifact = _target_template_artifacts(target).get(artifact_key, {})
    if not isinstance(artifact, dict):
        artifact = {}
    return {
        "section": section_name,
        "readiness_key": evidence_key,
        "title": title,
        "status": "template_only_not_evidence",
        "next_artifact_key": artifact_key,
        "next_artifact_status": artifact.get("status", "generic_missing_next_artifact"),
        "scope": artifact.get("scope"),
        "present_in_current_scan": present,
        "existing_files": files if isinstance(files, list) else [],
        "missing_requirement": missing,
        "required_work": list(required_work),
        "implementation_files_to_fill": _string_list(artifact.get("implementation_files")),
        "shared_scaffold_files": _string_list(artifact.get("shared_scaffold_files")),
        "onetrainer_reference_files_to_fill": _string_list(
            artifact.get("onetrainer_reference_files")
        ),
        "checkpoint_key_families_to_manifest": _string_list(
            artifact.get("checkpoint_key_families")
        ),
        "required_parity_paths_to_fill": _string_list(artifact.get("required_paths")),
        "mojo_consumers_to_add": _string_list(artifact.get("mojo_consumers_to_add")),
        "acceptance_evidence_to_fill": _string_list(artifact.get("acceptance_evidence")),
    }


def full_finetune_target_template(target: Target, record: dict[str, object]) -> dict[str, object]:
    sections = {
        section_name: _template_section(
            target, record, section_name, evidence_key, title, required_work
        )
        for section_name, evidence_key, title, required_work in TEMPLATE_SECTION_SPECS
    }
    return {
        "schema": "mojodiffusion.full_finetune_target_template.v1",
        "no_cuda": True,
        "template_only": True,
        "claim": (
            "not_support_evidence; this template only enumerates missing "
            "model-specific full-finetune work"
        ),
        "full_finetune_ready": bool(record.get("full_finetune_ready", False)),
        "support_claim": "not_claimed_by_template",
        "target": {
            "key": target.key,
            "template_slug": _target_template_slug(target),
            "label": target.label,
            "aliases": list(target.aliases),
            "model_types": list(target.model_types),
            "registered_model_types": record.get("registered_model_types", []),
            "registration_status": record.get("registration_status", "unknown"),
            "setup_file": str(target_setup_path(target)),
            "reference_root": str(target.reference_root),
            "train_loops": record.get("train_loops", []),
        },
        "next_implementation_artifacts": _target_template_artifacts(target),
        "current_blockers": {
            "loop": record.get("loop_blockers", []),
            "model_specific": record.get("model_specific_blockers", []),
            "product_loop": record.get("product_loop", {}),
        },
        "sections": sections,
    }


def write_full_finetune_templates(
    directory: Path, targets: list[Target], records: list[dict[str, object]]
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    records_by_key = {
        str(record.get("key")): record
        for record in records
        if record.get("key") is not None
    }
    written: list[Path] = []
    for target in targets:
        record = records_by_key[target.key]
        path = directory / (_target_template_slug(target) + "_full_finetune_template.json")
        path.write_text(
            json.dumps(full_finetune_target_template(target, record), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def target_readiness_record(
    target: Target,
    info: SetupInfo | None,
    shared: list[str],
    product_blockers: list[str],
    product_proof: ProductLoopProof,
    loop_blockers_for_target: list[str],
    proof: ModelProof | None,
    evidence_blockers_for_target: list[str],
) -> dict[str, object]:
    status, found = target_registration_status(target, info)
    behavior = summarize_setup_behavior(info)
    readiness = full_weight_readiness_status(
        found,
        len(shared),
        len(loop_blockers_for_target),
        len(evidence_blockers_for_target),
        product_proof,
    )
    full_finetune_ready = (
        bool(found)
        and not shared
        and not product_blockers
        and not loop_blockers_for_target
        and not evidence_blockers_for_target
        and product_proof.real_full_finetune_loop
    )
    record: dict[str, object] = {
        "key": target.key,
        "label": target.label,
        "aliases": list(target.aliases),
        "model_types": list(target.model_types),
        "registered_model_types": list(found),
        "registration_status": status,
        "setup_file": str(target_setup_path(target)),
        "reference_root": str(target.reference_root),
        "train_loops": _json_paths(target.train_loops),
        "readiness": readiness,
        "full_finetune_ready": full_finetune_ready,
        "support_claim": (
            "not_claimed" if not product_proof.real_full_finetune_loop else "claimed_by_product_loop"
        ),
        "requires_grad": behavior.requires_grad,
        "optimizer_parameter_setup": behavior.parameters,
        "loop_blockers": loop_blockers_for_target,
        "model_specific_blockers": evidence_blockers_for_target,
        "product_loop": {
            "status": product_proof.status,
            "support_claimed": product_proof.support_claimed,
            "full_runner_dispatch": product_proof.full_runner_dispatch,
            "positive_full_finetune_smoke": product_proof.positive_full_finetune_smoke,
            "fail_loud_smoke": product_proof.fail_loud_smoke,
            "lifecycle_rejects_finetune": product_proof.lifecycle_rejects_finetune,
            "lifecycle_lora_only_validator": product_proof.lifecycle_lora_only_validator,
            "full_finetune_blocker_preflight": product_proof.full_finetune_blocker_preflight,
            "detail": product_proof.detail,
            "global_blockers": product_blockers,
        },
    }
    if proof is None:
        record["source_files_scanned"] = 0
        record["model_specific_evidence"] = {}
        record["resume_evidence"] = {
            "product_full_resume_support": False,
            "product_fail_loud": product_proof.fail_loud_smoke,
            "resume_source_hooks": [],
            "parity_artifacts": [],
        }
    else:
        record["source_files_scanned"] = len(proof.source_files)
        record["model_specific_evidence"] = model_specific_evidence_report(proof)
        record["resume_evidence"] = {
            "product_full_resume_support": proof.product_resume_full_support,
            "product_loop_status": proof.product_loop_status,
            "product_fail_loud": proof.product_resume_fail_loud,
            "resume_source_hooks": _json_paths(proof.resume_files),
            "parity_artifacts": _json_paths(proof.parity_artifacts),
        }
    return record


def write_readiness_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory OneTrainer full-finetune registrations and Mojo readiness blockers."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail while any registered target has Mojo full-finetune blockers",
    )
    parser.add_argument(
        "--target-model",
        action="append",
        help="optional repeated target filter, e.g. qwen, flux2, sdxl, sd35",
    )
    parser.add_argument(
        "--write-readiness",
        type=Path,
        help=(
            "write a no-CUDA JSON readiness report with exact missing "
            "model-specific full-weight inventory/save/load/resume evidence"
        ),
    )
    parser.add_argument(
        "--write-template-dir",
        type=Path,
        help=(
            "write no-CUDA per-target full-finetune implementation templates "
            "for the selected targets"
        ),
    )
    args = parser.parse_args()

    selected = select_targets(args.target_model, parser)
    setup_infos = scan_onetrainer_setups(selected)
    shared = shared_mojo_blockers()
    product_proof = product_loop_proof()
    product = product_loop_blockers(product_proof)

    registered: list[str] = []
    absent: list[str] = []
    target_reports: list[dict[str, object]] = []
    loop_total = 0
    evidence_total = 0

    if shared:
        print("[full-ft] shared Mojo blockers:")
        for blocker in shared:
            print("[full-ft]   - " + blocker)

    print(
        "[full-ft] product full-finetune loop parity: "
        + product_proof.status
        + "; "
        + product_proof.detail
    )
    if product:
        print("[full-ft] product-loop claim/fail-loud blockers:")
        for blocker in product:
            print("[full-ft]   - " + blocker)

    for target in selected:
        info = setup_infos.get(target.key)
        status, found = target_registration_status(target, info)
        loops = loop_blockers(target) if found else []
        proof = collect_model_proof(target, product_proof) if found else None
        evidence = full_weight_evidence_blockers(target, proof) if proof else []
        if found:
            registered.append(target.label + " (" + ", ".join(found) + ")")
            loop_total += len(loops)
            evidence_total += len(evidence)
        else:
            absent.append(target.label)
        target_reports.append(
            target_readiness_record(
                target,
                info,
                shared,
                product,
                product_proof,
                loops,
                proof,
                evidence,
            )
        )

        print_target_line(
            target,
            info,
            len(shared),
            len(loops),
            proof,
            len(evidence),
            product_proof,
        )
        for blocker in loops:
            print("[full-ft]   - loop: " + blocker)
        for blocker in evidence:
            print("[full-ft]   - evidence: " + blocker)

    total_blockers = len(shared) + len(product) + loop_total + evidence_total
    print("[full-ft] registered targets: " + (", ".join(registered) if registered else "none"))
    print("[full-ft] absent/unsupported targets: " + (", ".join(absent) if absent else "none"))
    print(
        "[full-ft] Mojo blocker count: "
        + str(total_blockers)
        + f" (shared={len(shared)}, product={len(product)}, loop={loop_total}, evidence={evidence_total})"
    )

    readiness_report: dict[str, object] = {
        "schema": "mojodiffusion.full_finetune_readiness.v1",
        "no_cuda": True,
        "claim": (
            "blocker_inventory_only; do not claim full-finetune support unless "
            "full_finetune_ready is true for the target and strict mode passes"
        ),
        "selected_targets": [target.key for target in selected],
        "registered_targets": registered,
        "absent_or_unsupported_targets": absent,
        "shared_blockers": shared,
        "product_loop": {
            "status": product_proof.status,
            "support_claimed": product_proof.support_claimed,
            "full_runner_dispatch": product_proof.full_runner_dispatch,
            "positive_full_finetune_smoke": product_proof.positive_full_finetune_smoke,
            "fail_loud_smoke": product_proof.fail_loud_smoke,
            "lifecycle_rejects_finetune": product_proof.lifecycle_rejects_finetune,
            "lifecycle_lora_only_validator": product_proof.lifecycle_lora_only_validator,
            "full_finetune_blocker_preflight": product_proof.full_finetune_blocker_preflight,
            "detail": product_proof.detail,
            "blockers": product,
        },
        "targets": target_reports,
        "totals": {
            "blockers": total_blockers,
            "shared": len(shared),
            "product": len(product),
            "loop": loop_total,
            "model_specific_evidence": evidence_total,
        },
    }
    if args.write_readiness is not None:
        write_readiness_report(args.write_readiness, readiness_report)
        print("[full-ft] wrote readiness report: " + str(args.write_readiness))
    if args.write_template_dir is not None:
        for path in write_full_finetune_templates(
            args.write_template_dir, selected, target_reports
        ):
            print("[full-ft] wrote target template: " + str(path))

    if args.strict and total_blockers:
        print("[full-ft] FAIL: strict mode requires all registered full-finetune blockers to be closed")
        return 1
    if total_blockers:
        print("[full-ft] report-only PASS; use --strict for the production gate")
    else:
        print("[full-ft] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
