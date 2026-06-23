#!/usr/bin/env python3
"""Qwen-Image OneTrainer one-step replay readiness guard.

This is intentionally bounded: it inspects local OneTrainer dump artifacts and
current Mojo source files only. It reads safetensors headers without loading
tensor payloads, so the 1.6G adapter dump stays cheap to verify.
"""

from __future__ import annotations

import json
import re
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REPO = Path("/home/alex/mojodiffusion")
PARITY = Path("/home/alex/onetrainer-mojo/parity")
OT_ROOT = Path("/home/alex/OneTrainer")

CONTRACT = PARITY / "qwen_train_ref_contract.json"
META = PARITY / "qwen_train_ref_meta.json"
PROFILE_META = PARITY / "qwen_train_profile_meta.json"
STEP_DUMP = PARITY / "qwen_train_ref_step000.safetensors"
ADAPTER_DUMP = PARITY / "qwen_train_ref_step000_adapters.safetensors"


@dataclass(frozen=True)
class TensorSpec:
    key: str
    dtype: str
    shape: tuple[int, ...]


REQUIRED_STEP_TENSORS: tuple[TensorSpec, ...] = (
    TensorSpec("batch.latent_image", "BF16", (2, 16, 1, 64, 64)),
    TensorSpec("batch.tokens", "I64", (2, 512)),
    TensorSpec("batch.tokens_mask", "I64", (2, 512)),
    TensorSpec("batch.text_encoder_hidden_state", "BF16", (2, 512, 3584)),
    TensorSpec("batch.loss_weight", "F32", (2,)),
    TensorSpec("trace.encode_text.tokens", "I64", (2, 512)),
    TensorSpec("trace.encode_text.tokens_mask", "I64", (2, 512)),
    TensorSpec("trace.encode_text.cached_hidden_state", "BF16", (2, 512, 3584)),
    TensorSpec("trace.text_encoder_output", "BF16", (2, 144, 3584)),
    TensorSpec("trace.text_attention_mask", "BOOL", (2, 144)),
    TensorSpec("trace.latent_image_before_scale", "BF16", (2, 16, 1, 64, 64)),
    TensorSpec("trace.scaled_latent_image", "F32", (2, 16, 1, 64, 64)),
    TensorSpec("trace.noise_source_tensor", "F32", (2, 16, 1, 64, 64)),
    TensorSpec("trace.latent_noise", "F32", (2, 16, 1, 64, 64)),
    TensorSpec("trace.scaled_noisy_latent_image", "F32", (2, 16, 1, 64, 64)),
    TensorSpec("trace.sigma", "F32", (2, 1, 1, 1, 1)),
    TensorSpec("trace.latent_input", "F32", (2, 16, 1, 64, 64)),
    TensorSpec("trace.packed_latent_input", "F32", (2, 1024, 64)),
    TensorSpec("trace.transformer_hidden_states", "BF16", (2, 1024, 64)),
    TensorSpec("trace.transformer_timestep", "F32", (2,)),
    TensorSpec("trace.encoder_hidden_states", "BF16", (2, 144, 3584)),
    TensorSpec("trace.encoder_hidden_states_mask", "BOOL", (2, 144)),
    TensorSpec("trace.packed_predicted_flow", "BF16", (2, 1024, 64)),
    TensorSpec("trace.flow", "F32", (2, 16, 1, 64, 64)),
    TensorSpec("output.timestep", "I32", (2,)),
    TensorSpec("output.predicted", "BF16", (2, 16, 1, 64, 64)),
    TensorSpec("output.target", "F32", (2, 16, 1, 64, 64)),
    TensorSpec("output.loss_pre_scale", "F32", ()),
    TensorSpec("output.loss_for_backward", "F32", ()),
)

ADAPTER_PREFIXES = ("adapter_before", "adapter_pre", "adapter_post", "adapter_after")
EXPECTED_TRAINABLE_PARAMS = 1440
EXPECTED_ADAPTER_KEYS = EXPECTED_TRAINABLE_PARAMS * len(ADAPTER_PREFIXES)
EXPECTED_ADAPTER_NUMEL_PER_PHASE = 106168320
EXPECTED_OPTIMIZER_STATE_TENSORS = EXPECTED_TRAINABLE_PARAMS * 3
EXPECTED_OPTIMIZER_STATE_NUMEL = 212338080
EXPECTED_OPTIMIZER_STATE_KEYS = ("exp_avg", "exp_avg_sq", "step")
UPDATE_BEARING_STEP_INDEX = 1
QWEN_ADAPTER_UPDATE_GATE = REPO / "scripts/check_qwen_adapter_update_replay.py"

MOJO_GATES: tuple[tuple[Path, str, str], ...] = (
    (
        REPO / "serenitymojo/models/qwenimage/parity/qwen_train_ref_artifact_smoke.mojo",
        "artifact-header",
        "opens qwen_train_ref_step000 and adapter dumps and validates the OneTrainer tensor contract",
    ),
    (
        REPO / "../serenity-trainer/smoke/qwen_train_control_wiring_smoke.mojo",
        "source-only",
        "validates config/sample/save/offload controls; does not consume the dump",
    ),
    (
        REPO / "serenitymojo/models/qwenimage/parity/qwenimage_block_lora_parity.mojo",
        "block-oracle",
        "consumes local block oracle .bin files, not the OneTrainer one-step dump",
    ),
    (
        REPO / "serenitymojo/models/qwenimage/parity/qwenimage_real_smoke.mojo",
        "synthetic",
        "runs real-dim finite fwd/bwd/AdamW smoke on synthetic inputs",
    ),
    (
        REPO / "serenitymojo/models/qwenimage/parity/qwen_lora_resume_state_smoke.mojo",
        "save-resume",
        "exercises synthetic LoRA save/resume state, not the OneTrainer dump",
    ),
    (
        REPO / "serenitymojo/models/qwenimage/parity/qwen_lora_ot_save_key_smoke.mojo",
        "save-key",
        "emits reduced OneTrainer-format LoRA keys, not the one-step replay dump",
    ),
)

NEXT_REPLAY_GATES = (
    (
        "predict/loss replay",
        "must consume step dump keys trace.packed_latent_input, trace.transformer_hidden_states, "
        "trace.transformer_timestep, trace.encoder_hidden_states, trace.encoder_hidden_states_mask, "
        "trace.packed_predicted_flow, trace.flow, output.predicted, output.target, and output.loss_for_backward",
    ),
    (
        "LoRA backward replay",
        "must consume adapter_before/adapter_pre plus output.loss_for_backward to compare Mojo LoRA grads/update behavior",
    ),
    (
        "AdamW update replay",
        "must consume adapter_pre, adapter_post, adapter_after and qwen_train_ref_meta optimizer/lr fields",
    ),
    (
        "speed replay",
        "must consume qwen_train_profile_meta timing fields and report Mojo timing plus peak VRAM "
        "for the same Qwen batch/dtype/optimizer/LoRA shape",
    ),
)

TRAIN_TIMING_FIELDS: tuple[tuple[str, str], ...] = (
    ("predict_seconds", "predict"),
    ("loss_seconds", "loss"),
    ("backward_seconds", "backward"),
    ("optimizer_update_seconds", "optimizer"),
)

QWEN_CLAIM_ALIASES = (
    "qwen",
    "qwenimage",
    "qwen-image",
    "qwen_image",
    "qwen image",
    "qwen-image-2512",
)

TRAIN_SPEED_PARITY_CLAIM_MARKERS = (
    "train_speed_parity_accepted",
    "qwen train speed parity accepted",
    "accepted qwen train speed parity",
    "accepted train speed parity",
    "train speed parity accepted",
    "training speed parity accepted",
    "speed parity accepted",
    "speed_parity_accepted",
)

TRAIN_CONTEXT_MARKERS = (
    "train",
    "training",
    "predict",
    "loss",
    "backward",
    "optimizer",
    "adamw",
    "step",
)

NEGATIVE_CLAIM_MARKERS = (
    "not accepted",
    "not a speed-parity claim",
    "no mojo speed parity accepted",
    "no accepted",
    "cannot be accepted",
    "must not",
    "blocked",
    "does not claim",
    "does not accept",
    "evidence only",
    "ot timing evidence only",
)

QWEN_TRAIN_SPEED_EVIDENCE_GROUPS = (
    (
        "OneTrainer profile artifact",
        ("qwen_train_profile_meta.json", "onetrainer train timing", "ot train timing"),
    ),
    ("OneTrainer predict timing", ("predict_seconds", "ot_predict_seconds")),
    ("OneTrainer loss timing", ("loss_seconds", "ot_loss_seconds")),
    ("OneTrainer backward timing", ("backward_seconds", "ot_backward_seconds")),
    ("OneTrainer optimizer timing", ("optimizer_update_seconds", "ot_optimizer_update_seconds")),
    ("Mojo predict timing", ("mojo_predict_seconds", "mojo predict seconds")),
    ("Mojo loss timing", ("mojo_loss_seconds", "mojo loss seconds")),
    ("Mojo backward timing", ("mojo_backward_seconds", "mojo backward seconds")),
    ("Mojo optimizer timing", ("mojo_optimizer_update_seconds", "mojo optimizer seconds")),
    (
        "OneTrainer train peak VRAM",
        ("ot_train_peak_vram_mib", "onetrainer train peak vram", "onetrainer peak vram"),
    ),
    (
        "Mojo train peak VRAM",
        ("mojo_train_peak_vram_mib", "mojo peak vram", "mojo_peak_vram_mib"),
    ),
    ("matching Qwen batch", ("batch_size", "batch=2")),
    ("matching Qwen dtype", ("train_dtype", "bfloat_16", "bf16")),
    ("matching optimizer", ("optimizer", "adamw")),
    ("matching LoRA rank/filter", ("lora_rank", "rank=16", "layer_filter")),
)

TRAIN_SPEED_CLAIM_SCAN_FILES: tuple[Path, ...] = (
    REPO / "OT_MOJO_PORT_REMAINING.md",
    REPO / "../serenity-trainer/src/serenity_trainer/trainer/train_qwenimage_real.mojo",
    REPO / "../serenity-trainer/smoke/qwen_train_control_wiring_smoke.mojo",
    REPO / "serenitymojo/training/onetrainer_product_run.mojo",
    REPO / "serenitymojo/training/onetrainer_product_run_smoke.mojo",
    REPO / "serenitymojo/models/qwenimage/parity/qwen_train_ref_artifact_smoke.mojo",
    REPO / "serenitymojo/models/qwenimage/parity/qwenimage_real_smoke.mojo",
)


def die(msg: str) -> None:
    raise SystemExit(f"[qwen-replay-readiness] FAIL: {msg}")


def section(title: str) -> None:
    print(f"[qwen-replay-readiness] {title}")


def load_json(path: Path) -> dict:
    if not path.exists():
        die(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_safetensors_header(path: Path) -> dict:
    if not path.exists():
        die(f"missing required safetensors artifact: {path}")
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            die(f"truncated safetensors header length: {path}")
        (header_len,) = struct.unpack("<Q", raw_len)
        header_raw = fh.read(header_len)
        if len(header_raw) != header_len:
            die(f"truncated safetensors header: {path}")
    return json.loads(header_raw)


def tensor_keys(header: dict) -> list[str]:
    return sorted(k for k in header if k != "__metadata__")


def require_file(path: Path) -> None:
    if not path.exists():
        die(f"missing required source/gate file: {path}")


def first_float(value: object) -> float:
    if isinstance(value, list):
        if not value:
            return 0.0
        return float(value[0])
    if value is None:
        return 0.0
    return float(value)


def optimizer_state(step: dict, owner: str) -> dict:
    state = step.get(owner)
    if not isinstance(state, dict):
        die(f"metadata missing {owner} object")
    opt_state = state.get("state")
    if not isinstance(opt_state, dict):
        die(f"metadata missing {owner}.state object")
    return opt_state


def optimizer_param_group(step: dict, owner: str) -> dict:
    state = step.get(owner)
    if not isinstance(state, dict):
        die(f"metadata missing {owner} object")
    groups = state.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], dict):
        die(f"metadata {owner}.param_groups must contain one param group")
    group = groups[0]
    if group.get("param_count") != EXPECTED_TRAINABLE_PARAMS:
        die(
            f"metadata {owner}.param_groups[0].param_count="
            f"{group.get('param_count')!r}, expected {EXPECTED_TRAINABLE_PARAMS}"
        )
    if group.get("param_numel") != EXPECTED_ADAPTER_NUMEL_PER_PHASE:
        die(
            f"metadata {owner}.param_groups[0].param_numel="
            f"{group.get('param_numel')!r}, expected {EXPECTED_ADAPTER_NUMEL_PER_PHASE}"
        )
    return group


def expected_step_paths(step_index: int) -> tuple[Path, Path]:
    suffix = f"step{step_index:03d}"
    return (
        PARITY / f"qwen_train_ref_{suffix}.safetensors",
        PARITY / f"qwen_train_ref_{suffix}_adapters.safetensors",
    )


def path_status(path: Path) -> str:
    if not path.exists():
        return f"exists=false path={path}"
    return f"exists=true bytes={path.stat().st_size} path={path}"


def require_contract(contract: dict, meta: dict, profile: dict) -> None:
    if Path(contract.get("reference_repo", "")) != OT_ROOT:
        die(f"contract reference_repo is not {OT_ROOT}: {contract.get('reference_repo')!r}")
    if meta.get("onetrainer") != str(OT_ROOT):
        die(f"metadata onetrainer path is not {OT_ROOT}: {meta.get('onetrainer')!r}")
    if profile.get("onetrainer") != str(OT_ROOT):
        die(f"profile metadata onetrainer path is not {OT_ROOT}: {profile.get('onetrainer')!r}")
    if meta.get("max_steps") != 1 or profile.get("max_steps") != 1:
        die("Qwen replay dump/profile must be max_steps=1")
    runtime = meta.get("runtime_config", {})
    profile_runtime = profile.get("runtime_config", {})
    required_runtime = {
        "model_type": "QWEN",
        "training_method": "LORA",
        "train_dtype": "BFLOAT_16",
        "optimizer": "ADAMW",
        "batch_size": 2,
        "lora_rank": 16,
        "layer_filter": "attn,img_mlp,txt_mlp",
    }
    for key, expected in required_runtime.items():
        if runtime.get(key) != expected:
            die(f"metadata runtime_config.{key}={runtime.get(key)!r}, expected {expected!r}")
        if profile_runtime.get(key) != expected:
            die(
                "profile metadata runtime_config."
                f"{key}={profile_runtime.get(key)!r}, expected {expected!r}"
            )
    print(
        "[qwen-replay-readiness] PASS OneTrainer baseline "
        f"model={runtime['model_type']} method={runtime['training_method']} "
        f"batch={runtime['batch_size']} dtype={runtime['train_dtype']} "
        f"optimizer={runtime['optimizer']} rank={runtime['lora_rank']} "
        f"filter={runtime['layer_filter']}"
    )


def check_step_dump(header: dict) -> None:
    keys = tensor_keys(header)
    if len(keys) != len(REQUIRED_STEP_TENSORS):
        die(f"step dump has {len(keys)} tensor keys, expected {len(REQUIRED_STEP_TENSORS)}")

    for spec in REQUIRED_STEP_TENSORS:
        info = header.get(spec.key)
        if info is None:
            die(f"missing step tensor key: {spec.key}")
        dtype = info.get("dtype")
        shape = tuple(info.get("shape", ()))
        if dtype != spec.dtype:
            die(f"{spec.key} dtype {dtype}, expected {spec.dtype}")
        if shape != spec.shape:
            die(f"{spec.key} shape {shape}, expected {spec.shape}")

    print(f"[qwen-replay-readiness] PASS step dump keys={len(keys)} file={STEP_DUMP}")
    print("[qwen-replay-readiness] REQUIRED STEP KEYS")
    for spec in REQUIRED_STEP_TENSORS:
        shape = "[" + ",".join(str(x) for x in spec.shape) + "]"
        print(f"  {spec.key} dtype={spec.dtype} shape={shape}")


def check_adapter_dump(header: dict) -> None:
    keys = tensor_keys(header)
    if len(keys) != EXPECTED_ADAPTER_KEYS:
        die(f"adapter dump has {len(keys)} tensor keys, expected {EXPECTED_ADAPTER_KEYS}")

    prefix_counts = Counter(k.split(".", 1)[0] for k in keys)
    expected_counts = Counter({prefix: EXPECTED_TRAINABLE_PARAMS for prefix in ADAPTER_PREFIXES})
    if prefix_counts != expected_counts:
        die(f"adapter prefix counts {dict(prefix_counts)}, expected {dict(expected_counts)}")

    dtypes = Counter(info.get("dtype") for key, info in header.items() if key != "__metadata__")
    if dtypes != Counter({"F32": EXPECTED_ADAPTER_KEYS}):
        die(f"adapter dtypes {dict(dtypes)}, expected all F32 from OT lora_weight_dtype")

    sample_required = [
        f"{prefix}.transformer.transformer_blocks.0.attn.to_q.lora_down.weight"
        for prefix in ADAPTER_PREFIXES
    ] + [
        f"{prefix}.transformer.transformer_blocks.59.txt_mlp.net.2.lora_up.weight"
        for prefix in ADAPTER_PREFIXES
    ]
    for key in sample_required:
        if key not in header:
            die(f"missing adapter sample key: {key}")

    print(
        "[qwen-replay-readiness] PASS adapter dump "
        f"keys={len(keys)} prefixes={dict(prefix_counts)} file={ADAPTER_DUMP}"
    )


def check_meta(meta: dict, profile: dict) -> None:
    step = meta.get("steps", [{}])[0]
    profile_step = profile.get("steps", [{}])[0]
    expected_loss = 0.08948630839586258
    if abs(float(step.get("loss_for_backward")) - expected_loss) > 1e-12:
        die(f"metadata loss_for_backward {step.get('loss_for_backward')} != {expected_loss}")
    if step.get("safetensors") != str(STEP_DUMP):
        die(f"metadata step safetensors path mismatch: {step.get('safetensors')!r}")
    if step.get("adapter_safetensors") != str(ADAPTER_DUMP):
        die(f"metadata adapter safetensors path mismatch: {step.get('adapter_safetensors')!r}")
    if profile.get("profile_only") is not True:
        die(f"profile metadata profile_only={profile.get('profile_only')!r}, expected True")
    if profile.get("adapter_dump") != "none":
        die(f"profile metadata adapter_dump={profile.get('adapter_dump')!r}, expected 'none'")
    if profile_step.get("safetensors") is not None:
        die(f"profile metadata safetensors={profile_step.get('safetensors')!r}, expected None")
    if profile_step.get("adapter_safetensors") is not None:
        die(
            "profile metadata adapter_safetensors="
            f"{profile_step.get('adapter_safetensors')!r}, expected None"
        )

    section("LOSS-ONLY / ARTIFACT-CONSUMPTION EVIDENCE")
    print(
        "[qwen-replay-readiness] PASS OneTrainer loss scalar "
        f"loss_pre_scale={step['loss_pre_scale']} "
        f"loss_for_backward={step['loss_for_backward']} "
        f"step_dump={STEP_DUMP} adapter_dump={ADAPTER_DUMP}"
    )
    print(
        "[qwen-replay-readiness] INFO this bucket proves the dumped OT scalar "
        "and artifact availability; backward/update/speed parity are separate buckets"
    )


def check_zero_lr_state_init(meta: dict, profile: dict) -> None:
    section("ZERO-LR OPTIMIZER-STATE-INIT EVIDENCE")
    for label, step in (
        ("dump", meta.get("steps", [{}])[0]),
        ("profile", profile.get("steps", [{}])[0]),
    ):
        if not isinstance(step, dict):
            die(f"{label} step metadata is not an object")
        lr_before = first_float(step.get("lr_before"))
        lr_after = first_float(step.get("lr_after"))
        if lr_before != 0.0:
            die(f"{label} step lr_before={lr_before}, expected zero-lr state-init step")

        before_group = optimizer_param_group(step, "optimizer_before")
        after_group = optimizer_param_group(step, "optimizer_after")
        before_state = optimizer_state(step, "optimizer_before")
        after_state = optimizer_state(step, "optimizer_after")

        if first_float(before_group.get("lr")) != 0.0:
            die(f"{label} optimizer_before lr={before_group.get('lr')!r}, expected 0.0")
        if first_float(after_group.get("lr")) != lr_after:
            die(
                f"{label} optimizer_after lr={after_group.get('lr')!r} "
                f"does not match lr_after={lr_after}"
            )
        if before_state.get("parameter_entries") != 0:
            die(
                f"{label} optimizer_before parameter_entries="
                f"{before_state.get('parameter_entries')!r}, expected 0"
            )
        if before_state.get("tensor_count") != 0 or before_state.get("tensor_numel") != 0:
            die(
                f"{label} optimizer_before tensors="
                f"{before_state.get('tensor_count')!r}/"
                f"{before_state.get('tensor_numel')!r}, expected 0/0"
            )
        if after_state.get("parameter_entries") != EXPECTED_TRAINABLE_PARAMS:
            die(
                f"{label} optimizer_after parameter_entries="
                f"{after_state.get('parameter_entries')!r}, expected {EXPECTED_TRAINABLE_PARAMS}"
            )
        if after_state.get("tensor_count") != EXPECTED_OPTIMIZER_STATE_TENSORS:
            die(
                f"{label} optimizer_after tensor_count="
                f"{after_state.get('tensor_count')!r}, expected {EXPECTED_OPTIMIZER_STATE_TENSORS}"
            )
        if after_state.get("tensor_numel") != EXPECTED_OPTIMIZER_STATE_NUMEL:
            die(
                f"{label} optimizer_after tensor_numel="
                f"{after_state.get('tensor_numel')!r}, expected {EXPECTED_OPTIMIZER_STATE_NUMEL}"
            )
        keys = tuple(after_state.get("keys", ()))
        if keys != EXPECTED_OPTIMIZER_STATE_KEYS:
            die(f"{label} optimizer_after keys={keys!r}, expected {EXPECTED_OPTIMIZER_STATE_KEYS!r}")

        print(
            "[qwen-replay-readiness] PASS "
            f"{label} zero_lr_state_init lr_before={lr_before} lr_after={lr_after} "
            f"optimizer_entries=0->{after_state['parameter_entries']} "
            f"optimizer_tensors=0->{after_state['tensor_count']} "
            f"optimizer_numel=0->{after_state['tensor_numel']} "
            f"keys={','.join(EXPECTED_OPTIMIZER_STATE_KEYS)}"
        )
    print(
        "[qwen-replay-readiness] INFO accepted evidence here is optimizer state creation "
        "at lr_before=0.0, not a nonzero AdamW weight update"
    )


def check_update_bearing_evidence(meta: dict) -> None:
    section("UPDATE-BEARING EVIDENCE")
    require_file(QWEN_ADAPTER_UPDATE_GATE)
    gate_text = QWEN_ADAPTER_UPDATE_GATE.read_text(encoding="utf-8")
    for needle in ("--require-update-bearing", "update_bearing_status", "qwen_update_bearing_readiness"):
        if needle not in gate_text:
            die(f"Qwen update-bearing gate missing marker {needle!r}: {QWEN_ADAPTER_UPDATE_GATE}")

    steps = meta.get("steps", [])
    if not isinstance(steps, list):
        die("metadata steps is not a list")
    discovered = sorted(PARITY.glob("qwen_train_ref_step*.safetensors"))
    print(
        "[qwen-replay-readiness] discovered_qwen_step_files="
        + ",".join(path.name for path in discovered)
    )

    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        step_path = Path(str(step.get("safetensors", "")))
        adapter_path = Path(str(step.get("adapter_safetensors", "")))
        lr_before = first_float(step.get("lr_before"))
        before_entries = optimizer_state(step, "optimizer_before").get("parameter_entries")
        after_entries = optimizer_state(step, "optimizer_after").get("parameter_entries")
        candidate = (
            lr_before > 0.0
            and step_path.exists()
            and adapter_path.exists()
            and before_entries == EXPECTED_TRAINABLE_PARAMS
            and after_entries == EXPECTED_TRAINABLE_PARAMS
        )
        print(
            "[qwen-replay-readiness] meta_step "
            f"index={index} lr_before={lr_before} "
            f"optimizer_entries={before_entries}->{after_entries} "
            f"step_exists={str(step_path.exists()).lower()} "
            f"adapter_exists={str(adapter_path.exists()).lower()} "
            f"update_bearing_candidate={str(candidate).lower()}"
        )

    expected_step, expected_adapters = expected_step_paths(UPDATE_BEARING_STEP_INDEX)
    has_expected_meta = len(steps) > UPDATE_BEARING_STEP_INDEX
    has_candidate = any(
        isinstance(step, dict)
        and first_float(step.get("lr_before")) > 0.0
        and Path(str(step.get("safetensors", ""))).exists()
        and Path(str(step.get("adapter_safetensors", ""))).exists()
        for step in steps
    )
    print(
        "[qwen-replay-readiness] required_update_bearing_step "
        f"index={UPDATE_BEARING_STEP_INDEX} meta_exists={str(has_expected_meta).lower()} "
        f"step_file=({path_status(expected_step)}) "
        f"adapter_file=({path_status(expected_adapters)})"
    )
    if has_candidate:
        print(
            "[qwen-replay-readiness] INFO update_bearing_metadata_candidate=true; "
            "strict nonzero adapter delta must be verified by "
            f"{QWEN_ADAPTER_UPDATE_GATE.relative_to(REPO)} --require-update-bearing"
        )
    else:
        print(
            "[qwen-replay-readiness] BLOCKED update_bearing_evidence accepted=false; "
            "current local Qwen dump is the zero-lr state-init step only"
        )
        print(
            "[qwen-replay-readiness] BLOCKED missing later OneTrainer step with "
            "lr_before>0, optimizer_before state, step001 safetensors, and "
            "nonzero adapter_after - adapter_post delta"
        )


def check_train_speed_evidence(profile: dict) -> None:
    section("TRAIN-SPEED PARITY EVIDENCE")
    profile_step = profile.get("steps", [{}])[0]
    timings = profile_step.get("timing", {})
    reported_timings: list[tuple[str, float]] = []
    for key, label in TRAIN_TIMING_FIELDS:
        if key not in timings:
            die(f"profile metadata missing timing.{key}")
        try:
            value = float(timings[key])
        except (TypeError, ValueError):
            die(f"profile metadata timing.{key} is not numeric: {timings[key]!r}")
        if value <= 0.0:
            die(f"profile metadata timing.{key} must be > 0, got {value!r}")
        reported_timings.append((label, value))
    core_step_seconds = sum(value for _, value in reported_timings)
    print(
        "[qwen-replay-readiness] PASS OneTrainer train timing evidence only "
        + " ".join(f"{label}={value:.6f}s" for label, value in reported_timings)
        + f" core_step_sum={core_step_seconds:.6f}s "
        + f"profile={PROFILE_META}"
    )
    print(
        "[qwen-replay-readiness] INFO Mojo train speed parity accepted=false; "
        "missing matching Mojo predict/loss/backward/optimizer timing and Mojo peak VRAM evidence"
    )
    print(
        "[qwen-replay-readiness] INFO strict speed acceptance requires matching "
        "OneTrainer/Mojo run identity: batch=2 dtype=BFLOAT_16 optimizer=ADAMW "
        "lora_rank=16 layer_filter=attn,img_mlp,txt_mlp plus peak VRAM"
    )


def check_mojo_consumers() -> None:
    current_artifact_consumers: list[Path] = []
    needles = (str(STEP_DUMP), "qwen_train_ref_step000", "qwen_train_ref")

    section("LOSS-ONLY / ARTIFACT-CONSUMPTION MOJO GATES")
    for path, kind, note in MOJO_GATES:
        require_file(path)
        text = path.read_text(encoding="utf-8")
        consumes = any(needle in text for needle in needles)
        if consumes:
            current_artifact_consumers.append(path)
        rel = path.relative_to(REPO)
        print(f"  {rel} status={kind} consumes_one_step_dump={str(consumes).lower()} -- {note}")

    if current_artifact_consumers:
        print(
            "[qwen-replay-readiness] PASS current artifact-consuming gates="
            + ",".join(str(p.relative_to(REPO)) for p in current_artifact_consumers)
        )
    else:
        die("no current Mojo Qwen gate consumes qwen_train_ref_step000 artifacts")
    print(
        "[qwen-replay-readiness] INFO artifact consumers do not by themselves "
        "prove Qwen backward, AdamW update, or speed parity"
    )

    print("[qwen-replay-readiness] NEXT MOJO REPLAY GATES THAT MUST CONSUME THIS DUMP")
    for name, note in NEXT_REPLAY_GATES:
        print(f"  {name}: {note}")


def check_no_false_replay_claims() -> None:
    claim_re = re.compile(r"Qwen(?:-Image)?[^\n]{0,80}(?:one-step|replay|train parity)[^\n]{0,80}PASS", re.I)
    offenders: list[Path] = []
    for path in (REPO / "serenitymojo").rglob("*.mojo"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if claim_re.search(text) and "qwen_train_ref" not in text:
            offenders.append(path)
    if offenders:
        for path in offenders:
            print(f"[qwen-replay-readiness] WARN possible source-only Qwen replay claim: {path.relative_to(REPO)}")
    else:
        print("[qwen-replay-readiness] PASS no source-only Qwen one-step replay PASS claim found")


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _claim_context(lines: list[str], index: int, radius: int = 24) -> str:
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    return "\n".join(lines[start:end]).lower()


def require_no_unsupported_train_speed_claims() -> None:
    """Block accepted Qwen train-speed parity claims without Mojo timing/VRAM proof."""
    section("TRAIN-SPEED CLAIM GUARD")
    found_claims = 0
    for path in TRAIN_SPEED_CLAIM_SCAN_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for index, line in enumerate(lines):
            lowered = line.lower()
            if not _has_any(lowered, TRAIN_SPEED_PARITY_CLAIM_MARKERS):
                continue
            context = _claim_context(lines, index)
            if not _has_any(context, QWEN_CLAIM_ALIASES):
                continue
            if not _has_any(context, TRAIN_CONTEXT_MARKERS):
                continue
            if _has_any(context, NEGATIVE_CLAIM_MARKERS):
                continue

            found_claims += 1
            missing = [
                label
                for label, markers in QWEN_TRAIN_SPEED_EVIDENCE_GROUPS
                if not _has_any(context, markers)
            ]
            if missing:
                print(
                    "[qwen-replay-readiness] FAIL Qwen train speed-parity claim evidence: "
                    f"{path}:{index + 1}"
                )
                for label in missing:
                    print(f"  missing matching evidence near accepted claim: {label}")
                raise SystemExit(1)

    if found_claims:
        print(
            "[qwen-replay-readiness] PASS Qwen train speed-parity claim evidence gate "
            f"claims={found_claims}"
        )
    else:
        print(
            "[qwen-replay-readiness] PASS Qwen train speed-parity evidence gate: "
            "no accepted Mojo claim"
        )


def main() -> int:
    contract = load_json(CONTRACT)
    meta = load_json(META)
    profile = load_json(PROFILE_META)
    require_contract(contract, meta, profile)

    step_header = read_safetensors_header(STEP_DUMP)
    adapter_header = read_safetensors_header(ADAPTER_DUMP)
    check_step_dump(step_header)
    check_adapter_dump(adapter_header)
    check_meta(meta, profile)
    check_zero_lr_state_init(meta, profile)
    check_update_bearing_evidence(meta)
    check_train_speed_evidence(profile)
    check_mojo_consumers()
    check_no_false_replay_claims()
    require_no_unsupported_train_speed_claims()
    print("[qwen-replay-readiness] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
