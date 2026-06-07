#!/usr/bin/env python3
"""Static dtype guard for the ERNIE checkpoint loader and train boundaries.

Default mode keeps the accepted loader gate strict and reports known Ernie
train-boundary host-F32 carriers as warnings. Use --strict-train-boundaries to
make those non-production activation/grad carriers fatal.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ONETRAINER = Path("/home/alex/OneTrainer")
TRAIN_REF_DIR = Path("/home/alex/onetrainer-mojo/parity")
TRAIN_REF_PREFIX = "ernie_train_ref"

OT_CONFIG = ONETRAINER / "configs/ernie_eri2_100step_baseline.json"
OT_BASELINE_LORA = ONETRAINER / "output/ernie_eri2_100step_baseline/lora.safetensors"
OT_METRICS = ONETRAINER / "output/ernie_eri2_100step_baseline/metrics_nocompile.json"
OT_COMPILE_METRICS = ONETRAINER / "output/ernie_eri2_100step_baseline/metrics.json"

TRAIN_REF_CONTRACT = TRAIN_REF_DIR / f"{TRAIN_REF_PREFIX}_contract.json"
TRAIN_REF_META = TRAIN_REF_DIR / f"{TRAIN_REF_PREFIX}_meta.json"
TRAIN_REF_BLOCKERS = TRAIN_REF_DIR / f"{TRAIN_REF_PREFIX}_blockers.json"
TRAIN_REF_STEP = TRAIN_REF_DIR / f"{TRAIN_REF_PREFIX}_step000.safetensors"
TRAIN_REF_ADAPTERS = TRAIN_REF_DIR / f"{TRAIN_REF_PREFIX}_step000_adapters.safetensors"

WEIGHTS = REPO / "serenitymojo/models/ernie/weights.mojo"
STACK = REPO / "serenitymojo/models/ernie/ernie_stack.mojo"
LORA_STACK = REPO / "serenitymojo/models/ernie/ernie_stack_lora.mojo"
TRAIN = REPO / "serenitymojo/training/train_ernie_real.mojo"

MOJO_GATES: tuple[tuple[Path, str, str], ...] = (
    (
        REPO / "serenitymojo/training/ernie_train_control_wiring_smoke.mojo",
        "source-only",
        "validates Ernie config/sample/save/offload controls; does not replay the one-step dump",
    ),
    (
        REPO / "serenitymojo/models/ernie/parity/stack_parity.mojo",
        "torch-oracle",
        "consumes synthetic stack_oracle .bin files, not the OneTrainer one-step dump",
    ),
    (
        REPO / "serenitymojo/models/ernie/parity/lora_step_smoke.mojo",
        "synthetic-step",
        "runs a tiny deterministic LoRA step from stack_oracle inputs, not the OneTrainer dump",
    ),
    (
        REPO / "serenitymojo/models/ernie/parity/lora_stack_parity.mojo",
        "torch-oracle",
        "consumes local lora_stack_oracle .bin files, not the OneTrainer one-step dump",
    ),
    (
        REPO / "serenitymojo/models/ernie/parity/ernie_lora_ot_save_key_smoke.mojo",
        "save-key",
        "emits reduced OneTrainer-format LoRA keys, not the one-step replay dump",
    ),
    (
        REPO / "serenitymojo/models/ernie/parity/ernie_train_ref_artifact_smoke.mojo",
        "in-repo-artifact-consumer",
        "opens the real OneTrainer Ernie step000 and adapter dumps for header and scalar-payload checks",
    ),
    (
        REPO / "serenitymojo/models/ernie/parity/stack_real_smoke.mojo",
        "real-weight-finite",
        "loads real Ernie weights for finite/OOM coverage, not the OneTrainer one-step dump",
    ),
    (
        Path("/home/alex/onetrainer-mojo/smoke/ernie_train_ref_loss_replay.mojo"),
        "external-loss-replay",
        "loss-only replay of the existing OneTrainer Ernie step000 artifact",
    ),
)


@dataclass(frozen=True)
class TensorSpec:
    key: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ReplayGate:
    name: str
    note: str
    required_evidence: tuple[str, ...]
    claim_terms: tuple[str, ...]


@dataclass(frozen=True)
class TrainRefStepStatus:
    step_index: int
    lr_before: tuple[float, ...]
    lr_after: tuple[float, ...]
    optimizer_before_entries: int
    optimizer_after_entries: int
    optimizer_before_tensors: int
    optimizer_after_tensors: int
    zero_lr_state_init_oracle: bool
    update_bearing_step_indices: tuple[int, ...]


@dataclass(frozen=True)
class MojoConsumerSummary:
    artifact_only_consumers: tuple[Path, ...]
    loss_only_consumers: tuple[Path, ...]
    other_dump_consumers: tuple[Path, ...]
    in_repo_consumers: tuple[Path, ...]


REQUIRED_STEP_TENSORS: tuple[TensorSpec, ...] = (
    TensorSpec("batch.latent_image", "BF16", (2, 32, 80, 56)),
    TensorSpec("batch.loss_weight", "F32", (2,)),
    TensorSpec("batch.text_encoder_hidden_state", "BF16", (2, 512, 3072)),
    TensorSpec("batch.tokens", "I64", (2, 512)),
    TensorSpec("batch.tokens_mask", "I64", (2, 512)),
    TensorSpec("output.loss_for_backward", "F32", ()),
    TensorSpec("output.loss_pre_scale", "F32", ()),
    TensorSpec("output.predicted", "BF16", (2, 32, 80, 56)),
    TensorSpec("output.target", "F32", (2, 32, 80, 56)),
    TensorSpec("output.timestep", "I32", (2,)),
    TensorSpec("trace.encode_text.cached_hidden_state", "BF16", (2, 512, 3072)),
    TensorSpec("trace.encode_text.tokens", "I64", (2, 512)),
    TensorSpec("trace.encode_text.tokens_mask", "I64", (2, 512)),
    TensorSpec("trace.encoder_hidden_states", "BF16", (2, 201, 3072)),
    TensorSpec("trace.flow", "F32", (2, 128, 40, 28)),
    TensorSpec("trace.latent_image_before_scale", "F32", (2, 128, 40, 28)),
    TensorSpec("trace.latent_noise", "F32", (2, 128, 40, 28)),
    TensorSpec("trace.noise_source_tensor", "F32", (2, 128, 40, 28)),
    TensorSpec("trace.packed_predicted_flow", "BF16", (2, 128, 40, 28)),
    TensorSpec("trace.scaled_latent_image", "F32", (2, 128, 40, 28)),
    TensorSpec("trace.scaled_noisy_latent_image", "F32", (2, 128, 40, 28)),
    TensorSpec("trace.sigma", "F32", (2, 1, 1, 1)),
    TensorSpec("trace.text_encoder_output", "BF16", (2, 201, 3072)),
    TensorSpec("trace.transformer_hidden_states", "BF16", (2, 128, 40, 28)),
    TensorSpec("trace.transformer_timestep", "I32", (2,)),
)

CONTRACT_REQUIRED_TENSORS = (
    "output.predicted",
    "output.target",
    "output.loss_pre_scale",
    "output.loss_for_backward",
    "trace.scaled_latent_image",
    "trace.latent_noise",
    "trace.scaled_noisy_latent_image",
    "trace.sigma",
    "trace.transformer_hidden_states",
    "trace.transformer_timestep",
    "trace.encoder_hidden_states",
    "trace.packed_predicted_flow",
    "trace.flow",
)

ADAPTER_PHASES = ("adapter_before", "adapter_pre", "adapter_post", "adapter_after")
EXPECTED_ADAPTERS = 36 * 7
EXPECTED_ADAPTER_TRAINABLE_TENSORS = EXPECTED_ADAPTERS * 2
EXPECTED_ADAPTER_DUMP_KEYS = len(ADAPTER_PHASES) * EXPECTED_ADAPTER_TRAINABLE_TENSORS

TRAIN_REF_JSON_ARTIFACTS = (
    TRAIN_REF_CONTRACT,
    TRAIN_REF_BLOCKERS,
    TRAIN_REF_META,
)

TRAIN_REF_NEEDLES = (
    str(TRAIN_REF_STEP),
    TRAIN_REF_STEP.name,
    TRAIN_REF_PREFIX,
    "ernie_train_ref_step000",
)

NEXT_REPLAY_GATES = (
    ReplayGate(
        "transformer forward replay",
        "must consume trace.transformer_hidden_states, trace.transformer_timestep, "
        "trace.encoder_hidden_states, trace.packed_predicted_flow, and output.predicted",
        (
            "trace.transformer_hidden_states",
            "trace.transformer_timestep",
            "trace.encoder_hidden_states",
            "trace.packed_predicted_flow",
            "output.predicted",
        ),
        ("transformer", "forward", "predict"),
    ),
    ReplayGate(
        "LoRA backward replay",
        "must consume adapter_before/adapter_pre plus output.loss_for_backward to compare Mojo LoRA grads",
        ("adapter_before", "adapter_pre", "output.loss_for_backward"),
        ("lora", "backward", "grad"),
    ),
    ReplayGate(
        "AdamW update replay",
        "must consume adapter_pre, adapter_post, adapter_after plus Ernie meta optimizer/lr fields",
        ("adapter_pre", "adapter_post", "adapter_after", "ADAMW", "learning_rate"),
        ("adamw", "optimizer", "update"),
    ),
)

TRAIN_REF_MENTION_RE = re.compile(
    r"(ernie[_ -]train[_ -]ref|train[- ]ref|one[- ]step|step000|one_step)",
    re.I,
)
PASS_RE = re.compile(r"\bPASS\b", re.I)
REPLAY_OR_PARITY_RE = re.compile(r"\b(replay|parity|numeric|train parity)\b", re.I)
NUMERIC_REPLAY_TERM_RE = re.compile(
    r"\b(transformer|forward|predict|backward|grad|adamw|optimizer|update)\b",
    re.I,
)


@dataclass(frozen=True)
class Source:
    path: Path
    text: str

    @property
    def rel(self) -> str:
        return str(self.path.relative_to(REPO))

    def line_for_offset(self, offset: int) -> int:
        return self.text.count("\n", 0, offset) + 1


@dataclass(frozen=True)
class Block:
    source: Source
    name: str
    text: str
    code: str
    start_offset: int

    @property
    def start_line(self) -> int:
        return self.source.line_for_offset(self.start_offset)

    def line_for(self, pattern: str, *, regex: bool = False) -> int:
        if regex:
            match = re.search(pattern, self.text, flags=re.MULTILINE | re.DOTALL)
            if match:
                return self.source.line_for_offset(self.start_offset + match.start())
        else:
            idx = self.text.find(pattern)
            if idx >= 0:
                return self.source.line_for_offset(self.start_offset + idx)
        return self.start_line


def read_source(path: Path) -> Source:
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")
    return Source(path, path.read_text(encoding="utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"[ernie-train-ref] FAIL missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_safetensors_header(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"[ernie-train-ref] FAIL missing safetensors: {path}")
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise SystemExit(f"[ernie-train-ref] FAIL truncated safetensors header length: {path}")
        (header_len,) = struct.unpack("<Q", raw_len)
        header_raw = fh.read(header_len)
    if len(header_raw) != header_len:
        raise SystemExit(f"[ernie-train-ref] FAIL truncated safetensors header: {path}")
    return json.loads(header_raw.decode("utf-8"))


def tensor_keys(header: dict[str, Any]) -> list[str]:
    return sorted(key for key in header if key != "__metadata__")


def require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[ernie-train-ref] FAIL missing required file: {path}")


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def in_repo(path: Path) -> bool:
    try:
        path.relative_to(REPO)
        return True
    except ValueError:
        return False


def source_consumes_train_ref_dump(text: str) -> bool:
    return any(needle in text for needle in TRAIN_REF_NEEDLES)


def require_close(label: str, actual: float, expected: float, eps: float) -> None:
    if abs(actual - expected) > eps:
        raise SystemExit(
            f"[ernie-train-ref] FAIL {label} {actual!r} != {expected!r} within {eps}"
        )


def float_tuple(value: Any, label: str) -> tuple[float, ...]:
    if isinstance(value, list):
        return tuple(float(item) for item in value)
    if value is None:
        raise SystemExit(f"[ernie-train-ref] FAIL meta step missing {label}")
    return (float(value),)


def optimizer_state_counts(optimizer: Any, label: str) -> tuple[int, int, int]:
    if not isinstance(optimizer, dict):
        raise SystemExit(f"[ernie-train-ref] FAIL meta step {label} is not an object")
    if optimizer.get("class") != "AdamW":
        raise SystemExit(
            f"[ernie-train-ref] FAIL meta step {label}.class {optimizer.get('class')!r} != 'AdamW'"
        )
    state = optimizer.get("state")
    if not isinstance(state, dict):
        raise SystemExit(f"[ernie-train-ref] FAIL meta step {label}.state is not an object")
    return (
        int(state.get("parameter_entries", -1)),
        int(state.get("tensor_count", -1)),
        int(state.get("tensor_numel", -1)),
    )


def check_ot_baseline_status() -> None:
    cfg = read_json(OT_CONFIG)
    expected: dict[str, Any] = {
        "model_type": "ERNIE",
        "training_method": "LORA",
        "batch_size": 2,
        "learning_rate": 0.0003,
        "resolution": "512",
        "train_dtype": "BFLOAT_16",
        "output_dtype": "BFLOAT_16",
        "layer_filter": "self_attention,mlp",
        "layer_filter_regex": False,
        "timestep_distribution": "LOGIT_NORMAL",
        "gradient_checkpointing": "ON",
        "enable_async_offloading": True,
        "enable_activation_offloading": True,
        "compile": False,
        "seed": 42,
    }
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if cfg.get(key) != expected_value:
            mismatches.append(f"{key}: got {cfg.get(key)!r}, expected {expected_value!r}")
    if cfg.get("transformer", {}).get("train") is not True:
        mismatches.append("transformer.train must be true")
    if cfg.get("text_encoder", {}).get("train") is not False:
        mismatches.append("text_encoder.train must be false")
    if mismatches:
        print(f"[ernie-train-ref] FAIL OneTrainer Ernie baseline config: {OT_CONFIG}")
        for mismatch in mismatches:
            print(f"  {mismatch}")
        raise SystemExit(1)

    require_file(OT_BASELINE_LORA)
    metrics = read_json(OT_METRICS)
    if metrics.get("status") != "completed":
        raise SystemExit(f"[ernie-train-ref] FAIL metrics_nocompile status: {metrics.get('status')!r}")
    if metrics.get("global_steps_seen") != 100:
        raise SystemExit(
            f"[ernie-train-ref] FAIL metrics_nocompile global_steps_seen "
            f"{metrics.get('global_steps_seen')!r} != 100"
        )
    if metrics.get("loss_count") != 100 or metrics.get("grad_norm_count") != 100:
        raise SystemExit(
            "[ernie-train-ref] FAIL metrics_nocompile missing complete loss/grad_norm history: "
            f"loss_count={metrics.get('loss_count')!r} grad_norm_count={metrics.get('grad_norm_count')!r}"
        )
    last_loss = metrics.get("last_loss") or {}
    last_grad_norm = metrics.get("last_grad_norm") or {}
    print(
        "[ernie-train-ref] PASS OneTrainer 100-step baseline "
        f"steps={metrics['global_steps_seen']} loss={last_loss.get('value')} "
        f"grad_norm={last_grad_norm.get('value')} lora={OT_BASELINE_LORA}"
    )

    if OT_COMPILE_METRICS.exists():
        compile_metrics = read_json(OT_COMPILE_METRICS)
        if compile_metrics.get("status") != "completed":
            print(
                "[ernie-train-ref] WARN compile=true baseline is not a parity target: "
                f"status={compile_metrics.get('status')!r} "
                f"steps={compile_metrics.get('global_steps_seen')!r}"
            )


def check_train_ref_json_contract() -> None:
    for path in TRAIN_REF_JSON_ARTIFACTS:
        require_file(path)
    print(
        "[ernie-train-ref] PASS train-ref JSON artifacts "
        f"contract={TRAIN_REF_CONTRACT} blockers={TRAIN_REF_BLOCKERS} meta={TRAIN_REF_META}"
    )

    contract = read_json(TRAIN_REF_CONTRACT)
    blockers = read_json(TRAIN_REF_BLOCKERS)

    if contract.get("producer") != "scripts/ernie_dump_train_ref.py":
        raise SystemExit(f"[ernie-train-ref] FAIL unexpected contract producer: {contract.get('producer')!r}")
    if contract.get("schema_version") != 1:
        raise SystemExit(
            f"[ernie-train-ref] FAIL unexpected contract schema_version: {contract.get('schema_version')!r}"
        )
    if blockers.get("producer") != "scripts/ernie_dump_train_ref.py":
        raise SystemExit(f"[ernie-train-ref] FAIL unexpected blockers producer: {blockers.get('producer')!r}")
    if blockers.get("schema_version") != 1:
        raise SystemExit(
            f"[ernie-train-ref] FAIL unexpected blockers schema_version: {blockers.get('schema_version')!r}"
        )
    required = set(contract.get("required_tensors", []))
    missing_required = sorted(set(CONTRACT_REQUIRED_TENSORS) - required)
    if missing_required:
        raise SystemExit(f"[ernie-train-ref] FAIL contract missing required tensors: {missing_required}")
    if blockers.get("blocked"):
        raise SystemExit(
            "[ernie-train-ref] FAIL blocker report is blocked: "
            f"{blockers.get('structured_blockers')!r}"
        )
    dry = blockers.get("dry_run", {})
    if dry.get("onetrainer") != str(ONETRAINER):
        raise SystemExit(f"[ernie-train-ref] FAIL blocker onetrainer path: {dry.get('onetrainer')!r}")
    if dry.get("cache_present") is not True:
        raise SystemExit("[ernie-train-ref] FAIL blocker dry-run did not see the Ernie cache")
    print("[ernie-train-ref] PASS dry-run contract/blocker metadata")


def check_step_dump(header: dict[str, Any]) -> None:
    keys = tensor_keys(header)
    expected_keys = {spec.key for spec in REQUIRED_STEP_TENSORS}
    missing = sorted(expected_keys - set(keys))
    extra = sorted(set(keys) - expected_keys)
    if missing:
        raise SystemExit(f"[ernie-train-ref] FAIL step dump missing keys: {missing}")
    if extra:
        raise SystemExit(f"[ernie-train-ref] FAIL step dump unexpected keys: {extra}")

    for spec in REQUIRED_STEP_TENSORS:
        info = header[spec.key]
        dtype = info.get("dtype")
        shape = tuple(info.get("shape", ()))
        if dtype != spec.dtype:
            raise SystemExit(f"[ernie-train-ref] FAIL {spec.key} dtype {dtype} != {spec.dtype}")
        if shape != spec.shape:
            raise SystemExit(f"[ernie-train-ref] FAIL {spec.key} shape {shape} != {spec.shape}")

    dtypes = Counter(header[key].get("dtype") for key in keys)
    print(
        "[ernie-train-ref] PASS step dump "
        f"keys={len(keys)} dtypes={dict(sorted(dtypes.items()))} file={TRAIN_REF_STEP}"
    )


def expected_adapter_shape(layer: int, module: str, suffix: str) -> tuple[int, int]:
    del layer
    rank = 16
    hidden = 4096
    ffn = 12288
    if module.startswith("self_attention."):
        in_f = hidden
        out_f = hidden
    elif module in {"mlp.gate_proj", "mlp.up_proj"}:
        in_f = hidden
        out_f = ffn
    elif module == "mlp.linear_fc2":
        in_f = ffn
        out_f = hidden
    else:
        raise SystemExit(f"[ernie-train-ref] FAIL unknown adapter module: {module}")
    if suffix == "lora_down.weight":
        return rank, in_f
    if suffix == "lora_up.weight":
        return out_f, rank
    raise SystemExit(f"[ernie-train-ref] FAIL unknown adapter suffix: {suffix}")


def parse_adapter_key(key: str) -> tuple[str, int, str, str] | None:
    phase, sep, rest = key.partition(".")
    if not sep or phase not in ADAPTER_PHASES:
        return None
    prefix = "transformer.layers."
    if not rest.startswith(prefix):
        return None
    tail = rest[len(prefix) :]
    layer_s, sep, module_and_suffix = tail.partition(".")
    if not sep or not layer_s.isdigit():
        return None
    for suffix in ("lora_down.weight", "lora_up.weight"):
        trailer = "." + suffix
        if module_and_suffix.endswith(trailer):
            module = module_and_suffix[: -len(trailer)]
            return phase, int(layer_s), module, suffix
    return None


def check_adapter_dump(header: dict[str, Any]) -> None:
    keys = tensor_keys(header)
    if len(keys) != EXPECTED_ADAPTER_DUMP_KEYS:
        raise SystemExit(
            "[ernie-train-ref] FAIL adapter dump key count "
            f"{len(keys)} != {EXPECTED_ADAPTER_DUMP_KEYS}"
        )

    phase_counts = Counter()
    dtype_counts = Counter()
    for key in keys:
        parsed = parse_adapter_key(key)
        if parsed is None:
            raise SystemExit(f"[ernie-train-ref] FAIL unsupported adapter key: {key}")
        phase, layer, module, suffix = parsed
        if layer < 0 or layer >= 36:
            raise SystemExit(f"[ernie-train-ref] FAIL adapter layer out of range: {key}")
        expected_shape = expected_adapter_shape(layer, module, suffix)
        info = header[key]
        dtype = info.get("dtype")
        shape = tuple(info.get("shape", ()))
        if dtype != "F32":
            raise SystemExit(f"[ernie-train-ref] FAIL {key} dtype {dtype} != F32")
        if shape != expected_shape:
            raise SystemExit(f"[ernie-train-ref] FAIL {key} shape {shape} != {expected_shape}")
        phase_counts[phase] += 1
        dtype_counts[dtype] += 1

    expected_phase_counts = Counter({phase: EXPECTED_ADAPTER_TRAINABLE_TENSORS for phase in ADAPTER_PHASES})
    if phase_counts != expected_phase_counts:
        raise SystemExit(
            "[ernie-train-ref] FAIL adapter phase counts "
            f"{dict(phase_counts)} != {dict(expected_phase_counts)}"
        )
    print(
        "[ernie-train-ref] PASS adapter dump "
        f"keys={len(keys)} phases={dict(sorted(phase_counts.items()))} "
        f"dtypes={dict(dtype_counts)} file={TRAIN_REF_ADAPTERS}"
    )


def check_train_ref_meta() -> TrainRefStepStatus:
    meta = read_json(TRAIN_REF_META)
    if meta.get("producer") != "scripts/ernie_dump_train_ref.py":
        raise SystemExit(f"[ernie-train-ref] FAIL unexpected meta producer: {meta.get('producer')!r}")
    if meta.get("onetrainer") != str(ONETRAINER):
        raise SystemExit(f"[ernie-train-ref] FAIL meta onetrainer path: {meta.get('onetrainer')!r}")
    if meta.get("prefix") != TRAIN_REF_PREFIX:
        raise SystemExit(f"[ernie-train-ref] FAIL meta prefix: {meta.get('prefix')!r}")
    if meta.get("max_steps") != 1:
        raise SystemExit(f"[ernie-train-ref] FAIL meta max_steps: {meta.get('max_steps')!r}")
    if meta.get("adapter_dump") != "step":
        raise SystemExit(f"[ernie-train-ref] FAIL meta adapter_dump: {meta.get('adapter_dump')!r}")

    runtime = meta.get("runtime_config", {})
    expected_runtime: dict[str, Any] = {
        "model_type": "ERNIE",
        "training_method": "LORA",
        "train_dtype": "BFLOAT_16",
        "optimizer": "ADAMW",
        "batch_size": 2,
        "lora_rank": 16,
        "lora_alpha": 1.0,
        "layer_filter": "self_attention,mlp",
        "learning_rate": 0.0003,
    }
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected:
            raise SystemExit(
                f"[ernie-train-ref] FAIL meta runtime_config.{key} "
                f"{runtime.get(key)!r} != {expected!r}"
            )

    trainable = meta.get("trainable_parameters", {})
    if trainable.get("count") != EXPECTED_ADAPTER_TRAINABLE_TENSORS:
        raise SystemExit(
            f"[ernie-train-ref] FAIL trainable count {trainable.get('count')!r} "
            f"!= {EXPECTED_ADAPTER_TRAINABLE_TENSORS}"
        )
    if trainable.get("numel") != 47185920:
        raise SystemExit(f"[ernie-train-ref] FAIL trainable numel: {trainable.get('numel')!r}")

    steps = meta.get("steps")
    if not isinstance(steps, list) or len(steps) != 1:
        raise SystemExit("[ernie-train-ref] FAIL meta must contain exactly one step")
    step = steps[0]
    if step.get("safetensors") != str(TRAIN_REF_STEP):
        raise SystemExit(f"[ernie-train-ref] FAIL meta step safetensors path: {step.get('safetensors')!r}")
    if step.get("adapter_safetensors") != str(TRAIN_REF_ADAPTERS):
        raise SystemExit(
            f"[ernie-train-ref] FAIL meta adapter_safetensors path: {step.get('adapter_safetensors')!r}"
        )
    require_close("loss_for_backward", float(step.get("loss_for_backward")), 0.643847644329071, 1.0e-12)
    require_close("grad_norm_pre_clip", float(step.get("grad_norm_pre_clip")), 0.000828770047519356, 1.0e-15)

    lr_before = float_tuple(step.get("lr_before"), "lr_before")
    lr_after = float_tuple(step.get("lr_after"), "lr_after")
    before_entries, before_tensors, _ = optimizer_state_counts(
        step.get("optimizer_before"),
        "optimizer_before",
    )
    after_entries, after_tensors, _ = optimizer_state_counts(
        step.get("optimizer_after"),
        "optimizer_after",
    )
    zero_lr_state_init_oracle = (
        all(lr == 0.0 for lr in lr_before)
        and before_entries == 0
        and before_tensors == 0
        and after_entries == EXPECTED_ADAPTER_TRAINABLE_TENSORS
        and after_tensors > 0
    )
    update_bearing_step_indices = tuple(
        int(candidate.get("step_index", index))
        for index, candidate in enumerate(steps)
        if any(lr > 0.0 for lr in float_tuple(candidate.get("lr_before"), "lr_before"))
    )

    timings = step.get("timing", {})
    for key in ("predict_seconds", "loss_seconds", "backward_seconds", "optimizer_update_seconds"):
        if key not in timings:
            raise SystemExit(f"[ernie-train-ref] FAIL meta timing missing {key}")
    print(
        "[ernie-train-ref] PASS one-step meta "
        f"loss={step['loss_for_backward']} grad_norm={step['grad_norm_pre_clip']} "
        f"elapsed={meta.get('elapsed_seconds')} trainable={trainable.get('count')}"
    )
    if zero_lr_state_init_oracle:
        print(
            "[ernie-train-ref] STATUS zero-lr optimizer-state-init oracle=present "
            f"step={step.get('step_index')} lr_before={list(lr_before)} "
            f"lr_after={list(lr_after)} optimizer_state_entries={before_entries}->{after_entries} "
            f"optimizer_state_tensors={before_tensors}->{after_tensors}"
        )
    else:
        print(
            "[ernie-train-ref] STATUS zero-lr optimizer-state-init oracle=absent "
            f"step={step.get('step_index')} lr_before={list(lr_before)} "
            f"optimizer_state_entries={before_entries}->{after_entries}"
        )
    if update_bearing_step_indices:
        print(
            "[ernie-train-ref] STATUS update-bearing dump=present "
            f"steps={list(update_bearing_step_indices)}"
        )
    else:
        print(
            "[ernie-train-ref] STATUS update-bearing dump=missing "
            "current artifact is lr_before=0 optimizer-state initialization only; "
            "capture a later step with lr_before>0 and nonzero adapter_after-adapter_post"
        )
    return TrainRefStepStatus(
        step_index=int(step.get("step_index", 0)),
        lr_before=lr_before,
        lr_after=lr_after,
        optimizer_before_entries=before_entries,
        optimizer_after_entries=after_entries,
        optimizer_before_tensors=before_tensors,
        optimizer_after_tensors=after_tensors,
        zero_lr_state_init_oracle=zero_lr_state_init_oracle,
        update_bearing_step_indices=update_bearing_step_indices,
    )


def replay_claim_scan_paths() -> list[Path]:
    paths: dict[Path, None] = {}
    for root in (REPO / "serenitymojo/models/ernie", REPO / "serenitymojo/training"):
        if root.exists():
            for path in root.rglob("*.mojo"):
                paths[path] = None
    for path, _, _ in MOJO_GATES:
        if path.exists():
            paths[path] = None
    return sorted(paths)


def source_line(path: Path, line_no: int, line: str) -> str:
    return f"{rel_or_abs(path)}:{line_no}: {line.strip()}"


def line_has_ernie_train_ref_context(path: Path, line: str) -> bool:
    lower_path = str(path).lower()
    lower_line = line.lower()
    if "ernie" not in lower_line and "/ernie/" not in lower_path and "ernie_" not in path.name.lower():
        return False
    return TRAIN_REF_MENTION_RE.search(line) is not None


def allowed_narrow_train_ref_pass(line: str) -> bool:
    lower = line.lower()
    mentions_gate_math = NUMERIC_REPLAY_TERM_RE.search(line) is not None
    if "artifact" in lower and not mentions_gate_math and "replay" not in lower and "parity" not in lower:
        return True
    if "loss replay" in lower and not any(
        term in lower
        for term in (
            "transformer",
            "backward",
            "adamw",
            "optimizer",
            "update",
            "grad",
        )
    ):
        return True
    return False


def gate_line_matches(gate: ReplayGate, line: str) -> bool:
    lower = line.lower()
    primary = gate.claim_terms[0]
    alternates = gate.claim_terms[1:]
    return primary in lower and any(term in lower for term in alternates)


def collect_forbidden_train_ref_pass_claims() -> list[str]:
    offenders: list[str] = []
    for path in replay_claim_scan_paths():
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if PASS_RE.search(line) is None:
                continue
            if not line_has_ernie_train_ref_context(path, line):
                continue
            if allowed_narrow_train_ref_pass(line):
                continue
            if REPLAY_OR_PARITY_RE.search(line) or NUMERIC_REPLAY_TERM_RE.search(line):
                offenders.append(source_line(path, line_no, line))
    return offenders


def check_next_numeric_replay_gates_missing() -> None:
    stale_gate_claims: list[str] = []
    for path in replay_claim_scan_paths():
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if PASS_RE.search(line) is None:
                continue
            if not line_has_ernie_train_ref_context(path, line):
                continue
            if allowed_narrow_train_ref_pass(line):
                continue
            for gate in NEXT_REPLAY_GATES:
                if gate_line_matches(gate, line):
                    stale_gate_claims.append(f"{gate.name}: {source_line(path, line_no, line)}")

    if stale_gate_claims:
        print("[ernie-train-ref] FAIL forbidden Ernie train-ref numeric replay PASS claim(s):")
        for claim in stale_gate_claims:
            print(f"  {claim}")
        raise SystemExit(1)

    print("[ernie-train-ref] NEXT MOJO NUMERIC REPLAY GATES STILL MISSING")
    for gate in NEXT_REPLAY_GATES:
        evidence = ", ".join(gate.required_evidence)
        print(f"  MISSING {gate.name}: {gate.note}; required evidence={evidence}")


def check_mojo_consumers(
    require_consumer: bool,
    require_in_repo_consumer: bool,
) -> MojoConsumerSummary:
    consumers: list[Path] = []
    artifact_only_consumers: list[Path] = []
    loss_only_consumers: list[Path] = []
    other_dump_consumers: list[Path] = []
    in_repo_consumers: list[Path] = []

    print("[ernie-train-ref] CURRENT MOJO ERNIE GATES")
    for path, kind, note in MOJO_GATES:
        require_file(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        consumes = source_consumes_train_ref_dump(text)
        if consumes:
            consumers.append(path)
            if in_repo(path):
                in_repo_consumers.append(path)
            if kind == "in-repo-artifact-consumer":
                artifact_only_consumers.append(path)
            elif kind == "external-loss-replay":
                loss_only_consumers.append(path)
            else:
                other_dump_consumers.append(path)
        print(
            f"  {rel_or_abs(path)} status={kind} "
            f"consumes_one_step_dump={str(consumes).lower()} -- {note}"
        )

    if consumers:
        print(
            "[ernie-train-ref] PASS artifact-consuming Mojo gates="
            + ",".join(rel_or_abs(path) for path in consumers)
        )
    elif require_consumer:
        raise SystemExit("[ernie-train-ref] FAIL no listed Mojo Ernie gate consumes ernie_train_ref_step000")
    else:
        print("[ernie-train-ref] WARN no listed Mojo Ernie gate consumes ernie_train_ref_step000")

    if in_repo_consumers:
        print(
            "[ernie-train-ref] PASS in-repo artifact-consuming Mojo gates="
            + ",".join(rel_or_abs(path) for path in in_repo_consumers)
        )
    else:
        print("[ernie-train-ref] WARN no in-repo mojodiffusion Ernie Mojo gate consumes ernie_train_ref_step000")
        if require_in_repo_consumer:
            raise SystemExit("[ernie-train-ref] FAIL no in-repo Ernie Mojo gate consumes ernie_train_ref_step000")
    print("[ernie-train-ref] TRAIN-REF CONSUMPTION STATUS")
    print(
        "  artifact_only_consumers="
        + (",".join(rel_or_abs(path) for path in artifact_only_consumers) or "none")
    )
    print(
        "  loss_only_consumers="
        + (",".join(rel_or_abs(path) for path in loss_only_consumers) or "none")
    )
    print(
        "  other_one_step_dump_consumers="
        + (",".join(rel_or_abs(path) for path in other_dump_consumers) or "none")
    )
    if artifact_only_consumers or loss_only_consumers:
        print(
            "[ernie-train-ref] STATUS train-ref consumption=artifact/loss-only; "
            "this does not prove Mojo backward, LoRA gradients, or AdamW update parity"
        )
    return MojoConsumerSummary(
        artifact_only_consumers=tuple(artifact_only_consumers),
        loss_only_consumers=tuple(loss_only_consumers),
        other_dump_consumers=tuple(other_dump_consumers),
        in_repo_consumers=tuple(in_repo_consumers),
    )


def check_no_false_train_ref_claims() -> None:
    offenders = collect_forbidden_train_ref_pass_claims()
    if offenders:
        print("[ernie-train-ref] FAIL forbidden Ernie train-ref replay PASS claim(s):")
        for offender in offenders:
            print(f"  {offender}")
        raise SystemExit(1)
    else:
        print("[ernie-train-ref] PASS no forbidden Ernie train-ref replay PASS claim found")


def check_train_ref_gap_status(
    step_status: TrainRefStepStatus,
    consumer_summary: MojoConsumerSummary,
) -> None:
    print("[ernie-train-ref] ERNIE TRAIN-REF PARITY STATUS")
    if step_status.zero_lr_state_init_oracle:
        print(
            "  zero_lr_optimizer_state_init_oracle=present "
            f"step={step_status.step_index} "
            f"optimizer_entries={step_status.optimizer_before_entries}->{step_status.optimizer_after_entries}"
        )
    else:
        print("  zero_lr_optimizer_state_init_oracle=absent")
    if step_status.update_bearing_step_indices:
        print(
            "  update_bearing_dump=present "
            f"steps={list(step_status.update_bearing_step_indices)}"
        )
    else:
        print(
            "  update_bearing_dump=missing "
            "reason=no captured Ernie step has lr_before>0; current step only initializes AdamW state"
        )
    if consumer_summary.artifact_only_consumers or consumer_summary.loss_only_consumers:
        print(
            "  artifact_loss_only_consumption=present "
            f"artifact_only={len(consumer_summary.artifact_only_consumers)} "
            f"loss_only={len(consumer_summary.loss_only_consumers)}"
        )
    else:
        print("  artifact_loss_only_consumption=absent")
    if consumer_summary.other_dump_consumers:
        print(
            "  mojo_backward_adamw_consumer=unclassified_present "
            + ",".join(rel_or_abs(path) for path in consumer_summary.other_dump_consumers)
        )
    else:
        print(
            "  mojo_backward_consumer=missing "
            "required=adapter_before/adapter_pre + output.loss_for_backward -> Mojo LoRA grads"
        )
        print(
            "  mojo_adamw_consumer=missing "
            "required=adapter_pre/adapter_post/adapter_after + ADAMW/lr meta -> Mojo optimizer update"
        )


def check_train_ref_artifacts(
    require_consumer: bool,
    require_in_repo_consumer: bool,
) -> None:
    check_ot_baseline_status()
    check_train_ref_json_contract()
    step_status = check_train_ref_meta()
    check_step_dump(read_safetensors_header(TRAIN_REF_STEP))
    check_adapter_dump(read_safetensors_header(TRAIN_REF_ADAPTERS))
    consumer_summary = check_mojo_consumers(require_consumer, require_in_repo_consumer)
    check_train_ref_gap_status(step_status, consumer_summary)
    check_next_numeric_replay_gates_missing()
    check_no_false_train_ref_claims()


def strip_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        lines.append(line.split("#", 1)[0])
    return "\n".join(lines)


def top_level_span(text: str, keyword: str, name: str) -> tuple[int, int] | None:
    match = re.search(rf"^{keyword} {re.escape(name)}\b", text, flags=re.MULTILINE)
    if not match:
        return None
    boundary = re.search(
        r"^(?:def|struct|alias|comptime|from|import) \w+\b",
        text[match.end() :],
        flags=re.MULTILINE,
    )
    end = len(text) if boundary is None else match.end() + boundary.start()
    return match.start(), end


def top_level_body(source: Source, keyword: str, name: str) -> Block:
    span = top_level_span(source.text, keyword, name)
    if span is None:
        raise SystemExit(f"missing {keyword} {name} in {source.rel}")
    start, end = span
    text = source.text[start:end]
    return Block(source, name, text, strip_comments(text), start)


def maybe_top_level_body(source: Source, keyword: str, name: str) -> Block | None:
    span = top_level_span(source.text, keyword, name)
    if span is None:
        return None
    start, end = span
    text = source.text[start:end]
    return Block(source, name, text, strip_comments(text), start)


def function_body(source: Source, name: str) -> Block:
    return top_level_body(source, "def", name)


def struct_body(source: Source, name: str) -> Block:
    return top_level_body(source, "struct", name)


def maybe_function_body(source: Source, name: str) -> Block | None:
    return maybe_top_level_body(source, "def", name)


def maybe_struct_body(source: Source, name: str) -> Block | None:
    return maybe_top_level_body(source, "struct", name)


def fmt(source: Source, line: int, message: str) -> str:
    return f"{source.rel}:{line}: {message}"


def has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.MULTILINE | re.DOTALL) is not None


def add_absent(blockers: list[str], block: Block, pattern: str, message: str) -> None:
    if pattern not in block.code:
        blockers.append(fmt(block.source, block.start_line, message))


def add_present(
    blockers: list[str],
    block: Block,
    patterns: list[str],
    message: str,
) -> None:
    for pattern in patterns:
        if pattern in block.code:
            blockers.append(fmt(block.source, block.line_for(pattern), message))
            return


def add_regex(
    blockers: list[str],
    block: Block,
    pattern: str,
    message: str,
) -> None:
    if has(pattern, block.code):
        blockers.append(fmt(block.source, block.line_for(pattern, regex=True), message))


def add_count_less_than(
    blockers: list[str],
    block: Block,
    pattern: str,
    minimum: int,
    message: str,
) -> None:
    if block.code.count(pattern) < minimum:
        blockers.append(fmt(block.source, block.start_line, message))


def add_if_present(
    blockers: list[str],
    block: Block | None,
    patterns: list[str],
    message: str,
) -> None:
    if block is not None:
        add_present(blockers, block, patterns, message)


def add_if_regex(
    blockers: list[str],
    block: Block | None,
    pattern: str,
    message: str,
) -> None:
    if block is not None:
        add_regex(blockers, block, pattern, message)


def loader_blockers(weights: Source, lora_stack: Source, train: Source) -> list[str]:
    blockers: list[str] = []

    stored = function_body(weights, "_load_stored_device")
    reshaped = function_body(weights, "_load_stored_device_reshaped")
    block_loader = function_body(weights, "load_ernie_block_weights")
    bf16_block_loader = function_body(weights, "load_ernie_block_weights_bf16_normf32")
    base_loader = function_body(weights, "load_ernie_stack_base")
    all_bf16 = function_body(weights, "load_ernie_all_blocks_bf16_normf32")

    for block in (stored, reshaped):
        add_absent(
            blockers,
            block,
            "from_parts(info.dtype, info.shape.copy(), bytes)",
            f"{block.name} no longer builds tensor views from checkpoint dtype metadata",
        )
        add_absent(
            blockers,
            block,
            "Tensor.from_view(tv, ctx)",
            f"{block.name} no longer uploads checkpoint storage through Tensor.from_view",
        )
        add_present(
            blockers,
            block,
            ["cast_tensor", "STDtype.F32", ".to_host(ctx)", "Tensor.from_host"],
            f"{block.name} materializes checkpoint tensors across an F32 or host boundary",
        )

    add_count_less_than(
        blockers,
        block_loader,
        "_load_stored_device(",
        11,
        "load_ernie_block_weights no longer routes all block tensors through the stored-dtype loader",
    )
    add_present(
        blockers,
        block_loader,
        ["cast_tensor", "STDtype.F32", ".to_host(ctx)", "Tensor.from_host"],
        "load_ernie_block_weights creates an F32 or host checkpoint boundary",
    )
    add_count_less_than(
        blockers,
        bf16_block_loader,
        "_load_stored_device(",
        11,
        "load_ernie_block_weights_bf16_normf32 no longer routes all block tensors through the stored-dtype loader",
    )
    add_present(
        blockers,
        bf16_block_loader,
        ["cast_tensor", "STDtype.F32", ".to_host(ctx)", "Tensor.from_host"],
        "load_ernie_block_weights_bf16_normf32 creates an F32 or host checkpoint boundary",
    )
    add_count_less_than(
        blockers,
        base_loader,
        "_load_stored_device(",
        12,
        "load_ernie_stack_base no longer routes base tensors through the stored-dtype loader",
    )
    add_present(
        blockers,
        base_loader,
        ["cast_tensor", "STDtype.F32", ".to_host(ctx)", "Tensor.from_host"],
        "load_ernie_stack_base creates an F32 or host checkpoint boundary",
    )
    add_absent(
        blockers,
        all_bf16,
        "load_ernie_block_weights_bf16_normf32",
        "load_ernie_all_blocks_bf16_normf32 no longer delegates to the accepted dtype-preserving block loader",
    )

    for name in (
        "ernie_stack_lora_forward_streamed_device",
        "ernie_stack_lora_backward_streamed_device",
    ):
        block = maybe_function_body(lora_stack, name)
        if block is None:
            continue
        add_absent(
            blockers,
            block,
            "load_ernie_block_weights_bf16_normf32",
            f"{name} no longer uses the dtype-preserving streamed block loader",
        )
        add_present(
            blockers,
            block,
            ["load_ernie_block_weights(st"],
            f"{name} uses the generic streamed loader instead of the accepted BF16/norm dtype loader",
        )

    train_main = maybe_function_body(train, "main")
    if train_main is not None and "load_ernie_all_blocks" in train_main.code:
        add_absent(
            blockers,
            train_main,
            "load_ernie_all_blocks_bf16_normf32",
            "train main calls a block loader but not the accepted BF16/norm dtype loader",
        )
    return blockers


def train_boundary_blockers(stack: Source, lora_stack: Source, train: Source) -> list[str]:
    blockers: list[str] = []

    stack_forward = maybe_struct_body(stack, "ErnieStackForward")
    add_if_present(
        blockers,
        stack_forward,
        ["var out: List[Float32]"],
        "ErnieStackForward stores model outputs as host List[Float32]",
    )
    add_if_present(
        blockers,
        stack_forward,
        ["var img_in_act: List[Float32]", "var txt_in_act: List[Float32]"],
        "ErnieStackForward stores input projection activations as host List[Float32]",
    )

    stack_grads = maybe_struct_body(stack, "ErnieStackGrads")
    add_if_present(
        blockers,
        stack_grads,
        ["var d_img_tokens: List[Float32]", "var d_txt_tokens: List[Float32]"],
        "ErnieStackGrads stores token gradients as host List[Float32]",
    )
    add_if_present(
        blockers,
        stack_grads,
        ["var d_patch_w: List[Float32]", "var d_text_proj: List[Float32]", "var d_final_lin: List[Float32]"],
        "ErnieStackGrads stores base weight gradients as host List[Float32]",
    )
    add_if_present(
        blockers,
        stack_grads,
        ["var d_f_scale: List[Float32]", "var d_f_shift: List[Float32]", "var d_shared_mod: List[Float32]"],
        "ErnieStackGrads stores modulation gradients as host List[Float32]",
    )

    stack_lite = maybe_struct_body(stack, "ErnieStackGradsLite")
    add_if_present(
        blockers,
        stack_lite,
        ["var d_img_tokens: List[Float32]", "var d_txt_tokens: List[Float32]"],
        "ErnieStackGradsLite stores streamed token gradients as host List[Float32]",
    )
    add_if_present(
        blockers,
        stack_lite,
        ["var d_wq_deep: List[Float32]", "var d_wdown_deep: List[Float32]"],
        "ErnieStackGradsLite keeps probe block gradients as host List[Float32]",
    )

    stack_t = maybe_function_body(stack, "_t")
    add_if_present(
        blockers,
        stack_t,
        ["Tensor.from_host(vals, shape^, STDtype.F32, ctx)"],
        "_t still uploads train activations from host List[Float32] as STDtype.F32",
    )
    for name in ("_linear_wdev", "_linear_wdev_bias", "saved_x_out"):
        add_if_present(
            blockers,
            maybe_function_body(stack, name),
            [".to_host(ctx)"],
            f"{name} returns train activations through host F32 readback",
        )
    for name in (
        "ernie_stack_forward",
        "ernie_stack_forward_streamed",
        "ernie_stack_backward",
        "ernie_stack_backward_streamed",
    ):
        add_if_present(
            blockers,
            maybe_function_body(stack, name),
            [".to_host(ctx)"],
            f"{name} still stages stack activations/grads through to_host(ctx)",
        )
        add_if_regex(
            blockers,
            maybe_function_body(stack, name),
            r"\bList\[Float32\]",
            f"{name} exposes train-boundary List[Float32] carriers",
        )

    lora_grads = maybe_struct_body(lora_stack, "ErnieLoraGrads")
    add_if_present(
        blockers,
        lora_grads,
        ["var d_a: List[List[Float32]]", "var d_b: List[List[Float32]]"],
        "ErnieLoraGrads stores LoRA adapter gradients as host List[List[Float32]]",
    )
    add_if_present(
        blockers,
        lora_grads,
        ["var d_shared_mod: List[Float32]", "var d_f_scale: List[Float32]", "var d_f_shift: List[Float32]"],
        "ErnieLoraGrads stores modulation gradients as host List[Float32]",
    )

    host_grad_lists = maybe_struct_body(lora_stack, "_ErnieHostGradLists")
    add_if_present(
        blockers,
        host_grad_lists,
        ["var d_a: List[List[Float32]]", "var d_b: List[List[Float32]]"],
        "_ErnieHostGradLists keeps device LoRA grads in host F32 lists",
    )
    add_if_present(
        blockers,
        maybe_function_body(lora_stack, "_host_grad_slice"),
        ["bitcast[Float32]()", "-> List[Float32]"],
        "_host_grad_slice decodes device grad buffers as host Float32 lists",
    )
    add_if_present(
        blockers,
        maybe_function_body(lora_stack, "_grad_arc_f32"),
        ["cast_tensor(t[], STDtype.F32, ctx)"],
        "_grad_arc_f32 casts LoRA grad tensors to F32 before host readback",
    )
    add_if_present(
        blockers,
        maybe_function_body(lora_stack, "_ernie_tensor_grads_to_host"),
        ["var d_a_flat = List[List[Float32]]()", "ctx.enqueue_create_host_buffer"],
        "_ernie_tensor_grads_to_host batches LoRA grads into host F32 carriers",
    )

    for name in (
        "ernie_stack_lora_forward",
        "ernie_stack_lora_forward_streamed",
        "ernie_stack_lora_forward_streamed_device",
        "ernie_stack_lora_forward_resident_device",
        "ernie_stack_lora_predict_streamed_device",
        "ernie_stack_lora_predict_resident_device",
    ):
        add_if_present(
            blockers,
            maybe_function_body(lora_stack, name),
            [".to_host(ctx)"],
            f"{name} returns prediction/output through host F32 readback",
        )
    for name in (
        "ernie_stack_lora_backward",
        "ernie_stack_lora_backward_streamed",
        "ernie_stack_lora_backward_streamed_device",
        "ernie_stack_lora_backward_resident_device",
    ):
        add_if_present(
            blockers,
            maybe_function_body(lora_stack, name),
            [".to_host(ctx)", "_ernie_tensor_grads_to_host"],
            f"{name} still stages LoRA train grads through host F32 boundaries",
        )
        add_if_regex(
            blockers,
            maybe_function_body(lora_stack, name),
            r"\bList\[Float32\]",
            f"{name} exposes train-boundary List[Float32] carriers",
        )

    add_if_present(
        blockers,
        maybe_function_body(train, "_read_cache_f32"),
        ["cast_tensor(t, STDtype.F32, ctx)", "-> List[Float32]"],
        "_read_cache_f32 casts cached train tensors to host F32",
    )
    add_if_present(
        blockers,
        maybe_function_body(train, "_latent_to_img_tokens"),
        ["-> List[Float32]", "var out = List[Float32]()"],
        "_latent_to_img_tokens builds image tokens as host List[Float32]",
    )
    add_if_present(
        blockers,
        maybe_function_body(train, "_text_to_txt_tokens"),
        ["-> List[Float32]", "var out = List[Float32]()"],
        "_text_to_txt_tokens builds text tokens as host List[Float32]",
    )
    add_if_present(
        blockers,
        maybe_function_body(train, "_host_noise"),
        ["-> List[Float32]", "var out = List[Float32]()"],
        "_host_noise generates training noise as host List[Float32]",
    )
    add_if_present(
        blockers,
        maybe_function_body(train, "_chunk"),
        ["-> List[Float32]", "var o = List[Float32]()"],
        "_chunk carries shared/final modulation vectors as host List[Float32]",
    )
    add_if_present(
        blockers,
        maybe_function_body(train, "_shared_adaln_source"),
        ["adaln.to_host(ctx)", "fmod.to_host(ctx)", "Tuple[ErnieModVecs, List[Float32], List[Float32]]"],
        "_shared_adaln_source returns AdaLN/final modulation through host F32 lists",
    )

    main = maybe_function_body(train, "main")
    add_if_present(
        blockers,
        main,
        ["var target = List[Float32]()", "var noisy_tokens = List[Float32]()"],
        "train main stores noisy latents/targets as host List[Float32]",
    )
    add_if_present(
        blockers,
        main,
        ["var d_out = List[Float32]()", "var pred = fwd.out.copy()"],
        "train main computes loss/upstream gradients on host List[Float32]",
    )
    add_if_present(
        blockers,
        main,
        ["_clip_grads(grads, CLIP)", "ernie_lora_adamw_step(lora, grads"],
        "train main clips and steps adapter gradients through host F32 optimizer carriers",
    )

    return blockers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-train-boundaries",
        action="store_true",
        help="fail while known Ernie train/offload activation or grad carriers remain host-F32",
    )
    parser.add_argument(
        "--skip-one-step-train-ref",
        action="store_true",
        help="skip local OneTrainer one-step dump artifact/header checks",
    )
    parser.add_argument(
        "--require-one-step-consumer",
        action="store_true",
        help="fail unless at least one listed Mojo Ernie gate consumes ernie_train_ref_step000",
    )
    parser.add_argument(
        "--require-in-repo-one-step-consumer",
        action="store_true",
        help="fail unless a mojodiffusion Ernie Mojo gate consumes ernie_train_ref_step000",
    )
    args = parser.parse_args()

    weights = read_source(WEIGHTS)
    stack = read_source(STACK)
    lora_stack = read_source(LORA_STACK)
    train = read_source(TRAIN)

    blockers = loader_blockers(weights, lora_stack, train)
    train_blockers = train_boundary_blockers(stack, lora_stack, train)

    print("[ernie-dtype-contract] loader blockers:", len(blockers))
    for blocker in blockers:
        print("[ernie-dtype-contract] FAIL:", blocker)
    if blockers:
        return 1

    print("[ernie-dtype-contract] train-boundary blockers:", len(train_blockers))
    for blocker in train_blockers:
        print("[ernie-dtype-contract] WARN train-boundary:", blocker)
    if args.strict_train_boundaries and train_blockers:
        print("[ernie-dtype-contract] FAIL strict train-boundaries")
        return 1
    if train_blockers:
        print("[ernie-dtype-contract] PASS loader gate; train-boundaries are report-only")
    else:
        print("[ernie-dtype-contract] PASS loader gate and train-boundary audit")

    if not args.skip_one_step_train_ref:
        check_train_ref_artifacts(
            args.require_one_step_consumer,
            args.require_in_repo_one_step_consumer,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
