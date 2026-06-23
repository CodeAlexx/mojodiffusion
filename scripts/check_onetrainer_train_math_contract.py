#!/usr/bin/env python3
"""OneTrainer shared train-math contract guard.

This is a no-CUDA gate for the math every model train replay depends on:
optimizer dispatch/defaults, deterministic AdamW fixed inputs, LR schedule
factors, and OneTrainer masked-loss element scaling. It deliberately does not
claim model forward/backward/update parity.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PARITY = Path("/home/alex/onetrainer-mojo/parity")

ADAMW_REF = PARITY / "adamw_ref.json"
LR_REF = PARITY / "lr_ref.json"
MASKED_LOSS_REF = PARITY / "masked_loss_ref.json"

OPTIMIZER_DISPATCH = REPO / "serenitymojo/training/optimizer_dispatch.mojo"
OPTIMIZER_DISPATCH_SMOKE = REPO / "serenitymojo/training/optimizer_dispatch_smoke.mojo"
OPTIMIZER_UPDATE_SMOKE = REPO / "serenitymojo/training/optimizer_fixed_update_smoke.mojo"
OPTIM = REPO / "serenitymojo/training/optim.mojo"
FUSED_ADAMW = REPO / "serenitymojo/training/fused_adamw_multitensor.mojo"
TRAIN_STEP = REPO / "serenitymojo/training/train_step.mojo"
LOHA_ADAPTER = REPO / "serenitymojo/training/loha_adapter.mojo"
LOKR_ADAPTER = REPO / "serenitymojo/training/lokr_adapter.mojo"
DORA_ADAPTER = REPO / "serenitymojo/training/dora_adapter.mojo"
OFT_ADAPTER = REPO / "serenitymojo/training/oft_adapter.mojo"
BOFT_ADAPTER = REPO / "serenitymojo/training/boft_adapter.mojo"
LOCON_CONV_ADAPTER = REPO / "serenitymojo/training/locon_conv_adapter.mojo"
TUCKER_CONV_ADAPTER = REPO / "serenitymojo/training/tucker_conv_adapter.mojo"
MIXED_PRECISION_ORACLE = REPO / "serenitymojo/training/parity/mixed_precision_oracle.py"
LR_SCHEDULE = REPO / "serenitymojo/training/lr_schedule.mojo"
TRAIN_CONFIG = REPO / "serenitymojo/training/train_config.mojo"
TRAIN_LOOP_POLICY = REPO / "serenitymojo/training/onetrainer_train_loop_policy.mojo"
UPDATE_CONSUMER_GATE = REPO / "scripts/check_chroma_sdxl_mojo_update_consumers.py"
ZERO_LR_CONSUMER_GATE = REPO / "scripts/check_zero_lr_mojo_state_init_consumers.py"
TRAIN_LOOPS = tuple(sorted((REPO / "../serenity-trainer/src/serenity_trainer/trainer").glob("train_*_real.mojo")))

TARGET_TRAIN_LOOPS = {
    "train_qwenimage_real.mojo",
    "train_ernie_real.mojo",
    "train_anima_real.mojo",
    "train_sd35_real.mojo",
    "train_sdxl_real.mojo",
    "train_flux_real.mojo",
    "train_klein_real.mojo",
    "train_chroma_real.mojo",
    "train_zimage_real.mojo",
}

ADAMW_ORDER_SURFACES = (
    (OPTIM, "_adamw_kernel"),
    (FUSED_ADAMW, "_fused_adamw_kernel"),
    (TRAIN_STEP, "_adamw_host_list"),
    (LOHA_ADAPTER, "_adamw_host_list"),
    (LOKR_ADAPTER, "_adamw_host_list"),
    (DORA_ADAPTER, "_adamw_host_list"),
    (OFT_ADAPTER, "_adamw_host_list"),
    (BOFT_ADAPTER, "_adamw_host_list"),
    (LOCON_CONV_ADAPTER, "_adamw_host_list"),
    (TUCKER_CONV_ADAPTER, "_adamw_host_list"),
    (MIXED_PRECISION_ORACLE, "adamw_step_f32"),
)

STALE_ADAMW_ORDER_MARKERS = (
    "AFTER the Adam step",
    "after the Adam step",
    "after Adam adaptive",
    "post-Adam decay",
    "post Adam decay",
)


@dataclass
class Report:
    warnings: list[str]

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[ot-train-math] WARN {message}")


def fail(message: str) -> None:
    raise SystemExit(f"[ot-train-math] FAIL {message}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing required reference file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - surface local artifact damage.
        fail(f"could not parse {path}: {exc}")
    if not isinstance(data, dict):
        fail(f"{path} root must be a JSON object")
    return data


def read(path: Path) -> str:
    if not path.exists():
        fail(f"missing required source file: {path}")
    return path.read_text(encoding="utf-8")


def require_contains(path: Path, text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"{path.relative_to(REPO)} missing {label}: {needle!r}")


def without_line_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def top_level_source_slice(path: Path, text: str, name: str) -> str:
    start = text.find(f"def {name}")
    if start < 0:
        fail(f"{path.relative_to(REPO)} missing required function {name}")
    next_defs = [
        pos
        for token in ("\ndef ", "\nstruct ", "\ncomptime ")
        for pos in (text.find(token, start + 1),)
        if pos >= 0
    ]
    end = min(next_defs) if next_defs else len(text)
    return text[start:end]


def line_positions(text: str, predicate) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and predicate(stripped):
            out.append((offset, stripped))
        offset += len(line)
    return out


def close(actual: float, expected: float, tol: float, label: str) -> None:
    if not math.isfinite(actual):
        fail(f"{label} is not finite: {actual!r}")
    if abs(actual - expected) > tol:
        fail(f"{label} {actual!r} != {expected!r} within {tol}")


def f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", float(value)))[0]


def bf16_round(value: float) -> float:
    """Round a Python float through Float32 then BF16 round-to-nearest-even."""
    bits = int.from_bytes(struct.pack(">f", f32(value)), "big")
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return struct.unpack(">f", rounded.to_bytes(4, "big"))[0]


def compute_bf16_storage_adamw_after2(ref: dict[str, Any]) -> list[float]:
    inputs = ref.get("inputs", {})
    if not isinstance(inputs, dict):
        fail("adamw_ref.inputs must be an object")
    n = int(inputs.get("N", -1))
    lr = float(inputs.get("lr"))
    beta1 = float(inputs.get("b1"))
    beta2 = float(inputs.get("b2"))
    eps = float(inputs.get("eps"))
    weight_decay = float(inputs.get("wd"))
    steps = int(inputs.get("steps", -1))
    grad = bf16_round(float(inputs.get("grad")))
    p0 = ref.get("p0")
    if not isinstance(p0, list) or len(p0) != n:
        fail(f"adamw_ref.p0 must have N={n} values")

    out: list[float] = []
    for raw_p in p0:
        p = bf16_round(float(raw_p))
        m = 0.0
        v = 0.0
        for step in range(1, steps + 1):
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad * grad
            m_hat = m / (1.0 - beta1**step)
            v_hat = v / (1.0 - beta2**step)
            p = p * (1.0 - lr * weight_decay)
            p = p - lr * m_hat / (math.sqrt(v_hat) + eps)
            p = bf16_round(p)
        out.append(p)
    return out


def check_adamw_ref() -> None:
    ref = load_json(ADAMW_REF)
    expected_inputs = {
        "N": 8,
        "lr": 0.001,
        "b1": 0.9,
        "b2": 0.999,
        "eps": 1e-8,
        "wd": 0.01,
        "steps": 2,
        "grad": 0.05,
    }
    inputs = ref.get("inputs", {})
    for key, expected in expected_inputs.items():
        actual = inputs.get(key) if isinstance(inputs, dict) else None
        if isinstance(expected, float):
            close(float(actual), expected, 1e-12, f"adamw_ref.inputs.{key}")
        elif actual != expected:
            fail(f"adamw_ref.inputs.{key}={actual!r}, expected {expected!r}")

    got = compute_bf16_storage_adamw_after2(ref)
    expected = ref.get("p_after2_deterministic")
    if not isinstance(expected, list) or len(expected) != len(got):
        fail("adamw_ref.p_after2_deterministic length mismatch")
    max_delta = 0.0
    for i, (actual, want) in enumerate(zip(got, expected, strict=True)):
        delta = abs(actual - float(want))
        max_delta = max(max_delta, delta)
        close(actual, float(want), 0.0, f"adamw_ref.p_after2_deterministic[{i}]")
    print(
        "[ot-train-math] PASS AdamW ref "
        f"file={ADAMW_REF} steps=2 bf16-param-storage max_delta={max_delta}"
    )


def lr_factor(kind: str, step: int, total: int, min_factor: float) -> float:
    progress = min(max(step / total, 0.0), 1.0)
    if kind == "cosine":
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cos_factor
    if kind == "linear":
        return 1.0 - (1.0 - min_factor) * progress
    if kind == "rex":
        d = 0.9
        denom = (1.0 - d) + (d * (1.0 - progress))
        factor = 0.0 if step >= total else (1.0 - progress) / denom
        return min_factor + (1.0 - min_factor) * factor
    fail(f"unknown LR ref kind: {kind}")


def check_lr_ref() -> None:
    ref = load_json(LR_REF)
    total = int(ref.get("sched_steps", -1))
    min_factor = float(ref.get("min_factor"))
    if total != 1000:
        fail(f"lr_ref.sched_steps={total}, expected 1000")
    close(min_factor, 0.0, 0.0, "lr_ref.min_factor")
    for kind in ("cosine", "linear", "rex"):
        values = ref.get(kind)
        if not isinstance(values, dict):
            fail(f"lr_ref.{kind} must be an object")
        for step_text, expected in sorted(values.items(), key=lambda item: int(item[0])):
            step = int(step_text)
            actual = lr_factor(kind, step, total, min_factor)
            close(actual, float(expected), 1.0e-6, f"lr_ref.{kind}[{step}]")
    print("[ot-train-math] PASS LR refs cosine/linear/rex against local formulas")


def check_masked_loss_ref() -> None:
    ref = load_json(MASKED_LOSS_REF)
    losses = ref.get("losses")
    mask = ref.get("mask")
    out = ref.get("out")
    if not isinstance(losses, list) or not isinstance(mask, list) or not isinstance(out, list):
        fail("masked_loss_ref losses/mask/out must be arrays")
    if len(losses) != len(mask) or len(losses) != len(out):
        fail("masked_loss_ref losses/mask/out length mismatch")
    unmasked_weight = float(ref.get("unmasked_weight"))
    close(unmasked_weight, 0.1, 0.0, "masked_loss_ref.unmasked_weight")
    for i, (loss, mask_value, expected) in enumerate(zip(losses, mask, out, strict=True)):
        loss_f = float(loss)
        mask_f = float(mask_value)
        actual = loss_f * (mask_f if mask_f > 0.0 else unmasked_weight)
        close(actual, float(expected), 2.0e-8, f"masked_loss_ref.out[{i}]")
    print("[ot-train-math] PASS masked-loss element scaling ref")


def is_adamw_decay_line(line: str) -> bool:
    if "m_hat" in line or "v_hat" in line or "mhat" in line or "vhat" in line:
        return False
    if "weight_decay" in line and "lr" in line and "pv" in line:
        return ("pv =" in line or "pv *=" in line) and (
            "weight_decay * pv" in line
            or "pv * weight_decay" in line
            or "lr * weight_decay" in line
        )
    return (
        "wd" in line
        and "lr" in line
        and (line.startswith("p = p *") or line.startswith("p *= "))
    )


def is_adamw_adaptive_subtraction_line(line: str) -> bool:
    return (
        "pv" in line
        and "lr" in line
        and "m_hat" in line
        and "sqrt(v_hat)" in line
        and "-" in line
    ) or (
        line.startswith("p = p -")
        and "lr" in line
        and "mhat" in line
        and "sqrt(vhat)" in line
    )


def check_adamw_order_contracts() -> None:
    """Guard OneTrainer/PyTorch AdamW order: decay param, then Adam subtract."""
    errors: list[str] = []
    for path, func_name in ADAMW_ORDER_SURFACES:
        text = read(path)
        rel = path.relative_to(REPO)
        for marker in STALE_ADAMW_ORDER_MARKERS:
            if marker in text:
                errors.append(f"{rel} contains stale AdamW order marker {marker!r}")
        body = without_line_comments(top_level_source_slice(path, text, func_name))
        decay_lines = line_positions(body, is_adamw_decay_line)
        adaptive_lines = line_positions(body, is_adamw_adaptive_subtraction_line)
        if not decay_lines:
            errors.append(f"{rel}:{func_name} has no explicit decoupled decay line")
            continue
        if not adaptive_lines:
            errors.append(f"{rel}:{func_name} has no Adam adaptive subtraction line")
            continue
        first_adaptive_pos, first_adaptive = adaptive_lines[0]
        post_adam_decay = [
            line for pos, line in decay_lines if pos > first_adaptive_pos
        ]
        if post_adam_decay:
            errors.append(
                f"{rel}:{func_name} applies/mentions decay after adaptive subtraction: "
                + "; ".join(post_adam_decay)
            )
        if min(pos for pos, _line in decay_lines) > first_adaptive_pos:
            errors.append(
                f"{rel}:{func_name} missing PyTorch/OneTrainer order "
                f"(decoupled weight decay before adaptive Adam subtraction: {first_adaptive})"
            )
    if errors:
        fail(
            "AdamW order static contract failed; PyTorch/OneTrainer AdamW "
            "requires decoupled weight decay before the adaptive subtraction:\n  - "
            + "\n  - ".join(errors)
        )
    print(
        "[ot-train-math] PASS AdamW order static guard "
        "decoupled-decay-before-adaptive-subtraction"
    )


def check_source_contracts(report: Report) -> None:
    dispatch = read(OPTIMIZER_DISPATCH)
    dispatch_smoke = read(OPTIMIZER_DISPATCH_SMOKE)
    update_smoke = read(OPTIMIZER_UPDATE_SMOKE)
    lr_schedule = read(LR_SCHEDULE)
    train_config = read(TRAIN_CONFIG)
    train_policy = read(TRAIN_LOOP_POLICY)
    update_consumer_gate = read(UPDATE_CONSUMER_GATE)
    zero_lr_consumer_gate = read(ZERO_LR_CONSUMER_GATE)

    for needle, label in (
        ('canonical="ADAMW"', "default/ADAMW canonical mapping"),
        ("OPT_BACKEND_FUSED_ADAMW", "fused AdamW backend mapping"),
        ("OPT_STATE_ADAM_M_V_F32", "F32 Adam moment state mapping"),
        ('identifier == "ADAFACTOR"', "Adafactor mapping"),
        ('identifier == "ADAMW_8BIT"', "8-bit fail-loud mapping"),
        ('identifier == "SCHEDULE_FREE_ADAMW"', "schedule-free fail-loud mapping"),
    ):
        require_contains(OPTIMIZER_DISPATCH, dispatch, needle, label)

    for needle, label in (
        ("_assert_dispatch(", "dispatch smoke fixed cases"),
        ('"ADAMW"', "dispatch smoke AdamW"),
        ('"ADAFACTOR"', "dispatch smoke Adafactor"),
        ('"ADAMW_8BIT"', "dispatch smoke 8-bit unsupported"),
        ('"SCHEDULE_FREE_ADAMW"', "dispatch smoke schedule-free unsupported"),
    ):
        require_contains(OPTIMIZER_DISPATCH_SMOKE, dispatch_smoke, needle, label)

    for needle, label in (
        ("_test_adamw_fixed_update", "AdamW fixed update gate"),
        ("_test_adamw_order_sensitive_update", "AdamW order-sensitive update gate"),
        ("_test_adamw_resume_equivalence", "AdamW resume equivalence gate"),
        ("_test_adamw_storage_dtype", "AdamW storage dtype gate"),
        ("_test_adafactor_fixed_update", "Adafactor fixed update gate"),
        ("_test_adafactor_resume_equivalence", "Adafactor resume equivalence gate"),
        ("_test_cosine_lr_fixed_inputs", "cosine LR fixed-input gate"),
    ):
        require_contains(OPTIMIZER_UPDATE_SMOKE, update_smoke, needle, label)

    for needle, label in (
        ("def cosine_lr", "cosine LR implementation"),
        ("def linear_lr", "linear LR implementation"),
        ("def rex_lr", "rex LR implementation"),
        ("def lr_for_step", "LR dispatch implementation"),
        ("Float32(0.9)", "OneTrainer REX d=0.9 constant"),
    ):
        require_contains(LR_SCHEDULE, lr_schedule, needle, label)

    for needle, label in (
        ("var masked_training: Bool", "masked_training config field"),
        ("var unmasked_weight: Float32", "unmasked_weight config field"),
        ("var normalize_masked_area_loss: Bool", "normalize_masked_area_loss config field"),
        ("def validate_onetrainer_policy_config", "shared policy validation"),
    ):
        require_contains(TRAIN_CONFIG, train_config, needle, label)

    require_contains(
        TRAIN_LOOP_POLICY,
        train_policy,
        "validate_ot_lora_adamw_loop_policy",
        "shared train-loop policy validator",
    )
    for needle, label in (
        ("validate_ot_train_math_policy", "shared train-math policy validator"),
        ("ot_lr_for_optimizer_step", "shared optimizer-step LR resolver"),
        ("lr_for_step", "OneTrainer LR schedule binding"),
        ("masked_training", "masked training fail-loud policy"),
        ("normalize_masked_area_loss", "masked area fail-loud policy"),
        ("masked_prior_preservation_weight", "masked prior fail-loud policy"),
        ("cfg.beta1", "AdamW beta1 policy validation"),
        ("cfg.beta2", "AdamW beta2 policy validation"),
        ("cfg.eps", "AdamW eps policy validation"),
        ("cfg.weight_decay", "AdamW weight decay policy validation"),
    ):
        require_contains(TRAIN_LOOP_POLICY, train_policy, needle, label)

    for needle, label in (
        ("inspect_update_oracle", "shared adapter update oracle import"),
        ("--require-mojo-parity", "strict Mojo update parity flag"),
        ("adapter_post -> adapter_after", "OneTrainer update delta comparison wording"),
        ("grad_phase_tensor_count", "gradient-phase inventory blocker"),
        (
            "Update-delta artifact ",
            "update-delta artifact scope",
        ),
        (
            "consumption may be reported separately",
            "update-delta-vs-full-AdamW scope",
        ),
        (
            "it is not full Mojo ",
            "oracle-vs-Mojo-parity scope",
        ),
        ("AdamW parity", "AdamW parity wording"),
    ):
        require_contains(UPDATE_CONSUMER_GATE, update_consumer_gate, needle, label)

    for needle, label in (
        ("--require-mojo-state-init", "strict Mojo zero-lr state-init flag"),
        ("has_zero_lr_state_init_oracle", "zero-lr state-init oracle assertion"),
        ("optimizer_before_entries", "optimizer state-init before evidence"),
        ("adapters plus optimizer state initialization", "state-init blocker wording"),
        (
            "OneTrainer zero-lr adapter state-init oracle only; not Mojo backward",
            "oracle-vs-Mojo-state-init scope",
        ),
    ):
        require_contains(ZERO_LR_CONSUMER_GATE, zero_lr_consumer_gate, needle, label)

    check_adamw_order_contracts()
    print("[ot-train-math] PASS source contract surfaces are present")
    check_product_binding(report)


def check_product_binding(report: Report) -> None:
    found = {
        path.name: without_line_comments(read(path))
        for path in TRAIN_LOOPS
        if path.name in TARGET_TRAIN_LOOPS
    }
    missing = sorted(TARGET_TRAIN_LOOPS - set(found))
    if missing:
        fail("missing target train loops: " + ", ".join(missing))

    no_policy = [
        name
        for name, text in found.items()
        if "validate_ot_train_math_policy" not in text
        or "validate_ot_gradient_checkpointing_policy" not in text
    ]
    if no_policy:
        fail("target loops missing shared OneTrainer train-math policy calls: " + ", ".join(no_policy))

    lr_bound = [name for name, text in found.items() if "ot_lr_for_optimizer_step(" in text]
    if len(lr_bound) != len(found):
        report.warn(
            "LR scheduler binding is partial: "
            f"{len(lr_bound)}/{len(found)} target loops call ot_lr_for_optimizer_step "
            f"({', '.join(sorted(lr_bound)) or 'none'})"
        )

    masked_bound = [
        name
        for name, text in found.items()
        if "validate_ot_train_math_policy" in text
    ]
    if len(masked_bound) != len(found):
        report.warn(
            "masked/prior-loss config is not fail-loud product-bound in every target loop: "
            f"{len(masked_bound)}/{len(found)} loops call validate_ot_train_math_policy "
            f"({', '.join(sorted(masked_bound)) or 'none'})"
        )

    adamw_cfg_bound = [
        name
        for name, text in found.items()
        if all(token in text for token in ("cfg.beta1", "cfg.beta2", "cfg.eps", "cfg.weight_decay"))
    ]
    if len(adamw_cfg_bound) != len(found):
        report.warn(
            "AdamW hyperparameter threading is partial: "
            f"{len(adamw_cfg_bound)}/{len(found)} target loops reference cfg beta/eps/wd "
            f"({', '.join(sorted(adamw_cfg_bound)) or 'none'})"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-product",
        action="store_true",
        help="fail on product-loop binding warnings; use this for production parity claims",
    )
    args = parser.parse_args()

    report = Report(warnings=[])
    check_adamw_ref()
    check_lr_ref()
    check_masked_loss_ref()
    check_source_contracts(report)

    if report.warnings:
        print(f"[ot-train-math] WARNINGS: {len(report.warnings)}")
        if args.strict_product:
            return 1
        print("[ot-train-math] report-only PASS; use --strict-product for production binding")
        return 0

    print("[ot-train-math] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
