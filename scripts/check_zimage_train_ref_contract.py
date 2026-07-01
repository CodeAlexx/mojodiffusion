#!/usr/bin/env python3
"""Check whether ZImage has the required OneTrainer train-step reference dump."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_CONFIG = Path("/home/alex/OneTrainer/configs/eri2_zimage_base_2500.json")
DEFAULT_PARITY_DIR = Path("/home/alex/serenity-trainer/parity")
DEFAULT_INPUT_DUMP = DEFAULT_PARITY_DIR / "zimage_train_ref_step000_inputs.safetensors"
REQUIRED_ARTIFACTS = (
    "zimage_train_ref_meta.json",
    "zimage_train_ref_step000.safetensors",
    "zimage_train_ref_step000_adapters.safetensors",
)
PRODUCER = "scripts/zimage_dump_train_ref.py"
STEP_REQUIRED_TENSORS = {
    "scaled_latent_image": ((2, 16, 64, 64), "torch.bfloat16"),
    "latent_noise": ((2, 16, 64, 64), "torch.bfloat16"),
    "scaled_noisy_latent_image": ((2, 16, 64, 64), "torch.bfloat16"),
    "latent_input": ((2, 16, 1, 64, 64), "torch.bfloat16"),
    "timestep": ((2,), "torch.float32"),
    "sigma": ((2, 1, 1, 1), "torch.float32"),
    "flow_target": ((2, 16, 64, 64), "torch.bfloat16"),
    "predicted_flow": ((2, 16, 64, 64), "torch.bfloat16"),
    "output.loss_pre_scale": ((), "torch.float32"),
    "output.loss_for_backward": ((), "torch.float32"),
    "output.predicted": ((2, 16, 64, 64), "torch.bfloat16"),
    "output.target": ((2, 16, 64, 64), "torch.bfloat16"),
    "text_encoder_output_batch_size": ((), "torch.int64"),
}
ADAPTER_PHASES = (
    "adapter_before",
    "adapter_pre",
    "adapter_post",
    "adapter_after",
    "adapter_pre_clip",
    "adapter_post_clip",
    "adapter_pre_clip_grad",
    "adapter_post_clip_grad",
)


def _fail(message: str, expect_missing: bool = False) -> int:
    print(f"[zimage-train-ref-contract] {'BLOCKED' if expect_missing else 'FAIL'}: {message}")
    return 0 if expect_missing else 1


def _validate_input_dump(path: Path) -> str | None:
    if not path.is_file():
        return f"missing OneTrainer input dump: {path}"
    try:
        from safetensors.torch import load_file
    except Exception as exc:  # pragma: no cover - environment issue
        return f"could not import safetensors.torch: {exc}"
    tensors = load_file(str(path))
    expected = {
        "scaled_latent_image": ((2, 16, 64, 64), "torch.bfloat16"),
        "latent_noise": ((2, 16, 64, 64), "torch.bfloat16"),
        "scaled_noisy_latent_image": ((2, 16, 64, 64), "torch.bfloat16"),
        "latent_input": ((2, 16, 1, 64, 64), "torch.bfloat16"),
        "timestep": ((2,), "torch.float32"),
        "sigma": ((2, 1, 1, 1), "torch.float32"),
        "flow_target": ((2, 16, 64, 64), "torch.bfloat16"),
        "predicted_flow": ((2, 16, 64, 64), "torch.bfloat16"),
        "text_encoder_output_batch_size": ((), "torch.int64"),
    }
    missing = sorted(set(expected) - set(tensors))
    if missing:
        return "input dump missing tensors: " + ", ".join(missing)
    for key, (shape, dtype) in expected.items():
        tensor = tensors[key]
        if tuple(tensor.shape) != shape:
            return f"input dump {key} shape {tuple(tensor.shape)} != {shape}"
        if str(tensor.dtype) != dtype:
            return f"input dump {key} dtype {tensor.dtype} != {dtype}"
    batch_size = int(tensors["text_encoder_output_batch_size"].item())
    if batch_size != 2:
        return f"input dump text_encoder_output_batch_size {batch_size} != 2"
    for index in range(batch_size):
        key = f"text_encoder_output_{index}"
        if key not in tensors:
            return f"input dump missing {key}"
        tensor = tensors[key]
        if len(tensor.shape) != 2 or tensor.shape[1] != 2560:
            return f"input dump {key} shape {tuple(tensor.shape)} is not (*, 2560)"
        if str(tensor.dtype) != "torch.bfloat16":
            return f"input dump {key} dtype {tensor.dtype} != torch.bfloat16"
    if len(tensors) != 8 + batch_size + 1:
        return f"input dump tensor count {len(tensors)} != {8 + batch_size + 1}"
    return None


def _finite_float(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _load_safe_tensors(path: Path):
    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - environment issue
        return None, f"could not import safetensors.safe_open: {exc}"
    try:
        return safe_open(str(path), framework="pt", device="cpu"), None
    except Exception as exc:
        return None, f"could not open safetensors {path}: {exc}"


def _validate_tensor(fh, key: str, shape: tuple[int, ...], dtype: str) -> str | None:
    try:
        tensor = fh.get_tensor(key)
    except Exception as exc:
        return f"{key}: could not read tensor: {exc}"
    if tuple(tensor.shape) != shape:
        return f"{key}: shape {tuple(tensor.shape)} != {shape}"
    if str(tensor.dtype) != dtype:
        return f"{key}: dtype {tensor.dtype} != {dtype}"
    return None


def _validate_step_dump(path: Path) -> list[str]:
    errors: list[str] = []
    handle, error = _load_safe_tensors(path)
    if error is not None:
        return [error]
    assert handle is not None
    with handle as fh:
        keys = set(fh.keys())
        missing = sorted(set(STEP_REQUIRED_TENSORS) - keys)
        if missing:
            errors.append("step dump missing tensors: " + ", ".join(missing))
            return errors
        for key, (shape, dtype) in STEP_REQUIRED_TENSORS.items():
            tensor_error = _validate_tensor(fh, key, shape, dtype)
            if tensor_error is not None:
                errors.append("step dump " + tensor_error)
        try:
            batch_size = int(fh.get_tensor("text_encoder_output_batch_size").item())
        except Exception as exc:
            errors.append(f"step dump text_encoder_output_batch_size unreadable: {exc}")
            return errors
        if batch_size != 2:
            errors.append(f"step dump text_encoder_output_batch_size {batch_size} != 2")
        for index in range(batch_size):
            key = f"text_encoder_output_{index}"
            if key not in keys:
                errors.append(f"step dump missing {key}")
                continue
            tensor = fh.get_tensor(key)
            if len(tensor.shape) != 2 or tensor.shape[1] != 2560:
                errors.append(f"step dump {key} shape {tuple(tensor.shape)} is not (*, 2560)")
            if str(tensor.dtype) != "torch.bfloat16":
                errors.append(f"step dump {key} dtype {tensor.dtype} != torch.bfloat16")
        if len(keys) < 32:
            errors.append(f"step dump tensor count {len(keys)} is too small for a full trace")
    return errors


def _validate_adapter_dump(path: Path, meta: dict) -> list[str]:
    errors: list[str] = []
    names = meta.get("trainable_parameters", {}).get("names")
    if not isinstance(names, list) or not names:
        return ["meta.trainable_parameters.names is missing or empty"]
    expected_names = {str(name) for name in names}
    if any(name.startswith("unknown_") for name in expected_names):
        errors.append("adapter names include unknown_* fallback entries")
    if not all(name.startswith("transformer.") for name in expected_names):
        errors.append("adapter names are not all OneTrainer transformer.* keys")

    handle, error = _load_safe_tensors(path)
    if error is not None:
        return [error]
    assert handle is not None
    with handle as fh:
        keys = set(fh.keys())
        expected_count = len(expected_names)
        if len(keys) != expected_count * len(ADAPTER_PHASES):
            errors.append(
                f"adapter tensor count {len(keys)} != "
                f"{expected_count} trainables * {len(ADAPTER_PHASES)} phases"
            )

        for phase in ADAPTER_PHASES:
            prefix = phase + "."
            phase_names = {key.removeprefix(prefix) for key in keys if key.startswith(prefix)}
            if phase_names != expected_names:
                missing = sorted(expected_names - phase_names)[:8]
                extra = sorted(phase_names - expected_names)[:8]
                errors.append(f"{phase} key set mismatch missing={missing} extra={extra}")

        for phase in ADAPTER_PHASES[:4]:
            sample_key = f"{phase}.{names[0]}"
            if sample_key not in keys:
                continue
            tensor = fh.get_tensor(sample_key)
            if str(tensor.dtype) != "torch.float32":
                errors.append(f"{sample_key} dtype {tensor.dtype} != torch.float32")

        nonzero_grad = False
        for key in sorted(keys):
            if key.startswith("adapter_pre_clip_grad.") and key.endswith(".lora_up.weight"):
                tensor = fh.get_tensor(key)
                if float(tensor.detach().to(dtype=tensor.dtype).abs().max().item()) > 0.0:
                    nonzero_grad = True
                    break
        if not nonzero_grad:
            errors.append("adapter_pre_clip_grad has no nonzero sampled lora_up gradients")

    return errors


def _validate_meta(path: Path, config: Path, parity_dir: Path) -> tuple[list[str], dict | None, str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing meta artifact: {path}"], None, "missing"
    try:
        with path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception as exc:
        return [f"could not read meta artifact {path}: {exc}"], None, "invalid"

    if meta.get("producer") != PRODUCER:
        errors.append(f"meta producer {meta.get('producer')!r} != {PRODUCER!r}")
    if Path(str(meta.get("config_path", ""))).resolve() != config.resolve():
        errors.append(f"meta config_path {meta.get('config_path')!r} != {str(config.resolve())!r}")
    if meta.get("prefix") != "zimage_train_ref":
        errors.append(f"meta prefix {meta.get('prefix')!r} != 'zimage_train_ref'")
    if not isinstance(meta.get("max_steps"), int) or int(meta.get("max_steps", 0)) < 1:
        errors.append(f"meta max_steps {meta.get('max_steps')!r} must be >= 1")
    if meta.get("adapter_dump") != "step-with-grads":
        errors.append(f"meta adapter_dump {meta.get('adapter_dump')!r} != 'step-with-grads'")

    runtime = meta.get("runtime_config", {})
    expected_runtime = {
        "model_type": "Z_IMAGE",
        "training_method": "LORA",
        "train_dtype": "BFLOAT_16",
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.0003,
        "learning_rate_scheduler": "CONSTANT",
        "optimizer": "ADAMW",
        "clip_grad_norm": 1.0,
        "lora_rank": 16,
        "lora_alpha": 1.0,
        "lora_weight_dtype": "FLOAT_32",
        "layer_filter_preset": "attn-mlp",
    }
    for key, expected in expected_runtime.items():
        actual = runtime.get(key)
        if actual != expected:
            errors.append(f"meta runtime_config.{key} {actual!r} != {expected!r}")

    trainables = meta.get("trainable_parameters", {})
    if trainables.get("count") != 420:
        errors.append(f"meta trainable_parameters.count {trainables.get('count')!r} != 420")
    names = trainables.get("names")
    if not isinstance(names, list) or len(names) != trainables.get("count"):
        errors.append("meta trainable parameter names are missing or count-mismatched")

    steps = meta.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append("meta.steps is missing or empty")
        return errors, meta, "invalid"
    step0 = steps[0]
    expected_step_path = str((parity_dir / "zimage_train_ref_step000.safetensors").resolve())
    expected_adapter_path = str((parity_dir / "zimage_train_ref_step000_adapters.safetensors").resolve())
    if Path(str(step0.get("safetensors", ""))).resolve() != Path(expected_step_path):
        errors.append(f"meta step0 safetensors {step0.get('safetensors')!r} != {expected_step_path!r}")
    if Path(str(step0.get("adapter_safetensors", ""))).resolve() != Path(expected_adapter_path):
        errors.append(
            f"meta step0 adapter_safetensors {step0.get('adapter_safetensors')!r} "
            f"!= {expected_adapter_path!r}"
        )
    for key in ("loss_pre_scale", "loss_for_backward", "grad_norm_no_clip", "grad_norm_pre_clip"):
        if not _finite_float(step0.get(key)) or float(step0[key]) <= 0.0:
            errors.append(f"meta step0 {key} is not a positive finite scalar: {step0.get(key)!r}")
    if step0.get("lr_before") != [0.0]:
        errors.append(f"meta step0 lr_before {step0.get('lr_before')!r} != [0.0]")
    lr_after = step0.get("lr_after")
    if not isinstance(lr_after, list) or len(lr_after) != 1 or not _finite_float(lr_after[0]) or lr_after[0] <= 0:
        errors.append(f"meta step0 lr_after is not a positive one-entry list: {lr_after!r}")

    delta = step0.get("adapter_delta", {})
    if step0.get("lr_before") == [0.0]:
        evidence_level = "state-init"
        if delta.get("l2") != 0.0 or delta.get("max_abs") != 0.0:
            errors.append(f"meta step0 zero-lr adapter_delta is unexpectedly nonzero: {delta!r}")
    else:
        evidence_level = "update-bearing"
        if not _finite_float(delta.get("l2")) or float(delta["l2"]) <= 0.0:
            errors.append(f"meta step0 adapter_delta.l2 is not positive: {delta!r}")
    return errors, meta, evidence_level


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parity-dir", type=Path, default=DEFAULT_PARITY_DIR)
    parser.add_argument("--input-dump", type=Path, default=DEFAULT_INPUT_DUMP)
    parser.add_argument(
        "--require-input-dump",
        action="store_true",
        help="also validate the current input-only OneTrainer dump surface",
    )
    parser.add_argument(
        "--expect-missing",
        action="store_true",
        help="return success only for the currently documented missing-artifact blocker",
    )
    args = parser.parse_args()

    if not args.config.is_file():
        return _fail(f"missing OneTrainer config: {args.config}", args.expect_missing)
    with args.config.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    required_values = {
        "model_type": "Z_IMAGE",
        "training_method": "LORA",
        "batch_size": 2,
        "resolution": "512",
        "train_dtype": "BFLOAT_16",
        "weight_dtype": "BFLOAT_16",
        "output_dtype": "BFLOAT_16",
    }
    for key, expected in required_values.items():
        actual = cfg.get(key)
        if actual != expected:
            return _fail(
                f"{args.config}: expected {key}={expected!r}, got {actual!r}",
                args.expect_missing,
            )

    for key in ("base_model_name", "cache_dir", "output_model_destination"):
        if not cfg.get(key):
            return _fail(f"{args.config}: missing {key}", args.expect_missing)

    if args.require_input_dump:
        error = _validate_input_dump(args.input_dump)
        if error is not None:
            print(f"[zimage-train-ref-contract] FAIL: {error}")
            return 1
        print(f"[zimage-train-ref-contract] input dump PASS: {args.input_dump}")

    missing = [name for name in REQUIRED_ARTIFACTS if not (args.parity_dir / name).is_file()]
    if missing:
        return _fail(
            "missing required train-ref artifacts in "
            + str(args.parity_dir)
            + ": "
            + ", ".join(missing),
            args.expect_missing,
        )

    if args.expect_missing:
        print("[zimage-train-ref-contract] FAIL: expected missing artifacts, but all are present")
        return 1

    meta_path = args.parity_dir / "zimage_train_ref_meta.json"
    step_path = args.parity_dir / "zimage_train_ref_step000.safetensors"
    adapter_path = args.parity_dir / "zimage_train_ref_step000_adapters.safetensors"
    errors: list[str] = []
    meta_errors, meta, evidence_level = _validate_meta(meta_path, args.config, args.parity_dir)
    errors.extend(meta_errors)
    if meta is not None:
        errors.extend(_validate_step_dump(step_path))
        errors.extend(_validate_adapter_dump(adapter_path, meta))
    if errors:
        for error in errors:
            print(f"[zimage-train-ref-contract] FAIL: {error}")
        return 1

    print(
        "[zimage-train-ref-contract] PASS: "
        + str(args.config)
        + " has required train-ref artifacts in "
        + str(args.parity_dir)
        + f" (evidence={evidence_level})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
