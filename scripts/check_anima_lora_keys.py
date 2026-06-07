#!/usr/bin/env python3
"""Static/header guard for Anima OneTrainer raw LoRA key parity."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ONETRAINER_ANIMA = Path("/home/alex/OneTrainer-anima-ref")

OT_SETUP = ONETRAINER_ANIMA / "modules/modelSetup/AnimaLoRASetup.py"
OT_MODEL = ONETRAINER_ANIMA / "modules/model/AnimaModel.py"
OT_SAVER = ONETRAINER_ANIMA / "modules/modelSaver/anima/AnimaLoRASaver.py"
OT_LORA_MODULE = ONETRAINER_ANIMA / "modules/module/LoRAModule.py"
OT_MODULE_FILTER = ONETRAINER_ANIMA / "modules/util/ModuleFilter.py"
OT_CONFIG_CLASS = ONETRAINER_ANIMA / "modules/util/config/TrainConfig.py"
OT_CONFIG = ONETRAINER_ANIMA / "configs/anima_100step_baseline.json"
OT_BASELINE = ONETRAINER_ANIMA / "output/anima_100step_baseline/lora.safetensors"
OT_METRICS_WITH_GRAD = ONETRAINER_ANIMA / "output/anima_100step_baseline/metrics_with_grad.json"
OT_CACHE = ONETRAINER_ANIMA / "workspace-cache/anima_100step_baseline"
OT_STEP_PARITY = REPO / "serenitymojo/models/anima/parity/anima_ot_step_parity.mojo"
OT_ARTIFACT_SMOKE = REPO / "serenitymojo/models/anima/parity/anima_train_ref_artifact_smoke.mojo"
TRAIN_REF_DUMP = Path("/home/alex/onetrainer-mojo/parity/anima_train_ref_step000.safetensors")
TRAIN_REF_ADAPTER_DUMP = Path("/home/alex/onetrainer-mojo/parity/anima_train_ref_step000_adapters.safetensors")
TRAIN_REF_META = Path("/home/alex/onetrainer-mojo/parity/anima_train_ref_meta.json")
TRAIN_REF_ARTIFACT_SMOKE = REPO / "serenitymojo/models/anima/parity/anima_train_step_ref_artifact_smoke.mojo"
TRAIN_REF_ADAPTER_UPDATE_GATE = REPO / "scripts/check_anima_adapter_update_replay.py"

STACK = REPO / "serenitymojo/models/anima/anima_stack_lora.mojo"
LORA_BLOCK = REPO / "serenitymojo/models/anima/lora_block.mojo"
LORA_SAVE = REPO / "serenitymojo/training/lora_save.mojo"

NUM_BLOCKS = 28
RANK = 16
D_MODEL = 2048
JOINT = 1024
F_MLP = 8192

FILTERS = ("attn1", "attn2", "ff")
IMPLEMENTED_MODULES: tuple[tuple[str, str, str], ...] = (
    ("attn1.to_q", "D", "D"),
    ("attn1.to_k", "D", "D"),
    ("attn1.to_v", "D", "D"),
    ("attn1.to_out.0", "D", "D"),
    ("attn2.to_q", "D", "D"),
    ("attn2.to_k", "JOINT", "D"),
    ("attn2.to_v", "JOINT", "D"),
    ("attn2.to_out.0", "D", "D"),
    ("ff.net.0.proj", "D", "F"),
    ("ff.net.2", "F", "D"),
)
OT_MODULES = tuple(module for module, _, _ in IMPLEMENTED_MODULES)
REFERENCE_FILTERED_NON_LORA = (
    "attn1.norm_q",
    "attn1.norm_k",
    "attn2.norm_q",
    "attn2.norm_k",
)
REFERENCE_FILTERED_SURFACE = (
    "attn1.norm_q",
    "attn1.norm_k",
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.norm_q",
    "attn2.norm_k",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
SAVE_SUFFIXES = ("alpha", "lora_down.weight", "lora_up.weight")
KEY_RE = re.compile(
    r"^transformer\.transformer_blocks\.(?P<block>\d+)\."
    r"(?P<module>.+)\.(?P<suffix>alpha|lora_down\.weight|lora_up\.weight)$"
)


@dataclass(frozen=True)
class InventoryContract:
    name: str
    num_blocks: int
    rank: int
    d_model: int
    joint: int
    f_mlp: int

    @property
    def adapters(self) -> int:
        return self.num_blocks * len(OT_MODULES)

    @property
    def tensors(self) -> int:
        return self.adapters * len(SAVE_SUFFIXES)


PRODUCTION = InventoryContract("production", NUM_BLOCKS, RANK, D_MODEL, JOINT, F_MLP)
SMOKE = InventoryContract("reduced-smoke", 1, 2, 8, 4, 16)


def die(msg: str) -> None:
    raise SystemExit(f"[anima-lora-keys] FAIL: {msg}")


def read_text(path: Path) -> str:
    if not path.exists():
        die(f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        die(f"missing required JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        die(f"missing {label}: {needle}")


def saved_key(block: int, module: str, suffix: str) -> str:
    return f"transformer.transformer_blocks.{block}.{module}.{suffix}"


def _dim(contract: InventoryContract, token: str) -> int:
    if token == "D":
        return contract.d_model
    if token == "JOINT":
        return contract.joint
    if token == "F":
        return contract.f_mlp
    die(f"unhandled dimension token: {token}")


def module_shape(module: str, suffix: str, contract: InventoryContract) -> list[int]:
    if suffix == "alpha":
        return []
    dims = {name: (in_f, out_f) for name, in_f, out_f in IMPLEMENTED_MODULES}
    if module not in dims:
        die(f"unhandled Anima module shape: {module}")
    in_f = _dim(contract, dims[module][0])
    out_f = _dim(contract, dims[module][1])
    if suffix == "lora_down.weight":
        return [contract.rank, in_f]
    if suffix == "lora_up.weight":
        return [out_f, contract.rank]
    die(f"unhandled suffix shape: {suffix}")


def expected_key_set(contract: InventoryContract) -> set[str]:
    return {
        saved_key(block, module, suffix)
        for block in range(contract.num_blocks)
        for module in OT_MODULES
        for suffix in SAVE_SUFFIXES
    }


def read_safetensors_header(path: Path) -> dict:
    if not path.exists():
        die(f"missing safetensors file: {path}")
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            die(f"truncated safetensors header length: {path}")
        (header_len,) = struct.unpack("<Q", raw_len)
        header_raw = fh.read(header_len)
        if len(header_raw) != header_len:
            die(f"truncated safetensors header: {path}")
    return json.loads(header_raw)


def _require_under_anima_ref(label: str, path: Path) -> None:
    root = ONETRAINER_ANIMA.resolve(strict=False)
    resolved = path.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        die(f"{label} must stay under {ONETRAINER_ANIMA}, got {path}")


def _first_scalar(events: list[dict], tag: str) -> dict:
    for event in events:
        if event.get("tag") == tag:
            return event
    die(f"missing scalar event {tag!r} in {OT_METRICS_WITH_GRAD}")


def _require_finite_positive_scalar(event: dict, label: str) -> None:
    value = event.get("value")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
        die(f"{label} must be a finite positive scalar, got {value!r}")
    if event.get("global_step") != 0:
        die(f"{label} must pin the first OneTrainer optimizer step at global_step=0, got {event.get('global_step')!r}")


def _count_pt_files(root: Path) -> int:
    if not root.exists():
        die(f"missing OneTrainer cache directory: {root}")
    return sum(1 for _ in root.rglob("*.pt"))


def _require_variation_zero_cache(root: Path, label: str) -> None:
    matches = sorted(root.glob("*/variation-0/0.pt"))
    if not matches:
        die(f"missing first-step {label} cache shard under {root}")


def _mojo_artifact_gate_consumes_baseline() -> bool:
    smoke = read_text(OT_ARTIFACT_SMOKE)
    return str(OT_BASELINE) in smoke and "SafeTensors.open(path)" in smoke


def _local_train_ref_artifacts_present() -> bool:
    return (
        TRAIN_REF_DUMP.exists()
        and TRAIN_REF_DUMP.stat().st_size > 8
        and TRAIN_REF_ADAPTER_DUMP.exists()
        and TRAIN_REF_ADAPTER_DUMP.stat().st_size > 8
    )


def _mojo_step_artifact_gate_consumes_train_ref() -> bool:
    if not TRAIN_REF_ARTIFACT_SMOKE.exists():
        return False
    smoke = TRAIN_REF_ARTIFACT_SMOKE.read_text(encoding="utf-8")
    return (
        str(TRAIN_REF_DUMP) in smoke
        and str(TRAIN_REF_ADAPTER_DUMP) in smoke
        and str(TRAIN_REF_META) in smoke
        and "SafeTensors.open(String(STEP_DUMP))" in smoke
        and "SafeTensors.open(String(ADAPTER_DUMP))" in smoke
        and "optimizer_before" in smoke
        and "optimizer_after" in smoke
        and "parameter_entries" in smoke
        and "lr_before" in smoke
        and "lr_after" in smoke
        and "adapter_after - adapter_post" in smoke
        and "STATE_INIT_ATOL" in smoke
        and "anima_lora_adamw_step(" in smoke
        and "artifact/state-init" in smoke
        and "consumption only" in smoke
    )


def _first_lr(value: object) -> float:
    if isinstance(value, list) and value:
        return float(value[0])
    if value is None:
        return 0.0
    return float(value)


def _optimizer_state(step: dict, owner: str) -> dict:
    state_owner = step.get(owner, {})
    if not isinstance(state_owner, dict):
        die(f"train-ref meta step {owner} must be an object")
    state = state_owner.get("state", {})
    if not isinstance(state, dict):
        die(f"train-ref meta step {owner}.state must be an object")
    return state


def _read_train_ref_meta() -> dict | None:
    if not TRAIN_REF_META.exists():
        return None
    meta = read_json(TRAIN_REF_META)
    steps = meta.get("steps")
    if not isinstance(steps, list) or not steps:
        die(f"{TRAIN_REF_META} must contain at least one captured step")
    return meta


def _train_ref_steps(meta: dict) -> list[dict]:
    steps = meta.get("steps")
    if not isinstance(steps, list) or not steps:
        die(f"{TRAIN_REF_META} must contain at least one captured step")
    out: list[dict] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            die(f"{TRAIN_REF_META} steps[{index}] must be an object")
        out.append(step)
    return out


def _has_update_bearing_step(meta: dict) -> bool:
    for step in _train_ref_steps(meta):
        adapter_path = step.get("adapter_safetensors")
        if not isinstance(adapter_path, str) or not adapter_path:
            continue
        if _first_lr(step.get("lr_before")) <= 0.0:
            continue
        path = Path(adapter_path)
        if path.exists() and path.stat().st_size > 8:
            return True
    return False


def _adapter_update_gate_wired() -> bool:
    if not TRAIN_REF_ADAPTER_UPDATE_GATE.exists():
        return False
    text = TRAIN_REF_ADAPTER_UPDATE_GATE.read_text(encoding="utf-8")
    return "check_adapter_update_replay" in text and '"anima"' in text


def report_train_ref_update_status(meta: dict | None) -> None:
    if meta is None:
        print(f"[anima-lora-keys] WARN: OneTrainer-anima train-ref meta: MISSING {TRAIN_REF_META}")
        print("[anima-lora-keys] OneTrainer-anima zero-lr optimizer-state-init oracle: NOT PROVEN")
        print("[anima-lora-keys] OneTrainer-anima update-bearing dump: MISSING")
        print("[anima-lora-keys] Mojo backward/AdamW consumer: MISSING")
        return

    runtime = meta.get("runtime_config", {})
    if not isinstance(runtime, dict):
        die(f"{TRAIN_REF_META} runtime_config must be an object")
    expected_runtime = {
        "model_type": "ANIMA",
        "training_method": "LORA",
        "train_dtype": "BFLOAT_16",
        "optimizer": "ADAMW",
    }
    for key, value in expected_runtime.items():
        if runtime.get(key) != value:
            die(f"train-ref meta runtime_config {key}={runtime.get(key)!r}, expected {value!r}")

    first_step = _train_ref_steps(meta)[0]
    lr_before = _first_lr(first_step.get("lr_before"))
    lr_after = _first_lr(first_step.get("lr_after"))
    before_state = _optimizer_state(first_step, "optimizer_before")
    after_state = _optimizer_state(first_step, "optimizer_after")
    before_entries = int(before_state.get("parameter_entries", -1))
    after_entries = int(after_state.get("parameter_entries", -1))
    after_keys = tuple(after_state.get("keys", ()))
    if before_entries != 0:
        die(f"first train-ref optimizer_before entries {before_entries}, expected 0 for state init")
    if after_entries <= 0:
        die(f"first train-ref optimizer_after entries {after_entries}, expected initialized AdamW state")
    if after_keys != ("exp_avg", "exp_avg_sq", "step"):
        die(f"first train-ref optimizer_after keys {after_keys!r}, expected AdamW exp_avg/exp_avg_sq/step")
    if lr_before != 0.0:
        die(f"first train-ref lr_before {lr_before}, expected 0.0 for state-init oracle")

    print(
        "[anima-lora-keys] OneTrainer-anima zero-lr optimizer-state-init oracle: PRESENT "
        f"lr={lr_before:g}->{lr_after:g} optimizer_entries={before_entries}->{after_entries} "
        f"optimizer_tensors={before_state.get('tensor_count', 0)}->{after_state.get('tensor_count', 0)} "
        f"optimizer_tensor_numel={before_state.get('tensor_numel', 0)}->{after_state.get('tensor_numel', 0)}"
    )

    if _adapter_update_gate_wired():
        print(
            "[anima-lora-keys] OneTrainer-anima adapter update oracle gate: PRESENT "
            f"{TRAIN_REF_ADAPTER_UPDATE_GATE.relative_to(REPO)}"
        )
    else:
        print(
            "[anima-lora-keys] WARN: OneTrainer-anima adapter update oracle gate: MISSING "
            f"{TRAIN_REF_ADAPTER_UPDATE_GATE}"
        )

    if _has_update_bearing_step(meta):
        print("[anima-lora-keys] OneTrainer-anima update-bearing dump: PRESENT")
    else:
        print(
            "[anima-lora-keys] OneTrainer-anima update-bearing dump: MISSING "
            "(current local dump is first optimizer step with lr_before=0.0; "
            "capture a later OneTrainer step with lr_before > 0 and nonzero "
            "adapter_after - adapter_post before claiming AdamW update parity)"
        )

    print(
        "[anima-lora-keys] Mojo backward/AdamW consumer: MISSING "
        "(current Anima train-ref gates consume artifacts/state-init oracle only; "
        "no Mojo backward or AdamW update path consumes the train-ref tensors yet)"
    )


def check_reference_training_dump(strict_consumption: bool) -> None:
    cfg = read_json(OT_CONFIG)
    metrics = read_json(OT_METRICS_WITH_GRAD)
    train_ref_meta = _read_train_ref_meta()

    expected_config = {
        "model_type": "ANIMA",
        "training_method": "LORA",
        "train_dtype": "BFLOAT_16",
        "output_dtype": "BFLOAT_16",
        "layer_filter": ",".join(FILTERS),
        "layer_filter_regex": False,
        "stop_training_after": 100,
        "stop_training_after_unit": "STEP",
        "seed": 42,
        "resolution": "512",
    }
    for key, value in expected_config.items():
        if cfg.get(key) != value:
            die(f"training dump config {key}={cfg.get(key)!r}, expected {value!r}")

    output_path = Path(str(cfg.get("output_model_destination", "")))
    cache_path = Path(str(cfg.get("cache_dir", "")))
    _require_under_anima_ref("output_model_destination", output_path)
    _require_under_anima_ref("cache_dir", cache_path)
    if output_path != OT_BASELINE:
        die(f"training dump output path {output_path} != {OT_BASELINE}")
    if cache_path != OT_CACHE:
        die(f"training dump cache path {cache_path} != {OT_CACHE}")
    if not OT_BASELINE.exists() or OT_BASELINE.stat().st_size <= 8:
        die(f"missing or empty OneTrainer Anima LoRA dump: {OT_BASELINE}")

    if metrics.get("status") != "completed":
        die(f"training dump metrics status {metrics.get('status')!r} != completed")
    if metrics.get("global_steps_seen") != 100:
        die(f"training dump global_steps_seen {metrics.get('global_steps_seen')!r} != 100")
    if metrics.get("loss_count") != 100:
        die(f"training dump loss_count {metrics.get('loss_count')!r} != 100")
    if metrics.get("grad_norm_count") != 100:
        die(f"training dump grad_norm_count {metrics.get('grad_norm_count')!r} != 100")
    first_loss = _first_scalar(metrics.get("scalar_events", []), "loss/train_step")
    first_grad = _first_scalar(metrics.get("scalar_events", []), "grad_norm")
    _require_finite_positive_scalar(first_loss, "first OneTrainer loss/train_step")
    _require_finite_positive_scalar(first_grad, "first OneTrainer grad_norm")

    image_cache = OT_CACHE / "image"
    text_cache = OT_CACHE / "text"
    image_count = _count_pt_files(image_cache)
    text_count = _count_pt_files(text_cache)
    if image_count <= 0 or text_count <= 0:
        die(f"training dump cache must include image/text .pt shards, got image={image_count} text={text_count}")
    _require_variation_zero_cache(image_cache, "image")
    _require_variation_zero_cache(text_cache, "text")
    train_ref_artifacts_present = _local_train_ref_artifacts_present()

    parity = read_text(OT_STEP_PARITY)
    synthetic_markers = (
        'REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/anima/parity/"',
        '_in("ot_scaled_bthwc")',
        '_in("ot_noise_bthwc")',
        '_in("ot_context")',
        '_in("ot_pred")',
        '_in("ot_loss")',
    )
    uses_repo_oracle = all(marker in parity for marker in synthetic_markers)

    print(
        "[anima-lora-keys] OneTrainer training dump evidence: PASS "
        f"steps=100 first_loss={first_loss['value']:.9g} first_grad_norm={first_grad['value']:.9g} "
        f"image_cache_pt={image_count} text_cache_pt={text_count} "
        f"final_lora={OT_BASELINE.name}"
    )
    if _mojo_artifact_gate_consumes_baseline():
        print(
            "[anima-lora-keys] OneTrainer-anima final LoRA artifact consumption: PASS "
            f"({OT_ARTIFACT_SMOKE.relative_to(REPO)} opens the final OneTrainer LoRA safetensors)"
        )
    else:
        print(
            "[anima-lora-keys] WARN: OneTrainer-anima final LoRA artifact consumption: NOT PROVEN "
            f"{OT_ARTIFACT_SMOKE.relative_to(REPO)} does not open {OT_BASELINE}"
        )

    if train_ref_artifacts_present:
        print(
            "[anima-lora-keys] OneTrainer train-ref first-step safetensors: PRESENT "
            f"step={TRAIN_REF_DUMP} adapters={TRAIN_REF_ADAPTER_DUMP}"
        )
        first_step_kind = "local train-ref step/adapters safetensors"
    else:
        first_step_kind = "metrics JSON plus PyTorch .pt cache shards, not safetensors"
        print(
            "[anima-lora-keys] WARN: OneTrainer train-ref first-step safetensors: MISSING "
            f"step={TRAIN_REF_DUMP} adapters={TRAIN_REF_ADAPTER_DUMP}"
        )

    if train_ref_artifacts_present and _mojo_step_artifact_gate_consumes_train_ref():
        print(
            "[anima-lora-keys] OneTrainer-anima first-step artifact consumption: PASS "
            f"({TRAIN_REF_ARTIFACT_SMOKE.relative_to(REPO)} opens the local train-ref "
            "step/adapters safetensors; artifact consumption only, not transformer/backward/AdamW parity)"
        )
    else:
        print(
            "[anima-lora-keys] WARN: OneTrainer-anima first-step artifact consumption: NOT PROVEN "
            f"(available first-step evidence is {first_step_kind})"
        )
        if not TRAIN_REF_ARTIFACT_SMOKE.exists():
            print(
                "[anima-lora-keys] WARN: missing in-repo Anima first-step artifact consumer "
                f"{TRAIN_REF_ARTIFACT_SMOKE.relative_to(REPO)}"
            )
        elif train_ref_artifacts_present:
            print(
                "[anima-lora-keys] WARN: current "
                f"{TRAIN_REF_ARTIFACT_SMOKE.relative_to(REPO)} does not consume both local train-ref safetensors"
            )
        if strict_consumption:
            die(
                "strict dump consumption requested, but no in-repo Mojo gate consumes the "
                "local Anima first-step train-ref safetensors"
            )
    report_train_ref_update_status(train_ref_meta)
    if uses_repo_oracle:
        print(
            "[anima-lora-keys] WARN: current "
            f"{OT_STEP_PARITY.relative_to(REPO)} still consumes repo-local ot_*.bin "
            "oracle files generated by anima_ot_step_oracle.py"
        )


def _method_return_expr(tree: ast.Module, class_name: str, method_name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    returns = [stmt.value for stmt in item.body if isinstance(stmt, ast.Return)]
                    if len(returns) != 1:
                        die(f"{class_name}.{method_name} must have exactly one return")
                    return returns[0]
    die(f"missing method {class_name}.{method_name}")


def extract_anima_reference_block_modules(model_text: str) -> list[str]:
    tree = ast.parse(model_text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "diffusers_to_original":
            returns = [stmt.value for stmt in node.body if isinstance(stmt, ast.Return)]
            if len(returns) != 1:
                die("diffusers_to_original must have exactly one return")
            mapping = ast.literal_eval(returns[0])
            for entry in mapping:
                if entry[0] == "transformer_blocks.{i}":
                    return [child[0] for child in entry[2]]
    die("missing AnimaModel.diffusers_to_original transformer block mapping")


def check_reference_surface(model_text: str) -> None:
    block_modules = extract_anima_reference_block_modules(model_text)
    filtered = [m for m in block_modules if any(f in m for f in FILTERS)]
    if tuple(filtered) != REFERENCE_FILTERED_SURFACE:
        die(
            "Anima reference layer_filter=attn1,attn2,ff surface changed: "
            f"{filtered!r} != {list(REFERENCE_FILTERED_SURFACE)!r}"
        )
    lora_surface = [m for m in filtered if m not in REFERENCE_FILTERED_NON_LORA]
    if tuple(lora_surface) != OT_MODULES:
        die(f"implemented Anima LoRA surface {lora_surface!r} != {list(OT_MODULES)!r}")


def check_onetrainer_source() -> None:
    setup = read_text(OT_SETUP)
    model = read_text(OT_MODEL)
    saver = read_text(OT_SAVER)
    lora_module = read_text(OT_LORA_MODULE)
    module_filter = read_text(OT_MODULE_FILTER)
    config_class = read_text(OT_CONFIG_CLASS)

    check_reference_surface(model)

    require(setup, "model.transformer_lora = LoRAModuleWrapper(", "OT transformer LoRA wrapper")
    require(setup, 'model.transformer, "transformer", config, config.layer_filter.split(",")', "OT transformer LoRA prefix/filter")
    require(setup, "model.transformer_lora.set_dropout", "OT LoRA dropout setup")
    require(setup, "config.lora_weight_dtype.torch_dtype()", "OT LoRA dtype setup")
    require(setup, 'raise NotImplementedError("Embeddings not implemented for Anima")', "OT Anima embedding rejection")
    if "text_encoder_lora = LoRAModuleWrapper" in setup:
        die("Anima setup unexpectedly creates a text_encoder_lora surface")

    saver_tree = ast.parse(saver)
    convert_return = _method_return_expr(saver_tree, "AnimaLoRASaver", "_get_convert_key_sets")
    if not isinstance(convert_return, ast.Constant) or convert_return.value is not None:
        die("AnimaLoRASaver._get_convert_key_sets must return None for raw state_dict save")
    require(saver, "state_dict |= model.transformer_lora.state_dict()", "OT Anima raw transformer state_dict save")
    require(saver, "state_dict |= model.lora_state_dict", "OT Anima prior LoRA merge save")
    require(saver, "self._save(model, output_model_format, output_model_destination, dtype)", "OT Anima saver delegates dtype/output handling")

    require(lora_module, "self.prefix = prefix + '.'", "OT PeftBase dotted prefix")
    require(lora_module, "case nn.Linear():", "OT Linear LoRA support")
    require(lora_module, "case nn.Conv2d():", "OT Conv2d LoRA support")
    require(lora_module, "Only Linear and Conv2d are supported layers", "OT unsupported module rejection")
    require(lora_module, "self.register_buffer(\"alpha\", torch.tensor(alpha))", "OT alpha buffer")
    require(lora_module, "self.lora_down, self.lora_up = self.create_layer()", "OT down/up layer creation")
    require(lora_module, "nn.init.kaiming_uniform_(self.lora_down.weight", "OT lora_down init")
    require(lora_module, "nn.init.zeros_(self.lora_up.weight)", "OT lora_up init")
    require(lora_module, "return self.orig_forward(x) + ld * (self.alpha / self.rank)", "OT alpha/rank forward scale")
    require(lora_module, "prefixed_name = (self.prefix + \".\" + name)", "OT wrapper prefixed module names")
    require(lora_module, "if not isinstance(child_module, Linear | Conv2d):", "OT wrapper Linear/Conv2d surface filter")
    require(lora_module, "module.state_dict(prefix=module.prefix)", "OT module-prefixed state_dict")
    require(lora_module, 'PeftType.LORA: ".lora_down.weight"', "OT LORA rank-check suffix")
    require(lora_module, 'if name.endswith(".alpha"):', "OT raw alpha dummy-module load path")

    require(module_filter, "self._pattern in module_name", "OT non-regex substring layer filter")
    require(config_class, 'data.append(("peft_type", PeftType.LORA', "OT default PEFT type")
    require(config_class, 'data.append(("lora_rank", 16', "OT default LoRA rank")
    require(config_class, 'data.append(("lora_alpha", 1.0', "OT default LoRA alpha")
    require(config_class, 'data.append(("lora_decompose", False', "OT default DoRA disabled")

    if OT_CONFIG.exists():
        cfg = json.loads(OT_CONFIG.read_text(encoding="utf-8"))
        expected = {
            "model_type": "ANIMA",
            "training_method": "LORA",
            "train_dtype": "BFLOAT_16",
            "output_dtype": "BFLOAT_16",
            "layer_filter": ",".join(FILTERS),
            "layer_filter_regex": False,
        }
        for key, value in expected.items():
            if cfg.get(key) != value:
                die(f"baseline config {key}={cfg.get(key)!r}, expected {value!r}")
        if cfg.get("lora_rank", RANK) != RANK:
            die(f"baseline lora_rank={cfg.get('lora_rank')!r}, expected default {RANK}")
        if cfg.get("lora_alpha", 1.0) != 1.0:
            die(f"baseline lora_alpha={cfg.get('lora_alpha')!r}, expected default 1.0")
        if cfg.get("lora_decompose", False):
            die("baseline unexpectedly enables DoRA/lora_decompose")

    print("[anima-lora-keys] OneTrainer source/save contract: PASS")


def _between(text: str, start: str, end: str) -> str:
    if start not in text:
        die(f"missing source marker: {start}")
    rest = text.split(start, 1)[1]
    if end not in rest:
        die(f"missing source marker after {start}: {end}")
    return rest.split(end, 1)[0]


def check_reference_artifact_gate_source() -> None:
    smoke = read_text(OT_ARTIFACT_SMOKE)
    require(smoke, str(OT_BASELINE), "Mojo real OneTrainer Anima baseline path")
    require(smoke, "SafeTensors.open(path)", "Mojo real Anima safetensors open")
    require(smoke, "EXPECTED_TENSORS = NUM_BLOCKS * ANIMA_MODULES * SAVE_SUFFIXES", "Mojo bounded tensor-count gate")
    require(smoke, "info.size == expected_bytes", "Mojo tensor byte-size gate")
    require(smoke, "tensor_bytes(key)", "Mojo scalar payload read")
    require(smoke, "bytes[0] == UInt8(0x80) and bytes[1] == UInt8(0x3F)", "Mojo BF16 alpha payload gate")
    if "DeviceContext" in smoke or "std.gpu" in smoke:
        die(f"{OT_ARTIFACT_SMOKE.relative_to(REPO)} must remain a no-CUDA artifact gate")
    print("[anima-lora-keys] Mojo reference artifact source gate: PASS")


def check_mojo_sources(strict_port: bool) -> None:
    stack = read_text(STACK)
    lora_block = read_text(LORA_BLOCK)
    lora_save = read_text(LORA_SAVE)

    require(lora_block, "comptime ANIMA_SLOTS = 10", "Mojo Anima slot count")
    for slot in (
        "SLOT_SA_Q", "SLOT_SA_K", "SLOT_SA_V", "SLOT_SA_O",
        "SLOT_CA_Q", "SLOT_CA_K", "SLOT_CA_V", "SLOT_CA_O",
        "SLOT_MLP1", "SLOT_MLP2",
    ):
        require(lora_block, f"comptime {slot}", f"Mojo Anima slot {slot}")

    require(stack, "def build_anima_lora_set", "Mojo Anima LoRA builder")
    for snippet, label in (
        ("make_lora_adapter(rank, alpha, D, D, seed)", "D->D adapters"),
        ("make_lora_adapter(rank, alpha, JOINT, D, seed)", "JOINT->D adapters"),
        ("make_lora_adapter(rank, alpha, D, F, seed)", "D->F adapter"),
        ("make_lora_adapter(rank, alpha, F, D, seed)", "F->D adapter"),
    ):
        require(stack, snippet, f"Mojo Anima builder {label}")

    ot_module_body = _between(stack, "def _anima_ot_module", "def _anima_ot_prefix")
    mojo_modules = tuple(re.findall(r'return String\("([^"]+)"\)', ot_module_body))
    if mojo_modules != OT_MODULES:
        die(f"Mojo _anima_ot_module order {list(mojo_modules)!r} != {list(OT_MODULES)!r}")

    require(stack, "def save_anima_lora", "Mojo Anima product save hook")
    require(stack, "def save_anima_lora_ot", "Mojo Anima OT save alias")
    require(stack, "def load_anima_lora_resume", "Mojo Anima resume loader")
    require(stack, "save_lora_onetrainer", "Mojo shared OneTrainer save call")
    require(stack, "return save_lora_onetrainer(_anima_ot_named(set), path, ctx)", "Mojo raw OT product save")
    require(stack, "return save_anima_lora(set, path, ctx)", "Mojo OT save alias")
    require(stack, "var prefixes = anima_ot_prefixes(num_blocks)", "Mojo raw OT resume prefixes")
    require(stack, 'String("transformer.transformer_blocks.")', "Mojo OT wrapper prefix")

    require(lora_save, "def save_lora_onetrainer", "shared OneTrainer saver")
    require(lora_save, 'names.append(nl.prefix + ".alpha")', "shared OT alpha suffix")
    require(lora_save, 'names.append(nl.prefix + ".lora_down.weight")', "shared OT down suffix")
    require(lora_save, 'names.append(nl.prefix + ".lora_up.weight")', "shared OT up suffix")
    require(lora_save, "_bf16_scalar", "shared BF16 alpha helper")
    require(lora_save, "Tensor.from_host_bf16(values^, sh^, ctx)", "shared BF16 alpha tensor")
    require(lora_save, "_bf16_2d(a.a.copy(), a.rank, a.in_f, ctx)", "shared BF16 down tensor")
    require(lora_save, "_bf16_2d(a.b.copy(), a.out_f, a.rank, ctx)", "shared BF16 up tensor")
    require(lora_save, "adapter_scale = alpha_h[0] / Float32(rank)", "shared raw-key resume alpha")

    gaps: list[str] = []
    save_body = _between(stack, "def save_anima_lora", "def save_anima_lora_ot")
    resume_body = _between(stack, "def load_anima_lora_resume", "# \u2500\u2500 FAITHFUL-RESUME")
    if "save_lora_peft" in save_body:
        gaps.append("Mojo Anima product save still delegates to generic PEFT lora_A/lora_B")
    if "Tensor.from_host(al_v" in stack:
        gaps.append("Mojo Anima OT save still writes F32 shaped alpha directly")
    if "var prefixes = anima_lora_prefixes(num_blocks)" in resume_body:
        gaps.append("Mojo Anima resume loader still uses legacy Mojo/PEFT prefixes")

    if gaps:
        for gap in gaps:
            print(f"[anima-lora-keys] WARN: {gap}")
        if strict_port:
            die("strict port requested and Mojo Anima raw-key gaps remain")
    check_reference_artifact_gate_source()
    print("[anima-lora-keys] Mojo source scaffold: PASS")


def _key_context(key: str, contract: InventoryContract) -> str:
    match = KEY_RE.match(key)
    if not match:
        return "not a raw transformer block LoRA key"
    block = int(match.group("block"))
    module = match.group("module")
    suffix = match.group("suffix")
    if block >= contract.num_blocks:
        return f"block {block} outside 0..{contract.num_blocks - 1}"
    if module not in OT_MODULES:
        return f"unsupported module surface {module!r}"
    if suffix not in SAVE_SUFFIXES:
        return f"unsupported suffix {suffix!r}"
    return ""


def _shape_numel(shape: list[int]) -> int:
    out = 1
    for dim in shape:
        out *= dim
    return out


def check_saved_header(path: Path, contract: InventoryContract) -> None:
    header = read_safetensors_header(path)
    keys = [k for k in header if k != "__metadata__"]
    expected = expected_key_set(contract)
    unsupported = [(k, _key_context(k, contract)) for k in keys if k not in expected]
    unsupported = [(k, reason) for k, reason in unsupported if reason]
    if unsupported:
        shown = "\n  ".join(f"{k} ({reason})" for k, reason in unsupported[:25])
        more = "" if len(unsupported) <= 25 else f"\n  ... {len(unsupported) - 25} more"
        die(f"unsupported saved LoRA surfaces in {path}:\n  {shown}{more}")

    missing = sorted(expected - set(keys))
    if missing:
        shown = "\n  ".join(missing[:25])
        more = "" if len(missing) <= 25 else f"\n  ... {len(missing) - 25} more"
        die(f"missing saved LoRA keys in {path}:\n  {shown}{more}")

    if len(keys) != contract.tensors:
        die(f"tensor count mismatch: got {len(keys)} expected {contract.tensors}")

    for block in range(contract.num_blocks):
        for module in OT_MODULES:
            for suffix in SAVE_SUFFIXES:
                key = saved_key(block, module, suffix)
                info = header[key]
                if info.get("dtype") != "BF16":
                    die(f"{key} dtype {info.get('dtype')} != BF16")
                expected_shape = module_shape(module, suffix, contract)
                if info.get("shape") != expected_shape:
                    die(f"{key} shape {info.get('shape')} != {expected_shape}")
                offsets = info.get("data_offsets")
                if (
                    not isinstance(offsets, list)
                    or len(offsets) != 2
                    or not all(isinstance(v, int) for v in offsets)
                    or offsets[1] < offsets[0]
                ):
                    die(f"{key} invalid data_offsets {offsets!r}")
                expected_nbytes = _shape_numel(expected_shape) * 2
                if offsets[1] - offsets[0] != expected_nbytes:
                    die(
                        f"{key} byte size {offsets[1] - offsets[0]} != {expected_nbytes}"
                    )

    print(
        "[anima-lora-keys] saved inventory: PASS "
        f"profile={contract.name} blocks={contract.num_blocks} "
        f"adapters={contract.adapters} tensors={len(keys)} rank={contract.rank} "
        f"D={contract.d_model} JOINT={contract.joint} F={contract.f_mlp}"
    )


def contract_from_args(args: argparse.Namespace) -> InventoryContract:
    base = SMOKE if args.profile == "smoke" else PRODUCTION
    return InventoryContract(
        name=base.name if not args.contract_name else args.contract_name,
        num_blocks=args.num_blocks if args.num_blocks is not None else base.num_blocks,
        rank=args.rank if args.rank is not None else base.rank,
        d_model=args.d_model if args.d_model is not None else base.d_model,
        joint=args.joint if args.joint is not None else base.joint,
        f_mlp=args.f_mlp if args.f_mlp is not None else base.f_mlp,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-port", action="store_true")
    parser.add_argument(
        "--profile",
        choices=("production", "smoke"),
        default="production",
        help="shape/count defaults for saved-header inventory",
    )
    parser.add_argument("--contract-name", default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--joint", type=int, default=None)
    parser.add_argument("--f-mlp", type=int, default=None)
    parser.add_argument(
        "--safetensors",
        type=Path,
        default=None,
        help="optional Anima LoRA safetensors header to inventory",
    )
    parser.add_argument(
        "--use-baseline",
        action="store_true",
        help="validate the OneTrainer Anima 100-step baseline if present, including first-step dump evidence",
    )
    parser.add_argument(
        "--check-training-dump",
        action="store_true",
        help="validate existing OneTrainer Anima dump metrics/cache and report whether the Mojo OT step gate consumes them",
    )
    parser.add_argument(
        "--strict-dump-consumption",
        action="store_true",
        help="fail if no in-repo Mojo gate consumes the local Anima train-ref step/adapters safetensors",
    )
    args = parser.parse_args()

    check_onetrainer_source()
    check_mojo_sources(args.strict_port)
    if args.check_training_dump or args.strict_dump_consumption or (args.use_baseline and OT_BASELINE.exists()):
        check_reference_training_dump(args.strict_dump_consumption)
    header_path = args.safetensors
    if header_path is None and args.use_baseline:
        if OT_BASELINE.exists():
            header_path = OT_BASELINE
        else:
            print(f"[anima-lora-keys] saved inventory: SKIP (missing {OT_BASELINE})")
    if header_path is not None:
        check_saved_header(header_path, contract_from_args(args))
    print("[anima-lora-keys] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
