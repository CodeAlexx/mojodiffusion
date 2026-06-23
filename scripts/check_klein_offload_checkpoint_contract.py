#!/usr/bin/env python3
"""Static Klein low-memory offload/checkpoint/backward evidence guard.

This is intentionally no-CUDA and no-artifact: it reads current source/docs and
reports whether Klein has production-ready low-memory offload/checkpoint
backward evidence. Default mode is report-only and exits 0 while the known
blockers are tracked. Use --strict to make those blockers exit 2.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ONETRAINER = Path("/home/alex/OneTrainer")

TRAIN = REPO / "../serenity-trainer/src/serenity_trainer/trainer/train_klein_real.mojo"
TRAIN_CONFIG = REPO / "serenitymojo/training/train_config.mojo"
TRAIN_CONFIG_READER = REPO / "serenitymojo/io/train_config_reader.mojo"
TRAIN_LOOP_POLICY = REPO / "serenitymojo/training/onetrainer_train_loop_policy.mojo"
OFFLOAD_CONFIG_SMOKE = REPO / "serenitymojo/io/offload_checkpoint_config_smoke.mojo"
CHECKPOINT_PRIMITIVE = REPO / "serenitymojo/training/checkpoint.mojo"
CONDUCTOR_POLICY = REPO / "serenitymojo/offload/conductor_policy.mojo"
CONDUCTOR_POLICY_SMOKE = REPO / "serenitymojo/offload/conductor_policy_smoke.mojo"
PLAN_SMOKE = REPO / "serenitymojo/offload/plan_smoke.mojo"
PLANNED_LOADER_SMOKE = REPO / "serenitymojo/offload/planned_loader_smoke.mojo"
RESIDENCY_SMOKE = REPO / "serenitymojo/offload/residency_smoke.mojo"
TURBO_SLOTS_SMOKE = REPO / "serenitymojo/offload/turbo_slots_smoke.mojo"
KLEIN_CONTROL_SMOKE = REPO / "../serenity-trainer/smoke/klein_train_control_wiring_smoke.mojo"
KLEIN_CADENCE = REPO / "serenitymojo/training/train_klein_cadence.mojo"
TRAIN_REF_SMOKE = REPO / "serenitymojo/models/klein/parity/klein_train_ref_artifact_smoke.mojo"
ADAMW_STATE_INIT_MOJO_SMOKE = (
    REPO / "serenitymojo/models/klein/parity/klein_lora_adamw_state_init_smoke.mojo"
)
ACTIVATION_TAPE_PLAN = REPO / "serenitymojo/models/klein/activation_tape_plan.mojo"
ACTIVATION_TAPE_PLAN_SMOKE = (
    REPO / "serenitymojo/models/klein/parity/klein_activation_tape_plan_smoke.mojo"
)
ACTIVATION_TAPE = REPO / "serenitymojo/models/klein/activation_tape.mojo"
ACTIVATION_TAPE_OFFLOAD_SMOKE = (
    REPO / "serenitymojo/models/klein/parity/klein_activation_tape_offload_smoke.mojo"
)
OFFLOADED_TAPE_PARITY = (
    REPO / "serenitymojo/models/klein/parity/klein_stack_lora_offloaded_tape_parity.mojo"
)
TRAIN_REF_ADAMW_STATE_INIT_REPLAY = (
    REPO / "serenitymojo/models/klein/parity/klein_train_ref_adamw_state_init_replay.mojo"
)
ADAPTER_GATE = REPO / "scripts/check_klein_adapter_grad_update_replay.py"
ADAMW_STATE_INIT_GATE = REPO / "scripts/check_klein_adamw_state_init_replay.py"
ADAMW_POSITIVE_LR_ORACLE = REPO / "scripts/check_klein_adamw_positive_lr_oracle.py"
LORA_CONTRACT = REPO / "scripts/check_klein_lora_contract.py"
USE_STATUS = REPO / "MOJO_TRAINER_USE_STATUS.md"
CORE_STATUS = REPO / "serenitymojo/docs/ONETRAINER_CORE_PORT_STATUS_2026-06-06.md"
FULL_PORT_ROADMAP = REPO / "FULL_PORT_ROADMAP.md"
INFERENCE_AUDIT = REPO / "AUDIT_INFERENCE_AND_CROSSPOLLINATION_2026-05-30.md"
SDXL_FLUX_KLEIN_STATUS = REPO / "serenitymojo/docs/SDXL_FLUX_KLEIN_PORT_STATUS.md"
SERENITY_MODULES = REPO / "serenitymojo/docs/SERENITYMOJO_MODULES.md"
MOJO_MODULES = REPO / "docs/MOJO_MODULES.md"
DTYPE_AUDIT = REPO / "serenitymojo/docs/DTYPE_LOADING_AUDIT_2026-06-05.md"
DEFAULT_CONFIG = REPO / "serenitymojo/configs/klein9b.json"

OT_GRADIENT_CHECKPOINTING_METHOD = (
    ONETRAINER / "modules/util/enum/GradientCheckpointingMethod.py"
)
OT_CHECKPOINTING_UTIL = ONETRAINER / "modules/util/checkpointing_util.py"
OT_LAYER_OFFLOAD_CONDUCTOR = ONETRAINER / "modules/util/LayerOffloadConductor.py"
OT_QUANTIZATION_UTIL = ONETRAINER / "modules/util/quantization_util.py"
OT_FLUX2_LORA_8GB_PRESET = ONETRAINER / "training_presets/#flux2 LoRA 8GB.json"
OT_FLUX2_FINETUNE_16GB_PRESET = ONETRAINER / "training_presets/#flux2 Finetune 16GB.json"

TRAIN_REF_STEP = "/home/alex/onetrainer-mojo/parity/klein_train_ref_step000.safetensors"

KNOWN_BLOCKERS: tuple[str, ...] = (
    "No accepted Mojo replay reruns Klein `predict -> backward_lora` from "
    f"`{TRAIN_REF_STEP}` and compares all adapter gradients against OneTrainer; "
    "the CPU loss/d_loss bridge from dumped predicted/target tensors is covered "
    "by `scripts/check_klein_loss_replay.py`, but it is not model/backward parity.",
    "No accepted full Mojo `predict -> backward_lora -> AdamW` replay compares "
    "all gradients, optimizer state payloads, and adapter deltas against "
    "OneTrainer; `scripts/check_klein_adamw_state_init_replay.py`, "
    "`serenitymojo/models/klein/parity/klein_lora_adamw_state_init_smoke.mojo`, "
    "and `serenitymojo/models/klein/parity/klein_train_ref_adamw_state_init_replay.mojo` "
    "cover zero-lr optimizer support evidence; "
    "`scripts/check_klein_adamw_positive_lr_oracle.py` covers only a CPU "
    "host-list synthetic positive-lr optimizer oracle from the same captured "
    "gradients and is not CUDA/GPU parity.",
    "No later OneTrainer Klein dump exists yet with `lr_before > 0` and nonzero "
    "`adapter_after - adapter_post_clip`; bounded synthetic positive-lr AdamW math "
    "is support evidence only, not a real later-step oracle or model parity.",
    "No accepted train-ref bounded-CUDA/offload/checkpoint backward replay proves "
    "the production low-memory Klein path against the train-ref dump. A small "
    "Klein LoRA offloaded-tape backward parity gate exists and passes, but it is "
    "not the train-ref replay.",
)


@dataclass(frozen=True)
class Source:
    path: Path
    text: str | None

    @property
    def exists(self) -> bool:
        return self.text is not None

    @property
    def rel(self) -> str:
        try:
            return str(self.path.relative_to(REPO))
        except ValueError:
            return str(self.path)

    def has(self, needle: str) -> bool:
        return self.text is not None and needle in self.text

    def line(self, needle: str) -> int | None:
        if self.text is None:
            return None
        idx = self.text.find(needle)
        if idx < 0:
            return None
        return self.text.count("\n", 0, idx) + 1


@dataclass(frozen=True)
class Fact:
    status: str
    label: str
    detail: str
    refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContractReport:
    facts: tuple[Fact, ...]
    known_blockers: tuple[str, ...]
    strict_blockers: tuple[str, ...]
    missing_sources: tuple[str, ...]
    accepted_full_predict_backward_adamw_replay: bool
    accepted_low_memory_offload_checkpoint_backward_replay: bool

    @property
    def production_ready(self) -> bool:
        return (
            self.accepted_full_predict_backward_adamw_replay
            and self.accepted_low_memory_offload_checkpoint_backward_replay
            and not self.strict_blockers
            and not self.missing_sources
        )


def read_source(path: Path) -> Source:
    try:
        return Source(path, path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Source(path, None)


def ref(src: Source, line: int | None) -> str:
    if line is None:
        return f"{src.rel}:missing"
    return f"{src.rel}:{line}"


def fact_for_needles(
    *,
    status_if_ok: str,
    status_if_missing: str,
    label: str,
    source: Source,
    needles: tuple[str, ...],
    detail_ok: str,
    detail_missing: str,
) -> Fact:
    if not source.exists:
        return Fact("MISSING", label, f"missing source file {source.rel}", (ref(source, None),))
    missing = [needle for needle in needles if not source.has(needle)]
    refs = tuple(ref(source, source.line(needle)) for needle in needles if source.has(needle))
    if missing:
        return Fact(
            status_if_missing,
            label,
            detail_missing + " missing markers: " + ", ".join(repr(item) for item in missing),
            refs,
        )
    return Fact(status_if_ok, label, detail_ok, refs)


def any_negative_near(text: str, term: str) -> bool:
    term_l = term.lower()
    for match in re.finditer(re.escape(term_l), text.lower()):
        start = max(0, match.start() - 80)
        context = text.lower()[start : match.end() + 80]
        if re.search(r"\b(no|not|still needs|must still fail|missing)\b", context):
            return True
    return False


def positive_claim_present(sources: tuple[Source, ...], terms: tuple[str, ...]) -> bool:
    """Return true only for an explicit non-negated acceptance claim."""
    for source in sources:
        if source.text is None:
            continue
        text = source.text
        lower = text.lower()
        for term in terms:
            if term.lower() not in lower:
                continue
            if not any_negative_near(text, term):
                return True
    return False


def gather_report() -> ContractReport:
    sources = {
        "train": read_source(TRAIN),
        "train_config": read_source(TRAIN_CONFIG),
        "reader": read_source(TRAIN_CONFIG_READER),
        "policy": read_source(TRAIN_LOOP_POLICY),
        "offload_smoke": read_source(OFFLOAD_CONFIG_SMOKE),
        "checkpoint_primitive": read_source(CHECKPOINT_PRIMITIVE),
        "conductor_policy": read_source(CONDUCTOR_POLICY),
        "conductor_policy_smoke": read_source(CONDUCTOR_POLICY_SMOKE),
        "plan_smoke": read_source(PLAN_SMOKE),
        "planned_loader_smoke": read_source(PLANNED_LOADER_SMOKE),
        "residency_smoke": read_source(RESIDENCY_SMOKE),
        "turbo_slots_smoke": read_source(TURBO_SLOTS_SMOKE),
        "control_smoke": read_source(KLEIN_CONTROL_SMOKE),
        "cadence": read_source(KLEIN_CADENCE),
        "train_ref_smoke": read_source(TRAIN_REF_SMOKE),
        "adamw_state_init_mojo_smoke": read_source(ADAMW_STATE_INIT_MOJO_SMOKE),
        "activation_tape_plan": read_source(ACTIVATION_TAPE_PLAN),
        "activation_tape_plan_smoke": read_source(ACTIVATION_TAPE_PLAN_SMOKE),
        "activation_tape": read_source(ACTIVATION_TAPE),
        "activation_tape_offload_smoke": read_source(ACTIVATION_TAPE_OFFLOAD_SMOKE),
        "offloaded_tape_parity": read_source(OFFLOADED_TAPE_PARITY),
        "train_ref_adamw_state_init_replay": read_source(TRAIN_REF_ADAMW_STATE_INIT_REPLAY),
        "adapter_gate": read_source(ADAPTER_GATE),
        "adamw_state_init_gate": read_source(ADAMW_STATE_INIT_GATE),
        "adamw_positive_lr_oracle": read_source(ADAMW_POSITIVE_LR_ORACLE),
        "lora_contract": read_source(LORA_CONTRACT),
        "use_status": read_source(USE_STATUS),
        "core_status": read_source(CORE_STATUS),
        "full_port_roadmap": read_source(FULL_PORT_ROADMAP),
        "inference_audit": read_source(INFERENCE_AUDIT),
        "sdxl_flux_klein_status": read_source(SDXL_FLUX_KLEIN_STATUS),
        "serenity_modules": read_source(SERENITY_MODULES),
        "mojo_modules": read_source(MOJO_MODULES),
        "dtype_audit": read_source(DTYPE_AUDIT),
        "default_config": read_source(DEFAULT_CONFIG),
        "ot_grad_ckpt_method": read_source(OT_GRADIENT_CHECKPOINTING_METHOD),
        "ot_checkpointing_util": read_source(OT_CHECKPOINTING_UTIL),
        "ot_layer_offload_conductor": read_source(OT_LAYER_OFFLOAD_CONDUCTOR),
        "ot_quantization_util": read_source(OT_QUANTIZATION_UTIL),
        "ot_flux2_lora_8gb_preset": read_source(OT_FLUX2_LORA_8GB_PRESET),
        "ot_flux2_finetune_16gb_preset": read_source(OT_FLUX2_FINETUNE_16GB_PRESET),
    }
    missing_sources = tuple(src.rel for src in sources.values() if not src.exists)

    train = sources["train"]
    train_config = sources["train_config"]
    reader = sources["reader"]
    policy = sources["policy"]
    offload_smoke = sources["offload_smoke"]
    checkpoint_primitive = sources["checkpoint_primitive"]
    conductor_policy = sources["conductor_policy"]
    conductor_policy_smoke = sources["conductor_policy_smoke"]
    plan_smoke = sources["plan_smoke"]
    planned_loader_smoke = sources["planned_loader_smoke"]
    residency_smoke = sources["residency_smoke"]
    turbo_slots_smoke = sources["turbo_slots_smoke"]
    control_smoke = sources["control_smoke"]
    cadence = sources["cadence"]
    train_ref_smoke = sources["train_ref_smoke"]
    adamw_state_init_mojo_smoke = sources["adamw_state_init_mojo_smoke"]
    activation_tape_plan = sources["activation_tape_plan"]
    activation_tape_plan_smoke = sources["activation_tape_plan_smoke"]
    activation_tape = sources["activation_tape"]
    activation_tape_offload_smoke = sources["activation_tape_offload_smoke"]
    offloaded_tape_parity = sources["offloaded_tape_parity"]
    train_ref_adamw_state_init_replay = sources["train_ref_adamw_state_init_replay"]
    adapter_gate = sources["adapter_gate"]
    adamw_state_init_gate = sources["adamw_state_init_gate"]
    adamw_positive_lr_oracle = sources["adamw_positive_lr_oracle"]
    lora_contract = sources["lora_contract"]
    use_status = sources["use_status"]
    core_status = sources["core_status"]
    full_port_roadmap = sources["full_port_roadmap"]
    inference_audit = sources["inference_audit"]
    sdxl_flux_klein_status = sources["sdxl_flux_klein_status"]
    serenity_modules = sources["serenity_modules"]
    mojo_modules = sources["mojo_modules"]
    dtype_audit = sources["dtype_audit"]
    default_config = sources["default_config"]
    ot_grad_ckpt_method = sources["ot_grad_ckpt_method"]
    ot_checkpointing_util = sources["ot_checkpointing_util"]
    ot_layer_offload_conductor = sources["ot_layer_offload_conductor"]
    ot_quantization_util = sources["ot_quantization_util"]
    ot_flux2_lora_8gb_preset = sources["ot_flux2_lora_8gb_preset"]
    ot_flux2_finetune_16gb_preset = sources["ot_flux2_finetune_16gb_preset"]

    facts: list[Fact] = []

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="OneTrainer CPU_OFFLOADED is offload-enabled checkpointing",
            source=ot_grad_ckpt_method,
            needles=(
                "CPU_OFFLOADED = 'CPU_OFFLOADED'",
                "def enabled(self)",
                "or self == GradientCheckpointingMethod.CPU_OFFLOADED",
                "def offload(self)",
                "return self == GradientCheckpointingMethod.CPU_OFFLOADED",
            ),
            detail_ok=(
                "OneTrainer defines CPU_OFFLOADED as checkpointing enabled with "
                "the offload flag set, distinct from plain ON."
            ),
            detail_missing="OneTrainer GradientCheckpointingMethod markers changed.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="OneTrainer Flux2/Klein checkpoint wrapper uses conductor hooks",
            source=ot_checkpointing_util,
            needles=(
                "class OffloadCheckpointLayer",
                "self.conductor.before_layer",
                "self.conductor.after_layer",
                "use_reentrant=True",
                "def enable_checkpointing_for_qwen3_encoder_layers",
                "No activation offloading, because hidden states are taken from the middle of the network by Flux2",
                "def enable_checkpointing_for_flux2_transformer",
                '(model.transformer_blocks,        ["hidden_states", "encoder_hidden_states"])',
                '(model.single_transformer_blocks, ["hidden_states"                         ])',
            ),
            detail_ok=(
                "OneTrainer Flux2/Klein uses OffloadCheckpointLayer conductor "
                "hooks around Flux2 transformer blocks and a Qwen3 encoder "
                "checkpoint path with no Qwen activation-offload args."
            ),
            detail_missing="OneTrainer Flux2/Klein checkpointing markers changed.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="OneTrainer conductor moves activations and layers",
            source=ot_layer_offload_conductor,
            needles=(
                "class LayerOffloadStrategy",
                "target_loaded_bytes = int(total_bytes * (1.0 - layer_offload_fraction))",
                "self.__offload_activations = config.gradient_checkpointing.offload() and config.enable_activation_offloading",
                "self.__offload_layers = config.gradient_checkpointing.offload() and config.layer_offload_fraction > 0",
                'self.__async_transfer = self.__train_device.type == "cuda" and config.enable_async_offloading',
                "self.__activations_map[call_index] = activations",
                "def __schedule_layer_to",
                "def __schedule_activations_to_device",
            ),
            detail_ok=(
                "OneTrainer CPU_OFFLOADED is a conductor-driven activation and "
                "layer movement runtime keyed by layer_offload_fraction, not "
                "only checkpoint-file block streaming."
            ),
            detail_missing="OneTrainer LayerOffloadConductor markers changed.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="OneTrainer layer offload rewrites tensor storage",
            source=ot_quantization_util,
            needles=(
                "def offload_quantized",
                "tensors = get_offload_tensors(module)",
                "tensor.data = tensor.data.to(device=device, non_blocking=non_blocking)",
                "new_tensor.copy_(tensor.data, non_blocking=non_blocking)",
                "tensor.data = new_tensor",
            ),
            detail_ok=(
                "OneTrainer layer offload moves module tensor storage to the "
                "target device/temp allocator."
            ),
            detail_missing="OneTrainer quantized offload tensor-move markers changed.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="OneTrainer Flux2 LoRA low-VRAM preset selects CPU_OFFLOADED",
            source=ot_flux2_lora_8gb_preset,
            needles=(
                '"gradient_checkpointing": "CPU_OFFLOADED"',
                '"layer_offload_fraction": 0.7',
            ),
            detail_ok=(
                "The OneTrainer Flux2 LoRA 8GB preset uses CPU_OFFLOADED with "
                "layer_offload_fraction=0.7."
            ),
            detail_missing="OneTrainer Flux2 LoRA 8GB preset no longer has expected CPU_OFFLOADED markers.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="OneTrainer Flux2 finetune low-VRAM preset requires fused Adafactor path",
            source=ot_flux2_finetune_16gb_preset,
            needles=(
                '"gradient_checkpointing": "CPU_OFFLOADED"',
                '"layer_offload_fraction": 0.6',
                '"optimizer": "ADAFACTOR"',
                '"fused_back_pass": true',
            ),
            detail_ok=(
                "The OneTrainer Flux2 finetune 16GB preset combines "
                "CPU_OFFLOADED with fused-back-pass Adafactor, a separate "
                "full-finetune/runtime requirement from LoRA weight streaming."
            ),
            detail_missing="OneTrainer Flux2 finetune 16GB preset no longer has expected low-VRAM markers.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="OneTrainer offload/checkpoint config fields",
            source=train_config,
            needles=(
                "var gradient_checkpointing: Int",
                "var enable_async_offloading: Bool",
                "var enable_activation_offloading: Bool",
                "var layer_offload_fraction: Float64",
                "def gradient_checkpointing_offload",
                "def activation_offload_enabled",
                "def layer_offload_enabled",
                "def validate_offload_checkpoint_config",
            ),
            detail_ok=(
                "TrainConfig carries typed OneTrainer offload/checkpoint policy "
                "and derived CPU_OFFLOADED activation/layer helpers."
            ),
            detail_missing="TrainConfig offload/checkpoint policy markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="AdamW positive-lr optimizer oracle scope",
            source=adamw_positive_lr_oracle,
            needles=(
                "Klein/Flux2 CPU host-list positive-lr AdamW support oracle",
                "adapter_post_clip_grad",
                "deterministic stochastic BF16 rounding",
                "DEFAULT_EXPECT_CHANGED = 27_262_275",
                "has_optimizer_only_positive_lr_oracle",
                "does not execute Klein predict -> backward_lora",
                "does not compare OneTrainer optimizer moment tensor payloads",
                "does not prove CUDA/GPU parity",
                "does not use a real OneTrainer lr_before>0 adapter_after update",
            ),
            detail_ok=(
                "The positive-lr oracle covers only CPU host-list optimizer "
                "math from captured gradients; it is not CUDA/GPU, backward, "
                "OneTrainer moment payload, real later-step, or low-memory "
                "checkpoint/offload parity."
            ),
            detail_missing="AdamW positive-lr optimizer oracle markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="config reader parses offload/checkpoint fields",
            source=reader,
            needles=(
                'elif key == "gradient_checkpointing":',
                'elif key == "enable_async_offloading":',
                'elif key == "enable_activation_offloading":',
                'elif key == "layer_offload_fraction":',
                "cfg.validate_offload_checkpoint_config()",
            ),
            detail_ok=(
                "JSON config parsing is wired for OneTrainer checkpoint/offload "
                "fields before the trainer validates the config."
            ),
            detail_missing="train_config_reader offload/checkpoint parsing markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="offload config smoke is policy-only",
            source=offload_smoke,
            needles=(
                "CPU_OFFLOADED maps to active OneTrainer offload policy",
                "activation offload derived",
                "layer offload derived",
                "offload_checkpoint_config_smoke PASS",
            ),
            detail_ok=(
                "The offload smoke covers config derivation only; it does not "
                "run Klein model backward."
            ),
            detail_missing="offload_checkpoint_config_smoke no longer has the expected policy markers.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="HostOffload primitive preserves raw activation bytes",
            source=checkpoint_primitive,
            needles=(
                "host offload via raw device<->host",
                "NOT tensor.to_host",
                "struct HostOffload",
                "var host: List[UInt8]",
                "def offload_to_host",
                "copy t's raw storage bytes",
                "ctx.enqueue_copy(dst_buf=staging, src_buf=t.buf)",
                "return HostOffload(host^, t.shape(), t.dtype())",
                "def restore_to_device",
                "return Tensor(dev^, off.shape.copy(), off.dtype, 0)",
            ),
            detail_ok=(
                "The existing activation offload primitive carries raw bytes, "
                "shape, and dtype, and restores a Tensor without an F32 host "
                "list boundary."
            ),
            detail_missing="HostOffload primitive no longer has the expected raw-byte markers.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="OneTrainer conductor policy math is scalar-only",
            source=conductor_policy,
            needles=(
                "OneTrainer CPU_OFFLOADED conductor policy",
                "var layer_offload_fraction: Float64",
                "onetrainer_conductor_policy_from_fields",
                "var activation_offload = checkpointing_offload and enable_activation_offloading",
                "var layer_offload = checkpointing_offload and layer_offload_fraction > Float64(0.0)",
                "is_cuda and enable_async_offloading",
                "target_loaded_bytes = Int(",
                "Float64(total_layer_bytes)",
                "Float64(1.0) - layer_offload_fraction",
                "Float64 is scalar policy arithmetic only; no tensor boundary upcast.",
            ),
            detail_ok=(
                "The local conductor policy keeps the OneTrainer layer fraction "
                "as Float64 and mirrors activation/layer/async gates plus "
                "target-loaded-byte math without tensor movement."
            ),
            detail_missing="OneTrainer conductor policy scalar markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="Klein activation tape plan pins BF16 boundary bytes",
            source=activation_tape_plan,
            needles=(
                "Klein activation tape byte plan for OneTrainer CPU_OFFLOADED work",
                "not quietly become a Float32 host carrier",
                "input_projection_boundary_elems",
                "double_input_boundary_elems",
                "single_input_boundary_elems",
                "final_boundary_elems",
                "current_boundary_bytes",
                "live_backward_boundary_bytes",
                "unused_input_projection_boundary_bytes",
                "current_boundary_f32_bytes",
                "live_backward_boundary_f32_bytes",
                "STDtype.BF16",
            ),
            detail_ok=(
                "The Klein activation tape plan records the current saved "
                "boundary activation categories, separates unused input-proj "
                "retention from the live backward tape, and keeps the target "
                "storage dtype at BF16 for CPU_OFFLOADED work."
            ),
            detail_missing="Klein activation tape plan markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="Klein activation tape smoke rejects f32-sized carrier target",
            source=activation_tape_plan_smoke,
            needles=(
                "No-CUDA Klein activation tape accounting smoke",
                "current boundary bytes bf16",
                "432013312",
                "live backward boundary bytes bf16",
                "419430400",
                "unused input projection bytes bf16",
                "12582912",
                "current boundary bytes f32",
                "864026624",
                "live backward boundary bytes f32",
                "838860800",
                "current f32 extra bytes",
                "live f32 extra bytes",
                "internal tails recompute-only",
                "klein_activation_tape_plan_smoke PASS",
            ),
            detail_ok=(
                "The activation tape smoke proves the no-CUDA accounting target: "
                "Klein currently retains 432,013,312 BF16 boundary bytes, the "
                "minimal live backward tape is 419,430,400 BF16 bytes, and an "
                "F32 carrier would double those storage targets."
            ),
            detail_missing="Klein activation tape plan smoke markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="Klein LoRA activation offload tape preserves raw bytes",
            source=activation_tape,
            needles=(
                "Dtype-preserving Klein LoRA activation tape offload",
                "dbl_img_in, dbl_txt_in, sgl_x_in, img_out, ln_img_out",
                "input-projection activations are intentionally not carried",
                "KleinStackLoraOffloadedTape",
                "var dbl_img_in: List[HostOffload]",
                "var dbl_txt_in: List[HostOffload]",
                "var sgl_x_in: List[HostOffload]",
                "def offload_klein_stack_lora_backward_tape",
                "offload_to_host(saved.dbl_img_in",
                "def restore_klein_stack_lora_backward_tape",
                "restore_to_device(tape.dbl_img_in",
                "zeros_device(shape^, dtype, ctx)",
            ),
            detail_ok=(
                "Klein now has a LoRA-specific offloaded tape bridge for the "
                "block inputs plus final-layer tensors consumed by backward. "
                "It uses HostOffload raw-byte carriers and does not carry the "
                "unused input-projection activations."
            ),
            detail_missing="Klein activation offload tape markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="Klein activation offload smoke is bounded bridge evidence only",
            source=activation_tape_offload_smoke,
            needles=(
                "Bounded Klein activation tape offload smoke",
                "stores raw bytes as HostOffload and restores tensors with the original dtype",
                "It is not a full model replay and does not accept CPU_OFFLOADED parity",
                "host bytes",
                "storage dtype bf16",
                "storage dtype f32 rejected",
                "unused input projection dummy bytes",
                "klein_activation_tape_offload_smoke PASS",
            ),
            detail_ok=(
                "The bounded tape smoke checks BF16 raw-byte offload/restore "
                "on tiny tensors and explicitly remains bridge evidence only."
            ),
            detail_missing="Klein activation offload smoke markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="Klein offloaded-tape backward parity gate exists",
            source=offloaded_tape_parity,
            needles=(
                "Klein LoRA stack backward parity through the offloaded activation tape",
                "compute resident-tape backward as the local reference",
                "offload the backward tape through HostOffload raw bytes",
                "restore the tape and rerun backward",
                "compare all load-bearing LoRA/input/modulation grads",
                "build_klein_lora_set",
                "all_storage_dtype",
                "offloaded host bytes",
                "klein_stack_lora_offloaded_tape_parity PASS",
                "does not accept product CPU_OFFLOADED parity",
            ),
            detail_ok=(
                "A bounded Klein LoRA backward parity gate now compares "
                "offloaded/restored activation-tape backward against resident "
                "tape backward over the current production LoRA slot layout."
            ),
            detail_missing="Klein offloaded-tape backward parity markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="conductor policy smoke covers OT low-VRAM fractions",
            source=conductor_policy_smoke,
            needles=(
                "no-CUDA gate for OneTrainer CPU_OFFLOADED policy",
                "layer_offload_fraction=0.7",
                "layer_offload_fraction=0.6",
                "lora target loaded bytes",
                "finetune target loaded bytes",
                "async requires cuda",
                "plain checkpointing keeps all bytes loaded",
                "conductor_policy_smoke PASS",
            ),
            detail_ok=(
                "The conductor policy smoke checks the Flux2/Klein LoRA 8GB "
                "and finetune 16GB offload fractions, async CUDA gating, and "
                "plain-checkpointing behavior without CUDA."
            ),
            detail_missing="conductor_policy_smoke markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="shared offload plan smoke is topology-only",
            source=plan_smoke,
            needles=(
                "compile-only gate for shared offload planning",
                "build_klein9b_block_plan",
                'String("klein block count"), klein.count(), 32',
                'String("klein cfg visits"), klein.branch_visits(cfg), 64',
                "build_qwenimage_block_plan",
                "build_flux1_dev_block_plan",
            ),
            detail_ok=(
                "The plan smoke validates block counts, CFG visits, tensor "
                "hints, and prefixes for shared offload plans without loading "
                "weights, moving activations, or running model backward."
            ),
            detail_missing="offload plan_smoke markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="planned loader smoke is API-only",
            source=planned_loader_smoke,
            needles=(
                "compile gate for the BlockPlan-aware loader API",
                "intentionally avoids opening checkpoints or loading tensors",
                "PlannedBlockLoader",
                "PlannedOffloadStats",
                'String("klein blocks"), klein.count(), 32',
                'String("klein cfg branch visits"), klein.branch_visits(cfg), 64',
            ),
            detail_ok=(
                "The planned-loader smoke typechecks the block-plan-aware "
                "loader API and stats surface without checkpoint IO or CUDA "
                "activation/layer offload."
            ),
            detail_missing="planned_loader_smoke markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="offload residency smoke is pure logic",
            source=residency_smoke,
            needles=(
                "Build + run (no GPU needed",
                "State machine",
                "Budget",
                "Refcount",
                "Klein9b full plan",
                "ALL ASSERTIONS PASSED",
            ),
            detail_ok=(
                "The residency smoke covers state, budget, refcount, eviction, "
                "CFG revisit, and Klein9B plan logic without CUDA; this is "
                "runtime scaffolding evidence, not training parity."
            ),
            detail_missing="offload residency smoke markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="turbo slot smoke is metadata-only",
            source=turbo_slots_smoke,
            needles=(
                "metadata-only turbo slots",
                "TurboSlotBackend.from_plan",
                "planned pinned bytes",
                "metadata handle has no tensors",
                "metadata evictions",
            ),
            detail_ok=(
                "The turbo slot smoke covers slot planning, staged/prepared "
                "metadata, and evictions without device tensors; this is not "
                "bounded CUDA offload replay."
            ),
            detail_missing="turbo slot smoke markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="Klein product loop selects CPU_OFFLOADED activation tape branch",
            source=train,
            needles=(
                "OT_GRAD_POLICY_ON_OR_CPU_OFFLOADED",
                "validate_ot_gradient_checkpointing_policy(",
                "var use_activation_tape_offload = cfg.activation_offload_enabled()",
                "klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch_offloaded_tape",
                "klein_stack_lora_backward_offloaded_tape_turbo_moddev_rope_scratch",
                "activation_tape_host_bytes",
            ),
            detail_ok=(
                "Klein now accepts OneTrainer CPU_OFFLOADED policy and routes "
                "activation-offloaded runs through the raw-byte offloaded tape "
                "forward/backward branch. This is product plumbing, not train-ref "
                "runtime parity evidence."
            ),
            detail_missing="Klein train-loop CPU_OFFLOADED branch markers are incomplete.",
        )
    )

    if not train.exists:
        facts.append(
            Fact(
                "MISSING",
                "Klein train loop preserves cache/input dtypes",
                f"missing source file {train.rel}",
                (ref(train, None),),
            )
        )
    else:
        required_dtype_markers = (
            "var img_tok = _latent_to_img_tokens_device(sample.latent, cfg.in_channels, ctx)",
            "var txt_tok = reshape(sample.text_embedding, preload_txt_sh.copy(), ctx)",
            "zero list below is only a host scalar source",
            "var uncond_dtype = STDtype.BF16",
            "uncond_dtype = cached_txt_tokens[0][].dtype()",
            "[N_TXT, cfg.joint_attention_dim], uncond_dtype, ctx",
        )
        forbidden_dtype_markers = (
            "cast_tensor_if_needed(_latent_to_img_tokens_device",
            "_latent_to_img_tokens_device(sample.latent, cfg.in_channels, ctx), STDtype.F32",
            "[N_TXT, cfg.joint_attention_dim], STDtype.F32, ctx",
        )
        missing_required = [
            marker for marker in required_dtype_markers if not train.has(marker)
        ]
        present_forbidden = [
            marker for marker in forbidden_dtype_markers if train.has(marker)
        ]
        refs = tuple(
            ref(train, train.line(marker))
            for marker in required_dtype_markers
            if train.has(marker)
        ) + tuple(
            ref(train, train.line(marker))
            for marker in forbidden_dtype_markers
            if train.has(marker)
        )
        if missing_required or present_forbidden:
            detail_parts: list[str] = []
            if missing_required:
                detail_parts.append(
                    "missing dtype-preserving markers: "
                    + ", ".join(repr(item) for item in missing_required)
                )
            if present_forbidden:
                detail_parts.append(
                    "forbidden hard-coded F32 boundary markers present: "
                    + ", ".join(repr(item) for item in present_forbidden)
                )
            facts.append(
                Fact(
                    "WARN",
                    "Klein train loop preserves cache/input dtypes",
                    "; ".join(detail_parts),
                    refs,
                )
            )
        else:
            facts.append(
                Fact(
                    "PASS",
                    "Klein train loop preserves cache/input dtypes",
                    (
                        "Klein cache preload no longer casts latent tokens to "
                        "F32, cached text embeddings keep their stored dtype, "
                        "and the caption-dropout placeholder follows the cached "
                        "text-token dtype instead of hard-coding F32."
                    ),
                    refs,
                )
            )

    facts.append(
        fact_for_needles(
            status_if_ok="WARN",
            status_if_missing="WARN",
            label="shared policy records missing CPU_OFFLOADED runtime plumbing",
            source=policy,
            needles=(
                "currently requires gradient_checkpointing=ON",
                "OFF retains too much activation state and CPU_OFFLOADED",
                "needs activation/layer offload runtime plumbing",
                "cannot honor CPU_OFFLOADED activation/layer offload yet",
            ),
            detail_ok=(
                "The shared validator still treats CPU_OFFLOADED as unsupported "
                "for current product paths that do not have runtime plumbing."
            ),
            detail_missing="shared policy CPU_OFFLOADED fail-loud markers changed.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="Klein control smoke accepts CPU_OFFLOADED config",
            source=control_smoke,
            needles=(
                "Scope: TrainConfig + sample cadence + path/output reachability only",
                "not construct DeviceContext, open the 9B checkpoint, or run train math",
                "CPU_OFFLOADED policy reaches Klein activation-tape path",
                '"gradient_checkpointing":"CPU_OFFLOADED"',
                '"enable_activation_offloading":true',
                '"layer_offload_fraction":0.7',
                "cfg.activation_offload_enabled()",
                "CPU_OFFLOADED Klein config PASS",
            ),
            detail_ok=(
                "The Klein no-CUDA control smoke validates the parsed "
                "CPU_OFFLOADED activation/layer offload policy without creating "
                "a DeviceContext or claiming runtime parity."
            ),
            detail_missing="Klein control smoke CPU_OFFLOADED markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="INFO",
            status_if_missing="WARN",
            label="Klein train loop has offload-turbo implementation path",
            source=train,
            needles=(
                "TurboPlannedLoader.open",
                "OffloadConfig.synchronous_single()",
                "klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch",
                "klein_stack_lora_backward_offload_turbo_moddev_rope_scratch",
                "klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch_offloaded_tape",
                "klein_stack_lora_backward_offloaded_tape_turbo_moddev_rope_scratch",
            ),
            detail_ok=(
                "Implementation markers for streamed/offload-turbo forward and "
                "backward are present; implementation presence is not evidence "
                "that the path was replay-validated against OneTrainer."
            ),
            detail_missing="Klein offload-turbo forward/backward implementation markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="weight streaming is not CPU_OFFLOADED parity",
            source=use_status,
            needles=(
                "Klein weight streaming is distinct from OneTrainer CPU_OFFLOADED",
                "TurboPlannedLoader",
                "not activation/layer CPU offload",
                "bounded-CUDA offload/checkpoint backward replay",
            ),
            detail_ok=(
                "The status doc explicitly separates current turbo weight "
                "streaming from OneTrainer CPU_OFFLOADED activation/layer "
                "offload and from bounded CUDA replay parity."
            ),
            detail_missing=(
                "The docs no longer explicitly separate turbo weight streaming "
                "from OneTrainer CPU_OFFLOADED activation/layer parity."
            ),
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="roadmap scopes Klein offloaded smokes to image/inference",
            source=full_port_roadmap,
            needles=(
                "coherent block-streamed inference/image smokes exist",
                "not OneTrainer `CPU_OFFLOADED` activation/layer parity",
                "backward replay",
                "sampler trajectory parity",
                "speed parity",
            ),
            detail_ok=(
                "The roadmap no longer lets Klein block-streamed image smokes "
                "stand in for OneTrainer CPU_OFFLOADED or train parity."
            ),
            detail_missing="FULL_PORT_ROADMAP.md has stale or missing Klein offload caveats.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="inference audit scopes Klein coherence to image smoke",
            source=inference_audit,
            needles=(
                "Block-streamed inference offload",
                "image-smoke evidence only",
                "not OneTrainer `CPU_OFFLOADED` activation/layer parity",
                "training/offload backward parity",
                "coherence bar",
            ),
            detail_ok=(
                "The inference audit now separates visual coherence and "
                "block-streamed inference from training CPU_OFFLOADED parity."
            ),
            detail_missing="AUDIT_INFERENCE_AND_CROSSPOLLINATION has stale Klein offload wording.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="legacy Klein image docs avoid full-parity wording",
            source=sdxl_flux_klein_status,
            needles=(
                "exercises the current end-to-end",
                "wiring/memory proof",
                "not OneTrainer sampler/trajectory/VAE-PNG parity",
                "speed/VRAM parity",
                "`CPU_OFFLOADED` activation/layer parity",
            ),
            detail_ok=(
                "The SDXL/Flux/Klein status doc scopes the one-step Klein image "
                "path as wiring/memory evidence only."
            ),
            detail_missing="SDXL_FLUX_KLEIN_PORT_STATUS has stale one-step image wording.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="module inventories distinguish block streaming from OT offload",
            source=serenity_modules,
            needles=(
                "block-streamed 1024 forward used by the image smoke",
                "not OneTrainer `CPU_OFFLOADED` activation/layer parity",
                "training backward parity",
                "speed/VRAM parity",
            ),
            detail_ok=(
                "The Serenity module inventory does not present image "
                "block-streaming as OneTrainer training offload parity."
            ),
            detail_missing="SERENITYMOJO_MODULES has stale Klein offload wording.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="trainer module inventory scopes partial real run",
            source=mojo_modules,
            needles=(
                "block-streamed `klein_stack_lora` fwd/bwd",
                "not OneTrainer `CPU_OFFLOADED` activation/layer parity",
                "PARTIAL REAL RUN",
                "not full OneTrainer predict/backward/AdamW replay parity",
            ),
            detail_ok=(
                "The trainer module inventory now separates the partial "
                "block-streamed run from full OneTrainer replay/offload parity."
            ),
            detail_missing="docs/MOJO_MODULES.md has stale Klein trainer wording.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="Flux dtype audit scopes CPU_OFFLOADED policy wording",
            source=dtype_audit,
            needles=(
                "accepted `CPU_OFFLOADED` checkpoint policy",
                "policy/wiring evidence only",
                "not OneTrainer activation/layer `CPU_OFFLOADED` parity",
                "predict/backward/AdamW replay",
                "speed parity",
            ),
            detail_ok=(
                "The dtype audit now scopes Flux CPU_OFFLOADED wording to "
                "policy/wiring evidence only."
            ),
            detail_missing="DTYPE_LOADING_AUDIT has stale CPU_OFFLOADED wording.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="INFO",
            status_if_missing="WARN",
            label="cadence supervisor is process separation, not replay evidence",
            source=cadence,
            needles=(
                "process-separated Klein LoRA cadence supervisor",
                "let that worker exit so CUDA releases the training stack/scratch slabs",
                "In-process 1024px sampling after step 500 can",
                "OOM",
            ),
            detail_ok=(
                "The cadence supervisor reduces validation sampling residency by "
                "process separation; it is not a low-memory checkpoint/backward "
                "parity proof."
            ),
            detail_missing="Klein cadence process-separation markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="Klein train-ref artifact smoke scope",
            source=train_ref_smoke,
            needles=(
                "Header/payload-only, no-CUDA gate",
                "bounded synthetic positive-lr",
                "This does not claim transformer, backward, real update-bearing",
                "expected 1728 Klein adapter/grad tensors",
            ),
            detail_ok=(
                "The Mojo train-ref smoke consumes real artifact headers/payloads "
                "and synthetic AdamW math, while explicitly excluding transformer, "
                "backward, real update-bearing, sampler, speed, and image parity."
            ),
            detail_missing="Klein train-ref smoke no-CUDA/artifact-only scope markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="adapter gradient/update oracle scope",
            source=adapter_gate,
            needles=(
                "This is a CPU/no-CUDA checker",
                'os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")',
                "has_gradient_oracle",
                "has_zero_lr_state_init_oracle",
                "has_synthetic_positive_lr_adamw_update",
                '"has_mojo_backward_adamw_parity": False',
                "MOJO_PARITY_BLOCKERS",
                "--require-mojo-parity",
            ),
            detail_ok=(
                "The adapter gate tracks gradient/state-init/synthetic AdamW "
                "support evidence and intentionally marks Mojo backward/AdamW "
                "parity as missing."
            ),
            detail_missing="adapter grad/update oracle scope markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="AdamW state-init replay scope",
            source=adamw_state_init_gate,
            needles=(
                "Klein/Flux2 AdamW zero-lr state-init replay guard",
                "has_zero_lr_state_init_replay",
                "post_clip_bf16_import_minus_post_clip_f32",
                "does not execute Mojo backward/AdamW",
                "does not prove nonzero update parity",
            ),
            detail_ok=(
                "The AdamW state-init gate streams the Klein adapter dump and "
                "covers CPU zero-lr moment projection only."
            ),
            detail_missing="AdamW state-init replay gate scope markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="bounded Mojo AdamW state-init smoke scope",
            source=adamw_state_init_mojo_smoke,
            needles=(
                "Bounded Mojo AdamW zero-lr state-init smoke",
                "klein_lora_adamw_step",
                "optimizer-path support evidence only",
                "does not consume OneTrainer train-ref tensors",
                "does not execute Klein predict/backward_lora",
                "does not prove full Mojo predict/backward/AdamW parity",
                "does not prove nonzero update parity",
                "does not prove low-memory offload/checkpoint backward parity",
            ),
            detail_ok=(
                "The bounded Mojo smoke exercises only synthetic zero-lr "
                "AdamW state-init support behavior through the model-level "
                "Klein optimizer path; it does not satisfy full "
                "predict/backward/AdamW replay or low-memory offload/checkpoint "
                "backward parity."
            ),
            detail_missing="bounded Mojo AdamW state-init smoke scope markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="train-ref Mojo AdamW state-init replay scope",
            source=train_ref_adamw_state_init_replay,
            needles=(
                "real OneTrainer adapter dump",
                "all 288 train-ref adapters",
                "adapter_post_clip_grad",
                "klein_lora_adamw_step",
                "does not execute Klein predict/backward_lora",
                "does not compare optimizer moment tensors against OneTrainer payloads",
                "does not prove full Mojo predict/backward/AdamW parity",
                "does not prove nonzero OneTrainer update parity",
                "does not prove low-memory offload/checkpoint backward parity",
            ),
            detail_ok=(
                "The train-ref Mojo replay executes the model-level AdamW helper "
                "at zero lr over all train-ref adapters and gradients; it is "
                "optimizer support evidence only, not predict/backward, "
                "OneTrainer moment-payload, nonzero-update, or low-memory "
                "checkpoint/offload parity."
            ),
            detail_missing="train-ref Mojo AdamW state-init replay scope markers are incomplete.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="Klein LoRA contract keeps artifact evidence scoped",
            source=lora_contract,
            needles=(
                "missing Mojo Klein backward/AdamW consumer",
                "not Mojo backward or AdamW parity",
                "bounded optimizer-math replay",
                "AdamW state-init replay gate",
                "bounded Mojo AdamW state-init smoke",
                "train-ref Mojo AdamW state-init replay",
                "CPU AdamW positive-lr support oracle",
            ),
            detail_ok=(
                "The Klein LoRA contract cross-checks that the adapter gate remains "
                "scoped to oracle/artifact evidence."
            ),
            detail_missing="Klein LoRA contract no longer references the expected scoped evidence markers.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="WARN",
            status_if_missing="WARN",
            label="status doc records exact Klein replay/offload blockers",
            source=use_status,
            needles=(
                "No accepted Mojo replay reruns Klein `predict -> backward_lora`",
                "check_klein_loss_replay.py",
                "No accepted full Mojo `predict -> backward_lora -> AdamW` replay",
                "check_klein_adamw_state_init_replay.py",
                "klein_lora_adamw_state_init_smoke.mojo",
                "klein_train_ref_adamw_state_init_replay.mojo",
                "check_klein_adamw_positive_lr_oracle.py",
                "No later OneTrainer Klein dump exists yet with `lr_before > 0`",
                "No accepted train-ref bounded-CUDA/offload/checkpoint backward replay proves",
                "klein_activation_tape_offload_smoke",
                "klein_stack_lora_offloaded_tape_parity",
            ),
            detail_ok=(
                "MOJO_TRAINER_USE_STATUS.md records the exact current Klein "
                "production blockers."
            ),
            detail_missing="MOJO_TRAINER_USE_STATUS.md does not include all expected Klein blockers.",
        )
    )

    facts.append(
        fact_for_needles(
            status_if_ok="WARN",
            status_if_missing="WARN",
            label="core status doc says full replay still needed",
            source=core_status,
            needles=(
                "accepted loss/state-init support evidence only",
                "klein_train_ref_adamw_state_init_replay.mojo",
                "check_klein_adamw_positive_lr_oracle.py",
                "klein_activation_tape_offload_smoke",
                "klein_stack_lora_offloaded_tape_parity",
                "accepted full predict/backward/AdamW replay",
                "must still fail with exit `2` until a later Klein OneTrainer dump",
                "`lr_before > 0`",
                "Klein's synthetic positive-lr",
            ),
            detail_ok=(
                "The core status companion also classifies Klein synthetic AdamW "
                "as support evidence, not production replay readiness."
            ),
            detail_missing="core status doc no longer includes the expected Klein replay caveats.",
        )
    )

    if default_config.exists:
        if default_config.has('"gradient_checkpointing"'):
            facts.append(
                Fact(
                    "INFO",
                    "default Klein config overrides gradient_checkpointing",
                    "The default config has an explicit gradient_checkpointing field; inspect its value.",
                    (ref(default_config, default_config.line('"gradient_checkpointing"')),),
                )
            )
        else:
            facts.append(
                Fact(
                    "INFO",
                    "default Klein config uses TrainConfig checkpoint/offload defaults",
                    "The default klein9b.json has no explicit gradient_checkpointing field, so TrainConfig.default() supplies ON.",
                    (ref(default_config, 1),),
                )
            )

    full_replay_sources = (
        train,
        train_ref_smoke,
        train_ref_adamw_state_init_replay,
        adapter_gate,
        adamw_positive_lr_oracle,
        lora_contract,
        use_status,
        core_status,
    )
    accepted_full_replay = (
        adapter_gate.has('"has_mojo_backward_adamw_parity": True')
        or positive_claim_present(full_replay_sources, ("accepted full predict/backward/AdamW replay",))
    )
    accepted_lowmem_replay = positive_claim_present(
        full_replay_sources,
        ("accepted train-ref bounded-CUDA/offload/checkpoint backward replay",),
    )

    strict_blockers: list[str] = []
    if not accepted_full_replay:
        strict_blockers.extend(KNOWN_BLOCKERS[:3])
    if not accepted_lowmem_replay:
        strict_blockers.append(KNOWN_BLOCKERS[3])
    if missing_sources:
        strict_blockers.append(
            "Required source/doc files were missing, so this static guard cannot verify current Klein evidence: "
            + ", ".join(missing_sources)
        )

    return ContractReport(
        facts=tuple(facts),
        known_blockers=KNOWN_BLOCKERS,
        strict_blockers=tuple(strict_blockers),
        missing_sources=missing_sources,
        accepted_full_predict_backward_adamw_replay=accepted_full_replay,
        accepted_low_memory_offload_checkpoint_backward_replay=accepted_lowmem_replay,
    )


def status_counts(facts: tuple[Fact, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for fact in facts:
        out[fact.status] = out.get(fact.status, 0) + 1
    return out


def print_report(report: ContractReport, *, ref_limit: int) -> None:
    print("Klein offload/checkpoint/backward contract report")
    print(f"repo: {REPO}")
    print("mode: report-only by default; pass --strict to exit 2 for known blockers")
    print("scope: no-CUDA static source/doc inspection; does not import torch or run Mojo")
    print("")

    for fact in report.facts:
        print(f"{fact.status} {fact.label}")
        print(f"  {fact.detail}")
        if fact.refs:
            shown = fact.refs[:ref_limit]
            suffix = "" if len(fact.refs) <= ref_limit else f" ... +{len(fact.refs) - ref_limit} more"
            print(f"  refs: {', '.join(shown)}{suffix}")
        print("")

    print("verdict")
    print(f"  production_ready_low_memory_offload_checkpoint_backward: {report.production_ready}")
    print(
        "  accepted_full_predict_backward_adamw_replay: "
        f"{report.accepted_full_predict_backward_adamw_replay}"
    )
    print(
        "  accepted_low_memory_offload_checkpoint_backward_replay: "
        f"{report.accepted_low_memory_offload_checkpoint_backward_replay}"
    )
    print("")
    print("known blockers")
    for blocker in report.known_blockers:
        print(f"  - {blocker}")
    print("")

    counts = status_counts(report.facts)
    print(
        "summary: "
        f"facts={len(report.facts)} "
        + " ".join(f"{key.lower()}={counts[key]}" for key in sorted(counts))
        + f" strict_blockers={len(report.strict_blockers)}"
    )


def json_report(report: ContractReport) -> dict[str, Any]:
    return {
        "producer": "scripts/check_klein_offload_checkpoint_contract.py",
        "scope": "no-CUDA static source/doc inspection",
        "repo": str(REPO),
        "production_ready_low_memory_offload_checkpoint_backward": report.production_ready,
        "accepted_full_predict_backward_adamw_replay": (
            report.accepted_full_predict_backward_adamw_replay
        ),
        "accepted_low_memory_offload_checkpoint_backward_replay": (
            report.accepted_low_memory_offload_checkpoint_backward_replay
        ),
        "missing_sources": list(report.missing_sources),
        "known_blockers": list(report.known_blockers),
        "strict_blockers": list(report.strict_blockers),
        "facts": [
            {
                "status": fact.status,
                "label": fact.label,
                "detail": fact.detail,
                "refs": list(fact.refs),
            }
            for fact in report.facts
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report Klein low-memory offload/checkpoint/backward production "
            "evidence from source/docs."
        )
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 while known Klein production blockers remain",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--ref-limit",
        type=int,
        default=5,
        help="maximum source refs printed per fact in text mode",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = gather_report()
    if args.json:
        print(json.dumps(json_report(report), indent=2, sort_keys=True))
    else:
        print_report(report, ref_limit=max(0, args.ref_limit))

    if args.strict:
        if report.strict_blockers:
            if not args.json:
                print("")
                print("STRICT FAIL")
                for blocker in report.strict_blockers:
                    print(f"  - {blocker}")
            return 2
        if not args.json:
            print("")
            print("STRICT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
