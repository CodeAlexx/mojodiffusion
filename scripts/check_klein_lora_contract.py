#!/usr/bin/env python3
"""Audit Klein/Flux2 LoRA target parity against OneTrainer.

OneTrainer's Flux2/Klein LoRA setup wraps separate Linear modules for the double
blocks. The old mojodiffusion Klein trainer used fused qkv/proj LoRA slots
instead. That path is useful implementation history, but it is not OneTrainer
parity and its speed/loss numbers must not be promoted as parity evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ONETRAINER = Path("/home/alex/OneTrainer")
PORT = Path("/home/alex/onetrainer-mojo")
TRAIN_REF_PREFIX = PORT / "parity/klein_train_ref_step000"

OT_FLUX2_MODEL = ONETRAINER / "modules/model/Flux2Model.py"
OT_FLUX2_SETUP = ONETRAINER / "modules/modelSetup/Flux2LoRASetup.py"
CORE_STACK = REPO / "serenitymojo/models/klein/klein_stack_lora.mojo"
CORE_DOUBLE = REPO / "serenitymojo/models/klein/double_block.mojo"
CORE_SINGLE = REPO / "serenitymojo/models/klein/single_block.mojo"
CORE_LORA_ADAPTER = REPO / "serenitymojo/models/klein/lora_adapter.mojo"
CORE_SR = REPO / "serenitymojo/util/bf16_stochastic_rounding.mojo"
CORE_TRAIN = REPO / "serenitymojo/training/train_klein_real.mojo"
CORE_LORA_SAVE = REPO / "serenitymojo/training/lora_save.mojo"
CORE_TRAIN_REF_GATE = REPO / "serenitymojo/models/klein/parity/klein_train_ref_artifact_smoke.mojo"
CORE_ADAMW_STATE_INIT_MOJO_SMOKE = (
    REPO / "serenitymojo/models/klein/parity/klein_lora_adamw_state_init_smoke.mojo"
)
CORE_TRAIN_REF_ADAMW_STATE_INIT_REPLAY = (
    REPO / "serenitymojo/models/klein/parity/klein_train_ref_adamw_state_init_replay.mojo"
)
CORE_ADAPTER_GRAD_UPDATE_GATE = REPO / "scripts/check_klein_adapter_grad_update_replay.py"
CORE_LOSS_REPLAY_GATE = REPO / "scripts/check_klein_loss_replay.py"
CORE_ADAMW_STATE_INIT_GATE = REPO / "scripts/check_klein_adamw_state_init_replay.py"
CORE_ADAMW_POSITIVE_LR_ORACLE = REPO / "scripts/check_klein_adamw_positive_lr_oracle.py"
PORT_STACK = PORT / "src/onetrainer_mojo/model/klein/klein_stack_lora.mojo"
PORT_DOUBLE = PORT / "src/onetrainer_mojo/model/klein/double_block.mojo"
PORT_SINGLE = PORT / "src/onetrainer_mojo/model/klein/single_block.mojo"
TRAIN_REF_STEP = TRAIN_REF_PREFIX.with_suffix(".safetensors")
TRAIN_REF_ADAPTERS = TRAIN_REF_PREFIX.with_name(TRAIN_REF_PREFIX.name + "_adapters.safetensors")
TRAIN_REF_META = PORT / "parity/klein_train_ref_meta.json"
TRAIN_REF_CONTRACT = PORT / "parity/klein_train_ref_contract.json"

DOUBLE_TARGETS = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.linear_in",
    "ff.linear_out",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff_context.linear_in",
    "ff_context.linear_out",
]
SINGLE_TARGETS = ["attn.to_qkv_mlp_proj", "attn.to_out"]
SAVE_SUFFIXES = ["alpha", "lora_down.weight", "lora_up.weight"]
TRAIN_REF_STEP_SHAPES: dict[str, tuple[str, list[int]]] = {
    "batch.latent_image": ("BF16", [1, 32, 64, 64]),
    "batch.loss_weight": ("F32", [1]),
    "batch.text_encoder_hidden_state": ("BF16", [1, 512, 12288]),
    "batch.tokens": ("I64", [1, 512]),
    "batch.tokens_mask": ("I64", [1, 512]),
    "output.loss_for_backward": ("F32", []),
    "output.loss_pre_scale": ("F32", []),
    "output.predicted": ("BF16", [1, 32, 64, 64]),
    "output.target": ("F32", [1, 32, 64, 64]),
    "output.timestep": ("I32", [1]),
    "trace.encoder_hidden_states": ("BF16", [1, 512, 12288]),
    "trace.flow": ("F32", [1, 128, 32, 32]),
    "trace.image_ids": ("I64", [1, 1024, 4]),
    "trace.latent_noise": ("F32", [1, 128, 32, 32]),
    "trace.packed_latent_input": ("BF16", [1, 1024, 128]),
    "trace.packed_predicted_flow": ("BF16", [1, 1024, 128]),
    "trace.predicted_flow": ("BF16", [1, 128, 32, 32]),
    "trace.scaled_latent_image": ("F32", [1, 128, 32, 32]),
    "trace.scaled_noisy_latent_image": ("F32", [1, 128, 32, 32]),
    "trace.sigma": ("F32", [1, 1, 1, 1]),
    "trace.text_ids": ("I64", [1, 512, 4]),
    "trace.transformer_timestep": ("F32", [1]),
}
TRAIN_REF_ADAPTER_SHAPES: dict[str, tuple[str, list[int]]] = {
    "adapter_before.transformer_blocks.0.attn.to_q.lora_down.weight": ("F32", [16, 4096]),
    "adapter_after.transformer_blocks.7.ff_context.linear_in.lora_up.weight": ("F32", [24576, 16]),
    "adapter_pre_clip_grad.transformer_blocks.0.attn.to_q.lora_up.weight": ("F32", [4096, 16]),
    "adapter_post_clip_grad.transformer_blocks.0.attn.to_q.lora_up.weight": ("F32", [4096, 16]),
    "adapter_after.single_transformer_blocks.23.attn.to_qkv_mlp_proj.lora_up.weight": ("F32", [36864, 16]),
    "adapter_after.single_transformer_blocks.23.attn.to_out.lora_down.weight": ("F32", [16, 16384]),
}
TRAIN_REF_ADAPTER_PHASES = {
    "adapter_after",
    "adapter_before",
    "adapter_post_clip",
    "adapter_post_clip_grad",
    "adapter_pre_clip",
    "adapter_pre_clip_grad",
}


def read(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def extract_dbl_slots(text: str, path: Path) -> int:
    m = re.search(r"comptime\s+DBL_SLOTS\s*=\s*(\d+)", text)
    if not m:
        raise SystemExit(f"missing DBL_SLOTS in {path}")
    return int(m.group(1))


def require_all(label: str, text: str, needles: list[str]) -> list[str]:
    return [needle for needle in needles if needle not in text]


def read_optional(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def build_call_has_ff_dim(text: str) -> bool:
    return "build_klein_lora_set(" in text and "cfg.d_model, cfg.mlp_hidden" in text


def single_builder_has_full_ff_dim(text: str) -> bool:
    m = re.search(r"var\s+sgl\s*=\s*List\[LoraAdapter\]\(\)(.*?)return\s+KleinLoraSet", text, re.S)
    if not m:
        return False
    block = m.group(1)
    return "3 * D + 2 * F" in block and "D + F, D" in block


def saved_key_double(block: int, module: str, suffix: str) -> str:
    return f"transformer.transformer_blocks.{block}.{module}.{suffix}"


def saved_key_single(block: int, module: str, suffix: str) -> str:
    return f"transformer.single_transformer_blocks.{block}.{module}.{suffix}"


def read_safetensors_header(path: Path) -> dict:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise SystemExit(f"safetensors too small: {path}")
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = raw[8 : 8 + header_len]
    return json.loads(header.decode("utf-8"))


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"[klein-lora-contract] missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_saved_header(path: Path, num_double: int, num_single: int) -> None:
    header = read_safetensors_header(path)
    keys = sorted(k for k in header if k != "__metadata__")
    expected_adapters = num_double * len(DOUBLE_TARGETS) + num_single * len(SINGLE_TARGETS)
    expected_tensors = expected_adapters * len(SAVE_SUFFIXES)
    if len(keys) != expected_tensors:
        raise SystemExit(
            f"[klein-lora-contract] saved tensor count mismatch: got {len(keys)} "
            f"expected {expected_tensors}"
        )

    for block in range(num_double):
        for module in DOUBLE_TARGETS:
            for suffix in SAVE_SUFFIXES:
                key = saved_key_double(block, module, suffix)
                if key not in header:
                    raise SystemExit(f"[klein-lora-contract] saved file missing {key}")
                info = header[key]
                if info.get("dtype") != "BF16":
                    raise SystemExit(f"[klein-lora-contract] {key} dtype {info.get('dtype')} != BF16")
                if suffix == "alpha" and info.get("shape") != []:
                    raise SystemExit(f"[klein-lora-contract] {key} alpha shape {info.get('shape')} != []")
                if suffix != "alpha" and len(info.get("shape", [])) != 2:
                    raise SystemExit(f"[klein-lora-contract] {key} shape is not rank-2: {info.get('shape')}")

    for block in range(num_single):
        for module in SINGLE_TARGETS:
            for suffix in SAVE_SUFFIXES:
                key = saved_key_single(block, module, suffix)
                if key not in header:
                    raise SystemExit(f"[klein-lora-contract] saved file missing {key}")
                info = header[key]
                if info.get("dtype") != "BF16":
                    raise SystemExit(f"[klein-lora-contract] {key} dtype {info.get('dtype')} != BF16")
                if suffix == "alpha" and info.get("shape") != []:
                    raise SystemExit(f"[klein-lora-contract] {key} alpha shape {info.get('shape')} != []")
                if suffix != "alpha" and len(info.get("shape", [])) != 2:
                    raise SystemExit(f"[klein-lora-contract] {key} shape is not rank-2: {info.get('shape')}")

    print(
        "[klein-lora-contract] saved inventory: PASS "
        f"double={num_double} single={num_single} adapters={expected_adapters} tensors={len(keys)}"
    )


def check_train_ref_step_header(path: Path) -> None:
    header = read_safetensors_header(path)
    keys = sorted(k for k in header if k != "__metadata__")
    if len(keys) != 22:
        raise SystemExit(f"[klein-lora-contract] train-ref step tensor count {len(keys)} != 22")
    meta = header.get("__metadata__", {})
    if meta.get("producer") != "scripts/klein_dump_train_ref.py" or meta.get("global_step") != "0":
        raise SystemExit(f"[klein-lora-contract] unexpected train-ref step metadata: {meta!r}")
    for key, (expected_dtype, expected_shape) in TRAIN_REF_STEP_SHAPES.items():
        if key not in header:
            raise SystemExit(f"[klein-lora-contract] train-ref step missing {key}")
        info = header[key]
        if info.get("dtype") != expected_dtype or info.get("shape") != expected_shape:
            raise SystemExit(
                f"[klein-lora-contract] {key} header mismatch: "
                f"dtype={info.get('dtype')!r} shape={info.get('shape')!r}, "
                f"expected dtype={expected_dtype!r} shape={expected_shape!r}"
            )
    print(f"[klein-lora-contract] train-ref step artifact: PASS file={path}")


def check_train_ref_adapter_header(path: Path) -> None:
    header = read_safetensors_header(path)
    keys = sorted(k for k in header if k != "__metadata__")
    if len(keys) != 1728:
        raise SystemExit(f"[klein-lora-contract] train-ref adapter tensor count {len(keys)} != 1728")
    meta = header.get("__metadata__", {})
    if meta.get("producer") != "scripts/klein_dump_train_ref.py" or meta.get("adapter_dump") != "step-with-grads":
        raise SystemExit(f"[klein-lora-contract] unexpected train-ref adapter metadata: {meta!r}")
    phases = {key.split(".", 1)[0] for key in keys}
    if phases != TRAIN_REF_ADAPTER_PHASES:
        raise SystemExit(f"[klein-lora-contract] train-ref adapter phases mismatch: {sorted(phases)}")
    for key, (expected_dtype, expected_shape) in TRAIN_REF_ADAPTER_SHAPES.items():
        if key not in header:
            raise SystemExit(f"[klein-lora-contract] train-ref adapters missing {key}")
        info = header[key]
        if info.get("dtype") != expected_dtype or info.get("shape") != expected_shape:
            raise SystemExit(
                f"[klein-lora-contract] {key} header mismatch: "
                f"dtype={info.get('dtype')!r} shape={info.get('shape')!r}, "
                f"expected dtype={expected_dtype!r} shape={expected_shape!r}"
            )
    print(f"[klein-lora-contract] train-ref adapter/grad artifact: PASS file={path}")


def check_train_ref_meta(path: Path) -> None:
    meta = load_json(path)
    if meta.get("producer") != "scripts/klein_dump_train_ref.py":
        raise SystemExit(f"[klein-lora-contract] unexpected train-ref producer: {meta.get('producer')!r}")
    runtime = meta.get("runtime_config", {})
    expected = {
        "model_type": "FLUX_2",
        "training_method": "LORA",
        "train_dtype": "BFLOAT_16",
        "optimizer": "ADAMW",
        "lora_rank": 16,
        "layer_filter": "blocks",
    }
    for key, expected_value in expected.items():
        if runtime.get(key) != expected_value:
            raise SystemExit(
                f"[klein-lora-contract] runtime_config.{key}={runtime.get(key)!r}, expected {expected_value!r}"
            )
    steps = meta.get("steps")
    if not isinstance(steps, list) or len(steps) != 1:
        raise SystemExit("[klein-lora-contract] train-ref meta must contain exactly one step")
    step = steps[0]
    if step.get("safetensors") != str(TRAIN_REF_STEP) or step.get("adapter_safetensors") != str(TRAIN_REF_ADAPTERS):
        raise SystemExit("[klein-lora-contract] train-ref meta paths do not match expected artifacts")
    if step.get("loss_for_backward") != 0.12243738770484924:
        raise SystemExit(f"[klein-lora-contract] unexpected Klein loss: {step.get('loss_for_backward')!r}")
    if step.get("grad_norm_pre_clip") != 0.005975008010864258:
        raise SystemExit(f"[klein-lora-contract] unexpected Klein grad norm: {step.get('grad_norm_pre_clip')!r}")
    print(
        "[klein-lora-contract] train-ref meta: PASS "
        f"loss={step.get('loss_for_backward')} grad_norm={step.get('grad_norm_pre_clip')}"
    )


def check_train_ref_contract(path: Path) -> None:
    contract = load_json(path)
    if contract.get("name") != "klein_train_ref":
        raise SystemExit(f"[klein-lora-contract] unexpected contract name: {contract.get('name')!r}")
    outputs = contract.get("default_outputs", {})
    for key in ("run_metadata", "step_tensors", "adapter_tensors"):
        if key not in outputs:
            raise SystemExit(f"[klein-lora-contract] train-ref contract missing default output {key}")
    print(f"[klein-lora-contract] train-ref contract: PASS file={path}")


def check_train_ref_mojo_gate(path: Path) -> None:
    text = read(path)
    for needle in (
        str(TRAIN_REF_META),
        str(TRAIN_REF_STEP),
        str(TRAIN_REF_ADAPTERS),
        "output.loss_for_backward",
        "adapter_pre_clip_grad.transformer_blocks.0.attn.to_q.lora_up.weight",
        "synthetic_adamw.double0.to_q.up[0]",
        "_adamw_synthetic_positive_lr_delta",
        "1728",
    ):
        if needle not in text:
            raise SystemExit(f"[klein-lora-contract] Mojo train-ref gate missing {needle}")
    print(f"[klein-lora-contract] Mojo train-ref artifact consumer: PASS file={path}")


def check_adamw_state_init_mojo_smoke(path: Path) -> None:
    text = read(path)
    for needle in (
        "Bounded Mojo AdamW zero-lr state-init smoke",
        "klein_lora_adamw_step",
        "build_klein_lora_set",
        "KleinLoraGrads",
        "synthetic 1-double/1-single LoRA set",
        "optimizer-path support evidence only",
        "does not consume OneTrainer train-ref tensors",
        "does not execute Klein predict/backward_lora",
        "does not prove full Mojo predict/backward/AdamW parity",
        "does not prove nonzero update parity",
        "does not prove low-memory offload/checkpoint backward parity",
        "positive_lr_changed",
    ):
        if needle not in text:
            raise SystemExit(f"[klein-lora-contract] AdamW Mojo state-init smoke missing {needle}")
    print(f"[klein-lora-contract] bounded Mojo AdamW state-init smoke: PASS file={path}")


def check_train_ref_adamw_state_init_replay(path: Path) -> None:
    text = read(path)
    for needle in (
        "real OneTrainer adapter dump",
        "klein_train_ref_step000_adapters.safetensors",
        "all 288 train-ref adapters",
        "adapter_post_clip",
        "adapter_post_clip_grad",
        "klein_lora_adamw_step",
        "Float32(0.0)",
        "does not execute Klein predict/backward_lora",
        "does not compare optimizer moment tensors against OneTrainer payloads",
        "does not prove full Mojo predict/backward/AdamW parity",
        "does not prove nonzero OneTrainer update parity",
        "does not prove low-memory offload/checkpoint backward parity",
        "synthetic_positive_lr_changed",
    ):
        if needle not in text:
            raise SystemExit(f"[klein-lora-contract] train-ref Mojo AdamW replay missing {needle}")
    print(f"[klein-lora-contract] train-ref Mojo AdamW state-init replay: PASS file={path}")


def check_adapter_grad_update_gate(path: Path) -> None:
    text = read(path)
    for needle in (
        "Klein/Flux2 OneTrainer adapter gradient/update oracle gate",
        "adapter_pre_clip_grad",
        "adapter_post_clip_grad",
        "has_zero_lr_state_init_oracle",
        "--require-update-bearing",
        "--require-synthetic-positive-lr",
        "has_synthetic_positive_lr_adamw_update",
        "--require-mojo-parity",
        "has_mojo_backward_adamw_parity",
        "mojo_parity_blockers",
        "bounded optimizer-math replay",
        "missing Mojo Klein backward/AdamW consumer",
        "klein_train_ref_adamw_state_init_replay.mojo",
        "not Mojo backward or AdamW parity",
    ):
        if needle not in text:
            raise SystemExit(f"[klein-lora-contract] adapter grad/update oracle gate missing {needle}")
    print(f"[klein-lora-contract] adapter grad/update oracle gate: PASS file={path}")


def check_loss_replay_gate(path: Path, require_artifacts: bool) -> None:
    text = read(path)
    for needle in (
        "Klein/Flux2 OneTrainer loss and d_loss replay guard",
        "has_loss_dloss_replay",
        "--require-loss-replay",
        "Float64 sum((predicted-target)^2)/N",
        "Float32 (2/N)*(predicted-target)",
        "not transformer forward, backward, AdamW, or sampler parity",
    ):
        if needle not in text:
            raise SystemExit(f"[klein-lora-contract] loss replay gate missing {needle}")
    if require_artifacts:
        result = subprocess.run(
            [sys.executable, str(path), "--require-loss-replay"],
            cwd=str(REPO),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
            check=False,
        )
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        if result.returncode != 0:
            raise SystemExit(
                f"[klein-lora-contract] loss replay gate failed with exit {result.returncode}"
            )
    print(f"[klein-lora-contract] loss/d_loss replay gate: PASS file={path}")


def check_adamw_state_init_gate(path: Path, require_artifacts: bool) -> None:
    text = read(path)
    for needle in (
        "Klein/Flux2 AdamW zero-lr state-init replay guard",
        "has_zero_lr_state_init_replay",
        "--require-state-init",
        "adapter_post_clip_grad",
        "post_clip_bf16_import_minus_post_clip_f32",
        "state_exp_avg_bf16_projected",
        "state_exp_avg_sq_bf16_projected",
        "does not execute Mojo backward/AdamW",
        "does not prove nonzero update parity",
    ):
        if needle not in text:
            raise SystemExit(f"[klein-lora-contract] AdamW state-init gate missing {needle}")
    if require_artifacts:
        result = subprocess.run(
            [sys.executable, str(path), "--require-state-init"],
            cwd=str(REPO),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
            check=False,
        )
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        if result.returncode != 0:
            raise SystemExit(
                f"[klein-lora-contract] AdamW state-init gate failed with exit {result.returncode}"
            )
    print(f"[klein-lora-contract] AdamW state-init replay gate: PASS file={path}")


def check_adamw_positive_lr_oracle(path: Path, require_artifacts: bool) -> None:
    text = read(path)
    for needle in (
        "Klein/Flux2 CPU host-list positive-lr AdamW support oracle",
        "adapter_post_clip_grad",
        "BF16-project moments",
        "deterministic stochastic BF16 rounding",
        "DEFAULT_EXPECT_CHANGED = 27_262_275",
        "has_optimizer_only_positive_lr_oracle",
        "does not execute Klein predict -> backward_lora",
        "does not compare OneTrainer optimizer moment tensor payloads",
        "does not prove CUDA/GPU parity",
        "does not use a real OneTrainer lr_before>0 adapter_after update",
        "--strict",
    ):
        if needle not in text:
            raise SystemExit(f"[klein-lora-contract] AdamW positive-lr oracle missing {needle}")
    if require_artifacts:
        result = subprocess.run(
            [sys.executable, str(path), "--strict"],
            cwd=str(REPO),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
            check=False,
        )
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        if result.returncode != 0:
            raise SystemExit(
                f"[klein-lora-contract] AdamW positive-lr oracle failed with exit {result.returncode}"
            )
    print(f"[klein-lora-contract] CPU AdamW positive-lr support oracle: PASS file={path}")


def check_train_ref_artifacts(require_artifacts: bool) -> None:
    missing = [
        path
        for path in (
            TRAIN_REF_STEP,
            TRAIN_REF_ADAPTERS,
            TRAIN_REF_META,
            TRAIN_REF_CONTRACT,
            CORE_TRAIN_REF_GATE,
            CORE_ADAMW_STATE_INIT_MOJO_SMOKE,
            CORE_TRAIN_REF_ADAMW_STATE_INIT_REPLAY,
            CORE_ADAPTER_GRAD_UPDATE_GATE,
            CORE_LOSS_REPLAY_GATE,
            CORE_ADAMW_STATE_INIT_GATE,
            CORE_ADAMW_POSITIVE_LR_ORACLE,
        )
        if not path.exists()
    ]
    if missing:
        for path in missing:
            print(f"[klein-lora-contract] MISSING train-ref artifact: {path}")
        if require_artifacts:
            raise SystemExit(
                "[klein-lora-contract] required train-ref artifact(s) missing: "
                + ", ".join(str(path) for path in missing)
            )
        return
    check_train_ref_step_header(TRAIN_REF_STEP)
    check_train_ref_adapter_header(TRAIN_REF_ADAPTERS)
    check_train_ref_meta(TRAIN_REF_META)
    check_train_ref_contract(TRAIN_REF_CONTRACT)
    check_train_ref_mojo_gate(CORE_TRAIN_REF_GATE)
    check_adamw_state_init_mojo_smoke(CORE_ADAMW_STATE_INIT_MOJO_SMOKE)
    check_train_ref_adamw_state_init_replay(CORE_TRAIN_REF_ADAMW_STATE_INIT_REPLAY)
    check_adapter_grad_update_gate(CORE_ADAPTER_GRAD_UPDATE_GATE)
    check_loss_replay_gate(CORE_LOSS_REPLAY_GATE, require_artifacts)
    check_adamw_state_init_gate(CORE_ADAMW_STATE_INIT_GATE, require_artifacts)
    check_adamw_positive_lr_oracle(CORE_ADAMW_POSITIVE_LR_ORACLE, require_artifacts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-core",
        action="store_true",
        help="fail unless mojodiffusion core Klein LoRA has the OneTrainer split-slot contract",
    )
    parser.add_argument(
        "--safetensors",
        type=Path,
        help="optional saved Klein LoRA safetensors header to validate",
    )
    parser.add_argument(
        "--require-train-ref-artifacts",
        action="store_true",
        help="fail if the local Klein train-ref safetensors/meta/contract or Mojo consumer is absent",
    )
    parser.add_argument("--num-double", type=int, default=8)
    parser.add_argument("--num-single", type=int, default=24)
    args = parser.parse_args()

    ot_model = read(OT_FLUX2_MODEL)
    ot_setup = read(OT_FLUX2_SETUP)
    core = read(CORE_STACK)
    port = read(PORT_STACK)
    core_double = read_optional(CORE_DOUBLE)
    core_single = read_optional(CORE_SINGLE)
    core_train = read_optional(CORE_TRAIN)
    core_lora_save = read_optional(CORE_LORA_SAVE)
    port_double = read(PORT_DOUBLE)
    port_single = read(PORT_SINGLE)

    missing_model = require_all("OneTrainer Flux2Model", ot_model, DOUBLE_TARGETS + SINGLE_TARGETS)
    missing_setup = require_all("OneTrainer Flux2LoRASetup", ot_setup, ["LoRAModuleWrapper"])
    if missing_model or missing_setup:
        print("[klein-lora-contract] FAIL: OneTrainer source scan did not find expected targets")
        if missing_model:
            print("  missing in Flux2Model.py:", ", ".join(missing_model))
        if missing_setup:
            print("  missing in Flux2LoRASetup.py:", ", ".join(missing_setup))
        return 1

    core_slots = extract_dbl_slots(core, CORE_STACK)
    port_slots = extract_dbl_slots(port, PORT_STACK)
    expected_dbl_slots = len(DOUBLE_TARGETS)
    expected_single_slots = len(SINGLE_TARGETS)
    expected_adapters = 8 * expected_dbl_slots + 24 * expected_single_slots

    print("[klein-lora-contract] OneTrainer double targets:", expected_dbl_slots)
    print("[klein-lora-contract] OneTrainer single targets:", expected_single_slots)
    print("[klein-lora-contract] expected Klein 9B adapters:", expected_adapters)
    print("[klein-lora-contract] port mirror DBL_SLOTS:", port_slots)
    print("[klein-lora-contract] mojodiffusion core DBL_SLOTS:", core_slots)

    if port_slots != expected_dbl_slots:
        print("[klein-lora-contract] FAIL: OneTrainer-named port mirror is not split-slot parity")
        return 1

    port_missing = []
    port_missing.extend(require_all(
        "port double block",
        port_double,
        ["q_d_a", "k_d_a", "v_d_a", "out_d_a", "ff_in_d_a", "ff_out_d_a"],
    ))
    port_missing.extend(require_all(
        "port single block",
        port_single,
        ["3 * D + 2 * F", "qkv_d_a", "out_d_a"],
    ))
    port_missing.extend(require_all(
        "port stack",
        port,
        ["SGL_SAVE_TAIL = 0", "D: Int, F: Int, rank: Int", "3 * D + 2 * F"],
    ))
    if port_missing:
        print("[klein-lora-contract] FAIL: OneTrainer-named port mirror is missing split-slot markers")
        print("  missing:", ", ".join(sorted(set(port_missing))))
        return 1

    core_blockers: list[str] = []

    if core_slots != expected_dbl_slots:
        core_blockers.append(
            "core stack still uses legacy fused DBL_SLOTS="
            f"{core_slots}; OneTrainer requires {expected_dbl_slots}"
        )
    if "SGL_SAVE_TAIL = 0" not in core:
        core_blockers.append("core stack still saves the old single-block tail instead of all OneTrainer single blocks")
    if "D: Int, F: Int, rank: Int" not in core:
        core_blockers.append("core build_klein_lora_set still lacks the F/mlp_hidden argument")
    if not single_builder_has_full_ff_dim(core):
        core_blockers.append("core single-block adapters still lack the OneTrainer F/mlp_hidden dimensions")
    if not build_call_has_ff_dim(core_train):
        core_blockers.append("train_klein_real.mojo still calls build_klein_lora_set without cfg.mlp_hidden/F")
    missing_double = require_all(
        "core double block",
        core_double,
        ["q_d_a", "k_d_a", "v_d_a", "out_d_a", "ff_in_d_a", "ff_out_d_a"],
    )
    if missing_double:
        core_blockers.append(
            "core double_block still has fused qkv/proj LoRA grads, missing "
            + ", ".join(sorted(set(missing_double)))
        )
    missing_single = require_all(
        "core single block",
        core_single,
        ["3 * D + 2 * F", "qkv_d_a", "out_d_a"],
    )
    if missing_single:
        core_blockers.append(
            "core single_block is not full OneTrainer fused-qkv-mlp coverage, missing "
            + ", ".join(sorted(set(missing_single)))
        )
    if "klein_lora_fwd_qkv_rows_device_resident_scratch" in core_single:
        core_blockers.append("core single_block still calls the old qkv-row-only LoRA scratch helper")
    if "saved.att_flat[], lora.out" in core_single:
        core_blockers.append("core single_block still uses attention-only input for the OneTrainer to_out LoRA")
    if not CORE_LORA_ADAPTER.exists():
        core_blockers.append("core lacks serenitymojo/models/klein/lora_adapter.mojo")
    if not CORE_SR.exists():
        core_blockers.append("core lacks serenitymojo/util/bf16_stochastic_rounding.mojo")
    if "stochastic" not in core and "stochastic" not in read_optional(CORE_LORA_ADAPTER):
        core_blockers.append("core LoRA AdamW is not the OneTrainer stochastic-BF16 rounding helper")
    if "save_lora_onetrainer" not in core:
        core_blockers.append("core save_klein_lora is not using the OneTrainer raw LoRA saver")
    if ".lora_down.weight" not in core_lora_save or ".lora_up.weight" not in core_lora_save:
        core_blockers.append("shared LoRA saver does not emit raw OneTrainer lora_down/lora_up tensors")
    if ".alpha" not in core_lora_save or "_bf16_scalar" not in core_lora_save:
        core_blockers.append("shared LoRA saver does not emit BF16 OneTrainer alpha tensors")
    if ".lora_down.weight" not in core_lora_save or "adapter_scale = alpha_h[0] / Float32(rank)" not in core_lora_save:
        core_blockers.append("shared LoRA resume loader cannot read raw OneTrainer alpha/down/up files")

    if core_blockers:
        print("[klein-lora-contract] WARN: mojodiffusion core Klein trainer is not OneTrainer parity")
        for blocker in core_blockers:
            print("  -", blocker)
        return 1 if args.strict_core else 0

    if args.safetensors is not None:
        check_saved_header(args.safetensors, args.num_double, args.num_single)

    check_train_ref_artifacts(args.require_train_ref_artifacts)

    print("[klein-lora-contract] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
