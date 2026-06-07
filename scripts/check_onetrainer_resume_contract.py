#!/usr/bin/env python3
"""Static guard for the OneTrainer checkpoint/resume contract.

This checks the Mojo metadata contract against source markers in OneTrainer. It
is not a training resume parity test.
"""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
OT = Path("/home/alex/OneTrainer")
CONTRACT = ROOT / "serenitymojo/training/onetrainer_resume_contract.mojo"
PRODUCT = ROOT / "serenitymojo/training/onetrainer_product_run.mojo"
SMOKE = ROOT / "serenitymojo/training/onetrainer_resume_contract_smoke.mojo"
PRODUCT_SMOKE = ROOT / "serenitymojo/training/onetrainer_product_run_smoke.mojo"


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR missing file: {path}")
        raise


def require(label: str, haystack: str, needle: str) -> None:
    if needle not in haystack:
        raise AssertionError(f"{label}: missing marker {needle!r}")


def require_regex(label: str, haystack: str, pattern: str) -> None:
    if re.search(pattern, haystack, re.S) is None:
        raise AssertionError(f"{label}: missing pattern {pattern!r}")


def main() -> int:
    contract = read(CONTRACT)
    product = read(PRODUCT)
    smoke = read(SMOKE)
    product_smoke = read(PRODUCT_SMOKE)
    generic = read(OT / "modules/trainer/GenericTrainer.py")
    progress = read(OT / "modules/util/TrainProgress.py")
    internal_saver = read(OT / "modules/modelSaver/mixin/InternalModelSaverMixin.py")
    internal_loader = read(OT / "modules/modelLoader/mixin/InternalModelLoaderMixin.py")
    generic_lora_saver = read(OT / "modules/modelSaver/GenericLoRAModelSaver.py")
    generic_lora_loader = read(OT / "modules/modelLoader/GenericLoRAModelLoader.py")
    lora_loader = read(OT / "modules/modelLoader/mixin/LoRALoaderMixin.py")

    require("progress filename", progress, 'return f"{self.global_step}-{self.epoch}-{self.epoch_step}"')
    require("step save name", generic, "-save-")
    require("backup name", generic, "-backup-")
    require("training sample name", generic, "-training-sample-")
    require("save folder", generic, '"save"')
    require("backup folder", generic, '"backup"')
    require("sample queued before save command", generic, "self.__enqueue_sample_during_training")
    require("sample executes before backup/save", generic, "self.__execute_sample_during_training()")
    require("backup before save final", generic, "self.config.backup_before_save")

    require("internal meta save", internal_saver, '"meta.json"')
    require("internal optimizer save", internal_saver, '"optimizer", "optimizer.pt"')
    require("internal progress epoch", internal_saver, "'epoch': model.train_progress.epoch")
    require("internal progress epoch step", internal_saver, "'epoch_step': model.train_progress.epoch_step")
    require("internal progress global step", internal_saver, "'global_step': model.train_progress.global_step")
    require("internal param mapping", internal_saver, '"param_group_mapping"')
    require("internal optimizer mapping", internal_saver, '"param_group_optimizer_mapping"')

    require("internal meta load", internal_loader, '"meta.json"')
    require("internal optimizer load", internal_loader, '"optimizer", "optimizer.pt"')
    require("lora internal path", lora_loader, '"lora", "lora.safetensors"')
    require("lora model saver saves internal data", generic_lora_saver, "self._save_internal_data(model, output_model_destination)")
    require("lora loader loads internal data", generic_lora_loader, "self._load_internal_data(model, model_names.lora)")

    for marker in [
        "ot_step_save_path",
        "ot_backup_path",
        "ot_training_sample_leaf",
        "ot_save_before_sample",
        "ot_save_runs_before_sample",
        "ot_lora_resume_requires_optimizer_state",
        "ot_full_finetune_resume_expects_state_sidecars",
        "has_param_group_mapping",
        "has_param_group_optimizer_mapping",
        "full-finetune resume is unsupported",
        "full-finetune optimizer state missing param_group_mapping",
        "full-finetune optimizer state missing param_group_optimizer_mapping",
    ]:
        require("Mojo contract", contract, marker)

    require_regex("save-before-sample contract", contract, r"def ot_save_runs_before_sample\(\).*?return False")
    require("smoke validates missing optimizer", smoke, "missing_optimizer.has_optimizer_state = False")
    require("smoke validates raw LoRA rejection", smoke, "OT_RESUME_SURFACE_RAW_LORA")
    require("smoke validates full FT fail loud", smoke, "full-finetune loop not wired")
    require("smoke validates Klein resume alias", smoke, 'String("klein")')
    require("smoke validates FLUX_2 resume alias", smoke, 'String("FLUX_2")')
    require(
        "smoke validates full FT missing mapping",
        smoke,
        "full_ft_missing_mapping.has_param_group_mapping = False",
    )
    for marker in [
        "continue_last_backup",
        "_latest_backup_path",
        "OTProductResumePreflight",
        "validate_onetrainer_product_resume_preflight",
        "ot_internal_optimizer_state_path",
    ]:
        require("product resume preflight", product, marker)
    require(
        "product smoke validates missing optimizer",
        product_smoke,
        "missing optimizer state for LoRA resume",
    )
    require(
        "product smoke validates latest backup",
        product_smoke,
        "latest backup selected",
    )

    forbidden = ["STDtype.F32", "DType.float32", "to_f32", "Float32Tensor"]
    for marker in forbidden:
        if marker in contract or marker in smoke or marker in product:
            raise AssertionError(f"dtype boundary guard: forbidden marker {marker!r}")

    print("OneTrainer resume contract static guard PASS")
    print("  checked source markers: OneTrainer trainer/load/save internals")
    print("  evidence type: contract/static only, not end-to-end resume parity")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"OneTrainer resume contract static guard FAIL: {exc}")
        raise SystemExit(1)
