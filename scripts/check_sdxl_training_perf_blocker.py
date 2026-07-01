#!/usr/bin/env python3
"""Guard SDXL trainer-speed evidence while the scorecard row is blocked."""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TRAINER = Path("/home/alex/serenity-trainer/src/serenity_trainer/trainer/train_sdxl_real.mojo")
BLOCKER = REPO / "artifacts/training_perf/sdxl_scorecard_blocked_2026-06-30.md"
SCORECARD = REPO / "artifacts/training_perf/scorecard_coverage_2026-06-30.md"
MANIFEST = REPO / "artifacts/training_perf/benchmark_collection_2026-06-30.md"
MANIFEST_WRITER = REPO / "scripts/write_training_benchmark_collection_manifest.py"
SUMMARIZER = REPO / "scripts/summarize_training_perf.py"
SDXL_REAL = REPO / "serenitymojo/models/sdxl/sdxl_real_train.mojo"
SDXL_STACK = REPO / "serenitymojo/models/sdxl/sdxl_unet_stack_lora.mojo"
SDXL_LORA_BLOCK = REPO / "serenitymojo/models/sdxl/lora_block.mojo"

TRAIN_REF_PATHS = (
    Path("/home/alex/OneTrainer/output/sdxl_100step_baseline/step000_replay.safetensors"),
    Path("/home/alex/OneTrainer/output/sdxl_100step_baseline/step000_replay_manifest.json"),
    Path("/home/alex/onetrainer-mojo/parity/sdxl_train_ref_meta.json"),
    Path("/home/alex/onetrainer-mojo/parity/sdxl_train_ref_step000.safetensors"),
    Path("/home/alex/onetrainer-mojo/parity/sdxl_train_ref_step000_adapters.safetensors"),
)


def read(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"[sdxl-perf-blocker] missing required file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise SystemExit(f"[sdxl-perf-blocker] missing {label}: {needle!r}")


def require_absent(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise SystemExit(f"[sdxl-perf-blocker] unexpected {label}: {needle!r}")


def main() -> int:
    sdxl_jsonl = sorted((REPO / "artifacts/training_perf").glob("sdxl*.jsonl"))
    if sdxl_jsonl:
        raise SystemExit(
            "[sdxl-perf-blocker] SDXL perf JSONL exists while blocker is active: "
            + ", ".join(str(path) for path in sdxl_jsonl)
        )

    trainer = read(TRAINER)
    for needle in ("TrainingPerfRecord", "emit_training_perf_record", "[training-perf-json]"):
        require_absent(trainer, needle, "SDXL TrainingPerfRecord emission")

    blocker = read(BLOCKER)
    for needle in (
        "blocked-not-collected",
        "not numeric Mojo grad/update evidence",
        "not shared device-ABI evidence",
        "not production parity",
        "host-list LoRA grads",
        "DeviceGradSet",
        "step000_replay.safetensors",
        "sdxl_train_ref_step000.safetensors",
    ):
        require(blocker, needle, "blocker artifact text")

    real = read(SDXL_REAL)
    for needle in (
        "var d_a: List[List[List[Float32]]]",
        "SdxlRealGrads",
        "sdxl_real_backward",
    ):
        require(real, needle, "SDXL host-list real backward marker")

    stack = read(SDXL_STACK)
    for needle in (
        "SdxlStLoraGrads",
        "d_x.to_host(ctx)",
        "d_context.to_host(ctx)",
        "sdxl_lora_unsupported_onetrainer_targets",
    ):
        require(stack, needle, "SDXL stack host-list/blocker marker")

    lora_block = read(SDXL_LORA_BLOCK)
    for needle in (
        "to_host/from_host",
        "var x_h = x_in.to_host(ctx)",
        "to_host(ctx)",
        "Tensor.from_host",
    ):
        require(lora_block, needle, "SDXL LoRA host compatibility marker")

    missing_refs = [path for path in TRAIN_REF_PATHS if not path.exists()]
    if not missing_refs:
        raise SystemExit(
            "[sdxl-perf-blocker] SDXL replay/train-ref artifacts now exist; "
            "reassess blocker before keeping blocked-not-collected status"
        )

    for path in (SCORECARD, MANIFEST, MANIFEST_WRITER, SUMMARIZER):
        text = read(path)
        require(text, "blocked-not-collected", f"{path.name} SDXL blocked status")
        require(text, "sdxl_scorecard_blocked_2026-06-30.md", f"{path.name} blocker path")

    print("[sdxl-perf-blocker] PASS: SDXL scorecard remains blocked-not-collected")
    print("[sdxl-perf-blocker] missing refs:", ", ".join(str(path) for path in missing_refs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
