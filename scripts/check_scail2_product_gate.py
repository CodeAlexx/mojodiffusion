#!/usr/bin/env python3
"""Regenerate the machine-local SCAIL-2 product admission report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


SOURCE_COMMIT = "5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5"
MODEL_REVISION = "150cc0ca4e98e50e60b9295dacde39442fdccab2"
CHECKPOINT_SHA256 = "d6c73e94c57eb36e6351c800d1228e41ed7e45db1ccf410dd875bcfdd2945e7f"
LATENT_SHA256 = "7281f726de405cc5ef7931db45a9541e86a782537911c9cc9c735fc14286959b"
VIDEO_SHA256 = "af292cfe7c8c18a66e3348f57797c21350257c35bf0528533b2ff6803a20234c"
ONE_STEP_SHA256 = "e959bd00f99160b4db03656870558390de33e0bf4e6ce57761dcf20045436651"
SERENITY_HOME = Path(os.environ.get("SERENITY_HOME", Path.home() / ".serenity"))
EVIDENCE_ROOT = Path(
    os.environ.get(
        "SCAIL2_EVIDENCE_ROOT",
        SERENITY_HOME / "runs/scail2-gates/animation-regression",
    )
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-inspection-passed", action="store_true")
    parser.add_argument(
        "--latent",
        type=Path,
        default=EVIDENCE_ROOT / "latent_animation_40step.safetensors",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=EVIDENCE_ROOT / "decode/scail2_animation.mp4",
    )
    parser.add_argument(
        "--one-step",
        type=Path,
        default=EVIDENCE_ROOT / "latent_animation_compact_1step.safetensors",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/checks/scail2/product_gate.json")
    )
    args = parser.parse_args()

    binaries = {
        "stage_runner_sha256": Path("output/bin/scail2_stage_inputs"),
        "prompt_runner_sha256": Path("output/bin/scail2_encode_prompt"),
        "clip_runner_sha256": Path("output/bin/scail2_encode_clip"),
        "vae_runner_sha256": Path("output/bin/scail2_encode_vae"),
        "cache_runner_sha256": Path("output/bin/scail2_prepare_fp8_cache"),
        "denoise_runner_sha256": Path("output/bin/scail2_animation"),
        "decode_runner_sha256": Path("output/bin/scail2_decode"),
    }
    for path in binaries.values():
        require(path.is_file() and os.access(path, os.X_OK), f"missing executable: {path}")

    require(sha256(args.latent) == LATENT_SHA256, "accepted 40-step latent digest mismatch")
    require(sha256(args.video) == VIDEO_SHA256, "accepted MP4 digest mismatch")
    require(sha256(args.one_step) == ONE_STEP_SHA256, "current binary one-step digest mismatch")

    run = load_json(Path(f"{args.latent}.scail2_run.json"))
    result = load_json(Path(f"{args.video}.scail2_result.json"))
    require(run.get("schema") == "serenity.scail2.animation.v1", "invalid run manifest")
    require(run.get("mode") == "animation", "product run must use animation mode")
    require(
        run.get("additional_reference_latent") in (None, "-"),
        "base product run must not carry additional references",
    )
    require(run.get("source_commit") == SOURCE_COMMIT, "run source commit mismatch")
    require(run.get("model_revision") == MODEL_REVISION, "run model revision mismatch")
    require(run.get("checkpoint_sha256") == CHECKPOINT_SHA256, "run checkpoint mismatch")
    require(
        [run.get(k) for k in ("width", "height", "frames", "steps", "cfg")]
        == [896, 512, 65, 40, 5.0],
        "run profile mismatch",
    )
    require(result.get("schema") == "serenity.scail2.video_result.v1", "invalid result manifest")
    require(
        [result.get(k) for k in ("width", "height", "frames", "fps")]
        == [896, 512, 65, 16],
        "result profile mismatch",
    )

    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-count_frames", "-show_entries",
                "stream=codec_type,codec_name,width,height,nb_read_frames,avg_frame_rate",
                "-of", "json", str(args.video),
            ],
            text=True,
        )
    )
    video_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    require(len(video_streams) == 1, "accepted MP4 must have one video stream")
    stream = video_streams[0]
    require(
        stream.get("codec_name") == "h264"
        and stream.get("width") == 896
        and stream.get("height") == 512
        and stream.get("nb_read_frames") == "65"
        and stream.get("avg_frame_rate") == "16/1",
        "accepted MP4 probe mismatch",
    )
    require(args.visual_inspection_passed, "visual inspection acknowledgement is required")

    model_root = Path(os.environ.get("SERENITY_MODEL_ROOT", SERENITY_HOME / "models"))
    cache = model_root / "checkpoints/SCAIL-2-Mojo/transformer_fp8"
    provenance = (cache / "source_provenance.sha256").read_text(encoding="utf-8")
    require(f"source_commit={SOURCE_COMMIT}" in provenance, "cache source mismatch")
    require(f"model_revision={MODEL_REVISION}" in provenance, "cache revision mismatch")
    require(f"checkpoint_sha256={CHECKPOINT_SHA256}" in provenance, "cache checkpoint mismatch")
    require((cache / "shared.safetensors.sha256").is_file(), "shared cache checksum missing")
    require(
        all((cache / f"block_{index:02}.safetensors.sha256").is_file() for index in range(40)),
        "block cache checksum missing",
    )

    report = {
        "schema": "serenity.scail2.product_gate.v1",
        "passed": True,
        "pins": {
            "source_commit": SOURCE_COMMIT,
            "model_revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            **{key: sha256(path) for key, path in binaries.items()},
        },
        "profile": {
            "width": 896,
            "height": 512,
            "frames": 65,
            "fps": 16,
            "steps": 40,
            "guidance": 5.0,
            "seed": 42,
            "scheduler": "creator_unipc_order2_shift3",
        },
        "evidence": {
            "full_latent": str(args.latent),
            "full_latent_sha256": LATENT_SHA256,
            "full_video": str(args.video),
            "full_video_sha256": VIDEO_SHA256,
            "current_binary_one_step": str(args.one_step),
            "current_binary_one_step_sha256": ONE_STEP_SHA256,
            "visual_inspection_passed": True,
            "probe": stream,
        },
        "performance": {
            "requires_isolated_process_stages": True,
            "full_40_step_wall_seconds": 1528.62,
            "decode_wall_seconds": 29.69,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"GATE PASS {args.output}")


if __name__ == "__main__":
    main()
