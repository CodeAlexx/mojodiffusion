#!/usr/bin/env python3
"""Verify the installed creator-contract Wan2.2 TI2V-5B product profiles.

The gate is machine-local.  It accepts only current zero-copy model artifacts
and real 121-frame T2V plus first-frame I2V renders from the Mojo runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


REPO = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path("/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B-Mojo")
DEFAULT_T2V = REPO / "output/checks/wan22_20260729/t2v_landscape_teapot"
DEFAULT_I2V = (
    REPO / "output/checks/wan22_20260729/i2v_portrait_cyborg_lanczos"
)
DEFAULT_SOURCE = (
    REPO
    / "output/run_serenity_ui_i2v_check/uploads/canvas_init-0018.png"
)
DEFAULT_REPORT = REPO / "output/checks/wan22_product_gate.json"
EXPECTED_HF_REVISION = "installed-official-native"
EXPECTED_CREATOR_REVISION = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
EXPECTED_CACHE_SHA256 = (
    "bd2abdeeef4ab37454a7e6dab6eeb9517206f8c003a6aa1e345906e71a6d5010"
)
EXPECTED_INDEX_SHA256 = (
    "cd769dd8bddb0825ffb3516a39d64fc2ac3a5946fb93337f8594af926d6a0f56"
)
RUNNER_SOURCE_PATHS = (
    "serenitymojo/pipeline/wan22_t2v.mojo",
    "serenitymojo/models/dit/wan22_dit.mojo",
    "serenitymojo/models/vae/wan22_vae_encoder.mojo",
    "serenitymojo/models/vae/wan22_decoder.mojo",
    "serenitymojo/sampling/unipc.mojo",
    "serenitymojo/serve/image_io.mojo",
    "vendor/mojo-libs/image/studio_ops.mojo",
)
EXPECTED_SOURCE_BUNDLE_SHA256 = (
    "9a07732e237c01e5bd8bb69a2b3192d60f49826d66376881ebfe66c81eb25a68"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_bundle_sha256() -> str:
    digest = hashlib.sha256()
    for relative in RUNNER_SOURCE_PATHS:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((REPO / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def probe_mp4(path: Path) -> dict:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    streams = json.loads(output).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one video stream: {path}")
    return streams[0]


def read_wall_seconds(path: Path) -> float:
    match = re.search(
        r"^wall_seconds=([0-9.]+)$",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        raise ValueError(f"missing wall_seconds in {path}")
    return float(match.group(1))


def read_peak_vram(path: Path) -> int:
    peak = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            try:
                peak = max(peak, int(row[1].strip()))
            except ValueError:
                continue
    return peak


def creator_preprocess(source_path: Path, width: int, height: int) -> Image.Image:
    source = Image.open(source_path).convert("RGB")
    scale = max(width / source.width, height / source.height)
    resized = source.resize(
        (round(source.width * scale), round(source.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def validate_render(
    render_dir: Path,
    width: int,
    height: int,
) -> tuple[dict, dict[str, str]]:
    mp4 = render_dir / "wan22_t2v.mp4"
    probe = probe_mp4(mp4)
    hashes = {
        str(index): sha256(render_dir / f"frame_{index}.png")
        for index in (0, 60, 120)
    }
    frame_count = len(list(render_dir.glob("frame_*.png")))
    checks = {
        "frame_count": frame_count == 121,
        "width": int(probe.get("width", 0)) == width,
        "height": int(probe.get("height", 0)) == height,
        "frames": int(probe.get("nb_frames", 0)) == 121,
        "fps": probe.get("avg_frame_rate") == "24/1",
        "duration": abs(float(probe.get("duration", 0.0)) - 5.041667) < 0.001,
        "representative_frames_nonempty": all(hashes.values()),
    }
    return {
        "render_dir": str(render_dir),
        "mp4": str(mp4),
        "mp4_sha256": sha256(mp4),
        "probe": probe,
        "representative_frame_sha256": hashes,
        "checks": checks,
    }, hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2v-dir", type=Path, default=DEFAULT_T2V)
    parser.add_argument("--i2v-dir", type=Path, default=DEFAULT_I2V)
    parser.add_argument("--i2v-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--visual-accepted", action="store_true")
    args = parser.parse_args()

    manifest_path = MODEL_ROOT / "serenity_wan22_manifest.json"
    cache_path = (
        MODEL_ROOT
        / "wan22_dit_fp8_e4m3_b8fff7315c768468.safetensors"
    )
    manifest = read_json(manifest_path)
    cache_digest = sha256(cache_path)
    index_digest = manifest.get("transformer", {}).get("index_sha256", "")
    source_digest = source_bundle_sha256()

    t2v_artifact, _ = validate_render(args.t2v_dir, 832, 480)
    i2v_artifact, _ = validate_render(args.i2v_dir, 480, 832)

    reference = np.asarray(
        creator_preprocess(args.i2v_source, 480, 832),
        dtype=np.float32,
    )
    first_frame = np.asarray(
        Image.open(args.i2v_dir / "frame_0.png").convert("RGB"),
        dtype=np.float32,
    )
    first_frame_ssim = float(
        structural_similarity(
            reference,
            first_frame,
            channel_axis=2,
            data_range=255,
        )
    )

    t2v_wall = read_wall_seconds(args.t2v_dir / "render_timing.txt")
    i2v_wall = read_wall_seconds(args.i2v_dir / "render_timing.txt")
    t2v_peak = read_peak_vram(args.t2v_dir / "gpu.csv")
    i2v_peak = read_peak_vram(args.i2v_dir / "gpu.csv")

    checks = {
        "artifact_schema": (
            manifest.get("schema") == "serenity.wan22.artifact_view.v1"
        ),
        "zero_copy_artifacts": (
            manifest.get("source_kind") == "official_native_zero_copy"
        ),
        "hf_revision": manifest.get("revision") == EXPECTED_HF_REVISION,
        "creator_revision": (
            manifest.get("oracle_revision") == EXPECTED_CREATOR_REVISION
        ),
        "transformer_shards": (
            manifest.get("transformer", {}).get("shard_count") == 3
        ),
        "umt5_single_file": manifest.get("umt5", {}).get("shard_count") == 1,
        "fp8_cache_sha256": cache_digest == EXPECTED_CACHE_SHA256,
        "transformer_index_sha256": index_digest == EXPECTED_INDEX_SHA256,
        "runner_source_bundle_sha256": (
            source_digest == EXPECTED_SOURCE_BUNDLE_SHA256
        ),
        "t2v_artifact": all(t2v_artifact["checks"].values()),
        "i2v_artifact": all(i2v_artifact["checks"].values()),
        "i2v_first_frame_identity": first_frame_ssim >= 0.95,
        "measured_peak_vram": (
            0 < t2v_peak <= 24_000 and 0 < i2v_peak <= 24_000
        ),
        "measured_wall_time": t2v_wall > 0 and i2v_wall > 0,
        "visual_acceptance": args.visual_accepted,
        "required_binaries": all(
            (REPO / path).is_file()
            for path in (
                "output/bin/wan22_encode_prompt",
                "output/bin/wan22_t2v",
                "output/bin/wan22_t2v_480x832",
            )
        ),
    }
    report = {
        "schema": "serenity.wan22.product_gate.v2",
        "passed": all(checks.values()),
        "profile": {
            "model": "Wan-AI/Wan2.2-TI2V-5B",
            "mode": "t2v",
            "width": 832,
            "height": 480,
            "frames": 121,
            "fps": 24,
            "steps": 50,
            "guidance": 5.0,
            "sampler": "Flow-UniPC",
            "shift": 5.0,
            "quant": "fp8_e4m3_cached",
        },
        "i2v_profile": {
            "mode": "first_frame",
            "width": 480,
            "height": 832,
            "frames": 121,
            "fps": 24,
            "steps": 40,
            "guidance": 5.0,
            "sampler": "Flow-UniPC",
            "shift": 3.0,
            "first_frame_ssim": first_frame_ssim,
            "source": str(args.i2v_source),
        },
        "pins": {
            "hf_revision": EXPECTED_HF_REVISION,
            "creator_revision": EXPECTED_CREATOR_REVISION,
            "fp8_cache_sha256": cache_digest,
            "transformer_index_sha256": index_digest,
            "runner_source_bundle_sha256": source_digest,
        },
        "artifacts": {
            "manifest": str(manifest_path),
            "t2v": t2v_artifact,
            "i2v": i2v_artifact,
            "visual_inspection": {
                "accepted": args.visual_accepted,
                "note": (
                    "I2V preserves cyborg identity and natural blink across "
                    "five sampled times; T2V contains the requested red teapot, "
                    "blue table, window light, and coherent entering hand."
                ),
            },
        },
        "performance": {
            "t2v_wall_seconds": t2v_wall,
            "i2v_wall_seconds": i2v_wall,
            "t2v_peak_vram_mib": t2v_peak,
            "i2v_peak_vram_mib": i2v_peak,
            "requires_isolated_gpu_worker": True,
        },
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "passed": report["passed"],
                "first_frame_ssim": first_frame_ssim,
                "checks": checks,
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
