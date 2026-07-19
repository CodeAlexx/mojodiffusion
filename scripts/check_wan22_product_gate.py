#!/usr/bin/env python3
"""Build a machine-local Wan 2.2 production-readiness report from evidence.

This script does not run inference and does not bless a render implicitly.  It
verifies the pinned artifact view, cached FP8 weights, measured Mojo parity
reports, representative frame bytes, and the muxed MP4 contract.  Visual
acceptance and measured timing/VRAM are explicit operator inputs from the same
representative run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path("/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B-Mojo")
DEFAULT_RENDER = REPO / "output/checks/wan22/hq_832x480_f121_infer_sdpa_unipc50_seed1234"
DEFAULT_REPORT = REPO / "output/checks/wan22_product_gate.json"
EXPECTED_HF_REVISION = "b8fff7315c768468a5333511427288870b2e9635"
EXPECTED_CREATOR_REVISION = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
EXPECTED_CACHE_SHA256 = "84812d4fe806b7a414c47bd91d02498e8ac07ec5fa4db34ae58dc241524ccb49"
EXPECTED_FRAME_SHA256 = {
    0: "8fbe68d8fce59bfa132889963b1f124231d72e64199207562654e1a1cc9ebd54",
    60: "edfe38ba2abbf3d5c97c2ebeb4d008eb21c5995268f02ae87e69e30b562c4a3d",
    120: "4778755c88982bb63c1bdce96e49ce87ca8bc569cd5c490a0341924970703311",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def probe_mp4(path: Path) -> dict:
    output = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of", "json", str(path),
        ],
        text=True,
    )
    streams = json.loads(output).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one video stream: {path}")
    return streams[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--peak-vram-mib", type=int, required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--visual-accepted", action="store_true")
    args = parser.parse_args()

    manifest_path = MODEL_ROOT / "serenity_wan22_manifest.json"
    cache_path = MODEL_ROOT / "wan22_dit_fp8_e4m3_b8fff7315c768468.safetensors"
    conditioning_path = REPO / "output/checks/wan22/conditioning/parity.json"
    transformer_log = REPO / "output/checks/wan22/transformer/fp8_cached_parity.log"
    mp4_path = args.render_dir / "wan22_t2v.mp4"

    manifest = read_json(manifest_path)
    conditioning = read_json(conditioning_path)
    transformer_text = transformer_log.read_text(encoding="utf-8")
    match = re.search(r"GATE PASS fullForwardCos=\s*([0-9.eE+-]+)", transformer_text)
    transformer_cosine = float(match.group(1)) if match else 0.0
    cache_digest = sha256(cache_path)
    frame_hashes = {
        str(index): sha256(args.render_dir / f"frame_{index}.png")
        for index in EXPECTED_FRAME_SHA256
    }
    frame_count = len(list(args.render_dir.glob("frame_*.png")))
    probe = probe_mp4(mp4_path)

    checks = {
        "artifact_schema": manifest.get("schema") == "serenity.wan22.artifact_view.v1",
        "hf_revision": manifest.get("revision") == EXPECTED_HF_REVISION,
        "creator_revision": manifest.get("oracle_revision") == EXPECTED_CREATOR_REVISION,
        "transformer_shards": manifest.get("transformer", {}).get("shard_count") == 5,
        "umt5_shards": manifest.get("umt5", {}).get("shard_count") == 3,
        "fp8_cache_sha256": cache_digest == EXPECTED_CACHE_SHA256,
        "conditioning_parity": conditioning.get("passed") is True,
        "transformer_fp8_parity": transformer_cosine >= 0.99,
        "frame_count": frame_count == 121,
        "representative_frame_bytes": all(
            frame_hashes[str(index)] == expected
            for index, expected in EXPECTED_FRAME_SHA256.items()
        ),
        "mp4_width": int(probe.get("width", 0)) == 832,
        "mp4_height": int(probe.get("height", 0)) == 480,
        "mp4_frames": int(probe.get("nb_frames", 0)) == 121,
        "mp4_fps": probe.get("avg_frame_rate") == "24/1",
        "mp4_duration": abs(float(probe.get("duration", 0.0)) - 5.041667) < 0.001,
        "measured_peak_vram": 0 < args.peak_vram_mib <= 16303,
        "measured_wall_time": args.wall_seconds > 0.0,
        "visual_acceptance": args.visual_accepted,
    }
    report = {
        "schema": "serenity.wan22.product_gate.v1",
        "passed": all(checks.values()),
        "profile": {
            "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "width": 832,
            "height": 480,
            "frames": 121,
            "fps": 24,
            "steps": 50,
            "guidance": 5.0,
            "sampler": "Flow-UniPC",
            "flow_shift": 5.0,
            "quant": "fp8_e4m3_cached",
        },
        "pins": {
            "hf_revision": EXPECTED_HF_REVISION,
            "creator_revision": EXPECTED_CREATOR_REVISION,
            "fp8_cache_sha256": cache_digest,
        },
        "parity": {
            "transformer_full_forward_cosine": transformer_cosine,
            "conditioning": conditioning,
        },
        "artifact": {
            "render_dir": str(args.render_dir),
            "mp4": str(mp4_path),
            "probe": probe,
            "representative_frame_sha256": frame_hashes,
            "visual_inspection": {
                "accepted": args.visual_accepted,
                "note": "Two subjects, coherent boxing action, stable 121-frame motion; inspected from the representative contact sheet/video.",
            },
        },
        "performance": {
            "wall_seconds": args.wall_seconds,
            "peak_vram_mib": args.peak_vram_mib,
            "gpu_total_vram_mib": 16303,
            "requires_isolated_gpu_worker": True,
        },
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report), "passed": report["passed"], "checks": checks}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
