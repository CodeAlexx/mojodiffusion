#!/usr/bin/env python3
"""Fail-closed Bernini-R product gate from measured local evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODEL = Path("/home/alex/.serenity/models/checkpoints/Bernini-R-Diffusers")
REVISION = "de8c4621d3ac75cc33efe3db8deaed2023e9ac8c"
ORACLE_REVISION = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
EXPECTED_UMT5 = {
    "a8e861969c7433e707cc5a74065d795d36cca07ec96eb6763eb4083df7248f58",
    "d57d948ece4837d850b7a859a4415121d57cacf8b9ee1d4db200c67f592902d7",
    "0da9ee284e21d1406df708788db1d502d95d75f69faa25cd26151bf8829b7c5f",
}
DEFAULT_RENDER = REPO / "output/checks/bernini_r/hq_848x480_f81_seed42"
DEFAULT_REPORT = REPO / "output/checks/bernini_r/product_gate.json"
ENCODE_RUNNER = REPO / "output/bin/wan22_encode_prompt"
DENOISE_RUNNER = REPO / "output/bin/bernini_t2v"
DECODE_RUNNER = REPO / "output/bin/bernini_decode"
CUDNN_RUNTIME = Path("/home/alex/.serenity/cudnn/lib/libcudnn.so.9")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def forward_cos(path: Path, component: str) -> float:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"GATE PASS\s+{re.escape(component)}\s+fullForwardCos=\s*([0-9.eE+-]+)", text)
    return float(match.group(1)) if match else 0.0


def probe(path: Path) -> list[dict]:
    output = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=index,codec_type,width,height,avg_frame_rate,nb_frames,duration",
            "-of", "json", str(path),
        ],
        text=True,
    )
    return json.loads(output).get("streams", [])


def verify_cache(manifest_path: Path, component: str) -> tuple[bool, dict]:
    manifest = read_json(manifest_path)
    ok = (
        manifest.get("schema") == "serenity.bernini_r.fp8_cache.v1"
        and manifest.get("passed") is True
        and manifest.get("revision") == REVISION
        and manifest.get("source_component") == component
        and len(manifest.get("cache_files", [])) == 41
    )
    cache = Path(manifest.get("cache_dir", ""))
    if ok:
        for record in manifest["cache_files"]:
            path = cache / record["path"]
            if not path.is_file() or sha256(path) != record.get("sha256"):
                ok = False
                break
    return ok, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--high-cache-manifest", type=Path, required=True)
    parser.add_argument("--low-cache-manifest", type=Path, required=True)
    parser.add_argument("--peak-vram-mib", type=int, required=True)
    parser.add_argument("--denoise-seconds", type=float, required=True)
    parser.add_argument("--decode-seconds", type=float, required=True)
    parser.add_argument("--visual-accepted", action="store_true")
    args = parser.parse_args()

    evidence = REPO / "output/checks/bernini_r"
    render = args.render_dir.resolve(strict=True)
    artifact = read_json(MODEL / "serenity_bernini_r_manifest.json")
    conditioning = read_json(REPO / "output/checks/wan22/conditioning/parity.json")
    conditioning_sensitivity = read_json(
        render / "conditioning_sensitivity.json"
    )
    conditioning_velocity_cos = float(
        conditioning_sensitivity.get(
            "oracle_conditioning_vs_mojo_conditioning_after_real_product_first_step",
            {},
        ).get("cosine", 0.0)
    )
    conditioning_inputs = conditioning_sensitivity.get("inputs", {})
    conditioning_inputs_ok = set(conditioning_inputs) == {
        "creator_conditioned_latent",
        "mojo_conditioned_latent",
    }
    for record in conditioning_inputs.values():
        path = Path(record.get("path", ""))
        if not path.is_file() or sha256(path) != record.get("sha256"):
            conditioning_inputs_ok = False
            break
    vae_reuse = read_json(evidence / "vae_reuse.json")
    apg_text = (evidence / "apg_parity.log").read_text(encoding="utf-8")
    scheduler_text = (evidence / "scheduler_parity.log").read_text(encoding="utf-8")
    padmask_text = (evidence / "text_padmask_parity.log").read_text(encoding="utf-8")
    block_text = (evidence / "block_parity.log").read_text(encoding="utf-8")
    high_cos = forward_cos(evidence / "forward_high.log", "transformer")
    low_cos = forward_cos(evidence / "forward_low.log", "transformer_2")
    high_cache_ok, high_cache = verify_cache(
        args.high_cache_manifest.resolve(strict=True), "high_noise_transformer"
    )
    low_cache_ok, low_cache = verify_cache(
        args.low_cache_manifest.resolve(strict=True), "low_noise_transformer"
    )

    mp4 = render / "bernini_r_t2v.mp4"
    streams = probe(mp4)
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    frames = sorted(render.glob("frame_*.png"))
    representative = {
        str(index): sha256(render / f"frame_{index}.png") for index in (0, 40, 80)
    }
    video = videos[0] if len(videos) == 1 else {}
    umt5_hashes = {
        item.get("sha256") for item in artifact.get("umt5", {}).get("shards", [])
    }

    checks = {
        "artifact_gate": artifact.get("artifact_gate_passed") is True,
        "model_revision": artifact.get("revision") == REVISION,
        "creator_revision": artifact.get("oracle_revision") == ORACLE_REVISION,
        "umt5_content_identity": umt5_hashes == EXPECTED_UMT5,
        "conditioning_parity_reused_by_hash": conditioning.get("passed") is True,
        "representative_conditioning_velocity_cosine": (
            conditioning_sensitivity.get("schema")
            == "serenity.bernini_r.conditioning_sensitivity.v1"
            and conditioning_sensitivity.get("passed") is True
            and conditioning_inputs_ok
            and conditioning_velocity_cos >= 0.999
        ),
        "apg_creator_parity": "GATE PASS Bernini APG" in apg_text,
        "scheduler_creator_parity": "GATE PASS Bernini UniPC" in scheduler_text,
        "text_padding_masked": "GATE PASS Bernini text padmask" in padmask_text,
        "real_block_parity": "GATE PASS Bernini-R block" in block_text,
        "high_full_forward_parity": high_cos >= 0.99,
        "low_full_forward_parity": low_cos >= 0.99,
        "high_cache_bound": high_cache_ok,
        "low_cache_bound": low_cache_ok,
        "cache_sources_distinct": {
            item.get("sha256") for item in high_cache.get("source_shards", [])
        } != {
            item.get("sha256") for item in low_cache.get("source_shards", [])
        },
        "vae_content_and_temporal_parity": vae_reuse.get("passed") is True,
        "frame_count": len(frames) == 81,
        "one_video_stream": len(videos) == 1,
        "no_audio_expected_for_bernini_r": len(audios) == 0,
        "mp4_width": int(video.get("width", 0)) == 848,
        "mp4_height": int(video.get("height", 0)) == 480,
        "mp4_frames": int(video.get("nb_frames", 0)) == 81,
        "mp4_fps": video.get("avg_frame_rate") == "16/1",
        "mp4_duration": abs(float(video.get("duration", 0.0)) - 5.0625) < 0.001,
        "peak_vram_within_card": 0 < args.peak_vram_mib <= 16303,
        "denoise_measured": args.denoise_seconds > 0,
        "decode_measured": args.decode_seconds > 0,
        "visual_acceptance": args.visual_accepted,
    }
    report = {
        "schema": "serenity.bernini_r.product_gate.v1",
        "passed": all(checks.values()),
        "pins": {
            "hf_revision": REVISION,
            "creator_revision": ORACLE_REVISION,
            "high_cache_aggregate_sha256": high_cache.get("cache_aggregate_sha256"),
            "low_cache_aggregate_sha256": low_cache.get("cache_aggregate_sha256"),
            "encode_runner_sha256": sha256(ENCODE_RUNNER.resolve(strict=True)),
            "denoise_runner_sha256": sha256(DENOISE_RUNNER.resolve(strict=True)),
            "decode_runner_sha256": sha256(DECODE_RUNNER.resolve(strict=True)),
            "cudnn_runtime_sha256": sha256(CUDNN_RUNTIME.resolve(strict=True)),
        },
        "profile": {
            "model": "ByteDance/Bernini-R-Diffusers",
            "revision": REVISION,
            "width": 848,
            "height": 480,
            "frames": 81,
            "fps": 16,
            "steps": 40,
            "seed": 42,
            "sampler": "UniPC bh2 flow",
            "flow_shift": 5.0,
            "guidance": "t2v_apg",
            "omega_high": 4.0,
            "omega_low": 3.2,
            "eta": 0.5,
            "norm_threshold": 50.0,
            "quant": "persistent per-row E4M3; one block resident",
        },
        "parity": {
            "high_full_forward_cosine": high_cos,
            "low_full_forward_cosine": low_cos,
            "representative_conditioning_velocity_cosine": conditioning_velocity_cos,
        },
        "artifact": {
            "render_dir": str(render),
            "mp4": str(mp4),
            "streams": streams,
            "representative_frame_sha256": representative,
        },
        "performance": {
            "denoise_seconds": args.denoise_seconds,
            "decode_seconds": args.decode_seconds,
            "peak_vram_mib": args.peak_vram_mib,
            "requires_isolated_process_stages": True,
        },
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report), "passed": report["passed"], "checks": checks}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
