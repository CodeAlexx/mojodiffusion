#!/usr/bin/env python3
"""Fail-closed Wan2.2 TI2V-5B creator-parity product gate.

The gate accepts only the creator-native 720p BF16 profile. It binds the local
artifact view and current runtime sources to numeric conditioning, scheduler,
30-block transformer, and VAE parity evidence plus real native T2V/I2V videos.
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
VAE_PATH = Path("/home/alex/.serenity/models/vaes/wan2.2_vae.safetensors")
CHECK_ROOT = REPO / "output/checks/wan22_20260729"
DEFAULT_T2V = CHECK_ROOT / "t2v_native_bf16_teapot_extended"
DEFAULT_I2V = CHECK_ROOT / "i2v_native_bf16_cyborg"
DEFAULT_SOURCE = (
    REPO / "output/run_serenity_ui_i2v_check/uploads/canvas_init-0018.png"
)
DEFAULT_CONDITIONING = (
    CHECK_ROOT
    / "t2v_landscape_teapot_creator_clean/conditioning_parity.json"
)
DEFAULT_SCHEDULER_SOURCE = CHECK_ROOT / "scheduler/source_equivalence.json"
DEFAULT_SCHEDULER_MOJO = CHECK_ROOT / "scheduler/mojo_parity.log"
DEFAULT_TRANSFORMER_SMALL = CHECK_ROOT / "transformer_bf16_small.log"
DEFAULT_TRANSFORMER_LARGE = CHECK_ROOT / "transformer_bf16_large.log"
DEFAULT_TRANSFORMER_STREAM = (
    CHECK_ROOT / "transformer_bf16_stream/mojo_parity.log"
)
DEFAULT_VAE_ORACLE = CHECK_ROOT / "vae/creator_decode_oracle.json"
DEFAULT_VAE_MOJO = CHECK_ROOT / "vae/mojo_parity.log"
DEFAULT_VAE_ENCODER_MOJO = CHECK_ROOT / "vae_encoder/mojo_parity.log"
DEFAULT_LORA_SMOKE = CHECK_ROOT / "bf16_lora_smoke"
DEFAULT_PROMPT = CHECK_ROOT / "t2v_native_bf16_teapot_extended/prompt.txt"
DEFAULT_NEGATIVE = (
    CHECK_ROOT / "t2v_landscape_teapot_creator_clean/negative.txt"
)
DEFAULT_PROMPT_EXTENSION = (
    CHECK_ROOT / "t2v_creator_prompt_extend_qwen3_4b/manifest.json"
)
DEFAULT_REPORT = REPO / "output/checks/wan22_product_gate.json"

EXPECTED_HF_REVISION = "installed-official-native"
EXPECTED_CREATOR_REVISION = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
EXPECTED_SOURCE_INDEX_SHA256 = (
    "cd769dd8bddb0825ffb3516a39d64fc2ac3a5946fb93337f8594af926d6a0f56"
)
EXPECTED_LOCAL_INDEX_SHA256 = (
    "ff3fe4b6936ac924f881863bcaeda0e5e1e54c8b7e2202b2990aba8fcf18ce47"
)
EXPECTED_VAE_SHA256 = (
    "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156"
)
EXPECTED_SHARD_SHA256 = (
    "07cddfa20368c5e0884ee6660ed82b29d7ac97a9207b31fb630e4557c5308eb7",
    "38b79f68c95618f5341d4deae5ab364f9c74f10e8e903326499d0cb95353f1ff",
    "8d76abc71dee3e61a59ccc3a2e40889bb52ec9697acebfa7110de73f2a510452",
)
EXPECTED_SOURCE_BUNDLE_SHA256 = (
    "ea317b6ae0914c4828d85489c1e5a2d0952d2ca3880a122e56287771f65d24fe"
)
MIN_CONDITIONING_COSINE = 0.999
MIN_TRANSFORMER_COSINE = 0.999
MIN_VAE_COSINE = 0.999
MIN_VAE_ENCODER_COSINE = 0.99

RUNNER_SOURCE_PATHS = (
    "serenitymojo/pipeline/wan22_encode_prompt.mojo",
    "serenitymojo/pipeline/wan22_encode_first_frame.mojo",
    "serenitymojo/pipeline/wan22_t2v.mojo",
    "serenitymojo/tokenizer/wan_prompt_clean.mojo",
    "serenitymojo/tokenizer/t5_tokenizer.mojo",
    "serenitymojo/models/text_encoder/umt5_encoder.mojo",
    "serenitymojo/models/dit/wan22_dit.mojo",
    "serenitymojo/models/vae/wan22_vae_encoder.mojo",
    "serenitymojo/models/vae/wan22_decoder.mojo",
    "serenitymojo/sampling/unipc.mojo",
    "serenitymojo/serve/image_io.mojo",
    "vendor/mojo-libs/image/studio_ops.mojo",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_bundle_sha256() -> str:
    digest = hashlib.sha256()
    for relative in RUNNER_SOURCE_PATHS:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((REPO / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def read_json(path: Path, schema: str | None = None) -> dict:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    if schema is not None and value.get("schema") != schema:
        raise ValueError(
            f"{path}: schema {value.get('schema')!r}, expected {schema!r}"
        )
    return value


def metric_from_log(path: Path, name: str) -> float:
    text = path.resolve(strict=True).read_text(encoding="utf-8")
    match = re.search(
        rf"GATE PASS {re.escape(name)}=\s*([0-9.eE+-]+)",
        text,
    )
    if not match:
        raise ValueError(f"{path}: missing GATE PASS {name}")
    return float(match.group(1))


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
            str(path.resolve(strict=True)),
        ],
        text=True,
    )
    streams = json.loads(output).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one video stream: {path}")
    return streams[0]


def read_wall_seconds(path: Path) -> float:
    match = re.search(
        r"(?:^|\s)(?:wall_seconds|elapsed_seconds)=([0-9.]+)(?:\s|$)",
        path.resolve(strict=True).read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        raise ValueError(f"missing wall_seconds in {path}")
    return float(match.group(1))


def read_peak_vram(path: Path) -> int:
    peak = 0
    with path.resolve(strict=True).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            try:
                peak = max(peak, int(row[1].strip()))
            except ValueError:
                pass
    return peak


def creator_preprocess(source_path: Path, width: int, height: int) -> Image.Image:
    source = Image.open(source_path.resolve(strict=True)).convert("RGB")
    scale = max(width / source.width, height / source.height)
    resized = source.resize(
        (round(source.width * scale), round(source.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def validate_render(render_dir: Path, width: int, height: int) -> dict:
    mp4 = render_dir / "wan22_t2v.mp4"
    probe = probe_mp4(mp4)
    representative_paths = {
        str(index): render_dir / f"frame_{index}.png"
        for index in (0, 30, 60, 90, 120)
    }
    representative = {
        index: sha256(path)
        for index, path in representative_paths.items()
    }
    representative_std = {
        index: float(
            np.asarray(
                Image.open(path).convert("RGB"), dtype=np.float32
            ).std()
        )
        for index, path in representative_paths.items()
    }
    log = (render_dir / "run.log").resolve(strict=True).read_text(
        encoding="utf-8"
    )
    checks = {
        "frame_count": len(list(render_dir.glob("frame_*.png"))) == 121,
        "width": int(probe.get("width", 0)) == width,
        "height": int(probe.get("height", 0)) == height,
        "frames": int(probe.get("nb_frames", 0)) == 121,
        "fps": probe.get("avg_frame_rate") == "24/1",
        "duration": abs(float(probe.get("duration", 0.0)) - 5.041667) < 0.001,
        "bf16_runtime": "precision= bf16" in log,
        "creator_sampling": (
            "steps= 50" in log
            and "guidance= 5.0" in log
            and "shift= 5.0" in log
        ),
        "representative_frames_nonempty": all(
            path.stat().st_size > 0
            for path in representative_paths.values()
        ),
        "representative_frames_nonuniform": all(
            value >= 5.0 for value in representative_std.values()
        ),
        "temporal_frames_change": len(set(representative.values())) > 1,
    }
    return {
        "render_dir": str(render_dir),
        "mp4": str(mp4),
        "mp4_sha256": sha256(mp4),
        "probe": probe,
        "representative_frame_sha256": representative,
        "representative_frame_std": representative_std,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2v-dir", type=Path, default=DEFAULT_T2V)
    parser.add_argument("--i2v-dir", type=Path, default=DEFAULT_I2V)
    parser.add_argument("--i2v-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--conditioning", type=Path, default=DEFAULT_CONDITIONING)
    parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    parser.add_argument(
        "--scheduler-mojo", type=Path, default=DEFAULT_SCHEDULER_MOJO
    )
    parser.add_argument(
        "--transformer-small", type=Path, default=DEFAULT_TRANSFORMER_SMALL
    )
    parser.add_argument(
        "--transformer-large", type=Path, default=DEFAULT_TRANSFORMER_LARGE
    )
    parser.add_argument(
        "--transformer-stream", type=Path, default=DEFAULT_TRANSFORMER_STREAM
    )
    parser.add_argument("--vae-oracle", type=Path, default=DEFAULT_VAE_ORACLE)
    parser.add_argument("--vae-mojo", type=Path, default=DEFAULT_VAE_MOJO)
    parser.add_argument(
        "--vae-encoder-mojo", type=Path, default=DEFAULT_VAE_ENCODER_MOJO
    )
    parser.add_argument("--lora-smoke-dir", type=Path, default=DEFAULT_LORA_SMOKE)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--negative", type=Path, default=DEFAULT_NEGATIVE)
    parser.add_argument(
        "--prompt-extension", type=Path, default=DEFAULT_PROMPT_EXTENSION
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--visual-accepted", action="store_true")
    parser.add_argument("--visual-note", default="")
    args = parser.parse_args()

    manifest_path = MODEL_ROOT / "serenity_wan22_manifest.json"
    index_path = MODEL_ROOT / "diffusion_pytorch_model.safetensors.index.json"
    shard_paths = tuple(
        MODEL_ROOT / f"diffusion_pytorch_model-{index:05}-of-00003.safetensors"
        for index in range(1, 4)
    )
    manifest = read_json(
        manifest_path, "serenity.wan22.artifact_view.v1"
    )
    conditioning = read_json(
        args.conditioning, "serenity.wan22.conditioning_parity.v1"
    )
    scheduler_source = read_json(
        args.scheduler_source,
        "serenity.wan22.unipc_source_equivalence.v1",
    )
    vae_oracle = read_json(
        args.vae_oracle, "serenity.wan22.vae_decode_oracle.v1"
    )
    prompt_extension = read_json(
        args.prompt_extension, "serenity.wan22.prompt_extension.v1"
    )
    scheduler_log = args.scheduler_mojo.resolve(strict=True).read_text(
        encoding="utf-8"
    )
    small_cos = metric_from_log(args.transformer_small, "fullForwardCos")
    large_cos = metric_from_log(
        args.transformer_large, "largeFullForwardCos"
    )
    stream_cos = metric_from_log(
        args.transformer_stream, "fullForwardCos"
    )
    vae_cos = metric_from_log(args.vae_mojo, "wan22VaeDecodeCos")
    vae_encoder_cos = metric_from_log(
        args.vae_encoder_mojo, "wan22VaeEncodeCosMin"
    )
    source_digest = source_bundle_sha256()
    local_index_digest = sha256(index_path)
    shard_digests = tuple(sha256(path) for path in shard_paths)
    vae_digest = sha256(VAE_PATH)

    t2v = validate_render(args.t2v_dir, 1280, 704)
    i2v = validate_render(args.i2v_dir, 704, 1248)
    i2v_log = (args.i2v_dir / "run.log").resolve(strict=True).read_text(
        encoding="utf-8"
    )
    i2v["checks"]["process_isolated_first_frame"] = (
        "loading process-isolated creator first frame:" in i2v_log
    )
    reference = np.asarray(
        creator_preprocess(args.i2v_source, 704, 1248),
        dtype=np.float32,
    )
    first_frame = np.asarray(
        Image.open(args.i2v_dir / "frame_0.png").convert("RGB"),
        dtype=np.float32,
    )
    first_frame_ssim = float(
        structural_similarity(
            reference, first_frame, channel_axis=2, data_range=255
        )
    )

    t2v_wall = read_wall_seconds(args.t2v_dir / "render_timing.txt")
    i2v_wall = read_wall_seconds(args.i2v_dir / "render_timing.txt")
    i2v_first_frame_wall = read_wall_seconds(
        args.i2v_dir / "first_frame_timing.txt"
    )
    t2v_peak = read_peak_vram(args.t2v_dir / "gpu.csv")
    i2v_peak = read_peak_vram(args.i2v_dir / "gpu.csv")
    i2v_first_frame_peak = read_peak_vram(
        args.i2v_dir / "first_frame_gpu.csv"
    )
    lora_log = (args.lora_smoke_dir / "run.log").resolve(strict=True).read_text(
        encoding="utf-8"
    )
    lora_wall = read_wall_seconds(args.lora_smoke_dir / "render_timing.txt")
    lora_peak = read_peak_vram(args.lora_smoke_dir / "gpu.csv")
    prompt_text = args.prompt.resolve(strict=True).read_text(encoding="utf-8")
    negative_text = args.negative.resolve(strict=True).read_text(
        encoding="utf-8"
    )

    checks = {
        "zero_copy_artifacts": (
            manifest.get("source_kind") == "official_native_zero_copy"
        ),
        "hf_revision": manifest.get("revision") == EXPECTED_HF_REVISION,
        "creator_revision": (
            manifest.get("oracle_revision") == EXPECTED_CREATOR_REVISION
        ),
        "source_transformer_index": (
            manifest.get("transformer", {}).get("index_sha256")
            == EXPECTED_SOURCE_INDEX_SHA256
        ),
        "local_transformer_index": (
            local_index_digest == EXPECTED_LOCAL_INDEX_SHA256
        ),
        "bf16_transformer_shards": (
            shard_digests == EXPECTED_SHARD_SHA256
        ),
        "bf16_vae": vae_digest == EXPECTED_VAE_SHA256,
        "runner_source_bundle": (
            source_digest == EXPECTED_SOURCE_BUNDLE_SHA256
        ),
        "conditioning_parity": (
            conditioning.get("passed") is True
            and min(
                float(conditioning["tensors"]["pos"]["valid_rows"]["cosine"]),
                float(conditioning["tensors"]["neg"]["valid_rows"]["cosine"]),
            )
            >= MIN_CONDITIONING_COSINE
        ),
        "scheduler_source_equivalence": scheduler_source.get("passed") is True,
        "scheduler_mojo_parity": (
            "PASS: Cosmos RF + UniPC step-parity vs canonical >= 0.999"
            in scheduler_log
        ),
        "transformer_small_parity": small_cos >= MIN_TRANSFORMER_COSINE,
        "transformer_large_parity": large_cos >= MIN_TRANSFORMER_COSINE,
        "transformer_bf16_stream_parity": (
            stream_cos >= MIN_TRANSFORMER_COSINE
        ),
        "vae_creator_oracle": (
            vae_oracle.get("passed") is True
            and vae_oracle.get("oracle_revision")
            == EXPECTED_CREATOR_REVISION
        ),
        "vae_mojo_parity": vae_cos >= MIN_VAE_COSINE,
        "vae_encoder_mojo_parity": (
            vae_encoder_cos >= MIN_VAE_ENCODER_COSINE
        ),
        "bf16_streamed_lora_smoke": (
            "precision= bf16" in lora_log
            and "Wan BF16 streamed LoRA attached: mappings= 300" in lora_log
            and "GATE denoise-only final latent numel=" in lora_log
            and lora_wall > 0
            and 0 < lora_peak <= 24_064
        ),
        "t2v_native_bf16_artifact": all(t2v["checks"].values()),
        "i2v_native_bf16_artifact": all(i2v["checks"].values()),
        "i2v_first_frame_identity": first_frame_ssim >= 0.95,
        "measured_peak_vram": (
            0 < t2v_peak <= 24_064
            and 0 < i2v_peak <= 24_064
            and 0 < i2v_first_frame_peak <= 24_064
        ),
        "measured_wall_time": (
            t2v_wall > 0 and i2v_wall > 0 and i2v_first_frame_wall > 0
        ),
        "prompt_provenance": bool(prompt_text.strip() and negative_text.strip()),
        "creator_prompt_extension": (
            prompt_extension.get("method")
            == "creator_system_prompt_local_qwen"
            and prompt_extension.get("extended_prompt", "").strip()
            == prompt_text.strip()
        ),
        "visual_acceptance": (
            args.visual_accepted and bool(args.visual_note.strip())
        ),
        "required_binaries": all(
            (REPO / path).is_file()
            for path in (
                "output/bin/wan22_encode_prompt",
                "output/bin/wan22_encode_first_frame_704x1248",
                "output/bin/wan22_t2v_1280x704",
                "output/bin/wan22_t2v_704x1248",
            )
        ),
    }
    report = {
        "schema": "serenity.wan22.product_gate.v3",
        "passed": all(checks.values()),
        "profile": {
            "model": "Wan-AI/Wan2.2-TI2V-5B",
            "mode": "t2v",
            "width": 1280,
            "height": 704,
            "frames": 121,
            "fps": 24,
            "steps": 50,
            "guidance": 5.0,
            "sampler": "Flow-UniPC",
            "shift": 5.0,
            "quant": "bf16",
        },
        "i2v_profile": {
            "mode": "first_frame",
            "width": 704,
            "height": 1248,
            "frames": 121,
            "fps": 24,
            "steps": 50,
            "guidance": 5.0,
            "sampler": "Flow-UniPC",
            "shift": 5.0,
            "quant": "bf16",
            "first_frame_ssim": first_frame_ssim,
            "source": str(args.i2v_source),
        },
        "pins": {
            "hf_revision": EXPECTED_HF_REVISION,
            "creator_revision": EXPECTED_CREATOR_REVISION,
            "source_transformer_index_sha256": (
                EXPECTED_SOURCE_INDEX_SHA256
            ),
            "local_transformer_index_sha256": local_index_digest,
            "bf16_transformer_shard_sha256": list(shard_digests),
            "bf16_vae_sha256": vae_digest,
            "runner_source_bundle_sha256": source_digest,
        },
        "numeric_parity": {
            "conditioning": conditioning,
            "scheduler_source": scheduler_source,
            "transformer_small_cosine": small_cos,
            "transformer_large_cosine": large_cos,
            "transformer_bf16_stream_cosine": stream_cos,
            "vae_decode_cosine": vae_cos,
            "vae_encode_min_cosine": vae_encoder_cos,
            "vae_oracle": vae_oracle,
        },
        "artifacts": {
            "manifest": str(manifest_path),
            "prompt": {
                "path": str(args.prompt),
                "sha256": sha256(args.prompt),
            },
            "prompt_extension": prompt_extension,
            "negative_prompt": {
                "path": str(args.negative),
                "sha256": sha256(args.negative),
            },
            "t2v": t2v,
            "i2v": i2v,
            "visual_inspection": {
                "accepted": args.visual_accepted,
                "note": args.visual_note.strip(),
            },
        },
        "performance": {
            "t2v_wall_seconds": t2v_wall,
            "i2v_wall_seconds": i2v_wall,
            "i2v_first_frame_encode_wall_seconds": i2v_first_frame_wall,
            "bf16_lora_smoke_wall_seconds": lora_wall,
            "t2v_peak_vram_mib": t2v_peak,
            "i2v_peak_vram_mib": i2v_peak,
            "i2v_first_frame_encode_peak_vram_mib": i2v_first_frame_peak,
            "bf16_lora_smoke_peak_vram_mib": lora_peak,
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
