#!/usr/bin/env python3
"""Gate Creator and Mojo LTX-2 audio decodes of one shared latent artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path

import numpy as np


CREATOR_REVISION = "780984275fd47128b02bef9b5c085404276866ee"
REPO = Path(__file__).resolve().parents[1]
MOJO_RUNNER = REPO / "output/bin/ltx2_video_smoke_runner"
MOJO_CSHIM = REPO / "serenitymojo/ops/cshim/lib/libserenity_cudnn_sdpa.so"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected PCM16, got {wav.getsampwidth()} bytes/sample")
        channels = wav.getnchannels()
        rate = wav.getframerate()
        frames = wav.getnframes()
        pcm = np.frombuffer(wav.readframes(frames), dtype="<i2")
    if pcm.size != frames * channels:
        raise ValueError(f"{path}: truncated PCM payload")
    return pcm.reshape(frames, channels), rate


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.astype(np.float64, copy=False).ravel()
    b = right.astype(np.float64, copy=False).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else float(np.array_equal(a, b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("creator_wav", type=Path)
    ap.add_argument("mojo_wav", type=Path)
    ap.add_argument("--shared-latents", type=Path, required=True)
    ap.add_argument("--bar", type=float, default=0.999)
    ap.add_argument(
        "--mojo-runner",
        type=Path,
        default=MOJO_RUNNER,
        help="exact Mojo binary that produced mojo_wav",
    )
    ap.add_argument(
        "--mojo-cshim",
        type=Path,
        default=MOJO_CSHIM,
        help="exact native shim loaded by the Mojo binary",
    )
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    creator, creator_rate = load_pcm16(args.creator_wav)
    mojo, mojo_rate = load_pcm16(args.mojo_wav)
    if creator_rate != mojo_rate:
        raise SystemExit(f"sample-rate mismatch: Creator={creator_rate}, Mojo={mojo_rate}")
    if creator.shape != mojo.shape:
        raise SystemExit(f"waveform shape mismatch: Creator={creator.shape}, Mojo={mojo.shape}")

    channel_cosines = [cosine(creator[:, i], mojo[:, i]) for i in range(creator.shape[1])]
    global_cosine = cosine(creator, mojo)
    creator_f = creator.astype(np.float64)
    mojo_f = mojo.astype(np.float64)
    creator_rms = float(np.sqrt(np.mean(creator_f * creator_f)) / 32768.0)
    mojo_rms = float(np.sqrt(np.mean(mojo_f * mojo_f)) / 32768.0)
    rel_l2 = float(
        np.linalg.norm(mojo_f - creator_f) / max(np.linalg.norm(creator_f), 1e-12)
    )
    metrics = {
        "schema": "serenity.ltx2.audio_parity.v1",
        "creator_revision": CREATOR_REVISION,
        "mojo_runner": str(args.mojo_runner.resolve()),
        "mojo_runner_sha256": sha256_file(args.mojo_runner),
        "mojo_cshim": str(args.mojo_cshim.resolve()),
        "mojo_cshim_sha256": sha256_file(args.mojo_cshim),
        "shared_latents": str(args.shared_latents.resolve()),
        "shared_latents_sha256": sha256_file(args.shared_latents),
        "bar": args.bar,
        "sample_rate": creator_rate,
        "frames": int(creator.shape[0]),
        "channels": int(creator.shape[1]),
        "duration_seconds": creator.shape[0] / creator_rate,
        "global_cosine": global_cosine,
        "channel_cosines": channel_cosines,
        "worst_channel_cosine": min(channel_cosines),
        "relative_l2": rel_l2,
        "creator_rms": creator_rms,
        "mojo_rms": mojo_rms,
        "rms_ratio": mojo_rms / max(creator_rms, 1e-12),
        "max_abs_i16": int(np.max(np.abs(mojo_f - creator_f))),
    }
    metrics["passed"] = (
        global_cosine >= args.bar and min(channel_cosines) >= args.bar
    )
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if metrics["passed"] else 1)


if __name__ == "__main__":
    main()
