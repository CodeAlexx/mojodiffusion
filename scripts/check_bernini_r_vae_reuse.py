#!/usr/bin/env python3
"""Bind Bernini's official VAE bytes to the parity-gated Wan decoder path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL = Path(
    "/home/alex/.serenity/models/checkpoints/Bernini-R-Diffusers/vae/"
    "diffusion_pytorch_model.safetensors"
)
DEFAULT_REUSED = Path(
    "/mnt/disk1/models/lingbot-video-dense/vae/diffusion_pytorch_model.safetensors"
)
DEFAULT_LOG = REPO / "output/checks/bernini_r/vae_temporal_parity.log"
DEFAULT_REPORT = REPO / "output/checks/bernini_r/vae_reuse.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--reused", type=Path, default=DEFAULT_REUSED)
    parser.add_argument("--parity-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    official = args.official.resolve(strict=True)
    reused = args.reused.resolve(strict=True)
    parity_text = args.parity_log.resolve(strict=True).read_text(encoding="utf-8")
    official_hash = sha256(official)
    reused_hash = sha256(reused)
    checks = {
        "content_identical": official_hash == reused_hash,
        "temporal_decoder_parity": "TEMPORAL VAE GATE PASS" in parity_text,
    }
    report = {
        "schema": "serenity.bernini_r.vae_reuse.v1",
        "passed": all(checks.values()),
        "official": {"path": str(official), "sha256": official_hash},
        "reused": {"path": str(reused), "sha256": reused_hash},
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
