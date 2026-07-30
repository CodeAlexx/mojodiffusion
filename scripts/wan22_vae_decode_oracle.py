#!/usr/bin/env python3
"""Generate a pinned creator Wan2.2 high-compression VAE decode fixture.

This is a development-only oracle. It imports the clean, pinned Wan creator
checkout and decodes deterministic standard-normal video latents with the
official Wan2_2_VAE implementation and checkpoint. Product inference remains
pure Mojo and never imports this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file


REPO = Path(__file__).resolve().parents[1]
ORACLE_ROOT = Path("/home/alex/Wan2.2")
ORACLE_REVISION = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
DEFAULT_CHECKPOINT = Path(
    "/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
)
DEFAULT_FIXTURE = (
    REPO / "output/checks/wan22_20260729/vae/creator_decode_fixture.safetensors"
)
DEFAULT_MANIFEST = (
    REPO / "output/checks/wan22_20260729/vae/creator_decode_oracle.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_oracle() -> None:
    head = subprocess.check_output(
        ["git", "-C", str(ORACLE_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(ORACLE_ROOT), "status", "--porcelain"],
        text=True,
    ).strip()
    if head != ORACLE_REVISION or dirty:
        raise RuntimeError(
            "Wan creator checkout must be clean at "
            f"{ORACLE_REVISION}; head={head!r} dirty={bool(dirty)}"
        )


def tensor_stats(value: torch.Tensor) -> dict[str, object]:
    flat = value.float().reshape(-1)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "minimum": float(flat.min()),
        "maximum": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    check_oracle()
    checkpoint = args.checkpoint.resolve(strict=True)
    sys.path.insert(0, str(ORACLE_ROOT))
    from wan.modules.vae2_2 import Wan2_2_VAE

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latent_cthw = torch.randn(
        48,
        5,
        16,
        16,
        generator=generator,
        dtype=torch.float32,
    )
    latent_tokens = (
        latent_cthw.permute(1, 2, 3, 0).contiguous().reshape(5 * 16 * 16, 48)
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    vae = Wan2_2_VAE(
        z_dim=48,
        c_dim=160,
        vae_pth=str(checkpoint),
        dtype=torch.bfloat16,
        device="cuda",
    )
    loaded = time.perf_counter()
    with torch.inference_mode():
        decoded = vae.decode([latent_cthw.to("cuda")])[0]
    torch.cuda.synchronize()
    finished = time.perf_counter()
    frames = decoded.unsqueeze(0).float().cpu().contiguous()
    peak_vram_mib = int(torch.cuda.max_memory_allocated() / (1024 * 1024))

    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "lat_vid": latent_tokens,
            "frames": frames,
        },
        str(args.fixture),
        metadata={
            "schema": "serenity.wan22.vae_decode_fixture.v1",
            "oracle_revision": ORACLE_REVISION,
            "seed": str(args.seed),
        },
    )
    manifest = {
        "schema": "serenity.wan22.vae_decode_oracle.v1",
        "passed": True,
        "oracle_revision": ORACLE_REVISION,
        "oracle_impl": "wan.modules.vae2_2.Wan2_2_VAE",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "fixture": str(args.fixture),
        "fixture_sha256": sha256(args.fixture),
        "seed": args.seed,
        "latent": tensor_stats(latent_tokens),
        "frames": tensor_stats(frames),
        "load_seconds": loaded - started,
        "decode_seconds": finished - loaded,
        "peak_vram_mib": peak_vram_mib,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
