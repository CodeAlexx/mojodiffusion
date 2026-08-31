#!/usr/bin/env python3
"""Print a deterministic H3 policy reference receipt.

This script independently retypes the reviewed equations. It is not a Musubi
parity oracle and is never a product dependency or fallback.
"""

from __future__ import annotations

import json
import math


ORACLE_RECEIPT_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
DATASET_IDENTITY = "eri_with_trigger"


def require_pair(a: list[float], b: list[float]) -> None:
    if not a or len(a) != len(b):
        raise ValueError("vectors must be nonempty and equal length")
    if not all(math.isfinite(value) for value in (*a, *b)):
        raise ValueError("vectors must be finite")


def shift_sigma(base: float, shift: float) -> float:
    if not 0.0 <= base <= 1.0 or not 0.01 <= shift <= 100.0:
        raise ValueError("invalid H3 schedule input")
    return shift * base / (1.0 + (shift - 1.0) * base)


def noisy(latent: list[float], noise: list[float], sigma: float) -> list[float]:
    require_pair(latent, noise)
    return [(1.0 - sigma) * x0 + sigma * eps for x0, eps in zip(latent, noise)]


def target(latent: list[float], noise: list[float]) -> list[float]:
    require_pair(latent, noise)
    return [x0 - eps for x0, eps in zip(latent, noise)]


def mse(prediction: list[float], expected: list[float]) -> float:
    require_pair(prediction, expected)
    return sum((pred - want) ** 2 for pred, want in zip(prediction, expected)) / len(prediction)


def main() -> None:
    base = 0.25
    sigma_video = shift_sigma(base, 12.0)
    sigma_audio = shift_sigma(base, 3.0)
    video_loss = mse([0.0, 2.0, -1.0, 3.0], [1.0, 0.0, -1.0, 1.0])
    audio_loss = mse([2.0, -1.0, 1.0, 3.0], [0.0, -1.0, 2.0, 1.0])
    print(json.dumps({
        "evidence_level": "deterministic_policy_reference_not_parity",
        "oracle_receipt_commit": ORACLE_RECEIPT_COMMIT,
        "dataset_identity": DATASET_IDENTITY,
        "dataset_path": None,
        "base_sigma": base,
        "sigma_video": sigma_video,
        "sigma_audio": sigma_audio,
        "model_t_video": 1.0 - sigma_video,
        "model_t_audio": 1.0 - sigma_audio,
        "noisy_video": noisy([1.0, -2.0, 0.5], [-1.0, 2.0, 1.5], sigma_video),
        "native_target": target([1.0, -2.0, 0.5], [-1.0, 2.0, 1.5]),
        "video_loss": video_loss,
        "audio_loss": audio_loss,
        "total_loss": video_loss + 0.5 * audio_loss,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
