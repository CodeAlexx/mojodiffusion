#!/usr/bin/env python3
"""Generate the versioned MiniMax H3 schedule/flow-loss oracle fixture.

The schedule function is executed from the immutable pinned Musubi source, not
retyped here. No generated artifact is a product dependency.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import torch


ORACLE_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
SOURCE_FILE = "src/musubi_tuner/minimax_h3_train_network.py"
SOURCE_URL = (
    "https://raw.githubusercontent.com/kohya-ss/musubi-tuner/"
    f"{ORACLE_COMMIT}/{SOURCE_FILE}"
)
DEFAULT_OUTPUT = Path(
    "serenitymojo/training/parity/fixtures/minimax_h3_schedule_flow_loss_v1.json"
)


def load_exact_shift_function(source: str):
    tree = ast.parse(source, filename=SOURCE_FILE)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == "_shift_noise_amount"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, SOURCE_FILE, "exec"), namespace)
    return namespace["_shift_noise_amount"]


def tensor_values(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().flatten()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with urlopen(SOURCE_URL, timeout=30) as response:
        source_bytes = response.read()
    source = source_bytes.decode("utf-8")
    shift = load_exact_shift_function(source)

    dtype = torch.float32
    base = torch.tensor(0.25, dtype=dtype)
    sigma_video = shift(base, 12.0)
    sigma_audio = shift(base, 3.0)
    video_latent = torch.tensor([1.0, -2.0, 0.5], dtype=dtype)
    video_noise = torch.tensor([-1.0, 2.0, 1.5], dtype=dtype)
    audio_latent = torch.tensor([0.25, -0.75], dtype=dtype)
    audio_noise = torch.tensor([1.25, 0.25], dtype=dtype)
    noisy_video = (1.0 - sigma_video) * video_latent + sigma_video * video_noise
    noisy_audio = (1.0 - sigma_audio) * audio_latent + sigma_audio * audio_noise
    video_target = video_latent - video_noise
    audio_target = audio_latent - audio_noise
    video_prediction = torch.tensor([0.0, 2.0, -1.0, 3.0], dtype=dtype)
    video_loss_target = torch.tensor([1.0, 0.0, -1.0, 1.0], dtype=dtype)
    audio_prediction = torch.tensor([2.0, -1.0, 1.0, 3.0], dtype=dtype)
    audio_loss_target = torch.tensor([0.0, -1.0, 2.0, 1.0], dtype=dtype)
    video_loss = torch.nn.functional.mse_loss(video_prediction, video_loss_target, reduction="mean")
    audio_loss = torch.nn.functional.mse_loss(audio_prediction, audio_loss_target, reduction="mean")
    audio_weight = torch.tensor(0.5, dtype=dtype)

    payload = {
        "schema": "serenity.minimax_h3.schedule_flow_loss_oracle.v1",
        "oracle_repository": "https://github.com/kohya-ss/musubi-tuner",
        "oracle_commit": ORACLE_COMMIT,
        "source_file": SOURCE_FILE,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "torch_dtype": str(dtype),
        "inputs": {
            "base_sigma": float(base),
            "video_shift": 12.0,
            "audio_shift": 3.0,
            "video_latent": tensor_values(video_latent),
            "video_noise": tensor_values(video_noise),
            "audio_latent": tensor_values(audio_latent),
            "audio_noise": tensor_values(audio_noise),
            "video_prediction": tensor_values(video_prediction),
            "video_loss_target": tensor_values(video_loss_target),
            "audio_prediction": tensor_values(audio_prediction),
            "audio_loss_target": tensor_values(audio_loss_target),
            "audio_weight": float(audio_weight),
        },
        "outputs": {
            "sigma_video": float(sigma_video),
            "sigma_audio": float(sigma_audio),
            "model_t_video": float(1.0 - sigma_video),
            "model_t_audio": float(1.0 - sigma_audio),
            "noisy_video": tensor_values(noisy_video),
            "noisy_audio": tensor_values(noisy_audio),
            "video_target": tensor_values(video_target),
            "audio_target": tensor_values(audio_target),
            "video_loss": float(video_loss),
            "audio_loss": float(audio_loss),
            "total_loss": float(video_loss + audio_weight * audio_loss),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
