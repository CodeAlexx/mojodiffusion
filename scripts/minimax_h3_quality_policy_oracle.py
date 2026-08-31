#!/usr/bin/env python3
"""Generate a pinned Musubi H3 quality-policy value/gradient fixture.

Standalone upstream functions are AST-extracted and executed directly from the
immutable source. Guidance gating is outside the upstream helper; its exact
target assignments are source-verified before the same Torch F32 expression is
evaluated here. This is a development oracle only.
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
    "serenitymojo/training/parity/fixtures/minimax_h3_quality_policy_v1.json"
)
EXECUTED_FUNCTIONS = (
    "_apply_timestep_focus",
    "_preservation_density_compensation",
    "_decomposed_flow_loss",
    "_dc_attenuated_prediction",
)


def load_exact_functions(source: str) -> dict[str, object]:
    tree = ast.parse(source, filename=SOURCE_FILE)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in EXECUTED_FUNCTIONS
    ]
    if {node.name for node in selected} != set(EXECUTED_FUNCTIONS):
        raise RuntimeError("pinned Musubi quality-policy functions are missing")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {"torch": torch}
    exec(compile(module, SOURCE_FILE, "exec"), namespace)
    return namespace


def verify_guidance_contract(source: str) -> None:
    required = (
        "applied = float(timesteps) >= float(args.h3_guidance_loss_sigma_min)",
        "video_target = uncond_video + float(args.h3_guidance_loss_scale) * video_gap",
        "audio_target = uncond_audio + self._guidance_audio_scale(args) * audio_gap",
    )
    for text in required:
        if text not in source:
            raise RuntimeError(f"pinned guidance contract changed: {text}")


def values(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().flatten()]


def guidance_case(
    *,
    base_sigma: float,
    sigma_min: float,
    video_velocity: torch.Tensor,
    video_uncond: torch.Tensor,
    video_scale: float,
    audio_velocity: torch.Tensor,
    audio_uncond: torch.Tensor,
    audio_scale: float,
) -> dict[str, object]:
    applied = base_sigma >= sigma_min
    video = (
        video_uncond + video_scale * (video_velocity - video_uncond)
        if applied
        else video_velocity
    )
    audio = (
        audio_uncond + audio_scale * (audio_velocity - audio_uncond)
        if applied
        else audio_velocity
    )
    return {
        "inputs": {
            "base_sigma": base_sigma,
            "sigma_min": sigma_min,
            "video_velocity": values(video_velocity),
            "video_uncond": values(video_uncond),
            "video_scale": video_scale,
            "audio_velocity": values(audio_velocity),
            "audio_uncond": values(audio_uncond),
            "audio_scale": audio_scale,
        },
        "outputs": {
            "applied": applied,
            "video": values(video),
            "audio": values(audio),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with urlopen(SOURCE_URL, timeout=30) as response:
        source_bytes = response.read()
    source = source_bytes.decode("utf-8")
    exact = load_exact_functions(source)
    verify_guidance_contract(source)
    dtype = torch.float32

    focus_inputs = {
        "draws": [0.11, 0.41, 0.73, 0.96],
        "focus_min": 0.3,
        "focus_max": 0.77,
        "focus_probability": 0.55,
    }
    focus_outputs = [
        float(
            exact["_apply_timestep_focus"](
                torch.tensor(draw, dtype=dtype),
                focus_inputs["focus_min"],
                focus_inputs["focus_max"],
                focus_inputs["focus_probability"],
            )
        )
        for draw in focus_inputs["draws"]
    ]
    preservation_inputs = {
        "sigma_max": 0.74,
        "focus_min": 0.3,
        "focus_max": 0.77,
        "focus_probability": 0.55,
    }
    preservation_output = float(
        exact["_preservation_density_compensation"](
            preservation_inputs["sigma_max"],
            preservation_inputs["focus_min"],
            preservation_inputs["focus_max"],
            preservation_inputs["focus_probability"],
        )
    )

    prediction = torch.tensor(
        [0.37, -1.21, 2.03, 0.48, -0.66, 1.17], dtype=dtype, requires_grad=True
    )
    target = torch.tensor([-0.42, 0.91, 1.34, -1.08, 0.23, 0.76], dtype=dtype)
    magnitude_weight = 0.35
    direction_weight = 1.0
    decomposed = exact["_decomposed_flow_loss"](
        prediction, target, magnitude_weight, direction_weight
    )
    decomposed.backward()
    pred_norm = prediction.detach().norm()
    target_norm = target.norm()
    cosine = torch.dot(prediction.detach(), target) / (pred_norm * target_norm + 1.0e-12)
    magnitude_term = (pred_norm - target_norm).square() / prediction.numel()
    direction_term = (
        2.0 * pred_norm * target_norm * (1.0 - cosine) / prediction.numel()
    )

    dc_prediction = torch.tensor(
        [0.2, 1.4, -0.7, 2.1, -1.3, 0.8, 1.9, -0.2],
        dtype=dtype,
        requires_grad=True,
    ).reshape(1, 2, 4)
    dc_prediction.retain_grad()
    dc_target = torch.tensor(
        [-0.4, 0.9, 0.1, 1.2, -0.8, -0.1, 1.1, 0.5], dtype=dtype
    ).reshape(1, 2, 4)
    dc_weight = 0.27
    shaped = exact["_dc_attenuated_prediction"](
        dc_prediction, dc_target, dc_weight
    )
    dc_loss = torch.nn.functional.mse_loss(shaped, dc_target, reduction="mean")
    dc_loss.backward()

    video_velocity = torch.tensor([0.4, -1.2, 2.3, 0.7], dtype=dtype)
    video_uncond = torch.tensor([-0.3, 0.5, 1.1, -0.8], dtype=dtype)
    audio_velocity = torch.tensor([1.7, -0.2, 0.6], dtype=dtype)
    audio_uncond = torch.tensor([0.1, 0.9, -1.3], dtype=dtype)
    guidance_common = {
        "sigma_min": 0.15,
        "video_velocity": video_velocity,
        "video_uncond": video_uncond,
        "video_scale": 3.6,
        "audio_velocity": audio_velocity,
        "audio_uncond": audio_uncond,
        "audio_scale": 2.2,
    }

    payload = {
        "schema": "serenity.minimax_h3.quality_policy_oracle.v1",
        "oracle_repository": "https://github.com/kohya-ss/musubi-tuner",
        "oracle_commit": ORACLE_COMMIT,
        "source_file": SOURCE_FILE,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "torch_dtype": str(dtype),
        "executed_upstream_functions": list(EXECUTED_FUNCTIONS),
        "guidance_source_contract_verified": True,
        "guidance": {
            "applied": guidance_case(base_sigma=0.62, **guidance_common),
            "skipped": guidance_case(base_sigma=0.08, **guidance_common),
        },
        "focus": {"inputs": focus_inputs, "outputs": focus_outputs},
        "preservation": {
            "inputs": preservation_inputs,
            "output": preservation_output,
        },
        "decomposed": {
            "inputs": {
                "prediction": values(prediction),
                "target": values(target),
                "magnitude_weight": magnitude_weight,
                "direction_weight": direction_weight,
            },
            "outputs": {
                "value": float(decomposed.detach()),
                "magnitude_term": float(magnitude_term),
                "direction_term": float(direction_term),
                "prediction_gradient": values(prediction.grad),
            },
        },
        "dc": {
            "inputs": {
                "prediction": values(dc_prediction),
                "target": values(dc_target),
                "batch_channel_groups": 2,
                "dc_weight": dc_weight,
            },
            "outputs": {
                "shaped_prediction": values(shaped),
                "value": float(dc_loss.detach()),
                "prediction_gradient": values(dc_prediction.grad),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
