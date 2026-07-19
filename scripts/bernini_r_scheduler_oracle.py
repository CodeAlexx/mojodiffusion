#!/usr/bin/env python3
"""Creator-bound Bernini-R UniPC schedule and trajectory oracle."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler


ORACLE_REVISION = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
MODEL_REVISION = "de8c4621d3ac75cc33efe3db8deaed2023e9ac8c"
DEFAULT_CREATOR = Path("/home/alex/Bernini")
DEFAULT_MODEL = Path("/home/alex/.serenity/models/checkpoints/Bernini-R-Diffusers")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "output/checks/bernini_r/scheduler_oracle"


def write_f32(path: Path, values: torch.Tensor | np.ndarray) -> None:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    path.write_bytes(struct.pack(f"<{array.size}f", *array.tolist()))


def check_creator(creator: Path) -> tuple[Path, str]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=creator, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=creator, text=True
    ).strip()
    if revision != ORACLE_REVISION or dirty:
        raise RuntimeError(
            f"creator authority mismatch: revision={revision}, dirty={bool(dirty)}"
        )
    source_path = creator / "bernini/models/wan_diffusion.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imports = {
        (node.module, alias.name)
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    expected = (
        "diffusers.schedulers.scheduling_unipc_multistep",
        "UniPCMultistepScheduler",
    )
    if expected not in imports:
        raise RuntimeError("creator no longer imports the expected Diffusers UniPC authority")
    return source_path, revision


def scheduler(model_dir: Path, steps: int) -> UniPCMultistepScheduler:
    sch = UniPCMultistepScheduler.from_pretrained(
        model_dir, subfolder="scheduler", flow_shift=5.0
    )
    sch.set_timesteps(steps)
    return sch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator", type=Path, default=DEFAULT_CREATOR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    creator = args.creator.resolve(strict=True)
    model_dir = args.model.resolve(strict=True)
    source_path, creator_revision = check_creator(creator)
    config_path = model_dir / "scheduler/scheduler_config.json"
    if not config_path.is_file():
        raise RuntimeError(f"missing official scheduler config: {config_path}")

    args.output.mkdir(parents=True, exist_ok=True)
    files: dict[str, np.ndarray] = {}

    sch40 = scheduler(model_dir, 40)
    files["sigmas_40"] = sch40.sigmas.cpu().numpy()
    files["timesteps_40"] = sch40.timesteps.cpu().numpy()

    steps = 6
    dim = 8
    sch6 = scheduler(model_dir, steps)
    generator = torch.Generator(device="cpu").manual_seed(20260601)
    # Creator UniPC's einsum contract is [B, token, channel...].  Keep the
    # fixture tiny while exercising that real rank, not a scalar shortcut.
    x = torch.randn((1, 1, dim), generator=generator, dtype=torch.float32)
    velocities = torch.randn((steps, 1, 1, dim), generator=generator, dtype=torch.float32)
    trajectory = []
    x_initial = x.clone()
    for index, timestep in enumerate(sch6.timesteps):
        x = sch6.step(velocities[index], timestep, x, return_dict=False)[0]
        trajectory.append(x.clone())
    files.update(
        {
            "sigmas_6": sch6.sigmas.cpu().numpy(),
            "timesteps_6": sch6.timesteps.cpu().numpy(),
            "x_initial": x_initial.numpy(),
            "velocities": velocities.numpy(),
            "trajectory": torch.stack(trajectory).numpy(),
        }
    )

    hashes = {}
    for name, values in files.items():
        path = args.output / f"{name}.bin"
        write_f32(path, values)
        hashes[name] = {
            "path": path.name,
            "shape": list(np.asarray(values).shape),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    manifest = {
        "schema": "serenity.bernini_r.scheduler_oracle.v1",
        "creator_revision": creator_revision,
        "creator_source": str(source_path),
        "creator_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "model_revision": MODEL_REVISION,
        "scheduler_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "diffusers_version": __import__("diffusers").__version__,
        "flow_shift": 5.0,
        "num_train_timesteps": 1000,
        "production_steps": 40,
        "trajectory_steps": steps,
        "trajectory_dim": dim,
        "files": hashes,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
