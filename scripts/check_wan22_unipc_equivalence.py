#!/usr/bin/env python3
"""Prove the shared UniPC oracle is equivalent to pinned Wan 2.2 UniPC."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
WAN_ROOT = Path("/home/alex/Wan2.2")
WAN_REVISION = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
WAN_SCHEDULER = WAN_ROOT / "wan/utils/fm_solvers_unipc.py"
COSMOS_SCHEDULER = Path(
    "/home/alex/refs/cosmos-predict2.5/cosmos_predict2/"
    "_src/predict2/models/fm_solvers_unipc.py"
)
DEFAULT_REPORT = (
    REPO
    / "output/checks/wan22_20260729/scheduler/source_equivalence.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scheduler: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_pin() -> None:
    head = subprocess.check_output(
        ["git", "-C", str(WAN_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(WAN_ROOT), "status", "--porcelain"],
        text=True,
    ).strip()
    if head != WAN_REVISION or dirty:
        raise RuntimeError(
            f"Wan checkout must be clean at {WAN_REVISION}; "
            f"head={head!r} dirty={bool(dirty)}"
        )


def run(module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scheduler = module.FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(50, device="cpu", shift=5)
    rng = np.random.default_rng(1234)
    sample = torch.from_numpy(
        rng.standard_normal((1, 1, 8))
    ).to(torch.float64)
    outputs = []
    for index in range(50):
        velocity = torch.from_numpy(
            rng.standard_normal((1, 1, 8))
        ).to(torch.float64)
        sample = scheduler.step(
            velocity,
            scheduler.timesteps[index],
            sample,
            return_dict=False,
        )[0]
        outputs.append(sample.clone())
    return (
        scheduler.sigmas.to(torch.float64),
        scheduler.timesteps.to(torch.float64),
        torch.stack(outputs),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    check_pin()
    wan = run(load_module("wan22_unipc", WAN_SCHEDULER))
    shared = run(load_module("cosmos_unipc", COSMOS_SCHEDULER))
    schedule_max = float((wan[0] - shared[0]).abs().max())
    timestep_max = float((wan[1] - shared[1]).abs().max())
    trajectory_max = float((wan[2] - shared[2]).abs().max())
    trajectory_cosine = float(
        torch.nn.functional.cosine_similarity(
            wan[2].flatten(),
            shared[2].flatten(),
            dim=0,
        )
    )
    passed = (
        schedule_max == 0.0
        and timestep_max == 0.0
        and trajectory_max == 0.0
    )
    report = {
        "schema": "serenity.wan22.unipc_source_equivalence.v1",
        "passed": passed,
        "wan_revision": WAN_REVISION,
        "steps": 50,
        "shift": 5.0,
        "solver_order": 2,
        "wan_scheduler": str(WAN_SCHEDULER),
        "wan_scheduler_sha256": sha256(WAN_SCHEDULER),
        "shared_oracle": str(COSMOS_SCHEDULER),
        "shared_oracle_sha256": sha256(COSMOS_SCHEDULER),
        "schedule_max_abs": schedule_max,
        "timestep_max_abs": timestep_max,
        "trajectory_max_abs": trajectory_max,
        "trajectory_cosine": trajectory_cosine,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
