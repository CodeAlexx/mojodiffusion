"""MiniMax-H3 rectified-flow Euler scheduler oracle.

Generates the reference schedule and step trajectories for
`serenitymojo/models/minimax_h3/scheduler.mojo` from the reference's OWN
runtime — the diffusers PR huggingface/diffusers#14355, cloned at
/home/alex/minimax_h3_ref/diffusers-src and pinned to head
e1b518dfd5e390e7ba09a79a1d39fe1c6cb52dc1.

Reference module: src/diffusers/schedulers/scheduling_minimax_h3.py
  MiniMaxH3Scheduler.set_timesteps, .scale_noise, .step

Everything dumped is produced by calling the reference. The sample and velocity
inputs are dumped alongside the outputs so the Mojo gate consumes byte-identical
inputs rather than regenerating them.

Run:
    python3 scripts/minimax_h3_scheduler_oracle.py
Writes: output/minimax_h3_scheduler/scheduler_ref.safetensors
"""

import json
import os
import sys

DIFFUSERS_SRC = "/home/alex/minimax_h3_ref/diffusers-src/src"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_scheduler"

sys.path.insert(0, DIFFUSERS_SRC)

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler  # noqa: E402

# (name, shift, num_inference_steps). 12.0 is the released video shift, 3.0 the
# audio one. The large-shift case exercises the `unique_consecutive` collapse the
# reference warns about, where the shift compresses the grid near sigma = 1
# hard enough to create float32 collisions.
CASES = [
    ("video_30", 12.0, 30),
    ("audio_30", 3.0, 30),
    ("video_50", 12.0, 50),
    ("audio_50", 3.0, 50),
    ("video_8", 12.0, 8),
    ("min_2", 12.0, 2),
    ("collapse_1000_400", 1000.0, 400),
]

SAMPLE_LEN = 16


def main() -> None:
    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, object] = {}

    for name, shift, steps in CASES:
        scheduler = MiniMaxH3Scheduler(shift=shift)
        scheduler.set_timesteps(num_inference_steps=steps)
        tensors[f"{name}.sigmas"] = scheduler.sigmas.clone()
        tensors[f"{name}.timesteps"] = scheduler.timesteps.clone()
        meta[name] = {
            "shift": shift,
            "requested_steps": steps,
            "num_sigmas": int(scheduler.sigmas.numel()),
            "num_inference_steps": int(scheduler.num_inference_steps),
        }

        # A full denoising trajectory: one fixed sample, one fixed velocity per
        # step, walked through every `step` call in order.
        generator = torch.Generator().manual_seed(1234)
        sample = torch.randn(SAMPLE_LEN, generator=generator, dtype=torch.float32)
        velocities = torch.randn(
            scheduler.num_inference_steps, SAMPLE_LEN, generator=generator, dtype=torch.float32
        )
        tensors[f"{name}.sample_in"] = sample.clone()
        tensors[f"{name}.velocities"] = velocities.clone()

        trajectory = torch.empty(scheduler.num_inference_steps, SAMPLE_LEN, dtype=torch.float32)
        current = sample.clone()
        for index, timestep in enumerate(scheduler.timesteps):
            current = scheduler.step(velocities[index], timestep, current).prev_sample
            trajectory[index] = current
        tensors[f"{name}.trajectory"] = trajectory

    # scale_noise: the conditioning-anchor forward process, at the noise-aug
    # levels the released model pins its keyframe rows to.
    scheduler = MiniMaxH3Scheduler(shift=12.0)
    generator = torch.Generator().manual_seed(99)
    clean = torch.randn(SAMPLE_LEN, generator=generator, dtype=torch.float32)
    noise = torch.randn(SAMPLE_LEN, generator=generator, dtype=torch.float32)
    tensors["scale_noise.clean"] = clean.clone()
    tensors["scale_noise.noise"] = noise.clone()
    levels = [0.999, 1.0, 0.0, 0.5, 0.001]
    tensors["scale_noise.levels"] = torch.tensor(levels, dtype=torch.float32)
    scaled = torch.empty(len(levels), SAMPLE_LEN, dtype=torch.float32)
    for index, level in enumerate(levels):
        scaled[index] = scheduler.scale_noise(clean, level, noise)
    tensors["scale_noise.out"] = scaled

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "scheduler_ref.safetensors")
    save_file(tensors, path, metadata={"format": "pt"})
    with open(os.path.join(OUT_DIR, "scheduler_ref.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"wrote {len(tensors)} tensors, {total / 1024:.1f} KiB -> {path}")
    for name, info in meta.items():
        print(f"  {name:<20} shift={info['shift']:<8} requested={info['requested_steps']:<4} "
              f"sigmas={info['num_sigmas']:<4} model_calls={info['num_inference_steps']}")


if __name__ == "__main__":
    main()
