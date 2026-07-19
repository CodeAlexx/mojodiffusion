#!/usr/bin/env python
# Oracle-C: FlowUniPCMultistepScheduler black-box trajectory. Captures set_timesteps
# (steps=40, shift=3) sigmas/timesteps, then a deterministic synthetic denoise loop
# (seeded noise_pred sequence) recording (timestep, sample_in, model_output,
# sample_out) per step. The Mojo scheduler must replay the SAME model_output
# sequence and reproduce the latent trajectory (UniPC is multistep/stateful).
import os, sys, json
import torch
from safetensors.torch import save_file

sys.path.insert(0, "/mnt/disk1/lingbot-src/lingbot-video")
from lingbot_video.scheduling_flow_unipc import FlowUniPCMultistepScheduler

OUT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
STEPS, SHIFT = 40, 3.0
SHAPE = (1, 16, 1, 8, 8)          # T2I-ish latent (B,C,T,H,W)
DEV = "cuda"


def main():
    torch.manual_seed(20260709)
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1,
                                      use_dynamic_shifting=False)
    sch.set_timesteps(STEPS, device=DEV, shift=SHIFT)
    timesteps = sch.timesteps.detach().float().cpu()
    sigmas = sch.sigmas.detach().float().cpu() if hasattr(sch, "sigmas") else torch.zeros(0)

    caps = {"timesteps": timesteps, "sigmas": sigmas}
    meta = {"steps": STEPS, "shift": SHIFT, "shape": list(SHAPE),
            "sigma_max": float(getattr(sch, "sigma_max", 0.0)),
            "sigma_min": float(getattr(sch, "sigma_min", 0.0)),
            "n_timesteps": int(timesteps.numel())}

    latents = torch.randn(SHAPE, device=DEV, dtype=torch.float32)
    caps["latent_init"] = latents.detach().float().cpu()
    for i, t in enumerate(sch.timesteps):
        # deterministic pseudo-velocity: seeded fn of step index (self-consistent replay)
        g = torch.Generator(device=DEV).manual_seed(1000 + i)
        model_output = torch.randn(SHAPE, generator=g, device=DEV, dtype=torch.float32)
        caps[f"in_{i}"] = latents.detach().float().cpu()
        caps[f"mo_{i}"] = model_output.detach().float().cpu()
        latents = sch.step(model_output, t, latents, return_dict=False)[0]
        caps[f"out_{i}"] = latents.detach().float().cpu()

    save_file(caps, os.path.join(OUT, "oracle_c.safetensors"))
    json.dump(meta, open(os.path.join(OUT, "oracle_c_meta.json"), "w"), indent=2)
    print(f"SAVED oracle_c.safetensors  steps={STEPS} shift={SHIFT}")
    print(f"  timesteps[:5]={timesteps[:5].tolist()}  ...[-3:]={timesteps[-3:].tolist()}")
    print(f"  sigma_max={meta['sigma_max']} sigma_min={meta['sigma_min']} n_sigmas={sigmas.numel()}")
    print(f"  final latent mean {latents.mean():.5f} std {latents.std():.5f}")


if __name__ == "__main__":
    main()
