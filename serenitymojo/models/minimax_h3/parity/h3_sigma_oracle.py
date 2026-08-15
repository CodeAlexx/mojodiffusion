# h3_sigma_oracle.py — torch oracle for the H3 training sigma policy, noising,
# joint velocity loss, and loss gradient (musubi akane/minimax-h3 @ 04324c28,
# functions called DIRECTLY — prepare_joint_noisy_inputs + joint_velocity_loss).
#
# Emits output/checks/h3_sigma_oracle.safetensors with, per base-sigma case i:
#   case{i}_video_xt / case{i}_audio_xt          bf16 noisy inputs
#   case{i}_video_target / case{i}_audio_target  bf16 velocity targets
#   case{i}_sigma  [4] f32: video_sigma, audio_sigma, video_ts, audio_ts
#   case{i}_loss   [1] f32 joint token loss (w_v = w_a = 1)
#   case{i}_pred_video / case{i}_pred_audio      bf16 fake predictions
#   case{i}_grad_video / case{i}_grad_audio      bf16 d loss / d pred
# Shared: video_x0/video_noise [1,24,5,24,40] bf16, audio_x0/audio_noise
# [1,2,32,54] bf16, audio_mask [54] bool (last 2 False), video_mask
# [5,24,40] bool (first 2 latent rows of every frame False), base_sigmas [N].
# Case 4 additionally exercises the video mask; other cases mask audio only.
#
# Run:  /home/alex/musubi-tuner/.venv/bin/python h3_sigma_oracle.py
import sys

sys.path.insert(0, "/home/alex/musubi-h3/src")

import torch
from safetensors.torch import save_file

from musubi_tuner.minimax_h3.training import (
    H3ModelPrediction,
    joint_velocity_loss,
    prepare_joint_noisy_inputs,
    shift_sigma,
)

OUT = "/home/alex/mojodiffusion/output/checks/h3_sigma_oracle.safetensors"

torch.manual_seed(7)
video_x0 = torch.randn(1, 24, 5, 24, 40, dtype=torch.float32).to(torch.bfloat16)
video_noise = torch.randn(1, 24, 5, 24, 40, dtype=torch.float32).to(torch.bfloat16)
audio_x0 = torch.randn(1, 2, 32, 54, dtype=torch.float32).to(torch.bfloat16)
audio_noise = torch.randn(1, 2, 32, 54, dtype=torch.float32).to(torch.bfloat16)
audio_mask = torch.ones(54, dtype=torch.bool)
audio_mask[-2:] = False
video_mask = torch.ones(5, 24, 40, dtype=torch.bool)
video_mask[:, :2, :] = False

BASE_SIGMAS = [0.0, 0.25, 0.5, 0.9, 0.73]

out = {
    "video_x0": video_x0,
    "video_noise": video_noise,
    "audio_x0": audio_x0,
    "audio_noise": audio_noise,
    "audio_mask": audio_mask,
    "video_mask": video_mask,
    "base_sigmas": torch.tensor(BASE_SIGMAS, dtype=torch.float32),
}

for i, base in enumerate(BASE_SIGMAS):
    base_sigma = torch.tensor([base], dtype=torch.float32)
    inputs = prepare_joint_noisy_inputs(
        video_x0, audio_x0, video_noise, audio_noise, base_sigma,
    )
    use_video_mask = i == 4
    pred_video = torch.randn_like(video_x0.float()).to(torch.bfloat16).requires_grad_(True)
    pred_audio = torch.randn_like(audio_x0.float()).to(torch.bfloat16).requires_grad_(True)
    result = joint_velocity_loss(
        H3ModelPrediction(pred_video, pred_audio),
        inputs,
        video_mask=video_mask.unsqueeze(0) if use_video_mask else None,
        audio_mask=audio_mask.unsqueeze(0),
    )
    result.loss.backward()

    out[f"case{i}_video_xt"] = inputs.video.detach()
    out[f"case{i}_audio_xt"] = inputs.audio.detach()
    out[f"case{i}_video_target"] = inputs.video_target.detach()
    out[f"case{i}_audio_target"] = inputs.audio_target.detach()
    out[f"case{i}_sigma"] = torch.stack(
        [inputs.video_sigma[0], inputs.audio_sigma[0], inputs.video_timestep[0], inputs.audio_timestep[0]]
    ).detach()
    out[f"case{i}_loss"] = result.loss.detach().reshape(1).float()
    out[f"case{i}_pred_video"] = pred_video.detach()
    out[f"case{i}_pred_audio"] = pred_audio.detach()
    out[f"case{i}_grad_video"] = pred_video.grad.detach()
    out[f"case{i}_grad_audio"] = pred_audio.grad.detach()

    # cross-check the standalone shift against the joint path
    assert torch.allclose(shift_sigma(base_sigma, 12.0), inputs.video_sigma)
    assert torch.allclose(shift_sigma(base_sigma, 3.0), inputs.audio_sigma)

save_file({k: v.contiguous() for k, v in out.items()}, OUT)
print("wrote", OUT)
for i in range(len(BASE_SIGMAS)):
    s = out[f"case{i}_sigma"]
    print(f"  case{i}: base={BASE_SIGMAS[i]} sigma_v={float(s[0]):.6f} sigma_a={float(s[1]):.6f} loss={float(out[f'case{i}_loss'][0]):.6f}")
