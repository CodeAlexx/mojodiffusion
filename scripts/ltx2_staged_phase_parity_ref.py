#!/usr/bin/env python3
"""Oracle tensors for the LTX2 staged-HQ phase handoff gate.

This is development/oracle tooling only.  The paired Mojo gate recomputes the
same stage-1 final denoise, stage-2 GaussianNoiser init, and first stage-2
res2s/SDE/bong step on-device.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
UP_DIR = REPO / "serenitymojo/models/upsampler/parity"
OUT = REPO / "output/ltx2_staged_phase"

VIDEO_S1 = (1, 128, 2, 8, 8)
VIDEO_S2 = (1, 128, 2, 16, 16)
AUDIO = (1, 8, 8, 16)

RES2S_TERMINAL_SIGMA = np.float32(0.0011)
S2_SIGMA = np.float32(0.909375)
S2_SIGMA_NEXT = np.float32(0.725)
BONG_ITERS = 100


def read_bin(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def write_bin(name: str, value: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    value.astype(np.float32).tofile(OUT / f"{name}.bin")


def pattern(shape: tuple[int, ...], scale: float, offset: float) -> np.ndarray:
    idx = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    value = np.sin(idx * np.float32(0.017) + np.float32(offset))
    value += np.cos(idx * np.float32(0.011) - np.float32(offset * 0.5))
    return (value * np.float32(scale)).astype(np.float32)


def phi(j: int, z: float) -> float:
    if abs(z) < 1.0e-10:
        return 1.0 / math.factorial(j)
    rem = 0.0
    zp = 1.0
    for k in range(j):
        rem += zp / math.factorial(k)
        zp *= z
    return (math.exp(z) - rem) / zp


def coeffs(sigma: np.float32, sigma_next: np.float32) -> tuple[float, float, float, float, np.float32]:
    s = float(sigma)
    sn = float(sigma_next)
    h = math.log(s / sn)
    c2 = 0.5
    a21 = c2 * phi(1, -h * c2)
    b2 = phi(2, -h) / c2
    b1 = phi(1, -h) - b2
    return h, a21, b1, b2, np.float32(math.sqrt(s * sn))


def f32(value: np.ndarray | np.float32 | float) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def denoise_from_vel(x: np.ndarray, vel: np.ndarray, sigma: np.float32) -> np.ndarray:
    return f32(x - f32(sigma) * vel)


def sde_coeffs(sigma_next: float) -> tuple[np.float32, np.float32, np.float32]:
    sigma_up = min(sigma_next * 0.5, sigma_next * 0.9999)
    sigma_signal = 1.0 - sigma_next
    sigma_residual = math.sqrt(max(sigma_next * sigma_next - sigma_up * sigma_up, 0.0))
    alpha_ratio = sigma_signal + sigma_residual
    sigma_down = sigma_residual / alpha_ratio
    return np.float32(alpha_ratio), np.float32(sigma_down), np.float32(sigma_up)


def sde_step(
    sample: np.ndarray,
    denoised: np.ndarray,
    sigma: np.float32,
    sigma_next: np.float32,
    noise: np.ndarray,
) -> np.ndarray:
    if float(sigma_next) == 0.0:
        return denoised.astype(np.float32)
    alpha_ratio, sigma_down, sigma_up = sde_coeffs(float(sigma_next))
    if float(sigma_up) == 0.0:
        return denoised.astype(np.float32)
    eps_next = f32(f32(sample - denoised) * np.float32(1.0 / (float(sigma) - float(sigma_next))))
    denoised_next = f32(sample - np.float32(sigma) * eps_next)
    inner = f32(denoised_next + sigma_down * eps_next)
    return f32(alpha_ratio * inner + sigma_up * noise)


def res2s_first_step(
    x: np.ndarray,
    vel1: np.ndarray,
    vel2: np.ndarray,
    noise_sub: np.ndarray,
    noise_step: np.ndarray,
) -> np.ndarray:
    h, a21, b1, b2, sub_sigma = coeffs(S2_SIGMA, S2_SIGMA_NEXT)
    den1 = denoise_from_vel(x, vel1, S2_SIGMA)
    eps1 = f32(den1 - x)
    x_mid = f32(x + np.float32(h * a21) * eps1)
    x_mid = sde_step(x, x_mid, S2_SIGMA, sub_sigma, noise_sub)

    anchor = x.astype(np.float32).copy()
    if h < 0.5 and float(S2_SIGMA) > 0.03:
        eps1_bong = f32(den1 - x)
        w = np.float32(h * a21)
        for _ in range(BONG_ITERS):
            anchor = f32(x_mid - w * eps1_bong)
            eps1_bong = f32(den1 - anchor)

    den2 = denoise_from_vel(x_mid, vel2, sub_sigma)
    e1 = f32(den1 - anchor)
    e2 = f32(den2 - anchor)
    x_next = f32(anchor + np.float32(h * b1) * e1 + np.float32(h * b2) * e2)
    return sde_step(anchor, x_next, S2_SIGMA, S2_SIGMA_NEXT, noise_step)


def main() -> None:
    spatial_in = read_bin(UP_DIR / "spatial_in.bin", VIDEO_S1)
    spatial_out = read_bin(UP_DIR / "spatial_out.bin", VIDEO_S2)

    # Stage-1 final denoise: x0 = x - velocity * terminal_sigma.  Choose x and
    # velocity so the denoised output is exactly the spatial-upscaler oracle input.
    stage1_video_vel = pattern(VIDEO_S1, 0.031, 0.4)
    stage1_video_x = f32(spatial_in + RES2S_TERMINAL_SIGMA * stage1_video_vel)

    audio_stage1_final = pattern(AUDIO, 0.22, 1.7)
    stage1_audio_vel = pattern(AUDIO, 0.047, 2.1)
    stage1_audio_x = f32(audio_stage1_final + RES2S_TERMINAL_SIGMA * stage1_audio_vel)

    video_init_noise = pattern(VIDEO_S2, 0.61, 3.0)
    audio_init_noise = pattern(AUDIO, 0.58, 4.0)
    video_stage2_x = f32(video_init_noise * S2_SIGMA + spatial_out * np.float32(1.0 - S2_SIGMA))
    audio_stage2_x = f32(audio_init_noise * S2_SIGMA + audio_stage1_final * np.float32(1.0 - S2_SIGMA))

    video_vel1 = pattern(VIDEO_S2, 0.083, 5.0)
    video_vel2 = pattern(VIDEO_S2, 0.071, 6.0)
    audio_vel1 = pattern(AUDIO, 0.067, 7.0)
    audio_vel2 = pattern(AUDIO, 0.059, 8.0)
    video_noise_sub = pattern(VIDEO_S2, 0.93, 9.0)
    video_noise_step = pattern(VIDEO_S2, 0.87, 10.0)
    audio_noise_sub = pattern(AUDIO, 0.79, 11.0)
    audio_noise_step = pattern(AUDIO, 0.73, 12.0)

    video_stage2_next = res2s_first_step(video_stage2_x, video_vel1, video_vel2, video_noise_sub, video_noise_step)
    audio_stage2_next = res2s_first_step(audio_stage2_x, audio_vel1, audio_vel2, audio_noise_sub, audio_noise_step)

    tensors = {
        "stage1_video_x": stage1_video_x,
        "stage1_video_vel": stage1_video_vel,
        "stage1_video_final_ref": spatial_in,
        "stage1_audio_x": stage1_audio_x,
        "stage1_audio_vel": stage1_audio_vel,
        "stage1_audio_final_ref": audio_stage1_final,
        "stage2_video_upscaled_ref": spatial_out,
        "stage2_video_init_noise": video_init_noise,
        "stage2_audio_init_noise": audio_init_noise,
        "stage2_video_x_ref": video_stage2_x,
        "stage2_audio_x_ref": audio_stage2_x,
        "stage2_video_vel1": video_vel1,
        "stage2_video_vel2": video_vel2,
        "stage2_audio_vel1": audio_vel1,
        "stage2_audio_vel2": audio_vel2,
        "stage2_video_noise_sub": video_noise_sub,
        "stage2_video_noise_step": video_noise_step,
        "stage2_audio_noise_sub": audio_noise_sub,
        "stage2_audio_noise_step": audio_noise_step,
        "stage2_video_next_ref": video_stage2_next,
        "stage2_audio_next_ref": audio_stage2_next,
    }
    for name, value in tensors.items():
        write_bin(name, value)

    print(f"wrote {len(tensors)} staged phase oracle tensors -> {OUT}")
    print(f"video stage2 next shape={video_stage2_next.shape} audio stage2 next shape={audio_stage2_next.shape}")


if __name__ == "__main__":
    main()
