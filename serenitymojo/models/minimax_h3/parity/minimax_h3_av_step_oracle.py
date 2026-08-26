#!/usr/bin/env python3
# minimax_h3_av_step_oracle.py — torch oracle for ONE AV training step's recipe
# math: per-modality sigma shift, bf16 noising, velocity targets, token-balanced
# joint loss, and d_pred for BOTH modalities. Mirrors the pinned Musubi H3
# recipe (musubi-tuner h3-temporal-stretch @ 8191ec1, minimax_h3_train_network
# _shift_noise_amount :565 — identical formula to torchref _shift_unchecked)
# and the torchref bf16 dtype contract already gated in h3_train_sigma.mojo.
#
# The DiT forward is NOT run here: model output is a seeded stand-in tensor.
# Packed-layout/block/stack numerics have their own gates; this dump gates the
# TRAINER's AV wiring of the recipe (x_t / target / loss / d_pred, both
# modalities, with the audio loss mask).
#
# Usage: python minimax_h3_av_step_oracle.py <latent_cache.safetensors> <out.safetensors>
import sys

import torch
from safetensors.torch import load_file, save_file

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
U = 0.4375  # exact in bf16-land; the Mojo gate passes the same scalar


def shift(sigma: float, s: float) -> float:
    return s * sigma / (1.0 + (s - 1.0) * sigma)


def noisy(x0: torch.Tensor, noise: torch.Tensor, sigma: float) -> torch.Tensor:
    # torchref: sigma expanded with .to(dtype=bf16); elementwise f32 compute
    # rounded to bf16 (torch CUDA bf16 semantics — matches h3_noisy_input).
    s = torch.tensor(sigma, dtype=torch.bfloat16)
    one_minus = (1.0 - s.float()).to(torch.bfloat16)
    a = (x0.float() * one_minus.float()).to(torch.bfloat16)
    b = (noise.float() * s.float()).to(torch.bfloat16)
    return (a.float() + b.float()).to(torch.bfloat16)


def main() -> None:
    lat = load_file(sys.argv[1])
    vkey = [k for k in lat if k.startswith("latents_") and not k.startswith("latents_audio")][0]
    akey = [k for k in lat if k.startswith("latents_audio_2x32x")][0]
    x0_v = lat[vkey]                       # [24,F,H,W] bf16
    x0_a = lat[akey]                       # [2,32,T] bf16
    mask_a = lat["audio_loss_mask"]        # [T] bool

    g = torch.Generator().manual_seed(2026)
    noise_v = torch.randn(x0_v.shape, generator=g, dtype=torch.float32).to(torch.bfloat16)
    noise_a = torch.randn(x0_a.shape, generator=g, dtype=torch.float32).to(torch.bfloat16)
    pred_v = torch.randn(x0_v.shape, generator=g, dtype=torch.float32).to(torch.bfloat16)
    pred_a = torch.randn(x0_a.shape, generator=g, dtype=torch.float32).to(torch.bfloat16)

    sigma_v = shift(U, VIDEO_SHIFT)
    sigma_a = shift(U, AUDIO_SHIFT)

    xt_v = noisy(x0_v, noise_v, sigma_v)
    xt_a = noisy(x0_a, noise_a, sigma_a)
    tgt_v = (x0_v.float() - noise_v.float()).to(torch.bfloat16)  # one rounding
    tgt_a = (x0_a.float() - noise_a.float()).to(torch.bfloat16)

    # token-balanced joint loss (training.py _joint_loss balance="token"):
    # bf16 diff, .float().square(), f64 accumulate; mask [T] broadcasts over
    # [2,32,T]'s trailing dim.
    diff_v = (pred_v - tgt_v).float().double()
    sum_v = float(diff_v.square().sum())
    n_v = diff_v.numel()
    diff_a = (pred_a - tgt_a).float().double()
    m = mask_a.view(1, 1, -1).expand_as(diff_a)
    sum_a = float(diff_a.square()[m].sum())
    n_a = int(m.sum())
    w_v = 1.0 if n_v > 0 else 0.0
    w_a = 1.0 if n_a > 0 else 0.0
    denom = w_v * n_v + w_a * n_a
    loss = (w_v * sum_v + w_a * sum_a) / denom

    # d loss / d pred (h3_loss_grad contract): bf16 diff, f32 scale 2*w/denom,
    # one bf16 rounding, exact 0/1 mask multiply.
    def dgrad(pred, tgt, w, mask=None):
        d = (pred - tgt)  # bf16 sub
        s = torch.tensor(2.0 * w / denom, dtype=torch.float32)
        gt = (d.float() * s).to(torch.bfloat16)
        if mask is not None:
            gt = gt * mask.to(torch.bfloat16)
        return gt

    d_pred_v = dgrad(pred_v, tgt_v, w_v)
    d_pred_a = dgrad(pred_a, tgt_a, w_a, mask_a.view(1, 1, -1).expand_as(pred_a))

    save_file(
        {
            "noise_v": noise_v, "noise_a": noise_a,
            "pred_v": pred_v, "pred_a": pred_a,
            "xt_v": xt_v, "xt_a": xt_a,
            "tgt_v": tgt_v, "tgt_a": tgt_a,
            "d_pred_v": d_pred_v, "d_pred_a": d_pred_a,
            "scalars_f64": torch.tensor(
                [U, sigma_v, sigma_a, loss, float(n_v), float(n_a), denom],
                dtype=torch.float64,
            ),
        },
        sys.argv[2],
    )
    print("u", U, "sigma_v", sigma_v, "sigma_a", sigma_a)
    print("loss", loss, "n_v", n_v, "n_a", n_a)


if __name__ == "__main__":
    main()
