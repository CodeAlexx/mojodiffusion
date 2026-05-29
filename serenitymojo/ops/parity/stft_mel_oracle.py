#!/usr/bin/env python3
"""P-stft oracle: dump real STFT/mel bases + a fixed test signal + the
reference mel produced by the exact LTX-2 `compute_mel` algorithm.

This is the binding numeric gate for serenitymojo/ops/stft.mojo. It reproduces
`LTX2VocoderWithBWE::compute_mel` (inference-flame/src/vae/ltx2_vocoder.rs:1018-1047)
in torch on the REAL checkpoint bases, on a FIXED 16 kHz stereo sine, and writes
everything the Mojo smoke needs to load IDENTICAL inputs and compare against an
identical reference:

  forward_basis  [514, 1, 512]  (bf16-on-disk -> stored bf16, dumped f32)
  mel_basis      [64, 257]      (bf16-on-disk -> stored bf16, dumped f32)
  audio          [1, 2, T]      fixed stereo sine (f32)
  mel_ref        [1, 2, 64, Tf] reference log-mel (f32)

Algorithm (matches Rust exactly):
  flat        = audio.reshape(B*C, T)
  win_length  = forward_basis.shape[2] = 512
  left_pad    = win_length - hop_length = 512 - 80 = 432
  flat_padded = flat.unsqueeze(1).pad1d(left_pad, 0)      # ZERO pad, left only
  spec        = conv1d(flat_padded, forward_basis, stride=hop)   # [B*C,514,Tf]
  n_freqs     = 514 // 2 = 257
  real        = spec[:, 0:257]
  imag        = spec[:, 257:514]
  magnitude   = sqrt(real^2 + imag^2)                    # [B*C,257,Tf]
  mel         = (magnitude^T @ mel_basis^T)^T            # [B*C,64,Tf]
  mel         = mel.clamp(1e-5, 1e10).log()
  mel         = mel.reshape(B, C, 64, Tf)

Compute is done in BF16 (matching Rust `to_dtype(BF16)` on the bases and the
vocoder running in bf16) so the Mojo BF16 path can hit cos >= 0.999.

Usage:
    python3 serenitymojo/ops/parity/stft_mel_oracle.py
Writes to: serenitymojo/ops/parity/stft_mel_oracle.safetensors
"""
from __future__ import annotations
import math
import os
import sys
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stft_mel_oracle.safetensors")

HOP = 80
SR = 16000
T = 2400  # 30 frames at hop 80 after the 432 left-pad; small, GPU-light


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with safe_open(CKPT, framework="pt") as f:
        forward_basis = f.get_tensor("vocoder.mel_stft.stft_fn.forward_basis")
        mel_basis = f.get_tensor("vocoder.mel_stft.mel_basis")
    # Match Rust: bases consumed as BF16.
    forward_basis = forward_basis.to(device=dev, dtype=torch.bfloat16)
    mel_basis = mel_basis.to(device=dev, dtype=torch.bfloat16)
    print(f"forward_basis {tuple(forward_basis.shape)} {forward_basis.dtype}")
    print(f"mel_basis     {tuple(mel_basis.shape)} {mel_basis.dtype}")

    win_length = forward_basis.shape[2]   # 512
    n_freqs = forward_basis.shape[0] // 2  # 257
    n_mels = mel_basis.shape[0]            # 64
    assert win_length == 512 and n_freqs == 257 and n_mels == 64

    # Fixed stereo sine [1, 2, T]: L = 220 Hz, R = 440 Hz, amplitude 0.5.
    t = torch.arange(T, dtype=torch.float64)
    left = 0.5 * torch.sin(2 * math.pi * 220.0 * t / SR)
    right = 0.5 * torch.sin(2 * math.pi * 440.0 * t / SR)
    audio = torch.stack([left, right], dim=0).unsqueeze(0)  # [1,2,T]
    audio_bf16 = audio.to(device=dev, dtype=torch.bfloat16)

    B, C, Tlen = audio_bf16.shape
    flat = audio_bf16.reshape(B * C, Tlen)
    left_pad = win_length - HOP  # 432
    flat_padded = F.pad(flat.unsqueeze(1), (left_pad, 0), mode="constant", value=0.0)
    spec = F.conv1d(flat_padded, forward_basis, bias=None, stride=HOP)
    Tf = spec.shape[2]
    real = spec[:, 0:n_freqs]
    imag = spec[:, n_freqs:2 * n_freqs]
    magnitude = (real * real + imag * imag).sqrt()           # [B*C,257,Tf]

    magnitude_t = magnitude.permute(0, 2, 1)                  # [B*C,Tf,257]
    mel_basis_t = mel_basis.permute(1, 0)                     # [257,64]
    mel = magnitude_t.matmul(mel_basis_t).permute(0, 2, 1)    # [B*C,64,Tf]
    mel = mel.clamp(1e-5, 1e10).log()
    mel = mel.reshape(B, C, n_mels, Tf)

    print(f"T={T} Tf={Tf} -> mel {tuple(mel.shape)}")
    mf = mel.float()
    print(f"mel mean={mf.mean():.4f} std={mf.std():.4f} "
          f"min={mf.min():.4f} max={mf.max():.4f}")

    tensors = {
        # f32 for Mojo Tensor.from_host (which takes host F32 and casts to bf16)
        "forward_basis": forward_basis.float().cpu().contiguous(),
        "mel_basis": mel_basis.float().cpu().contiguous(),
        "audio": audio.float().cpu().contiguous(),       # [1,2,T]
        "mel_ref": mel.float().cpu().contiguous(),       # [1,2,64,Tf]
    }
    save_file(tensors, OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
