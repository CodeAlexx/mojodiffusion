#!/usr/bin/env python3
"""FIRST REAL-WEIGHT parity oracle for MiniMax-H3 — the audio VAE.

Units 10 (BigVGAN decoder) and 15 (DAC encoder + causal-attention projection)
were gated on tiny random-weight fixtures because no checkpoint existed. This
runs the reference on the ACTUAL released audio VAE at its ACTUAL config, so
the Mojo port can be diffed against it at full width for the first time.

Weights come from the released file directly and are NOT copied into the
reference output — the Mojo gate opens the same file. The only thing written
here is the input waveform and the reference's intermediates, a few hundred KB
instead of 577 MiB.

WHAT CHANGES AT FULL SCALE, and why a tiny fixture cannot have caught it:
  * encoder_rates (2, 4, 4, 5, 5) — hop 800, five stages, three of them with
    ODD strides whose `k=2*stride, padding=ceil(stride/2)` geometry is not
    L/stride. The fixture had two stages.
  * 172 real weight_norm pairs to fold, against six in the fixture.
  * the causal attention runs 2048 -> 32 with 8 heads, so head_dim is 256 and
    the adaptive average pool is 256 -> 32, an exact 8:1 divide. The fixture
    deliberately used an UNEVEN 6 -> 8 to catch a uniform-stride bug; this
    checks the code still agrees where the windows are even.
  * decoder_rates (5, 5, 2, 2, 2, 2, 2) with resblock kernels (3, 7, 11) at
    dilations (1, 3, 5) — 7 stages of 18 convolutions each.

INPUT: 3200 samples, 4 latent frames. Deliberately NOT 2 — `_decode` in the
reference raises on 2 latent frames, a defect unit 10 recorded rather than
worked around.

Usage:
    python3 scripts/minimax_h3_audio_real_weights_oracle.py [path/to/audio_vae.safetensors]
"""

import os
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, "/home/alex/minimax_h3_ref/diffusers-src/src")

from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (  # noqa: E402
    AutoencoderKLMiniMaxH3Audio,
)

DEFAULT_CKPT = "/home/alex/Downloads/MiniMax-H3-audio_vae.safetensors"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_audio"
SAMPLES = 3200


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"checkpoint: {ckpt}")
    print(f"            {os.path.getsize(ckpt)/1024**2:.1f} MiB")

    state = load_file(ckpt)
    model = AutoencoderKLMiniMaxH3Audio().eval()
    missing, unexpected = model.load_state_dict(state, strict=True), None
    print(f"loaded strict=True: {missing}")
    print(f"hop_length {model.hop_length}, sampling_rate "
          f"{model.config.sampling_rate}")

    torch.manual_seed(0x48_33_AA)
    wave = torch.randn(1, 1, SAMPLES, dtype=torch.float32) * 0.1

    tensors = {"in.samples": wave[0, 0].contiguous()}
    with torch.no_grad():
        trunk = model.encoder(wave)
        tensors["out.trunk"] = trunk[0].contiguous()

        pre = model.pre_block(trunk.transpose(1, 2)).transpose(1, 2)
        tensors["out.pre_block"] = pre[0].contiguous()

        mean = model.encode(wave).latent_dist.mode()
        tensors["out.mean"] = mean[0].contiguous()

        decoded = model.decode(mean).sample
        tensors["out.waveform"] = decoded[0, 0].contiguous()

    print()
    print(f"  samples    {tuple(wave.shape)}")
    print(f"  trunk      {tuple(trunk.shape)}")
    print(f"  pre_block  {tuple(pre.shape)}")
    print(f"  mean       {tuple(mean.shape)}   "
          f"range [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  waveform   {tuple(decoded.shape)}   "
          f"range [{decoded.min():.4f}, {decoded.max():.4f}]")

    # Round-trip quality is the reference's own, not the port's — recorded so
    # the Mojo numbers have something to be judged against.
    err = (decoded[0, 0] - wave[0, 0]).abs()
    print(f"  reference round trip: max_abs {err.max():.6f}, "
          f"rms {err.pow(2).mean().sqrt():.6f}")

    save_file(tensors, f"{OUT_DIR}/audio_real_weights_ref.safetensors")
    total = sum(t.numel() * 4 for t in tensors.values())
    print()
    print(f"wrote {OUT_DIR}/audio_real_weights_ref.safetensors ({total/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
