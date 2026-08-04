#!/usr/bin/env python3
"""FIRST REAL-WEIGHT parity oracle for MiniMax-H3 — the audio VAE.

Units 10 (BigVGAN decoder) and 15 (DAC encoder + causal-attention projection)
were gated on tiny random-weight fixtures because no checkpoint existed. This
runs the reference on the ACTUAL released audio VAE at its ACTUAL config, so
the Mojo port can be diffed against it at full width for the first time.

THE REFERENCE PASS RUNS IN FLOAT64 and stores float32 downcasts — the 34a648c
vision-oracle precedent, adopted here after the precision sweep measured WHY it
matters: against the original f32-pass reference the Mojo trunk read 1.79e-4
(90% of the 2e-4 bar) and "passed", but torch's own f32-vs-f64 error at the
trunk is 1.554e-4 — 78% of the bar was the ANCHOR's own accumulation noise,
partially cancelling the port's. Anchored to this f64 pass, the pre-fix port
FAILED the trunk at 2.33e-4 (116% of bar) — a real sequential-f32 accumulation
defect in the encoder's 10240-wide convs, invisible against the noisy anchor —
and the fixed port sits at 7.6e-6 (3.8%). An anchor at truth makes the bar
mean what it says. Bars are unchanged.

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

DECODER SELF-CONSISTENCY: the gate decodes the STORED `out.mean` (float32), so
the decoder references here are computed from that same stored float32 mean
upcast to float64 — not from the float64 mean the encoder pass produced. The
encoder and decoder halves stay independently attributable, exactly as the
gate assumes.

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

# The dev-time copy (/home/alex/Downloads/MiniMax-H3-audio_vae.safetensors) was
# cleaned up; this is the product location the t2va pipeline reads (H3_ROOT in
# pipeline/minimax_h3_t2va.mojo) — same 1086-tensor file.
DEFAULT_CKPT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/audio_vae/model.safetensors"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_audio"
SAMPLES = 3200


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"checkpoint: {ckpt}")
    print(f"            {os.path.getsize(ckpt)/1024**2:.1f} MiB")

    state = load_file(ckpt)
    model = AutoencoderKLMiniMaxH3Audio().double().eval()
    missing, unexpected = (
        model.load_state_dict({k: v.double() for k, v in state.items()}, strict=True),
        None,
    )
    print(f"loaded strict=True (float64 pass): {missing}")
    print(f"hop_length {model.hop_length}, sampling_rate "
          f"{model.config.sampling_rate}")

    # The input stays the FLOAT32 waveform, stored bit-identically — the port
    # reads the same f32 samples; only the reference ARITHMETIC is float64.
    torch.manual_seed(0x48_33_AA)
    wave32 = torch.randn(1, 1, SAMPLES, dtype=torch.float32) * 0.1
    wave = wave32.double()

    tensors = {"in.samples": wave32[0, 0].contiguous()}
    with torch.no_grad():
        trunk = model.encoder(wave)
        tensors["out.trunk"] = trunk[0].float().contiguous()

        pre = model.pre_block(trunk.transpose(1, 2)).transpose(1, 2)
        tensors["out.pre_block"] = pre[0].float().contiguous()

        mean = model.encode(wave).latent_dist.mode()
        tensors["out.mean"] = mean[0].float().contiguous()

        # The decoder is driven from the STORED float32 mean (see header).
        mean_in = tensors["out.mean"].unsqueeze(0).double()
        decoded = model.decode(mean_in).sample
        tensors["out.waveform"] = decoded[0, 0].float().contiguous()

        # ── decoder intermediates, stage by stage ────────────────────────────
        # Re-walks MiniMaxH3AudioBigVGANDecoder.forward using the reference's
        # OWN submodules, so these are the reference's numbers and not a second
        # implementation. Without them the Mojo decoder's staged entry points
        # have nothing to be compared against and the split buys only tidiness.
        dec = model.decoder
        h = model.dec_in_proj(mean_in)
        tensors["dec.in_proj"] = h[0].float().contiguous()
        h = dec.conv_pre(h)
        tensors["dec.pre"] = h[0].float().contiguous()
        print()
        print("  decoder stages:")
        print(f"    dec_in_proj -> {tuple(h.shape)}")
        for i in range(dec.num_upsamples):
            h = dec.ups[i][0](h)
            residual = None
            for j in range(dec.num_kernels):
                block = dec.resblocks[i * dec.num_kernels + j](h)
                residual = block if residual is None else residual + block
            h = residual / dec.num_kernels
            tensors[f"dec.stage{i}"] = h[0].float().contiguous()
            print(f"    stage {i}: rate {dec.ups[i][0].stride[0]:2d} -> "
                  f"[{h.shape[1]:5d}, {h.shape[2]:5d}]")

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
