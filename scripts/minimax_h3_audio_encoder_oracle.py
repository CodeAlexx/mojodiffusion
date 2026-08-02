#!/usr/bin/env python3
"""Oracle for MiniMax-H3 unit 15 — the audio encoder (DAC + causal attention).

Runs the reference's own `AutoencoderKLMiniMaxH3Audio.encode` from the pinned
diffusers clone on a tiny config, and writes the state dict, the input waveform,
and every intermediate the Mojo gate checks.

EVERY PARAMETER IS RE-RANDOMIZED. Three of them default to values that make a
gate vacuous:

  * `Snake1d.alpha` is ones, so `x + sin(x)^2 / (1+1e-9)` — a wrong per-channel
    broadcast still passes because every channel has the same alpha.
  * `q_bias` and `v_bias` are zeros, so the whole "assemble the bias as
    cat(q_bias, ZERO, v_bias)" trap is invisible.
  * LayerNorm weight is ones and bias zeros, so norm2 and mlp.norm are
    indistinguishable from each other.

`zero_k_bias` is deliberately LEFT AT ZERO — it is a frozen buffer, not a
parameter, and randomizing it would gate behaviour the checkpoint cannot
produce.

The config is small but structurally complete: rates (2, 5) exercise both an
exact halving and the awkward stride-5 geometry (k=10, padding=3), the residual
units run at all three dilations 1/3/9, and latent_dim 24 -> latent_channels 8
with 4 heads makes head_dim 6, so the adaptive average pool runs 6 -> 8 on
genuinely UNEVEN overlapping windows (lengths 1,2,2,1,1,2,2,1) rather than a
clean divide, which is the only arrangement that can tell the real pooling rule
apart from a uniform stride.

Writes: output/minimax_h3_audio/audio_encoder_ref.safetensors
"""

import math
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, "/home/alex/minimax_h3_ref/diffusers-src/src")

from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (  # noqa: E402
    AutoencoderKLMiniMaxH3Audio,
)

OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_audio"

ENCODER_DIM = 4
ENCODER_RATES = (2, 5)
LATENT_DIM = 24
LATENT_CHANNELS = 8
NUM_HEADS = 4


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(0x48_33_15)

    model = AutoencoderKLMiniMaxH3Audio(
        encoder_dim=ENCODER_DIM,
        encoder_rates=ENCODER_RATES,
        latent_dim=LATENT_DIM,
        latent_channels=LATENT_CHANNELS,
        num_attention_heads=NUM_HEADS,
        decoder_dim=8,
        decoder_rates=(5, 2),
        decoder_kernel_sizes=(9, 4),
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1, 3),),
        sampling_rate=32000,
    ).eval()

    hop = math.prod(ENCODER_RATES)
    print(f"hop_length {hop}, head_dim {LATENT_DIM // NUM_HEADS} -> out {LATENT_CHANNELS}")

    # ── re-randomize everything the defaults would leave uniform ─────────────
    randomized = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not name.startswith(("encoder.", "pre_block.", "mean_proj.")):
                continue
            p.copy_(torch.randn_like(p) * 0.5 + (0.9 if name.endswith("alpha") else 0.0))
            randomized += 1
        # zero_k_bias stays zero: it is a frozen buffer, and the point of the
        # trap is that the key bias must NOT be loaded as a parameter.
        assert bool((model.pre_block.attn.zero_k_bias == 0).all())
    print(f"re-randomized {randomized} encoder-side parameters (zero_k_bias left at 0)")

    # snake alpha must stay positive-ish: alpha + 1e-9 is inverted
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("alpha") and name.startswith("encoder."):
                p.copy_(p.abs() + 0.3)

    # ── input: not a multiple of hop, so the right-pad is exercised ──────────
    samples = 47
    wave = torch.randn(1, 1, samples, dtype=torch.float32)
    print(f"input {samples} samples -> padded {math.ceil(samples/hop)*hop} -> "
          f"{math.ceil(samples/hop)*hop//hop} latent frames")

    tensors = {"in.samples": wave[0, 0].contiguous()}

    with torch.no_grad():
        # intermediates, so a failure lands on a stage instead of on the end
        padded = torch.nn.functional.pad(
            wave, (0, math.ceil(samples / hop) * hop - samples)
        )
        trunk = model.encoder(padded)
        tensors["out.trunk"] = trunk[0].contiguous()

        pre = model.pre_block(trunk.transpose(1, 2)).transpose(1, 2)
        tensors["out.pre_block"] = pre[0].contiguous()

        posterior = model.encode(wave).latent_dist
        mode = posterior.mode()
        tensors["out.mean"] = mode[0].contiguous()

    print(f"trunk      {tuple(trunk.shape)}")
    print(f"pre_block  {tuple(pre.shape)}")
    print(f"mean       {tuple(mode.shape)}")

    # ── the state dict, encoder side only ────────────────────────────────────
    kept = 0
    for name, tensor in model.state_dict().items():
        if not name.startswith(("encoder.", "pre_block.", "mean_proj.")):
            continue
        if name.endswith("zero_k_bias"):
            continue  # a frozen zero buffer; the port must not read it
        tensors["w." + name] = tensor.detach().float().contiguous()
        kept += 1
    print(f"wrote {kept} weight tensors")

    save_file(tensors, f"{OUT_DIR}/audio_encoder_ref.safetensors")
    print(f"wrote {OUT_DIR}/audio_encoder_ref.safetensors")

    # the names the Mojo side must ask for, so a rename in the reference shows
    # up here rather than as a missing-key error inside the gate
    names = sorted(n[2:] for n in tensors if n.startswith("w."))
    open(f"{OUT_DIR}/audio_encoder_keys.txt", "w").write("\n".join(names) + "\n")
    print(f"wrote {OUT_DIR}/audio_encoder_keys.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
