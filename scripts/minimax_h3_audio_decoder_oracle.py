"""MiniMax-H3 audio decoder oracle: BigVGAN on a tiny random-weight model.

Reference: diffusers PR huggingface/diffusers#14355, cloned at
/home/alex/minimax_h3_ref/diffusers-src, head e1b518df.
  models/autoencoders/autoencoder_kl_minimax_h3_audio.py
    AutoencoderKLMiniMaxH3Audio.decode -> dec_in_proj + MiniMaxH3AudioBigVGANDecoder

The decoder is what GENERATION needs — the encoder only runs for reference
audio, and is a later unit. Config is the reference's own test fixture.

Two things make this gateable without the real checkpoint:

  * `weight_norm` stores `weight_g` / `weight_v`, and the effective weight is
    `g * v / ||v||` with the norm taken over every dim but the first. The
    oracle dumps BOTH the parametrization AND the folded weight, so the port can
    be gated on doing the fold itself rather than being handed the answer.
  * the Kaiser-window resampling filters of the anti-aliased activations are
    registered BUFFERS, so they ship in the checkpoint and are dumped here.
    Porting `kaiser_window`/`sinc` is therefore unnecessary — and reproducing
    them would be a worse gate than loading the tensor the model was trained
    with.

Run:
    python3 scripts/minimax_h3_audio_decoder_oracle.py
Writes: output/minimax_h3_audio/audio_decoder_ref.safetensors
"""

import json
import os
import sys

DIFFUSERS_SRC = "/home/alex/minimax_h3_ref/diffusers-src/src"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_audio"

sys.path.insert(0, DIFFUSERS_SRC)

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from diffusers import AutoencoderKLMiniMaxH3Audio  # noqa: E402

INIT = {
    "encoder_dim": 4,
    "encoder_rates": (2, 2),
    "latent_dim": 32,
    "latent_channels": 8,
    "num_attention_heads": 2,
    "decoder_dim": 16,
    "decoder_rates": (2, 2),
    "decoder_kernel_sizes": (4, 4),
    "resblock_kernel_sizes": (3, 7),
    "resblock_dilation_sizes": ((1, 3), (1, 3)),
    "sampling_rate": 32000,
    "latents_mean": [0.0] * 8,
    "latents_std": [1.0] * 8,
}

NUM_LATENTS = 6


def main() -> None:
    torch.manual_seed(0)
    model = AutoencoderKLMiniMaxH3Audio(**INIT).eval().to(torch.float32)

    # Re-randomize everything: Snake alphas default to ones and SnakeBeta
    # log-params to zeros, both of which would make the activation nearly an
    # identity and hide ordering errors.
    generator = torch.Generator().manual_seed(4321)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * 0.2
            )

    gen = torch.Generator().manual_seed(11)
    latents = torch.randn(1, INIT["latent_channels"], NUM_LATENTS, generator=gen)

    with torch.no_grad():
        decoded = model.decode(latents).sample

    tensors: dict[str, torch.Tensor] = {}

    # Parametrized weights, as stored, plus the folded result the port must
    # reproduce for itself.
    folded = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight_g") and hasattr(module, "weight_v"):
            tensors[f"wn.{name}.weight_g"] = module.weight_g.detach().clone()
            tensors[f"wn.{name}.weight_v"] = module.weight_v.detach().clone()
            tensors[f"wn.{name}.weight"] = module.weight.detach().clone()
            folded += 1

    for name, parameter in model.named_parameters():
        tensors[f"w.{name}"] = parameter.detach().clone()
    for name, buffer in model.named_buffers():
        tensors[f"b.{name}"] = buffer.detach().clone()

    tensors["in.latents"] = latents
    tensors["out.sample"] = decoded

    meta = {
        **{k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in INIT.items()},
        "num_latents": NUM_LATENTS,
        "hop_length": 4,
        "decoded_shape": list(decoded.shape),
        "weight_norm_modules": folded,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "audio_decoder_ref.safetensors")
    save_file(tensors, path, metadata={"format": "pt"})
    with open(os.path.join(OUT_DIR, "audio_decoder_ref.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {len(tensors)} tensors -> {path}")
    print(f"weight_norm modules: {folded}")
    print(f"latents {list(latents.shape)} -> decoded {list(decoded.shape)}")
    print(f"decoded mean {decoded.mean():.6f}  min {decoded.min():.6f}  max {decoded.max():.6f}")
    print("\ndecoder tensors:")
    for name in sorted(n for n in tensors if ".decoder." in n or "dec_in_proj" in n):
        print(f"  {name:<62} {list(tensors[name].shape)}")


if __name__ == "__main__":
    main()
