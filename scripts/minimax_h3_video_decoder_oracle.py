"""MiniMax-H3 video ViT decoder oracle, on a tiny random-weight model.

Reference: diffusers PR huggingface/diffusers#14355, cloned at
/home/alex/minimax_h3_ref/diffusers-src, head e1b518df.
  models/autoencoders/autoencoder_kl_minimax_h3.py
    MiniMaxH3VideoViTDecoder3d.forward      :447-498
    MiniMaxH3VideoTransformerBlock.forward  :387-395
    MiniMaxH3VideoAttnProcessor.__call__    :303-340
    MiniMaxH3VideoRotaryPosEmbed.forward    :293-296

The last math surface between latents and pixels. Unusually for a video VAE
this decoder is a TRANSFORMER, not a convolutional stack: every latent voxel is
one token, `num_register_tokens` learned registers plus one all-zero token are
appended, full self-attention runs over the lot, and the suffix is dropped again
before `proj_out` expands each token into a `patch_t x patch x patch` pixel
block.

Four things the oracle is built to pin down, all of which produce
plausible-looking output when wrong:
  * position ids are length-normalized to `[-1, 1)` per axis — `2*(arange(size)
    + 0.5)/size - 1` — and the suffix tokens sit at position ZERO, not at the
    end of the grid;
  * the rotary angles carry a `2*pi` scale and `theta` is 100.0, not 10000.0;
  * q/k RMSNorm is `elementwise_affine=False`, i.e. NO weight, and the reference
    runs it in float32 regardless of compute dtype;
  * the residual branches are scaled by learned per-channel `scale1` / `scale2`
    vectors rather than by adaLN.

`post_quant_conv` is included so the gate covers the whole latent-to-pixel path.

Run:
    python3 scripts/minimax_h3_video_decoder_oracle.py
Writes: output/minimax_h3_video/video_decoder_ref.safetensors
"""

import json
import os
import sys

DIFFUSERS_SRC = "/home/alex/minimax_h3_ref/diffusers-src/src"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_video"

sys.path.insert(0, DIFFUSERS_SRC)

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from diffusers import AutoencoderKLMiniMaxH3  # noqa: E402

INIT = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "block_out_channels": (8, 16),
    "layers_per_block": 1,
    "spatial_downsample_factors": (2, 2),
    "temporal_downsample_factors": (2, 2),
    "norm_num_groups": 8,
    "decoder_num_layers": 2,
    "decoder_num_attention_heads": 2,
    "decoder_attention_head_dim": 8,
    "decoder_num_register_tokens": 2,
    "decoder_ffn_mult": 2,
    "clip_length": 17,
    "token_drop": 3,
    "latents_mean": (0.0,) * 4,
    "latents_std": (1.0,) * 4,
}

LATENT_FRAMES = 2
LATENT_HEIGHT = 3
LATENT_WIDTH = 3


def main() -> None:
    torch.manual_seed(0)
    model = AutoencoderKLMiniMaxH3(**INIT).eval().to(torch.float32)

    # Re-randomize: `register_tokens` and both residual `scale` vectors are
    # zero-initialized, which would make the whole transformer a no-op.
    generator = torch.Generator().manual_seed(999)
    with torch.no_grad():
        for _, parameter in model.named_parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * 0.2
            )

    gen = torch.Generator().manual_seed(21)
    latents = torch.randn(
        1, INIT["latent_channels"], LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH, generator=gen
    )

    with torch.no_grad():
        after_quant = model.post_quant_conv(latents)
        pixels = model.decoder(after_quant)

    tensors: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if name.startswith("decoder.") or name.startswith("post_quant_conv."):
            tensors[f"w.{name}"] = parameter.detach().clone()

    tensors["in.latents"] = latents
    tensors["mid.after_quant"] = after_quant
    tensors["out.pixels"] = pixels

    meta = {
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in INIT.items()},
        "latent_frames": LATENT_FRAMES,
        "latent_height": LATENT_HEIGHT,
        "latent_width": LATENT_WIDTH,
        "spatial_compression_ratio": model.spatial_compression_ratio,
        "temporal_compression_ratio": model.temporal_compression_ratio,
        "decoder_dim": INIT["decoder_num_attention_heads"] * INIT["decoder_attention_head_dim"],
        "rope_dim": int(INIT["decoder_attention_head_dim"] * 0.75),
        "pixels_shape": list(pixels.shape),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "video_decoder_ref.safetensors")
    save_file(tensors, path, metadata={"format": "pt"})
    with open(os.path.join(OUT_DIR, "video_decoder_ref.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {len(tensors)} tensors -> {path}")
    print(f"spatial ratio {meta['spatial_compression_ratio']}  "
          f"temporal ratio {meta['temporal_compression_ratio']}  "
          f"decoder dim {meta['decoder_dim']}  rope dim {meta['rope_dim']}")
    print(f"latents {list(latents.shape)} -> pixels {list(pixels.shape)}")
    print(f"pixels mean {pixels.mean():.6f}  min {pixels.min():.6f}  max {pixels.max():.6f}")
    print("\ndecoder tensors:")
    for name in sorted(n for n in tensors if n.startswith("w.")):
        print(f"  {name:<58} {list(tensors[name].shape)}")


if __name__ == "__main__":
    main()
