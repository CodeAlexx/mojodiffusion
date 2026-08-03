"""MiniMax-H3 video encoder oracle: the 3D causal CNN, on a tiny random model.

Reference: diffusers PR huggingface/diffusers#14355, cloned at
/home/alex/minimax_h3_ref/diffusers-src, head e1b518df.
  models/autoencoders/autoencoder_kl_minimax_h3.py
    MiniMaxH3VideoEncoder3d.forward     :271-276
    MiniMaxH3VideoResnetBlock3d.forward :118-126
    MiniMaxH3VideoDownsample3d.forward  :156-159
    MiniMaxH3VideoCausalConv3d.forward  :57-65
    MiniMaxH3VideoGroupNorm.forward     :74-80

Needed for keyframes and video references, not for text-to-video — so this is
gated after the decoders rather than before them.

Three paddings, all different, all easy to conflate:
  * SPATIAL padding is symmetric and REFLECTING;
  * TEMPORAL padding is causal — `temporal_padding` zero frames prepended and
    nothing appended;
  * the strided downsampler adds an ASYMMETRIC bottom/right reflect pad of 1
    BEFORE its convolution, which itself carries no spatial padding.

And the normalization is per-FRAME: `MiniMaxH3VideoGroupNorm` folds time into
batch so statistics never mix across frames. A plain GroupNorm over the whole
clip produces output of exactly the right shape.

Run:
    python3 scripts/minimax_h3_video_encoder_oracle.py
Writes: output/minimax_h3_video/video_encoder_ref.safetensors
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

# Small enough to gate by hand, large enough that both downsample levels and the
# causal temporal pad actually do something.
FRAMES = 5
HEIGHT = 8
WIDTH = 8


def main() -> None:
    torch.manual_seed(0)
    model = AutoencoderKLMiniMaxH3(**INIT).eval().to(torch.float32)

    generator = torch.Generator().manual_seed(555)
    with torch.no_grad():
        for _, parameter in model.named_parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * 0.2
            )

    gen = torch.Generator().manual_seed(31)
    pixels = torch.randn(1, INIT["in_channels"], FRAMES, HEIGHT, WIDTH, generator=gen)

    with torch.no_grad():
        # `_encode_clip` without tiling is exactly `quant_conv(encoder(x))`; call
        # the parts so the intermediates can be dumped and a failure localized.
        encoded = model.encoder(pixels)
        moments = model.quant_conv(encoded)

    tensors: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if name.startswith("encoder.") or name.startswith("quant_conv."):
            tensors[f"w.{name}"] = parameter.detach().clone()

    tensors["in.pixels"] = pixels
    tensors["mid.encoded"] = encoded
    tensors["out.moments"] = moments

    meta = {
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in INIT.items()},
        "frames": FRAMES,
        "height": HEIGHT,
        "width": WIDTH,
        "encoded_shape": list(encoded.shape),
        "moments_shape": list(moments.shape),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "video_encoder_ref.safetensors")
    save_file(tensors, path, metadata={"format": "pt"})
    with open(os.path.join(OUT_DIR, "video_encoder_ref.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {len(tensors)} tensors -> {path}")
    print(f"pixels {list(pixels.shape)} -> encoded {list(encoded.shape)} -> moments {list(moments.shape)}")
    print(f"moments mean {moments.mean():.6f}  min {moments.min():.6f}  max {moments.max():.6f}")
    print("\nencoder tensors:")
    for name in sorted(n for n in tensors if n.startswith("w.")):
        print(f"  {name:<56} {list(tensors[name].shape)}")


if __name__ == "__main__":
    main()
