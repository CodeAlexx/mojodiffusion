"""MiniMax-H3 video VAE geometry oracle: spatial tiling + temporal chunking.

Reference: diffusers PR huggingface/diffusers#14355, cloned at
/home/alex/minimax_h3_ref/diffusers-src, head e1b518df.
Module: src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py
  AutoencoderKLMiniMaxH3._split_tiles, ._blend, ._encode, ._decode

This dumps the arithmetic that decides WHICH pixels and WHICH latent frames get
processed — not the convolutions. It is the part that silently corrupts output
when a port gets it subtly wrong, and it needs no weights.

Method: the model is instantiated with the RELEASED temporal and spatial
factors (so `spatial_compression_ratio` 16, `temporal_compression_ratio` 4,
`clip_length` 17, `token_drop` 3, and every derived constant are the real ones)
but with tiny channel counts and a 1-layer decoder, then `_encode_clip` and
`_decode_clip` are replaced by shape-correct zero producers. The reference's own
control flow therefore runs unmodified and decides every boundary; only the
convolutions are stubbed out.

Run:
    python3 scripts/minimax_h3_vae_geometry_oracle.py
Writes: output/minimax_h3_vae_geometry/vae_geometry_ref.safetensors
"""

import json
import os
import sys

DIFFUSERS_SRC = "/home/alex/minimax_h3_ref/diffusers-src/src"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_vae_geometry"

sys.path.insert(0, DIFFUSERS_SRC)

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3  # noqa: E402


def build_geometry_model() -> AutoencoderKLMiniMaxH3:
    """Real geometry, negligible weights."""
    model = AutoencoderKLMiniMaxH3(
        in_channels=3,
        out_channels=3,
        latent_channels=24,
        block_out_channels=(32, 32, 32, 32, 32, 32),  # GroupNorm needs channels % 32 == 0
        layers_per_block=1,
        spatial_downsample_factors=(2, 2, 2, 2, 1, 1),
        temporal_downsample_factors=(1, 2, 2, 1, 1, 1),
        decoder_num_layers=1,
        decoder_num_attention_heads=1,
        decoder_attention_head_dim=8,
        clip_length=17,
        token_drop=3,
    )
    ratio = model.spatial_compression_ratio
    temporal_ratio = model.temporal_compression_ratio

    def fake_encode_clip(x: torch.Tensor) -> torch.Tensor:
        frames = x.shape[2]
        # One latent frame per `temporal_ratio` pixel frames, ceil — the causal
        # encoder's own output length for a clip.
        latent_frames = -(-frames // temporal_ratio)
        return torch.zeros(
            x.shape[0], 48, latent_frames, x.shape[3] // ratio, x.shape[4] // ratio, dtype=x.dtype
        )

    def fake_decode_clip(z: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            z.shape[0],
            3,
            z.shape[2] * temporal_ratio,
            z.shape[3] * ratio,
            z.shape[4] * ratio,
            dtype=z.dtype,
        )

    model._encode_clip = fake_encode_clip
    model._decode_clip = fake_decode_clip
    return model


def main() -> None:
    model = build_geometry_model()
    tensors: dict[str, torch.Tensor] = {}

    meta = {
        "spatial_compression_ratio": model.spatial_compression_ratio,
        "temporal_compression_ratio": model.temporal_compression_ratio,
        "frame_pre_padding": model.frame_pre_padding,
        "tokens_chunk_size": model.tokens_chunk_size,
        "token_overlap": model.token_overlap,
        "frame_overlap": model.frame_overlap,
        "tile_sample_min_height": model.tile_sample_min_height,
        "tile_sample_min_overlap_height": model.tile_sample_min_overlap_height,
    }
    tensors["derived"] = torch.tensor(
        [
            model.spatial_compression_ratio,
            model.temporal_compression_ratio,
            model.frame_pre_padding,
            model.tokens_chunk_size,
            model.token_overlap,
            model.frame_overlap,
        ],
        dtype=torch.int64,
    )

    # 1. _split_tiles over a sweep of frame sizes, at the released 256/64 geometry
    lengths = [64, 128, 256, 257, 320, 384, 512, 768, 1024, 1088, 1280, 1344, 1920, 2048]
    starts, sizes, overlaps, counts = [], [], [], []
    for length in lengths:
        s, l, o = model._split_tiles(length, 256, 64)
        counts.append(len(s))
        starts += s
        sizes += l
        overlaps += o
    tensors["split.lengths"] = torch.tensor(lengths, dtype=torch.int64)
    tensors["split.num_tiles"] = torch.tensor(counts, dtype=torch.int64)
    tensors["split.starts"] = torch.tensor(starts, dtype=torch.int64)
    tensors["split.sizes"] = torch.tensor(sizes, dtype=torch.int64)
    tensors["split.overlaps"] = torch.tensor(overlaps, dtype=torch.int64)

    # a second tile geometry, to prove the port is not fitted to 256/64
    starts2, sizes2, overlaps2, counts2 = [], [], [], []
    for length in lengths:
        s, l, o = model._split_tiles(length, 384, 96)
        counts2.append(len(s))
        starts2 += s
        sizes2 += l
        overlaps2 += o
    tensors["split384.num_tiles"] = torch.tensor(counts2, dtype=torch.int64)
    tensors["split384.starts"] = torch.tensor(starts2, dtype=torch.int64)
    tensors["split384.sizes"] = torch.tensor(sizes2, dtype=torch.int64)
    tensors["split384.overlaps"] = torch.tensor(overlaps2, dtype=torch.int64)

    # 2. the linear blend weights, whose extent clamps to the shorter side
    blend_cases = [(16, 16, 8), (16, 16, 16), (8, 16, 12), (4, 4, 1), (32, 8, 20)]
    blend_a, blend_b, blend_out = [], [], []
    for a_len, b_len, extent in blend_cases:
        a = torch.arange(a_len, dtype=torch.float32).view(1, 1, 1, a_len, 1)
        b = torch.arange(b_len, dtype=torch.float32).view(1, 1, 1, b_len, 1) + 100.0
        out = model._blend(a, b, extent, dim=-2)
        blend_a.append(a_len)
        blend_b.append(b_len)
        blend_out += out.flatten().tolist()
    tensors["blend.cases"] = torch.tensor(
        [[c[0], c[1], c[2]] for c in blend_cases], dtype=torch.int64
    )
    tensors["blend.out"] = torch.tensor(blend_out, dtype=torch.float32)

    # 3. encode / decode temporal plans over a sweep of legal frame counts.
    # 5 pixel frames encode to 2 latent frames, which the reference CANNOT decode
    # (see the DECODE FLOOR note below), so it is encode-only here.
    frame_counts = [5, 22, 39, 56, 124, 141, 260, 362]
    latent_counts, decoded_counts = [], []
    for frames in frame_counts:
        x = torch.zeros(1, 3, frames, 64, 64)
        moments = model._encode(x)
        latent_counts.append(moments.shape[2])
        if moments.shape[2] < 3:
            decoded_counts.append(-1)  # reference raises; recorded as a divergence
            continue
        z = torch.zeros(1, 24, moments.shape[2], 4, 4)
        decoded_counts.append(model._decode(z).shape[2])
    tensors["temporal.frames"] = torch.tensor(frame_counts, dtype=torch.int64)
    tensors["temporal.latent_frames"] = torch.tensor(latent_counts, dtype=torch.int64)
    tensors["temporal.decoded_frames"] = torch.tensor(decoded_counts, dtype=torch.int64)

    # 4. decode from an arbitrary latent length, including counts that are not a
    # whole number of chunks (the padding path).
    #
    # DECODE FLOOR: `latent_frames == 2` yields `num_chunks == 0` in the
    # reference — `num_tokens = 2 + 3 = 5`, `pad_tokens = 0`,
    # `num_chunks = 5 // 5 - 1 = 0` — so its `torch.cat(decoded_chunks)` gets an
    # empty list and raises. That is exactly `video_latent_num_frames(5)`, the
    # 5-frame minimum. Every length >= 3 is fine. The ComfyUI implementation
    # guards this case explicitly ("too few tokens for one chunk (e.g.
    # T_lat == 2): pad one extra chunk"); the diffusers draft does not. The
    # sweep therefore starts at 3 and the port raises a clear error at 2 rather
    # than inventing behaviour the vendor has not published.
    latent_lengths = list(range(3, 40))
    from_latent = []
    for length in latent_lengths:
        z = torch.zeros(1, 24, length, 4, 4)
        from_latent.append(model._decode(z).shape[2])
    tensors["decode.latent_lengths"] = torch.tensor(latent_lengths, dtype=torch.int64)
    tensors["decode.frames"] = torch.tensor(from_latent, dtype=torch.int64)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "vae_geometry_ref.safetensors")
    save_file(tensors, path, metadata={"format": "pt"})
    with open(os.path.join(OUT_DIR, "vae_geometry_ref.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {len(tensors)} tensors -> {path}")
    print("derived geometry:", json.dumps(meta))
    print("frames -> latents -> decoded:")
    for f, l, d in zip(frame_counts, latent_counts, decoded_counts):
        print(f"  {f:>4} -> {l:>3} -> {d:>4}")


if __name__ == "__main__":
    main()
