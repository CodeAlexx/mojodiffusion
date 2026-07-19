#!/usr/bin/env python3
"""Synthetic paired cache for the LTX-2 IC-LoRA / v2v SMOKE (P5 u3 + tail).

CPU-only, no GPU, no Mojo compile. Writes a tiny musubi-native dataset the
train_ltx2_av.mojo v2v arms can read end-to-end:
  <root>/cache/sampleN_ltx2.safetensors      target latent  (geometry key)
  <root>/cache/sampleN_ltx2_te.safetensors   video_prompt_embeds_bfloat16 [1024,4096]
                                             + prompt_attention_mask [1024] i64 all-ones
  <root>/ref_cache/sampleN_ltx2.safetensors  clean reference latent (downscaled key)

Two geometries (matched to the trainer dispatch in train_ltx2_av.mojo):
  image512  _run_geometry[320, 1,16,16, 64,1,8,8, 2]
            target latents_1x16x16_bfloat16 [128,1,16,16] + ref latents_1x8x8_bfloat16 [128,1,8,8]
  video512  _run_geometry[608, 4,9,16, 32,1,4,8, 2]
            target latents_4x9x16_bfloat16 [128,4,9,16] + ref latents_1x4x8_bfloat16 [128,1,4,8]

Contract pinned from train_ltx2_av.mojo: LAT_KEY = the geometry's target key
(C=128), ENC_KEY video_prompt_embeds_bfloat16 (N_TXT=1024, VD=4096),
assert_prompt_mask_all_ones needs prompt_attention_mask I64 all-ones; the ref
grid is discovered via v2v_cache.discover_ref_latent_key (any single latents_*),
paired by same-basename in the reference_cache_directory.

Non-degenerate (randn): finite loss + nonzero LoRA grads. LIVENESS smoke, not a
parity gate. Video runs exercise the first-frame Bernoulli when the trainer is
given --ltx2_first_frame_conditioning_p 0.1.

Run:  /home/alex/serenityflow-v2/.venv/bin/python scripts/ltx2_v2v_smoke_cache.py \
          [--geometry image512|video512] [--root DIR] [--n 2]
"""
import argparse
import os

import torch
from safetensors.torch import save_file

C, N_TXT, VD = 128, 1024, 4096
META = {"architecture": "ltx2", "format_version": "1.0.1"}

# (target shape, target key, ref shape, ref key) per geometry — trainer-matched.
GEOM = {
    "image512": (
        (C, 1, 16, 16), "latents_1x16x16_bfloat16",
        (C, 1, 8, 8), "latents_1x8x8_bfloat16",
    ),
    "video512": (
        (C, 4, 9, 16), "latents_4x9x16_bfloat16",
        (C, 1, 4, 8), "latents_1x4x8_bfloat16",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="image512", choices=list(GEOM.keys()))
    ap.add_argument("--root", default=None)
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()

    tgt_shape, tgt_key, ref_shape, ref_key = GEOM[args.geometry]
    root = args.root or ("/tmp/ltx2_v2v_smoke" if args.geometry == "image512"
                         else "/tmp/ltx2_v2v_smoke_" + args.geometry)
    cache = os.path.join(root, "cache")
    ref = os.path.join(root, "ref_cache")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(ref, exist_ok=True)
    g = torch.Generator().manual_seed(1234)

    for i in range(args.n):
        stem = f"sample{i}_ltx2"
        tgt = torch.randn(*tgt_shape, generator=g).to(torch.bfloat16)      # target latent
        save_file({tgt_key: tgt}, os.path.join(cache, stem + ".safetensors"), metadata=META)
        emb = (torch.randn(N_TXT, VD, generator=g) * 0.1).to(torch.bfloat16)
        mask = torch.ones(N_TXT, dtype=torch.int64)
        save_file({"video_prompt_embeds_bfloat16": emb, "prompt_attention_mask": mask},
                  os.path.join(cache, stem + "_te.safetensors"), metadata=META)
        rlat = torch.randn(*ref_shape, generator=g).to(torch.bfloat16)     # CLEAN ref latent
        save_file({ref_key: rlat}, os.path.join(ref, stem + ".safetensors"), metadata=META)

    print(f"[{args.geometry}] wrote {args.n} samples -> {cache} + {ref}")
    print(f"  target {tgt_key} {tuple(tgt_shape)} | ref {ref_key} {tuple(ref_shape)}")


if __name__ == "__main__":
    main()
