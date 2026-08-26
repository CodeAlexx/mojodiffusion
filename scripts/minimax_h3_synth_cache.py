#!/usr/bin/env python3
# minimax_h3_synth_cache.py — write SYNTHETIC mmh3 training cache items that
# satisfy the gated Mojo reader contract (h3_train_cache.mojo header; reader
# gated 36/36 bit-exact vs the upstream akane writer fixture). The akane
# musubi-h3 clone this repo's fixture imported is no longer on disk, and the
# pinned kohya h3-temporal-stretch branch writes a DIFFERENT layout generation
# (latents_audio_32x2x*, [32,2,A]) — incompatible with the reader. For smoke
# and step-gate items the reader contract itself is the format authority.
#
# Usage:
#   python minimax_h3_synth_cache.py OUT_DIR image  N   # [24,1,30,52] items
#   python minimax_h3_synth_cache.py OUT_DIR av256  N   # 256x448 124f AV items
import os
import sys

import torch
from safetensors.torch import save_file

out_dir = sys.argv[1]
kind = sys.argv[2]
count = int(sys.argv[3])
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(7)

for n in range(count):
    base = f"synth_{kind}_{n:03d}"
    if kind == "image":
        w, h = 832, 480
        lat = {"latents_1x30x52_bfloat16": torch.randn(24, 1, 30, 52).to(torch.bfloat16)}
        tokens = 87
    elif kind == "av256":
        # smallest legal AV clip at the bring-up geometry: 124 pixel frames ->
        # 37 video latent frames, 207 audio latents, 16x28 latent grid.
        w, h = 448, 256
        lat = {
            "latents_37x16x28_bfloat16": torch.randn(24, 37, 16, 28).to(torch.bfloat16),
            "latents_audio_2x32x207_bfloat16": torch.randn(2, 32, 207).to(torch.bfloat16),
            "audio_loss_mask": torch.ones(207, dtype=torch.bool),
            # first+last keyframe rows: [2R, RW], R = (16//2)*(28//2), RW = 24*2*2
            "varlen_mmh3_keyframe_video_rows_bfloat16": torch.randn(
                2 * (16 // 2) * (28 // 2), 24 * 2 * 2
            ).to(torch.bfloat16),
        }
        tokens = 87
    else:
        raise SystemExit(f"unknown kind {kind}")

    save_file(lat, os.path.join(out_dir, f"{base}_{w:04d}x{h:04d}_mmh3.safetensors"))
    te = {
        "varlen_mmh3_hidden_states_bfloat16": torch.randn(tokens, 5120).to(torch.bfloat16),
        "varlen_mmh3_token_tags_int64": torch.ones(tokens, dtype=torch.long),
        "mmh3_conditioning_task": torch.tensor(0, dtype=torch.long),  # t2va
        # empty-cond entries so --guidance_scale runs can use these items
        "varlen_mmh3_empty_hidden_states_bfloat16": torch.randn(12, 5120).to(torch.bfloat16),
        "varlen_mmh3_empty_token_tags_int64": torch.ones(12, dtype=torch.long),
    }
    save_file(te, os.path.join(out_dir, f"{base}_mmh3_te.safetensors"))
    print("wrote", base)
