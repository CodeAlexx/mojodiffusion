#!/usr/bin/env python3
"""Synthetic TRI-PAIR cache for the LTX-2 AV DRIVER smoke (P6.2).

CPU-only. Writes a torchref-native tri-pair the AV arm reads end-to-end:
  <root>/cache/sampleN_ltx2.safetensors        video latents_{F}x{H}x{W}_bfloat16 [128,F,H,W]
  <root>/cache/sampleN_ltx2_te.safetensors      video_prompt_embeds_bfloat16 [N_TXT,4096]
                                                + audio_prompt_embeds_bfloat16 [N_TXT,2048]
                                                + prompt_attention_mask [N_TXT] i64 all-ones
  <root>/cache/sampleN_ltx2_audio.safetensors   audio_latents_{T}x{mel}x{C}_bfloat16 [C,T,mel]
                                                + audio_lengths_int32 scalar

Format pinned from torchref ltx2_cache_latents.py:473-490 (audio latents stored
[C,T,mel]; key {time_steps}x{mel_bins}x{channels}; audio_lengths int32; the audio
patchify in_features = C*mel, S_A tokens = T). Metadata architecture ltx2_v1.

Emits >=1 sample with audio_length < T so the loss length-mask path is non-trivial
(lead pin). Non-degenerate randn. LIVENESS smoke, not a parity gate.

Run: /home/alex/serenityflow-v2/.venv/bin/python scripts/ltx2_av_smoke_cache.py \
        [--root DIR] [--n 2]
"""
import argparse
import os

import torch
from safetensors.torch import save_file

C, N_TXT, VD, AD = 128, 1024, 4096, 2048
# audio geometry: S_A = T = 16 tokens; patch_in = C_a*mel = 8*16 = 128 (matches
# audio_patchify_proj.in_features).
AC, AT, AMEL = 8, 16, 16
VIDEO_SHAPE = (C, 4, 9, 16)          # [C, F, H, W] -> S_V = 4*9*16 = 576
VIDEO_KEY = "latents_4x9x16_bfloat16"
VMETA = {"architecture": "ltx2", "format_version": "1.0.1"}
AMETA = {"architecture": "ltx2_v1", "format_version": "1.0.1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/ltx2_av_smoke")
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()

    cache = os.path.join(args.root, "cache")
    os.makedirs(cache, exist_ok=True)
    g = torch.Generator().manual_seed(6202)

    for i in range(args.n):
        stem = f"sample{i}_ltx2"
        # video latent
        vlat = torch.randn(*VIDEO_SHAPE, generator=g).to(torch.bfloat16)
        save_file({VIDEO_KEY: vlat}, os.path.join(cache, stem + ".safetensors"), metadata=VMETA)
        # tri-pair text: video + audio prompt embeds + mask
        vemb = (torch.randn(N_TXT, VD, generator=g) * 0.1).to(torch.bfloat16)
        aemb = (torch.randn(N_TXT, AD, generator=g) * 0.1).to(torch.bfloat16)
        mask = torch.ones(N_TXT, dtype=torch.int64)
        save_file(
            {"video_prompt_embeds_bfloat16": vemb,
             "audio_prompt_embeds_bfloat16": aemb,
             "prompt_attention_mask": mask},
            os.path.join(cache, stem + "_te.safetensors"), metadata=VMETA)
        # audio latent [C,T,mel] + length (sample 0 padded: length < T)
        alat = torch.randn(AC, AT, AMEL, generator=g).to(torch.bfloat16)
        alen = AT - 4 if i == 0 else AT       # sample0 has 4 padded time steps
        akey = f"audio_latents_{AT}x{AMEL}x{AC}_bfloat16"
        # audio file = video stem with _ltx2 -> _ltx2_audio (torchref
        # ltx2_cache_latents.py:260), i.e. sampleN_ltx2_audio.safetensors.
        save_file(
            {akey: alat, "audio_lengths_int32": torch.tensor(alen, dtype=torch.int32)},
            os.path.join(cache, f"sample{i}_ltx2_audio.safetensors"), metadata=AMETA)

    print(f"[av-smoke] wrote {args.n} tri-pairs -> {cache}")
    print(f"  video {VIDEO_KEY} {VIDEO_SHAPE} | audio audio_latents_{AT}x{AMEL}x{AC}_bfloat16 "
          f"[{AC},{AT},{AMEL}] (S_A={AT}, patch_in={AC*AMEL})")
    print(f"  sample0 audio_length={AT-4} < T={AT} (non-trivial length-mask)")


if __name__ == "__main__":
    main()
