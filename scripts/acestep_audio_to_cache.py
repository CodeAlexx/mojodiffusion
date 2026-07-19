#!/usr/bin/env python3
"""ACE-Step-1.5 raw audio → Mojo training cache (Tier-3 T3.C real-audio preprocess).

Chains the two halves of the offline preprocess into the pure-Mojo trainer's cache:

  1. AUDIO FILE → .pt   — upstream `acestep/training_v2/preprocess.py::
     preprocess_audio_files` (two-pass): Oobleck VAE encode → target_latents;
     Qwen3-Embedding text+lyric encode; DIT encoder → encoder_hidden_states +
     context_latents; masks. Writes one final `.pt` per audio file.
  2. .pt → Mojo cache   — this repo's `scripts/acestep_pt_to_cache.py` (--pt-dir):
     per-sample BF16 safetensors + manifest.txt (the format
     `serenitymojo/models/acestep/acestep_cache_reader.mojo` streams).

Then train:  ACESTEP_CACHE=<cache_out> output/bin/train_acestep   (or the UI
`acestep` preset with the cache set).

VERIFIED end-to-end (2026-07-14):
  * PASS-1 on a REAL 5s WAV → target_latents [125,64] (Oobleck VAE) + text/lyric
    Qwen3 encodes [1,256,1024]/[1,512,1024]. Needs the torchaudio→soundfile patch
    below (torch 2.12's torchaudio delegates decode to torchcodec, which fails to
    load against FFmpeg 6 — a torch-2.12/torchcodec version incompat, NOT missing
    FFmpeg).
  * .pt → Mojo cache → train: upstream-format .pt (make_test_fixtures, 3 samples)
    → converter --pt-dir → cache → streamed training (Σ|B| grows).

ENV REQUIREMENT for the FULL run (pass-2):
  Pass-2's DIT-encoder load (`AutoModel.from_pretrained`) hits "Tensor.item()
  cannot be called on meta tensors" under torch 2.12 (the FSQ AudioTokenizer
  meta-init bug the parity oracle also documented). Run this in an env with
  ACE-Step's PINNED torch (~2.10) — where both torchcodec AND AutoModel work —
  or apply the oracle's direct-build workaround (build AceStepDiTModel directly,
  gen_acestep_train_oracle.py). Models (all under $CKPT): vae/ (Oobleck),
  Qwen3-Embedding-0.6B, acestep-v15-xl-base (DIT encoder).

Usage:
  python acestep_audio_to_cache.py --audio-dir <dir> --cache-out <dir> \
      [--checkpoint-dir DIR] [--variant xl_base] [--dataset-json captions.json] \
      [--max-duration 240] [--no-audio-patch]
"""
from __future__ import annotations
import argparse
import glob
import os
import subprocess
import sys

ACE_ROOT = "/home/alex/ACE-Step-1.5"
DEFAULT_CKPT = f"{ACE_ROOT}/checkpoints"
CONVERTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "acestep_pt_to_cache.py")


def _patch_torchaudio_load() -> None:
    """torch 2.12's torchaudio.load → torchcodec (fails vs FFmpeg 6). Route decode
    through soundfile (verified to work): (path) → (audio [C,N] float32, sr)."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    def _sf_load(path, *a, **k):
        data, sr = sf.read(path, dtype="float32", always_2d=True)   # [N, C]
        return torch.from_numpy(np.ascontiguousarray(data.T)), sr   # [C, N]

    torchaudio.load = _sf_load


def main() -> int:
    ap = argparse.ArgumentParser(description="ACE-Step raw audio → Mojo training cache")
    ap.add_argument("--audio-dir", required=True, help="dir of audio files (wav/flac/mp3/…)")
    ap.add_argument("--cache-out", required=True, help="output Mojo cache dir")
    ap.add_argument("--pt-dir", default=None, help="intermediate .pt dir (default <cache-out>_pt)")
    ap.add_argument("--checkpoint-dir", default=DEFAULT_CKPT)
    ap.add_argument("--variant", default="xl_base")
    ap.add_argument("--dataset-json", default=None, help="captions/lyrics metadata json (optional)")
    ap.add_argument("--max-duration", type=float, default=240.0)
    ap.add_argument("--no-audio-patch", action="store_true",
                    help="skip the torchaudio→soundfile patch (use if torchcodec works)")
    a = ap.parse_args()

    pt_dir = a.pt_dir or (a.cache_out.rstrip("/") + "_pt")
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(a.cache_out, exist_ok=True)

    sys.path.insert(0, ACE_ROOT)
    if not a.no_audio_patch:
        _patch_torchaudio_load()
        print("[env] patched torchaudio.load → soundfile (torchcodec workaround)")

    # ── 1) audio → .pt (upstream two-pass) ──
    from acestep.training_v2.preprocess import preprocess_audio_files
    print(f"[1/2] audio → .pt   {a.audio_dir} → {pt_dir}")
    res = preprocess_audio_files(
        audio_dir=a.audio_dir, output_dir=pt_dir, checkpoint_dir=a.checkpoint_dir,
        variant=a.variant, max_duration=a.max_duration, dataset_json=a.dataset_json,
        device="cuda", precision="bf16",
    )
    print("   ", res)
    pts = [p for p in glob.glob(os.path.join(pt_dir, "*.pt")) if not p.endswith(".tmp.pt")]
    if not pts:
        raise SystemExit(
            "no final .pt produced. Pass-1 (VAE+text) works with the audio patch; if pass-2 "
            "failed with 'Tensor.item() cannot be called on meta tensors', run in ACE-Step's "
            "pinned-torch env (see the module docstring)."
        )
    print(f"   {len(pts)} final .pt")

    # ── 2) .pt → Mojo cache (this repo's converter) ──
    print(f"[2/2] .pt → Mojo cache   {pt_dir} → {a.cache_out}")
    subprocess.run([sys.executable, CONVERTER, "--pt-dir", pt_dir, "--out", a.cache_out], check=True)
    print(f"DONE. Train with:  ACESTEP_CACHE={a.cache_out} output/bin/train_acestep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
