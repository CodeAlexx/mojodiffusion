#!/usr/bin/env python3
"""Produce LTX2 inpaint mask caches (P5.5 unit 2) from mask images/videos.

CPU-only. Mirrors the official LTX-2 preprocessing
(LTX-2-upstream/packages/ltx-trainer/scripts/process_videos.py:905-988):
grayscale -> resize to pixel dims -> avg_pool2d(VAE_SPATIAL_FACTOR=32) spatial
downsample -> temporal amax over VAE_TEMPORAL_FACTOR=8 pixel-frame groups ->
binarize >0.5. Output is a per-sample safetensors `{"mask": [F,H,W] f32}` in
the mask_cache_dir under the SAME basename as the target latent cache (the route
train_ltx2_av.mojo's --mask_cache_dir / _pair_mask_cache reads).

The trainer's Mojo loader (training/ltx2/mask_cache.mojo) re-thresholds >0.5 and
reshapes frame->h->w to the token mask, so the binarize here is the load-time
match; F/H/W is the TARGET latent grid (read from the latent cache key
`latents_{F}x{H}x{W}_bfloat16`, or given via --frames/--height/--width).

Run:
  # single: mask image -> cache for a 4x9x16 target
  python scripts/ltx2_make_inpaint_mask_cache.py --mask m.png --out ref_ci.safetensors \
      --frames 4 --height 9 --width 16
  # batch: one mask per target latent in a cache dir (same basename)
  python scripts/ltx2_make_inpaint_mask_cache.py --mask_dir masks/ \
      --latent_cache_dir cache/ --out_dir mask_cache/
"""
import argparse
import glob
import os
import re

import torch
import torch.nn.functional as F
from safetensors.torch import save_file, safe_open

VAE_SPATIAL_FACTOR = 32
VAE_TEMPORAL_FACTOR = 8
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_LAT_KEY_RE = re.compile(r"^latents_(\d+)x(\d+)x(\d+)_")


def _target_grid_from_latent(path):
    with safe_open(path, framework="pt") as st:
        for k in st.keys():
            m = _LAT_KEY_RE.match(k)
            if m and not k.startswith("latents_clean_"):
                return int(m.group(1)), int(m.group(2)), int(m.group(3))
    raise ValueError(f"no latents_FxHxW key in {path}")


def _load_mask_pixels(mask_file, pixel_f, pixel_h, pixel_w):
    ext = os.path.splitext(mask_file)[1].lower()
    if ext in IMAGE_EXT:
        from PIL import Image
        img = torch.from_numpy(
            __import__("numpy").asarray(Image.open(mask_file).convert("L"), dtype="float32") / 255.0
        )[None, None]  # [1,1,H,W]
        img = F.interpolate(img, size=(pixel_h, pixel_w), mode="nearest")[0, 0]  # [H,W]
        return img.expand(pixel_f, -1, -1)  # tile across frames [F,H,W]
    import imageio.v3 as iio
    vid = iio.imread(mask_file)  # [F,H,W,C] or [H,W,C]
    t = torch.from_numpy(vid).float() / 255.0
    if t.ndim == 3:
        t = t[None]
    frames = t[:pixel_f].mean(dim=-1)  # grayscale [F,H,W]
    frames = F.interpolate(frames[:, None], size=(pixel_h, pixel_w), mode="nearest")[:, 0]
    return frames


def make_mask_latent(mask_file, latent_f, latent_h, latent_w):
    pixel_h = latent_h * VAE_SPATIAL_FACTOR
    pixel_w = latent_w * VAE_SPATIAL_FACTOR
    pixel_f = (latent_f - 1) * VAE_TEMPORAL_FACTOR + 1
    mask_pixels = _load_mask_pixels(mask_file, pixel_f, pixel_h, pixel_w)  # [F,H,W]
    # spatial avg_pool2d(32) -> [F, H', W']
    ml = F.avg_pool2d(mask_pixels.unsqueeze(1), kernel_size=VAE_SPATIAL_FACTOR).squeeze(1)
    # temporal amax over groups of 8 (any masked pixel-frame masks the latent frame)
    fsp = ml.shape[0]
    pad = (VAE_TEMPORAL_FACTOR - fsp % VAE_TEMPORAL_FACTOR) % VAE_TEMPORAL_FACTOR
    if pad:
        ml = F.pad(ml, (0, 0, 0, 0, 0, pad))
    hp, wp = ml.shape[1], ml.shape[2]
    ml = ml.reshape(-1, VAE_TEMPORAL_FACTOR, hp, wp).amax(dim=1)[:latent_f]
    return (ml > 0.5).float().contiguous()   # [F,H',W'] binary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask")
    ap.add_argument("--out")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--latent_cache")   # read F/H/W from this latent cache instead
    ap.add_argument("--mask_dir")       # batch
    ap.add_argument("--latent_cache_dir")
    ap.add_argument("--out_dir")
    args = ap.parse_args()

    if args.mask_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        n = 0
        for lat in sorted(glob.glob(os.path.join(args.latent_cache_dir, "*_ltx2.safetensors"))):
            base = os.path.basename(lat)
            if base.endswith(("_te.safetensors", "_audio.safetensors", "_dino.safetensors")):
                continue
            stem = base[: -len("_ltx2.safetensors")]
            masks = [m for e in IMAGE_EXT | {".mp4", ".webm", ".mov"}
                     for m in glob.glob(os.path.join(args.mask_dir, stem + e))]
            if not masks:
                print(f"  no mask for {stem}, skip")
                continue
            f, h, w = _target_grid_from_latent(lat)
            ml = make_mask_latent(masks[0], f, h, w)
            save_file({"mask": ml}, os.path.join(args.out_dir, base),
                      metadata={"architecture": "ltx2", "format_version": "1.0.1"})
            n += 1
        print(f"wrote {n} mask caches -> {args.out_dir}")
        return

    if args.latent_cache:
        f, h, w = _target_grid_from_latent(args.latent_cache)
    else:
        f, h, w = args.frames, args.height, args.width
    ml = make_mask_latent(args.mask, f, h, w)
    save_file({"mask": ml}, args.out, metadata={"architecture": "ltx2", "format_version": "1.0.1"})
    print(f"wrote mask cache {tuple(ml.shape)} (binary) -> {args.out}")


if __name__ == "__main__":
    main()
