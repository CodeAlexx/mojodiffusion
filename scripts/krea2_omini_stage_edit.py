#!/usr/bin/env python3
# krea2_omini_stage_edit.py — stage A for the krea2 OminiControl EDIT vertical.
#
# Mirrors serenitymojo/training/krea2_stage_images.py (same bucketing, same
# [-1,1] CHW f32 tensors, same sample_NNNNN record names krea2_prepare_cache's
# _stage_name reads) but stages the OminiControl CONDITION-BRANCH data model
# (ominicontrol_krea2_intake.md §1.1/§3.6): ONE record per sample holding BOTH
# the target image and the condition image, plus per-sample position metadata.
# This is NOT the existing --edit ref-dir layout in krea2_stage_images.py (that
# one writes references to <out>/ref for the klein-style channel/ref cache);
# the Omini cache prepare will read both tensors from the SAME record and
# VAE-encode each through the identical path.
#
# Dataset convention (dev set /home/alex/datasets/qwen_edit_test, 96 triplets):
#   <stem>.jpg            target (edited result)
#   <stem>-condlabel.png  condition (source image)
#   <stem>.txt            raw edit instruction
#
# Per kept sample i -> three files in <out_dir>:
#   sample_NNNNN.safetensors   keys:
#       "image"       [1,3,H,W] f32 [-1,1] CHW  (target; the loss/velocity image)
#       "cond_image"  [1,3,H,W] f32 [-1,1] CHW  (condition; SAME bucket as target
#                     — EDIT overlaps the target spatially, so the cond grid must
#                     equal the img grid for the delta-[0,0] position map)
#   sample_NNNNN.txt           raw instruction text, stripped (the Mojo prepare
#                              wraps it in the krea2 template itself — do NOT
#                              pre-wrap, same rule as krea2_stage_images.py)
#   sample_NNNNN.json          sidecar the Omini prepare/reader consume:
#       {"position_delta": [0, 0], "position_scale": 1.0, "condition_type": "edit"}
#       EDIT condition type: delta [0,0] + scale 1.0 (condition tokens SHARE
#       positions with the image tokens, flux_omini train_spatial_alignment.py:44);
#       kept per-sample so subject ([0, -w/16]) and OC2 compact-token (scale 2.0)
#       staging reuse this script later with only a metadata change.
#
# --uncond: same contract as krea2_stage_images.py — write uncond.txt = "" for
# the drop_text_prob=0.1 empty-caption render. (Condition-image drop — the black
# cond latent for drop_image_prob=0.1 — is Stage B's job: encode a black image
# once, store as a constant. Intake §3.6.)
#
# AUTHORED, NOT YET RUN — do not treat staged output as verified.
#
# Run (default bucket 512 — the OminiControl training resolution). Needs
# numpy+PIL+safetensors; the krea2_stage_images.py-cited serenityflow-v2 venv
# does not exist on this box — the ai-toolkit venv does (verified importable):
#   /home/alex/ai-toolkit/venv/bin/python \
#     scripts/krea2_omini_stage_edit.py \
#     /home/alex/datasets/qwen_edit_test /home/alex/trainings/krea2_omini_edit_stage 512

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import save_file

# vae_scale_factor (8) * patch (2) — the krea2 bucket divisibility
# (krea2_stage_images.py KREA2_BUCKET_DIV; krea2.py:166-167).
KREA2_BUCKET_DIV = 16

# Caption-length guard, verbatim rationale from krea2_stage_images.py: the
# Qwen3-VL encoder in krea2_prepare_cache fails loud above 2048 SDPA tokens
# MID-RUN; catch overlong captions here where re-staging is seconds. Measured
# 2.81 chars/token on the JSON-shaped caption style -> 5000 chars ~ 1780 tokens
# (conservative). The prepare's own fail-loud remains the authority.
KREA2_MAX_CAPTION_CHARS = 5000

# EDIT condition type (OminiControl spatial family): condition overlaps the
# target spatially — position delta [0,0], scale 1.0 (intake §1.2/§2).
EDIT_POSITION_DELTA = [0, 0]
EDIT_POSITION_SCALE = 1.0


def _parse_bucket(text: str) -> tuple[int, int]:
    text = text.lower().strip()
    if "x" not in text:
        size = int(text)
        return size, size
    w_s, h_s = text.split("x", 1)
    return int(w_s), int(h_s)


def _parse_args(argv: list[str]):
    emit_uncond = "--uncond" in argv
    args = [a for a in argv if a != "--uncond"]
    buckets_arg = None
    if "--buckets" in args:
        idx = args.index("--buckets")
        if idx + 1 >= len(args):
            raise SystemExit("--buckets requires a comma-separated WxH list")
        buckets_arg = args[idx + 1]
        del args[idx : idx + 2]
    max_caption_chars = KREA2_MAX_CAPTION_CHARS
    if "--max-caption-chars" in args:
        idx = args.index("--max-caption-chars")
        if idx + 1 >= len(args):
            raise SystemExit("--max-caption-chars requires an integer")
        max_caption_chars = int(args[idx + 1])
        del args[idx : idx + 2]
    if len(args) < 2:
        print(
            "usage: krea2_omini_stage_edit.py <src_dir> <out_dir> [SIZE|WxH] "
            "[--buckets WxH,WxH,...] [--uncond] "
            f"[--max-caption-chars N (default {KREA2_MAX_CAPTION_CHARS}, 0=off)]\n"
            "  OminiControl EDIT staging: per <stem>.txt, target=<stem>.jpg, "
            "condition=<stem>-condlabel.png; one record with image + cond_image "
            "+ a position-metadata json sidecar (delta [0,0], scale 1.0)."
        )
        sys.exit(1)
    src = Path(args[0])
    out = Path(args[1])
    if buckets_arg:
        buckets = [_parse_bucket(x) for x in buckets_arg.split(",") if x.strip()]
    else:
        # Default 512: the OminiControl training resolution (intake §1.8).
        buckets = [_parse_bucket(args[2] if len(args) > 2 else "512")]
    for w, h in buckets:
        if w % KREA2_BUCKET_DIV != 0 or h % KREA2_BUCKET_DIV != 0:
            raise SystemExit(
                f"bucket {w}x{h} must be divisible by {KREA2_BUCKET_DIV} "
                f"(vae_scale_factor*patch)"
            )
    return src, out, buckets, emit_uncond, max_caption_chars


def _choose_bucket(width: int, height: int, buckets: list[tuple[int, int]]) -> tuple[int, int]:
    if len(buckets) == 1:
        return buckets[0]
    ar = width / height
    area = width * height

    def score(bucket: tuple[int, int]) -> float:
        bw, bh = bucket
        return abs(math.log(ar / (bw / bh))) + 0.15 * abs(math.log(area / (bw * bh)))

    return min(buckets, key=score)


def _center_crop_to_aspect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    w, h = img.size
    target_ar = target_w / target_h
    ar = w / h
    if ar > target_ar:
        new_w = int(round(h * target_ar))
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    new_h = int(round(w / target_ar))
    top = (h - new_h) // 2
    return img.crop((0, top, w, top + new_h))


def _decode_to_bucket(path: Path, bucket_w: int, bucket_h: int) -> np.ndarray:
    """Center-crop to the bucket aspect, resize to bucket, -> [1,3,H,W] f32 [-1,1]."""
    img = Image.open(path).convert("RGB")
    img = _center_crop_to_aspect(img, bucket_w, bucket_h)
    img = img.resize((bucket_w, bucket_h), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0   # [H,W,3] in [-1,1]
    arr = arr.transpose(2, 0, 1)[None]                       # [1,3,H,W]
    return np.ascontiguousarray(arr)


def _caption_too_long(caption: str, limit: int) -> bool:
    return limit > 0 and len(caption) > limit


def _report_skipped_captions(skipped: list, limit: int):
    if not skipped:
        return
    print(
        f"\nSKIPPED {len(skipped)} sample(s): caption over --max-caption-chars "
        f"{limit} (the Qwen3-VL encoder in krea2_prepare_cache fails loud past "
        f"2048 tokens, mid-run):"
    )
    for name, n_chars in skipped:
        print(f"  {name}: {n_chars} chars")
    print(
        "  Shorten those captions and re-stage to include the samples, or raise "
        "--max-caption-chars if you believe they tokenize under 2048.\n"
    )


def _stage_name(i: int) -> str:
    """The record name krea2_prepare_cache reads (_stage_name, 5-wide zero pad)."""
    return f"sample_{i:05d}"


def main():
    src, out, buckets, emit_uncond, max_caption_chars = _parse_args(sys.argv[1:])
    out.mkdir(parents=True, exist_ok=True)
    for pattern in ("sample_*.safetensors", "sample_*.txt", "sample_*.json"):
        for old in out.glob(pattern):
            old.unlink()

    stems = sorted(
        p.stem
        for p in src.iterdir()
        if p.suffix.lower() == ".txt" and not p.stem.endswith("-condlabel")
    )
    bucket_counts = Counter()
    skipped_captions = []
    kept = 0
    for stem in stems:
        tgt = src / f"{stem}.jpg"
        cond = src / f"{stem}-condlabel.png"
        cap = src / f"{stem}.txt"
        if not tgt.exists():
            print(f"skip {stem}: no target {stem}.jpg")
            continue
        if not cond.exists():
            print(f"skip {stem}: no condition {stem}-condlabel.png")
            continue
        caption = cap.read_text().strip()
        if _caption_too_long(caption, max_caption_chars):
            skipped_captions.append((cap.name, len(caption)))
            continue
        # Bucket chosen from the TARGET; the condition is forced to the SAME
        # bucket — EDIT overlaps the target spatially (delta [0,0]) so the cond
        # latent grid must equal the target grid (S_COND == S_IMG).
        w, h = Image.open(tgt).size
        bucket_w, bucket_h = _choose_bucket(w, h, buckets)
        name = _stage_name(kept)
        save_file(
            {
                "image": _decode_to_bucket(tgt, bucket_w, bucket_h),
                "cond_image": _decode_to_bucket(cond, bucket_w, bucket_h),
            },
            str(out / f"{name}.safetensors"),
        )
        (out / f"{name}.txt").write_text(caption)
        (out / f"{name}.json").write_text(
            json.dumps(
                {
                    "position_delta": EDIT_POSITION_DELTA,
                    "position_scale": EDIT_POSITION_SCALE,
                    "condition_type": "edit",
                },
                indent=None,
            )
        )
        bucket_counts[(bucket_w, bucket_h)] += 1
        kept += 1

    if kept == 0:
        raise SystemExit(
            f"no <stem>.txt + <stem>.jpg + <stem>-condlabel.png triplets in {src}"
        )
    _report_skipped_captions(skipped_captions, max_caption_chars)

    if emit_uncond:
        # Empty caption: the Mojo prepare wraps it as PREFIX + "" + SUFFIX.
        (out / "uncond.txt").write_text("")
        print(f"staged uncond.txt (empty-caption render) -> {out}")
    summary = ", ".join(
        f"{w}x{h}:{count}" for (w, h), count in sorted(bucket_counts.items())
    )
    print(
        f"staged {kept} OMINI-EDIT samples -> {out} "
        f"(buckets {summary}, bucket-div {KREA2_BUCKET_DIV}; keys image + "
        f"cond_image per record, position sidecar delta {EDIT_POSITION_DELTA} "
        f"scale {EDIT_POSITION_SCALE})"
    )


if __name__ == "__main__":
    main()
