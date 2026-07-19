#!/usr/bin/env python3
# krea2_stage_images.py — stage A of the Krea-2 LoRA-training cache prepare.
# (Stage B = serenitymojo/models/krea2/krea2_prepare_cache.mojo: VAE + Qwen3-VL
# encoders through the GATED Mojo encoders.)
#
# No JPEG decoder exists in the Mojo stack yet, so this one-shot stager does the
# image decode + bucketing in Python and hands Mojo a single safetensors of f32
# image tensors + the RAW captions (one .txt per sample). This mirrors
# serenity-trainer/scripts/ideogram4_stage_images.py, with two krea2 deltas:
#
#   1) BUCKET DIVISIBILITY = vae_scale_factor * patch = 8 * 2 = 16
#      (krea2.py:166-167 `vae_scale_factor * patch_size`). A bucket HxW divisible
#      by 16 satisfies it (512x512, 768x1024, 1024x1024, ...). Center-crop to the
#      selected bucket aspect ratio, resize to that bucket.
#   2) NO caption rendering. Krea-2's conditioning is NOT the ideogram4 structured-
#      JSON/chat-template digest. The Mojo prepare tokenizes
#      KREA2_TPL_PREFIX + <raw caption> + KREA2_TPL_SUFFIX itself (krea2_paths.mojo,
#      == ai-toolkit text_encoder.py:26-34 PROMPT_TEMPLATE_ENCODE_PREFIX/SUFFIX),
#      so this stager writes the RAW caption text unchanged. We must NOT pre-wrap it
#      (the DROP_IDX=34 prefix-drop in encode_krea2_stack assumes the exact template).
#
# Per sample (dataset: <dir>/N.jpg + N.txt):
#   image: center-crop to selected aspect, resize HxW, RGB f32 [-1,1] CHW
#          -> image.<i> [1,3,H,W] f32   (the QwenImageVaeEncoder input)
#   caption: the raw .txt, stripped -> prompt.<i>.txt  (the Mojo prepare wraps it)
#
# --uncond (caption dropout, flag-gated, default-off): ALSO write uncond.txt = the
#   empty caption "". The Mojo prepare encodes it through the SAME PREFIX+""+SUFFIX
#   path into the llm_uncond cache tensor; trainers substitute it when the seeded
#   dropout schedule fires (matches ai-toolkit caption_dropout / the ideogram4
#   precedent). Note: for "" the template is just PREFIX+SUFFIX, whose length must
#   still exceed DROP_IDX=34 — it does (the prefix alone is 34 tokens, the suffix
#   adds 5, so LT=5 for the empty render, the same LT_NEG the inference pipeline uses).
#
# Run:
#   /home/alex/serenityflow-v2/.venv/bin/python \
#     serenitymojo/training/krea2_stage_images.py \
#     /home/alex/datasets/<dataset> /home/alex/trainings/krea2_<name>_stage 1024 [--uncond]
#
# Bucketed run:
#   ... krea2_stage_images.py <src> <out> --buckets 1024x1024,768x1024,1024x768 --uncond

import sys
import math
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from safetensors.numpy import save_file

# vae_scale_factor (8) * patch (2) — the krea2 bucket divisibility (krea2.py:166-167).
KREA2_BUCKET_DIV = 16


def _parse_bucket(text: str) -> tuple[int, int]:
    text = text.lower().strip()
    if "x" not in text:
        size = int(text)
        return size, size
    w_s, h_s = text.split("x", 1)
    return int(w_s), int(h_s)


def _parse_args(argv: list[str]):
    emit_uncond = "--uncond" in argv
    # IMAGE-EDIT (task #7): single-dir suffix convention. For each <id>.txt the
    # TARGET (edited result) is <id>.jpg and the REFERENCE (source/condition) is
    # <id>-condlabel.png (both in the SAME dir). Reference staged tensors go to
    # <out>/ref/images.safetensors (image.<i> at the SAME index+bucket as target),
    # which is exactly the <ref_dir> krea2_prepare_cache's edit mode consumes.
    edit_mode = "--edit" in argv
    args = [a for a in argv if a not in ("--uncond", "--edit")]
    buckets_arg = None
    if "--buckets" in args:
        idx = args.index("--buckets")
        if idx + 1 >= len(args):
            raise SystemExit("--buckets requires a comma-separated WxH list")
        buckets_arg = args[idx + 1]
        del args[idx : idx + 2]
    if len(args) < 2:
        print(
            "usage: krea2_stage_images.py <src_dir> <out_dir> [SIZE|WxH] "
            "[--buckets WxH,WxH,...] [--uncond] [--edit]\n"
            "  --edit: IMAGE-EDIT single-dir suffix layout — per <id>.txt, "
            "target=<id>.jpg, reference=<id>-condlabel.png; references staged to "
            "<out_dir>/ref for krea2_prepare_cache's 5th (ref_stage_dir) arg."
        )
        sys.exit(1)
    src = Path(args[0])
    out = Path(args[1])
    if buckets_arg:
        buckets = [_parse_bucket(x) for x in buckets_arg.split(",") if x.strip()]
    else:
        buckets = [_parse_bucket(args[2] if len(args) > 2 else "1024")]
    for w, h in buckets:
        if w % KREA2_BUCKET_DIV != 0 or h % KREA2_BUCKET_DIV != 0:
            raise SystemExit(
                f"bucket {w}x{h} must be divisible by {KREA2_BUCKET_DIV} "
                f"(vae_scale_factor*patch)"
            )
    return src, out, buckets, emit_uncond, edit_mode


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


def _stage_edit(src: Path, out: Path, buckets, emit_uncond: bool):
    """IMAGE-EDIT stage A (single-dir suffix convention). Per <id>.txt:
    target=<id>.jpg, reference=<id>-condlabel.png, caption=<id>.txt. The reference
    is decoded to the SAME bucket as its target and staged to <out>/ref."""
    out.mkdir(parents=True, exist_ok=True)
    ref_out = out / "ref"
    ref_out.mkdir(parents=True, exist_ok=True)

    stems = sorted(
        p.stem
        for p in src.iterdir()
        if p.suffix.lower() == ".txt" and not p.stem.endswith("-condlabel")
    )
    tensors, ref_tensors, captions = {}, {}, {}
    bucket_counts = Counter()
    kept = 0
    for stem in stems:
        tgt = src / f"{stem}.jpg"
        ref = src / f"{stem}-condlabel.png"
        cap = src / f"{stem}.txt"
        if not tgt.exists():
            print(f"skip {stem}: no target {stem}.jpg")
            continue
        if not ref.exists():
            print(f"skip {stem}: no reference {stem}-condlabel.png")
            continue
        # Bucket is chosen from the TARGET; the reference is forced to the SAME
        # bucket so the Mojo edit prepare's same-bucket assertion holds.
        w, h = Image.open(tgt).size
        bucket_w, bucket_h = _choose_bucket(w, h, buckets)
        tensors[f"image.{kept}"] = _decode_to_bucket(tgt, bucket_w, bucket_h)
        ref_tensors[f"image.{kept}"] = _decode_to_bucket(ref, bucket_w, bucket_h)
        captions[str(kept)] = cap.read_text().strip()
        bucket_counts[(bucket_w, bucket_h)] += 1
        kept += 1

    if kept == 0:
        raise SystemExit(f"no <id>.txt + <id>.jpg + <id>-condlabel.png trios in {src}")

    save_file(tensors, str(out / "images.safetensors"))
    save_file(ref_tensors, str(ref_out / "images.safetensors"))
    for k, v in captions.items():
        (out / f"prompt.{k}.txt").write_text(v)
    if emit_uncond:
        (out / "uncond.txt").write_text("")
        print(f"staged uncond.txt (empty-caption render) -> {out}")
    summary = ", ".join(
        f"{w}x{h}:{count}" for (w, h), count in sorted(bucket_counts.items())
    )
    print(
        f"staged {kept} EDIT samples -> {out} (+ references -> {ref_out}) "
        f"(buckets {summary}, bucket-div {KREA2_BUCKET_DIV})"
    )
    print(
        "next: krea2_prepare_cache <out> <cache.safetensors> "
        f"{kept} <SIZE> {ref_out}   (5th arg = the reference stage dir)"
    )


def main():
    src, out, buckets, emit_uncond, edit_mode = _parse_args(sys.argv[1:])
    if edit_mode:
        _stage_edit(src, out, buckets, emit_uncond)
        return
    out.mkdir(parents=True, exist_ok=True)

    # Accept .jpg and .png; sort numerically when stems are digits (stable order).
    imgs = sorted(
        [p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda p: int(p.stem) if p.stem.isdigit() else 1 << 30,
    )
    tensors, captions = {}, {}
    bucket_counts = Counter()
    kept = 0
    for p in imgs:
        cap_path = p.with_suffix(".txt")
        if not cap_path.exists():
            print(f"skip {p.name}: no caption")
            continue
        img = Image.open(p).convert("RGB")
        w, h = img.size
        bucket_w, bucket_h = _choose_bucket(w, h, buckets)
        img = _center_crop_to_aspect(img, bucket_w, bucket_h)
        img = img.resize((bucket_w, bucket_h), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0   # [H,W,3] in [-1,1]
        arr = arr.transpose(2, 0, 1)[None]                       # [1,3,H,W]
        tensors[f"image.{kept}"] = np.ascontiguousarray(arr)

        captions[str(kept)] = cap_path.read_text().strip()
        bucket_counts[(bucket_w, bucket_h)] += 1
        kept += 1

    if kept == 0:
        raise SystemExit(f"no image+caption pairs found in {src}")

    save_file(tensors, str(out / "images.safetensors"))
    for k, v in captions.items():
        (out / f"prompt.{k}.txt").write_text(v)
    if emit_uncond:
        # Empty caption: the Mojo prepare wraps it as PREFIX + "" + SUFFIX.
        (out / "uncond.txt").write_text("")
        print(f"staged uncond.txt (empty-caption render) -> {out}")
    summary = ", ".join(
        f"{w}x{h}:{count}" for (w, h), count in sorted(bucket_counts.items())
    )
    print(
        f"staged {kept} samples -> {out} "
        f"(buckets {summary}, bucket-div {KREA2_BUCKET_DIV})"
    )


if __name__ == "__main__":
    main()
