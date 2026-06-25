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
#      (krea2.py:166-167 `vae_scale_factor * patch_size`). A square SIZE divisible
#      by 16 satisfies it (512, 1024, ...). Center-crop to square, resize SIZExSIZE.
#   2) NO caption rendering. Krea-2's conditioning is NOT the ideogram4 structured-
#      JSON/chat-template digest. The Mojo prepare tokenizes
#      KREA2_TPL_PREFIX + <raw caption> + KREA2_TPL_SUFFIX itself (krea2_paths.mojo,
#      == ai-toolkit text_encoder.py:26-34 PROMPT_TEMPLATE_ENCODE_PREFIX/SUFFIX),
#      so this stager writes the RAW caption text unchanged. We must NOT pre-wrap it
#      (the DROP_IDX=34 prefix-drop in encode_krea2_stack assumes the exact template).
#
# Per sample (dataset: <dir>/N.jpg + N.txt):
#   image: center-crop to square, resize SIZExSIZE, RGB f32 [-1,1] CHW
#          -> image.<i> [1,3,SIZE,SIZE] f32   (the QwenImageVaeEncoder input)
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

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import save_file

# vae_scale_factor (8) * patch (2) — the krea2 bucket divisibility (krea2.py:166-167).
KREA2_BUCKET_DIV = 16


def main():
    args = [a for a in sys.argv[1:] if a != "--uncond"]
    emit_uncond = "--uncond" in sys.argv[1:]
    if len(args) < 2:
        print("usage: krea2_stage_images.py <src_dir> <out_dir> [SIZE] [--uncond]")
        sys.exit(1)
    src = Path(args[0])
    out = Path(args[1])
    size = int(args[2]) if len(args) > 2 else 1024
    if size % KREA2_BUCKET_DIV != 0:
        raise SystemExit(
            f"SIZE={size} must be divisible by {KREA2_BUCKET_DIV} "
            f"(vae_scale_factor*patch); pick e.g. 512 or 1024"
        )
    out.mkdir(parents=True, exist_ok=True)

    # Accept .jpg and .png; sort numerically when stems are digits (stable order).
    imgs = sorted(
        [p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda p: int(p.stem) if p.stem.isdigit() else 1 << 30,
    )
    tensors, captions = {}, {}
    kept = 0
    for p in imgs:
        cap_path = p.with_suffix(".txt")
        if not cap_path.exists():
            print(f"skip {p.name}: no caption")
            continue
        img = Image.open(p).convert("RGB")
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize((size, size), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0   # [H,W,3] in [-1,1]
        arr = arr.transpose(2, 0, 1)[None]                       # [1,3,H,W]
        tensors[f"image.{kept}"] = np.ascontiguousarray(arr)

        captions[str(kept)] = cap_path.read_text().strip()
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
    print(f"staged {kept} samples -> {out} (size {size}, bucket-div {KREA2_BUCKET_DIV})")


if __name__ == "__main__":
    main()
