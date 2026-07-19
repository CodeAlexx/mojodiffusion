#!/usr/bin/env python3
"""Stage the Alina dataset for the pure-Mojo Z-Image prepare driver.

Dev tooling only (the runtime path is pipeline/zimage_prepare.mojo, pure Mojo):
reads JPEGs + sibling .txt captions, resizes/center-crops each image to the
nearest Z-Image production bucket, and writes what zimage_prepare expects in
output/alina_zimage_stage/:
    <stem>.safetensors   image: [1,3,H,W] BF16, RGB, [-1,1]
    <stem>.txt           caption (copied verbatim)

Buckets (train_zimage_real comptime contract; latent = pixel/8):
    576x448  (latent 72x56)     704x384  (latent 88x48)

Run: /home/alex/serenityflow-v2/.venv/bin/python scripts/zimage_stage_alina.py
"""
import os
import shutil

import torch
from PIL import Image
from safetensors.torch import save_file

SRC = "/home/alex/datasets/AlinaAignatova"
DST = "/home/alex/mojodiffusion/output/alina_zimage_stage"
BUCKETS = [(576, 448), (704, 384)]  # (H, W)


def pick_bucket(w: int, h: int) -> tuple[int, int]:
    ar = h / w
    return min(BUCKETS, key=lambda b: abs((b[0] / b[1]) - ar))


def stage(path: str, bucket: tuple[int, int]) -> torch.Tensor:
    bh, bw = bucket
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # resize so the bucket fits inside, then center-crop
    scale = max(bw / w, bh / h)
    nw, nh = round(w * scale), round(h * scale)
    img = img.resize((nw, nh), Image.BICUBIC)
    left = (nw - bw) // 2
    top = (nh - bh) // 2
    img = img.crop((left, top, left + bw, top + bh))
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    t = t.view(bh, bw, 3).permute(2, 0, 1).float() / 127.5 - 1.0  # CHW [-1,1]
    return t.unsqueeze(0).to(torch.bfloat16).contiguous()  # [1,3,H,W] BF16


def main() -> None:
    os.makedirs(DST, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".webp")
    stems = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(SRC)
        if f.lower().endswith(exts)
    )
    ext_of = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(SRC)
        if f.lower().endswith(exts)
    }
    n_main = n_tall = 0
    staged = 0
    for stem in stems:
        cap = os.path.join(SRC, stem + ".txt")
        if not os.path.exists(cap):
            print("SKIP (no caption):", stem)
            continue
        img_path = os.path.join(SRC, ext_of[stem])
        with Image.open(img_path) as im:
            bucket = pick_bucket(*im.size)
        t = stage(img_path, bucket)
        save_file({"image": t}, os.path.join(DST, stem + ".safetensors"))
        shutil.copyfile(cap, os.path.join(DST, stem + ".txt"))
        if bucket == BUCKETS[0]:
            n_main += 1
        else:
            n_tall += 1
        staged += 1
    print(f"staged {staged} samples -> {DST}  (576x448: {n_main}, 704x384: {n_tall})")


if __name__ == "__main__":
    main()
