#!/usr/bin/env python3
"""Stage the raw Alina set for the Klein 9B prepare driver.

Klein's prepare expects, per sample i, in DST:
    sample_<i>.safetensors  key "image" = [1,3,512,512] F32 in [-1,1]
    sample_<i>.txt          the caption

That differs from the zimage stager (scripts/zimage_stage_alina.py), which
buckets to 576x448 / 704x384 in BF16 and keys files by source stem. Klein is
square-512 and F32, so it gets its own stager rather than a flag on that one.

Pairs without a caption are skipped, and the surviving pairs are renumbered
contiguously from 0 so the prepare driver can index them without gaps. The
sample -> source stem mapping is written to sample_index.txt for provenance.

Run:
    python3 scripts/klein_stage_alina.py [SIZE]
"""

import os
import shutil
import sys

import torch
from PIL import Image
from safetensors.torch import save_file

SRC = "/home/alex/datasets/AlinaAignatova"
DST = "/home/alex/mojodiffusion/output/alina_stage"
SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 512


def stage(path: str, size: int) -> torch.Tensor:
    """Resize so the square fits inside, center-crop, scale to [-1,1] F32."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    img = img.resize((nw, nh), Image.BICUBIC)
    left = (nw - size) // 2
    top = (nh - size) // 2
    img = img.crop((left, top, left + size, top + size))
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    t = t.view(size, size, 3).permute(2, 0, 1).float() / 127.5 - 1.0
    return t.unsqueeze(0).contiguous()  # [1,3,S,S] F32


def main() -> None:
    os.makedirs(DST, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".webp")
    ext_of = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(SRC)
        if f.lower().endswith(exts)
    }
    stems = sorted(ext_of)

    staged = 0
    index = []
    for stem in stems:
        cap = os.path.join(SRC, stem + ".txt")
        if not os.path.exists(cap):
            print("SKIP (no caption):", stem)
            continue
        t = stage(os.path.join(SRC, ext_of[stem]), SIZE)
        assert t.shape == (1, 3, SIZE, SIZE), t.shape
        assert t.dtype == torch.float32
        save_file({"image": t}, os.path.join(DST, f"sample_{staged}.safetensors"))
        shutil.copyfile(cap, os.path.join(DST, f"sample_{staged}.txt"))
        index.append(f"{staged}\t{stem}")
        staged += 1

    with open(os.path.join(DST, "sample_index.txt"), "w") as fh:
        fh.write("\n".join(index) + "\n")

    lo, hi = t.min().item(), t.max().item()
    print(f"staged {staged} samples at {SIZE}x{SIZE} F32 -> {DST}")
    print(f"last sample range [{lo:.4f}, {hi:.4f}] (expect within [-1,1])")
    print("NOTE: feed this count to klein_prepare (stage_dir cache_dir N)")


if __name__ == "__main__":
    main()
