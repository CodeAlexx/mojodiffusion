#!/usr/bin/env python3
"""klein_stage_images.py — stage raw jpg+caption pairs into the Klein "alina-style"
prepare input format.

For each <name>.jpg with a sibling <name>.txt in the SOURCE dir, produce one
staged sample the Mojo prepare (serenitymojo/pipeline/klein_prepare.mojo) can
consume:

  <STAGE_DIR>/sample_<i>.safetensors   key "image" = [1,3,512,512] F32 in [-1,1]
  <STAGE_DIR>/sample_<i>.txt           the raw caption text (verbatim)

Preprocess: open RGB -> center-crop to square (min side) -> resize to 512x512
(LANCZOS) -> float32 NCHW normalized (px/127.5 - 1.0). This mirrors the
output/alina_stage contract that klein_prepare_alina.mojo's _load_image reads
(SafeTensors key "image", rank-4 [1,3,512,512], F32).

CPU-only. Uses the torchref venv's numpy + PIL + safetensors:
  /home/alex/torchref-image/venv/bin/python \
      serenitymojo/training/klein_stage_images.py

Run with no args for the defaults below, or:
  ... klein_stage_images.py <SOURCE_DIR> <STAGE_DIR> [SIZE]
"""

import os
import sys
import glob

import numpy as np
from PIL import Image
from safetensors.numpy import save_file

SOURCE_DIR = "/home/alex/eri2_with_trigger"
STAGE_DIR = "/home/alex/mojodiffusion/output/eri2_klein_stage"
SIZE = 512


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def to_image_tensor(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = center_crop_square(img)
    img = img.resize((size, size), Image.LANCZOS)
    # HWC uint8 -> float32 [-1,1] -> CHW -> add batch axis => [1,3,size,size]
    arr = np.asarray(img, dtype=np.float32)          # [H,W,3]
    arr = arr / 127.5 - 1.0                           # [-1, 1]
    arr = np.transpose(arr, (2, 0, 1))                # [3,H,W]
    arr = np.ascontiguousarray(arr[None, ...], dtype=np.float32)  # [1,3,H,W]
    return arr


def main() -> int:
    source_dir = sys.argv[1] if len(sys.argv) > 1 else SOURCE_DIR
    stage_dir = sys.argv[2] if len(sys.argv) > 2 else STAGE_DIR
    size = int(sys.argv[3]) if len(sys.argv) > 3 else SIZE

    os.makedirs(stage_dir, exist_ok=True)
    # clear any stale staged samples so a re-run is clean
    for old in glob.glob(os.path.join(stage_dir, "sample_*.safetensors")):
        os.remove(old)
    for old in glob.glob(os.path.join(stage_dir, "sample_*.txt")):
        os.remove(old)

    jpgs = sorted(glob.glob(os.path.join(source_dir, "*.jpg")))
    print(f"[stage] source={source_dir}  found {len(jpgs)} jpg")
    print(f"[stage] stage_dir={stage_dir}  size={size}x{size}")

    written = 0
    skipped = 0
    for jpg in jpgs:
        base = os.path.splitext(jpg)[0]
        txt = base + ".txt"
        if not os.path.isfile(txt):
            print(f"  skip (no caption): {os.path.basename(jpg)}")
            skipped += 1
            continue
        with open(txt, "r", encoding="utf-8") as fh:
            caption = fh.read()

        arr = to_image_tensor(jpg, size)
        assert arr.shape == (1, 3, size, size), arr.shape
        assert arr.dtype == np.float32

        out_st = os.path.join(stage_dir, f"sample_{written}.safetensors")
        out_txt = os.path.join(stage_dir, f"sample_{written}.txt")
        save_file({"image": arr}, out_st)
        with open(out_txt, "w", encoding="utf-8") as fh:
            fh.write(caption)
        written += 1

    print("")
    print(f"PASS: wrote {written} staged samples to {stage_dir} "
          f"(skipped {skipped} without caption)")
    print(f"      pass N={written} to klein_prepare.mojo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
