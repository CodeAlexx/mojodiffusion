#!/usr/bin/env python3
# krea2_stage_one.py — stage a SINGLE source image to a [1,3,SIZE,SIZE] f32 [-1,1]
# safetensors for the krea2 EDIT inference CLI (Mojo has no image decoder). Default
# mirrors training/krea2_stage_images._decode_to_bucket (center-crop to square,
# LANCZOS resize, /127.5 - 1, CHW).
#
# ASPECT MODES (2026-07-14 — a blind center-crop on the 1696x2624 demo photo cut the
# faces out of the model input entirely; the "reframed" outputs were input-prep, not
# the model):
#   --crop center|top|bottom : which square band of a non-square image to keep
#   --crop fit               : NO crop — letterbox-pad the whole image onto a square
#                              canvas (mean-gray pad) so nothing is lost; output is
#                              the full scene at slightly lower effective resolution
#   --crop smart             : top for portrait (faces usually in the upper band),
#                              center for landscape/square — a heuristic default
#
# Run:
#   <py> serenitymojo/pipeline/krea2_stage_one.py <src.png> <out.safetensors> <SIZE> [--crop MODE]
import sys
import numpy as np
from PIL import Image
from safetensors.numpy import save_file


def square(img: Image.Image, mode: str) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    if mode == "smart":
        mode = "top" if h > w else "center"
    if mode == "fit":
        s = max(w, h)
        pad = int(round(float(np.asarray(img, dtype=np.float32).mean())))
        canvas = Image.new("RGB", (s, s), (pad, pad, pad))
        canvas.paste(img, ((s - w) // 2, (s - h) // 2))
        return canvas
    s = min(w, h)
    if mode == "center":
        left, top = (w - s) // 2, (h - s) // 2
    elif mode == "top":
        left, top = ((w - s) // 2, 0) if h > w else (0, (h - s) // 2)
    elif mode == "bottom":
        left, top = ((w - s) // 2, h - s) if h > w else (w - s, (h - s) // 2)
    else:
        raise SystemExit(f"unknown --crop mode: {mode}")
    return img.crop((left, top, left + s, top + s))


def main():
    args = [a for a in sys.argv[1:]]
    mode = "center"
    if "--crop" in args:
        i = args.index("--crop")
        mode = args[i + 1]
        del args[i : i + 2]
    src, out, size = args[0], args[1], int(args[2])
    img = Image.open(src).convert("RGB")
    img = square(img, mode)
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0   # [H,W,3] in [-1,1]
    arr = arr.transpose(2, 0, 1)[None]                       # [1,3,H,W]
    arr = np.ascontiguousarray(arr)
    save_file({"image": arr}, out)
    print(f"[stage_one] {src} -> {out}  shape={arr.shape} range=[{arr.min():.3f},{arr.max():.3f}]")


if __name__ == "__main__":
    main()
