#!/usr/bin/env python
"""Phase-2 mxfp4 IMAGE CONSISTENCY gate: does 4-bit change the picture?

The acceptance bar (set by the user): the mxfp4 apple must match the fp8/bf16
apple — quantization must NOT alter the image. So the decisive numbers are
mxfp4-vs-fp8 and mxfp4-vs-bf16 pixel cosine (NOT mxfp4-vs-torch, which is the
separate MoE-non-reproducibility axis).

Reads (after e_t2i_probe.mojo runs with LINGBOT_CKPT=transformer_mxfp4):
  mojo_e_pixels.safetensors        <- mxfp4 run
  mojo_e_pixels_fp8.safetensors    <- fp8 run (backup)
  mojo_e_pixels_bf16.safetensors   <- bf16 run (backup)
  oracle_e.safetensors ('pixels')  <- torch reference
Writes mojo_e_generated_mxfp4.png + a 4-panel contact sheet
(oracle | bf16 | fp8 | mxfp4) to output/media/lingbotvideo/.
"""
import os
import numpy as np
from safetensors import safe_open
from PIL import Image

D = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
MEDIA = "/home/alex/mojodiffusion/output/media/lingbotvideo"


def load_px(path, key="pixels"):
    with safe_open(path, framework="np") as f:
        return f.get_tensor(key)


def to_img(px):
    img = px[0, :, 0].transpose(1, 2, 0)
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def cos(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


mxfp4 = load_px(os.path.join(D, "mojo_e_pixels.safetensors"))
fp8 = load_px(os.path.join(D, "mojo_e_pixels_fp8.safetensors"))
bf16 = load_px(os.path.join(D, "mojo_e_pixels_bf16.safetensors"))
orac = load_px(os.path.join(D, "oracle_e.safetensors"))

img_mx = to_img(mxfp4)
Image.fromarray(img_mx).save(os.path.join(D, "mojo_e_generated_mxfp4.png"))
Image.fromarray(img_mx).save(os.path.join(MEDIA, "mojo_e_generated_mxfp4.png"))

print("=== mxfp4 IMAGE CONSISTENCY (pixel cosine) ===")
print(f"  mxfp4 vs fp8   : {cos(mxfp4, fp8):.4f}   <-- consistency gate (want >= ~0.99)")
print(f"  mxfp4 vs bf16  : {cos(mxfp4, bf16):.4f}   <-- consistency gate (want >= ~0.99)")
print(f"  fp8   vs bf16  : {cos(fp8, bf16):.4f}   (reference: fp8 held at 0.9969)")
print(f"  mxfp4 vs oracle: {cos(mxfp4, orac):.4f}   (MoE-vs-torch axis, informational)")
print(f"  mxfp4 mean/std : {float(mxfp4.mean()):.4f} / {float(mxfp4.std()):.4f}")

imgs = [to_img(orac), to_img(bf16), to_img(fp8), img_mx]
labels = ["oracle", "bf16", "fp8", "mxfp4"]
h, w, _ = imgs[0].shape
pad = 8
sheet = np.full((h + 24, w * 4 + pad * 3, 3), 255, np.uint8)
for i, im in enumerate(imgs):
    x = i * (w + pad)
    sheet[24:24 + h, x:x + w] = im
out = os.path.join(MEDIA, "e_mxfp4_vs_fp8_vs_bf16_contactsheet.png")
Image.fromarray(sheet).save(out)
print("SAVED", out)
print("SAVED", os.path.join(MEDIA, "mojo_e_generated_mxfp4.png"))
