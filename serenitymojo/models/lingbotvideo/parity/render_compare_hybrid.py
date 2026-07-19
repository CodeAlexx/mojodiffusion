#!/usr/bin/env python
"""Phase-2 HYBRID (w1/w3 mxfp4 + w2 fp8) image-consistency verdict.

Bar: hybrid apple must match fp8/bf16 (quantization must not change the picture).
Decisive numbers: hybrid-vs-fp8 and hybrid-vs-bf16 pixel cosine.
Compares against the pure-mxfp4 result (which failed at 0.9774 vs fp8).

Reads (after the hybrid e_t2i gate runs; mojo_e_pixels.safetensors = hybrid):
  mojo_e_pixels.safetensors        <- hybrid (w1/w3 mxfp4, w2 fp8)
  mojo_e_pixels_fp8.safetensors    <- fp8 backup
  mojo_e_pixels_bf16.safetensors   <- bf16 backup
  oracle_e.safetensors ('pixels')  <- torch reference
Writes mojo_e_generated_hybrid.png + a 4-panel sheet (oracle|bf16|fp8|hybrid)
and prints a VERDICT line.
"""
import os
import numpy as np
from safetensors import safe_open
from PIL import Image

D = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
MEDIA = "/home/alex/mojodiffusion/output/media/lingbotvideo"


def load_px(p, k="pixels"):
    with safe_open(p, framework="np") as f:
        return f.get_tensor(k)


def to_img(px):
    return np.clip(px[0, :, 0].transpose(1, 2, 0) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def cos(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


hyb = load_px(os.path.join(D, "mojo_e_pixels.safetensors"))
fp8 = load_px(os.path.join(D, "mojo_e_pixels_fp8.safetensors"))
bf16 = load_px(os.path.join(D, "mojo_e_pixels_bf16.safetensors"))
orac = load_px(os.path.join(D, "oracle_e.safetensors"))

img_h = to_img(hyb)
Image.fromarray(img_h).save(os.path.join(D, "mojo_e_generated_hybrid.png"))
Image.fromarray(img_h).save(os.path.join(MEDIA, "mojo_e_generated_hybrid.png"))

c_hf = cos(hyb, fp8); c_hb = cos(hyb, bf16)
print("=== HYBRID (w1/w3 mxfp4 + w2 fp8) IMAGE CONSISTENCY ===")
print(f"  hybrid vs fp8  : {c_hf:.4f}   <-- gate (want >= ~0.99; pure-mxfp4 was 0.9774)")
print(f"  hybrid vs bf16 : {c_hb:.4f}   <-- gate")
print(f"  fp8    vs bf16 : {cos(fp8, bf16):.4f}   (reference bar 0.9969)")
print(f"  hybrid vs oracle: {cos(hyb, orac):.4f}   (MoE-vs-torch, informational)")
verdict = "PASS (consistent)" if (c_hf >= 0.99 and c_hb >= 0.99) else \
          ("MARGINAL" if c_hf >= 0.985 else "FAIL (still drifts -> keep fp8)")
print(f"  VERDICT: {verdict}")

imgs = [to_img(orac), to_img(bf16), to_img(fp8), img_h]
h, w, _ = imgs[0].shape; pad = 8
sheet = np.full((h + 24, w * 4 + pad * 3, 3), 255, np.uint8)
for i, im in enumerate(imgs):
    sheet[24:24 + h, i * (w + pad):i * (w + pad) + w] = im
out = os.path.join(MEDIA, "e_hybrid_vs_fp8_vs_bf16_contactsheet.png")
Image.fromarray(sheet).save(out)
print("SAVED", out)
