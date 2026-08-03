# Oracle for serenitymojo/pipeline/minimax_h3_keyframe_image.mojo.
#
# CPU only (PIL + numpy; torch is used ONLY as the safetensors container and
# never touches CUDA).
#
# The vendor function under test is NOT transcribed here — it is EXEC'd
# straight out of /home/alex/minimax_h3_ref/diffusers/packing.py, so the gate
# runs the vendor's own bytes and cannot drift from them.
import os, re, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

REF = "/home/alex/minimax_h3_ref/diffusers/packing.py"
src = open(REF).read()
m = re.search(r"^def prepare_keyframe_image\(.*?(?=^def |\Z)", src, re.S | re.M)
assert m, "prepare_keyframe_image not found in the reference"
ns = {"Image": Image}
exec(m.group(0), ns)
prepare_keyframe_image = ns["prepare_keyframe_image"]
print("exec'd vendor prepare_keyframe_image,", len(m.group(0)), "bytes from", REF)

# Deterministic, structured test images: smooth gradients plus high-frequency
# detail, so a filter that is merely "close" shows up rather than averaging out.
def make_image(h, w, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 127.5 + 127.0 * np.sin(xx / 7.3 + yy / 11.7)
    g = 255.0 * (xx / max(w - 1, 1))
    b = 255.0 * (yy / max(h - 1, 1))
    img = np.stack([r, g, b], -1)
    img += rng.integers(-24, 25, size=img.shape)          # break up the smoothness
    img[h // 3 : h // 3 + max(1, h // 20), :, :] = 255.0   # a hard edge
    img[:, w // 4 : w // 4 + max(1, w // 25), :] = 0.0
    return np.clip(img, 0, 255).astype(np.uint8)

CASES = [
    # name,           src_h, src_w, dst_h, dst_w, stretch
    ("small_stretch",   100,   137,    64,    96, True),
    ("up_stretch",      270,   480,   768,  1344, True),
    ("mixed_stretch",  1600,   900,   768,  1344, True),
    ("cover_crop",     1000,  1000,   768,  1344, False),
    ("cover_crop_tall", 900,   400,   768,  1344, False),
    ("identity",        768,  1344,   768,  1344, True),
]

tensors = {}
meta = []
for i, (name, sh, sw, dh, dw, stretch) in enumerate(CASES):
    arr = make_image(sh, sw, seed=1000 + i)
    pil = Image.fromarray(arr, mode="RGB")
    out = prepare_keyframe_image(pil, dh, dw, stretch)
    out_arr = np.array(out)
    assert out_arr.shape == (dh, dw, 3), (name, out_arr.shape)
    tensors[f"in_{name}"] = torch.from_numpy(arr.copy())
    tensors[f"out_{name}"] = torch.from_numpy(out_arr.copy())
    meta.append((name, sh, sw, dh, dw, stretch))
    print(f"  {name:16s} {sw}x{sh} -> {dw}x{dh} stretch={stretch}  "
          f"out mean={out_arr.mean():.4f}")

# A plain resize too, so a failure can be localized to the filter rather than
# to the cover-crop bookkeeping around it.
for i, (sh, sw, dh, dw) in enumerate([(100, 137, 64, 96), (240, 320, 480, 640), (1080, 1920, 768, 1344)]):
    arr = make_image(sh, sw, seed=2000 + i)
    out = np.array(Image.fromarray(arr, mode="RGB").resize((dw, dh), Image.Resampling.LANCZOS))
    tensors[f"rin_{i}"] = torch.from_numpy(arr.copy())
    tensors[f"rout_{i}"] = torch.from_numpy(out.copy())
    print(f"  resize_{i:<9d} {sw}x{sh} -> {dw}x{dh}                out mean={out.mean():.4f}")

# ── EXIF orientation: the transpose itself, and the tag detection ───────────
# The vendor applies ImageOps.exif_transpose before the canvas is resolved, so
# an orientation this port gets wrong changes the canvas, not just the framing.
from PIL import ImageOps
orient_src = make_image(90, 140, seed=3000)
tensors["orient_in"] = torch.from_numpy(orient_src.copy())
_METHOD = {
    1: None,
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}
pil_src = Image.fromarray(orient_src, mode="RGB")
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_keyframe"
os.makedirs(OUT_DIR, exist_ok=True)
jpeg_dir = os.path.join(OUT_DIR, "exif_fixtures")
os.makedirs(jpeg_dir, exist_ok=True)
for o, method in _METHOD.items():
    expected = pil_src if method is None else pil_src.transpose(method)
    tensors[f"orient_out_{o}"] = torch.from_numpy(np.array(expected).copy())
    # A JPEG carrying exactly this orientation tag, for the byte-level parser.
    exif = Image.Exif()
    exif[0x0112] = o
    pil_src.save(os.path.join(jpeg_dir, f"orient_{o}.jpg"), exif=exif, quality=92)
    # Cross-check: PIL's own exif_transpose of that JPEG has the shape we expect.
    with Image.open(os.path.join(jpeg_dir, f"orient_{o}.jpg")) as im:
        got = ImageOps.exif_transpose(im)
        assert got.size == expected.size, (o, got.size, expected.size)
pil_src.save(os.path.join(jpeg_dir, "no_exif.jpg"), quality=92)
pil_src.save(os.path.join(jpeg_dir, "plain.png"))
# The PNG round-trip the ffmpeg ingest route has to reproduce byte for byte.
tensors["png_in"] = torch.from_numpy(np.array(Image.open(os.path.join(jpeg_dir, "plain.png")).convert("RGB")).copy())
print("wrote EXIF fixtures to", jpeg_dir)

# A canvas-ALIGNED prepared keyframe (both axes a multiple of the VAE's 16), for
# scripts/minimax_h3_keyframe_encode_oracle.py — that gate takes an image that is
# already on the canvas, because the resize is gated separately and bit-exactly
# right here.
prepared = make_image(128, 192, seed=4000)
prepared_path = os.path.join(OUT_DIR, "prepared_keyframe_192x128.png")
Image.fromarray(prepared, mode="RGB").save(prepared_path)
print("wrote prepared keyframe", prepared_path)


out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT_DIR, "keyframe_image_ref.safetensors")
save_file(tensors, out_path)
print("wrote", out_path)
print("PIL", Image.__version__ if hasattr(Image, "__version__") else "?", "cuda_initialized", torch.cuda.is_initialized())
