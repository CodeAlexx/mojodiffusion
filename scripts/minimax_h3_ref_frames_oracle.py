# Oracle for serenitymojo/pipeline/minimax_h3_ref_frames.mojo.
#
# CPU only (PIL + numpy; torch is used ONLY as the safetensors container and
# never touches CUDA). Host uint8 image resampling — CPU parity is the right
# kind here, same as the keyframe-image gate.
#
# The vendor functions under test are NOT transcribed — they are EXEC'd
# straight out of the reference tree, so the gate runs the vendor's own bytes
# and cannot drift from them:
#   * resolve_canvas_size            modular_pipelines/minimax_h3/packing.py
#   * resample_reference_frames      modular_pipelines/minimax_h3/packing_ref2va.py:622
#   * prepare_reference_frames       modular_pipelines/minimax_h3/packing_ref2va.py:654
#
# The REAL frames come from output/h3_ref2va_media/ref_video.mp4 (1920x1080
# @ 24 fps), decoded through the SAME ffmpeg rawvideo route
# pipeline/minimax_h3_media_in.mojo uses (`-f rawvideo -pix_fmt rgb24`, no
# filters), so oracle and probe start from byte-identical pixels.
import math
import os
import re
import subprocess
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

REF_DIR = "/home/alex/minimax_h3_ref/diffusers-src/src/diffusers/modular_pipelines/minimax_h3"
VIDEO = "/home/alex/mojodiffusion/output/h3_ref2va_media/ref_video.mp4"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_ref_frames"

ns = {"np": np, "Image": Image, "math": math}

packing_src = open(os.path.join(REF_DIR, "packing.py")).read()
for const in (
    "MINIMAX_H3_FPS",
    "MINIMAX_H3_SHORT_EDGE",
    "MINIMAX_H3_MAX_PIXELS",
    "MINIMAX_H3_CANVAS_MULTIPLE",
    "MINIMAX_H3_MIN_ASPECT_RATIO",
    "MINIMAX_H3_MAX_ASPECT_RATIO",
):
    m = re.search(rf"^{const} = .*$", packing_src, re.M)
    assert m, f"{const} not found in the reference packing.py"
    exec(m.group(0), ns)
m = re.search(r"^def resolve_canvas_size\(.*?(?=^def |\Z)", packing_src, re.S | re.M)
assert m, "resolve_canvas_size not found in the reference"
exec(m.group(0), ns)

ref2va_src = open(os.path.join(REF_DIR, "packing_ref2va.py")).read()
for fname in ("resample_reference_frames", "prepare_reference_frames"):
    m = re.search(rf"^def {fname}\(.*?(?=^def |\Z)", ref2va_src, re.S | re.M)
    assert m, f"{fname} not found in the reference"
    exec(m.group(0), ns)
print("exec'd vendor resolve_canvas_size + resample/prepare_reference_frames from", REF_DIR)

resolve_canvas_size = ns["resolve_canvas_size"]
resample_reference_frames = ns["resample_reference_frames"]
prepare_reference_frames = ns["prepare_reference_frames"]

# ── real frames, decoded the way media_in decodes them ──────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
raw = os.path.join(OUT_DIR, "ref_video_head.rgb")
subprocess.run(
    ["ffmpeg", "-v", "error", "-y", "-i", VIDEO, "-frames:v", "22",
     "-f", "rawvideo", "-pix_fmt", "rgb24", raw],
    check=True,
)
head = np.fromfile(raw, dtype=np.uint8).reshape(22, 1080, 1920, 3)
os.remove(raw)
real5 = head[[0, 5, 11, 16, 21]].copy()
print(f"decoded {head.shape[0]} real frames, kept 5 of them: {real5.shape}")

tensors = {}

# [1] the real thing: 5 x 1920x1080 -> the canvas its own 16:9 resolves to.
canvas = resolve_canvas_size(real5.shape[2], real5.shape[1])
out5 = prepare_reference_frames(real5, 5)
assert out5.shape == (5, canvas[0], canvas[1], 3), (out5.shape, canvas)
tensors["real_in"] = torch.from_numpy(real5)
tensors["real_out"] = torch.from_numpy(out5.copy())
print(f"  real5     1920x1080 -> {canvas[1]}x{canvas[0]}  out mean={out5.mean():.4f}")

# [2] truncation inside prepare: cap at 3 of the 5 frames.
out3 = prepare_reference_frames(real5, 3)
assert out3.shape == (3, canvas[0], canvas[1], 3)
assert np.array_equal(out3, out5[:3]), "truncation must commute with the per-frame resize"
tensors["realcap_out"] = torch.from_numpy(out3.copy())
print(f"  realcap   5 frames capped at 3 -> {out3.shape}")

# [3] identity: frames already at the canvas flow through untouched — the
# vendor slices first (`frames[:num_frames]`, a view), so the passthrough is
# a view of the input, not the input object: same memory, no resampling pass.
ident = prepare_reference_frames(out5, 5)
assert np.shares_memory(ident, out5) and np.array_equal(ident, out5), (
    "vendor passthrough must be an untouched view of the input"
)
print("  identity  canvas-sized input passed through untouched (shared memory)")

# [4] a portrait reference exercises the width<height canvas branch.
def make_frame(h, w, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 127.5 + 127.0 * np.sin(xx / 7.3 + yy / 11.7)
    g = 255.0 * (xx / max(w - 1, 1))
    b = 255.0 * (yy / max(h - 1, 1))
    img = np.stack([r, g, b], -1)
    img += rng.integers(-24, 25, size=img.shape)
    img[h // 3 : h // 3 + max(1, h // 20), :, :] = 255.0
    img[:, w // 4 : w // 4 + max(1, w // 25), :] = 0.0
    return np.clip(img, 0, 255).astype(np.uint8)

portrait = np.stack([make_frame(854, 480, seed=5000 + i) for i in range(2)])
pcanvas = resolve_canvas_size(portrait.shape[2], portrait.shape[1])
portrait_out = prepare_reference_frames(portrait, 2)
assert portrait_out.shape == (2, pcanvas[0], pcanvas[1], 3)
tensors["portrait_in"] = torch.from_numpy(portrait)
tensors["portrait_out"] = torch.from_numpy(portrait_out.copy())
print(f"  portrait  480x854 -> {pcanvas[1]}x{pcanvas[0]}  out mean={portrait_out.mean():.4f}")

# [5] the canvas law's own refusal: aspect ratio beyond 4:1 must raise.
wide = np.zeros((1, 64, 320, 3), dtype=np.uint8)
try:
    prepare_reference_frames(wide, 1)
    raise SystemExit("vendor accepted a 5:1 reference — law drift, refusing to write the oracle")
except ValueError as err:
    print(f"  aspect    5:1 refused by the vendor: {err}")

# [6] the 24 fps resample applied to pixels: a DROP case and a DUPLICATE case.
fps30 = np.stack([make_frame(48, 64, seed=6000 + i) for i in range(8)])
fps30_out = resample_reference_frames(fps30, 30.0)
tensors["fps30_in"] = torch.from_numpy(fps30)
tensors["fps30_out"] = torch.from_numpy(fps30_out.copy())
print(f"  fps30     8 frames @30 -> {fps30_out.shape[0]} on the 24 grid")

fps12 = np.stack([make_frame(48, 64, seed=7000 + i) for i in range(5)])
fps12_out = resample_reference_frames(fps12, 12.0)
tensors["fps12_in"] = torch.from_numpy(fps12)
tensors["fps12_out"] = torch.from_numpy(fps12_out.copy())
print(f"  fps12     5 frames @12 -> {fps12_out.shape[0]} on the 24 grid")

ident24 = resample_reference_frames(fps30, 24.0)
assert ident24 is fps30, "vendor 24 fps resample must return the input array itself"
print("  fps24     identity (same array)")

out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT_DIR, "ref_frames_ref.safetensors")
save_file(tensors, out_path)
print("wrote", out_path)
print("PIL", Image.__version__, "cuda_initialized", torch.cuda.is_initialized())
