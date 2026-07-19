#!/usr/bin/env python
# Render a T2V pixels safetensors [1,3,F,H,W] (values in [0,1]) to mp4 (cv2) + gif
# (PIL). Usage:
#   python render_t2v.py <pixels.safetensors> <out_basename> [fps] [tensor_key]
import sys, os
import numpy as np
from safetensors import safe_open
from PIL import Image

path = sys.argv[1]
base = sys.argv[2]
fps = int(sys.argv[3]) if len(sys.argv) > 3 else 24
key = sys.argv[4] if len(sys.argv) > 4 else "pixels"

with safe_open(path, framework="np") as f:
    px = f.get_tensor(key)   # [1,3,F,H,W]
px = np.asarray(px, dtype=np.float32)
_, C, F, H, W = px.shape
frames = []
for i in range(F):
    img = px[0, :, i].transpose(1, 2, 0)          # HWC
    img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    frames.append(img)

# GIF (PIL)
gif_path = base + ".gif"
pil_frames = [Image.fromarray(fr) for fr in frames]
dur_ms = max(1, int(round(1000.0 / fps)))
pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                   duration=dur_ms, loop=0, optimize=False)
print(f"wrote {gif_path}  ({F} frames {W}x{H} @ {fps}fps)")

# MP4 (cv2)
try:
    import cv2
    mp4_path = base + ".mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(mp4_path, fourcc, float(fps), (W, H))
    for fr in frames:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()
    # cv2's mp4v (FMP4) is unplayable in most desktop players -> re-encode H.264
    import subprocess, os as _os
    _tmp = mp4_path + ".h264.mp4"
    _r = subprocess.run(["/home/alex/.local/bin/ffmpeg", "-y", "-v", "error", "-i", mp4_path,
                         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                         "-movflags", "+faststart", _tmp])
    if _r.returncode == 0:
        _os.replace(_tmp, mp4_path)
    print(f"wrote {mp4_path}")
except Exception as e:
    print(f"mp4 skipped: {e}")
