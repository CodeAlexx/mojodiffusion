#!/usr/bin/env python
# FlowEdit gate metrics: edited pixels vs staged source clip.
#   per-frame MAD (overall / crude center box / outside border)
#   temporal consistency: mean |frame[i+1]-frame[i]| for edited vs source
#   optional: per-frame MAD inside/outside the ACTUAL auto-mask (per-latent-
#   frame PNGs <mask_base>_mask_t{t}.png written by lingbot_flowedit
#   --auto-mask; pixel frame f maps to latent frame 0 if f==0 else (f+3)//4).
# Usage: flowedit_metrics.py <edited_pixels.safetensors> <src_clip.safetensors>
#                            [mask_base]
import sys
import numpy as np
from safetensors import safe_open

ed_p, src_p = sys.argv[1], sys.argv[2]
mask_base = sys.argv[3] if len(sys.argv) > 3 else None
with safe_open(ed_p, framework="np") as f:
    ed = f.get_tensor("pixels").astype(np.float32)   # [1,3,F,H,W]
with safe_open(src_p, framework="np") as f:
    src = f.get_tensor("pixel").astype(np.float32)   # [1,3,F,H,W]
assert ed.shape == src.shape, (ed.shape, src.shape)
_, C, F, H, W = ed.shape

# crude center box (subject is centered): x in [W/4,3W/4), y in [H/8,7H/8)
y0, y1 = H // 8, 7 * H // 8
x0, x1 = W // 4, 3 * W // 4
cmask = np.zeros((H, W), bool)
cmask[y0:y1, x0:x1] = True

d = np.abs(ed - src)[0]                              # [3,F,H,W]
print(f"clip {F}f {H}x{W}   center box y[{y0}:{y1}] x[{x0}:{x1}]")
print("frame   MAD_all   MAD_center   MAD_outside")
for i in range(F):
    di = d[:, i]
    print(f"{i:5d}   {di.mean():.5f}   {di[:, cmask].mean():.5f}      {di[:, ~cmask].mean():.5f}")
print(f"MEAN    {d.mean():.5f}   {d[:, :, cmask].mean():.5f}      {d[:, :, ~cmask].mean():.5f}")

# temporal consistency: consecutive-frame MAD
td_ed = np.abs(ed[0, :, 1:] - ed[0, :, :-1]).mean(axis=(0, 2, 3))
td_src = np.abs(src[0, :, 1:] - src[0, :, :-1]).mean(axis=(0, 2, 3))
print("\ntemporal MAD (frame i -> i+1):  edited vs source")
for i in range(F - 1):
    print(f"  {i:2d}->{i+1:2d}   {td_ed[i]:.5f}   {td_src[i]:.5f}")
print(f"  MEAN     {td_ed.mean():.5f}   {td_src.mean():.5f}")

# per-frame MAD vs source, split by the ACTUAL auto-mask (if provided).
if mask_base is not None:
    from PIL import Image
    GT = 1 + (F - 1) // 4
    masks = []
    for t in range(GT):
        m = np.asarray(Image.open(f"{mask_base}_mask_t{t}.png").convert("L"))
        assert m.shape == (H, W), m.shape
        masks.append(m > 127)
    print(f"\nauto-mask MAD vs source (mask_base={mask_base})")
    print("frame  latf  mask%   MAD_inside   MAD_outside")
    ins, outs = [], []
    for f in range(F):
        lt = 0 if f == 0 else (f + 3) // 4
        m = masks[lt]
        di = d[:, f]
        mi, mo = di[:, m].mean(), di[:, ~m].mean()
        ins.append(mi); outs.append(mo)
        print(f"{f:5d}  {lt:4d}  {100.0*m.mean():5.1f}   {mi:.5f}      {mo:.5f}")
    print(f"MEAN                {np.mean(ins):.5f}      {np.mean(outs):.5f}")
