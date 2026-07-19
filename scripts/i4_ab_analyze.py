#!/usr/bin/env python3
"""Analyze the ideogram4 Rust-vs-Mojo A/B (MJ-1047/MJ-1051).

Latent-space: cos / relL2 / per-channel bias between the two stacks' final
latents (identical inputs -> any difference is pipeline drift, and its
STRUCTURE distinguishes bias (tint-capable) from noise-like drift).
Pixel-space: channel means + mean|diff| for every image pair of interest.
"""
import sys, json, struct
import numpy as np

AB = sys.argv[1] if len(sys.argv) > 1 else "/home/alex/mojodiffusion/output/checks/i4_ab"

def load_st(path, name="tensor"):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        info = hdr[name]
        off = info["data_offsets"]
        f.seek(8 + n + off[0])
        buf = f.read(off[1] - off[0])
    dt = {"F32": np.float32, "BF16": None}[info["dtype"]]
    if dt is None:
        raw = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
        return raw.view(np.float32).reshape(info["shape"]).astype(np.float32)
    return np.frombuffer(buf, dtype=dt).reshape(info["shape"]).astype(np.float32)

def latent_compare(a, b, la, lb):
    fa, fb = a.ravel(), b.ravel()
    cos = float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-12))
    rel = float(np.linalg.norm(fa - fb) / (np.linalg.norm(fb) + 1e-12))
    print(f"\n[latent] {la} vs {lb}: cos {cos:.8f}  relL2 {rel:.6f}")
    # channel structure: [1, 1024, 128] -> per-channel over dim1
    ca, cb = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    dmean = ca.mean(0) - cb.mean(0)
    print(f"  per-channel mean-bias: max|d| {np.abs(dmean).max():.5f}  "
          f"mean|d| {np.abs(dmean).mean():.5f}  (noise-like drift ~0; a few big channels = structured bias)")
    top = np.argsort(-np.abs(dmean))[:6]
    print("  top-bias channels:", ", ".join(f"ch{c}:{dmean[c]:+.4f}" for c in top))
    print(f"  std ratio (a/b): {ca.std():.5f}/{cb.std():.5f} = {ca.std()/cb.std():.4f}")

def img_stats(path):
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return im

def img_compare(pa, pb, la, lb):
    a, b = img_stats(pa), img_stats(pb)
    if a.shape != b.shape:
        print(f"\n[image] {la} vs {lb}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return
    d = np.abs(a - b)
    print(f"\n[image] {la} vs {lb}: mean|d| {d.mean():.3f}  max|d| {d.max():.0f}  "
          f"pct>8 {100*(d.max(-1)>8).mean():.1f}%")
    for im, l in ((a, la), (b, lb)):
        r, g, bl = im[..., 0].mean(), im[..., 1].mean(), im[..., 2].mean()
        print(f"  {l:14s} R{r:6.1f} G{g:6.1f} B{bl:6.1f}   G-B {g-bl:+5.1f}  B-R {bl-r:+5.1f}")

rust = load_st(f"{AB}/rust_final_latent.safetensors")
mojo = load_st(f"{AB}/mojo_sched_final_latent.safetensors")
latent_compare(mojo, rust, "mojo_sched", "rust")

img_compare(f"{AB}/mojo_sched_whole.png", f"{AB}/rust.png", "mojo_whole", "rust")
img_compare(f"{AB}/mojo_sched_tiled.png", f"{AB}/mojo_sched_whole.png", "mojo_tiled", "mojo_whole")
img_compare(f"{AB}/mojo_const7_whole.png", f"{AB}/mojo_sched_whole.png", "mojo_const7", "mojo_sched")

print("\nVERDICT GUIDE:")
print("  latent cos ~0.999+ & flat channel bias  -> kernels/plumbing equivalent; gap is recipe/UI-path")
print("  latent cos low OR big few-channel bias  -> measured Mojo denoise drift (kernel hypothesis) -> bisect per-step dumps")
print("  tiled vs whole visibly different        -> MJ-1051 tiling cost confirmed")
print("  const7 vs sched visibly different       -> MJ-1051 constant-CFG cost confirmed")
