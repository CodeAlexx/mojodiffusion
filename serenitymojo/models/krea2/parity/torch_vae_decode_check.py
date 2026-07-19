#!/usr/bin/env python3
# DECODE-vs-LATENT isolation: decode the SAME Mojo latent with torch (F32, the
# reference VAE) and compare per-channel RGB means to the Mojo bf16 render.
#   torch-F32 decode WARM (R>G, matches data)  -> Mojo bf16 DECODE is the culprit.
#   torch-F32 decode GREEN (like Mojo)         -> the LATENT is green (denoise), decode innocent.
import sys, struct, numpy as np, torch
from PIL import Image
from diffusers import AutoencoderKLQwenImage

MEAN = [-0.7571,-0.7089,-0.9113,0.1075,-0.1745,0.9653,-0.1517,1.5508,
        0.4134,-0.0715,0.5517,-0.3632,-0.1922,-0.9497,0.2503,-0.2921]
STD  = [2.8184,1.4541,2.3275,2.6558,1.2196,1.7708,2.6052,2.0743,
        3.2687,2.1526,2.8652,1.5579,1.6382,1.1253,2.8251,1.9160]

def read_mojo_bin(p):
    with open(p,'rb') as f:
        magic=struct.unpack('<q',f.read(8))[0]
        dtag =struct.unpack('<q',f.read(8))[0]
        rank =struct.unpack('<q',f.read(8))[0]
        shape=[struct.unpack('<q',f.read(8))[0] for _ in range(rank)]
        raw=f.read()
    # latent is F32 (dtag for F32); numel*4 bytes
    n=int(np.prod(shape))
    arr=np.frombuffer(raw[:n*4], dtype=np.float32).reshape(shape).copy()
    return arr, shape

def chan(a):  # a: HWC uint8/float
    return float(a[...,0].mean()), float(a[...,1].mean()), float(a[...,2].mean())

lat_path=sys.argv[1]
vae_dir=sys.argv[2]
out_png=sys.argv[3]
lat, shape = read_mojo_bin(lat_path)     # [1,16,LH,LW] model-space (normalized)
print("latent", shape, "mean/std", float(lat.mean()), float(lat.std()))

dev="cuda"
vae=AutoencoderKLQwenImage.from_pretrained(vae_dir, torch_dtype=torch.float32).to(dev).eval()
z=torch.from_numpy(lat).to(dev, torch.float32)          # [1,16,LH,LW]
m=torch.tensor(MEAN,device=dev).view(1,16,1,1)
s=torch.tensor(STD,device=dev).view(1,16,1,1)
zu=z*s+m                                                  # unnormalize (inverse of (z-m)/s)
# Qwen-Image VAE is 3D-causal: decode expects [B,C,T,H,W].
zu5=zu.unsqueeze(2)                                       # T=1
with torch.no_grad():
    img=vae.decode(zu5).sample                           # F32 decode, [-1,1]
img=img.float().clamp(-1,1)[:, :, 0]                     # drop T -> [1,3,H,W]
arr=((img+1.0)*127.5).round().clamp(0,255).byte().permute(0,2,3,1)[0].cpu().numpy()
Image.fromarray(arr).save(out_png)
r,g,b=chan(arr.astype(np.float32))
print(f"torch-F32 decode  R={r:.1f} G={g:.1f} B={b:.1f}  G-R={g-r:+.1f}  B-R={b-r:+.1f}  -> {out_png}")
print("(compare: Mojo bf16 render R=102 G=111 B=93 G-R=+9 ; eri2 data R=114 G=97 B=91 G-R=-17.5)")
