#!/usr/bin/env python3
"""Build a small LTX-2 legacy-trainer v2 cache from local mp4s (5080 box).

Same math as serenitymojo/training/ltx2_cache_v2_converter.py (the PROVEN v2
recipe that produced the 2026-07-08 training anchor), with the disney
dataset.json replaced by a directory of mp4s:

  middle frame -> center square crop -> 512x512 -> [1,3,1,512,512] in [-1,1]
  -> AutoencoderKLLTX2Video.encode().latent_dist.mode()   [1,128,1,16,16]
  -> (x - latents_mean) * scaling / latents_std           <- the v2 fix
  -> _pack_latents(p=1)                                   [256,128] f32
  -> save {"latent": ...} per sample

Gate: norm-std ~1.0 class (vs raw ~0.16).
Run (CPU-only): /home/alex/ai-toolkit/venv/bin/python \
    scripts/ltx2_build_cache_v2_local.py --limit 12
"""
import argparse, glob, os, time
import torch
import torch.nn.functional as F
import imageio.v3 as iio

torch.set_num_threads(8)
torch.manual_seed(0)

VIDEO_DIR = "/mnt/disk2/output_comfui"
LTX2_VAE = "/home/alex/.serenity/models/checkpoints/ltx2-diffusers/vae"
OUT = "/home/alex/datasets/ltx2_cache_512_v2"
RES = 512


def load_center_frame(mp4_path):
    frames = iio.imread(mp4_path, plugin="pyav")            # [T,H,W,3] uint8
    t = frames.shape[0] // 2
    img = torch.from_numpy(frames[t].copy()).float() / 127.5 - 1.0
    img = img.permute(2, 0, 1)
    _, h, w = img.shape
    s = min(h, w)
    img = img[:, (h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    img = F.interpolate(img.unsqueeze(0), size=(RES, RES), mode="bilinear", align_corners=False)
    return img.unsqueeze(2)                                 # [1,3,1,512,512]


def pack_latents(latents, patch_size=1, patch_size_t=1):
    b, c, f, h, w = latents.shape
    pf, ph, pw = f // patch_size_t, h // patch_size, w // patch_size
    latents = latents.reshape(b, -1, pf, patch_size_t, ph, patch_size, pw, patch_size)
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def normalize_latents(latents, latents_mean, latents_std, scaling_factor=1.0):
    m = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    s = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return (latents - m) * scaling_factor / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--videos", default=VIDEO_DIR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--vae", default=LTX2_VAE)
    args = ap.parse_args()

    from safetensors.torch import save_file
    from diffusers import AutoencoderKLLTX2Video

    os.makedirs(args.out, exist_ok=True)
    vids = sorted(glob.glob(os.path.join(args.videos, "*.mp4")))
    if args.limit:
        vids = vids[: args.limit]
    if not vids:
        raise SystemExit(f"no mp4s in {args.videos}")

    print("[load] AutoencoderKLLTX2Video (F32, CPU) ...")
    vae = AutoencoderKLLTX2Video.from_pretrained(args.vae, torch_dtype=torch.float32).to("cpu").eval()
    lat_mean = vae.latents_mean.float()
    lat_std = vae.latents_std.float()
    scaling = float(vae.config.scaling_factor)
    print(f"[load] latents_mean |.|={lat_mean.abs().mean():.4f}  latents_std mean={lat_std.mean():.4f}  scaling={scaling}")

    stds, means = [], []
    t_start = time.time()
    for i, vp in enumerate(vids):
        name = os.path.splitext(os.path.basename(vp))[0]
        t0 = time.time()
        frame = load_center_frame(vp)
        with torch.no_grad():
            enc = vae.encode(frame)
            raw = enc.latent_dist.mode() if hasattr(enc, "latent_dist") else enc.latents
            norm = normalize_latents(raw, lat_mean, lat_std, scaling)
            ll = pack_latents(norm).squeeze(0).contiguous().float()   # [256,128]
        raw_std = float(raw.std())
        n_std = float(ll.std()); n_mean = float(ll.mean())
        stds.append(n_std); means.append(n_mean)
        save_file({"latent": ll}, os.path.join(args.out, f"{name}.safetensors"))
        print(f"[{i+1:>2}/{len(vids)}] {name}  latent[256,128] raw_std={raw_std:.4f} "
              f"-> norm mean={n_mean:+.4f} std={n_std:.4f}  ({time.time()-t0:.1f}s)")

    import statistics as st
    print(f"\n[gate] {len(stds)} samples  norm-std: min={min(stds):.3f} "
          f"mean={st.mean(stds):.3f} max={max(stds):.3f}  |mean|-avg={st.mean([abs(m) for m in means]):.4f}")
    print(f"[gate] expect norm-std ~1.0 class; total {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
