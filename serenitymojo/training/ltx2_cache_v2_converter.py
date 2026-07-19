#!/usr/bin/env python3
"""Build the CORRECT LTX-2 Mojo training cache (v2) from the disney dataset.

CPU-only. Run with the ai-toolkit venv:
  /home/alex/ai-toolkit/venv/bin/python /home/alex/disney_cache_tools/build_ltx2_cache_v2.py [--limit N]

WHAT THIS FIXES vs v1 (/home/alex/datasets/ltx2_cache_512):
  v1 (build_disney_caches.py) saved the RAW VAE posterior mean (std ~0.164) and
  SKIPPED the LTX-2 per-channel latent normalization. The wan22 arm in that same
  script DID normalize ((z-mean)/std); the ltx2 arm did not. The diffusers LTX2
  pipeline ALWAYS normalizes encoded latents before they enter the transformer:
    diffusers/pipelines/ltx2/pipeline_ltx2.py:566  _normalize_latents(...)
    diffusers/pipelines/ltx2/pipeline_ltx2.py:572  (latents - mean)*scaling/std
    diffusers/pipelines/ltx2/pipeline_ltx2.py:660  called from prepare_latents
  The per-channel stats are REAL buffers in the VAE checkpoint:
    vae/diffusion_pytorch_model.safetensors :: latents_mean[128], latents_std[128] (BF16)
  So the base transformer expects std ~1.0 normalized latents; feeding raw std~0.164
  latents is out-of-distribution -> the wrong-latent loss class.

CACHE CONTRACT (measured from train_ltx2_real.mojo):
  key "latent"  -> [256,128] f32, token-major channel-last.
    line 403  var lat = _load_host(st, "latent", ctx)   # reads flat, first S*IN_CH
    line 114  S = GRID_F*GRID_H*GRID_W = 1*16*16 = 256
    line 100  IN_CH = 128
  Token order = (F,H,W)-major, channel last  == diffusers _pack_latents(p=1,p_t=1):
    pipeline_ltx2.py:530-550 permute(0,2,4,6,1,3,5,7).flatten(4,7).flatten(1,3)
    which for patch=1 reduces to latent[0].permute(1,2,3,0).reshape(-1,128).
  transformer config: patch_size=1, patch_size_t=1 -> IN_CH stays 128 (no repack).

ENCODE (identical to v1 EXCEPT the added normalization, so it is a clean A/B):
  middle frame -> center square crop -> 512x512 -> [1,3,1,512,512] in [-1,1]
  -> AutoencoderKLLTX2Video.encode(...).latent_dist.mode()  -> [1,128,1,16,16]
  -> _normalize_latents((x-mean)/std, scaling=1.0)          <-- THE FIX
  -> _pack_latents(p=1)                                     -> [256,128]
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F
import imageio.v3 as iio

torch.set_num_threads(8)
torch.manual_seed(0)

DATASET  = "/home/alex/disney/dataset.json"
LTX2_VAE = "/home/alex/.serenity/models/checkpoints/ltx2-diffusers/vae"  # AutoencoderKLLTX2Video
LTX2_OUT = "/home/alex/datasets/ltx2_cache_512_v2"
RES      = 512


def load_center_frame(mp4_path):
    """Middle frame -> center square crop -> 512x512 -> [1,3,1,512,512] in [-1,1]."""
    frames = iio.imread(mp4_path, plugin="pyav")            # [T,H,W,3] uint8
    t = frames.shape[0] // 2
    img = torch.from_numpy(frames[t].copy()).float() / 127.5 - 1.0  # [H,W,3] in [-1,1]
    img = img.permute(2, 0, 1)                              # [3,H,W]
    _, h, w = img.shape
    s = min(h, w)
    img = img[:, (h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    img = F.interpolate(img.unsqueeze(0), size=(RES, RES), mode="bilinear", align_corners=False)
    return img.unsqueeze(2)                                 # [1,3,1,512,512]


def pack_latents(latents, patch_size=1, patch_size_t=1):
    """Faithful copy of diffusers pipeline_ltx2._pack_latents (patch=1 here).

    [B,C,F,H,W] -> [B, F*H*W, C]  (token order F,H,W-major; channel last).
    """
    b, c, f, h, w = latents.shape
    pf, ph, pw = f // patch_size_t, h // patch_size, w // patch_size
    latents = latents.reshape(b, -1, pf, patch_size_t, ph, patch_size, pw, patch_size)
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def normalize_latents(latents, latents_mean, latents_std, scaling_factor=1.0):
    """Faithful copy of diffusers pipeline_ltx2._normalize_latents."""
    m = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    s = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return (latents - m) * scaling_factor / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only process first N samples (validation)")
    ap.add_argument("--out", default=LTX2_OUT, help="output cache dir")
    args = ap.parse_args()

    from safetensors.torch import save_file
    from diffusers import AutoencoderKLLTX2Video

    os.makedirs(args.out, exist_ok=True)

    print("[load] AutoencoderKLLTX2Video (F32, CPU) ...")
    vae = AutoencoderKLLTX2Video.from_pretrained(LTX2_VAE, torch_dtype=torch.float32).to("cpu").eval()
    lat_mean = vae.latents_mean.float()          # [128] real per-channel stats from ckpt
    lat_std = vae.latents_std.float()            # [128]
    scaling = float(vae.config.scaling_factor)   # 1.0
    print(f"[load] latents_mean |.|={lat_mean.abs().mean():.4f}  latents_std mean={lat_std.mean():.4f}  scaling={scaling}")

    ds = json.load(open(DATASET))
    if args.limit:
        ds = ds[:args.limit]
    n = len(ds)
    print(f"[run] {n} samples -> {args.out}")

    stds, means = [], []
    t_start = time.time()
    for i, e in enumerate(ds):
        vp = e["video_path"]
        hash_ = os.path.splitext(os.path.basename(vp))[0]
        t0 = time.time()

        frame = load_center_frame(vp)                       # [1,3,1,512,512]
        with torch.no_grad():
            enc = vae.encode(frame)
            raw = enc.latent_dist.mode() if hasattr(enc, "latent_dist") else enc.latents  # [1,128,1,16,16]
            norm = normalize_latents(raw, lat_mean, lat_std, scaling)                      # [1,128,1,16,16]
            ll = pack_latents(norm).squeeze(0).contiguous().float()                        # [256,128]

        raw_std = float(raw.std())
        n_std = float(ll.std()); n_mean = float(ll.mean())
        stds.append(n_std); means.append(n_mean)

        save_file({"latent": ll}, os.path.join(args.out, f"{hash_}.safetensors"))
        dt = time.time() - t0
        print(f"[{i+1:>3}/{n}] {hash_}  latent[256,128] raw_std={raw_std:.4f} "
              f"-> norm mean={n_mean:+.4f} std={n_std:.4f}  ({dt:.1f}s)")

    if stds:
        import statistics as st
        print(f"\n[gate] {len(stds)} samples  norm-std: min={min(stds):.3f} "
              f"mean={st.mean(stds):.3f} max={max(stds):.3f}  |mean|-avg={st.mean([abs(m) for m in means]):.4f}")
        print(f"[gate] expect norm-std ~1.0 class (vs raw ~0.16 class); total {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
