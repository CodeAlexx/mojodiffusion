"""MiniMax-H3 KEYFRAME encode oracle — REAL released weights. NEEDS THE GPU.

Produces the reference for
`serenitymojo/pipeline/parity/minimax_h3_keyframe_encode_probe.mojo`: the whole
I2VA / FL2VA / L2VA conditioning chain, stage by stage, so a mismatch localizes
to a stage instead of to "the keyframe is wrong".

    1. pixel normalize   ImageNet over [0,1] (packing.py:70-71)
    2. _adaptive_encode  the tiled SPATIAL encoder, one frame  <- the GPU part
    3. posterior SAMPLE  torch.Generator().manual_seed(42)     (encoders.py:297)
    4. float16 round     .to(torch.float16).float()            (encoders.py:300)
    5. normalize         (z - latents_mean) / latents_std
    6. patchify          patchify_video_latents, channel-SLOWEST
    7. mix at 0.999      scheduler.scale_noise(rows, 0.999, keyframe noise)

WHICH ENCODE ENTRY POINT, AND WHY: the diffusers rewrite calls `_encode_clip`;
the CREATOR bundle (which is what the released checkpoint's config.json
auto_maps to, and what serenitymojo's device encoder was ported from) calls
`_adaptive_encode`. They are the same spatial-tiled single-clip path under two
names. This oracle uses the creator's, because the creator's is the code the
released weights ship with — `load_state_dict(strict=True)` in
minimax_h3_video_vae.py proves there is no key conversion between them.

STAGES 3-7 ARE THE DIFFUSERS PIPELINE'S, NOT THE CREATOR'S. `encode_base`
(klvae.py:1231) samples WITHOUT a generator and never rounds to fp16; the seeded
sample and the fp16 round are `MiniMaxH3KeyframeVaeEncoderStep`'s, and they are
what the released model was conditioned with. Mixing the two APIs here is
deliberate and is the point of the comment.

Run (GPU, ~11 GiB for the VAE in fp32):
    python3 scripts/minimax_h3_keyframe_encode_oracle.py <prepared_keyframe.png> \
        output/minimax_h3_keyframe/keyframe_encode_ref.safetensors [request_seed]

The keyframe must ALREADY be on the target canvas — `prepare_keyframe_image`'s
output, not a raw photo. That split is intentional: the resize is gated
separately and bit-exactly against PIL
(pipeline/parity/minimax_h3_keyframe_image_probe.mojo), so this gate is about
the VAE chain alone and cannot fail for a resampling reason.

    --dry-run   validate imports, the input image and every non-VAE stage
                WITHOUT loading weights or touching CUDA. Runnable while the GPU
                is held.
"""

import argparse
import os
import sys

import numpy as np

REPO = "/home/alex/mojodiffusion"
H3_ROOT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA"
VIDEO_VAE_DIR = os.path.join(H3_ROOT, "video_vae")
VIDEO_VAE_SOURCE = os.path.join(VIDEO_VAE_DIR, "source")
DIFFUSERS_SRC = "/home/alex/minimax_h3_ref/diffusers-src/src"

# packing.py:70-71 / :84 / :87
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
KEYFRAME_NOISE_AUG = 0.999
KEYFRAME_ENCODE_SEED = 42
PATCH = (1, 2, 2)
LATENT_CHANNELS = 24


def patchify_video_latents(latents, patch_size):
    """packing.py:246-275, transcribed — CHANNEL-SLOWEST within a row."""
    import torch

    patch_t, patch_h, patch_w = patch_size
    b, c, f, h, w = latents.shape
    assert f % patch_t == 0 and h % patch_h == 0 and w % patch_w == 0
    x = latents.reshape(b, c, f // patch_t, patch_t, h // patch_h, patch_h, w // patch_w, patch_w)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(-1, c * patch_t * patch_h * patch_w).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("seed", nargs="?", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    from PIL import Image
    from safetensors.torch import save_file

    image = Image.open(args.image).convert("RGB")
    arr = np.array(image)
    print(f"keyframe {args.image}: {arr.shape[1]}x{arr.shape[0]}")
    if arr.shape[0] % 16 or arr.shape[1] % 16:
        raise SystemExit(
            f"the keyframe must be canvas-aligned (a multiple of the VAE's 16), got "
            f"{arr.shape[1]}x{arr.shape[0]} — run prepare_keyframe_image first"
        )

    device = "cpu" if args.dry_run else "cuda"
    pixel_mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1, 1)

    # STAGE 1 — encoders.py:291-292, byte for byte.
    pixels = torch.from_numpy(arr).to(device).permute(2, 0, 1)[None, :, None]
    pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
    print(f"stage 1 pixels {tuple(pixels.shape)} mean={pixels.mean().item():+.6f}")

    tensors = {"pixels_normalized": pixels.float().cpu()}

    if args.dry_run:
        print("--dry-run: skipping the VAE (stages 2-3); "
              "checking the input plumbing only")
        latent_h, latent_w = arr.shape[0] // 16, arr.shape[1] // 16
        torch.manual_seed(0)
        moments = torch.randn(1, 2 * LATENT_CHANNELS, 1, latent_h, latent_w)
    else:
        sys.path.insert(0, os.path.dirname(VIDEO_VAE_DIR))
        from video_vae.klvae import AutoencoderKLLegacy  # noqa: E402
        from video_vae.parallel import get_parallel_state  # noqa: E402

        state = get_parallel_state()
        if not state:
            state.update({
                "group_size": 1, "group_rank": 0, "local_process_group": None,
                "sp_size": 1, "sp_rank": 0, "sp_enabled": False,
                "sp_process_group": None, "tp_size": 1, "tp_rank": 0,
            })
        import json
        import safetensors.torch

        cfg = json.load(open(os.path.join(VIDEO_VAE_SOURCE, "config.json")))
        cfg.pop("_class_name", None)
        cfg.pop("_diffusers_version", None)
        vae = AutoencoderKLLegacy(**cfg)
        sd = safetensors.torch.load_file(os.path.join(VIDEO_VAE_SOURCE, "model.safetensors"))
        missing, unexpected = vae.load_state_dict(sd, strict=False)
        print(f"loaded VAE: {len(missing)} missing, {len(unexpected)} unexpected keys")
        vae = vae.to(device).eval()
        # The released config turns tiling ON by default; keep it, because that
        # is what a real request runs and what the Mojo side reproduces.
        with torch.no_grad():
            moments = vae._adaptive_encode(pixels)
    print(f"stage 2 moments {tuple(moments.shape)}")
    tensors["moments"] = moments.float().cpu()

    # STAGE 3 — the SEEDED posterior sample. The clamp is
    # DiagonalGaussianDistribution's (diffusers vae.py:691).
    mean, logvar = torch.chunk(moments, 2, dim=1)
    logvar = torch.clamp(logvar, -30.0, 20.0)
    std = torch.exp(0.5 * logvar)
    noise = torch.randn(
        mean.shape, generator=torch.Generator().manual_seed(KEYFRAME_ENCODE_SEED),
        dtype=torch.float32,
    ).to(mean.device)
    sample = mean + std * noise
    tensors["posterior_noise"] = noise.float().cpu()
    tensors["sample"] = sample.float().cpu()

    # STAGES 4-6.
    latents = sample.to(torch.float16).float().cpu()
    tensors["sample_fp16"] = latents.clone()
    lm = torch.tensor(json_latents("latents_mean")).view(1, -1, 1, 1, 1)
    ls = torch.tensor(json_latents("latents_std")).view(1, -1, 1, 1, 1)
    normalized = (latents - lm) / ls
    tensors["normalized"] = normalized.clone()
    rows = patchify_video_latents(normalized, PATCH)
    tensors["condition_rows_clean"] = rows.clone()
    print(f"stages 4-6 rows {tuple(rows.shape)}")

    # STAGE 7 — the request's OWN generator draws the conditioning noise, in
    # packed order, BEFORE any target noise (packing.py:511-515).
    gen = torch.Generator().manual_seed(args.seed)
    cond_noise = torch.randn(
        (1, LATENT_CHANNELS, 1, moments.shape[-2], moments.shape[-1]),
        generator=gen, dtype=torch.float32,
    )
    noise_rows = patchify_video_latents(cond_noise, PATCH)
    tensors["condition_noise_rows"] = noise_rows.clone()

    sys.path.insert(0, DIFFUSERS_SRC)
    from diffusers import MiniMaxH3Scheduler  # noqa: E402

    scheduler = MiniMaxH3Scheduler(shift=12.0)
    mixed = scheduler.scale_noise(rows, KEYFRAME_NOISE_AUG, noise_rows)
    tensors["condition_rows_mixed"] = mixed.float().cpu()
    print(f"stage 7 mixed rows {tuple(mixed.shape)} "
          f"mean={mixed.mean().item():+.6f} std={mixed.std().item():.6f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, args.out)
    print("wrote", args.out)
    for k, v in tensors.items():
        print(f"  {k:22s} {tuple(v.shape)}")


_CFG_CACHE = {}


def json_latents(key):
    import json
    if not _CFG_CACHE:
        _CFG_CACHE.update(json.load(open(os.path.join(VIDEO_VAE_DIR, "config.json"))))
    return _CFG_CACHE[key]


if __name__ == "__main__":
    main()
