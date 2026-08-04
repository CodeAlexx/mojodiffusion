"""MiniMax-H3 ref2va reference-ENCODE oracle — REAL Ref2VA weights. NEEDS THE GPU.

Producer side of the contract in
`serenitymojo/models/vae/parity/minimax_h3_ref_encode_gate.mojo` (read that
header first — the key set, shapes and bars are fixed THERE, this script only
implements them). One safetensors, eight keys:

    in.frames        uint8 [T, H, W, 3]  the prepared reference frames,
                                         channels-last, at the reference's own
                                         canvas, 24 fps, trimmed to 17n+5
    in.is_image      int32 []            0 — this is a VIDEO reference (the
                                         `_encode` path, not `_encode_clip`)
    out.pixels       f32 [3, T, H, W]    after (x/255 - mean)/std
    out.moments      f32 [1, 2C, T', H', W']  raw VAE moments
    out.sample       f32 [1, C, T', H', W']   posterior.sample(gen seed 42),
                                         BEFORE the fp16 round
    out.rows         f32 [N, C*pt*ph*pw] fp16 round -> normalize -> patchify
    out.noise        f32 [N, C*pt*ph*pw] keyframe_condition_noise draw
    out.rows_mixed   f32 [N, C*pt*ph*pw] scale_noise(rows, 0.999, noise)

EVERY stage is the vendor's own code, not a transcription:

  * frame prep — `MiniMaxH3Reference(video=...)` decodes via PyAV
    (packing_ref2va.py:276-301), then the VIDEO branch of
    `MiniMaxH3Ref2VASetupStep.prepare_references` verbatim: 24 fps CFR
    resample + own-aspect 768-canvas rescale + truncation
    (before_encoder.py:373-375). `prepare_references` itself is NOT called
    because its audio branch needs torchaudio (packing_ref2va.py:745-753)
    and this venv has none — the `num_frames` derive it would have done
    (before_encoder.py:336-361: duration of the single audio-bearing
    reference -> `align_num_frames(round(duration * 24))`) is replicated
    with the same functions and the same bound checks.
  * trim + pixel normalize — encoders.py:574-576, byte for byte, ON the GPU.
  * encode — the CREATOR bundle's `encode_temporal` (klvae.py:461-512), the
    same spatial-tiled temporal-chunk path the diffusers rewrite calls
    `_encode` (autoencoder_kl_minimax_h3.py:772-795 — pad-last-frame to a
    clip_length multiple, per-clip `_encode_clip`/`_adaptive_encode`, concat,
    drop `token_drop` trailing moment frames; verified line-for-line
    equivalent for the released config's all-False isolated_* flags). The
    creator's code is used because it is what the released checkpoint ships
    with — `minimax_h3_video_vae.py`'s `load_state_dict(strict=True)` proves
    there is no key conversion — same reasoning as
    scripts/minimax_h3_keyframe_encode_oracle.py. The production
    `load_kwargs` come from the OUTER video_vae/config.json exactly as
    `minimax_h3_video_vae.py:90-101` builds them; without them the class
    default `token_drop=0` would build an oracle for a DIFFERENT chunking
    than the one that ships (the vvae temporal oracle learned this).
  * posterior sample — the diffusers PR's own `DiagonalGaussianDistribution`
    (vae.py:687-709) under `torch.Generator().manual_seed(42)`
    (encoders.py:584-585; the seed is packing.py:87 and is independent of
    the request seed). `randn_tensor` draws on the CPU because the generator
    is a CPU one, and moves the result — that is the vendor's own behavior.
  * fp16 round -> normalize -> patchify — encoders.py:588-593, byte for
    byte, on the CPU exactly where the vendor's `.cpu()` puts it, with the
    vendor's own `patchify_video_latents`.
  * noise + mix — `keyframe_condition_noise` (packing.py:501-538) off a
    request generator (seed 0 here — the request seed is free in this gate;
    it is an INPUT to the value chain, not something the Mojo side must
    re-derive), then `MiniMaxH3Scheduler.scale_noise(rows, 0.999, noise)` on
    the device (encoders.py:617-630).

Run (GPU, VAE in fp32; check nvidia-smi first):
    /home/alex/OneTrainer/venv/bin/python scripts/minimax_h3_ref_encode_oracle.py
Writes: output/minimax_h3_ref2va/ref_encode_ref.safetensors (+ .json sidecar)
"""

import json
import os
import sys
import time

DIFFUSERS_SRC = "/home/alex/minimax_h3_ref/diffusers-src/src"
CKPT_ROOT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/Ref2VA"
VIDEO_VAE_DIR = os.path.join(CKPT_ROOT, "video_vae")
VIDEO_VAE_SOURCE = os.path.join(VIDEO_VAE_DIR, "source")
REF_VIDEO = "/home/alex/mojodiffusion/output/h3_ref2va_media/ref_video.mp4"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_ref2va"
OUT_PATH = os.path.join(OUT_DIR, "ref_encode_ref.safetensors")

# The request generator's seed — draws `out.noise` only. Free in this gate's
# contract (the gate consumes the noise as an input), recorded in the sidecar.
REQUEST_SEED = 0

# The pinned PR clone FIRST, before any `import diffusers`: the venv's own
# diffusers is a different commit and has no minimax_h3 modular pipeline.
sys.path.insert(0, DIFFUSERS_SRC)

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from diffusers import MiniMaxH3Scheduler  # noqa: E402
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution  # noqa: E402
from diffusers.modular_pipelines.minimax_h3.packing import (  # noqa: E402
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MIN_DURATION,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    align_num_frames,
    keyframe_condition_noise,
    patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3.packing_ref2va import (  # noqa: E402
    MiniMaxH3Reference,
    prepare_reference_frames,
    resample_reference_frames,
    reference_media_to_uint8,
    trim_reference_num_frames,
)

PATCH = (1, 2, 2)  # transformer.config.patch_size, the same value every
# minimax-h3 oracle in this repo uses (scripts/minimax_h3_keyframe_encode_oracle.py)
LATENT_CHANNELS = 24  # video_vae/config.json `latent_channels`


def load_vendor_vae(device: torch.device):
    """The creator bundle, loaded the way `minimax_h3_video_vae.py` loads it.

    Same recipe as the session's proven vvae_temporal_oracle_gen.py: source
    config + the OUTER config.json's `load_kwargs` (minimax_h3_video_vae.py:
    90-101), `_ensure_vae_parallel_state()`, fp32, eval."""
    sys.path.insert(0, CKPT_ROOT)
    from video_vae.klvae import AutoencoderKLLegacy  # noqa: E402
    from video_vae.minimax_h3_video_vae import _ensure_vae_parallel_state  # noqa: E402

    import safetensors.torch as st

    with open(os.path.join(VIDEO_VAE_SOURCE, "config.json")) as f:
        cfg = json.load(f)
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    with open(os.path.join(VIDEO_VAE_DIR, "config.json")) as f:
        outer = json.load(f)
    cfg.update(
        {
            "clip_length": int(outer["vae_clip_length"]),
            "token_drop": int(outer["vae_token_drop"]),
            "encoder_tiling": int(outer["vae_encoder_tiling"]),
            "decoder_tiling": int(outer["vae_decoder_tiling"]),
            "parallel_tiling": int(outer["vae_parallel_tiling"]),
            "tile_size": int(outer["vae_tile_size"]),
            "tile_overlap_min": int(outer["vae_tile_overlap_min"]),
            "encoder_parallel": int(outer["vae_encoder_parallel"]),
            "decoder_parallel": int(outer["vae_decoder_parallel"]),
            "chunk_dim": int(outer["vae_chunk_dim"]),
        }
    )
    _ensure_vae_parallel_state()
    vae = AutoencoderKLLegacy(**cfg)
    sd = st.load_file(os.path.join(VIDEO_VAE_SOURCE, "model.safetensors"))
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    print(f"loaded VAE: {len(missing)} missing, {len(unexpected)} unexpected keys", flush=True)
    if missing:
        raise SystemExit(f"missing weights: {missing[:8]} ...")
    vae = vae.to(device, dtype=torch.float32).eval()
    print(
        "vendor attrs:",
        "clip_length", vae.clip_length, "token_drop", vae.token_drop,
        "vae_ratio_t", vae.vae_ratio_t, "encoder_tiling", vae.encoder_tiling,
        "tile_size", vae.tile_size, "tile_overlap_min", vae.tile_overlap_min,
        flush=True,
    )
    return vae


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda")

    # 1. Decode the reference the vendor's way (packing_ref2va.py:276-301).
    reference = MiniMaxH3Reference(video=REF_VIDEO)
    src = reference_media_to_uint8(reference.video)
    print(
        f"decoded {REF_VIDEO}: {src.shape[0]} frames {src.shape[2]}x{src.shape[1]} "
        f"@ {reference.fps:g} fps, soundtrack "
        f"{tuple(reference.audio.shape) if reference.audio is not None else None} "
        f"@ {reference.sample_rate} Hz ({time.time() - t0:.1f}s)",
        flush=True,
    )
    if reference.audio is None:
        raise SystemExit("the reference video carries no soundtrack — the "
                         "num_frames derive (before_denoise.py:336-361) needs it")

    # 2. `num_frames` derive, before_encoder.py:336-361 verbatim: the duration
    #    of the single audio-bearing reference, aligned up to 17n+5.
    duration = reference.audio.shape[-1] / reference.sample_rate
    if not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
        raise SystemExit(f"reference is {duration:g}s, outside "
                         f"{MINIMAX_H3_MIN_DURATION}-{MINIMAX_H3_MAX_DURATION}s")
    num_frames = align_num_frames(round(duration * MINIMAX_H3_FPS))
    if num_frames / MINIMAX_H3_FPS > MINIMAX_H3_MAX_DURATION:
        raise SystemExit("aligned frame count past the duration ceiling")
    print(f"duration {duration:.4f}s -> num_frames {num_frames}", flush=True)

    # 3. The VIDEO branch of prepare_references, before_encoder.py:373-375.
    frames = resample_reference_frames(src, float(reference.fps))
    frames = prepare_reference_frames(frames, num_frames)
    print(f"prepared frames: {frames.shape[0]} @ {frames.shape[2]}x{frames.shape[1]} "
          f"(canvas of the reference's own aspect)", flush=True)

    # 4. Trim to 17n+5 and pixel-normalize ON the GPU, encoders.py:574-576.
    trimmed = frames[: trim_reference_num_frames(frames.shape[0])]
    print(f"trimmed to {trimmed.shape[0]} frames (17n+5)", flush=True)
    pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    pixels = torch.from_numpy(trimmed.copy()).to(device).permute(3, 0, 1, 2)[None]
    pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
    print(f"pixels {tuple(pixels.shape)} mean={pixels.mean().item():+.6f}", flush=True)

    tensors = {
        "in.frames": torch.from_numpy(trimmed.copy()),
        "in.is_image": torch.tensor(0, dtype=torch.int32),
        "out.pixels": pixels[0].float().cpu(),
    }

    # 5. The `_encode` path: creator `encode_temporal` (klvae.py:461-512).
    vae = load_vendor_vae(device)
    t_enc = time.time()
    with torch.no_grad():
        moments = vae.encode_temporal(pixels)
    torch.cuda.synchronize()
    print(f"moments {tuple(moments.shape)} ({time.time() - t_enc:.1f}s encode)", flush=True)
    tensors["out.moments"] = moments.float().cpu()

    # 6. Posterior SAMPLE off a fresh generator seeded 42 (encoders.py:584-585).
    posterior = DiagonalGaussianDistribution(moments)
    latents = posterior.sample(
        generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
    )
    tensors["out.sample"] = latents.float().cpu()

    # 7. fp16 round -> normalize -> patchify (encoders.py:588-593).
    latents = latents.to(torch.float16).float().cpu()
    with open(os.path.join(VIDEO_VAE_DIR, "config.json")) as f:
        vae_cfg = json.load(f)
    latents_mean = torch.tensor(vae_cfg["latents_mean"]).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae_cfg["latents_std"]).view(1, -1, 1, 1, 1)
    rows = patchify_video_latents((latents - latents_mean) / latents_std, PATCH)
    tensors["out.rows"] = rows.clone()
    print(f"rows {tuple(rows.shape)}", flush=True)

    # 8. The condition noise + the 0.999 mix (encoders.py:617-630).
    num_latent_frames, latent_h, latent_w = latents.shape[2], latents.shape[3], latents.shape[4]
    noise = keyframe_condition_noise(
        ((num_latent_frames, latent_h, latent_w),),
        PATCH,
        LATENT_CHANNELS,
        generator=torch.Generator().manual_seed(REQUEST_SEED),
        device=device,
    )
    scheduler = MiniMaxH3Scheduler(shift=12.0)
    mixed = scheduler.scale_noise(rows.to(device), MINIMAX_H3_KEYFRAME_NOISE_AUG, noise)
    tensors["out.noise"] = noise.float().cpu()
    tensors["out.rows_mixed"] = mixed.float().cpu()
    print(f"mixed rows {tuple(mixed.shape)} mean={mixed.mean().item():+.6f} "
          f"std={mixed.std().item():.6f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, OUT_PATH, metadata={"format": "pt"})
    meta = {
        "ref_video": REF_VIDEO,
        "source_frames": int(src.shape[0]),
        "source_hw": [int(src.shape[1]), int(src.shape[2])],
        "source_fps": float(reference.fps),
        "soundtrack_sample_rate": int(reference.sample_rate),
        "duration_seconds": float(duration),
        "num_frames": int(num_frames),
        "prepared_hw": [int(frames.shape[1]), int(frames.shape[2])],
        "trimmed_frames": int(trimmed.shape[0]),
        "latent_tchw": [int(num_latent_frames), int(latent_h), int(latent_w)],
        "encode_seed": int(MINIMAX_H3_KEYFRAME_ENCODE_SEED),
        "request_seed": REQUEST_SEED,
        "noise_aug": float(MINIMAX_H3_KEYFRAME_NOISE_AUG),
        "patch_size": list(PATCH),
        "vae_dtype": "float32",
    }
    with open(OUT_PATH.replace(".safetensors", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"wrote {len(tensors)} tensors, {total / (1 << 20):.1f} MiB -> {OUT_PATH} "
          f"({time.time() - t0:.1f}s total)", flush=True)
    for k, v in tensors.items():
        print(f"  {k:<16} {str(v.dtype):<14} {tuple(v.shape)}", flush=True)


if __name__ == "__main__":
    main()
