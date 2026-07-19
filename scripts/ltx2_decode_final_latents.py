#!/usr/bin/env python3
"""Decode a run's final_latents.safetensors (video+audio) with the Creator
oracle, using LTX Desktop's 16 GB VAE tiling contract.
Writes frames, wav, and the muxed mp4.

Usage: ltx2_decode_final_latents.py <run_dir> [fps] [mp4_name] [latents_name]
       [--out-dir PATH]
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ltx2_hq_ref_run import CKPT  # noqa: E402  (sets LTX-2 sys.paths + OIIO stub)

import torch  # noqa: E402


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("fps", nargs="?", type=float, default=24.0)
    ap.add_argument("mp4_name", nargs="?", default="ltx2_oracle.mp4")
    ap.add_argument("latents_name", nargs="?", default="final_latents.safetensors")
    ap.add_argument(
        "--out-dir",
        help="write oracle frames/audio/video here (default: run_dir)",
    )
    args = ap.parse_args()
    d = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.out_dir or args.run_dir)
    os.makedirs(out_dir, exist_ok=True)
    fps = int(round(args.fps))
    mp4_name = args.mp4_name
    lat_file = args.latents_name
    from safetensors.torch import load_file
    from ltx_core.model.video_vae.tiling import (
        SpatialTilingConfig,
        TemporalTilingConfig,
        TilingConfig,
    )
    from ltx_pipelines.utils.blocks import AudioDecoder, VideoDecoder

    dev = torch.device("cuda")
    dtype = torch.bfloat16
    lat = load_file(os.path.join(d, lat_file))
    v = lat["video"].to(dev, dtype)
    a = lat["audio"].to(dev, dtype)
    print("video latent", tuple(v.shape), " audio latent", tuple(a.shape))

    gen = torch.Generator(device=dev).manual_seed(42)
    print("[decode] video (tiled)")
    vd = VideoDecoder(CKPT, dtype, dev)
    # Do not use TilingConfig.default(): Creator currently defaults to 768/80,
    # while the installed LTX Desktop 15 GB policy is deliberately more
    # conservative.  This explicit contract is the Mojo product parity oracle.
    tiling = TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=512,
            tile_overlap_in_pixels=64,
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=64,
            tile_overlap_in_frames=24,
        ),
    )
    print("tiling spatial=512/64 temporal=64/24")
    frames = []
    from PIL import Image
    n = 0
    for chunk in vd(v, tiling, gen):
        c8 = (chunk.float().clamp(0, 1) * 255).round().byte().cpu().numpy()
        for f in range(c8.shape[0]):
            Image.fromarray(c8[f]).save(os.path.join(out_dir, f"ref_frame{n:03d}.png"))
            n += 1
        print(f"  chunk {tuple(chunk.shape)} -> total {n} frames", flush=True)
        del chunk
    del vd
    torch.cuda.empty_cache()
    print("  frames:", n)

    print("[decode] audio")
    ad = AudioDecoder(CKPT, dtype, dev)
    audio = ad(a)
    wav = audio.waveform.float().cpu()
    print("  audio:", tuple(wav.shape), "rms", wav.pow(2).mean().sqrt().item())
    import wave as wavemod

    w = wavemod.open(os.path.join(out_dir, "ref_audio.wav"), "wb")
    w.setnchannels(wav.shape[0] if wav.dim() == 2 else 2)
    w.setsampwidth(2)
    w.setframerate(48000)
    pcm = (wav.clamp(-1, 1) * 32767).short().numpy()
    w.writeframes(pcm.T.tobytes() if wav.dim() == 2 else pcm.tobytes())
    w.close()

    # Bound the mux to the decoded video duration. Infinite apad with -shortest
    # was measured to leave a silent AAC tail. Derive this from the actual
    # latent/decode result so the Creator 121-frame HQ contract and future
    # admitted durations cannot silently drift.
    duration = float(n) / fps
    rc = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
            "-i", os.path.join(out_dir, "ref_frame%03d.png"),
            "-i", os.path.join(out_dir, "ref_audio.wav"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-af", "apad",
            "-c:a", "aac", "-t", f"{duration:.6f}",
            "-movflags", "+faststart", os.path.join(out_dir, mp4_name),
        ],
        check=False,
    ).returncode
    print("mux rc=", rc)
    print("DONE ->", os.path.join(out_dir, mp4_name))


if __name__ == "__main__":
    main()
