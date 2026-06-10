#!/usr/bin/env python3
"""Decode the EXACT Mojo-generated LTX2 latents with the official ltx_core stack.

Quality-funnel step 1 (LTX2_INFERENCE_TRAINER_HANDOFF §6): the Mojo pipeline
produces coherent-but-distorted faces. Decoding the Mojo final latents with the
reference decoder separates:
  - frames STILL distorted  -> the generated latent is bad (loop/conditioning)
  - frames clean            -> the Mojo VAE decode path is at fault

Reference pinned to /home/alex/LTX-2 — MEASURED 2026-06-09: the LTX-2-official
copy's VideoDecoderConfigurator ignores `decoder_base_channels` (the 22B VAE
needs 256) and fails load_state_dict on the 22B ckpt; the LTX-2 copy reads it
and is the runnable reference. ltx_core ONLY (ltx_pipelines pulls OpenImageIO).

Run:
  /home/alex/serenityflow-v2/.venv/bin/python scripts/ltx2_decode_latent_oracle.py \
      [latents.safetensors] [out_dir]
Defaults: output/ltx2_mvp/final_latents.safetensors -> output/ltx2_decode_oracle/
"""
import os
import sys

sys.path.insert(0, "/home/alex/LTX-2/packages/ltx-core/src")

import torch
from safetensors.torch import load_file, save_file

CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
LATENTS = sys.argv[1] if len(sys.argv) > 1 else "output/ltx2_mvp/final_latents.safetensors"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "output/ltx2_decode_oracle"


def save_frames_png(video_fhwc01: torch.Tensor, out_dir: str, prefix: str) -> int:
    # [F,H,W,C] float in [0,1] (ltx_core decode_video chunk format)
    from PIL import Image
    import numpy as np

    v = (video_fhwc01.float().clamp(0, 1) * 255.0).round().byte().cpu().numpy()
    for f in range(v.shape[0]):
        Image.fromarray(v[f]).save(os.path.join(out_dir, f"{prefix}{f:02d}.png"))
    return int(v.shape[0])


def build(configurator, sd_ops, dev, dtype):
    import json
    import safetensors

    from ltx_core.loader.sft_loader import (
        SafetensorsModelStateDictLoader,
        SafetensorsStateDictLoader,
    )

    loader = SafetensorsModelStateDictLoader()
    config = loader.metadata(CKPT)
    model = configurator.from_config(config)
    sd = SafetensorsStateDictLoader().load(CKPT, sd_ops=sd_ops, device=torch.device(dev))
    missing, unexpected = model.load_state_dict(sd.sd, strict=False)
    print(f"  {configurator.__name__}: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("   first missing:", missing[:3])
    return model.to(dev, dtype).eval()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    import ltx_core

    assert "/LTX-2/" in ltx_core.__file__, f"wrong ltx_core: {ltx_core.__file__}"
    print("ltx_core:", ltx_core.__file__)

    from ltx_core.model.video_vae.model_configurator import (
        VAE_DECODER_COMFY_KEYS_FILTER,
        VideoDecoderConfigurator,
    )
    from ltx_core.model.audio_vae.model_configurator import (
        AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
        VOCODER_COMFY_KEYS_FILTER,
        AudioDecoderConfigurator,
        VocoderConfigurator,
    )
    from ltx_core.model.audio_vae.audio_vae import decode_audio as vae_decode_audio

    dumped = load_file(LATENTS)
    video_x = dumped["video_x"]
    audio_x = dumped.get("audio_x")
    print("video_x:", tuple(video_x.shape), video_x.dtype,
          "std", video_x.float().std().item())
    if audio_x is not None:
        print("audio_x:", tuple(audio_x.shape), audio_x.dtype,
              "std", audio_x.float().std().item())

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    decoder = build(VideoDecoderConfigurator, VAE_DECODER_COMFY_KEYS_FILTER, dev, dtype)
    gen = torch.Generator(device=dev).manual_seed(0)
    lat = video_x.to(dev, dtype)
    with torch.inference_mode():
        chunks = [c for c in decoder.decode_video(lat, None, gen)]
    video = torch.cat(chunks, dim=0)  # [F,H,W,C] in [0,1]
    print("decoded video:", tuple(video.shape))
    n = save_frames_png(video, OUT_DIR, "oracle_frame")
    print(f"  wrote {n} frames -> {OUT_DIR}/oracle_frame*.png")

    out_tensors = {"decoded_video": video.float().cpu()}

    if audio_x is not None:
        try:
            adec = build(AudioDecoderConfigurator, AUDIO_VAE_DECODER_COMFY_KEYS_FILTER, dev, dtype)
            voc = build(VocoderConfigurator, VOCODER_COMFY_KEYS_FILTER, dev, dtype)
            with torch.inference_mode():
                audio = vae_decode_audio(audio_x.to(dev, dtype), adec, voc)
            wav = audio.waveform.float().cpu() if hasattr(audio, "waveform") else audio.float().cpu()
            print("decoded audio:", tuple(wav.shape),
                  "rms", wav.pow(2).mean().sqrt().item())
            out_tensors["decoded_audio"] = wav
        except Exception as e:  # report loudly, video verdict still stands
            print("AUDIO DECODE FAILED (video verdict still valid):", repr(e))

    save_file(out_tensors, os.path.join(OUT_DIR, "oracle_decode.safetensors"))
    print("wrote", os.path.join(OUT_DIR, "oracle_decode.safetensors"))


if __name__ == "__main__":
    main()
