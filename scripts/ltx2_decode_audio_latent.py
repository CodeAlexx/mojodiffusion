#!/usr/bin/env python3
"""Decode an LTX2 audio latent (patchified [1,S_A,128] or unpatchified
[1,8,S_A,16]) to a 48 kHz wav with the reference AudioDecoder — early audio
verdict for in-flight runs (stage-1 audio is dumped long before decode).

Usage: ltx2_decode_audio_latent.py <latents.safetensors> <key> <out.wav>
"""
import sys

import torch

sys.path.insert(0, "/home/alex/LTX-2/packages/ltx-core/src")
sys.path.insert(0, "/home/alex/LTX-2/packages/ltx-pipelines/src")
import types

sys.modules.setdefault("OpenImageIO", types.ModuleType("OpenImageIO"))

CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"


def main() -> None:
    path, key, out = sys.argv[1], sys.argv[2], sys.argv[3]
    from safetensors.torch import load_file
    from ltx_pipelines.utils.blocks import AudioDecoder

    lat = load_file(path)[key]
    if lat.dim() == 3 and lat.shape[2] == 128:
        # patchified [1,S_A,128] -> [1,8,S_A,16]  (channels 8, mel_bins 16;
        # patchify packs (c,mel) into 128 with c-major rows: 128 = 8*16)
        b, s_a, _ = lat.shape
        lat = lat.reshape(b, s_a, 8, 16).permute(0, 2, 1, 3).contiguous()
    print("latent:", tuple(lat.shape), lat.dtype)
    dev = torch.device(sys.argv[4] if len(sys.argv) > 4 else "cuda")
    dt = torch.float32 if dev.type == "cpu" else torch.bfloat16
    ad = AudioDecoder(CKPT, dt, dev)
    audio = ad(lat.to(dev, dt))
    wav = audio.waveform.float().cpu()
    print("wav:", tuple(wav.shape), "rms", wav.pow(2).mean().sqrt().item())
    import wave as wavemod

    w = wavemod.open(out, "wb")
    w.setnchannels(wav.shape[0] if wav.dim() == 2 else 2)
    w.setsampwidth(2)
    w.setframerate(48000)
    pcm = (wav.clamp(-1, 1) * 32767).short().numpy()
    w.writeframes(pcm.T.tobytes() if wav.dim() == 2 else pcm.tobytes())
    w.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
