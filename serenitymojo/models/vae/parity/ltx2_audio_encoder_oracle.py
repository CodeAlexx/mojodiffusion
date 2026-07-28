#!/usr/bin/env python3
"""Dump the pinned LTX Desktop source-audio encoder contract.

This is an oracle producer only.  It must run in LTX Desktop's exact locked
Python environment and imports the creator packages directly; it does not
transcribe or approximate their implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from ltx_core.model.audio_vae import encode_audio
from ltx_core.model.audio_vae.ops import AudioProcessor
from ltx_pipelines.utils.blocks import AudioConditioner
from ltx_pipelines.utils.media_io import (
    decode_audio_from_file,
    get_videostream_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    shape = get_videostream_metadata(args.video)
    duration = shape.frames / shape.fps
    decoded = decode_audio_from_file(
        args.video,
        device=device,
        start_time=0.0,
        max_duration=duration,
    )
    if decoded is None:
        raise RuntimeError("oracle source has no audio stream")

    dumped: dict[str, torch.Tensor] = {}

    def run(encoder: torch.nn.Module) -> torch.Tensor:
        processor = AudioProcessor(
            target_sample_rate=encoder.sample_rate,
            mel_bins=encoder.mel_bins,
            mel_hop_length=encoder.mel_hop_length,
            n_fft=encoder.n_fft,
        ).to(device=device)
        resampled = processor.resample_audio(decoded)
        mel = processor.waveform_to_mel(decoded)
        latent_unconformed = encode_audio(decoded, encoder, processor)
        required = round(duration * 16000 / 160 / 4)
        latent = latent_unconformed
        if latent.shape[2] > required:
            latent = latent[:, :, :required]
        elif latent.shape[2] < required:
            pad = torch.zeros(
                (*latent.shape[:2], required - latent.shape[2], latent.shape[3]),
                device=latent.device,
                dtype=latent.dtype,
            )
            latent = torch.cat((latent, pad), dim=2)
        dumped.update(
            waveform_native=decoded.waveform.detach().float().cpu().contiguous(),
            waveform_16k=resampled.waveform.detach().float().cpu().contiguous(),
            mel=mel.detach().float().cpu().contiguous(),
            latent_unconformed=latent_unconformed.detach().float().cpu().contiguous(),
            latent=latent.detach().float().cpu().contiguous(),
            metadata=torch.tensor(
                [
                    decoded.sampling_rate,
                    decoded.waveform.shape[-1],
                    resampled.waveform.shape[-1],
                    shape.frames,
                    shape.fps,
                    duration,
                    required,
                ],
                dtype=torch.float64,
            ),
        )
        return latent

    AudioConditioner(
        checkpoint_path=args.checkpoint,
        dtype=dtype,
        device=device,
    )(run)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(dumped, output)
    for name, tensor in dumped.items():
        print(
            f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"min={tensor.min().item():.7g} max={tensor.max().item():.7g} "
            f"mean={tensor.mean().item():.7g}"
        )


if __name__ == "__main__":
    main()
