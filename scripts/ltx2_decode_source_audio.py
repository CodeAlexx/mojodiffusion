#!/usr/bin/env python3
"""Decode an LTX Retake/Extend source exactly as the creator pipeline does.

This copies the media-I/O contract, not model logic, from the LTX Desktop
locked `ltx_pipelines.utils.media_io.decode_audio_from_file` implementation:
PyAV frames, native planar/interleaved sample conversion, time-window trimming,
and no resampling.  Mojo owns the torchaudio-equivalent resample, log-mel, and
learned AudioVAE encoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np


_INT_FORMAT_MAX = {
    "u8": 128.0,
    "u8p": 128.0,
    "s16": 32768.0,
    "s16p": 32768.0,
    "s32": 2147483648.0,
    "s32p": 2147483648.0,
}


def _audio_frame_to_float(frame: av.AudioFrame) -> np.ndarray:
    fmt = frame.format.name
    array = frame.to_ndarray().astype(np.float32)
    if fmt in _INT_FORMAT_MAX:
        array = array / _INT_FORMAT_MAX[fmt]
    if not frame.format.is_planar:
        channels = len(frame.layout.channels)
        array = array.reshape(-1, channels).T
    return array


def decode(path: str, max_duration: float) -> tuple[np.ndarray, int]:
    container = av.open(path)
    try:
        audio_stream = next(stream for stream in container.streams if stream.type == "audio")
    except StopIteration:
        container.close()
        raise RuntimeError("source has no audio stream")

    sample_rate = audio_stream.rate
    start_time = 0.0
    start_pts = int(start_time / audio_stream.time_base)
    end_time = start_time + max_duration
    container.seek(start_pts, stream=audio_stream)

    samples: list[np.ndarray] = []
    first_frame_time: float | None = None
    for frame in container.decode(audio=0):
        if frame.pts is None:
            continue
        frame_time = float(frame.pts * audio_stream.time_base)
        frame_end = frame_time + frame.samples / frame.sample_rate
        if frame_end < start_time:
            continue
        if frame_time > end_time:
            break
        if first_frame_time is None:
            first_frame_time = frame_time
        samples.append(_audio_frame_to_float(frame))
    container.close()

    if not samples or first_frame_time is None:
        raise RuntimeError("source audio decoded no samples")
    audio = np.concatenate(samples, axis=-1)
    skip_samples = round((start_time - first_frame_time) * sample_rate)
    if skip_samples > 0:
        audio = audio[..., skip_samples:]
    max_samples = round(max_duration * sample_rate)
    audio = audio[..., :max_samples]
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-duration", required=True, type=float)
    args = parser.parse_args()
    audio, sample_rate = decode(args.input, args.max_duration)
    if audio.shape[0] != 2:
        raise RuntimeError(
            f"creator LTX audio encoder requires stereo input; got {audio.shape[0]} channels"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # f32le raw is sample-major/interleaved; Mojo converts it back to [C,T].
    audio.T.astype("<f4", copy=False).tofile(output)
    print(
        json.dumps(
            {
                "schema": "serenity.ltx2.creator_audio_decode.v1",
                "path": str(output),
                "sample_rate": sample_rate,
                "channels": int(audio.shape[0]),
                "samples_per_channel": int(audio.shape[1]),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
