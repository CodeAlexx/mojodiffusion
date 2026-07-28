#!/usr/bin/env python3
"""Apply the Lightricks LTX image-conditioning CRF round trip.

This intentionally mirrors `ltx_pipelines.utils.media_io.preprocess` without
importing the full Torch pipeline.  PyAV owns both the RGB-to-YUV conversion and
the libx264 encode/decode so the pixels match the creator implementation.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import av
import numpy as np
from PIL import Image


def preprocess(image: np.ndarray, crf: int) -> np.ndarray:
    if crf == 0:
        return image

    with BytesIO() as output_file:
        container = av.open(output_file, "w", format="mp4")
        try:
            stream = container.add_stream(
                "libx264",
                rate=1,
                options={"crf": str(crf), "preset": "veryfast"},
            )
            height = image.shape[0] // 2 * 2
            width = image.shape[1] // 2 * 2
            image = image[:height, :width]
            stream.height = height
            stream.width = width
            frame = av.VideoFrame.from_ndarray(image, format="rgb24").reformat(
                format="yuv420p"
            )
            container.mux(stream.encode(frame))
            container.mux(stream.encode())
        finally:
            container.close()
        payload = output_file.getvalue()

    with BytesIO(payload) as video_file:
        container = av.open(video_file)
        try:
            stream = next(row for row in container.streams if row.type == "video")
            frame = next(container.decode(stream))
            return frame.to_ndarray(format="rgb24")
        finally:
            container.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--crf", type=int, default=33)
    args = parser.parse_args()
    if not 0 <= args.crf <= 51:
        raise SystemExit("--crf must be in [0, 51]")
    image = np.asarray(Image.open(args.input).convert("RGB"))
    Image.fromarray(preprocess(image, args.crf)).save(args.output, format="PNG")


if __name__ == "__main__":
    main()
