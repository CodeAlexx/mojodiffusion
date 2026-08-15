"""MiniMax-H3 ref2va REAL-conditioning oracle: the vendor's own presentation
(token ids + tags), video-processor grid and pixel patches for ONE concrete
request — the trimmed reference clip + the vendor 768p prompt.

Reference: diffusers PR huggingface/diffusers#14355, cloned at
/home/alex/minimax_h3_ref/diffusers-src, head e1b518df.
  packing_ref2va.py  resample_reference_frames, prepare_reference_frames,
                     sample_reference_video_frames, build_ref2va_presentation
  encoders.py        MiniMaxH3Ref2VATextEncoderStep.encode_prompt :399-427
                     (the video_processor call + video_block_token_counts law)

INPUT PARITY: the frames are decoded with the EXACT ffmpeg command
`pipeline/minimax_h3_media_in.mojo::minimax_h3_ffmpeg_extract_rgb` runs
(`ffmpeg -v error -y -i X -f rawvideo -pix_fmt rgb24 out.rgb`), so the Mojo
probe and this oracle read byte-identical pixels. Everything after that is the
vendor's own functions plus the REAL Qwen3-VL processor + tokenizer from the
Ref2VA checkpoint.

The request mirrors the pipeline invocation exactly:
  references = [video:ref_video_trim.mp4 (with its own soundtrack),
                audio:ref_audio.wav]
  target 832x480, 22 frames  ->  the reference video is truncated to 22 frames
  prompt = output/minimax_h3_prompts/ref2va_vendor_768p.txt (3554 bytes)

Run:
    /home/alex/torchref/venv/bin/python scripts/minimax_h3_ref2va_conditioning_oracle.py
Writes: output/minimax_h3_ref2va/conditioning_oracle.safetensors (+ .json)
"""

import hashlib
import json
import os
import subprocess
import sys

DIFFUSERS_SRC = "/home/alex/minimax_h3_ref/diffusers-src/src"
PROCESSOR_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/Ref2VA/processor"
CLIP = "/home/alex/mojodiffusion/output/h3_ref2va_media/ref_video_trim.mp4"
PROMPT_FILE = "/home/alex/mojodiffusion/output/minimax_h3_prompts/ref2va_vendor_768p.txt"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_ref2va"
TARGET_FRAMES = 22  # the generated video's frame count (17n+5), H3_FRAMES=22

sys.path.insert(0, DIFFUSERS_SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoTokenizer, Qwen3VLProcessor  # noqa: E402

from diffusers.modular_pipelines.minimax_h3.packing_ref2va import (  # noqa: E402
    MiniMaxH3PreparedReference,
    build_ref2va_presentation,
    prepare_reference_frames,
    resample_reference_frames,
    sample_reference_video_frames,
)


def ffprobe_geometry(path: str) -> tuple[int, int, float]:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json", path,
        ]
    )
    stream = json.loads(out)["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return int(stream["width"]), int(stream["height"]), float(num) / float(den)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    width, height, fps = ffprobe_geometry(CLIP)
    rgb_path = os.path.join(OUT_DIR, "oracle_trim.rgb")
    # The EXACT extraction command the Mojo pipeline runs (media_in.mojo:407-410).
    subprocess.check_call(
        f"ffmpeg -v error -y -i '{CLIP}' -f rawvideo -pix_fmt rgb24 '{rgb_path}'",
        shell=True,
    )
    raw = np.fromfile(rgb_path, dtype=np.uint8)
    frame_bytes = height * width * 3
    assert raw.size % frame_bytes == 0, "rgb dump is not a whole number of frames"
    frames = raw.reshape(-1, height, width, 3)
    print(f"decoded {frames.shape[0]} frames {width}x{height} @ {fps} fps")

    # Vendor order (before_encoder.py:373-375): 24 fps resample, then the
    # canvas LANCZOS resize + truncation to the target's frame count.
    frames = resample_reference_frames(frames, fps)
    prepared = prepare_reference_frames(frames, TARGET_FRAMES)
    print(f"prepared: {prepared.shape} (canvas {prepared.shape[2]}x{prepared.shape[1]})")

    sampled_frames, block_timestamps = sample_reference_video_frames(prepared)
    print(f"conditioner sees {len(sampled_frames)} frames, blocks at {block_timestamps}")

    processor = Qwen3VLProcessor.from_pretrained(PROCESSOR_DIR)
    tokenizer = AutoTokenizer.from_pretrained(PROCESSOR_DIR)
    merge_size = processor.image_processor.merge_size**2

    # encoders.py:409-423, verbatim for the single-video case.
    vision = processor.video_processor(
        videos=[np.stack(sampled_frames)], do_sample_frames=False, return_tensors="pt"
    )
    pixel_values_videos = vision["pixel_values_videos"]
    video_grid_thw = vision["video_grid_thw"]
    video_block_token_counts = [int(grid[1]) * int(grid[2]) // merge_size for grid in video_grid_thw]
    print(f"video_grid_thw={video_grid_thw.tolist()} block_token_counts={video_block_token_counts}")
    assert int(video_grid_thw[0][0]) == len(block_timestamps), (
        f"processor merged into {int(video_grid_thw[0][0])} blocks, "
        f"H3 labels {len(block_timestamps)}"
    )

    references = [
        MiniMaxH3PreparedReference(
            kind="video", has_audio=True, block_timestamps=block_timestamps
        ),
        MiniMaxH3PreparedReference(kind="audio", has_audio=True),
    ]

    prompt = open(PROMPT_FILE, "rb").read().decode("utf-8")
    token_ids, token_tags = build_ref2va_presentation(
        tokenizer, prompt, references, [], video_block_token_counts
    )
    print(f"presentation: {len(token_ids)} tokens")

    tensors = {
        "token_ids": torch.tensor(token_ids, dtype=torch.int64),
        "token_tags": torch.tensor(token_tags, dtype=torch.int64),
        "video_grid_thw": video_grid_thw.to(torch.int64),
        "block_timestamps": torch.tensor(block_timestamps, dtype=torch.float64),
        "sampled_indices": torch.tensor(
            [i for i in range(prepared.shape[0])
             if any(np.array_equal(prepared[i], f) for f in sampled_frames)],
            dtype=torch.int64,
        ),
        "pixel_values_videos": pixel_values_videos.to(torch.float32),
        "prepared_shape": torch.tensor(list(prepared.shape), dtype=torch.int64),
    }
    out_path = os.path.join(OUT_DIR, "conditioning_oracle.safetensors")
    save_file(tensors, out_path)

    sidecar = {
        "clip": CLIP,
        "prompt_file": PROMPT_FILE,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_bytes": len(prompt.encode("utf-8")),
        "target_frames": TARGET_FRAMES,
        "source": f"{width}x{height}@{fps}",
        "decoded_frames": int(raw.size // frame_bytes),
        "canvas": [int(prepared.shape[1]), int(prepared.shape[2])],
        "num_sampled_frames": len(sampled_frames),
        "block_timestamps": block_timestamps,
        "video_grid_thw": video_grid_thw.tolist(),
        "video_block_token_counts": video_block_token_counts,
        "num_text_tokens": len(token_ids),
        "num_video_tagged": int(sum(1 for t in token_tags if t == 0)),
    }
    with open(os.path.join(OUT_DIR, "conditioning_oracle.json"), "w") as f:
        json.dump(sidecar, f, indent=1)
    print(f"wrote {out_path}")
    print(json.dumps(sidecar, indent=1))


if __name__ == "__main__":
    main()
