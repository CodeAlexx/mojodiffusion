"""MiniMax-H3 KEYFRAME presentation oracle — the COMPOSITION, on real inputs.

CPU only, no torch model, no CUDA.

── WHAT IS NEW HERE (two gated units have never been JOINED) ────────────────
`models/minimax_h3/image_grid.mojo` is gated: source size -> conditioner grid.
`models/minimax_h3/presentation.mojo` is gated: prompt + token COUNTS -> ids/tags.
Neither gate composes them, and the composition is where the three numbers a
keyframe request actually runs on come from:

  1. TEXT_TOKENS — the total presentation length. It is COMPTIME in
     minimax_h3_i2va.mojo, so getting it wrong is a rebuild, and getting it
     SILENTLY wrong shifts every media row's rotary coordinate.
  2. the per-row TAGS — a keyframe's vision block is tagged VIDEO (0) inside the
     text run, which is the one way a keyframe layout's text region differs from
     t2va's. These feed minimax_h3_build_sampling_geometry.
  3. the `<|image_pad|>` POSITIONS — where the vision tower's embeds are spliced
     in, and where the deepstack features are added at language layers 0/1/2.
     A position list that is right in LENGTH and wrong in PLACEMENT produces a
     perfectly-shaped, completely wrong conditioning.

The image processor is fed the PREPARED keyframe — already on the target canvas
(before_encoder.py:190-193 prepares them, encoders.py:154 processes them) — so
the grid is the CANVAS's grid, not the source photo's. This oracle takes a
canvas size and synthesizes an image at exactly that size to make that explicit.

Run:
    CUDA_VISIBLE_DEVICES="" python3 scripts/minimax_h3_keyframe_presentation_oracle.py
Writes: output/minimax_h3_keyframe/keyframe_presentation_ref.json
"""

import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

H3 = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA"
PROCESSOR_DIR = os.path.join(H3, "processor")
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_keyframe"

# packing.py:49-51
TEXT_TAG, VIDEO_TAG = 1, 0

# The canvases a real request resolves to, plus two awkward ones. 768x1184 and
# 1344x768 are what resolve_canvas_size produces for common aspect ratios.
CANVASES = [(768, 1184), (768, 1344), (1184, 768), (480, 832), (128, 192)]

PROMPTS = [
    "For the target video, at 0.00 seconds into the target video, <Picture 1>"
    " (from [Shot 1]) is fully referenced.\n\n"
    "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a test.\n\n"
    "overall_soundscape: Room tone.\n\n"
    "non_diegetic_music: N/A",
    "a short prompt",
]


def build_presentation(tokenizer, prompt, image_token_counts):
    """encoders.py:152-169, transcribed. Labels are 1-based and in packed order."""
    ids, tags = [], []
    for index, count in enumerate(image_token_counts):
        label = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
        ids += label
        tags += [TEXT_TAG] * len(label)
        block = ([tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                 + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * count
                 + [tokenizer.convert_tokens_to_ids("<|vision_end|>")])
        ids += block
        tags += [VIDEO_TAG] * len(block)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids += prompt_ids
    tags += [TEXT_TAG] * len(prompt_ids)
    return ids, tags


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    proc = AutoProcessor.from_pretrained(PROCESSOR_DIR)
    tok = AutoTokenizer.from_pretrained(PROCESSOR_DIR)
    ip = proc.image_processor
    merge = ip.merge_size
    pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    print(f"processor: patch {ip.patch_size} merge {merge} "
          f"min_pixels {ip.size['shortest_edge']} max_pixels {ip.size['longest_edge']}")
    print(f"<|image_pad|> id = {pad_id}")

    cases = []
    for (ch, cw) in CANVASES:
        # A REAL prepared keyframe is exactly canvas-sized; synthesize one so the
        # processor sees what it would see at runtime.
        arr = (np.mgrid[0:ch, 0:cw][0] % 251).astype(np.uint8)
        img = Image.fromarray(np.stack([arr, arr, arr], -1), "RGB")
        vision = ip(images=[img], return_tensors="pt")
        t, h, w = vision["image_grid_thw"][0].tolist()
        count = t * h * w // (merge * merge)
        for ki, nkf in enumerate((1, 2)):
            for pi, prompt in enumerate(PROMPTS):
                ids, tags = build_presentation(tok, prompt, [count] * nkf)
                pads = [i for i, v in enumerate(ids) if v == pad_id]
                cases.append({
                    "canvas_h": ch, "canvas_w": cw,
                    "grid_t": t, "grid_h": h, "grid_w": w,
                    "vision_tokens": count,
                    "num_keyframes": nkf,
                    "prompt_index": pi,
                    "prompt": prompt,
                    "num_text_tokens": len(ids),
                    "token_ids": ids,
                    "token_tags": tags,
                    "pad_positions": pads,
                })
                if ki == 0 and pi == 0:
                    print(f"  canvas {cw}x{ch} -> grid {t}x{h}x{w} = {count} vision tokens; "
                          f"{nkf} kf, prompt {pi}: {len(ids)} text tokens, "
                          f"{len(pads)} pads")

    out = {
        "image_pad_id": pad_id,
        "vision_start_id": tok.convert_tokens_to_ids("<|vision_start|>"),
        "vision_end_id": tok.convert_tokens_to_ids("<|vision_end|>"),
        "cases": cases,
    }
    path = os.path.join(OUT_DIR, "keyframe_presentation_ref.json")
    json.dump(out, open(path, "w"))
    print(f"wrote {path}: {len(cases)} cases")


if __name__ == "__main__":
    main()
