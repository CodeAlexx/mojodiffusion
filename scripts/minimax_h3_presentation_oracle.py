"""MiniMax-H3 prompt-presentation oracle.

Reference: diffusers PR huggingface/diffusers#14355, cloned at
/home/alex/minimax_h3_ref/diffusers-src, head e1b518df.
  modular_pipelines/minimax_h3/packing_ref2va.py  build_ref2va_presentation
  modular_pipelines/minimax_h3/encoders.py        the t2va / fl2va label loop

MiniMax-H3 does NOT chat-template its prompt. The presentation is raw token ids
with no special tokens added, labels numbered per modality, and vision blocks
spliced in as `<|vision_start|>` + N pad tokens + `<|vision_end|>`. Each token
also carries a modality tag: text 1, vision block 0 — which is what the
transformer's AdaLN keys off, so a mistagged run silently modulates rows with
the wrong parameters.

Tokenized with the REAL Qwen3-VL tokenizer we fetched, so the Mojo port is
gated against the same tokenizer.json it will load at runtime.

Vision TOKEN COUNTS are inputs here, not outputs: they come from the image
processor's grid, which is a separate unit. This oracle fixes the presentation
structure given those counts.

Run:
    python3 scripts/minimax_h3_presentation_oracle.py
Writes: output/minimax_h3_presentation/presentation_ref.safetensors
"""

import json
import os
import sys

DIFFUSERS_SRC = "/home/alex/minimax_h3_ref/diffusers-src/src"
TOKENIZER_DIR = "/home/alex/minimax_h3_ref/creator-MiniMax-H3/FL2VA/processor"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_presentation"

sys.path.insert(0, DIFFUSERS_SRC)

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from diffusers.modular_pipelines.minimax_h3.packing import (  # noqa: E402
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
)
from diffusers.modular_pipelines.minimax_h3.packing_ref2va import (  # noqa: E402
    MiniMaxH3PreparedReference,
    build_ref2va_presentation,
)

PROMPTS = {
    "plain": "A red fox trotting through a snowy pine forest, snow crunching underfoot",
    "punct": "Close-up: the subject's face, lit by neon. 35mm, f/1.4 — shallow depth!",
    "unicode": "夜の街を歩く女性、ネオンの光 — cinematic, 24fps",
    "short": "a cat",
}


def t2va_fl2va_presentation(tokenizer, prompt: str, image_token_counts: list[int]):
    """The t2va / fl2va presentation.

    Copied verbatim from `MiniMaxH3TextEncoderStep.encode_prompt` (encoders.py
    :152-169), with the vision token counts passed in rather than derived from
    the image processor — that derivation is a separate unit.
    """
    token_ids, token_tags = [], []
    for index, num_image_tokens in enumerate(image_token_counts):
        label_ids = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
        vision_ids = (
            [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
            + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        )
        token_ids += label_ids + vision_ids
        token_tags += [MINIMAX_H3_TEXT_TAG] * len(label_ids) + [MINIMAX_H3_VIDEO_TAG] * len(vision_ids)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    token_ids += prompt_ids
    token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)
    return token_ids, token_tags


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, object] = {}

    # The four special ids the presentation splices in.
    specials = {
        name: tokenizer.convert_tokens_to_ids(name)
        for name in ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>")
    }
    tensors["specials"] = torch.tensor(list(specials.values()), dtype=torch.int64)
    meta["specials"] = specials

    # 1. t2va (no keyframes) and fl2va (one and two keyframes)
    cases = [
        ("t2va_plain", "plain", []),
        ("t2va_punct", "punct", []),
        ("t2va_unicode", "unicode", []),
        ("t2va_short", "short", []),
        ("fl2va_one", "plain", [12]),
        ("fl2va_two", "plain", [12, 7]),
        ("fl2va_unicode", "unicode", [3]),
    ]
    for name, prompt_key, counts in cases:
        ids, tags = t2va_fl2va_presentation(tokenizer, PROMPTS[prompt_key], counts)
        tensors[f"{name}.ids"] = torch.tensor(ids, dtype=torch.int64)
        tensors[f"{name}.tags"] = torch.tensor(tags, dtype=torch.int64)
        meta[name] = {"prompt": PROMPTS[prompt_key], "image_token_counts": counts, "len": len(ids)}

    # 2. ref2va, through the reference's own builder
    def image_ref():
        return MiniMaxH3PreparedReference(kind="image")

    def audio_ref():
        return MiniMaxH3PreparedReference(kind="audio", has_audio=True)

    def video_ref(block_timestamps, has_audio):
        return MiniMaxH3PreparedReference(
            kind="video", has_audio=has_audio, block_timestamps=list(block_timestamps)
        )

    # The 0.25 / 0.75 timestamps are the interesting ones: `"{:.1f}"` rounds half
    # to EVEN, so they render "0.2" and "0.8", not "0.3" and "0.8".
    ref_cases = [
        ("ref_image", [image_ref()], [9], []),
        ("ref_audio", [audio_ref()], [], []),
        ("ref_video_silent", [video_ref([0.25, 1.25], False)], [], [6]),
        ("ref_video_sound", [video_ref([0.25, 1.25, 2.25], True)], [], [6]),
        (
            "ref_mixed",
            [image_ref(), video_ref([0.25, 0.75], True), audio_ref()],
            [9],
            [4],
        ),
        (
            "ref_reordered",
            [audio_ref(), video_ref([0.25, 0.75], True), image_ref()],
            [9],
            [4],
        ),
        ("ref_two_images", [image_ref(), image_ref()], [9, 5], []),
    ]
    for name, references, image_counts, video_counts in ref_cases:
        ids, tags = build_ref2va_presentation(
            tokenizer, PROMPTS["plain"], references, image_counts, video_counts
        )
        tensors[f"{name}.ids"] = torch.tensor(ids, dtype=torch.int64)
        tensors[f"{name}.tags"] = torch.tensor(tags, dtype=torch.int64)
        meta[name] = {
            "kinds": [r.kind for r in references],
            "has_audio": [r.has_audio for r in references],
            "block_timestamps": [r.block_timestamps for r in references],
            "image_token_counts": image_counts,
            "video_block_token_counts": video_counts,
            "len": len(ids),
        }

    # 3. The label strings themselves, tokenized in isolation — so a gate can
    # localize a mismatch to the BPE rather than to the presentation.
    labels = [
        "<Picture 1>: ", "<Picture 2>: ", "<Picture 9>: ",
        "<Audio 1>: ", "<Audio 3>: ",
        "<Video 1>: ", "<Video 2>: ",
        "<0.2 seconds>", "<0.8 seconds>", "<1.2 seconds>", "<10.5 seconds>",
    ]
    for index, label in enumerate(labels):
        tensors[f"label.{index}"] = torch.tensor(
            tokenizer(label, add_special_tokens=False)["input_ids"], dtype=torch.int64
        )
    meta["labels"] = labels

    # 4. Python's `"{:.1f}"` on the timestamps a 2 fps pairing produces.
    stamps = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.75, 2.25, 10.5, 0.05, 0.15]
    meta["format_1f"] = {str(s): f"{s:.1f}" for s in stamps}
    tensors["format_1f.values"] = torch.tensor(stamps, dtype=torch.float64)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "presentation_ref.safetensors")
    save_file(tensors, path, metadata={"format": "pt"})
    with open(os.path.join(OUT_DIR, "presentation_ref.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"wrote {len(tensors)} tensors -> {path}")
    print("specials:", specials)
    print("format_1f:", meta["format_1f"])
    for name in [c[0] for c in cases] + [c[0] for c in ref_cases]:
        print(f"  {name:<18} len={meta[name]['len']}")


if __name__ == "__main__":
    main()
