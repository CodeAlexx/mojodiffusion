# serenitymojo/models/minimax_h3/presentation.mojo
#
# MiniMax-H3 prompt presentation — unit 7 of the H3 port.
#
# H3 does NOT chat-template its prompt. The presentation is raw token ids with
# no special tokens added: labels numbered per modality, vision blocks spliced
# in as `<|vision_start|>` + N pad tokens + `<|vision_end|>`, then the prompt
# verbatim. Every token also carries a modality tag — text 1, vision block 0 —
# and that tag is what the transformer's AdaLN keys off, so a mistagged run
# modulates those rows with the wrong parameters and nothing crashes.
#
# Source (op-for-op port, diffusers PR huggingface/diffusers#14355 at head
# e1b518dfd5e390e7ba09a79a1d39fe1c6cb52dc1):
#   src/diffusers/modular_pipelines/minimax_h3/encoders.py
#     MiniMaxH3TextEncoderStep.encode_prompt  :152-169  (t2va / fl2va)
#   src/diffusers/modular_pipelines/minimax_h3/packing_ref2va.py
#     build_ref2va_presentation               :756-819  (ref2va)
#
# Order is a contract in ref2va: references are labelled in REQUEST order,
# numbered per modality, and a video that carries sound emits its
# `"<Audio j>: "` label BEFORE its `"<Video k>: "` label — mirroring the order
# its rows are packed in. Reordering the same references is a different request.
#
# Vision TOKEN COUNTS are inputs, not outputs: they come from the image
# processor's grid (`prod(grid_thw) // merge_size**2`), which is a separate
# unit. This module fixes the presentation structure given those counts.
#
# Tokenization runs on `Qwen3Tokenizer` over the conditioner's own
# tokenizer.json, so the ids here are the ids the runtime will produce.

from std.collections import List
from std.math import floor

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer

comptime MINIMAX_H3_TEXT_TAG = 1
comptime MINIMAX_H3_VIDEO_TAG = 0

comptime MINIMAX_H3_VISION_START = "<|vision_start|>"
comptime MINIMAX_H3_VISION_END = "<|vision_end|>"
comptime MINIMAX_H3_IMAGE_PAD = "<|image_pad|>"
comptime MINIMAX_H3_VIDEO_PAD = "<|video_pad|>"

# Reference kinds, matching packing_ref2va.mojo.
comptime MINIMAX_H3_REF_IMAGE = 0
comptime MINIMAX_H3_REF_VIDEO = 1
comptime MINIMAX_H3_REF_AUDIO = 2


@fieldwise_init
struct MiniMaxH3Presentation(Copyable, Movable):
    """Token ids and the per-token modality tag."""

    var ids: List[Int]
    var tags: List[Int]


@fieldwise_init
struct MiniMaxH3PresentationReference(Copyable, Movable):
    """One reference as the presentation sees it.

    `block_timestamps` is empty except for a video; `has_audio` is true for an
    audio reference and for a video whose soundtrack the request supplied."""

    var kind: Int
    var has_audio: Bool
    var block_timestamps: List[Float64]
    var vision_token_count: Int


def minimax_h3_format_seconds(value: Float64) raises -> String:
    """Python's `"{:.1f}"` for a timestamp, as `"<{t:.1f} seconds>"` renders it.

    Python formats the EXACT BINARY value and breaks ties to even, which is not
    the same as rounding the decimal you meant. Measured against CPython:
    `0.25 -> "0.2"` and `0.75 -> "0.8"` (real ties, resolved to even), while
    `0.05 -> "0.1"` and `0.15 -> "0.1"` come out equal for OPPOSITE reasons —
    0.05 as a double sits just above the midpoint and 0.15 just below it. A
    `floor(x*10 + 0.5)` shortcut gets 0.15 wrong; a scale-then-round gets 0.05
    wrong.

    H3's timestamps are always multiples of 0.25: a conditioner frame is at
    `index / 2.0` and a merged block's stamp is the mean of two of those. Those
    are exactly representable, `value * 10` is exact for them, and ties-to-even
    on that product is then correct. This function REFUSES anything else rather
    than return a string CPython would not have produced."""
    if value < 0.0:
        raise Error("MiniMax-H3: a block timestamp cannot be negative")
    var quarters = value * 4.0
    if quarters != floor(quarters):
        raise Error(
            "MiniMax-H3: block timestamps are multiples of 0.25; formatting"
            " anything else needs exact decimal conversion, not this shortcut"
        )

    var scaled = value * 10.0
    var lower = floor(scaled)
    var frac = scaled - lower
    var tenths: Float64
    if frac > 0.5:
        tenths = lower + 1.0
    elif frac < 0.5:
        tenths = lower
    else:
        # Exact tie -> even.
        tenths = lower if (Int(lower) % 2) == 0 else lower + 1.0

    var whole = Int(tenths) // 10
    var decimal = Int(tenths) % 10
    return String(whole) + "." + String(decimal)


def minimax_h3_special_id(ref tokenizer: Qwen3Tokenizer, name: String) raises -> Int:
    """Resolve an added/special token to its id."""
    for i in range(len(tokenizer.special_tokens)):
        if tokenizer.special_tokens[i] == name:
            return tokenizer.special_ids[i]
    raise Error(String("MiniMax-H3: tokenizer has no special token ") + name)


def _emit_text(
    ref tokenizer: Qwen3Tokenizer,
    mut ids: List[Int],
    mut tags: List[Int],
    text: String,
) raises:
    var encoded = tokenizer.encode(text)
    for i in range(len(encoded)):
        ids.append(encoded[i])
        tags.append(MINIMAX_H3_TEXT_TAG)


def _emit_vision(
    mut ids: List[Int],
    mut tags: List[Int],
    start_id: Int,
    pad_id: Int,
    end_id: Int,
    num_tokens: Int,
):
    """`<|vision_start|>` + N pad + `<|vision_end|>`, all tagged video."""
    ids.append(start_id)
    tags.append(MINIMAX_H3_VIDEO_TAG)
    for _ in range(num_tokens):
        ids.append(pad_id)
        tags.append(MINIMAX_H3_VIDEO_TAG)
    ids.append(end_id)
    tags.append(MINIMAX_H3_VIDEO_TAG)


def minimax_h3_t2va_presentation(
    ref tokenizer: Qwen3Tokenizer,
    prompt: String,
    image_token_counts: List[Int],
) raises -> MiniMaxH3Presentation:
    """The t2va / fl2va presentation (encoders.py:152-169).

    t2va is the prompt verbatim. Every keyframe prepends `"<Picture i>: "` and a
    vision block, numbered from 1 in packed order."""
    var start_id = minimax_h3_special_id(tokenizer, String(MINIMAX_H3_VISION_START))
    var end_id = minimax_h3_special_id(tokenizer, String(MINIMAX_H3_VISION_END))
    var pad_id = minimax_h3_special_id(tokenizer, String(MINIMAX_H3_IMAGE_PAD))

    var ids = List[Int]()
    var tags = List[Int]()
    for index in range(len(image_token_counts)):
        _emit_text(tokenizer, ids, tags, "<Picture " + String(index + 1) + ">: ")
        _emit_vision(ids, tags, start_id, pad_id, end_id, image_token_counts[index])
    _emit_text(tokenizer, ids, tags, prompt)
    return MiniMaxH3Presentation(ids^, tags^)


def minimax_h3_ref2va_presentation(
    ref tokenizer: Qwen3Tokenizer,
    prompt: String,
    references: List[MiniMaxH3PresentationReference],
) raises -> MiniMaxH3Presentation:
    """The ref2va presentation (packing_ref2va.py:756-819).

    Per reference, in request order and numbered per modality:
      image -> `"<Picture i>: "` + vision block
      audio -> `"<Audio j>: "` alone (a waveform never reaches the conditioner)
      video -> `"<Video k>: "` + one `"<t.t seconds>"` + vision block per
               merged frame pair
    A reference that carries audio emits its `"<Audio j>: "` label FIRST,
    including a video with a soundtrack. The prompt follows verbatim."""
    var start_id = minimax_h3_special_id(tokenizer, String(MINIMAX_H3_VISION_START))
    var end_id = minimax_h3_special_id(tokenizer, String(MINIMAX_H3_VISION_END))
    var image_pad = minimax_h3_special_id(tokenizer, String(MINIMAX_H3_IMAGE_PAD))
    var video_pad = minimax_h3_special_id(tokenizer, String(MINIMAX_H3_VIDEO_PAD))

    var ids = List[Int]()
    var tags = List[Int]()
    var image_count = 0
    var video_count = 0
    var audio_count = 0

    for i in range(len(references)):
        ref block = references[i]
        if block.has_audio:
            audio_count += 1
            _emit_text(tokenizer, ids, tags, "<Audio " + String(audio_count) + ">: ")
        if block.kind == MINIMAX_H3_REF_IMAGE:
            image_count += 1
            _emit_text(tokenizer, ids, tags, "<Picture " + String(image_count) + ">: ")
            _emit_vision(ids, tags, start_id, image_pad, end_id, block.vision_token_count)
        elif block.kind == MINIMAX_H3_REF_VIDEO:
            video_count += 1
            _emit_text(tokenizer, ids, tags, "<Video " + String(video_count) + ">: ")
            for b in range(len(block.block_timestamps)):
                var stamp = minimax_h3_format_seconds(block.block_timestamps[b])
                _emit_text(tokenizer, ids, tags, "<" + stamp + " seconds>")
                _emit_vision(ids, tags, start_id, video_pad, end_id, block.vision_token_count)
        elif block.kind != MINIMAX_H3_REF_AUDIO:
            raise Error("MiniMax-H3: a reference must be an image, a video or an audio")

    _emit_text(tokenizer, ids, tags, prompt)
    return MiniMaxH3Presentation(ids^, tags^)
