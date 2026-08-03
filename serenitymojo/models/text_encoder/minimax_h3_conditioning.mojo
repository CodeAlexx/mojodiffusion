# serenitymojo/models/text_encoder/minimax_h3_conditioning.mojo
#
# MiniMax-H3 end-to-end TEXT conditioning: prompt string -> token ids ->
# streamed Qwen3-VL-32B encode -> hidden_states[50], ready for
# `minimax_h3_condition_embed` (models/dit/minimax_h3_frontend.mojo).
#
# NO CHAT TEMPLATE — verified against the actual reference source, not
# assumed: /home/alex/minimax_h3_ref/diffusers-src/src/diffusers/
# modular_pipelines/minimax_h3/encoders.py, class MiniMaxH3TextEncoderStep.
# Its own docstring: "Encodes MiniMax-H3's presentation of a `t2va` / `fl2va`
# request: the prompt verbatim ... with no chat template and no special
# tokens." `encode_prompt` calls
# `components.tokenizer(prompt, add_special_tokens=False)["input_ids"]`
# directly for the text — no `apply_chat_template` anywhere in the file.
# H3's `processor/chat_template.json` IS present on disk (it ships with the
# Qwen3-VL-32B-Instruct base processor bundle, confirmed real content, not a
# stub) but is genuinely UNUSED by H3's own conditioning code. Do not wire
# it in.
#
# TOKENIZATION mirrors the reference exactly for t2va (no keyframes/images):
#   token_ids  = tokenizer(prompt, add_special_tokens=False)   — no BOS/EOS
#   token_tags = [MINIMAX_H3_TEXT_TAG] * len(token_ids)        — every row TEXT
# `Qwen3Tokenizer.encode()` (serenitymojo/tokenizer/tokenizer.mojo:1100)
# never inserts BOS/EOS itself, matching `add_special_tokens=False` with no
# extra work needed on this side. fl2va's `"<Picture i>: "` + vision-block
# presentation (encoders.py:154-165, image conditioning) is NOT built here —
# t2va has no conditioning image, out of this port's scope (see the H3
# conditioner scoping report).
#
# SPECIAL TOKENS: H3 ships seven tokens (<d> </d> <|cutoff|> <|lyrics_start|>
# <|lyrics_end|> <|caption_start|> <|caption_end|>) that exist ONLY in
# `tokenizer_config.json`'s `additional_special_tokens` array (confirmed on
# the real, landed processor/tokenizer_config.json), not in tokenizer.json's
# vocab. `Qwen3Tokenizer.merge_additional_special_tokens`
# (tokenizer.mojo:631) is the ALREADY-GATED fix for this (4 checks, exact
# ids 151669..151675 vs transformers, models/minimax_h3/parity/
# minimax_h3_tokenizer_parity.mojo) — called below, not re-derived.
#
# EXTRACTION: hidden_states[50], pre-final-norm, via
# minimax_h3_qwen3vl_streamed.minimax_h3_encode_conditioning_streamed —
# matches the reference's `outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]`
# (encoders.py:198, MINIMAX_H3_TEXT_ENCODER_LAYER = 50 in packing.py).
#
# TEXT-ONLY MRoPE, now confirmed by the reference mechanism (not just
# precedent): encoders.py builds `mm_token_type_ids` from
# `processor.create_mm_token_type_ids([token_ids])` — 0 for text, 1 image, 2
# video — and Qwen3-VL derives its 3-axis rotary positions from those ids
# ("lays its 3D rotary positions out per modality run"). A t2va prompt is
# ALL text, so mm_token_type_ids is all zeros and every axis position is
# identical for every token — MRoPE collapses to plain 1-D rope by
# construction here, not merely by analogy to the two other precedents
# (ideogram_qwen3vl_streamed.mojo, qwen25vl_encoder.mojo) already cited.
# Still unverified numerically (see below).
#
# STILL UNGATED as of this writing: H3's text_encoder/ shards + index.json
# have not landed (only config.json/chat_template.json/merges.txt are on
# disk), so the streamed-encode half of this file has never run against
# real weights — see minimax_h3_qwen3vl_streamed.mojo's header for that
# status. The TOKENIZER half HAS run for real: H3's processor/ directory
# (tokenizer.json + tokenizer_config.json) is complete on disk today, and
# minimax_h3_conditioning_probe.mojo exercises it for real, printing the
# actual token ids and the merge's actual added-token count. A clean build
# of this whole file is NOT a numeric gate on the encoder half until the
# text_encoder weights land.
#
# LINKER: once the streamed-encode half actually runs (weights present),
# expect plain `mojo run -I .` to fail the moment `_layer`'s sdpa call is
# reached with "Symbols not found: flame_cudnn_sdpa_bf16". Add:
#   -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.minimax_h3.packing import MINIMAX_H3_TEXT_TAG
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_streamed import (
    minimax_h3_encode_conditioning_streamed,
)


struct MiniMaxH3ConditioningOutput(Movable):
    """`embeds` [1, seq, 5120] pre-final-norm, and the per-row modality tag
    `token_tags` — matches encoders.py's `_conditioner_outputs()` contract
    (`prompt_embeds`, `text_token_tags`)."""

    var embeds: Tensor
    var token_tags: List[Int]

    def __init__(out self, var embeds: Tensor, var token_tags: List[Int]):
        self.embeds = embeds^
        self.token_tags = token_tags^


def minimax_h3_tokenize_prompt(
    processor_dir: String, prompt: String
) raises -> List[Int]:
    """Prompt -> token ids, t2va (no keyframes). No chat template, no
    BOS/EOS — see file header. `processor_dir` is H3's `processor/`
    (tokenizer.json + tokenizer_config.json)."""
    var tok = Qwen3Tokenizer(processor_dir + "/tokenizer.json")
    var _added = tok.merge_additional_special_tokens(
        processor_dir + "/tokenizer_config.json"
    )
    return tok.encode(prompt)


def minimax_h3_encode_conditioning(
    processor_dir: String,
    text_encoder_dir: String,
    prompt: String,
    ctx: DeviceContext,
) raises -> MiniMaxH3ConditioningOutput:
    """End-to-end: prompt string -> ids -> streamed encode -> hidden_states[50].
    t2va only (no keyframes/images — see file header)."""
    var ids = minimax_h3_tokenize_prompt(processor_dir, prompt)
    if len(ids) == 0:
        raise Error("minimax_h3_encode_conditioning: prompt tokenized to zero ids")
    var token_tags = List[Int]()
    for _ in range(len(ids)):
        token_tags.append(MINIMAX_H3_TEXT_TAG)

    # PAD-TO-DISPATCH-CASE. ops/attention's sdpa_dispatch enumerates comptime
    # sequence lengths (8..2048 powers of two for h=64/dh=128); an arbitrary
    # prompt length (e.g. 245) has no case and raises. The encoder is CAUSAL,
    # so trailing pad tokens cannot alter hidden states at positions before
    # them — pad up to the next enumerated size, run, slice the rows back.
    # Pad id 151643 (<|endoftext|>) — any id works; its rows are discarded.
    var real_len = len(ids)
    var padded_len = 8
    while padded_len < real_len:
        padded_len *= 2
    if padded_len > 2048:
        raise Error(
            String("minimax_h3_encode_conditioning: prompt is ")
            + String(real_len)
            + " tokens; the sdpa dispatch table tops out at 2048"
        )
    var padded_ids = ids.copy()
    for _ in range(padded_len - real_len):
        padded_ids.append(151643)

    var embeds_padded = minimax_h3_encode_conditioning_streamed(
        text_encoder_dir, padded_ids, ctx
    )
    var embeds = slice(embeds_padded, 1, 0, real_len, ctx)
    return MiniMaxH3ConditioningOutput(embeds^, token_tags^)
