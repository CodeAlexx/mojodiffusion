# serenitymojo/models/text_encoder/minimax_h3_conditioning_probe.mojo
#
# Probe for minimax_h3_conditioning.mojo. Two independent halves, run
# separately because only one is currently possible:
#
#   * TOKENIZER half: H3's real, landed processor/ (tokenizer.json +
#     tokenizer_config.json) is complete on disk today. RUNS FOR REAL: tokenizes
#     a prompt containing one of the seven H3-only special tokens (`<d>...</d>`)
#     and checks it lands as ONE id each (not the three pieces `<`,`d`,`>` a
#     tokenizer.json-only load would produce) — the concrete, observable effect
#     of `merge_additional_special_tokens`, not just an added-count check
#     (that exact-id gate already exists and is not re-derived here:
#     models/minimax_h3/parity/minimax_h3_tokenizer_parity.mojo).
#   * ENCODER half: H3's text_encoder/ shards + model.safetensors.index.json
#     have NOT landed (checked at runtime, same pattern as
#     minimax_h3_qwen3vl_streamed_probe.mojo). Runs for real if present,
#     otherwise says so plainly and skips — NOT a numeric gate either way
#     until then.
#
# LINKER: once the encoder half actually runs, expect plain `mojo run -I .`
# to fail with "Symbols not found: flame_cudnn_sdpa_bf16" the moment
# `_layer`'s sdpa call is reached (two other agents lost time to this same
# night). Build/run with:
#   pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/text_encoder/minimax_h3_conditioning_probe.mojo
#
# Run (tokenizer half only needs the plain form):
#   pixi run mojo run -I . serenitymojo/models/text_encoder/minimax_h3_conditioning_probe.mojo

from std.gpu.host import DeviceContext
from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY
from serenitymojo.models.text_encoder.minimax_h3_conditioning import (
    minimax_h3_tokenize_prompt,
    minimax_h3_encode_conditioning,
    MiniMaxH3ConditioningOutput,
)
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_streamed import H3_HIDDEN

comptime _PROCESSOR_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/processor"
comptime _TEXT_ENCODER_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder"


def _path_readable(path: String) -> Bool:
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def main() raises:
    var ctx = DeviceContext()

    # ── TOKENIZER half: real run against H3's real, landed processor/ ──────
    print("=== tokenizer half (real H3 processor/, RUNS FOR REAL) ===")
    var plain = minimax_h3_tokenize_prompt(
        String(_PROCESSOR_DIR), "a video of a cat playing piano"
    )
    if len(plain) == 0:
        raise Error("minimax_h3_conditioning_probe: plain prompt tokenized to zero ids")
    print("  plain prompt ->", len(plain), "ids, first =", plain[0])

    # The concrete effect of merge_additional_special_tokens: <d>...</d>
    # embedded in prompt text (adjacent, no whitespace, exactly how H3's
    # README uses it) must land as ONE id each, in [151669, 151675].
    var special = minimax_h3_tokenize_prompt(
        String(_PROCESSOR_DIR), "<d>a cutoff marker</d> then continue"
    )
    var d_open = special[0]
    if d_open < 151669 or d_open > 151675:
        raise Error(
            "minimax_h3_conditioning_probe: <d> did not tokenize to a single"
            " H3 special-token id — merge_additional_special_tokens regressed"
            " or was not applied"
        )
    print(
        "  '<d>...</d>' prompt -> first id =", d_open,
        "(single H3 special token, not the 3-piece '<','d','>' split)",
    )

    # ── ENCODER half: only if the real weights have landed ─────────────────
    print("")
    print("=== encoder half (H3 text_encoder/ weights) ===")
    if not _path_readable(String(_TEXT_ENCODER_DIR) + "/model.safetensors.index.json"):
        print(
            "SKIPPED: text_encoder/model.safetensors.index.json not on disk"
            " yet. Tokenizer half above is real; this half is NOT a numeric"
            " gate until the 62.13 GiB text_encoder/ download completes."
        )
        return

    var out = minimax_h3_encode_conditioning(
        String(_PROCESSOR_DIR), String(_TEXT_ENCODER_DIR),
        "a video of a cat playing piano", ctx,
    )
    var sh = out.embeds.shape()
    if len(sh) != 3 or sh[0] != 1 or sh[2] != H3_HIDDEN:
        raise Error("minimax_h3_conditioning_probe: unexpected embeds shape")
    if len(out.token_tags) != sh[1]:
        raise Error("minimax_h3_conditioning_probe: token_tags length != seq")
    print("  embeds shape=", sh[0], "x", sh[1], "x", sh[2], " token_tags all TEXT")
    print("RAN FOR REAL end-to-end (not a parity check — no reference output exists yet)")
