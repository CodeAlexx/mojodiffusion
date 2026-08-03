# serenitymojo/models/text_encoder/minimax_h3_qwen3vl_streamed_probe.mojo
#
# Probe for minimax_h3_qwen3vl_streamed.mojo. Two modes, chosen at RUNTIME by
# whether the real checkpoint has landed:
#
#   * Weights present (model.safetensors.index.json exists under
#     .../MiniMax-H3/FL2VA/text_encoder/): runs
#     minimax_h3_encode_conditioning_streamed for real on a tiny fixed
#     prompt, prints the output shape/dtype and a sample value. Still NOT a
#     parity check (no reference `hidden_states[50]` to compare against
#     exists yet) — a real execution, not a numeric gate.
#   * Weights absent (current state as of this probe's writing — only
#     config.json has landed): the function is still referenced and called
#     in this file's source, so its body is fully compiled/typechecked either
#     way; execution is skipped and said so PLAINLY, exit 0. Do not read a
#     clean build here as a pass on the actual encoder.
#
# LINKER: once the weights-present branch actually reaches `_layer`'s sdpa
# call, expect plain `mojo run -I .` to fail with
# "Symbols not found: flame_cudnn_sdpa_bf16" (two other agents lost time to
# this same night). Build/run with:
#   pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/text_encoder/minimax_h3_qwen3vl_streamed_probe.mojo
#
# Run (SKIPPED branch only needs the plain form):
#   pixi run mojo run -I . serenitymojo/models/text_encoder/minimax_h3_qwen3vl_streamed_probe.mojo

from std.gpu.host import DeviceContext
from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_streamed import (
    minimax_h3_encode_conditioning_streamed,
    H3_HIDDEN,
    H3_EXTRACT_LAYER,
)

comptime _TEXT_ENCODER_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder"


def _index_present() -> Bool:
    var path = String(_TEXT_ENCODER_DIR) + "/model.safetensors.index.json"
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def main() raises:
    var ctx = DeviceContext()

    if not _index_present():
        print(
            "SKIPPED: MiniMax-H3 text_encoder/model.safetensors.index.json"
            " not on disk yet (only config.json has landed as of this run)."
        )
        print(
            "This file's functions are still referenced above, so"
            " minimax_h3_qwen3vl_streamed.mojo's body compiled/typechecked"
            " cleanly — that is NOT a numeric gate. Re-run once the 62.13"
            " GiB text_encoder/ download completes."
        )
        return

    # Tiny fixed prompt (token ids are placeholders, not from H3's real
    # tokenizer/chat template — that step is explicitly not wired yet).
    var ids = List[Int]()
    ids.append(9906)
    ids.append(1917)
    ids.append(0)

    var hidden = minimax_h3_encode_conditioning_streamed(_TEXT_ENCODER_DIR, ids, ctx)
    var sh = hidden.shape()
    if len(sh) != 3 or sh[0] != 1 or sh[1] != len(ids) or sh[2] != H3_HIDDEN:
        raise Error("minimax_h3_qwen3vl_streamed_probe: unexpected output shape")

    var host = hidden.to_host(ctx)
    print(
        "minimax_h3_encode_conditioning_streamed: hidden_states[", H3_EXTRACT_LAYER, "]",
        " shape=", sh[0], "x", sh[1], "x", sh[2],
        " sample=", host[0],
    )
    print("RAN FOR REAL (not a parity check — no reference output exists yet)")
