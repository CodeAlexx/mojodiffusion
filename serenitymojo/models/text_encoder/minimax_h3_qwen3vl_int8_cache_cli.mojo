# One-time/resumable GPU cache builder for the MiniMax-H3 INT8 conditioner.
#
# Build:
#   pixi run mojo build -I . -O2 -Xlinker -lm \
#     -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/text_encoder/minimax_h3_qwen3vl_int8_cache_cli.mojo \
#     -o /tmp/minimax_h3_qwen_int8_cache
# Run:
#   /tmp/minimax_h3_qwen_int8_cache <text_encoder_dir> [layers=50]

from std.gpu.host import DeviceContext
from std.sys import argv

from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_int8 import (
    minimax_h3_build_int8_encoder_cache,
)


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: minimax_h3_qwen_int8_cache <text_encoder_dir> [layers=50]")
        return
    var layers = 50
    if len(args) >= 3:
        layers = atol(String(args[2]))
    var ctx = DeviceContext()
    var built = minimax_h3_build_int8_encoder_cache(
        String(args[1]), layers, ctx
    )
    print("MiniMax-H3 encoder INT8 cache complete; new matrices:", built)
