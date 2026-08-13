# Bounded real-weight execution probe for the direct-W8A8 H3 conditioner.
# It never invokes the BF16 streamed encoder. Cache creation, if needed, uses
# the fixed-slab GPU converter from minimax_h3_qwen3vl_int8.mojo.

from max.gpu.host import DeviceContext
from std.sys import argv

from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_int8 import (
    minimax_h3_encode_conditioning_int8_streamed_depth,
)


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: minimax_h3_qwen_int8_depth_probe <text_encoder_dir> [layers=1]")
        return
    var layers = 1
    if len(args) >= 3:
        layers = atol(String(args[2]))
    # Eight real vocabulary rows; sequence 8 is an existing compiled SDPA case.
    var ids: List[Int] = [9707, 314, 264, 1741, 4762, 323, 3290, 13]
    var ctx = DeviceContext()
    var hidden = minimax_h3_encode_conditioning_int8_streamed_depth(
        String(args[1]), ids, layers, ctx
    )
    ctx.synchronize()
    print("PASS: H3 direct INT8 conditioner depth", layers)
    print("  output shape:", hidden.shape(), "dtype:", hidden.dtype().name())
