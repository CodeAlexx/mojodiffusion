# qwenimage_rect_rope_parity.mojo — product landscape RoPE vs local diffusers.

from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor
from serenitymojo.models.dit.qwenimage_dit import (
    QwenImageConfig, build_qwenimage_rope_tables,
)

comptime FIXTURE = "/tmp/qwenimage_rect_rope.safetensors"
comptime FRAME = 1
comptime HEIGHT = 56
comptime WIDTH = 72
comptime TEXT = 512
comptime HEADS = 2
comptime HALF = 64
comptime ROWS = (TEXT + FRAME * HEIGHT * WIDTH) * HEADS


def main() raises:
    var ctx = DeviceContext()
    var cfg = QwenImageConfig.qwen_image()
    var rope = build_qwenimage_rope_tables(
        FRAME, HEIGHT, WIDTH, TEXT, HEADS, cfg, STDtype.F32, ctx
    )
    var sh = rope[0].shape()
    if len(sh) != 2 or sh[0] != ROWS or sh[1] != HALF:
        raise Error("Qwen-Image rectangular RoPE shape mismatch")

    var fx = ShardedSafeTensors.open(FIXTURE)
    var cos_ref = Tensor.from_view(fx.tensor_view("cos"), ctx).to_host(ctx)
    var sin_ref = Tensor.from_view(fx.tensor_view("sin"), ctx).to_host(ctx)
    var h = ParityHarness(0.99999)
    var cos_result = h.compare(rope[0], cos_ref, ctx)
    var sin_result = h.compare(rope[1], sin_ref, ctx)
    print("qwenimage landscape rope cos:", cos_result)
    print("qwenimage landscape rope sin:", sin_result)
    if not cos_result.passed or not sin_result.passed:
        raise Error("Qwen-Image rectangular RoPE parity failed")
    print("PASS: Qwen-Image 56x72 patch-grid RoPE matches local diffusers")
