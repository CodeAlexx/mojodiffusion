# serenitymojo/ops/tests/evg_attention_rescue_parity.mojo
#
# Isolated all-rescue numerical gate for the Mojo-native EVG executor.  With
# every video-to-video tile routed through the BF16 rescue phase, EVG attends
# the complete document and must match dense cuDNN self-attention closely.

from max.gpu.host import DeviceContext
from std.math import isfinite, sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.attention_flash import sdpa_flash_infer_fwd_cross_dynamic
from serenitymojo.ops.evg_attention_int8 import (
    EVGH3RaggedLayout,
    evg_h3_attention_int8_fwd,
)
from serenitymojo.ops.random import randn
from serenitymojo.tensor import Tensor


comptime PREFIX = 5
comptime FRAMES = 2
comptime HEIGHT = 7
comptime WIDTH = 9
comptime HEADS = 2
comptime HEAD_DIM = 128


def _all_rescue_routes(heads: Int, blocks: Int, ctx: DeviceContext) raises -> Tensor:
    var elements = heads * blocks * blocks
    var host = ctx.enqueue_create_host_buffer[DType.uint8](elements)
    for i in range(elements):
        host.unsafe_ptr()[i] = UInt8(2)
    var device = ctx.enqueue_create_buffer[DType.uint8](elements)
    ctx.enqueue_copy(dst_buf=device, src_buf=host)
    return Tensor(device^, [heads, blocks, blocks], STDtype.U8)


def main() raises:
    var ctx = DeviceContext()
    var layout = EVGH3RaggedLayout(PREFIX, FRAMES, HEIGHT, WIDTH, ctx)
    var sequence = layout.sequence_tokens()
    var shape: List[Int] = [1, sequence, HEADS, HEAD_DIM]
    var q = randn(shape.copy(), 4101, STDtype.BF16, ctx)
    var k = randn(shape.copy(), 4102, STDtype.BF16, ctx)
    var v = randn(shape.copy(), 4103, STDtype.BF16, ctx)
    ctx.synchronize()
    print("stage=random-ready")
    var scale = Float32(1.0) / sqrt(Float32(HEAD_DIM))
    var routes = _all_rescue_routes(HEADS, layout.video_blocks, ctx)

    var reference = sdpa_flash_infer_fwd_cross_dynamic(q, k, v, scale, ctx)
    ctx.synchronize()
    print("stage=dense-ready")
    var got = evg_h3_attention_int8_fwd(q, k, v, scale, layout, routes, ctx)
    ctx.synchronize()
    print("stage=evg-ready")
    var rh = reference.to_host(ctx)
    var gh = got.to_host(ctx)
    var dot = Float64(0.0)
    var nr = Float64(0.0)
    var ng = Float64(0.0)
    var max_abs = Float32(0.0)
    var nonfinite = 0
    for i in range(len(rh)):
        if not isfinite(rh[i]) or not isfinite(gh[i]):
            nonfinite += 1
        var delta = gh[i] - rh[i]
        var absolute = delta if delta >= 0.0 else -delta
        if absolute > max_abs:
            max_abs = absolute
        dot += Float64(rh[i]) * Float64(gh[i])
        nr += Float64(rh[i]) * Float64(rh[i])
        ng += Float64(gh[i]) * Float64(gh[i])
    var cosine = dot / (sqrt(nr) * sqrt(ng) + 1.0e-30)
    print(
        "EVG all-rescue vs cuDNN: sequence=", sequence,
        " video_blocks=", layout.video_blocks,
        " cosine=", cosine,
        " max_abs=", max_abs,
        " nonfinite=", nonfinite,
    )
    if nonfinite != 0 or cosine < 0.999:
        raise Error("EVG BF16 rescue numerical gate failed")
    print("PASS: EVG BF16 rescue executor matches dense cuDNN")
