# Microbench: shared SDPA backward at the active Klein training dimensions.
#
# The current 512px Klein LoRA run uses F32 tensors with:
#   B=1, S=1536, H=24, Dh=128  (D=3072, 4B checkpoint path)
# Every single and double block calls the shared `sdpa_backward`, so this is an
# all-model training-speed lever, not a Klein-specific LoRA helper.

from std.math import sqrt
from std.gpu.host import DeviceContext
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.attention_backward import sdpa_backward


def zeros_bshd(
    b: Int, s: Int, h: Int, dh: Int, dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    var n = b * s * h * dh
    var buf = ctx.enqueue_create_buffer[DType.uint8](n * dtype.byte_size())
    buf.enqueue_fill(UInt8(0))
    var shp = List[Int]()
    shp.append(b)
    shp.append(s)
    shp.append(h)
    shp.append(dh)
    return Tensor(buf^, shp^, dtype)


def bench_sdpa_bwd[
    B: Int, S: Int, H: Int, Dh: Int
](
    name: String, dtype: STDtype, warmup: Int, iters: Int, ctx: DeviceContext
) raises:
    var q = zeros_bshd(B, S, H, Dh, dtype, ctx)
    var k = zeros_bshd(B, S, H, Dh, dtype, ctx)
    var v = zeros_bshd(B, S, H, Dh, dtype, ctx)
    var d_out = zeros_bshd(B, S, H, Dh, dtype, ctx)
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    for _ in range(warmup):
        var _g = sdpa_backward[B, S, H, Dh](q, k, v, d_out, scale, ctx)
        ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(iters):
        var _g = sdpa_backward[B, S, H, Dh](q, k, v, d_out, scale, ctx)
        ctx.synchronize()
    var t1 = perf_counter_ns()

    var ms = Float64(t1 - t0) / 1.0e6 / Float64(iters)
    var score_gib = (
        Float64(B) * Float64(H) * Float64(S) * Float64(S) * 4.0
        / 1073741824.0
    )
    print(
        name, " B=", B, " S=", S, " H=", H, " Dh=", Dh,
        " dtype=", dtype.name(), " score_buf=", score_gib,
        "GiB -> ", ms, " ms/iter",
    )


def main() raises:
    var ctx = DeviceContext()
    print("=== shared sdpa_backward microbench (Klein real dims) ===")
    bench_sdpa_bwd[1, 1536, 24, 128](
        String("Klein 4B 512px active"), STDtype.F32, 1, 3, ctx
    )
