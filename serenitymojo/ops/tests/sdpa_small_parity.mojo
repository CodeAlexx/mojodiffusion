# ops/tests/sdpa_small_parity.mojo — fused small-S SDPA vs the sdpa_nomask
# math path, on the krea2 txtfusion layerwise shape [B=384, S=12, H=20, Dh=128].
# Gate: cos >= 0.9999 (same F32 QK/softmax; P@V here is F32-accum vs the loop
# path's storage-dtype GEMM — bf16-rounding-level differences only).
#
# Build:
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 -I . \
#     -I /home/alex/MOJO-libs -Xlinker -lm \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/ops/tests/sdpa_small_parity.mojo -o /tmp/sdpa_small_parity
# Mojo 1.0.0b1, NVIDIA GPU.

from max.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, pi
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.attention_small import sdpa_nomask_small


def _gaussian(n: Int, seed: Int, sd: Float32) -> List[Float32]:
    var out = List[Float32]()
    var st = UInt64(seed * 2654435761 + 12345)
    for _i in range(n):
        st = st * 6364136223846793005 + 1442695040888963407
        var u1 = (Float64(st >> 11) + 1.0) / Float64(1 << 53)
        st = st * 6364136223846793005 + 1442695040888963407
        var u2 = Float64(st >> 11) / Float64(1 << 53)
        var r = sqrt(-2.0 * flog(u1))
        out.append(Float32(r * fcos(2.0 * pi * u2)) * sd)
    return out^


def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var dot: Float64 = 0.0; var na: Float64 = 0.0; var nb: Float64 = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i]); na += Float64(a[i]) * Float64(a[i]); nb += Float64(b[i]) * Float64(b[i])
    return dot / (sqrt(na) * sqrt(nb) + 1e-30)


def main() raises:
    var ctx = DeviceContext()
    comptime B = 384
    comptime S = 12
    comptime H = 20
    comptime Dh = 128
    comptime n = B * S * H * Dh

    var qh = _gaussian(n, 3, 1.0)
    var kh = _gaussian(n, 9, 1.0)
    var vh = _gaussian(n, 5, 1.0)
    var q = Tensor.from_host(qh.copy(), [B, S, H, Dh], STDtype.BF16, ctx)
    var k = Tensor.from_host(kh.copy(), [B, S, H, Dh], STDtype.BF16, ctx)
    var v = Tensor.from_host(vh.copy(), [B, S, H, Dh], STDtype.BF16, ctx)

    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var base = sdpa_nomask[B, S, H, Dh](q, k, v, scale, ctx)
    var fused = sdpa_nomask_small[B, S, H, Dh](q, k, v, scale, ctx)

    var ref_h = base.to_host(ctx)
    var fused_h = fused.to_host(ctx)
    var c = _cos(ref_h, fused_h)
    var max_diff: Float32 = 0.0
    for i in range(len(ref_h)):
        var d = ref_h[i] - fused_h[i]
        if d < 0: d = -d
        if d > max_diff: max_diff = d
    print("sdpa_small vs sdpa_nomask: cos =", c, " max|diff| =", max_diff)
    if c < 0.9999:
        print("FAIL: cos", c); return
    print("PASS: fused small-S SDPA matches the math path (cos>=0.9999)")
