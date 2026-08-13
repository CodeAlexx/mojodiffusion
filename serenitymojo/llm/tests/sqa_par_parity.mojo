# sqa_par_parity.mojo — gate the PARALLEL single-query attention (sqa_device_par)
# against the serial sqa_device (already validated end-to-end by decoder_cache_test)
# on random inputs across L. Same F32-accumulate math → expect cos ~1.0.
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.llm.sqa import sqa_device, sqa_device_par
from std.math import sqrt


def _fill(n: Int, seed: Int) -> List[Float32]:
    # deterministic LCG in [-1, 1]
    var out = List[Float32]()
    var s = UInt64(seed * 2654435761 + 12345)
    for _ in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float32((s >> 33) & UInt64(0x7FFFFF)) / Float32(0x7FFFFF)
        out.append(u * 2.0 - 1.0)
    return out^


def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var dot = Float64(0.0); var na = Float64(0.0); var nb = Float64(0.0)
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


def _to_f32(t: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    var h = t.to_host(ctx)
    var out = List[Float32]()
    for i in range(len(h)):
        out.append(Float32(h[i]))
    return out^


def main() raises:
    var ctx = DeviceContext()
    var H = 32; var H_kv = 8; var dh = 128    # Qwen3-8B attention shape
    var Ls = List[Int]()
    for L in [1, 7, 64, 257, 512, 2000]:
        Ls.append(L)

    var worst = Float64(2.0)
    for li in range(len(Ls)):
        var L = Ls[li]
        var q = Tensor.from_host(_fill(H * dh, 1), [1, 1, H, dh], STDtype.BF16, ctx)
        var kc = Tensor.from_host(_fill(L * H_kv * dh, 2), [1, L, H_kv, dh], STDtype.BF16, ctx)
        var vc = Tensor.from_host(_fill(L * H_kv * dh, 3), [1, L, H_kv, dh], STDtype.BF16, ctx)
        var a = _to_f32(sqa_device(q, kc, vc, H, H_kv, L, dh, ctx), ctx)
        var b = _to_f32(sqa_device_par(q, kc, vc, H, H_kv, L, dh, ctx), ctx)
        var c = _cos(a, b)
        if c < worst: worst = c
        print("L=", L, " cos(serial, parallel)=", c)
    print("worst cos:", worst)
    if worst >= 0.9999:
        print("SQA PARALLEL PARITY PASS")
    else:
        print("SQA PARALLEL PARITY FAIL")
