# norm_modulate_smoke.mojo — gate the fused norm_modulate AGAINST
# modulate(layer_norm(x, ones, zeros, eps), scale, shift). (MJ-1005)
#
# Run: cd /home/alex/md-perf && pixi run mojo run -I serenitymojo \
#      serenitymojo/ops/parity/norm_modulate_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.norm import layer_norm, norm_modulate
from serenitymojo.ops.elementwise import modulate


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _const(n: Int, v: Float32) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(v)
    return out^


def main() raises:
    var ctx = DeviceContext()
    var rows = 4096
    var d = 1024
    var eps = Float32(1e-6)
    print("=== norm_modulate parity vs modulate(layer_norm(x,1,0)) (rows=", rows, " d=", d, ") ===")

    var x = Tensor.from_host(_fill(rows * d, 1, 4.0), [rows, d], STDtype.F32, ctx)
    var scale = Tensor.from_host(_fill(d, 2, 1.0), [d], STDtype.F32, ctx)
    var shift = Tensor.from_host(_fill(d, 3, 1.0), [d], STDtype.F32, ctx)
    var ones = Tensor.from_host(_const(d, 1.0), [d], STDtype.F32, ctx)
    var zeros = Tensor.from_host(_const(d, 0.0), [d], STDtype.F32, ctx)

    var refh = modulate(layer_norm(x, ones, zeros, eps, ctx), scale, shift, ctx).to_host(ctx)
    var got = norm_modulate(x, scale, shift, eps, ctx)

    var h = ParityHarness(0.999)
    var r = h.compare(got, refh, ctx)
    print("    norm_modulate:", r)

    if r.passed:
        print("PASS: norm_modulate matches modulate(layer_norm) cos>=0.999")
    else:
        raise Error("norm_modulate_smoke gate FAILED")
