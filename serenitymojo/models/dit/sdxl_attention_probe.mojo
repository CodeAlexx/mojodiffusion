# sdxl_attention_probe.mojo — compile/typecheck driver for rectangular SDPA.
# Instantiates sdxl_sdpa at cross-attn (Sq=64, Skv=77) and self-attn (Sq=Skv=64)
# comptime shapes. Guarded so no GPU work runs (GPU wedged; compile-only).

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.sdxl_attention import sdxl_sdpa


def _z(n: Int) -> List[Float32]:
    var v = List[Float32]()
    for _ in range(n):
        v.append(0.0)
    return v^


def _sh(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    return s^


def _instantiate(ctx: DeviceContext) raises:
    if False:
        # cross-attn: q [1,64,5,64], kv [1,77,5,64]
        var q = Tensor.from_host(_z(1 * 64 * 5 * 64), _sh(1, 64, 5, 64), STDtype.BF16, ctx)
        var k = Tensor.from_host(_z(1 * 77 * 5 * 64), _sh(1, 77, 5, 64), STDtype.BF16, ctx)
        var v = Tensor.from_host(_z(1 * 77 * 5 * 64), _sh(1, 77, 5, 64), STDtype.BF16, ctx)
        var o = sdxl_sdpa[1, 64, 77, 5, 64](q, k, v, Float32(0.125), ctx)
        print(o.shape()[1])
        # self-attn: q=kv [1,64,5,64]
        var q2 = Tensor.from_host(_z(1 * 64 * 5 * 64), _sh(1, 64, 5, 64), STDtype.BF16, ctx)
        var o2 = sdxl_sdpa[1, 64, 64, 5, 64](q2, q2, q2, Float32(0.125), ctx)
        print(o2.shape()[1])


def main() raises:
    var ctx = DeviceContext()
    print("ctx ok", Int(ctx.id()))
    _instantiate(ctx)
    print("sdxl_attention compile OK")
