# sdxl_unet_probe.mojo — compile/typecheck driver for the SDXL UNet.
# Monomorphizes SDXLUNet[32,32] (image 256²; L0=32,L1=16,L2=8) load()+forward()
# behind an `if False:` guard so the entire UNet graph type-checks without any
# GPU execution (GPU wedged; compile-only verification).

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.sdxl_unet import SDXLUNet


def _z(n: Int) -> List[Float32]:
    var v = List[Float32]()
    for _ in range(n):
        v.append(0.0)
    return v^


def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    return s^


def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^


def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


# Comptime-parameterized instantiation helper. The `if False` keeps the device
# calls in the compilation unit (full type-check) without executing them.
def _instantiate[LH: Int, LW: Int](ctx: DeviceContext) raises:
    if False:
        var unet = SDXLUNet[LH, LW].load(String("/nonexistent"), ctx)
        var x = Tensor.from_host(_z(1 * 4 * LH * LW), _sh4(1, 4, LH, LW), STDtype.BF16, ctx)
        var ctxt = Tensor.from_host(_z(1 * 77 * 2048), _sh3(1, 77, 2048), STDtype.BF16, ctx)
        var y = Tensor.from_host(_z(1 * 2816), _sh2(1, 2816), STDtype.BF16, ctx)
        var eps = unet.forward(x, Float32(999.0), ctxt, y, ctx)
        print(eps.shape()[1])


def main() raises:
    var ctx = DeviceContext()
    print("ctx ok", Int(ctx.id()))
    _instantiate[32, 32](ctx)
    print("sdxl_unet compile OK")
