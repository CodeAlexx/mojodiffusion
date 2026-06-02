# ldm_decoder_probe.mojo — compile/typecheck driver for the SDXL LDM decoder.
# Instantiates SDXLLdmDecoder at a small comptime latent (8x8) so the kit's
# comptime conv shapes are tiny; does NOT load weights or run the GPU forward
# (GPU wedged; compile-only verification).

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.ldm_decoder import SDXLLdmDecoder


# Force monomorphization of load()+decode() at a tiny comptime latent (8x8 ->
# image 64x64) without executing GPU work. The `if False:` guard keeps the
# device calls in the compilation unit (so the kit type-checks end-to-end) yet
# never runs them.
def _instantiate(ctx: DeviceContext) raises:
    if False:
        var dec = SDXLLdmDecoder[8, 8].load(String("/nonexistent"), ctx)
        var dummy = Tensor.from_host(
            _zeros(1 * 4 * 8 * 8), _shape4(1, 4, 8, 8), STDtype.BF16, ctx
        )
        var img = dec.decode(dummy, ctx)
        print(img.shape()[0])


def _zeros(n: Int) -> List[Float32]:
    var v = List[Float32]()
    for _ in range(n):
        v.append(0.0)
    return v^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    return s^


def main() raises:
    var ctx = DeviceContext()
    print("ctx ok", Int(ctx.id()))
    _instantiate(ctx)
    print("ldm_decoder compile OK")
