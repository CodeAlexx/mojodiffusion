# klein_vae_1024_smoke.mojo - FLUX.2/Klein VAE native-size decode smoke.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.klein_decoder import KleinVaeDecoder
from serenitymojo.image.png import save_png, ValueRange


comptime VAE_PATH = "/home/alex/.serenity/models/vaes/flux2-vae.safetensors"
comptime LH = 64
comptime LW = 64
comptime OUT = "/home/alex/mojodiffusion/output/klein_vae_smoke_1024.png"


def _latent(ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32]()
    var n = 1 * 128 * LH * LW
    for i in range(n):
        vals.append(Float32((i % 23) - 11) * 0.003)
    var sh = List[Int]()
    sh.append(1)
    sh.append(128)
    sh.append(LH)
    sh.append(LW)
    return Tensor.from_host(vals, sh^, STDtype.F32, ctx)


def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var n = len(h)
    var s = 0.0
    var s2 = 0.0
    var amax = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v
        s2 += v * v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    var mean = s / Float64(n)
    var var_ = s2 / Float64(n) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print(
        "  [stat]", name, "mean=", Float32(mean), "std=", Float32(sqrt(var_)),
        "absmax=", Float32(amax), "n=", n,
    )


def main() raises:
    var ctx = DeviceContext()
    print("=== Klein VAE smoke - packed [1,128,64,64] -> 1024x1024 ===")
    print("[load]", VAE_PATH)
    var vae = KleinVaeDecoder[LH, LW].load(VAE_PATH, ctx)
    var z = _latent(ctx)
    print("[decode]")
    var img = vae.decode(z, ctx)
    var sh = img.shape()
    print("  image shape:", sh[0], sh[1], sh[2], sh[3])
    _stats("image", img, ctx)
    save_png(img, OUT, ctx, ValueRange.SIGNED)
    print("saved", OUT)
