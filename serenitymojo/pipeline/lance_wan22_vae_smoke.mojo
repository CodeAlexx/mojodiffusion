# lance_wan22_vae_smoke.mojo — Wan2.2 VAE first-frame decode smoke.
#
# This uses a tiny random Lance latent grid (`T_lat=1, H=W=1`) to exercise the
# full Wan2.2 decoder body and write a 16x16 RGB PNG. It is a VAE wiring/load
# gate, not a quality sample.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.vae.wan22_decoder import Wan22VaeImageDecoder
from serenitymojo.ops.random import randn


comptime VAE_PATH = "/home/alex/.serenity/models/vaes/wan2.2_vae.safetensors"
comptime LH = 1
comptime LW = 1
comptime OUT = "/home/alex/mojodiffusion/output/lance_wan22_vae_smoke_16.png"


def main() raises:
    var ctx = DeviceContext()
    var shape = List[Int]()
    shape.append(LH * LW)
    shape.append(48)
    var z = randn(shape^, UInt64(20260527), STDtype.F32, ctx)
    print("[wan22-vae] loading first-frame decoder")
    var vae = Wan22VaeImageDecoder[LH, LW].load(String(VAE_PATH), ctx)
    print("[wan22-vae] decoding tiny latent")
    var img = vae.decode_tokens(z, ctx)
    print("[wan22-vae] image shape:", img.shape()[0], img.shape()[1], img.shape()[2], img.shape()[3])
    save_png(img, String(OUT), ctx, ValueRange.SIGNED)
    print("[wan22-vae] saved ->", OUT)
