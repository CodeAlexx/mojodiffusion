# lance_wan22_vae_video_smoke.mojo - Wan2.2 VAE temporal decode smoke.
#
# This is a tiny Lance VAE wiring gate for the cached video decode path. It
# uses random latent tokens with T_lat=3 and writes the first decoded frame as a
# 16x16 RGB PNG. It is not a quality sample.

from std.gpu.host import DeviceContext

from serenitymojo.components.artifacts import save_video_frame_png
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.wan22_decoder import Wan22VaeImageDecoder
from serenitymojo.ops.random import randn


comptime VAE_PATH = "/home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors"
comptime LH = 1
comptime LW = 1
comptime LT = 3
comptime OUT = "/home/alex/mojodiffusion/output/lance_wan22_vae_video_t3_frame0_16.png"


def main() raises:
    var ctx = DeviceContext()
    var shape = List[Int]()
    shape.append(LT * LH * LW)
    shape.append(48)
    var z = randn(shape^, UInt64(20260528), STDtype.F32, ctx)
    print("[wan22-vae-video] loading decoder")
    var vae = Wan22VaeImageDecoder[LH, LW].load(String(VAE_PATH), ctx)
    print("[wan22-vae-video] decoding T_lat=3 tiny latent")
    var video = vae.decode_video_tokens(z, LT, ctx)
    var vs = video.shape()
    print("[wan22-vae-video] video shape:", vs[0], vs[1], vs[2], vs[3], vs[4])
    save_video_frame_png(video, 0, String(OUT), LH, LW, ctx)
    print("[wan22-vae-video] saved ->", OUT)
