# ltx2_vae_decode_hq121_smoke.mojo — target-shape LTX2 video VAE decode gate.
#
# Exercises the production HQ staged decode shape only:
#   latent  [1,128,16,16,24]
#   output  [1,3,121,512,768]
#
# This is a runtime/performance gate for the decoder path, not a generation
# artifact. It uses a deterministic synthetic BF16 latent.
#
# Build:
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/pipeline/ltx2_vae_decode_hq121_smoke.mojo \
#     -o /tmp/ltx2_vae_decode_hq121_smoke

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.ltx2_vae_decoder import (
    LTX2VaeDecoderWeights,
    decode,
)


comptime CKPT = (
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
)

comptime B = 1
comptime C = 128
comptime F = 16
comptime H = 16
comptime W = 24


def main() raises:
    var ctx = DeviceContext()

    print("=== LTX2 VAE HQ121 decode smoke ===")
    print("  checkpoint:", CKPT)
    print("  latent target: [", B, ",", C, ",", F, ",", H, ",", W, "]")

    var weights = LTX2VaeDecoderWeights.load(CKPT, ctx)
    print("  weights loaded")

    var numel = B * C * F * H * W
    var host = List[Float32]()
    host.resize(numel, Float32(0.0))
    for i in range(numel):
        host[i] = Float32((i % 17) - 8) * Float32(0.03)

    var sh = List[Int]()
    sh.append(B); sh.append(C); sh.append(F); sh.append(H); sh.append(W)
    var latent = Tensor.from_host(host, sh^, STDtype.BF16, ctx)

    var out = decode[B, C, F, H, W](weights, latent, ctx)
    var os = out.shape()
    print(
        "  decoded shape: [", os[0], ",", os[1], ",", os[2], ",", os[3],
        ",", os[4], "]",
    )

    var f_out = 1 + (F - 1) * 8
    if (
        os[0] != B or os[1] != 3 or os[2] != f_out
        or os[3] != H * 32 or os[4] != W * 32
    ):
        raise Error("hq121 VAE decode output shape mismatch")

    print("  PASS: target-shape decode completed")
