# Standalone Ideogram4 latent-to-PNG decoder.
#
#   ideogram4_latent_decode_cli <latent.safetensors> <out.png> [resolution]
#
# The inline trainer sampler writes `latent` as [1,128,GH,GW] F32. This CLI uses
# the existing Ideogram4 tiled VAE decoder, so 2048 output avoids the full-frame
# VAE activation peak.

from std.gpu.host import DeviceContext
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.tensor_algebra import add, mul, reshape, permute
from serenitymojo.image.png import save_png
from serenitymojo.models.vae.ideogram4_tiled_decode import (
    ideogram4_tiled_decode, ideogram4_tiled_decode_5x5_lowmem,
)


comptime I4_LATENTNORM = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors"
comptime I4_VAE = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors"


def decode_ideogram4_latent_to_png[GH: Int, GW: Int](
    latent_path: String,
    out_path: String,
    ctx: DeviceContext,
) raises:
    print("[Ideogram4-decode] latent ", latent_path)
    print("[Ideogram4-decode] out    ", out_path)

    var st = ShardedSafeTensors.open(latent_path)
    var z = Tensor.from_view(st.tensor_view(String("latent")), ctx)

    var ln = ShardedSafeTensors.open(I4_LATENTNORM)
    var scale = reshape(
        Tensor.from_view(ln.tensor_view("latent_scale"), ctx), [1, 1, 128], ctx
    )
    var shift = reshape(
        Tensor.from_view(ln.tensor_view("latent_shift"), ctx), [1, 1, 128], ctx
    )

    var z_hwc = permute(z, [0, 2, 3, 1], ctx)
    var z_tok = reshape(z_hwc, [1, GH * GW, 128], ctx)
    var zd = add(mul(z_tok, scale, ctx), shift, ctx)

    var z6 = reshape(zd, [1, GH, GW, 2, 2, 32], ctx)
    var zp = permute(z6, [0, 5, 1, 3, 2, 4], ctx)
    var latent = reshape(zp, [1, 32, 2 * GH, 2 * GW], ctx)

    comptime if GH >= 128 or GW >= 128:
        print("[Ideogram4-decode] tiled VAE decode 5x5 lowmem")
        var img = ideogram4_tiled_decode_5x5_lowmem[2 * GH, 2 * GW](
            latent, I4_VAE, ctx
        )
        print("[Ideogram4-decode] saving PNG")
        save_png(img, out_path, ctx)
    else:
        print("[Ideogram4-decode] tiled VAE decode 3x3")
        var img = ideogram4_tiled_decode[2 * GH, 2 * GW](latent, I4_VAE, ctx)
        print("[Ideogram4-decode] saving PNG")
        save_png(img, out_path, ctx)
    print("[Ideogram4-decode] done")


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: ideogram4_latent_decode_cli <latent.safetensors> <out.png> [resolution]")

    var latent_path = String(args[1])
    var out_path = String(args[2])
    var resolution = 2048
    if len(args) > 3:
        resolution = atol(String(args[3]))

    var ctx = DeviceContext()
    if resolution == 2048:
        decode_ideogram4_latent_to_png[128, 128](latent_path, out_path, ctx)
    elif resolution == 1024:
        decode_ideogram4_latent_to_png[64, 64](latent_path, out_path, ctx)
    elif resolution == 512:
        decode_ideogram4_latent_to_png[32, 32](latent_path, out_path, ctx)
    else:
        raise Error("resolution must be 512, 1024, or 2048")
