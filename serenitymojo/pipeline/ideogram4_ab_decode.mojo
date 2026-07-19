# pipeline/ideogram4_ab_decode.mojo — decode-only stage for the 1024 A/B (MJ-1054).
# Loads a final packed latent ("tensor" F32 [1,4096,128]) with NO DiTs resident,
# then decodes the SAME latent BOTH whole-image and 3x3-tiled. Split from the
# harness because dual-fp8-DiT + whole 1024 decode does not co-fit 24GB (the
# harness OOMs post-latent-save — the very pressure that motivated worker tiling).
#
# argv: 1 latent safetensors ("tensor" F32 [1,4096,128]) · 2 out prefix
from std.gpu.host import DeviceContext
from std.sys import argv
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import mul, add, reshape, permute
from serenitymojo.image.png import save_png
from serenitymojo.models.vae.ldm_decoder import load_ideogram4_vae_decoder
from serenitymojo.models.vae.ideogram4_tiled_decode import ideogram4_tiled_decode, ideogram4_tiled_decode_5x5_lowmem

comptime VAE = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors"
comptime LATENTNORM = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors"
comptime GH = 64
comptime GW = 64
comptime NIMG = GH * GW
comptime VAE_H = 2 * GH
comptime VAE_W = 2 * GW


def main() raises:
    var a = argv()
    if len(a) < 3:
        raise Error("usage: ideogram4_ab_decode <final_latent.st> <out_prefix>")
    var latent_path = String(a[1])
    var prefix = String(a[2])
    var ctx = DeviceContext()

    var nz = ShardedSafeTensors.open(latent_path)
    var z = cast_tensor(Tensor.from_view(nz.tensor_view("tensor"), ctx), STDtype.F32, ctx)
    z = reshape(z, [1, NIMG, 128], ctx)

    var ln = ShardedSafeTensors.open(LATENTNORM)
    var scale = reshape(Tensor.from_view(ln.tensor_view("latent_scale"), ctx), [1, 1, 128], ctx)
    var shift = reshape(Tensor.from_view(ln.tensor_view("latent_shift"), ctx), [1, 1, 128], ctx)
    var zd = add(mul(z, scale, ctx), shift, ctx)
    var z6 = reshape(zd, [1, GH, GW, 2, 2, 32], ctx)
    var zp = permute(z6, [0, 5, 1, 3, 2, 4], ctx)
    var latent = reshape(zp, [1, 32, VAE_H, VAE_W], ctx)
    var latent_bf = cast_tensor(latent, STDtype.BF16, ctx)

    var dec = load_ideogram4_vae_decoder[VAE_H, VAE_W](VAE, ctx)
    var img_whole = dec.decode(latent_bf, ctx)
    save_png(img_whole, prefix + "_whole.png", ctx)
    print("saved", prefix + "_whole.png")
    var img_tiled = ideogram4_tiled_decode[VAE_H, VAE_W](latent_bf, String(VAE), ctx)
    save_png(img_tiled, prefix + "_tiled.png", ctx)
    print("saved", prefix + "_tiled.png")
    var img_5x5 = ideogram4_tiled_decode_5x5_lowmem[VAE_H, VAE_W](latent_bf, String(VAE), ctx)
    save_png(img_5x5, prefix + "_tiled5x5.png", ctx)
    print("saved", prefix + "_tiled5x5.png")
