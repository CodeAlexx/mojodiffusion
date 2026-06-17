# serenitymojo/pipeline/anima_decode_cli.mojo
#
# STANDALONE Qwen-Image VAE decode CLI (process-SEPARATED from the sampler).
#
#   anima_decode_cli <latent.safetensors> <out.png>
#
# Reads a SCALED latent [1,16,128,128] BF16/F32 (key `latent`, the schema
# anima_sample_cli writes) and decodes it through the tiled Qwen/Wan image VAE
# to a 1024x1024 PNG ([-1,1] SIGNED range). The Mojo decoder internally applies
# z/inv_std + mean, matching OneTrainer AnimaModel.unscale_latents before
# diffusers VAE decode. Mirrors anima_vae_latent_smoke.mojo, but takes the latent
# path + out path as args. A FRESH DeviceContext with NO DiT resident — the
# 1024 3D-conv upsample decode is multi-GiB, so this uses 3x3 overlapping
# half-latent tiles and runs in its own process.
#
# Build:
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda -Xlinker -lpng16 \
#       serenitymojo/pipeline/anima_decode_cli.mojo -o /tmp/anima_decode_cli
#   /tmp/anima_decode_cli /tmp/x.latent.safetensors /tmp/x.png

from std.sys import argv
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape


# Qwen-Image VAE per-channel latent normalization (config.json latents_mean/std,
# z_dim=16). Standard sampler/trainer latents are already in OneTrainer SCALED
# space. Do not pre-unscale them here: wan21_image_tiled_decode does
# z/inv_std + mean internally. The helper below is retained only for explicit
# debugging of non-standard latents.
def _lat_mean() -> List[Float32]:
    var m = List[Float32]()
    var v = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
             0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
    for i in range(16): m.append(Float32(v[i]))
    return m^

def _lat_std() -> List[Float32]:
    var s = List[Float32]()
    var v = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
             3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916]
    for i in range(16): s.append(Float32(v[i]))
    return s^

def _mean_abs16(v: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        s += v[i] if v[i] >= 0.0 else -v[i]
    return s / Float32(len(v)) if len(v) > 0 else Float32(0.0)
from serenitymojo.models.dit.anima_contract import (
    ANIMA_LATENT_H, ANIMA_LATENT_W, ANIMA_LATENT_CHANNELS, ANIMA_VAE_PATH,
)
from serenitymojo.models.vae.qwenimage_tiled_decode import wan21_image_tiled_decode
from serenitymojo.image.png import save_png, ValueRange


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: anima_decode_cli <latent.safetensors> <out.png> [unscale]")
    var latent_path = String(args[1])
    var out_png = String(args[2])
    # OPTIONAL 3rd arg "unscale" is debug-only and normally wrong for Anima
    # trainer/sampler latents, because the tiled Wan21 decode already performs
    # the scaled->raw unnormalize internally.
    var do_unscale = (len(args) > 3 and String(args[3]) == String("unscale"))

    var ctx = DeviceContext()
    print("=== Anima tiled VAE decode CLI ===")
    print("  latent:", latent_path)
    print("  out   :", out_png, " unscale=", do_unscale)

    var st = ShardedSafeTensors.open(latent_path)
    var lat_raw = Tensor.from_view(st.tensor_view(String("latent")), ctx)
    var sh = List[Int]()
    sh.append(1)
    sh.append(ANIMA_LATENT_CHANNELS)
    sh.append(ANIMA_LATENT_H)
    sh.append(ANIMA_LATENT_W)
    var lat4 = reshape(lat_raw, sh^, ctx)
    var hv = lat4.to_host(ctx)
    print("  latent mean_abs:", _mean_abs16(hv))
    var lat: Tensor
    if do_unscale:
        var lmean = _lat_mean()
        var lstd = _lat_std()
        var hw = ANIMA_LATENT_H * ANIMA_LATENT_W
        for c in range(ANIMA_LATENT_CHANNELS):
            for i in range(hw):
                hv[c * hw + i] = hv[c * hw + i] * lstd[c] + lmean[c]
        print("  latent (raw) mean_abs after unscale:", _mean_abs16(hv))
        var sh2 = List[Int]()
        sh2.append(1); sh2.append(ANIMA_LATENT_CHANNELS)
        sh2.append(ANIMA_LATENT_H); sh2.append(ANIMA_LATENT_W)
        lat = Tensor.from_host(hv, sh2^, STDtype.BF16, ctx)
    else:
        lat = cast_tensor(lat4, STDtype.BF16, ctx)

    var rgb = wan21_image_tiled_decode[ANIMA_LATENT_H, ANIMA_LATENT_W](
        lat, String(ANIMA_VAE_PATH), ctx
    )  # NCHW [1,3,1024,1024], [-1,1]
    print("  decoded RGB:", rgb.shape()[2], "x", rgb.shape()[3])
    save_png(rgb, out_png, ctx, ValueRange.SIGNED)
    print("IMAGE SAVED:", out_png)
