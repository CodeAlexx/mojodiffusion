# Parity: Ideogram-4 VAE *encoder* (training path) vs torch oracle.
# Oracle = ideogram4_vae_encode_oracle.py (ae.encoder -> mean[:,:32] -> patchify
# -> (patched-shift)/scale). Gates moments / mean / final normalized latents.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.vae.ldm_encoder import (
    load_ideogram4_vae_encoder,
    ideogram4_patchify_latents,
    ideogram4_normalize_latents,
    encode_ideogram4_latents,
)

comptime VAE = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity/ideogram4_fx_vae_encode.safetensors"


def main() raises:
    var ctx = DeviceContext()
    # image 256x256 -> latent 32x32 -> LH=LW=32; packed gh=gw=16.
    var enc = load_ideogram4_vae_encoder[32, 32](VAE, ctx)
    var fx = ShardedSafeTensors.open(FX)

    var img = cast_tensor(Tensor.from_view(fx.tensor_view("image"), ctx), STDtype.BF16, ctx)
    var shift = Tensor.from_view(fx.tensor_view("latent_shift"), ctx)  # [128] F32
    var scale = Tensor.from_view(fx.tensor_view("latent_scale"), ctx)  # [128] F32

    # stage 1: raw moments [1,64,32,32] (encode_moments == ae.encoder incl quant_conv)
    var moments = enc.encode_moments(img, ctx)
    var mom_exp = Tensor.from_view(fx.tensor_view("moments"), ctx).to_host(ctx)
    print("moments parity:", ParityHarness(0.999).compare(moments, mom_exp, ctx))

    # stage 2: deterministic mean latent [1,32,32,32]
    var mean = enc.encode_mean(img, ctx)
    var mean_exp = Tensor.from_view(fx.tensor_view("mean"), ctx).to_host(ctx)
    print("mean parity:", ParityHarness(0.999).compare(mean, mean_exp, ctx))

    # stage 3: full normalized packed latents [1,128,16,16]
    var lat = encode_ideogram4_latents[32, 32](enc, img, shift, scale, ctx)
    print("latents:", lat.shape()[0], lat.shape()[1], lat.shape()[2], lat.shape()[3])
    var lat_exp = Tensor.from_view(fx.tensor_view("latents"), ctx).to_host(ctx)
    print("latents parity:", ParityHarness(0.999).compare(lat, lat_exp, ctx))
