# chunk8 parity: Ideogram-4 Flux2 VAE decode (z=32) vs Wave-0 fixture.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.vae.ldm_decoder import load_ideogram4_vae_decoder

comptime VAE = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_vae.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var dec = load_ideogram4_vae_decoder[32, 32](VAE, ctx)
    var fx = ShardedSafeTensors.open(FX)
    var z = cast_tensor(Tensor.from_view(fx.tensor_view("chunk8.latent"), ctx), STDtype.BF16, ctx)
    var out = dec.decode(z, ctx)   # [1,3,256,256]
    print("decoded:", out.shape()[0], out.shape()[1], out.shape()[2], out.shape()[3])
    var exp_host = Tensor.from_view(fx.tensor_view("chunk8.decoded"), ctx).to_host(ctx)
    print("chunk8 vae decode parity:", ParityHarness(0.999).compare(out, exp_host, ctx))
