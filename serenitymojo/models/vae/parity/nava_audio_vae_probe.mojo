# NAVA audio VAE decoder parity: Mojo ltx2_audio_vae.decode on NAVA's
# ltx-2.3-22b-dev_audio_vae.safetensors vs torch decoder mel. PRODUCTION = F32
# (init_ltx_vae builds dtype=torch.float32), so we decode in F32 (f32=True).
# Latent [1,8,34,16] -> mel [1,2,133,64].  Gate cos>=0.999.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.vae.ltx2_audio_vae import (
    LTX2AudioVaeDecoderWeights,
    decode,
)

comptime CKPT = "/home/alex/.serenity/models/checkpoints/NAVA/params/LTX2/ltx-2.3-22b-dev_audio_vae.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity/nava_audio_vae_fx.safetensors"


def main() raises:
    var ctx = DeviceContext()
    print("=== NAVA audio VAE decoder parity (F32, production dtype) ===")
    var weights = LTX2AudioVaeDecoderWeights.load(CKPT, ctx, f32=True)

    var fx = ShardedSafeTensors.open(FX)
    var latent = Tensor.from_view_as_f32(fx.tensor_view("latent_in"), ctx)  # [1,8,34,16] F32
    var ls = latent.shape()
    print("  latent_in: [", ls[0], ",", ls[1], ",", ls[2], ",", ls[3], "]")

    var mel = decode(weights, latent, ctx)  # [1,2,133,64] NCHW
    var ms = mel.shape()
    print("  mel out:  [", ms[0], ",", ms[1], ",", ms[2], ",", ms[3], "]")

    var ref_host = Tensor.from_view(fx.tensor_view("mel"), ctx).to_host(ctx)
    print("NAVA audio VAE decode vs torch:", ParityHarness(0.999).compare(mel, ref_host, ctx))
