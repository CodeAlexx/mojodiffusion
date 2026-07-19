# NAVA video VAE decode parity: Mojo Wan22VaeImageDecoder.decode_video_tokens on
# NAVA's Wan2.2_VAE vs torch (F32 production). latent [1280,48] (5x16x16) ->
# frames [1,3,17,256,256]. Gate cos>=0.999.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.vae.wan22_decoder import Wan22VaeImageDecoder

comptime CKPT = "/home/alex/.serenity/models/checkpoints/NAVA/Wan2.2_VAE.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity/nava_video_vae_fx.safetensors"


def main() raises:
    var ctx = DeviceContext()
    print("=== NAVA video VAE decode parity (Wan2.2, 5 lat frames -> 17 video) ===")
    var dec = Wan22VaeImageDecoder[16, 16].load(CKPT, ctx, f32=True)

    var fx = ShardedSafeTensors.open(FX)
    var lat = Tensor.from_view_as_f32(fx.tensor_view("lat_vid"), ctx)  # [1280,48]
    var ls = lat.shape()
    print("  lat_vid: [", ls[0], ",", ls[1], "]")

    var frames = dec.decode_video_tokens(lat, 5, ctx)  # [1,3,17,256,256]
    var fs = frames.shape()
    print("  frames out: [", fs[0], ",", fs[1], ",", fs[2], ",", fs[3], ",", fs[4], "]")

    var ref_host = Tensor.from_view(fx.tensor_view("frames"), ctx).to_host(ctx)
    print("NAVA video VAE decode vs torch:", ParityHarness(0.999).compare(frames, ref_host, ctx))
