# Wan2.2 TI2V-5B product VAE decoder parity against the pinned creator source.
#
# Generate the fixture with:
#   python3 scripts/wan22_vae_decode_oracle.py
# Then run:
#   pixi run mojo run -I . \
#     serenitymojo/models/vae/parity/wan22_vae_decoder_parity.mojo
#
# The product path loads the converted BF16 high-compression VAE and decodes
# five 16x16 latent frames into 17 256x256 RGB frames, exactly as T2V does.

from std.gpu.host import DeviceContext

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.vae.wan22_decoder import Wan22VaeImageDecoder
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor


comptime CKPT = "/home/alex/.serenity/models/vaes/wan2.2_vae.safetensors"
comptime FIXTURE = (
    "/home/alex/mojodiffusion/output/checks/wan22_20260729/vae/"
    "creator_decode_fixture.safetensors"
)
comptime COS_THRESHOLD = 0.999


def main() raises:
    var ctx = DeviceContext()
    print("=== Wan2.2 creator VAE decode parity (BF16 product path) ===")
    var fixture = ShardedSafeTensors.open(FIXTURE)
    var latent = Tensor.from_view_as_f32(
        fixture.tensor_view("lat_vid"), ctx
    )
    var reference = Tensor.from_view_as_f32(
        fixture.tensor_view("frames"), ctx
    ).to_host(ctx)

    var decoder = Wan22VaeImageDecoder[16, 16].load(CKPT, ctx)
    var frames = decoder.decode_video_tokens(latent, 5, ctx)
    var result = ParityHarness(COS_THRESHOLD).compare(
        frames, reference, ctx
    )
    print("Wan2.2 creator VAE decode:", result)
    if not result.passed:
        raise Error(
            String("Wan2.2 creator VAE decode parity failed: cos=")
            + String(result.cos)
            + " max_abs="
            + String(result.max_abs)
        )
    print(
        "GATE PASS wan22VaeDecodeCos=", result.cos,
        " maxAbs=", result.max_abs,
    )
