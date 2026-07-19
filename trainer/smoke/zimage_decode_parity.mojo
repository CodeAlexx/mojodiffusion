# VAE decode parity: decode Serenity's EXACT latent_final, compare to zi_SERENITY_1024.png.
# decode() _rescales internally → pass the RAW (scaled) latent, NOT pre-unscaled.
from std.time import monotonic
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.image.png import save_png, ValueRange
from serenity_trainer.model.ZImageVAE import ZImageDecoder, decode_latent

comptime LH = 128
comptime LW = 128

def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(String("trainer/parity/zi_gen1024_ref.safetensors"))
    var lat = cast_tensor(Tensor.from_view(st.tensor_view(String("latent_final")), ctx), STDtype.BF16, ctx)  # RAW scaled latent
    print("loading VAE ...")
    var dec = ZImageDecoder[LH, LW].load(String("/home/alex/.serenity/models/zimage_base/vae"), ctx)
    print("decoding (raw latent, decode() rescales internally) ...")
    var t0 = monotonic()
    var img = decode_latent[LH, LW](dec, lat, ctx)   # NO external unscale — decode does _rescale
    ctx.synchronize()
    var t1 = monotonic()
    print("decode wall =", Float64(t1 - t0) / 1e9, "s   image", img.shape()[2], "x", img.shape()[3])
    save_png(img, String("trainer/parity/zi_MOJO_decode.png"), ctx, ValueRange.SIGNED)
    print("WROTE parity/zi_MOJO_decode.png")
