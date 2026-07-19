# Anima VAE latent runtime smoke.
#
# Decodes the Rust cached-context Anima latent oracle through the local
# tiled Wan/Qwen-style image VAE and writes a 1024 PNG. This proves the VAE half
# of Anima's image path without porting MiniTrainDIT or prompt encoders.

from std.gpu.host import DeviceContext

from serenitymojo.image.png import ValueRange, save_png
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.anima_contract import (
    ANIMA_LATENT_H,
    ANIMA_LATENT_W,
    ANIMA_VAE_PATH,
    anima_default_rust_latent_path,
    validate_anima_rust_latent_header,
)
from serenitymojo.models.vae.qwenimage_tiled_decode import wan21_image_tiled_decode
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.tensor import Tensor


comptime OUT = "/home/alex/mojodiffusion/output/anima_vae_from_rust_latent_1024.png"


def main() raises:
    var ctx = DeviceContext()
    print("=== Anima tiled VAE latent runtime smoke ===")
    var latent_path = anima_default_rust_latent_path()
    _ = validate_anima_rust_latent_header(latent_path)
    var st = ShardedSafeTensors.open(latent_path)
    var latent5 = Tensor.from_view(st.tensor_view(String("latent")), ctx)
    var sh = List[Int]()
    sh.append(1)
    sh.append(16)
    sh.append(ANIMA_LATENT_H)
    sh.append(ANIMA_LATENT_W)
    var latent4 = reshape(latent5, sh^, ctx)
    var latent = cast_tensor(latent4, STDtype.BF16, ctx)

    var image = wan21_image_tiled_decode[ANIMA_LATENT_H, ANIMA_LATENT_W](
        latent, String(ANIMA_VAE_PATH), ctx
    )
    print("  decoded", image.shape()[2], "x", image.shape()[3])
    save_png(image, String(OUT), ctx, ValueRange.SIGNED)
    print("IMAGE SAVED:", OUT)
    print("Anima VAE latent runtime smoke PASS")
