# Qwen-Image VAE runtime smoke.
#
# Exercises the real Qwen-Image VAE decoder against deterministic BF16 latent
# noise. This isolates decoder key/dtype/runtime issues from the slower
# text-encoder + streamed-DiT path.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder
from serenitymojo.ops.random import randn
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.image.png import save_png, ValueRange


comptime VAE_DIR = "/home/alex/.serenity/models/checkpoints/qwen-image-2512/vae"
comptime OUT = "/home/alex/mojodiffusion/output/qwenimage_vae_noise_512.png"
comptime LH = 64
comptime LW = 64
comptime SEED = UInt64(42)


def main() raises:
    var ctx = DeviceContext()
    print("=== Qwen-Image VAE runtime smoke ===")
    var shape = List[Int]()
    shape.append(1)
    shape.append(16)
    shape.append(LH)
    shape.append(LW)
    var latent = cast_tensor(randn(shape^, SEED, STDtype.F32, ctx), STDtype.BF16, ctx)
    var vae = QwenImageVaeDecoder[LH, LW].load(String(VAE_DIR), ctx)
    var image = vae.decode(latent, ctx)
    print("  decoded", image.shape()[2], "x", image.shape()[3])
    save_png(image, String(OUT), ctx, ValueRange.SIGNED)
    print("IMAGE SAVED:", OUT)
    print("Qwen-Image VAE runtime smoke PASS")
