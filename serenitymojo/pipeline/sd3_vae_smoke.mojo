# SD3.5 Large embedded-VAE runtime smoke.
#
# This decodes deterministic latent noise through the VAE stored inside the
# SD3.5 Large checkpoint. It intentionally does not run MMDiT or text encoders.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.sd3_contract import (
    SD3_LARGE_LATENT_CHANNELS,
    SD3_LARGE_LATENT_H,
    SD3_LARGE_LATENT_W,
    validate_sd3_large_pipeline_contract,
)
from serenitymojo.models.vae.ldm_decoder import load_sd3_embedded_ldm_decoder
from serenitymojo.registry.checkpoints import default_manifest_by_id
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.image.png import save_png, ValueRange


comptime OUT = "/home/alex/mojodiffusion/output/sd3_vae_noise_1024.png"
comptime SEED = UInt64(42)


def main() raises:
    var manifest = default_manifest_by_id(String("sd3_5_large"))
    validate_sd3_large_pipeline_contract(manifest)

    var ctx = DeviceContext()
    print("=== SD3.5 Large embedded VAE runtime smoke ===")
    print(
        "  latent",
        1,
        "x",
        SD3_LARGE_LATENT_CHANNELS,
        "x",
        SD3_LARGE_LATENT_H,
        "x",
        SD3_LARGE_LATENT_W,
    )
    print("  checkpoint", manifest.vae_path)

    var shape = List[Int]()
    shape.append(1)
    shape.append(SD3_LARGE_LATENT_CHANNELS)
    shape.append(SD3_LARGE_LATENT_H)
    shape.append(SD3_LARGE_LATENT_W)
    var latent_f32 = randn(shape^, SEED, STDtype.F32, ctx)
    var latent = cast_tensor(latent_f32, STDtype.BF16, ctx)

    var vae = load_sd3_embedded_ldm_decoder[
        SD3_LARGE_LATENT_H, SD3_LARGE_LATENT_W
    ](manifest.vae_path, ctx)
    var image = vae.decode(latent, ctx)
    print("  decoded", image.shape()[2], "x", image.shape()[3])

    save_png(image, String(OUT), ctx, ValueRange.SIGNED)
    print("IMAGE SAVED:", OUT)
    print("SD3.5 Large embedded VAE runtime smoke PASS")
