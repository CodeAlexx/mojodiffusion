# SD1.5 VAE runtime smoke.
#
# This exercises the real SD1.5 diffusers VAE checkpoint through the Mojo LDM
# decoder path, including the legacy `query/key/value/proj_attn` mid-attention
# key spelling. It intentionally starts from deterministic latent noise; the
# CLIP+UNet denoise path remains separate work.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.sd15_contract import validate_sd15_metadata_contract
from serenitymojo.models.vae.ldm_decoder import load_sd15_ldm_decoder
from serenitymojo.registry.checkpoints import default_manifest_by_id
from serenitymojo.ops.random import randn
from serenitymojo.image.png import save_png, ValueRange


comptime OUT = "/home/alex/mojodiffusion/output/sd15_vae_noise_512.png"
comptime WIDTH = 512
comptime HEIGHT = 512
comptime LH = HEIGHT // 8
comptime LW = WIDTH // 8
comptime SEED = UInt64(42)


def main() raises:
    var manifest = default_manifest_by_id(String("sd15"))
    _ = validate_sd15_metadata_contract(manifest)

    var ctx = DeviceContext()
    print("=== SD1.5 VAE runtime smoke ===")
    print("  latent", 1, "x", 4, "x", LH, "x", LW)
    print("  checkpoint", manifest.vae_path)

    var shape = List[Int]()
    shape.append(1)
    shape.append(4)
    shape.append(LH)
    shape.append(LW)
    var latent = randn(shape^, SEED, STDtype.F32, ctx)

    var vae = load_sd15_ldm_decoder[LH, LW](manifest.vae_path, ctx)
    var image = vae.decode(latent, ctx)
    print("  decoded", image.shape()[2], "x", image.shape()[3])

    save_png(image, String(OUT), ctx, ValueRange.SIGNED)
    print("IMAGE SAVED:", OUT)
    print("SD1.5 VAE runtime smoke PASS")
