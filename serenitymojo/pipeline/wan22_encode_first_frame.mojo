# Wan2.2 TI2V-5B creator first-frame VAE staging, pure Mojo.
#
# The VAE encoder runs in a dedicated process so its CUDA allocator and
# temporary 3D-convolution buffers are destroyed before the 5B DiT process
# starts. This is the same phase isolation already used for UMT5 conditioning
# and is required for native BF16 I2V on a 24 GB GPU.
#
# argv: <source_image> <out.safetensors>
# output key: first_latent [48,1,H/16,W/16] F32

from std.collections import List
from std.gpu.host import DeviceContext
from std.math import round
from std.memory import ArcPointer
from std.sys import argv
from std.sys.defines import get_defined_int

from image.studio_ops import resize_lanczos
from image.transform import crop

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.models.vae.wan22_vae_encoder import Wan22VaeImageEncoder
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.serve.image_io import (
    decode_image_any,
    image_to_signed_nchw,
)
from serenitymojo.tensor import Tensor


comptime VAE_PATH = "/home/alex/.serenity/models/vaes/wan2.2_vae.safetensors"
comptime HEIGHT = get_defined_int["WAN22_HEIGHT", 704]()
comptime WIDTH = get_defined_int["WAN22_WIDTH", 1280]()
comptime H_LAT = HEIGHT // 16
comptime W_LAT = WIDTH // 16


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error(
            "usage: wan22_encode_first_frame <source_image> <out.safetensors>"
        )
    var image_path = String(args[1])
    var output_path = String(args[2])
    var image = decode_image_any(image_path)

    # Exact creator cover-resize and center-crop for the already-derived
    # 32-aligned output profile.
    var scale = max(
        Float64(WIDTH) / Float64(image.width),
        Float64(HEIGHT) / Float64(image.height),
    )
    var resized_width = Int(round(Float64(image.width) * scale))
    var resized_height = Int(round(Float64(image.height) * scale))
    var resized = resize_lanczos(image, resized_width, resized_height)
    var crop_left = (resized.width - WIDTH) // 2
    var crop_top = (resized.height - HEIGHT) // 2
    var cropped = crop(resized, crop_left, crop_top, WIDTH, HEIGHT)
    var values = image_to_signed_nchw(cropped)

    var ctx = DeviceContext()
    var pixels = Tensor.from_host(
        values, [1, 3, HEIGHT, WIDTH], STDtype.BF16, ctx
    )
    var vae = Wan22VaeImageEncoder[HEIGHT, WIDTH].load(
        String(VAE_PATH), ctx
    )
    var encoded = vae.encode_image(pixels, ctx)
    var first = cast_tensor(
        reshape(encoded, [48, 1, H_LAT, W_LAT], ctx),
        STDtype.F32,
        ctx,
    )
    var names = List[String]()
    names.append("first_latent")
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer(first^))
    save_safetensors(names, tensors, output_path, ctx)
    print(
        "GATE wrote Wan creator first latent [48,1,",
        H_LAT, ",", W_LAT, "] ->", output_path,
    )
