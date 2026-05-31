# klein_encode_smoke.mojo - FLUX.2/Klein VAE ENCODER correctness gate.
#
# Loads flux2-vae.safetensors, encodes a REAL image (a 512x512 Alina crop,
# preprocessed exactly like prepare_klein.rs: resize-exact, HWC->CHW, *2-1),
# and asserts the packed latent has:
#   shape [1,128,32,32]
#   std ~= 0.96  (the BN-normalised target; a HWC->CHW channel scramble would
#                 give ~0.85 -- this is the footgun gate, project memory
#                 feedback_prepare_bins_chw_transpose).
#
# The image is provided as a single-file safetensors `image` [1,3,512,512] F32
# (PNG decode is not yet available in pure Mojo -- png.mojo is encode-only --
# so the image is staged offline to a raw-tensor file; see RETURN notes).
#
# Run:
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/pipeline/klein_encode_smoke.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.vae.klein_encoder import KleinVaeEncoder


comptime VAE_PATH = "/home/alex/.serenity/models/vaes/flux2-vae.safetensors"
comptime IMG_PATH = "/home/alex/mojodiffusion/output/alina_512_image.safetensors"
comptime IH = 512
comptime IW = 512


def _load_image(ctx: DeviceContext) raises -> Tensor:
    var st = SafeTensors.open(IMG_PATH)
    var info = st.tensor_info(String("image"))
    var bytes = st.tensor_bytes(String("image"))
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def _std(t: Tensor, ctx: DeviceContext) raises -> Float64:
    var h = t.to_host(ctx)
    var n = len(h)
    var s = 0.0
    var s2 = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v
        s2 += v * v
    var mean = s / Float64(n)
    var var_ = s2 / Float64(n) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    return sqrt(var_)


def main() raises:
    var ctx = DeviceContext()
    print("=== Klein VAE ENCODER smoke (image -> packed latent) ===")
    print("[load image]", IMG_PATH)
    var img = _load_image(ctx)
    var ish = img.shape()
    print("  image shape:", ish[0], ish[1], ish[2], ish[3], "dtype", img.dtype().name())

    print("[load encoder]", VAE_PATH)
    var enc = KleinVaeEncoder[IH, IW].load(VAE_PATH, ctx)

    print("[encode]")
    var z = enc.encode(img, ctx)
    var zsh = z.shape()
    print("  latent shape:", zsh[0], zsh[1], zsh[2], zsh[3])
    var std = _std(z, ctx)
    print("  latent std =", Float32(std))

    var shape_ok = (
        len(zsh) == 4
        and zsh[0] == 1
        and zsh[1] == 128
        and zsh[2] == IH // 16
        and zsh[3] == IW // 16
    )
    # std target ~0.96; channel scramble -> ~0.85. Accept [0.90, 1.05].
    var std_ok = std >= 0.90 and std <= 1.05
    if not shape_ok:
        print("FAIL: latent shape != [1,128,", IH // 16, ",", IW // 16, "]")
        return
    if not std_ok:
        print("FAIL: latent std", Float32(std), "outside [0.90, 1.05]")
        print("  (std ~0.85 indicates the HWC->CHW channel scramble footgun)")
        return
    print("PASS: shape [1,128,", IH // 16, ",", IW // 16, "] and std ~= 0.96 (", Float32(std), ")")
