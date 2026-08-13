# zimage_vae_compile_check.mojo — compile gate for the Z-Image VAE port.
# References scale/unscale + encoder/decoder structs + the encode_image/
# decode_latent seams so the compiler instantiates them.
# Compile only (the .load() paths need real weights, so we don't run them):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -I trainer/src \
#       trainer/smoke/zimage_vae_compile_check.mojo -o /tmp/zvae_check

from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenity_trainer.model.ZImageVAE import (
    scale_latents,
    unscale_latents,
    ZImageVaeEncoder,
    ZImageDecoder,
    encode_image,
    decode_latent,
    ZIMAGE_VAE_SHIFT_FACTOR,
    ZIMAGE_VAE_SCALING_FACTOR,
)

comptime LH = 16   # tiny latent for a fast compile-gate (128px image)
comptime LW = 16


def main() raises:
    var ctx = DeviceContext()

    # latent affine round-trip (the predict/sampler seam).
    var v = List[Float32]()
    for i in range(16):
        v.append(Float32(i) * 0.05 - 0.3)
    var sh = List[Int](); sh.append(16)
    var lat = Tensor.from_host(v.copy(), sh.copy(), STDtype.BF16, ctx)
    var scaled = scale_latents(lat, ctx)
    var back = unscale_latents(scaled, ctx)
    print("scale/unscale OK; dtype =", back.dtype().name())
    print("shift =", ZIMAGE_VAE_SHIFT_FACTOR, " scaling =", ZIMAGE_VAE_SCALING_FACTOR)

    # Type-instantiate the encoder/decoder structs (forces their fields +
    # comptime params to typecheck). No .load()/.encode()/.decode() — those need
    # the real VAE checkpoint; this gate is compile-only. encode_image /
    # decode_latent / encode_mean / decode are typechecked at struct definition.
    comptime _Enc = ZImageVaeEncoder[LH, LW]
    comptime _Dec = ZImageDecoder[LH, LW]
    print("encoder IH =", _Enc.IH, " decoder out =", 8 * LH)

    # Force monomorphization of the parametric high-level seams without
    # executing GPU work. The `if False:` guard keeps the .load()/seam calls in
    # the compilation unit (so encode_image[LH,LW]/decode_latent[LH,LW] type-check
    # end-to-end) yet never runs them — needs no real weights. This is the proven
    # serenitymojo pattern (cf. models/vae/ldm_decoder_probe.mojo:_instantiate);
    # a free parametric `def` bound to a comptime alias is NOT a proven form.
    if False:
        var enc = ZImageVaeEncoder[LH, LW].load(String("/nonexistent"), ctx)
        var img = Tensor.from_host(
            _zeros(1 * 3 * 8 * LH * 8 * LW), _shape4(1, 3, 8 * LH, 8 * LW),
            STDtype.BF16, ctx,
        )
        var latent = encode_image[LH, LW](enc, img, ctx)
        var dec = ZImageDecoder[LH, LW].load(String("/nonexistent"), ctx)
        var out_img = decode_latent[LH, LW](dec, latent, ctx)
        print(out_img.shape()[0])

    print("ZIMAGE VAE COMPILE-CHECK OK")


def _zeros(n: Int) -> List[Float32]:
    var v = List[Float32]()
    for _ in range(n):
        v.append(Float32(0.0))
    return v^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    return s^
