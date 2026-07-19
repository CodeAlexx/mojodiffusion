# SDXL model BF16 tensor round-trip contract gate.
#
# This is split out of `sdxl_model_compile_check.mojo` because `DeviceContext`
# requires GPU architecture detection and fails inside the current sandbox.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenity_trainer.model.StableDiffusionXLModel import (
    stable_diffusion_xl_latents_from_vae_input,
    stable_diffusion_xl_vae_decode_input,
)


def _expect_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name + String(": got ") + String(got)
            + String(", expected ") + String(expected)
        )


def _expect_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(name + String(": got ") + got + String(", expected ") + expected)


def main() raises:
    var ctx = DeviceContext()
    var vals = List[Float32]()
    for i in range(4 * 2 * 2):
        vals.append(Float32(i % 17) * 0.01)
    var latent_dims = List[Int]()
    latent_dims.append(1)
    latent_dims.append(4)
    latent_dims.append(2)
    latent_dims.append(2)
    var latent = Tensor.from_host(vals^, latent_dims^, STDtype.BF16, ctx)

    var decode_input = stable_diffusion_xl_vae_decode_input(
        latent, Float32(0.13025), ctx
    )
    var scaled_latents = stable_diffusion_xl_latents_from_vae_input(
        decode_input, Float32(0.13025), ctx
    )
    _expect_string("scaled latent dtype", scaled_latents.dtype().name(), String("BF16"))
    _expect_int("scaled latent rank", len(scaled_latents.shape()), 4)
    _expect_int("scaled latent batch", scaled_latents.shape()[0], 1)
    _expect_int("scaled latent channels", scaled_latents.shape()[1], 4)

    print("SDXL MODEL TENSOR CONTRACT OK")
