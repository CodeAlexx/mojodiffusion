# Compile-only rectangular specialization gate for both FLUX tiled VAE paths.
#
# Build this file; do not run it as a routine gate because it performs four VAE
# decodes.  Instantiating landscape + portrait arms catches height-derived
# width offsets and rectangular decoder-shape regressions at compile time.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.pipeline.flux_tiled_decode import (
    flux_tiled_decode,
    flux_tiled_decode_5x5_lowmem,
)


comptime VAE_PATH = "/home/alex/.serenity/models/vaes/ae.safetensors"


def _probe_3x3[LH: Int, LW: Int](ctx: DeviceContext) raises:
    var latent = randn([1, 16, LH, LW], UInt64(7), STDtype.F32, ctx)
    var image = flux_tiled_decode[LH, LW](latent, VAE_PATH, ctx)
    var shape = image.shape()
    if shape[2] != LH * 8 or shape[3] != LW * 8:
        raise Error("FLUX rectangular 3x3 tiled decode shape mismatch")


def _probe_5x5[LH: Int, LW: Int](ctx: DeviceContext) raises:
    var latent = randn([1, 16, LH, LW], UInt64(11), STDtype.F32, ctx)
    var image = flux_tiled_decode_5x5_lowmem[LH, LW](latent, VAE_PATH, ctx)
    var shape = image.shape()
    if shape[2] != LH * 8 or shape[3] != LW * 8:
        raise Error("FLUX rectangular 5x5 tiled decode shape mismatch")


def main() raises:
    var ctx = DeviceContext()
    _probe_3x3[96, 168](ctx)
    _probe_3x3[168, 96](ctx)
    _probe_5x5[96, 168](ctx)
    _probe_5x5[168, 96](ctx)
