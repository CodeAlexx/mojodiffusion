# ldm_encoder_probe.mojo — real-weight run probe for the SDXL LDM VAE encoder.
#
# Loads the standalone sdxl_vae.safetensors (LDM-format encoder keys + quant_conv)
# at a small comptime latent (8x8 -> image 64x64) and runs encode_moments /
# encode_mean / encode on a deterministic input, printing finite stats. Real GPU
# forward to exit 0; this is the compiles+runs gate for the chunk.
#
# Run: cd /home/alex/mojodiffusion &&
#   pixi run mojo run -I . serenitymojo/models/vae/ldm_encoder_probe.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.ldm_encoder import (
    LdmVaeEncoder,
    load_sdxl_ldm_encoder,
)


comptime SDXL_VAE = "/home/alex/.serenity/models/vaes/OfficialStableDiffusion/sdxl_vae.safetensors"


def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var n = len(h)
    var mn = h[0]
    var mx = h[0]
    var sum = Float64(0.0)
    var sumsq = Float64(0.0)
    var bad = 0
    for i in range(n):
        var v = h[i]
        if v < mn:
            mn = v
        if v > mx:
            mx = v
        sum += Float64(v)
        sumsq += Float64(v) * Float64(v)
        # NaN: v != v ; Inf: |v| huge
        if v != v or v > 3.0e38 or v < -3.0e38:
            bad += 1
    var mean = sum / Float64(n)
    var var_ = sumsq / Float64(n) - mean * mean
    print(
        name,
        "n=", n,
        "min=", Float64(mn),
        "max=", Float64(mx),
        "mean=", mean,
        "std=", sqrt(var_ if var_ > 0.0 else 0.0),
        "bad=", bad,
    )
    if bad != 0:
        raise Error(name + ": non-finite values present")


def _input(ctx: DeviceContext) raises -> Tensor:
    # [1,3,64,64] deterministic ramp in [-1,1], BF16 (the Rust encoder feeds a
    # BF16 [B,3,H,W] image; ldm_encoder.rs encode() docstring + to_dtype(BF16)).
    var n = 1 * 3 * 64 * 64
    var v = List[Float32]()
    for i in range(n):
        var x = Float32(i % 251) / Float32(125.0) - 1.0
        v.append(x)
    var sh = List[Int]()
    sh.append(1)
    sh.append(3)
    sh.append(64)
    sh.append(64)
    return Tensor.from_host(v^, sh^, STDtype.BF16, ctx)


def main() raises:
    var ctx = DeviceContext()
    print("ctx ok", Int(ctx.id()))

    # latent 8x8 -> image 64x64.
    var enc = load_sdxl_ldm_encoder[8, 8](String(SDXL_VAE), ctx)
    print("encoder loaded (SDXL LDM, latent_ch=4, quant_conv present)")

    var img = _input(ctx)
    _stats(String("input"), img, ctx)

    var moments = enc.encode_moments(img, ctx)
    var ms = moments.shape()
    print(
        "moments shape:",
        ms[0], ms[1], ms[2], ms[3],
        "(expect 1 8 8 8)",
    )
    if ms[0] != 1 or ms[1] != 8 or ms[2] != 8 or ms[3] != 8:
        raise Error("moments shape mismatch")
    _stats(String("moments"), moments, ctx)

    var mean = enc.encode_mean(img, ctx)
    var meanshape = mean.shape()
    print(
        "mean latent shape:",
        meanshape[0], meanshape[1], meanshape[2], meanshape[3],
        "(expect 1 4 8 8)",
    )
    if (
        meanshape[0] != 1
        or meanshape[1] != 4
        or meanshape[2] != 8
        or meanshape[3] != 8
    ):
        raise Error("mean latent shape mismatch")
    _stats(String("mean"), mean, ctx)

    var sampled = enc.encode(img, UInt64(42), ctx)
    _stats(String("sampled"), sampled, ctx)

    print("ldm_encoder probe OK")
