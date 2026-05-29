# ERNIE-Image resident DiT math smoke.
#
# Runs real checkpoint math for the resident pre-block path:
# latent patch projection, timestep MLP, and Mistral-hidden text projection.
# It intentionally bypasses Mistral3B and the 36 ERNIE DiT blocks.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.ernie_contract import (
    ERNIE_DIT_HIDDEN,
    ERNIE_DIT_TEXT_IN_DIM,
    ERNIE_LATENT_CHANNELS,
    ERNIE_LATENT_H,
    ERNIE_LATENT_W,
    ERNIE_TEXT_MAX_TOKENS,
    validate_ernie_metadata_contract,
)
from serenitymojo.models.dit.ernie_image import (
    ErnieImageResident,
    validate_ernie_adaln_shape,
    validate_ernie_resident_shapes,
)
from serenitymojo.ops.random import randn
from serenitymojo.runtime.model_manifest import ernie_image_default_manifest
from serenitymojo.tensor import Tensor


comptime SEED = UInt64(20260528)


def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var n = len(h)
    if n == 0:
        raise Error(String("empty tensor stats: ") + name)
    var sum = Float64(0.0)
    var sum2 = Float64(0.0)
    var amax = Float64(0.0)
    for i in range(n):
        var v = Float64(h[i])
        sum += v
        sum2 += v * v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    var mean = sum / Float64(n)
    var var_ = sum2 / Float64(n) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print(
        "  [stat]",
        name,
        "mean=",
        Float32(mean),
        "std=",
        Float32(sqrt(var_)),
        "absmax=",
        Float32(amax),
        "n=",
        n,
    )
    if amax == 0.0:
        raise Error(String("all-zero tensor stats: ") + name)


def main() raises:
    var manifest = ernie_image_default_manifest()
    _ = validate_ernie_metadata_contract(manifest)

    var ctx = DeviceContext()
    print("=== ERNIE-Image resident DiT math smoke ===")
    print(
        "  latent",
        1,
        "x",
        ERNIE_LATENT_CHANNELS,
        "x",
        ERNIE_LATENT_H,
        "x",
        ERNIE_LATENT_W,
    )
    print(
        "  text",
        1,
        "x",
        ERNIE_TEXT_MAX_TOKENS,
        "x",
        ERNIE_DIT_TEXT_IN_DIM,
    )

    var model = ErnieImageResident.load_default(ctx)

    var latent_shape = List[Int]()
    latent_shape.append(1)
    latent_shape.append(ERNIE_LATENT_CHANNELS)
    latent_shape.append(ERNIE_LATENT_H)
    latent_shape.append(ERNIE_LATENT_W)
    var latent = randn(latent_shape^, SEED, STDtype.BF16, ctx)

    var text_shape = List[Int]()
    text_shape.append(1)
    text_shape.append(ERNIE_TEXT_MAX_TOKENS)
    text_shape.append(ERNIE_DIT_TEXT_IN_DIM)
    var text = randn(text_shape^, SEED + 1, STDtype.BF16, ctx)

    var t_vals = List[Float32]()
    t_vals.append(1000.0)
    var t_shape = List[Int]()
    t_shape.append(1)
    var timestep = Tensor.from_host(t_vals, t_shape^, STDtype.F32, ctx)

    var patch_tokens = model.patch_embed_1024(latent, ctx)
    var temb = model.time_embed(timestep, ctx)
    var text_tokens = model.project_text(text, ctx)
    var adaln = model.shared_adaln(temb, ctx)
    validate_ernie_resident_shapes(patch_tokens, temb, text_tokens)
    validate_ernie_adaln_shape(adaln)

    print(
        "  patch/text/temb:",
        patch_tokens.shape()[1],
        "x",
        patch_tokens.shape()[2],
        text_tokens.shape()[1],
        "x",
        text_tokens.shape()[2],
        temb.shape()[1],
    )
    _stats("patch_tokens", patch_tokens, ctx)
    _stats("temb", temb, ctx)
    _stats("adaln", adaln, ctx)
    _stats("text_tokens", text_tokens, ctx)
    print("ERNIE-Image resident DiT math smoke PASS")
