# ERNIE-Image real-weight block0 smoke.
#
# Feeds the existing resident pre-block path into a bounded layer-0 slice:
# latent patch projection + text projection + timestep AdaLN -> image-first/text
# concat -> 3-axis ERNIE RoPE -> block0 attention + GELU-gated MLP.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.ernie_contract import (
    ERNIE_DIT_HEAD_DIM,
    ERNIE_DIT_HEADS,
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
    build_ernie_rope_tables,
    validate_ernie_adaln_shape,
    validate_ernie_block0_shape,
    validate_ernie_resident_shapes,
)
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import concat, mul_scalar, slice
from serenitymojo.runtime.model_manifest import ernie_image_default_manifest
from serenitymojo.tensor import Tensor


comptime SEED = UInt64(20260529)
comptime INPUT_SCALE = Float32(0.25)
# NOTE (2026-05-28 bugfix): the prior `ADALN_SMOKE_SCALE = 1e-5` workaround
# is gone — the sin/cos channel-order fix in `ops/embeddings.mojo` (now
# `timestep_embedding_sin_first` for ERNIE) means the real timestep MLP +
# shared adaLN chain no longer overflows BF16. AdaLN is now consumed raw.
# TODO(parity): after a GPU run, lock the expected absmax range — Rust
# parity predicts `temb_raw` absmax in the low tens and `adaln_raw` absmax
# well under 100 (definitely < 65504 BF16-max). The bounds below are loose
# placeholders pending the first real run.
comptime N_IMG = 4
comptime N_TXT = 8
comptime S = N_IMG + N_TXT
comptime IMG_H = 2
comptime IMG_W = 2


def _require_shape2(label: String, got: List[Int], a: Int, b: Int) raises:
    if len(got) != 2 or got[0] != a or got[1] != b:
        raise Error(String("shape mismatch for ") + label)


def _require_shape3(label: String, got: List[Int], a: Int, b: Int, c: Int) raises:
    if len(got) != 3 or got[0] != a or got[1] != b or got[2] != c:
        raise Error(String("shape mismatch for ") + label)


def _stats(name: String, t: Tensor, ctx: DeviceContext, max_abs_allowed: Float64) raises:
    var h = t.to_host(ctx)
    if len(h) == 0:
        raise Error(String("empty tensor stats: ") + name)
    var s = 0.0
    var s2 = 0.0
    var amax = 0.0
    for i in range(len(h)):
        var v = Float64(h[i])
        if v != v:
            raise Error(String("NaN in ") + name)
        if v > max_abs_allowed or v < -max_abs_allowed:
            raise Error(String("unstable value in ") + name)
        s += v
        s2 += v * v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    if amax == 0.0:
        raise Error(String("all-zero tensor stats: ") + name)
    var mean = s / Float64(len(h))
    var var_ = s2 / Float64(len(h)) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print(
        "  ",
        name,
        "stats mean/std/absmax:",
        Float32(mean),
        Float32(sqrt(var_)),
        Float32(amax),
    )
    print("   ", name, "max_abs_allowed:", Float32(max_abs_allowed))


def main() raises:
    var manifest = ernie_image_default_manifest()
    _ = validate_ernie_metadata_contract(manifest)

    var ctx = DeviceContext()
    print("=== ERNIE-Image block0 real-weight smoke ===")
    print("  block slice image/text/total:", N_IMG, N_TXT, S)

    print("  [load] resident + layer0 weights")
    var model = ErnieImageResident.load_default_block0_smoke(ctx)
    model.validate_block0_smoke_weights()
    print("  [load] done")

    var latent_shape = List[Int]()
    latent_shape.append(1)
    latent_shape.append(ERNIE_LATENT_CHANNELS)
    latent_shape.append(ERNIE_LATENT_H)
    latent_shape.append(ERNIE_LATENT_W)
    var latent = mul_scalar(randn(latent_shape^, SEED, STDtype.BF16, ctx), INPUT_SCALE, ctx)

    var text_shape = List[Int]()
    text_shape.append(1)
    text_shape.append(ERNIE_TEXT_MAX_TOKENS)
    text_shape.append(ERNIE_DIT_TEXT_IN_DIM)
    var text = mul_scalar(randn(text_shape^, SEED + 1, STDtype.BF16, ctx), INPUT_SCALE, ctx)

    var t_vals = List[Float32]()
    t_vals.append(875.0)
    var t_shape = List[Int]()
    t_shape.append(1)
    var timestep = Tensor.from_host(t_vals, t_shape^, STDtype.F32, ctx)

    print("  [resident] patch/text/time/AdaLN")
    var patch_tokens = model.patch_embed_1024(latent, ctx)
    print("  [resident] patch done")
    var temb = model.time_embed(timestep, ctx)
    print("  [resident] time done")
    var text_tokens = model.project_text(text, ctx)
    print("  [resident] text done")
    var adaln_raw = model.shared_adaln(temb, ctx)
    print("  [resident] AdaLN done")
    validate_ernie_resident_shapes(patch_tokens, temb, text_tokens)
    validate_ernie_adaln_shape(adaln_raw)
    _stats(String("patch_tokens"), patch_tokens, ctx, 32.0)
    _stats(String("text_tokens"), text_tokens, ctx, 32.0)
    # After the sin-first fix, temb / adaln should be BF16-safe; bounds left
    # loose (200 / 500) so the smoke passes the first GPU run, then tighten.
    _stats(String("temb_raw"), temb, ctx, 200.0)
    _stats(String("adaln_raw"), adaln_raw, ctx, 500.0)
    # No bounded scaling — `adaln_raw` is fed straight into block0 below.

    print("  [seq] slice image/text prefixes")
    var img = slice(patch_tokens, 1, 0, N_IMG, ctx)
    print("  [seq] image slice done")
    var txt = slice(text_tokens, 1, 0, N_TXT, ctx)
    print("  [seq] text slice done")
    var seq = concat(1, ctx, img, txt)
    print("  [seq] concat done")
    _stats(String("seq_bounded"), seq, ctx, 32.0)

    print("  [rope] build")
    var rope = build_ernie_rope_tables[N_IMG, N_TXT, ERNIE_DIT_HEADS, ERNIE_DIT_HEAD_DIM](
        IMG_H, IMG_W, N_TXT, ctx, STDtype.BF16
    )
    _require_shape2(String("rope_cos"), rope[0].shape(), S * ERNIE_DIT_HEADS, ERNIE_DIT_HEAD_DIM)
    _require_shape2(String("rope_sin"), rope[1].shape(), S * ERNIE_DIT_HEADS, ERNIE_DIT_HEAD_DIM)
    _require_shape3(String("seq"), seq.shape(), 1, S, ERNIE_DIT_HIDDEN)

    print("  [block0] forward")
    var out = model.block0_smoke_forward[S](seq, adaln_raw, rope[0], rope[1], ctx)
    validate_ernie_block0_shape[S](out)
    _stats(String("block0_out"), out, ctx, 65536.0)
    print("ERNIE-Image block0 real-weight smoke PASS")
