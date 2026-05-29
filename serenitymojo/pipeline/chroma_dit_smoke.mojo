# Chroma DiT real-weight staged smoke.
#
# This is not a full image generator. It proves the Chroma-specific runtime
# path that is absent from FLUX: distilled_guidance_layer -> pooled_temb,
# two double blocks, the first two single blocks, and final image projection on
# a static token grid.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.chroma_contract import (
    CHROMA_DIT_HIDDEN,
    CHROMA_DIT_MOD_INDEX,
)
from serenitymojo.models.dit.chroma_dit import (
    ChromaDitCache,
    chroma_step_cache_stats,
)
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.tensor import Tensor


comptime N_IMG = 4
comptime N_TXT = 8
comptime S = N_IMG + N_TXT
comptime IMG_H2 = 2
comptime IMG_W2 = 2


def _require_shape3(label: String, got: List[Int], a: Int, b: Int, c: Int) raises:
    if len(got) != 3 or got[0] != a or got[1] != b or got[2] != c:
        raise Error(String("shape mismatch for ") + label)


def _require_shape2(label: String, got: List[Int], a: Int, b: Int) raises:
    if len(got) != 2 or got[0] != a or got[1] != b:
        raise Error(String("shape mismatch for ") + label)


def _tiny_tokens[
    D: Int
](n: Int, scale: Float32, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32]()
    for tok in range(n):
        for ch in range(D):
            var raw = ((tok + 1) * (ch + 3)) % 23
            vals.append((Float32(raw) - 11.0) * scale)
    var sh = List[Int]()
    sh.append(1)
    sh.append(n)
    sh.append(D)
    return Tensor.from_host(vals, sh^, STDtype.BF16, ctx)


def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    if len(h) == 0:
        raise Error(String("empty tensor stats: ") + name)
    var s = 0.0
    var s2 = 0.0
    var amax = 0.0
    for i in range(len(h)):
        var v = Float64(h[i])
        if v != v:
            raise Error(String("non-finite NaN in ") + name)
        s += v
        s2 += v * v
        var av = v if v >= 0.0 else -v
        if av > 1.0e30:
            raise Error(String("non-finite or unstable value in ") + name)
        if av > amax:
            amax = av
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


def main() raises:
    var ctx = DeviceContext()
    print("=== Chroma DiT staged smoke ===")
    var model = ChromaDitCache.load_default_stage_smoke(ctx)
    var cache = model.precompute_step_cache[N_IMG, N_TXT, S](
        Float32(0.5), IMG_H2, IMG_W2, ctx
    )

    var pooled_shape = cache.pooled_temb.shape()
    var cos_shape = cache.rope_cos.shape()
    var sin_shape = cache.rope_sin.shape()
    _require_shape3(String("pooled_temb"), pooled_shape, 1, CHROMA_DIT_MOD_INDEX, 3072)
    _require_shape2(String("rope_cos"), cos_shape, S * 24, 64)
    _require_shape2(String("rope_sin"), sin_shape, S * 24, 64)

    var stats = chroma_step_cache_stats(cache, ctx)
    print(
        "  pooled_temb shape:",
        pooled_shape[0],
        pooled_shape[1],
        pooled_shape[2],
    )
    print("  rope shape:", cos_shape[0], cos_shape[1])
    print(
        "  pooled_temb stats mean/std/absmax:",
        stats[0],
        stats[1],
        stats[2],
    )

    var img_tokens = _tiny_tokens[64](N_IMG, Float32(0.01), ctx)
    var txt_tokens = _tiny_tokens[4096](N_TXT, Float32(0.001), ctx)
    var img = model.project_image_tokens(img_tokens, ctx)
    var txt = model.project_text_tokens(txt_tokens, ctx)
    var merged = model.two_double_blocks_smoke_forward[N_IMG, N_TXT, S](
        img, txt, cache, ctx
    )

    var merged_shape = merged.shape()
    _require_shape3(String("double01 merged"), merged_shape, 1, S, CHROMA_DIT_HIDDEN)
    var txt_out = slice(merged, 1, 0, N_TXT, ctx)
    var img_out = slice(merged, 1, N_TXT, N_IMG, ctx)
    _stats(String("double01.txt_out"), txt_out, ctx)
    _stats(String("double01.img_out"), img_out, ctx)

    var single0 = model.single_block_smoke_forward[S](0, merged, cache, ctx)
    var single0_shape = single0.shape()
    _require_shape3(String("single0 merged"), single0_shape, 1, S, CHROMA_DIT_HIDDEN)
    _stats(String("single0.merged"), single0, ctx)

    var single = model.single_block_smoke_forward[S](1, single0, cache, ctx)
    var single_shape = single.shape()
    _require_shape3(String("single1 merged"), single_shape, 1, S, CHROMA_DIT_HIDDEN)
    _stats(String("single1.merged"), single, ctx)

    var pred = model.final_image_projection_smoke[N_IMG, N_TXT, S](single, cache, ctx)
    var pred_shape = pred.shape()
    _require_shape3(String("final image projection"), pred_shape, 1, N_IMG, 64)
    _stats(String("final.img_patch64"), pred, ctx)
    print("Chroma DiT staged smoke PASS")
