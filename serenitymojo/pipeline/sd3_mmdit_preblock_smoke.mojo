# SD3.5 Large/Medium MMDiT pre/post-block real-weight smoke.
#
# Gates the resident MMDiT math around the missing joint transformer blocks:
# latent patch embedding + cropped pos_embed, timestep+pooled conditioning, and
# context projection for both SD3.5 Large and the local "small" SD3.5 Medium.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.sd3_contract import (
    SD3_LARGE_CONTEXT_DIM,
    SD3_LARGE_HIDDEN,
    SD3_LARGE_IMAGE_TOKENS,
    SD3_LARGE_LATENT_CHANNELS,
    SD3_LARGE_LATENT_H,
    SD3_LARGE_LATENT_W,
    SD3_MEDIUM_CONTEXT_DIM,
    SD3_MEDIUM_HIDDEN,
    SD3_MEDIUM_IMAGE_TOKENS,
    SD3_MEDIUM_LATENT_CHANNELS,
    SD3_MEDIUM_LATENT_H,
    SD3_MEDIUM_LATENT_W,
)
from serenitymojo.models.dit.sd3_mmdit import SD3MMDiTPreBlockGate
from serenitymojo.tensor import Tensor


comptime CTX_TOKENS = 8


def _shape2(a: Int, b: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    return sh^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    sh.append(c)
    return sh^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    sh.append(c)
    sh.append(d)
    return sh^


def _make_values(n: Int, scale: Float32) -> List[Float32]:
    var vals = List[Float32](capacity=n)
    for i in range(n):
        vals.append((Float32(i % 257) - 128.0) * scale)
    return vals^


def _require_shape2(label: String, got: List[Int], a: Int, b: Int) raises:
    if len(got) != 2 or got[0] != a or got[1] != b:
        raise Error(String("SD3 shape mismatch for ") + label)


def _require_shape3(label: String, got: List[Int], a: Int, b: Int, c: Int) raises:
    if len(got) != 3 or got[0] != a or got[1] != b or got[2] != c:
        raise Error(String("SD3 shape mismatch for ") + label)


def _stats_nonzero(label: String, t: Tensor, ctx: DeviceContext) raises -> Float32:
    var h = t.to_host(ctx)
    if len(h) == 0:
        raise Error(String("SD3 empty tensor: ") + label)
    var max_abs = Float32(0.0)
    for i in range(len(h)):
        var v = h[i]
        if v < 0.0:
            v = -v
        if v > max_abs:
            max_abs = v
    if max_abs == 0.0:
        raise Error(String("SD3 all-zero tensor: ") + label)
    return max_abs


def _run_large(ctx: DeviceContext) raises:
    print("=== SD3.5 Large MMDiT pre/post-block real-weight smoke ===")
    var gate = SD3MMDiTPreBlockGate.load_large_default(ctx)
    var latents = Tensor.from_host(
        _make_values(
            1
            * SD3_LARGE_LATENT_CHANNELS
            * SD3_LARGE_LATENT_H
            * SD3_LARGE_LATENT_W,
            0.002,
        ),
        _shape4(
            1,
            SD3_LARGE_LATENT_CHANNELS,
            SD3_LARGE_LATENT_H,
            SD3_LARGE_LATENT_W,
        ),
        STDtype.BF16,
        ctx,
    )
    var x = gate.latent_patch_embed[SD3_LARGE_LATENT_H, SD3_LARGE_LATENT_W](
        latents, ctx
    )
    _require_shape3(
        String("large latent_patch_embed"),
        x.shape(),
        1,
        SD3_LARGE_IMAGE_TOKENS,
        SD3_LARGE_HIDDEN,
    )

    var pooled = Tensor.from_host(
        _make_values(1 * 2048, 0.001),
        _shape2(1, 2048),
        STDtype.BF16,
        ctx,
    )
    var c = gate.conditioning(Float32(1.0), pooled, ctx)
    _require_shape2(String("large conditioning"), c.shape(), 1, SD3_LARGE_HIDDEN)

    var enc = Tensor.from_host(
        _make_values(1 * CTX_TOKENS * SD3_LARGE_CONTEXT_DIM, 0.0005),
        _shape3(1, CTX_TOKENS, SD3_LARGE_CONTEXT_DIM),
        STDtype.BF16,
        ctx,
    )
    var ctx_emb = gate.context_embed[CTX_TOKENS](enc, ctx)
    _require_shape3(
        String("large context_embed"),
        ctx_emb.shape(),
        1,
        CTX_TOKENS,
        SD3_LARGE_HIDDEN,
    )
    var patch_out = gate.final_layer_tokens(x, c, ctx)
    _require_shape3(
        String("large final_layer_tokens"),
        patch_out.shape(),
        1,
        SD3_LARGE_IMAGE_TOKENS,
        SD3_LARGE_LATENT_CHANNELS * 2 * 2,
    )
    var latent_out = gate.final_unpatchify[SD3_LARGE_LATENT_H, SD3_LARGE_LATENT_W](
        patch_out, ctx
    )
    var latent_shape = latent_out.shape()
    if (
        len(latent_shape) != 4
        or latent_shape[0] != 1
        or latent_shape[1] != SD3_LARGE_LATENT_CHANNELS
        or latent_shape[2] != SD3_LARGE_LATENT_H
        or latent_shape[3] != SD3_LARGE_LATENT_W
    ):
        raise Error("SD3 Large final unpatchify shape mismatch")
    print(
        "  large x/c/context/final/latent max_abs:",
        _stats_nonzero(String("large x"), x, ctx),
        _stats_nonzero(String("large conditioning"), c, ctx),
        _stats_nonzero(String("large context"), ctx_emb, ctx),
        _stats_nonzero(String("large final patches"), patch_out, ctx),
        _stats_nonzero(String("large final latent"), latent_out, ctx),
    )


def _run_medium(ctx: DeviceContext) raises:
    print("=== SD3.5 Medium MMDiT pre/post-block real-weight smoke ===")
    var gate = SD3MMDiTPreBlockGate.load_medium_default(ctx)
    var latents = Tensor.from_host(
        _make_values(
            1
            * SD3_MEDIUM_LATENT_CHANNELS
            * SD3_MEDIUM_LATENT_H
            * SD3_MEDIUM_LATENT_W,
            0.002,
        ),
        _shape4(
            1,
            SD3_MEDIUM_LATENT_CHANNELS,
            SD3_MEDIUM_LATENT_H,
            SD3_MEDIUM_LATENT_W,
        ),
        STDtype.BF16,
        ctx,
    )
    var x = gate.latent_patch_embed[SD3_MEDIUM_LATENT_H, SD3_MEDIUM_LATENT_W](
        latents, ctx
    )
    _require_shape3(
        String("medium latent_patch_embed"),
        x.shape(),
        1,
        SD3_MEDIUM_IMAGE_TOKENS,
        SD3_MEDIUM_HIDDEN,
    )

    var pooled = Tensor.from_host(
        _make_values(1 * 2048, 0.001),
        _shape2(1, 2048),
        STDtype.BF16,
        ctx,
    )
    var c = gate.conditioning(Float32(1.0), pooled, ctx)
    _require_shape2(String("medium conditioning"), c.shape(), 1, SD3_MEDIUM_HIDDEN)

    var enc = Tensor.from_host(
        _make_values(1 * CTX_TOKENS * SD3_MEDIUM_CONTEXT_DIM, 0.0005),
        _shape3(1, CTX_TOKENS, SD3_MEDIUM_CONTEXT_DIM),
        STDtype.BF16,
        ctx,
    )
    var ctx_emb = gate.context_embed[CTX_TOKENS](enc, ctx)
    _require_shape3(
        String("medium context_embed"),
        ctx_emb.shape(),
        1,
        CTX_TOKENS,
        SD3_MEDIUM_HIDDEN,
    )
    var patch_out = gate.final_layer_tokens(x, c, ctx)
    _require_shape3(
        String("medium final_layer_tokens"),
        patch_out.shape(),
        1,
        SD3_MEDIUM_IMAGE_TOKENS,
        SD3_MEDIUM_LATENT_CHANNELS * 2 * 2,
    )
    var latent_out = gate.final_unpatchify[
        SD3_MEDIUM_LATENT_H, SD3_MEDIUM_LATENT_W
    ](patch_out, ctx)
    var latent_shape = latent_out.shape()
    if (
        len(latent_shape) != 4
        or latent_shape[0] != 1
        or latent_shape[1] != SD3_MEDIUM_LATENT_CHANNELS
        or latent_shape[2] != SD3_MEDIUM_LATENT_H
        or latent_shape[3] != SD3_MEDIUM_LATENT_W
    ):
        raise Error("SD3 Medium final unpatchify shape mismatch")
    print(
        "  medium x/c/context/final/latent max_abs:",
        _stats_nonzero(String("medium x"), x, ctx),
        _stats_nonzero(String("medium conditioning"), c, ctx),
        _stats_nonzero(String("medium context"), ctx_emb, ctx),
        _stats_nonzero(String("medium final patches"), patch_out, ctx),
        _stats_nonzero(String("medium final latent"), latent_out, ctx),
    )


def main() raises:
    var ctx = DeviceContext()
    _run_large(ctx)
    _run_medium(ctx)
    print("SD3.5 Large/Medium MMDiT pre/post-block real-weight smoke PASS")
