# LTX-2 video VAE tiled decode for the 15 GB Desktop contract.
#
# Oracle: the locally installed LTX Desktop v1.0.5 / ltx_core VideoDecoder.
# Its low-VRAM defaults are:
#   spatial 512 px tiles, 64 px overlap
#   temporal 64 frame tiles, 24 frame overlap
# For the pinned Creator HQ 121-frame latent [1,128,16,H,W], that is:
#   temporal: [0:8], [4:13], [9:16]
# Each temporal group is decoded as the measured spatial tile grid, blended to
# one [1,3,F,H*32,W*32] chunk, then merged with the previous group. Callers emit
# finalized chunks [32,40,49] instead of retaining the full 121-frame tensor.

from std.collections import List
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.ltx2_vae_decoder import (
    LTX2VaeDecoderWeights,
    decode as decode_video,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, mul, slice, concat


comptime LTX2_DESKTOP_TEMPORAL_GROUPS = 3
comptime LTX2_DESKTOP_FIRST_OUT_CHUNK = 32
comptime LTX2_DESKTOP_MIDDLE_OUT_CHUNK = 40
comptime LTX2_DESKTOP_TEMPORAL_OVERLAP = 25
comptime LTX2_DESKTOP_SPATIAL_OVERLAP = 64


def _weight_5d(
    var left: List[Float32], var right: List[Float32], dim: Int,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var shl = List[Int]()
    var shr = List[Int]()
    for _ in range(5):
        shl.append(1)
        shr.append(1)
    shl[dim] = len(left)
    shr[dim] = len(right)
    return (
        Tensor.from_host(left^, shl^, STDtype.F32, ctx),
        Tensor.from_host(right^, shr^, STDtype.F32, ctx),
    )


def _creator_spatial_xfade(
    left: Tensor, right: Tensor, dim: Int, ctx: DeviceContext
) raises -> Tensor:
    """Creator trapezoid overlap; BF16 boundary, F32 blend interior."""
    var n = left.shape()[dim]
    var lh = List[Float32]()
    var rh = List[Float32]()
    for i in range(n):
        # Creator compute_trapezoidal_mask_1d(..., left_starts_from_0=False):
        # complementary interior samples of linspace(0, 1, n + 2).
        var wr = Float32(i + 1) / Float32(n + 1)
        lh.append(Float32(1.0) - wr)
        rh.append(wr)
    var weights = _weight_5d(lh^, rh^, dim, ctx)
    var lf = cast_tensor(left, STDtype.F32, ctx)
    var rf = cast_tensor(right, STDtype.F32, ctx)
    return cast_tensor(
        add(mul(lf, weights[0], ctx), mul(rf, weights[1], ctx), ctx),
        STDtype.BF16,
        ctx,
    )


def _creator_spatial_join(
    left: Tensor, right: Tensor, dim: Int, overlap: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var nl = left.shape()[dim]
    var nr = right.shape()[dim]
    if overlap <= 0 or overlap >= nl or overlap >= nr:
        raise Error("ltx2 tiled decode: invalid spatial overlap")
    var a = slice(left, dim, 0, nl - overlap, ctx)
    var b = _creator_spatial_xfade(
        slice(left, dim, nl - overlap, overlap, ctx),
        slice(right, dim, 0, overlap, ctx),
        dim,
        ctx,
    )
    var c = slice(right, dim, overlap, nr - overlap, ctx)
    return concat(dim, ctx, a, b, c)


def _creator_temporal_overlap(
    left: Tensor, right: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Desktop temporal accumulation+normalization for one 25-frame overlap."""
    var n = left.shape()[2]
    if n != LTX2_DESKTOP_TEMPORAL_OVERLAP or right.shape()[2] != n:
        raise Error("ltx2 tiled decode: temporal overlap shape mismatch")
    var lh = List[Float32]()
    var rh = List[Float32]()
    for i in range(n):
        # Desktop uses split_temporal_causal: the prior tile's 24-frame
        # right ramp plus the next tile's 25-frame zero-based left ramp form
        # complementary weights across their 25-frame output intersection.
        var wr = Float32(i) / Float32(LTX2_DESKTOP_TEMPORAL_OVERLAP)
        lh.append(Float32(1.0) - wr)
        rh.append(wr)
    var weights = _weight_5d(lh^, rh^, 2, ctx)
    var lf = cast_tensor(left, STDtype.F32, ctx)
    var rf = cast_tensor(right, STDtype.F32, ctx)
    return cast_tensor(
        add(mul(lf, weights[0], ctx), mul(rf, weights[1], ctx), ctx),
        STDtype.BF16,
        ctx,
    )


def ltx2_desktop_temporal_carry(
    previous: Tensor, current: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Return current with its leading overlap merged with previous's tail."""
    var np = previous.shape()[2]
    var nc = current.shape()[2]
    var ov = LTX2_DESKTOP_TEMPORAL_OVERLAP
    if np < ov or nc < ov:
        raise Error("ltx2 tiled decode: temporal chunk shorter than overlap")
    var merged = _creator_temporal_overlap(
        slice(previous, 2, np - ov, ov, ctx),
        slice(current, 2, 0, ov, ctx),
        ctx,
    )
    var suffix = slice(current, 2, ov, nc - ov, ctx)
    return concat(2, ctx, merged, suffix)


def _latent_tile[
    F_CT: Int, H_CT: Int, W_CT: Int
](
    latent: Tensor, f0: Int, h0: Int, w0: Int, ctx: DeviceContext
) raises -> Tensor:
    var tile = slice(latent, 2, f0, F_CT, ctx)
    tile = slice(tile, 3, h0, H_CT, ctx)
    return slice(tile, 4, w0, W_CT, ctx)


def _decode_tile[
    F_CT: Int, H_CT: Int, W_CT: Int
](
    weights: LTX2VaeDecoderWeights,
    latent: Tensor,
    f0: Int,
    h0: Int,
    w0: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var tile = _latent_tile[F_CT, H_CT, W_CT](
        latent, f0, h0, w0, ctx
    )
    return decode_video[1, 128, F_CT, H_CT, W_CT](weights, tile, ctx)


def ltx2_desktop_decode_temporal_group[
    F_CT: Int
](
    weights: LTX2VaeDecoderWeights,
    latent: Tensor,
    f0: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Decode one Desktop temporal group as the exact 3x2 spatial tile grid."""
    var a = _decode_tile[F_CT, 9, 16](weights, latent, f0, 0, 0, ctx)
    var b = _decode_tile[F_CT, 9, 16](weights, latent, f0, 0, 14, ctx)
    var row0 = _creator_spatial_join(
        a, b, 4, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )

    a = _decode_tile[F_CT, 9, 16](weights, latent, f0, 7, 0, ctx)
    b = _decode_tile[F_CT, 9, 16](weights, latent, f0, 7, 14, ctx)
    var row1 = _creator_spatial_join(
        a, b, 4, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )

    a = _decode_tile[F_CT, 3, 16](weights, latent, f0, 14, 0, ctx)
    b = _decode_tile[F_CT, 3, 16](weights, latent, f0, 14, 14, ctx)
    var row2 = _creator_spatial_join(
        a, b, 4, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )

    var upper = _creator_spatial_join(
        row0, row1, 3, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )
    return _creator_spatial_join(
        upper, row2, 3, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )


def _decode_portrait_column[
    F_CT: Int, W_CT: Int
](
    weights: LTX2VaeDecoderWeights,
    latent: Tensor,
    f0: Int,
    w0: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Decode one 960px column using the transposed Desktop H=30 split."""
    var column = _decode_tile[F_CT, 16, W_CT](
        weights, latent, f0, 0, w0, ctx
    )
    var tile = _decode_tile[F_CT, 16, W_CT](
        weights, latent, f0, 14, w0, ctx
    )
    return _creator_spatial_join(
        column, tile, 3, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )


def ltx2_desktop_decode_temporal_group_portrait[
    F_CT: Int
](
    weights: LTX2VaeDecoderWeights,
    latent: Tensor,
    f0: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Decode one 544x960 group with the transposed Desktop 2x3 grid."""
    var full = _decode_portrait_column[F_CT, 9](
        weights, latent, f0, 0, ctx
    )
    var column = _decode_portrait_column[F_CT, 9](
        weights, latent, f0, 7, ctx
    )
    full = _creator_spatial_join(
        full, column, 4, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )
    column = _decode_portrait_column[F_CT, 3](
        weights, latent, f0, 14, ctx
    )
    return _creator_spatial_join(
        full, column, 4, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )


def _decode_full_width_row[
    F_CT: Int, H_CT: Int
](
    weights: LTX2VaeDecoderWeights,
    latent: Tensor,
    f0: Int,
    h0: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Decode one 1920px row using Creator/Desktop's measured W=60 split.

    split_by_size(size=16, overlap=2) yields latent intervals
    [0:16], [14:30], [28:44], [42:58], [56:60].
    """
    var row = _decode_tile[F_CT, H_CT, 16](
        weights, latent, f0, h0, 0, ctx
    )
    var tile = _decode_tile[F_CT, H_CT, 16](
        weights, latent, f0, h0, 14, ctx
    )
    row = _creator_spatial_join(
        row, tile, 4, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )
    tile = _decode_tile[F_CT, H_CT, 16](
        weights, latent, f0, h0, 28, ctx
    )
    row = _creator_spatial_join(
        row, tile, 4, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )
    tile = _decode_tile[F_CT, H_CT, 16](
        weights, latent, f0, h0, 42, ctx
    )
    row = _creator_spatial_join(
        row, tile, 4, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )
    tile = _decode_tile[F_CT, H_CT, 4](
        weights, latent, f0, h0, 56, ctx
    )
    return _creator_spatial_join(
        row, tile, 4, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )


def ltx2_desktop_decode_temporal_group_full[
    F_CT: Int
](
    weights: LTX2VaeDecoderWeights,
    latent: Tensor,
    f0: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Decode one 1920x1088 temporal group with the exact Desktop grid.

    For latent H=34,W=60 and Desktop 512/64px tiling, Creator computes a
    proportional height tile size round(16*34/60)=9. Its measured H intervals
    are [0:9], [7:16], [14:23], [21:30], [28:34]; width uses the five
    intervals documented by `_decode_full_width_row`.
    """
    var full = _decode_full_width_row[F_CT, 9](
        weights, latent, f0, 0, ctx
    )
    var row = _decode_full_width_row[F_CT, 9](
        weights, latent, f0, 7, ctx
    )
    full = _creator_spatial_join(
        full, row, 3, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )
    row = _decode_full_width_row[F_CT, 9](
        weights, latent, f0, 14, ctx
    )
    full = _creator_spatial_join(
        full, row, 3, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )
    row = _decode_full_width_row[F_CT, 9](
        weights, latent, f0, 21, ctx
    )
    full = _creator_spatial_join(
        full, row, 3, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )
    row = _decode_full_width_row[F_CT, 6](
        weights, latent, f0, 28, ctx
    )
    return _creator_spatial_join(
        full, row, 3, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )


def _decode_full_height_column[
    F_CT: Int, W_CT: Int
](
    weights: LTX2VaeDecoderWeights,
    latent: Tensor,
    f0: Int,
    w0: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Decode one 1920px column using Creator/Desktop's measured H=60 split."""
    var column = _decode_tile[F_CT, 16, W_CT](
        weights, latent, f0, 0, w0, ctx
    )
    var tile = _decode_tile[F_CT, 16, W_CT](
        weights, latent, f0, 14, w0, ctx
    )
    column = _creator_spatial_join(
        column, tile, 3, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )
    tile = _decode_tile[F_CT, 16, W_CT](
        weights, latent, f0, 28, w0, ctx
    )
    column = _creator_spatial_join(
        column, tile, 3, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )
    tile = _decode_tile[F_CT, 16, W_CT](
        weights, latent, f0, 42, w0, ctx
    )
    column = _creator_spatial_join(
        column, tile, 3, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )
    tile = _decode_tile[F_CT, 4, W_CT](
        weights, latent, f0, 56, w0, ctx
    )
    return _creator_spatial_join(
        column, tile, 3, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )


def ltx2_desktop_decode_temporal_group_full_portrait[
    F_CT: Int
](
    weights: LTX2VaeDecoderWeights,
    latent: Tensor,
    f0: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Decode one 1088x1920 group with the transposed Desktop tile grid."""
    var full = _decode_full_height_column[F_CT, 9](
        weights, latent, f0, 0, ctx
    )
    var column = _decode_full_height_column[F_CT, 9](
        weights, latent, f0, 7, ctx
    )
    full = _creator_spatial_join(
        full, column, 4, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )
    column = _decode_full_height_column[F_CT, 9](
        weights, latent, f0, 14, ctx
    )
    full = _creator_spatial_join(
        full, column, 4, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )
    column = _decode_full_height_column[F_CT, 9](
        weights, latent, f0, 21, ctx
    )
    full = _creator_spatial_join(
        full, column, 4, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )
    column = _decode_full_height_column[F_CT, 6](
        weights, latent, f0, 28, ctx
    )
    return _creator_spatial_join(
        full, column, 4, LTX2_DESKTOP_SPATIAL_OVERLAP, ctx
    )
