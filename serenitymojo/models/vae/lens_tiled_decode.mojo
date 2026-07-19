# lens_tiled_decode.mojo — MS-Lens FLUX.2 (Klein) VAE tiled decode
# (3x3 overlap + feathered blend).
#
# WHY (MEMORY RISK D, lens_backend.mojo): the whole-frame 1024² decode goes
# through KleinVaeDecoder[64,64] (the Flux2 VAE, PACKED 128-ch latent → 16×
# upsample). A single monolithic 1024² Flux2 decode buffer OOMs a 24 GB card at
# the post-DiT high-water mark (same failure class the orchestrator note flagged
# for the SDXL / zimage monolithic decodes). lens_backend already does strict
# phase separation + cu_mempool_trim_current(0) before the decode, but the
# decode itself is a single big alloc; under a high pool water mark it still
# risks OOM. This is the tiled mitigation the lens_backend docstring asked for
# ("a tiled Flux2 decoder is required (none exists yet for the Flux2 VAE …)").
#
# This is the FLUX.2 / Klein PACKED-latent analogue of pipeline/flux_tiled_decode
# (which serves the FLUX.1 LdmVaeDecoder, 16-ch UNPACKED) — Lens uses the
# DISTINCT KleinVaeDecoder, so it needs its own tiled wrapper. Each crop decodes
# a TILE = LATENT/2 quadrant (a small alloc that fits the pool). Crops sit at
# latent stride TILE/2 (positions 0/half/TILE) so adjacent image tiles overlap;
# a separable feathered cross-fade (horizontal per row, then vertical) erases the
# conv-boundary seams. Tile vars are reassigned per row so prior tiles free →
# retained-memory peak stays near the single-tile working set.
#
# Blend math (_weight_tensor/_xfade/_blend3) is byte-identical to
# sdxl_tiled_decode / zimage_tiled_decode / flux_tiled_decode — it is
# VAE-agnostic (operates on the DECODED image tensors at spatial dims 2/3, and is
# upscale-factor-agnostic, so the Klein 16× upsample needs no change vs the FLUX.1
# 8× / Klein-Lens 16×). The ONLY thing that differs from flux_tiled_decode is the
# decoder instance (KleinVaeDecoder vs LdmVaeDecoder) and that the latent is
# PACKED 128-ch NCHW (channel dim 1; crops along spatial dims 2/3).
#
# Input is the SAME packed-NCHW latent KleinVaeDecoder.decode consumes:
# [1, PACKED_CH=128, LATENT_H, LATENT_W] (what lens vae_decode produces after its
# reshape+permute). Output: [1, 3, 16*LATENT_H, 16*LATENT_W].

from std.collections import List
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.klein_decoder import KleinVaeDecoder
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, mul, slice, concat


# ── feathered cross-fade weight (ramps 1→0 or 0→1 along `dim`), F32 ──────────
# Shape [1,1,n,1] for a height(dim 2) blend, [1,1,1,n] for a width(dim 3) blend,
# so it broadcasts over channels and the non-blend spatial axis.
def _weight_tensor(n: Int, dim: Int, ascending: Bool, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for i in range(n):
        var t = (Float32(i) + 0.5) / Float32(n)
        h.append(t if ascending else (1.0 - t))
    var sh = List[Int]()
    sh.append(1)
    sh.append(1)
    if dim == 2:
        sh.append(n)
        sh.append(1)
    else:
        sh.append(1)
        sh.append(n)
    return Tensor.from_host(h^, sh^, STDtype.F32, ctx)


# ── cross-fade two equal-shaped overlap slabs along `dim` (left fades out) ────
def _xfade(left: Tensor, right: Tensor, dim: Int, ctx: DeviceContext) raises -> Tensor:
    var n = left.shape()[dim]
    var wl = _weight_tensor(n, dim, False, ctx)
    var wr = _weight_tensor(n, dim, True, ctx)
    return add(mul(left, wl, ctx), mul(right, wr, ctx), ctx)


# ── blend 3 equal tiles (size T along `dim`) placed at offsets 0, T/2, T ──────
# Output size 2T: [pure t0 | xfade(t0,t1) | xfade(t1,t2) | pure t2].
def _blend3(t0: Tensor, t1: Tensor, t2: Tensor, dim: Int, ctx: DeviceContext) raises -> Tensor:
    var t = t0.shape()[dim]
    var s = t // 2
    var ov = t - s
    var a = slice(t0, dim, 0, s, ctx)
    var b = _xfade(slice(t0, dim, s, ov, ctx), slice(t1, dim, 0, ov, ctx), dim, ctx)
    var c = _xfade(slice(t1, dim, ov, ov, ctx), slice(t2, dim, 0, ov, ctx), dim, ctx)
    var d = slice(t2, dim, ov, s, ctx)
    return concat(dim, ctx, a, b, c, d)


# ── tiled Klein/FLUX.2 VAE decode — 3x3 OVERLAPPING packed-latent crops + blend ──
# PACKED latent NCHW [1,128,LATENT_H,LATENT_W] -> image NCHW
# [1,3,16*LATENT_H,16*LATENT_W]. Decoder supplied at the TILE shape (LATENT/2),
# reused for all 9 crops at latent stride TILE/2. Spatial crops are along dims 2/3
# (the 128 packed channels in dim 1 are taken whole per crop).
def lens_tiled_decode_with_decoder[
    LATENT_H: Int, LATENT_W: Int, TILE_H: Int, TILE_W: Int
](
    packed_nchw: Tensor, dec: KleinVaeDecoder[TILE_H, TILE_W], ctx: DeviceContext
) raises -> Tensor:
    comptime assert TILE_H == LATENT_H // 2, "tile height must be half latent height"
    comptime assert TILE_W == LATENT_W // 2, "tile width must be half latent width"
    var half = TILE_H // 2                          # latent stride
    # row 0 crop [0:TILE_H] (spatial dim 2), blend its 3 columns (spatial dim 3)
    var r = slice(packed_nchw, 2, 0, TILE_H, ctx)
    var a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var b = cast_tensor(dec.decode(slice(r, 3, half, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var c = cast_tensor(dec.decode(slice(r, 3, TILE_W, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row0 = _blend3(a, b, c, 3, ctx)
    # row 1 crop [half:half+TILE_H] (reassign a/b/c → prior tiles freed)
    r = slice(packed_nchw, 2, half, TILE_H, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, half, TILE_W, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, TILE_W, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row1 = _blend3(a, b, c, 3, ctx)
    # row 2 crop [TILE_H:2*TILE_H]
    r = slice(packed_nchw, 2, TILE_H, TILE_H, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, half, TILE_W, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, TILE_W, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row2 = _blend3(a, b, c, 3, ctx)
    # vertical feathered blend of the 3 full-width rows
    return _blend3(row0, row1, row2, 2, ctx)


# ── load Klein/FLUX.2 VAE at the TILE shape + tiled-decode a packed latent ────
# packed_nchw is [1,128,LATENT_H,LATENT_W]; vae_path is the flux2-vae safetensors.
def lens_tiled_decode[
    LATENT_H: Int, LATENT_W: Int
](packed_nchw: Tensor, vae_path: String, ctx: DeviceContext) raises -> Tensor:
    comptime TILE_H = LATENT_H // 2
    comptime TILE_W = LATENT_W // 2
    var dec = KleinVaeDecoder[TILE_H, TILE_W].load(vae_path, ctx)
    return lens_tiled_decode_with_decoder[
        LATENT_H, LATENT_W, TILE_H, TILE_W
    ](packed_nchw, dec, ctx)
