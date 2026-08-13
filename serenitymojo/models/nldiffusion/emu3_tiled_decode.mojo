# emu3_tiled_decode.mojo — NL-Diffusion-Image Emu3 VQ decoder tiled 1024² decode.
#
# THE PROBLEM. A whole-frame 64-latent Emu3VQDecoder[64,64] decode (→1024²) OOMs
# on the 16 GB 5080: the ops that PRODUCE a 1024²×256 tensor — up.1.upsample.conv,
# every conv in up.0, and head.conv_out — each im2col ~9.4 GB.
#
# WHY NAIVE TILING (à la vae/zimage_tiled_decode) IS NOT ENOUGH. The Emu3 VQ
# decoder is NOT purely convolutional. It has TWO spatially-GLOBAL op families that
# run at the latent resolution and whose result depends on the whole grid:
#   * self-attention — 6 AttnBlocks (mid + 5 in up.4), full S×S attention;
#   * GroupNorm      — every ResnetBlock + head, mean/var over the whole spatial map.
# Decoding independent latent crops gives each tile a SMALLER attention/GN window
# than the full grid, so the interior drifts everywhere (not a seam artifact):
# measured cos≈0.9934 (whole-decode tiling) — fails the ≥0.999 gate.
#
# THE FIX — DEEP SPLIT at the 512²/1024² boundary:
#   PHASE A (global, ≤512² → fits): run the decoder ONCE over the full 64×64 grid
#     from post_quant_conv through up.1's five ResnetBlocks (STOP before
#     up.1.upsample) → feat NHWC [1, 8·LH, 8·LW, 256] (512²×256). ALL attention and
#     15 of 21 GroupNorms run over the full context → bit-identical to full decode.
#   PHASE B (tiled, the 1024²-producing tail): 3×3 overlapping crops of the 512²
#     feature (stride ½ tile) → up.1.upsample + up.0 + head → 512² image tiles,
#     feather-blended. Only up.0's 5 GroupNorms + head are tiled, so the residual
#     GN-window drift is tiny → cos ≥ 0.999.
#
# The feather-blend math (_weight_tensor / _xfade / _blend3) is byte-identical to
# vae/zimage_tiled_decode / vae/ideogram4_tiled_decode — VAE-agnostic, operates on
# decoded image tensors.
#
# DTYPE. The Emu3 VQ-VAE is fp32 (config torch_dtype float32); everything here is
# F32 — NO bf16 cast (unlike zimage, whose VAE is bf16).
#
# MEMORY. Callers should FREE the full-shaped decoder after PHASE A (only its
# low-res blocks are used) and enable the runtime's SYNCHRONOUS device allocator
# (MODULAR_DEVICE_CONTEXT_SYNC_MODE=true) before creating the DeviceContext — the
# default async allocator's pool grows to the cumulative transient peak and OOMs.
# See parity/tiled_decode_probe.mojo for the wiring.

from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.nldiffusion.emu3_vq_decoder import Emu3VQDecoder
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.tensor_algebra import add, mul, slice, concat
from serenitymojo.models.vae.decoder2d import (
    nchw_to_nhwc,
    nhwc_to_nchw,
    GN_GROUPS,
    GN_EPS,
)
from serenitymojo.models.vae.vae_ops import clone


comptime ZCH = 256      # z_channels / embed_dim
comptime C1024 = 1024   # block_in (mid, top up-level)
comptime C256 = 256
comptime OUT_CH = 3


# ── feathered cross-fade weight (ramps 1→0 or 0→1 along `dim`), F32 ──────────
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


# ── PHASE A: global forward post_quant_conv … up.1 RESNETS (stop before up1_up) ─
# z NCHW [1,256,LH,LW] -> feat NHWC [1, 8·LH, 8·LW, 256] (512²×256 for LH=64).
# All attention + up.4/up.3/up.2/up.1 resnets run over the FULL grid (≤512², fits).
# Mirrors Emu3VQDecoder.decode up to (not including) up.1.upsample. No bf16 cast.
# ctx.synchronize() after each block forces the async allocator to reclaim the
# prior block's working set (bounds peak to ~one block).
def emu3_decode_to_up1res[
    LH: Int, LW: Int
](dec: Emu3VQDecoder[LH, LW], z_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
    var h = nchw_to_nhwc(z_nchw, ctx)
    h = conv2d[1, LH, LW, ZCH, 1, 1, ZCH, 1, 1, 0, 0](
        h, clone(dec.pqc_w, ctx), Optional[Tensor](clone(dec.pqc_b, ctx)), ctx
    )
    h = conv2d[1, LH, LW, ZCH, 3, 3, C1024, 1, 1, 1, 1](
        h, clone(dec.conv_in_w, ctx), Optional[Tensor](clone(dec.conv_in_b, ctx)), ctx
    )
    h = dec.mid_res0.forward(h, ctx); ctx.synchronize()
    h = dec.mid_attn.forward(h, ctx); ctx.synchronize()
    h = dec.mid_res1.forward(h, ctx); ctx.synchronize()
    h = dec.up4_r0.forward(h, ctx); h = dec.up4_a0.forward(h, ctx); ctx.synchronize()
    h = dec.up4_r1.forward(h, ctx); h = dec.up4_a1.forward(h, ctx); ctx.synchronize()
    h = dec.up4_r2.forward(h, ctx); h = dec.up4_a2.forward(h, ctx); ctx.synchronize()
    h = dec.up4_r3.forward(h, ctx); h = dec.up4_a3.forward(h, ctx); ctx.synchronize()
    h = dec.up4_r4.forward(h, ctx); h = dec.up4_a4.forward(h, ctx); ctx.synchronize()
    h = dec.up4_up.forward(h, ctx); ctx.synchronize()   # -> 2·LH
    h = dec.up3_r0.forward(h, ctx); ctx.synchronize()
    h = dec.up3_r1.forward(h, ctx); ctx.synchronize()
    h = dec.up3_r2.forward(h, ctx); ctx.synchronize()
    h = dec.up3_r3.forward(h, ctx); ctx.synchronize()
    h = dec.up3_r4.forward(h, ctx); ctx.synchronize()
    h = dec.up3_up.forward(h, ctx); ctx.synchronize()   # -> 4·LH
    h = dec.up2_r0.forward(h, ctx); ctx.synchronize()
    h = dec.up2_r1.forward(h, ctx); ctx.synchronize()
    h = dec.up2_r2.forward(h, ctx); ctx.synchronize()
    h = dec.up2_r3.forward(h, ctx); ctx.synchronize()
    h = dec.up2_r4.forward(h, ctx); ctx.synchronize()
    h = dec.up2_up.forward(h, ctx); ctx.synchronize()   # -> 8·LH  (512² for LH=64)
    h = dec.up1_r0.forward(h, ctx); ctx.synchronize()
    h = dec.up1_r1.forward(h, ctx); ctx.synchronize()
    h = dec.up1_r2.forward(h, ctx); ctx.synchronize()
    h = dec.up1_r3.forward(h, ctx); ctx.synchronize()
    h = dec.up1_r4.forward(h, ctx); ctx.synchronize()   # STOP before up1_up
    return h^                         # [1, 8·LH, 8·LW, 256]


# ── PHASE B tail: up.1.upsample + up.0 + head on ONE 8·TH crop ────────────────
# feat_tile NHWC [1, 8·TH, 8·TW, 256] -> image NCHW [1, 3, 16·TH, 16·TW].
# The 1024²-producing tail; pure conv + up.0's 5 GN + head GN. No attention.
def _decode_tail_tile[
    TH: Int, TW: Int
](dec: Emu3VQDecoder[TH, TW], feat_tile: Tensor, ctx: DeviceContext) raises -> Tensor:
    var h = dec.up1_up.forward(feat_tile, ctx); ctx.synchronize()   # 8·TH -> 16·TH
    h = dec.up0_r0.forward(h, ctx); ctx.synchronize()
    h = dec.up0_r1.forward(h, ctx); ctx.synchronize()
    h = dec.up0_r2.forward(h, ctx); ctx.synchronize()
    h = dec.up0_r3.forward(h, ctx); ctx.synchronize()
    h = dec.up0_r4.forward(h, ctx); ctx.synchronize()
    h = group_norm(h, dec.norm_out_w, dec.norm_out_b, GN_GROUPS, GN_EPS, ctx); ctx.synchronize()
    h = silu(h, ctx)
    h = conv2d[1, 16 * TH, 16 * TW, C256, 3, 3, OUT_CH, 1, 1, 1, 1](
        h, clone(dec.conv_out_w, ctx), Optional[Tensor](clone(dec.conv_out_b, ctx)), ctx
    )
    return nhwc_to_nchw(h, ctx)  # [1, 3, 16·TH, 16·TW]


# ── PHASE B tiling: 3×3 overlapping crops of the 512² feature → 1024² image ──
# feat NHWC [1, 8·LATENT_H, 8·LATENT_W, 256] -> image NCHW [1,3,16·LATENT_H,16·LATENT_W].
# TILE = LATENT/2. Crop size 8·TILE, stride 4·TILE (half-tile overlap); the tail's
# up.1.upsample doubles feature→image, so a feature crop of 8·TILE → a 16·TILE=512²
# image tile. Image tiles sit at offsets 0 / 8·STRIDE / 16·STRIDE → _blend3.
# ctx.synchronize() between tiles bounds the async allocator to ~one tile's set.
def emu3_tiled_tail_3x3[
    LATENT_H: Int, LATENT_W: Int, TILE_H: Int, TILE_W: Int
](
    feat: Tensor, dec_tile: Emu3VQDecoder[TILE_H, TILE_W], ctx: DeviceContext
) raises -> Tensor:
    comptime assert TILE_H == LATENT_H // 2, "tile height must be half latent height"
    comptime assert TILE_W == LATENT_W // 2, "tile width must be half latent width"
    comptime CROP = 8 * TILE_H     # feature crop (256): 3 crops at stride 4·TILE cover 8·LATENT
    comptime STRIDE = 4 * TILE_H   # feature stride (128)
    # row 0 crop H[0:CROP].
    var r = slice(feat, 1, 0, CROP, ctx)
    var a = _decode_tail_tile[TILE_H, TILE_W](dec_tile, slice(r, 2, 0, CROP, ctx), ctx); ctx.synchronize()
    var b = _decode_tail_tile[TILE_H, TILE_W](dec_tile, slice(r, 2, STRIDE, CROP, ctx), ctx); ctx.synchronize()
    var c = _decode_tail_tile[TILE_H, TILE_W](dec_tile, slice(r, 2, CROP, CROP, ctx), ctx); ctx.synchronize()
    var row0 = _blend3(a, b, c, 3, ctx); ctx.synchronize()
    # row 1 crop H[STRIDE:STRIDE+CROP] (reassign a/b/c → prior tiles freed).
    r = slice(feat, 1, STRIDE, CROP, ctx)
    a = _decode_tail_tile[TILE_H, TILE_W](dec_tile, slice(r, 2, 0, CROP, ctx), ctx); ctx.synchronize()
    b = _decode_tail_tile[TILE_H, TILE_W](dec_tile, slice(r, 2, STRIDE, CROP, ctx), ctx); ctx.synchronize()
    c = _decode_tail_tile[TILE_H, TILE_W](dec_tile, slice(r, 2, CROP, CROP, ctx), ctx); ctx.synchronize()
    var row1 = _blend3(a, b, c, 3, ctx); ctx.synchronize()
    # row 2 crop H[CROP:2·CROP].
    r = slice(feat, 1, CROP, CROP, ctx)
    a = _decode_tail_tile[TILE_H, TILE_W](dec_tile, slice(r, 2, 0, CROP, ctx), ctx); ctx.synchronize()
    b = _decode_tail_tile[TILE_H, TILE_W](dec_tile, slice(r, 2, STRIDE, CROP, ctx), ctx); ctx.synchronize()
    c = _decode_tail_tile[TILE_H, TILE_W](dec_tile, slice(r, 2, CROP, CROP, ctx), ctx); ctx.synchronize()
    var row2 = _blend3(a, b, c, 3, ctx); ctx.synchronize()
    return _blend3(row0, row1, row2, 2, ctx)
