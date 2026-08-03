# serenitymojo/pipeline/minimax_h3_video_vae_spatial_tiling.mojo
#
# MiniMax-H3 video VAE — SPATIAL TILING layer. Proven required (not
# large-canvas insurance) in the 2026-08-02 audit: `video_vae/config.json`
# sets `vae_encoder_tiling: 1` / `vae_decoder_tiling: 1` BY DEFAULT with
# `vae_tile_size: 256` PIXELS, and `split_tiles` only skips tiling when
# `tile_size >= input_len` — at a 1344x768 target, both axes exceed 256, so
# every temporal clip's encode/decode routes through this layer.
#
# Authority: klvae.py `AutoencoderKL.split_tiles`/`tiled_encode`/
# `tiled_decode` (:192-429). Read in full. Reuses `minimax_h3_video_blend`
# (pipeline/minimax_h3_video_vae_blend.mojo) — the SAME function the
# temporal layer uses — rather than a second blend implementation, exactly
# as the reference itself reuses ONE `blend()` for both spatial and
# temporal seams.
#
# COMPOSITION WITH THE TEMPORAL LAYER (tiles-within-chunks): this module's
# `tiled_encode`/`tiled_decode` are DROP-IN replacements for
# `minimax_h3_video_encode_device`/`minimax_h3_video_decode_device` — same
# input/output tensor contract, spatial tiling fully resolved inside one
# call (every spatial seam is blended and trimmed before the function
# returns). The temporal layer's blend only ever touches the TEMPORAL axis
# of a COMPLETE (already spatially-reconstructed) per-clip volume — it never
# sees a partial tile. The two blends cannot double-apply: they operate on
# disjoint axes at disjoint times (spatial resolved first, per clip;
# temporal resolved second, across clips). Swapping
# `pipeline/minimax_h3_video_vae_temporal.mojo`'s two per-clip call sites
# (`minimax_h3_video_encode_device(encoder, clip_x, ctx)` /
# `minimax_h3_video_decode_device[...](decoder, clip_z, ctx)`) to call this
# module's `tiled_encode`/`tiled_decode` instead is a one-line change at
# each site — NOT done here, to keep this unit independently gated first
# (same discipline the temporal layer itself was built and gated with).
#
# SCOPE / ASSUMPTION (stated, not silently assumed): every tile is EXACT
# `tile_size` pixels except the fit-in-one-tile case (`tile_size >=
# input_len`). `split_tiles`'s own overlap-redistribution loop is DESIGNED
# to make every tile land exactly on `tile_size` (that is the point of
# `remaining_units`) for any `input_len` that is itself a multiple of
# `vae_ratio` — true for every real generation resolution (video/image
# canvases are always sized in VAE-compatible units). The decoder path
# ADDITIONALLY requires the resulting LATENT tile size to be uniform across
# every tile (checked, not assumed) so `ops/attention.sdpa_nomask`'s
# compile-time S can be ONE value for every tile, the same simplification
# the temporal layer uses for chunks. A non-`vae_ratio`-multiple input
# fails loud here rather than silently misrendering a seam.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import concat, slice
from serenitymojo.pipeline.minimax_h3_video_vae_blend import minimax_h3_video_blend
from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice, minimax_h3_video_encode_device,
)
from serenitymojo.models.vae.minimax_h3_video_decoder_device import (
    MiniMaxH3VideoDecoderDevice, minimax_h3_video_decode_device,
)

comptime TArc = ArcPointer[Tensor]


@fieldwise_init
struct MiniMaxH3TilingConfig(Copyable, Movable):
    var tile_size: Int
    var tile_overlap_min: Int
    var decoder_tile_size: Int
    var decoder_tile_overlap_min: Int
    var vae_ratio: Int


def minimax_h3_video_released_tiling_config() -> MiniMaxH3TilingConfig:
    """`video_vae/config.json`: `vae_tile_size: 256`, `vae_tile_overlap_min:
    64`. `decoder_tile_size`/`decoder_tile_overlap_min` default to the SAME
    values (setup_forward:116-117, `kwargs.get("decoder_tile_size",
    self.tile_size)`) since `load_kwargs` doesn't set them separately.
    `vae_ratio` = product(space_down=[2,2,2,2,1,1]) = 16, matching
    `minimax_h3_video_released_encoder_config()`'s space_down list."""
    return MiniMaxH3TilingConfig(256, 64, 256, 64, 16)


@fieldwise_init
struct MiniMaxH3TileLayout(Movable):
    var start_idx: List[Int]
    var lengths: List[Int]   # nominal length (== tile_size); clamp against
                             # the real axis size when slicing the last tile
    var overlaps: List[Int]  # length N-1: overlap between tile i and i+1

    def num_tiles(self) -> Int:
        return len(self.start_idx)


def minimax_h3_split_tiles(
    input_len: Int, tile_size: Int, tile_overlap_min: Int, vae_ratio: Int,
) raises -> MiniMaxH3TileLayout:
    """`AutoencoderKL.split_tiles` (:192-218), verbatim arithmetic. Grows
    `N` (tile count) until `N` tiles of `tile_size` minus the overlaps cover
    at least `input_len`, then spreads the slack (in `vae_ratio` units)
    across the overlap slots round-robin so the tiling covers `input_len`
    exactly (for `input_len` a multiple of `vae_ratio` — see this file's
    header SCOPE note)."""
    if tile_size >= input_len:
        var s = List[Int]()
        s.append(0)
        var l = List[Int]()
        l.append(input_len)
        var o = List[Int]()
        return MiniMaxH3TileLayout(s^, l^, o^)

    var n = (input_len + tile_size - 1) // tile_size
    var overlaps = List[Int]()
    var remaining = 0
    while True:
        overlaps = List[Int]()
        for _ in range(n - 1):
            overlaps.append(tile_overlap_min)
        var sum_overlaps = 0
        for i in range(len(overlaps)):
            sum_overlaps += overlaps[i]
        remaining = tile_size * n - sum_overlaps - input_len
        if remaining < 0:
            n += 1
        else:
            break

    var remaining_units = remaining // vae_ratio
    for i in range(remaining_units):
        overlaps[i % (n - 1)] += vae_ratio

    var start_idx = List[Int]()
    start_idx.append(0)
    for i in range(n - 1):
        start_idx.append(start_idx[len(start_idx) - 1] + tile_size - overlaps[i])

    var lengths = List[Int]()
    for _ in range(n):
        lengths.append(tile_size)

    return MiniMaxH3TileLayout(start_idx^, lengths^, overlaps^)


def _clamp_tile_len(pos: Int, nominal_len: Int, axis_size: Int) -> Int:
    var length = nominal_len
    if pos + length > axis_size:
        length = axis_size - pos
    return length


# ─────────────────────────────────────────────────────────────────────────────
# ENCODE.
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_video_tiled_encode(
    encoder: MiniMaxH3VideoEncoderDevice,
    pixels: Tensor,  # [1, T, H, W, in_channels] NDHWC
    tiling: MiniMaxH3TilingConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    """`tiled_encode` (:297-363): per-tile `encode`, blend against the RAW
    (unblended) neighbor above then the RAW neighbor to the left — exactly
    the reference's own order, not "fixed" to chain through already-blended
    neighbors — then trim the trailing overlap, concat W then H."""
    var s = pixels.shape()
    var height = s[2]
    var width = s[3]
    var y_layout = minimax_h3_split_tiles(
        height, tiling.tile_size, tiling.tile_overlap_min, tiling.vae_ratio
    )
    var x_layout = minimax_h3_split_tiles(
        width, tiling.tile_size, tiling.tile_overlap_min, tiling.vae_ratio
    )
    var i_max = y_layout.num_tiles()
    var j_max = x_layout.num_tiles()

    if i_max == 1 and j_max == 1:
        return minimax_h3_video_encode_device(encoder, pixels, ctx)

    var raw = List[TArc]()
    for i in range(i_max):
        var i_pos = y_layout.start_idx[i]
        var i_len = _clamp_tile_len(i_pos, y_layout.lengths[i], height)
        var row_slice = slice(pixels, 2, i_pos, i_len, ctx)
        for j in range(j_max):
            var j_pos = x_layout.start_idx[j]
            var j_len = _clamp_tile_len(j_pos, x_layout.lengths[j], width)
            var tile = slice(row_slice, 3, j_pos, j_len, ctx)
            var z = minimax_h3_video_encode_device(encoder, tile, ctx)
            raw.append(TArc(z^))

    var latent_y_overlap = List[Int]()
    for i in range(len(y_layout.overlaps)):
        latent_y_overlap.append(y_layout.overlaps[i] // tiling.vae_ratio)
    var latent_x_overlap = List[Int]()
    for i in range(len(x_layout.overlaps)):
        latent_x_overlap.append(x_layout.overlaps[i] // tiling.vae_ratio)

    var result_rows = List[TArc]()
    for i in range(i_max):
        var row_out = List[TArc]()
        for j in range(j_max):
            var tile = raw[i * j_max + j][].clone(ctx)
            if i > 0:
                tile = minimax_h3_video_blend(
                    raw[(i - 1) * j_max + j][], tile, latent_y_overlap[i - 1], 2, ctx
                )
            if j > 0:
                tile = minimax_h3_video_blend(
                    raw[i * j_max + j - 1][], tile, latent_x_overlap[j - 1], 3, ctx
                )
            if i < i_max - 1:
                var h_now = tile.shape()[2]
                tile = slice(tile, 2, 0, h_now - latent_y_overlap[i], ctx)
            if j < j_max - 1:
                var w_now = tile.shape()[3]
                tile = slice(tile, 3, 0, w_now - latent_x_overlap[j], ctx)
            row_out.append(TArc(tile^))
        var row_cat = row_out[0][].clone(ctx)
        for j in range(1, j_max):
            row_cat = concat(3, ctx, row_cat, row_out[j][])
        result_rows.append(TArc(row_cat^))

    var z = result_rows[0][].clone(ctx)
    for i in range(1, i_max):
        z = concat(2, ctx, z, result_rows[i][])
    return z^


# ─────────────────────────────────────────────────────────────────────────────
# DECODE. Same composition as encode; PIXEL-space overlaps used directly
# (dec is pixel-space, unlike encode's latent-space blend).
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_video_tiled_decode[
    LATENT_TILE_H: Int, LATENT_TILE_W: Int, HEADS: Int, DIM_HEAD: Int,
    NUM_SUFFIX: Int, TOKENS_PER_CLIP: Int,
](
    decoder: MiniMaxH3VideoDecoderDevice,
    latents: Tensor,  # [1, latent_T, latent_H, latent_W, latent_channels] NDHWC
    tiling: MiniMaxH3TilingConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    if decoder.config.heads != HEADS or decoder.config.dim_head != DIM_HEAD:
        raise Error("MiniMax-H3 video tiled decode: comptime HEADS/DIM_HEAD mismatch")
    if decoder.config.num_register_tokens + 1 != NUM_SUFFIX:
        raise Error("MiniMax-H3 video tiled decode: comptime NUM_SUFFIX mismatch")

    var s = latents.shape()
    var latent_h = s[2]
    var latent_w = s[3]
    var height = latent_h * tiling.vae_ratio
    var width = latent_w * tiling.vae_ratio
    var y_layout = minimax_h3_split_tiles(
        height, tiling.decoder_tile_size, tiling.decoder_tile_overlap_min, tiling.vae_ratio
    )
    var x_layout = minimax_h3_split_tiles(
        width, tiling.decoder_tile_size, tiling.decoder_tile_overlap_min, tiling.vae_ratio
    )
    var i_max = y_layout.num_tiles()
    var j_max = x_layout.num_tiles()

    comptime S_TILE = TOKENS_PER_CLIP * LATENT_TILE_H * LATENT_TILE_W + NUM_SUFFIX

    if i_max == 1 and j_max == 1:
        if latent_h != LATENT_TILE_H or latent_w != LATENT_TILE_W:
            raise Error(
                "MiniMax-H3 video tiled decode: latent H/W does not match the"
                " comptime tile size (single-tile path)"
            )
        return minimax_h3_video_decode_device[S_TILE, HEADS, DIM_HEAD](decoder, latents, ctx)

    var raw = List[TArc]()
    for i in range(i_max):
        var i_pos_px = y_layout.start_idx[i]
        var i_len_px = _clamp_tile_len(i_pos_px, y_layout.lengths[i], height)
        if i_pos_px % tiling.vae_ratio != 0 or i_len_px % tiling.vae_ratio != 0:
            raise Error("MiniMax-H3 video tiled decode: tile boundary is not vae_ratio-aligned")
        var i_pos = i_pos_px // tiling.vae_ratio
        var i_len = i_len_px // tiling.vae_ratio
        if i_len != LATENT_TILE_H:
            raise Error(
                "MiniMax-H3 video tiled decode: a tile's latent H does not match"
                " the comptime LATENT_TILE_H -- non-uniform tiling unsupported here"
            )
        var row_slice = slice(latents, 2, i_pos, i_len, ctx)
        for j in range(j_max):
            var j_pos_px = x_layout.start_idx[j]
            var j_len_px = _clamp_tile_len(j_pos_px, x_layout.lengths[j], width)
            if j_pos_px % tiling.vae_ratio != 0 or j_len_px % tiling.vae_ratio != 0:
                raise Error("MiniMax-H3 video tiled decode: tile boundary is not vae_ratio-aligned")
            var j_pos = j_pos_px // tiling.vae_ratio
            var j_len = j_len_px // tiling.vae_ratio
            if j_len != LATENT_TILE_W:
                raise Error(
                    "MiniMax-H3 video tiled decode: a tile's latent W does not"
                    " match the comptime LATENT_TILE_W -- non-uniform tiling"
                    " unsupported here"
                )
            var tile_latent = slice(row_slice, 3, j_pos, j_len, ctx)
            var dec_tile = minimax_h3_video_decode_device[S_TILE, HEADS, DIM_HEAD](
                decoder, tile_latent, ctx
            )
            raw.append(TArc(dec_tile^))

    var result_rows = List[TArc]()
    for i in range(i_max):
        var row_out = List[TArc]()
        for j in range(j_max):
            var tile = raw[i * j_max + j][].clone(ctx)
            if i > 0:
                tile = minimax_h3_video_blend(
                    raw[(i - 1) * j_max + j][], tile, y_layout.overlaps[i - 1], 2, ctx
                )
            if j > 0:
                tile = minimax_h3_video_blend(
                    raw[i * j_max + j - 1][], tile, x_layout.overlaps[j - 1], 3, ctx
                )
            if i < i_max - 1:
                var h_now = tile.shape()[2]
                tile = slice(tile, 2, 0, h_now - y_layout.overlaps[i], ctx)
            if j < j_max - 1:
                var w_now = tile.shape()[3]
                tile = slice(tile, 3, 0, w_now - x_layout.overlaps[j], ctx)
            row_out.append(TArc(tile^))
        var row_cat = row_out[0][].clone(ctx)
        for j in range(1, j_max):
            row_cat = concat(3, ctx, row_cat, row_out[j][])
        result_rows.append(TArc(row_cat^))

    var dec = result_rows[0][].clone(ctx)
    for i in range(1, i_max):
        dec = concat(2, ctx, dec, result_rows[i][])
    return dec^
