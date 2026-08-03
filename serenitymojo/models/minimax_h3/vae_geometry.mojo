# serenitymojo/models/minimax_h3/vae_geometry.mojo
#
# MiniMax-H3 video VAE geometry — unit 4 of the H3 port.
#
# The arithmetic that decides WHICH pixels and WHICH latent frames get
# processed: spatial tile layout, the linear blend, and the temporal chunking of
# encode and decode. No convolutions, no weights. This is the part that silently
# corrupts output when a port gets it subtly wrong — a misplaced tile boundary
# or a dropped overlap frame produces a plausible video with seams or a stutter,
# not a crash.
#
# Source (op-for-op port, diffusers PR huggingface/diffusers#14355 at head
# e1b518dfd5e390e7ba09a79a1d39fe1c6cb52dc1):
#   src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py
#     derived geometry (__init__)  :593-599
#     _split_tiles                 :645-666
#     _blend                       :668-686
#     _encode                      :771-795
#     _decode                      :797-846
#
# Released geometry: spatial ratio 16, temporal ratio 4, clip_length 17,
# token_drop 3 — from which frame_pre_padding 3, tokens_chunk_size 5,
# token_overlap 2, frame_overlap 5 follow, and with them the product's
# 17n+5 pixel frames <-> 5n+2 latent frames grid.
#
# Tiling is ON by default in the release, and the published frames are the
# blended-tile ones, so disabling it changes the output. It is not an
# optimization to be skipped.
#
# DECODE FLOOR — a defect in the reference, reproduced deliberately:
# `_decode` on a 2-latent-frame input computes num_tokens = 2 + 3 = 5,
# pad_tokens = 0, num_chunks = 5 // 5 - 1 = 0, runs its chunk loop zero times,
# and hands an empty list to `torch.cat`. Swept over lengths 2..19, 2 is the
# ONLY failing length. It is exactly `video_latent_num_frames(5)`, the shortest
# clip on the model's own grid. The ComfyUI implementation guards it ("too few
# tokens for one chunk (e.g. T_lat == 2): pad one extra chunk"); the diffusers
# draft does not. This port implements the diffusers arithmetic and RAISES at 2
# rather than inventing behaviour the vendor has not published.

from std.collections import List

comptime MINIMAX_H3_VAE_SPATIAL_RATIO = 16
comptime MINIMAX_H3_VAE_TEMPORAL_RATIO = 4
comptime MINIMAX_H3_VAE_CLIP_LENGTH = 17
comptime MINIMAX_H3_VAE_TOKEN_DROP = 3
comptime MINIMAX_H3_VAE_TILE_SIZE = 256
comptime MINIMAX_H3_VAE_TILE_MIN_OVERLAP = 64


@fieldwise_init
struct MiniMaxH3VaeGeometry(Copyable, Movable):
    """The derived temporal geometry of the video VAE.

    `clip_length` is not a multiple of `temporal_ratio`, so the decoder has to
    re-derive the implicit leading pad and the overlap `token_drop` leaves."""

    var spatial_ratio: Int
    var temporal_ratio: Int
    var clip_length: Int
    var token_drop: Int
    var frame_pre_padding: Int
    var tokens_chunk_size: Int
    var token_overlap: Int
    var frame_overlap: Int


def minimax_h3_vae_geometry(
    clip_length: Int = MINIMAX_H3_VAE_CLIP_LENGTH,
    token_drop: Int = MINIMAX_H3_VAE_TOKEN_DROP,
    spatial_ratio: Int = MINIMAX_H3_VAE_SPATIAL_RATIO,
    temporal_ratio: Int = MINIMAX_H3_VAE_TEMPORAL_RATIO,
) -> MiniMaxH3VaeGeometry:
    """Derive the chunking constants (autoencoder_kl_minimax_h3.py:593-599)."""
    var frame_pre_padding = (-clip_length) % temporal_ratio
    if frame_pre_padding < 0:
        frame_pre_padding += temporal_ratio
    var tokens_chunk_size = -((-clip_length) // temporal_ratio)  # ceil
    var token_overlap = (-token_drop) % tokens_chunk_size
    if token_overlap < 0:
        token_overlap += tokens_chunk_size
    var frame_overlap = token_overlap * temporal_ratio - frame_pre_padding
    if frame_overlap < 0:
        frame_overlap = 0
    return MiniMaxH3VaeGeometry(
        spatial_ratio,
        temporal_ratio,
        clip_length,
        token_drop,
        frame_pre_padding,
        tokens_chunk_size,
        token_overlap,
        frame_overlap,
    )


@fieldwise_init
struct MiniMaxH3TileLayout(Copyable, Movable):
    """Where the tiles start, how long each is, and the overlap between
    consecutive pairs. `overlaps` has one fewer entry than `starts`."""

    var starts: List[Int]
    var sizes: List[Int]
    var overlaps: List[Int]


def minimax_h3_split_tiles(
    length: Int, tile_size: Int, min_overlap: Int, spatial_ratio: Int
) -> MiniMaxH3TileLayout:
    """Lay `tile_size`-wide tiles over `length` pixels.

    The tile count is the smallest whose union covers `length` while keeping
    every overlap at least `min_overlap`; the slack is then distributed
    round-robin over the overlaps in whole `spatial_ratio` steps, so every tile
    boundary stays latent-aligned."""
    if tile_size >= length:
        return MiniMaxH3TileLayout([0], [length], List[Int]())

    var num_tiles = -((-length) // tile_size)  # ceil
    while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
        num_tiles += 1

    var overlaps = List[Int]()
    for _ in range(num_tiles - 1):
        overlaps.append(min_overlap)
    var total_overlap = min_overlap * (num_tiles - 1)
    var remaining = tile_size * num_tiles - total_overlap - length
    for i in range(remaining // spatial_ratio):
        overlaps[i % (num_tiles - 1)] += spatial_ratio

    var starts = List[Int]()
    starts.append(0)
    for i in range(num_tiles - 1):
        starts.append(starts[len(starts) - 1] + tile_size - overlaps[i])
    var sizes = List[Int]()
    for _ in range(num_tiles):
        sizes.append(tile_size)
    return MiniMaxH3TileLayout(starts^, sizes^, overlaps^)


def minimax_h3_blend(
    a: List[Float32], b: List[Float32], blend_extent: Int
) -> List[Float32]:
    """Linear cross-fade of `a`'s tail into `b`'s head.

    The extent clamps to the shorter of the two. The output is `b`'s length: the
    first `extent` values are the blend, the rest is `b` untouched — which is
    why blending never changes how many rows a chunk contributes."""
    var extent = blend_extent
    if len(a) < extent:
        extent = len(a)
    if len(b) < extent:
        extent = len(b)

    var out = List[Float32]()
    for i in range(extent):
        var weight_b = Float32(i) / Float32(extent)
        var weight_a = Float32(1.0) - weight_b
        out.append(a[len(a) - extent + i] * weight_a + b[i] * weight_b)
    for i in range(extent, len(b)):
        out.append(b[i])
    return out^


def minimax_h3_encode_latent_frames(
    num_frames: Int, geometry: MiniMaxH3VaeGeometry
) -> Int:
    """Latent frames `_encode` produces for `num_frames` pixel frames.

    The input is repeat-padded up to a whole number of `clip_length` chunks,
    each chunk yields `ceil(clip_length / temporal_ratio)` latent frames, and
    the trailing `token_drop` are dropped from the concatenation."""
    var padded = num_frames
    var remainder = padded % geometry.clip_length
    if remainder != 0:
        padded += geometry.clip_length - remainder
    var chunks = padded // geometry.clip_length
    var latents = chunks * geometry.tokens_chunk_size
    if geometry.token_drop > 0:
        latents -= geometry.token_drop
    return latents


def minimax_h3_decode_num_frames(
    num_latent_frames: Int, geometry: MiniMaxH3VaeGeometry
) raises -> Int:
    """Pixel frames `_decode` produces from `num_latent_frames` latent frames.

    Walks the reference's own chunk assembly, counting rows rather than moving
    pixels: every chunk contributes its `j == 0` piece, the held `j == 1` piece
    is merged into the following chunk (so overlap frames are counted once), and
    the trailing frames that the repeat-padded latents produced are cut."""
    var chunk_num_frames = geometry.tokens_chunk_size * geometry.temporal_ratio
    var num_tokens = num_latent_frames + geometry.token_drop
    var pad_tokens = (-num_tokens) % geometry.tokens_chunk_size
    if pad_tokens < 0:
        pad_tokens += geometry.tokens_chunk_size
    var splits = 1 if geometry.token_drop > 0 else 0
    var num_chunks = (num_tokens + pad_tokens) // geometry.tokens_chunk_size - splits

    if num_chunks < 1:
        # See DECODE FLOOR in the module header: the reference produces zero
        # chunks here and its `torch.cat` raises on an empty list. Only
        # `num_latent_frames == 2` reaches this.
        raise Error(
            "MiniMax-H3 video VAE: too few latent frames to decode (the"
            " reference produces zero chunks and raises); 3 or more are needed"
        )

    var padded_tokens = num_latent_frames + pad_tokens
    var total_frames = 0
    var have_overlap = False
    var overlap_frames = 0

    for i in range(num_chunks):
        var start = i * geometry.tokens_chunk_size
        var stop = start + geometry.tokens_chunk_size + geometry.token_overlap
        if stop > padded_tokens:
            stop = padded_tokens
        var clip_frames = (stop - start) * geometry.temporal_ratio
        for j in range(splits + 1):
            var frame_start = j * chunk_num_frames
            var frame_stop = frame_start + chunk_num_frames
            if frame_stop > clip_frames:
                frame_stop = clip_frames
            var piece = frame_stop - frame_start
            if piece < 0:
                piece = 0
            piece -= geometry.frame_pre_padding
            if piece < 0:
                piece = 0
            if j == 0:
                # A blend with the held overlap returns the chunk's own length,
                # so the row count is unchanged by it.
                total_frames += piece
                have_overlap = False
            else:
                overlap_frames = piece
                have_overlap = True
    if have_overlap:
        total_frames += overlap_frames

    if pad_tokens > 0:
        var intra_tail = geometry.clip_length % geometry.temporal_ratio
        var tokens_before_pad = padded_tokens - pad_tokens
        var pad_frames = 0
        for k in range(pad_tokens):
            var on_boundary = (tokens_before_pad + k) % geometry.tokens_chunk_size == 0
            if intra_tail != 0 and on_boundary:
                pad_frames += intra_tail
            else:
                pad_frames += geometry.temporal_ratio
        total_frames -= pad_frames

    return total_frames
