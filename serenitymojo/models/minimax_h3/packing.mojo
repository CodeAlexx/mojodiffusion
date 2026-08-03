# serenitymojo/models/minimax_h3/packing.mojo
#
# MiniMax-H3 packed-sequence geometry — the first unit of the H3 port.
#
# H3 runs one transformer over a single packed 1-D sequence holding every
# modality at once. For the text/keyframe tasks the row order is
#
#   [ text (L) | keyframe conditions (C) | target audio (A) | target video (V) ]
#
# and this module places every row and gives it its (t, h, w) rotary
# coordinate. Video and audio share ONE 40-units-per-second rotary clock —
# video advances 5/3 rotary units per pixel frame at 24 fps, audio one unit per
# latent at 40 latents/s — and that shared clock IS the audio/video alignment.
#
# Source (op-for-op port, diffusers PR huggingface/diffusers#14355 at head
# e1b518dfd5e390e7ba09a79a1d39fe1c6cb52dc1):
#   src/diffusers/modular_pipelines/minimax_h3/packing.py
#     resolve_canvas_size        :131-166
#     align_num_frames           :169-183
#     video_latent_num_frames    :186-200
#     audio_latent_num_frames    :203-213
#     _spatial_position_grid     :331-341
#     _temporal_position_grid    :344-353
#     _temporal_position_span    :356-366
#     build_packed_sequence      :369-466
#     build_row_timesteps        :469-498
#
# Three float-order traps, all called out by the reference's own docstrings and
# all reproduced here rather than "cleaned up":
#
#   1. The spatial grid is numpy's `linspace(..., endpoint=False)`, i.e.
#      `start + arange(n) * (stop - start) / n` — NOT torch.linspace, and NOT
#      `start + i * ratio / n`. `stop - start` is recomputed from the rounded
#      `stop`, so the subtraction stays in the expression.
#   2. `_temporal_position_span` is summed by numpy's PAIRWISE summation, which
#      differs from a sequential sum in the last ulp from 16 latent frames on.
#      `_pairwise_sum` below reproduces numpy's algorithm (8 accumulators up to
#      a 128-element block, split-in-half above it).
#   3. Canvas rounding is Python's `round`, i.e. round-half-to-EVEN, not
#      round-half-away-from-zero. `_round_half_even` below.
#
# Positions are Float64 because the reference builds them in float64 and the
# gate compares them BIT-EXACTLY (parity/minimax_h3_packing_parity.mojo).
# Row timesteps are Float32 because the reference builds that tensor in float32.
#
# Host-only: no GPU, no DeviceContext, no Tensor. This is index and coordinate
# arithmetic that runs once per request shape.

from std.collections import List
from std.math import sqrt, floor

# Per-row modality tags. They index the transformer's AdaLN table, so the values
# are a checkpoint contract (packing.py:49-51).
comptime MINIMAX_H3_VIDEO_TAG = 0
comptime MINIMAX_H3_TEXT_TAG = 1
comptime MINIMAX_H3_AUDIO_TAG = 2

# H3 generates at a fixed 24 fps, released for a 768-pixel short edge only, with
# a soft 768x1344 area cap and both axes rounded to a multiple of 32.
comptime MINIMAX_H3_FPS = 24
comptime MINIMAX_H3_SHORT_EDGE = 768
comptime MINIMAX_H3_MAX_PIXELS = 768 * 1344
comptime MINIMAX_H3_CANVAS_MULTIPLE = 32
comptime MINIMAX_H3_MIN_ASPECT_RATIO = Float64(1.0) / Float64(4.0)
comptime MINIMAX_H3_MAX_ASPECT_RATIO = Float64(4.0)

# The video VAE encodes 17 pixel frames per chunk and drops the 3 trailing
# latent frames of every chunk: 17n+5 pixel frames -> 5n+2 latent frames.
comptime MINIMAX_H3_FRAMES_PER_CHUNK = 17
comptime MINIMAX_H3_LATENTS_PER_CHUNK = 5

# The audio VAE hops 800 samples at 32 kHz, i.e. 40 latents per second. Stereo
# is carried as two channel-major blocks of audio rows.
comptime MINIMAX_H3_AUDIO_LATENTS_PER_SECOND = 40
comptime MINIMAX_H3_AUDIO_CHANNELS = 2

# Rotary-time constants. One latent frame spans 5/3 * frames_per_latent rotary
# units; the pattern (1, 4, 4, 4, 4) mirrors the VAE's 17-to-5 grouping. The
# spatial axes are normalized by the square root of the latent area, then x32.
comptime ROPE_FRAME_RESCALE = Float64(5.0) / Float64(3.0)
comptime ROPE_SPATIAL_SCALE = Float64(32.0)

# Keyframe anchor codes (the reference uses the strings "first" / "last").
comptime MINIMAX_H3_ANCHOR_FIRST = 0
comptime MINIMAX_H3_ANCHOR_LAST = 1

# numpy's pairwise-summation block size (PW_BLOCKSIZE in numpy's loops.c.src).
comptime _PW_BLOCKSIZE = 128


@fieldwise_init
struct MiniMaxH3CanvasSize(Copyable, Movable):
    """A resolved canvas, in pixels."""

    var height: Int
    var width: Int


@fieldwise_init
struct MiniMaxH3PackedSequence(Copyable, Movable):
    """The structural description of one packed MiniMax-H3 sequence.

    `position_ids` is flat row-major `[sequence_length * 3]`: the (t, h, w)
    rotary coordinate of row `i` is `position_ids[3*i : 3*i+3]`.

    `video_indices` lists the keyframe conditioning rows first, then the target
    rows; `num_condition_video_rows` is how many leading entries are
    conditioning. Same contract for audio (always 0 conditioning rows on the
    t2va / fl2va path; ref2va is a separate builder)."""

    var sequence_length: Int
    var position_ids: List[Float64]
    var token_tags: List[Int]
    var video_indices: List[Int]
    var audio_indices: List[Int]
    var text_indices: List[Int]
    var num_condition_video_rows: Int
    var num_condition_audio_rows: Int


@fieldwise_init
struct MiniMaxH3RowTimesteps(Copyable, Movable):
    """The distinct row timesteps, sorted, and every row's index into them."""

    var values: List[Float32]
    var indices: List[Int]


def _round_half_even(x: Float64) -> Float64:
    """Python's `round` on a float: ties go to the even integer.

    Mojo's `round` rounds half away from zero, which disagrees with the
    reference on exact .5 canvas divisions."""
    var lower = floor(x)
    var frac = x - lower
    if frac > 0.5:
        return lower + 1.0
    if frac < 0.5:
        return lower
    # Exact tie: pick the even neighbour.
    if (Int(lower) % 2) == 0:
        return lower
    return lower + 1.0


def _pairwise_sum(a: List[Float64], start: Int, n: Int) -> Float64:
    """numpy's pairwise summation (`pairwise_sum_DOUBLE`, loops.c.src).

    Below 8 elements it sums sequentially; up to a 128-element block it runs 8
    independent accumulators and combines them as a balanced tree; above that it
    splits in half at a multiple of 8. Reproduced because the reference sums the
    temporal spans with `np.ndarray.sum` and the order is observable."""
    if n < 8:
        var res = Float64(0.0)
        for i in range(n):
            res += a[start + i]
        return res

    if n <= _PW_BLOCKSIZE:
        var r0 = a[start + 0]
        var r1 = a[start + 1]
        var r2 = a[start + 2]
        var r3 = a[start + 3]
        var r4 = a[start + 4]
        var r5 = a[start + 5]
        var r6 = a[start + 6]
        var r7 = a[start + 7]
        var i = 8
        var limit = n - (n % 8)
        while i < limit:
            r0 += a[start + i + 0]
            r1 += a[start + i + 1]
            r2 += a[start + i + 2]
            r3 += a[start + i + 3]
            r4 += a[start + i + 4]
            r5 += a[start + i + 5]
            r6 += a[start + i + 6]
            r7 += a[start + i + 7]
            i += 8
        var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[start + i]
            i += 1
        return res

    var n2 = n // 2
    n2 -= n2 % 8
    return _pairwise_sum(a, start, n2) + _pairwise_sum(a, start + n2, n - n2)


def minimax_h3_resolve_canvas_size(
    aspect_width: Float64, aspect_height: Float64
) raises -> MiniMaxH3CanvasSize:
    """Resolve a display aspect ratio into a MiniMax-H3 canvas (packing.py:131).

    The short edge starts at 768, the area is capped at 768*1344, and both axes
    are then rounded to the nearest multiple of 32 — so the final area may end
    up slightly above the pre-rounding budget. Only the ratio matters."""
    if aspect_width <= 0.0 or aspect_height <= 0.0:
        raise Error("MiniMax-H3: the aspect ratio must be positive")

    var ratio = aspect_width / aspect_height
    if ratio < MINIMAX_H3_MIN_ASPECT_RATIO or ratio > MINIMAX_H3_MAX_ASPECT_RATIO:
        raise Error("MiniMax-H3 supports aspect ratios from 1:4 to 4:1")

    var short_edge = Float64(MINIMAX_H3_SHORT_EDGE)
    var width: Float64
    var height: Float64
    if ratio >= 1.0:
        width = short_edge * ratio
        height = short_edge
    else:
        width = short_edge
        height = short_edge / ratio

    var area = width * height
    var max_pixels = Float64(MINIMAX_H3_MAX_PIXELS)
    if area > max_pixels:
        var scale = sqrt(max_pixels / area)
        width = width * scale
        height = height * scale

    var multiple = Float64(MINIMAX_H3_CANVAS_MULTIPLE)
    var rounded_h = Int(_round_half_even(height / multiple)) * MINIMAX_H3_CANVAS_MULTIPLE
    var rounded_w = Int(_round_half_even(width / multiple)) * MINIMAX_H3_CANVAS_MULTIPLE
    if rounded_h < MINIMAX_H3_CANVAS_MULTIPLE:
        rounded_h = MINIMAX_H3_CANVAS_MULTIPLE
    if rounded_w < MINIMAX_H3_CANVAS_MULTIPLE:
        rounded_w = MINIMAX_H3_CANVAS_MULTIPLE
    return MiniMaxH3CanvasSize(rounded_h, rounded_w)


def minimax_h3_align_num_frames(num_frames: Int) raises -> Int:
    """Snap a frame count up to the next 17n+5 the video VAE can encode."""
    if num_frames < 1:
        raise Error("MiniMax-H3: `num_frames` must be positive")
    var aligned = num_frames
    while (aligned % MINIMAX_H3_FRAMES_PER_CHUNK) != MINIMAX_H3_LATENTS_PER_CHUNK:
        aligned += 1
    return aligned


def minimax_h3_video_latent_num_frames(num_frames: Int) raises -> Int:
    """Latent frames the video VAE produces for a 17n+5 frame count: 5n+2."""
    if (num_frames % MINIMAX_H3_FRAMES_PER_CHUNK) != MINIMAX_H3_LATENTS_PER_CHUNK:
        raise Error("MiniMax-H3: `num_frames` must be of the form 17n+5")
    return (
        num_frames - MINIMAX_H3_LATENTS_PER_CHUNK
    ) // MINIMAX_H3_FRAMES_PER_CHUNK * MINIMAX_H3_LATENTS_PER_CHUNK + 2


def minimax_h3_audio_latent_num_frames(num_frames: Int) -> Int:
    """Audio latents covering `num_frames` video frames at 24 fps, on the 40 Hz
    latent grid. Same op order as the reference: `n / 24 * 40`, then `round`."""
    var seconds = Float64(num_frames) / Float64(MINIMAX_H3_FPS)
    return Int(_round_half_even(seconds * Float64(MINIMAX_H3_AUDIO_LATENTS_PER_SECOND)))


def minimax_h3_spatial_position_grid(
    dim: Int, patch: Int, sqrt_area: Float64
) -> List[Float64]:
    """One aspect-normalized spatial rotary axis (packing.py:331).

    `dim // patch` coordinates centred on the unit interval, right endpoint
    excluded, scaled up by 32 — so a square canvas spans [0, 32).

    numpy's `linspace(start, stop, num, endpoint=False)` computes
    `arange(num) * ((stop - start) / num) + start`. Reproduced exactly: `stop`
    is formed first, then the difference is taken from it."""
    var ratio = Float64(dim) / sqrt_area
    var left = (1.0 - ratio) / 2.0
    var stop = left + ratio
    var num = dim // patch
    var step = (stop - left) / Float64(num)
    var out = List[Float64]()
    for i in range(num):
        out.append((Float64(i) * step + left) * ROPE_SPATIAL_SCALE)
    return out^


def _temporal_spans(num_latent_frames: Int) -> List[Float64]:
    """`5/3 * (1, 4, 4, 4, 4)[i % 5]` per latent frame (packing.py:346)."""
    var spans = List[Float64]()
    for index in range(num_latent_frames):
        var slot = index % 5
        var frames_per_latent = Float64(1.0) if slot == 0 else Float64(4.0)
        spans.append(ROPE_FRAME_RESCALE * frames_per_latent)
    return spans^


def minimax_h3_temporal_position_grid(
    num_latent_frames: Int, origin: Float64
) -> List[Float64]:
    """The rotary time of every latent frame, starting at `origin`.

    Spacing is non-uniform. The reference builds this as
    `origin + cat([zeros(1), spans[:-1].cumsum(0)])`, i.e. an EXCLUSIVE
    sequential cumulative sum with `origin` added to each element."""
    var spans = _temporal_spans(num_latent_frames)
    var out = List[Float64]()
    var running = Float64(0.0)
    for index in range(num_latent_frames):
        if index == 0:
            out.append(origin)
        else:
            running += spans[index - 1]
            out.append(origin + running)
    return out^


def minimax_h3_temporal_position_span(num_latent_frames: Int) -> Float64:
    """The rotary time spanned by `num_latent_frames` latent frames.

    The reference builds `ones(n) * 5/3`, scales the strided slices by
    (1, 4, 4, 4, 4) and sums with numpy — pairwise, not sequentially. The
    keyframe "last" anchor is computed from this, so the order is load-bearing."""
    var spans = List[Float64]()
    for index in range(num_latent_frames):
        var slot = index % 5
        var frames_per_latent = Float64(1.0) if slot == 0 else Float64(4.0)
        spans.append(ROPE_FRAME_RESCALE * frames_per_latent)
    return _pairwise_sum(spans, 0, num_latent_frames)


def minimax_h3_build_packed_sequence(
    text_token_tags: List[Int],
    num_latent_frames: Int,
    latent_height: Int,
    latent_width: Int,
    num_audio_latents: Int,
    patch_h: Int,
    patch_w: Int,
    keyframe_anchors: List[Int],
) raises -> MiniMaxH3PackedSequence:
    """Build the [text | keyframe conditions | target audio | target video]
    layout used by the t2va and fl2va tasks (packing.py:369).

    `text_token_tags` is the modality tag of every text row: 1 for text, 0 for
    the rows of a keyframe's vision block. `keyframe_anchors` holds one
    MINIMAX_H3_ANCHOR_* code per conditioning block, in packed order."""
    var rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    var num_text_tokens = len(text_token_tags)
    var num_condition_rows = len(keyframe_anchors) * rows_per_frame
    var num_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    var num_video_rows = num_latent_frames * rows_per_frame
    var sequence_length = (
        num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows
    )

    var condition_start = num_text_tokens
    var audio_start = condition_start + num_condition_rows
    var video_start = audio_start + num_audio_rows

    # 1. The (t, h, w) grid. Text rows sit on the time axis at their row index,
    # and the media rows continue the time axis from there, so text length
    # shifts the whole media clock.
    var position_ids = List[Float64]()
    for _ in range(sequence_length * 3):
        position_ids.append(Float64(0.0))
    for i in range(num_text_tokens):
        position_ids[3 * i] = Float64(i)

    var sqrt_area = sqrt(Float64(latent_height * latent_width))
    var height_grid = minimax_h3_spatial_position_grid(latent_height, patch_h, sqrt_area)
    var width_grid = minimax_h3_spatial_position_grid(latent_width, patch_w, sqrt_area)
    # meshgrid(height, width, indexing="ij") flattened row-major.
    var frame_h = List[Float64]()
    var frame_w = List[Float64]()
    for h in range(len(height_grid)):
        for w in range(len(width_grid)):
            frame_h.append(height_grid[h])
            frame_w.append(width_grid[w])

    var text_time = Float64(num_text_tokens)
    for index in range(len(keyframe_anchors)):
        var anchor = keyframe_anchors[index]
        var anchor_time: Float64
        if anchor == MINIMAX_H3_ANCHOR_FIRST:
            anchor_time = text_time
        elif anchor == MINIMAX_H3_ANCHOR_LAST:
            anchor_time = (
                text_time
                + minimax_h3_temporal_position_span(num_latent_frames)
                - ROPE_FRAME_RESCALE
            )
        else:
            raise Error("MiniMax-H3: a keyframe anchor must be first or last")
        var row_start = condition_start + index * rows_per_frame
        for row in range(rows_per_frame):
            var base = 3 * (row_start + row)
            position_ids[base] = anchor_time
            position_ids[base + 1] = frame_h[row]
            position_ids[base + 2] = frame_w[row]

    # Audio rows are channel-major and share the video's rotary clock: one unit
    # per latent at 40 latents/s equals 24 fps * 5/3. They carry no height
    # coordinate and are pinned to the two extremes of the width grid.
    var width_low = width_grid[0]
    var width_high = width_grid[len(width_grid) - 1]
    for row in range(num_audio_rows):
        var base = 3 * (audio_start + row)
        position_ids[base] = text_time + Float64(row % num_audio_latents)
        position_ids[base + 2] = width_low if row < num_audio_latents else width_high

    var temporal_grid = minimax_h3_temporal_position_grid(num_latent_frames, text_time)
    for frame in range(num_latent_frames):
        var frame_time = temporal_grid[frame]
        for row in range(rows_per_frame):
            var base = 3 * (video_start + frame * rows_per_frame + row)
            position_ids[base] = frame_time
            position_ids[base + 1] = frame_h[row]
            position_ids[base + 2] = frame_w[row]

    # 2. Row indices and modality tags.
    var video_indices = List[Int]()
    for i in range(condition_start, audio_start):
        video_indices.append(i)
    for i in range(video_start, sequence_length):
        video_indices.append(i)
    var audio_indices = List[Int]()
    for i in range(audio_start, video_start):
        audio_indices.append(i)
    var text_indices = List[Int]()
    for i in range(num_text_tokens):
        text_indices.append(i)

    var token_tags = List[Int]()
    for _ in range(sequence_length):
        token_tags.append(0)
    for i in range(num_text_tokens):
        token_tags[i] = text_token_tags[i]
    for i in range(len(audio_indices)):
        token_tags[audio_indices[i]] = MINIMAX_H3_AUDIO_TAG
    for i in range(len(video_indices)):
        token_tags[video_indices[i]] = MINIMAX_H3_VIDEO_TAG

    return MiniMaxH3PackedSequence(
        sequence_length,
        position_ids^,
        token_tags^,
        video_indices^,
        audio_indices^,
        text_indices^,
        num_condition_rows,
        0,
    )


def minimax_h3_build_row_timesteps(
    layout: MiniMaxH3PackedSequence,
    video_timestep: Float32,
    audio_timestep: Float32,
    condition_video_timestep: Float32,
    condition_audio_timestep: Float32,
) -> MiniMaxH3RowTimesteps:
    """Assign a timestep to every row and reduce to (distinct sorted values,
    per-row index) — the transformer's modulation-row contract (packing.py:469).

    One forward serves rows at different noise levels: the generated video and
    audio rows step down their own schedules while the conditioning rows stay
    pinned at their noise-augmentation level. Text rows never reach an output
    head and inherit the video timestep."""
    var row_timesteps = List[Float32]()
    for _ in range(layout.sequence_length):
        row_timesteps.append(video_timestep)
    for i in range(layout.num_condition_video_rows):
        row_timesteps[layout.video_indices[i]] = condition_video_timestep
    for i in range(layout.num_condition_audio_rows, len(layout.audio_indices)):
        row_timesteps[layout.audio_indices[i]] = audio_timestep
    for i in range(layout.num_condition_audio_rows):
        row_timesteps[layout.audio_indices[i]] = condition_audio_timestep

    # Distinct values, ascending. At most four ever occur, so an insertion sort
    # over the distinct set is the whole reduction.
    var values = List[Float32]()
    for i in range(len(row_timesteps)):
        var value = row_timesteps[i]
        var seen = False
        for j in range(len(values)):
            if values[j] == value:
                seen = True
                break
        if not seen:
            values.append(value)
    for i in range(1, len(values)):
        var key = values[i]
        var j = i - 1
        while j >= 0 and values[j] > key:
            values[j + 1] = values[j]
            j -= 1
        values[j + 1] = key

    var indices = List[Int]()
    for i in range(len(row_timesteps)):
        var value = row_timesteps[i]
        var slot = 0
        for j in range(len(values)):
            if values[j] == value:
                slot = j
                break
        indices.append(slot)

    return MiniMaxH3RowTimesteps(values^, indices^)
