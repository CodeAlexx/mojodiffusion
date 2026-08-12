# serenitymojo/models/minimax_h3/packing_ref2va.mojo
#
# The ref2va half of MiniMax-H3's packed sequence — unit 3 of the H3 port.
#
# ref2va takes an ordered list of references (images, videos with their
# soundtrack, audio clips) and packs one block per reference ahead of the
# generated rows:
#
#   [ text (L) | ref block 1 | ref block 2 | ... | target audio (A) | target video (V) ]
#
# Order is semantic twice over: it fixes the "<Picture i>" / "<Audio j>" /
# "<Video k>" labels of the prompt presentation, and it advances the shared
# rotary clock. An image occupies ONE integer rotary slot, an audio reference as
# many slots as it has latents, and a video reference the longer of its two
# spans — its soundtrack rows packing immediately before its own video rows,
# both sharing one origin.
#
# Unlike an fl2va keyframe, a reference never binds the target geometry: each
# carries its own aspect-normalized spatial grid.
#
# Source (op-for-op port, diffusers PR huggingface/diffusers#14355 at head
# e1b518dfd5e390e7ba09a79a1d39fe1c6cb52dc1):
#   src/diffusers/modular_pipelines/minimax_h3/packing_ref2va.py
#     _temporal_position_span        :387-398
#     _frame_position_grid           :401-409
#     _fill_audio_positions          :412-432
#     build_ref2va_packed_sequence   :435-551
#     resolve_reference_image_size   :554-576
#     resample_reference_frames      :622-651
#     sample_reference_video_frames  :684-712
#     trim_reference_num_frames      :822-841
#
# THE TRAP IN THIS FILE: its `_temporal_position_span` sums SEQUENTIALLY, while
# `packing.py`'s sums with numpy's pairwise algorithm. The reference's own
# docstring: "the two orders differ in the last ulp from 16 latent frames
# onwards. The reference implementation keeps both, one per call site, so the
# port has to as well." Hence `minimax_h3_temporal_position_span_sequential`
# here alongside `minimax_h3_temporal_position_span` in packing.mojo. They are
# not interchangeable and must not be deduplicated.
#
# Scope: arithmetic only. Media decoding, PIL resampling and waveform
# resampling are a later unit; the frame-selection index math lives here because
# it is pure integer/float arithmetic and is gateable now.

from std.collections import List
from std.math import sqrt, floor, round

from serenitymojo.models.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3PackedSequence,
    ROPE_FRAME_RESCALE,
    _round_half_even,
    minimax_h3_spatial_position_grid,
    minimax_h3_temporal_position_grid,
)
from serenitymojo.models.minimax_h3.motion_context import (
    minimax_h3_motion_context_audio_latents,
    minimax_h3_motion_context_step_offsets,
    minimax_h3_motion_context_steps,
)

# Reference images are resized to a 2048 pixel short edge — upscaling included —
# and both axes rounded to a multiple of 32 independently. No area cap, so a 4:1
# reference is 8192x2048.
comptime MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048

# The conditioner sees a reference video at 2 fps, and Qwen3-VL merges every two
# of those frames into one vision block labelled with the pair's mean timestamp.
comptime MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS = Float64(2.0)
comptime MINIMAX_H3_QWEN_TEMPORAL_PATCH = 2

# Documented per-request limits of the omni-reference task.
comptime MINIMAX_H3_MAX_REFERENCE_IMAGES = 9
comptime MINIMAX_H3_MAX_REFERENCE_VIDEOS = 3
comptime MINIMAX_H3_MAX_REFERENCE_AUDIOS = 3
comptime MINIMAX_H3_MAX_REFERENCES = 12

# Reference kinds.
comptime MINIMAX_H3_REF_IMAGE = 0
comptime MINIMAX_H3_REF_VIDEO = 1
comptime MINIMAX_H3_REF_AUDIO = 2


@fieldwise_init
struct MiniMaxH3PreparedReference(Copyable, Movable):
    """One ref2va reference with its latent geometry already resolved.

    `kind` is MINIMAX_H3_REF_*. An image has one latent frame; an audio
    reference has no visual rows; a video reference may carry both."""

    var kind: Int
    var num_latent_frames: Int
    var latent_height: Int
    var latent_width: Int
    var num_audio_latents: Int

    def num_video_rows(self) -> Int:
        """Packed video rows, for the (1, 2, 2) patch H3 packs video with."""
        return self.num_latent_frames * (self.latent_height // 2) * (self.latent_width // 2)

    def num_audio_rows(self) -> Int:
        """One row per latent per stereo channel."""
        return self.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS


def minimax_h3_temporal_position_span_sequential(num_latent_frames: Int) -> Float64:
    """The rotary time a VIDEO REFERENCE advances the clock by.

    Deliberately NOT `minimax_h3_temporal_position_span` in packing.mojo, which
    reproduces a numpy PAIRWISE sum for the fl2va keyframe anchor. The reference
    keeps one per call site and so does this port.

    The reference sums with Python's builtin `sum()` over a generator, and since
    CPython 3.12 that applies NEUMAIER COMPENSATED summation to floats — it is
    not a naive left-to-right accumulation. Reproduced here: a naive sum misses
    the reference by ~4e-14 from 21 latent frames on, which this gate caught on
    a 22-frame video reference (214 of 939 coordinates off, first at row 99)."""
    var total = Float64(0.0)
    var compensation = Float64(0.0)
    for index in range(num_latent_frames):
        var slot = index % 5
        var frames_per_latent = Float64(1.0) if slot == 0 else Float64(4.0)
        var term = ROPE_FRAME_RESCALE * frames_per_latent
        var running = total + term
        var total_mag = -total if total < 0.0 else total
        var term_mag = -term if term < 0.0 else term
        if total_mag >= term_mag:
            compensation += (total - running) + term
        else:
            compensation += (term - running) + total
        total = running
    return total + compensation


def minimax_h3_resolve_reference_image_size(
    width: Int, height: Int
) raises -> List[Int]:
    """Resolution a reference image is encoded at: 2048 short edge, both axes
    rounded to a multiple of 32. Upscaling is intended and there is NO area cap,
    unlike the target canvas. Returns [height, width]."""
    if width <= 0 or height <= 0:
        raise Error("MiniMax-H3: a reference image must have a positive size")
    if width > 4 * height or height > 4 * width:
        raise Error("MiniMax-H3: a reference image must be within 1:4 and 4:1")

    var short_edge = width if width < height else height
    var scale = Float64(MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE) / Float64(short_edge)
    var multiple = Float64(MINIMAX_H3_CANVAS_MULTIPLE)
    var out_h = Int(_round_half_even(Float64(height) * scale / multiple)) * MINIMAX_H3_CANVAS_MULTIPLE
    var out_w = Int(_round_half_even(Float64(width) * scale / multiple)) * MINIMAX_H3_CANVAS_MULTIPLE
    if out_h < MINIMAX_H3_CANVAS_MULTIPLE:
        out_h = MINIMAX_H3_CANVAS_MULTIPLE
    if out_w < MINIMAX_H3_CANVAS_MULTIPLE:
        out_w = MINIMAX_H3_CANVAS_MULTIPLE
    return [out_h, out_w]


def minimax_h3_trim_reference_num_frames(num_frames: Int) raises -> Int:
    """Snap a reference video's frame count DOWN to a 17n+5 the video VAE
    encodes without padding. A reference is truncated to the target's own frame
    count, which is already of that form, so this only bites when the reference
    is shorter than the video being generated."""
    if num_frames < 1:
        raise Error("MiniMax-H3: a reference video must have at least one frame")
    # Python floor division, which differs from truncation when the numerator is
    # negative — and it is, for any reference shorter than 5 frames.
    var numerator = num_frames - MINIMAX_H3_LATENTS_PER_CHUNK
    var chunks = numerator // MINIMAX_H3_FRAMES_PER_CHUNK
    if numerator < 0 and chunks * MINIMAX_H3_FRAMES_PER_CHUNK != numerator:
        chunks -= 1
    if chunks < 1:
        chunks = 1
    return chunks * MINIMAX_H3_FRAMES_PER_CHUNK + MINIMAX_H3_LATENTS_PER_CHUNK


def minimax_h3_resample_reference_repeats(num_frames: Int, fps: Float64) raises -> List[Int]:
    """Per-source-frame repeat counts that put a reference video on H3's 24 fps.

    The reference decoded every video reference through ffmpeg's `fps` filter, a
    constant-frame-rate resampler: source frame `i` lands on output slot
    `floor(i * 24/fps + 0.5)`, a slot holds the last frame that landed on it, and
    the stream's end rounds onto the grid the same way. So frame `i` is repeated
    `slot(i+1) - slot(i)` times, with the last one running to
    `floor(n * 24/fps + 0.5)`.

    An identity (all ones) for frames already at 24 fps."""
    if fps <= 0.0:
        raise Error("MiniMax-H3: a reference video must have a positive frame rate")
    var out = List[Int]()
    if fps == Float64(MINIMAX_H3_FPS):
        for _ in range(num_frames):
            out.append(1)
        return out^

    var scale = Float64(MINIMAX_H3_FPS) / fps
    var slots = List[Int]()
    for index in range(num_frames):
        slots.append(Int(floor(Float64(index) * scale + 0.5)))
    var end_slot = Int(floor(Float64(num_frames) * scale + 0.5))
    for index in range(num_frames):
        var next_slot = slots[index + 1] if index + 1 < num_frames else end_slot
        out.append(next_slot - slots[index])
    return out^


@fieldwise_init
struct MiniMaxH3ConditionerFrames(Copyable, Movable):
    """Which frames the conditioner sees, and the timestamp of every merged
    vision block."""

    var indices: List[Int]
    var block_timestamps: List[Float64]


def minimax_h3_sample_reference_video_frames(
    num_frames: Int,
) -> MiniMaxH3ConditionerFrames:
    """Sample the frames the conditioner reads from a prepared reference video.

    2 fps, i.e. every 12th of the 24 fps frames, deduplicated. Qwen3-VL then
    merges the sampled frames in pairs — repeating the last when there is an odd
    number — and a merged pair is labelled with the mean of its two
    timestamps."""
    var indices = List[Int]()
    var cursor = Float64(0.0)
    var stride = Float64(MINIMAX_H3_FPS) / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS
    while Int(_round_half_even(cursor)) < num_frames:
        var slot = Int(_round_half_even(cursor))
        if len(indices) == 0 or slot > indices[len(indices) - 1]:
            indices.append(slot)
        cursor += stride

    var timestamps = List[Float64]()
    for index in range(len(indices)):
        timestamps.append(Float64(index) / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS)
    # Repeat-pad to the temporal patch of 2.
    var pad = (-len(timestamps)) % MINIMAX_H3_QWEN_TEMPORAL_PATCH
    if pad < 0:
        pad += MINIMAX_H3_QWEN_TEMPORAL_PATCH
    for _ in range(pad):
        timestamps.append(timestamps[len(timestamps) - 1])

    var block_timestamps = List[Float64]()
    var index = 0
    while index < len(timestamps):
        var last = index + MINIMAX_H3_QWEN_TEMPORAL_PATCH - 1
        block_timestamps.append((timestamps[index] + timestamps[last]) / 2.0)
        index += MINIMAX_H3_QWEN_TEMPORAL_PATCH

    return MiniMaxH3ConditionerFrames(indices^, block_timestamps^)


def _frame_position_grid(
    latent_height: Int, latent_width: Int, patch_h: Int, patch_w: Int
) -> List[Float64]:
    """The (h, w) rotary coordinates of one latent frame, flattened row-major as
    pairs: [h0, w0, h1, w1, ...]."""
    var sqrt_area = sqrt(Float64(latent_height * latent_width))
    var height_grid = minimax_h3_spatial_position_grid(latent_height, patch_h, sqrt_area)
    var width_grid = minimax_h3_spatial_position_grid(latent_width, patch_w, sqrt_area)
    var out = List[Float64]()
    for h in range(len(height_grid)):
        for w in range(len(width_grid)):
            out.append(height_grid[h])
            out.append(width_grid[w])
    return out^


def _width_extremes(latent_height: Int, latent_width: Int, patch_w: Int) -> List[Float64]:
    """The first and last coordinate of a block's own width grid — the two
    values its audio rows are pinned to."""
    var sqrt_area = sqrt(Float64(latent_height * latent_width))
    var width_grid = minimax_h3_spatial_position_grid(latent_width, patch_w, sqrt_area)
    return [width_grid[0], width_grid[len(width_grid) - 1]]


def _fill_audio_positions(
    mut position_ids: List[Float64],
    row_start: Int,
    num_audio_latents: Int,
    rotary_time: Float64,
    width_low: Float64,
    width_high: Float64,
):
    """Place one channel-major audio block. Audio rows carry no height
    coordinate and are pinned to the two extremes of the width grid of THEIR OWN
    block — the target grid for a standalone audio reference, the video's own
    grid for a soundtrack."""
    var rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    for row in range(rows):
        var base = 3 * (row_start + row)
        position_ids[base] = rotary_time + Float64(row % num_audio_latents)
        position_ids[base + 2] = width_low if row < num_audio_latents else width_high


def minimax_h3_build_ref2va_packed_sequence(
    text_token_tags: List[Int],
    references: List[MiniMaxH3PreparedReference],
    num_latent_frames: Int,
    latent_height: Int,
    latent_width: Int,
    num_audio_latents: Int,
    patch_h: Int,
    patch_w: Int,
) raises -> MiniMaxH3PackedSequence:
    """Build the [text | reference blocks | target audio | target video] layout
    of the ref2va task."""
    var num_text_tokens = len(text_token_tags)
    var target_rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    var num_target_video_rows = num_latent_frames * target_rows_per_frame
    var num_target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS

    var num_reference_video_rows = 0
    var num_reference_audio_rows = 0
    for i in range(len(references)):
        if references[i].kind != MINIMAX_H3_REF_AUDIO:
            num_reference_video_rows += references[i].num_video_rows()
        num_reference_audio_rows += references[i].num_audio_rows()

    var sequence_length = (
        num_text_tokens
        + num_reference_video_rows
        + num_reference_audio_rows
        + num_target_audio_rows
        + num_target_video_rows
    )

    var position_ids = List[Float64]()
    for _ in range(sequence_length * 3):
        position_ids.append(Float64(0.0))
    for i in range(num_text_tokens):
        position_ids[3 * i] = Float64(i)

    var target_frame_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    var target_width = _width_extremes(latent_height, latent_width, patch_w)

    # Reference blocks, in request order. `rotary_time` is the shared
    # audio/video clock: it starts where the text rows end and every block
    # pushes it forward by the time that block occupies.
    var video_indices = List[Int]()
    var audio_indices = List[Int]()
    var cursor = num_text_tokens
    var rotary_time = Float64(num_text_tokens)

    for i in range(len(references)):
        ref ref_block = references[i]
        if ref_block.kind == MINIMAX_H3_REF_IMAGE:
            var rows = ref_block.num_video_rows()
            var grid = _frame_position_grid(
                ref_block.latent_height, ref_block.latent_width, patch_h, patch_w
            )
            for row in range(rows):
                var base = 3 * (cursor + row)
                position_ids[base] = rotary_time
                position_ids[base + 1] = grid[2 * row]
                position_ids[base + 2] = grid[2 * row + 1]
                video_indices.append(cursor + row)
            cursor += rows
            # An image is a single frame and takes ONE integer rotary slot, not
            # a latent frame's 5/3 units.
            rotary_time += 1.0
        elif ref_block.kind == MINIMAX_H3_REF_AUDIO:
            var rows = ref_block.num_audio_rows()
            _fill_audio_positions(
                position_ids,
                cursor,
                ref_block.num_audio_latents,
                rotary_time,
                target_width[0],
                target_width[1],
            )
            for row in range(rows):
                audio_indices.append(cursor + row)
            cursor += rows
            rotary_time += Float64(ref_block.num_audio_latents)
        elif ref_block.kind == MINIMAX_H3_REF_VIDEO:
            # A video reference's soundtrack rows pack immediately BEFORE its
            # video rows and share their origin, so the two are rotary-aligned
            # exactly as the generated audio and video are.
            var audio_rows = ref_block.num_audio_rows()
            var video_rows = ref_block.num_video_rows()
            var own_width = _width_extremes(
                ref_block.latent_height, ref_block.latent_width, patch_w
            )
            if audio_rows > 0:
                _fill_audio_positions(
                    position_ids,
                    cursor,
                    ref_block.num_audio_latents,
                    rotary_time,
                    own_width[0],
                    own_width[1],
                )
            for row in range(audio_rows):
                audio_indices.append(cursor + row)
            var video_start = cursor + audio_rows

            var grid = _frame_position_grid(
                ref_block.latent_height, ref_block.latent_width, patch_h, patch_w
            )
            var rows_per_frame = (ref_block.latent_height // patch_h) * (
                ref_block.latent_width // patch_w
            )
            var frame_time = minimax_h3_temporal_position_grid(
                ref_block.num_latent_frames, rotary_time
            )
            for frame in range(ref_block.num_latent_frames):
                for row in range(rows_per_frame):
                    var index = video_start + frame * rows_per_frame + row
                    var base = 3 * index
                    position_ids[base] = frame_time[frame]
                    position_ids[base + 1] = grid[2 * row]
                    position_ids[base + 2] = grid[2 * row + 1]
                    video_indices.append(index)
            cursor = video_start + video_rows

            var audio_span = Float64(ref_block.num_audio_latents)
            var video_span = minimax_h3_temporal_position_span_sequential(
                ref_block.num_latent_frames
            )
            rotary_time += audio_span if audio_span > video_span else video_span
        else:
            raise Error("MiniMax-H3: a reference must be an image, a video or an audio")

    # The generated rows. Target audio and target video share the origin the
    # reference blocks left behind.
    var audio_start = cursor
    var video_start = audio_start + num_target_audio_rows
    _fill_audio_positions(
        position_ids,
        audio_start,
        num_audio_latents,
        rotary_time,
        target_width[0],
        target_width[1],
    )
    var target_frame_time = minimax_h3_temporal_position_grid(num_latent_frames, rotary_time)
    for frame in range(num_latent_frames):
        for row in range(target_rows_per_frame):
            var base = 3 * (video_start + frame * target_rows_per_frame + row)
            position_ids[base] = target_frame_time[frame]
            position_ids[base + 1] = target_frame_grid[2 * row]
            position_ids[base + 2] = target_frame_grid[2 * row + 1]

    for i in range(audio_start, video_start):
        audio_indices.append(i)
    for i in range(video_start, sequence_length):
        video_indices.append(i)
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
        num_reference_video_rows,
        num_reference_audio_rows,
    )


def minimax_h3_build_ref2va_motion_context_packed_sequence(
    text_token_tags: List[Int],
    references: List[MiniMaxH3PreparedReference],
    num_latent_frames: Int,
    latent_height: Int,
    latent_width: Int,
    num_audio_latents: Int,
    patch_h: Int,
    patch_w: Int,
    context_frames: Int,
    source_audio_overhang: Float64,
) raises -> MiniMaxH3PackedSequence:
    """Build Ref2VA references and native A/V continuation in one document.

    Physical rows follow the coexistence order H3 consumes:

      [text | fixed motion video | references | fixed motion audio |
       target audio | target video]

    Ordinary references retain their stock coordinates.  The motion-audio
    block still advances the reference cursor, but its rows are translated
    onto the target head; fixed motion-video rows land on that same shifted
    target timeline.  This preserves both identity/style reference distance
    from text and the previous clip's synchronized A/V join.
    """
    if source_audio_overhang < -0.5 or source_audio_overhang > 0.5:
        raise Error(
            "MiniMax-H3 source audio rounding residual must be in [-0.5,0.5]"
        )
    var base = minimax_h3_build_ref2va_packed_sequence(
        text_token_tags,
        references,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        patch_h,
        patch_w,
    )
    var rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    var context_steps = minimax_h3_motion_context_steps(context_frames)
    var context_audio_latents = minimax_h3_motion_context_audio_latents(
        context_frames
    )
    var motion_video_rows = context_steps * rows_per_frame
    var motion_audio_rows = context_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    var target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    var target_video_rows = num_latent_frames * rows_per_frame
    var base_target_video_start = base.sequence_length - target_video_rows
    var base_target_audio_start = base_target_video_start - target_audio_rows
    var text_rows = len(text_token_tags)
    if base_target_audio_start < text_rows or target_audio_rows <= 0:
        raise Error("MiniMax-H3 Ref2VA motion-context target geometry mismatch")

    var reference_rows = base_target_audio_start - text_rows
    var motion_video_start = text_rows
    var reference_start = motion_video_start + motion_video_rows
    var motion_audio_start = reference_start + reference_rows
    var target_audio_start = motion_audio_start + motion_audio_rows
    var target_video_start = target_audio_start + target_audio_rows
    var sequence_length = target_video_start + target_video_rows

    var position_ids = List[Float64]()
    var token_tags = List[Int]()
    for _ in range(sequence_length * 3):
        position_ids.append(Float64(0.0))
    for _ in range(sequence_length):
        token_tags.append(MINIMAX_H3_VIDEO_TAG)

    # Text stays byte-for-byte equivalent to stock Ref2VA.
    for row in range(text_rows):
        for axis in range(3):
            position_ids[3 * row + axis] = base.position_ids[3 * row + axis]
        token_tags[row] = base.token_tags[row]

    # The appended motion-audio reference advances the target origin by its
    # latent span even though its own rows are relocated onto the target head.
    var base_target_origin = base.position_ids[3 * base_target_video_start]
    var target_origin = base_target_origin + Float64(context_audio_latents)
    var offsets = minimax_h3_motion_context_step_offsets(context_steps)
    for frame in range(context_steps):
        var frame_time = (
            target_origin + ROPE_FRAME_RESCALE * Float64(offsets[frame])
        )
        for spatial in range(rows_per_frame):
            var dst = motion_video_start + frame * rows_per_frame + spatial
            var src = base_target_video_start + spatial
            position_ids[3 * dst] = frame_time
            position_ids[3 * dst + 1] = base.position_ids[3 * src + 1]
            position_ids[3 * dst + 2] = base.position_ids[3 * src + 2]
            token_tags[dst] = MINIMAX_H3_VIDEO_TAG

    # Ordered identity/style/video/audio references keep the stock clock.
    for row in range(reference_rows):
        var src = text_rows + row
        var dst = reference_start + row
        for axis in range(3):
            position_ids[3 * dst + axis] = base.position_ids[3 * src + axis]
        token_tags[dst] = base.token_tags[src]

    # End-align fixed audio with the pinned video window on the shifted target
    # timeline, preserving the source clip's signed audio-grid residual.
    var raw_end = (
        ROPE_FRAME_RESCALE * Float64(context_frames) + source_audio_overhang
    )
    var end_coord = Float64(round(raw_end))
    var motion_audio_time = (
        target_origin + end_coord - Float64(context_audio_latents)
    )
    var width_low = base.position_ids[3 * base_target_audio_start + 2]
    var width_high = base.position_ids[
        3 * (base_target_audio_start + num_audio_latents) + 2
    ]
    for row in range(motion_audio_rows):
        var dst = motion_audio_start + row
        position_ids[3 * dst] = (
            motion_audio_time + Float64(row % context_audio_latents)
        )
        position_ids[3 * dst + 2] = (
            width_low if row < context_audio_latents else width_high
        )
        token_tags[dst] = MINIMAX_H3_AUDIO_TAG

    # Target rows retain stock spatial coordinates while the appended motion
    # audio span translates only their time coordinate.
    for row in range(base_target_audio_start, base.sequence_length):
        var dst = row + motion_video_rows + motion_audio_rows
        position_ids[3 * dst] = (
            base.position_ids[3 * row] + Float64(context_audio_latents)
        )
        position_ids[3 * dst + 1] = base.position_ids[3 * row + 1]
        position_ids[3 * dst + 2] = base.position_ids[3 * row + 2]
        token_tags[dst] = base.token_tags[row]

    # Modality index order is the payload contract: motion video first,
    # ordinary reference media next, motion audio last among condition rows,
    # then each target suffix.
    var video_indices = List[Int]()
    for row in range(motion_video_rows):
        video_indices.append(motion_video_start + row)
    for i in range(base.num_condition_video_rows):
        video_indices.append(base.video_indices[i] + motion_video_rows)
    for i in range(base.num_condition_video_rows, len(base.video_indices)):
        video_indices.append(
            base.video_indices[i] + motion_video_rows + motion_audio_rows
        )

    var audio_indices = List[Int]()
    for i in range(base.num_condition_audio_rows):
        audio_indices.append(base.audio_indices[i] + motion_video_rows)
    for row in range(motion_audio_rows):
        audio_indices.append(motion_audio_start + row)
    for i in range(base.num_condition_audio_rows, len(base.audio_indices)):
        audio_indices.append(
            base.audio_indices[i] + motion_video_rows + motion_audio_rows
        )

    var text_indices = List[Int]()
    for row in range(text_rows):
        text_indices.append(row)

    return MiniMaxH3PackedSequence(
        sequence_length,
        position_ids^,
        token_tags^,
        video_indices^,
        audio_indices^,
        text_indices^,
        base.num_condition_video_rows + motion_video_rows,
        base.num_condition_audio_rows + motion_audio_rows,
    )
