# Pure-Mojo MiniMax-H3 long-form timeline planning.
#
# This is the native counterpart of SerenityFlow's MiniMaxH3EndlessLatent and
# MiniMaxH3EndlessSampler planning contract.  It deliberately owns no model
# weights and allocates no tensors: every entry describes one bounded H3 call,
# while the pipeline runner samples the entries serially and persists progress.

from std.collections import List


comptime MINIMAX_H3_ENDLESS_FPS = 24
comptime MINIMAX_H3_ENDLESS_AUDIO_HZ = 40
comptime MINIMAX_H3_ENDLESS_GRID = 17
comptime MINIMAX_H3_ENDLESS_GRID_LEAD = 5
comptime MINIMAX_H3_ENDLESS_MIN_FRAMES = 120
comptime MINIMAX_H3_ENDLESS_MAX_CHUNK_FRAMES = 362


@fieldwise_init
struct MiniMaxH3EndlessChunk(Copyable, Movable):
    var index: Int
    var frame_start: Int
    var frame_end: Int
    var sample_frames: Int
    var new_frames: Int
    var output_trim_frames: Int
    var sample_video_t: Int
    var boundary_video_t: Int
    var sample_audio_t: Int
    var context_audio_t: Int
    var new_audio_t: Int
    var reference_video_t: Int
    var reference_audio_t: Int


def minimax_h3_endless_is_on_grid(frames: Int) -> Bool:
    return frames >= MINIMAX_H3_ENDLESS_GRID_LEAD and (
        frames % MINIMAX_H3_ENDLESS_GRID
        == MINIMAX_H3_ENDLESS_GRID_LEAD
    )


def minimax_h3_endless_snap_frames_up(frames: Int) -> Int:
    var value = frames
    if value < MINIMAX_H3_ENDLESS_GRID_LEAD:
        value = MINIMAX_H3_ENDLESS_GRID_LEAD
    var remainder = (
        MINIMAX_H3_ENDLESS_GRID_LEAD - value
    ) % MINIMAX_H3_ENDLESS_GRID
    if remainder < 0:
        remainder += MINIMAX_H3_ENDLESS_GRID
    return value + remainder


def minimax_h3_endless_video_latents(frames: Int) raises -> Int:
    if not minimax_h3_endless_is_on_grid(frames):
        raise Error("MiniMax-H3 endless video frames must use the 17k+5 grid")
    return ((frames - MINIMAX_H3_ENDLESS_GRID_LEAD) // 17) * 5 + 2


def minimax_h3_endless_audio_latents(frames: Int) raises -> Int:
    if frames < 0:
        raise Error("MiniMax-H3 endless audio frame boundary cannot be negative")
    # round(frames / 24 * 40) == round(frames * 5 / 3).  The denominator is
    # odd, so no half-even tie exists; (n + 1) // 3 is exact for n >= 0.
    return (frames * 5 + 1) // 3


def minimax_h3_plan_endless_chunks(
    total_frames: Int,
    chunk_frames: Int = 124,
    reference_frames: Int = 22,
    boundary_frames: Int = 5,
) raises -> List[MiniMaxH3EndlessChunk]:
    """Plan bounded serial calls with phase-exact A/V append boundaries.

    The first call contributes all of its frames.  Every continuation carries
    a protected target prefix (`boundary_frames`) and appends only the new
    suffix.  A separate tail (`reference_frames`) is presented to H3 as the
    previous synchronized video/audio reference.
    """
    if total_frames < MINIMAX_H3_ENDLESS_MIN_FRAMES \
            or not minimax_h3_endless_is_on_grid(total_frames):
        raise Error(
            "MiniMax-H3 endless total_frames must be at least 120 and use"
            " the 17k+5 grid"
        )
    if chunk_frames < MINIMAX_H3_ENDLESS_MIN_FRAMES \
            or not minimax_h3_endless_is_on_grid(chunk_frames):
        raise Error(
            "MiniMax-H3 endless chunk_frames must be at least 120 and use"
            " the 17k+5 grid"
        )
    if chunk_frames > MINIMAX_H3_ENDLESS_MAX_CHUNK_FRAMES:
        raise Error("MiniMax-H3 endless chunk exceeds the 15-second H3 limit")
    if not minimax_h3_endless_is_on_grid(reference_frames):
        raise Error("MiniMax-H3 endless reference_frames must use the 17k+5 grid")
    if not minimax_h3_endless_is_on_grid(boundary_frames):
        raise Error("MiniMax-H3 endless boundary_frames must use the 17k+5 grid")
    if reference_frames > chunk_frames:
        raise Error("MiniMax-H3 endless reference exceeds the chunk")
    if boundary_frames >= chunk_frames:
        raise Error("MiniMax-H3 endless boundary must be smaller than the chunk")

    var chunks = List[MiniMaxH3EndlessChunk]()
    var covered = 0
    var global_audio_end = 0
    while covered < total_frames:
        var first = len(chunks) == 0
        var sample_frames: Int
        var new_frames: Int
        var trim_frames: Int
        if first:
            sample_frames = (
                total_frames if total_frames < chunk_frames else chunk_frames
            )
            new_frames = sample_frames
            trim_frames = 0
        else:
            var capacity = chunk_frames - boundary_frames
            var remaining = total_frames - covered
            new_frames = remaining if remaining < capacity else capacity
            sample_frames = boundary_frames + new_frames
            trim_frames = boundary_frames
        if not minimax_h3_endless_is_on_grid(sample_frames):
            raise Error("MiniMax-H3 endless planner lost the 17k+5 phase")

        var sample_audio_t = minimax_h3_endless_audio_latents(sample_frames)
        var next_audio_end = minimax_h3_endless_audio_latents(
            covered + new_frames
        )
        var new_audio_t = next_audio_end - global_audio_end
        var context_audio_t = sample_audio_t - new_audio_t
        if first and context_audio_t != 0:
            raise Error("MiniMax-H3 endless first chunk gained audio context")
        if not first and context_audio_t < 1:
            raise Error("MiniMax-H3 endless continuation lost its audio boundary")

        var reference_audio_t = 0
        if not first:
            var reference_start = covered - reference_frames
            if reference_start < 0:
                reference_start = 0
            reference_audio_t = global_audio_end \
                - minimax_h3_endless_audio_latents(reference_start)
            if reference_audio_t < 1:
                raise Error("MiniMax-H3 endless reference has no audio ticks")

        chunks.append(MiniMaxH3EndlessChunk(
            len(chunks),
            covered,
            covered + new_frames,
            sample_frames,
            new_frames,
            trim_frames,
            minimax_h3_endless_video_latents(sample_frames),
            0 if first else minimax_h3_endless_video_latents(boundary_frames),
            sample_audio_t,
            context_audio_t,
            new_audio_t,
            0 if first else minimax_h3_endless_video_latents(reference_frames),
            reference_audio_t,
        ))
        covered += new_frames
        global_audio_end = next_audio_end

    if covered != total_frames \
            or global_audio_end != minimax_h3_endless_audio_latents(total_frames):
        raise Error("MiniMax-H3 endless planner did not cover the target timeline")
    return chunks^


def minimax_h3_plan_endless_seconds(
    seconds: Int,
    chunk_frames: Int = 124,
    reference_frames: Int = 22,
    boundary_frames: Int = 5,
) raises -> List[MiniMaxH3EndlessChunk]:
    if seconds < 5 or seconds > 3600:
        raise Error("MiniMax-H3 endless duration must be from 5 to 3600 seconds")
    return minimax_h3_plan_endless_chunks(
        minimax_h3_endless_snap_frames_up(seconds * MINIMAX_H3_ENDLESS_FPS),
        chunk_frames,
        reference_frames,
        boundary_frames,
    )
