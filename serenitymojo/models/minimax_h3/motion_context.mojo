# Native MiniMax-H3 latent-tail continuation.
#
# A continuation request keeps a short tail of the previous clip's generated
# video and audio latents fixed while sampling a longer target whose head
# overlaps that tail.  The fixed rows use H3's existing condition timesteps;
# only their packed positions differ from ordinary first/last keyframes and
# reference media.  This module owns that model-specific layout and the small
# continuation artifact shared by all H3 runners.

from std.collections import List
from std.gpu.host import DeviceContext
from std.math import round, sqrt
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.tensor_algebra import concat, slice
from serenitymojo.models.dit.minimax_h3_sampling import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    ROPE_FRAME_RESCALE,
    MiniMaxH3SamplingGeometry,
    minimax_h3_sampling_spatial_grid,
    minimax_h3_sampling_temporal_grid,
)

comptime MINIMAX_H3_MOTION_CONTEXT_SCHEMA = Float32(1.0)
comptime MINIMAX_H3_MOTION_CONTEXT_FPS = 24
comptime MINIMAX_H3_MOTION_CONTEXT_AUDIO_HZ = 40
comptime MINIMAX_H3_MOTION_CONTEXT_META_LEN = 10


def minimax_h3_motion_context_steps(context_frames: Int) raises -> Int:
    """Return the phase-zero video latent steps for an admitted window."""
    if context_frames == 5:
        return 2
    if context_frames == 22:
        return 7
    if context_frames == 39:
        return 12
    raise Error("MiniMax-H3 motion context must be 5, 22, or 39 frames")


def minimax_h3_motion_context_pixel_frames(latent_steps: Int) -> Int:
    """Pixel frames covered by phase-zero H3 temporal latent steps."""
    var frames = 0
    for step in range(latent_steps):
        frames += 1 if step % 5 == 0 else 4
    return frames


def minimax_h3_motion_context_endpoint_steps(
    requested_pixel_frames: Int, available_latent_steps: Int
) raises -> Int:
    """Choose the native H3 endpoint nearest a delivered pixel-frame cut.

    H3 video endpoints lie on ``5 + 17*n`` pixel frames.  A request can be
    padded beyond its delivered duration for the VAE/DiT lattice, so blindly
    saving the final latent rows can hide as many as 16 frames at a Continue
    join.  Select the closest available endpoint instead; this bounds the
    unavoidable cut error to eight frames and is two/three frames for the
    common 15-second source/continuation geometries.
    """
    if available_latent_steps < 2 \
            or (available_latent_steps - 2) % 5 != 0:
        raise Error("MiniMax-H3 available endpoint is not phase-zero aligned")
    if requested_pixel_frames < 1:
        raise Error("MiniMax-H3 continuation endpoint must be positive")
    var available_groups = (available_latent_steps - 2) // 5
    var lower_groups = 0
    if requested_pixel_frames > 5:
        lower_groups = (requested_pixel_frames - 5) // 17
    if lower_groups > available_groups:
        lower_groups = available_groups
    var best_groups = lower_groups
    var lower_frames = 5 + 17 * lower_groups
    if lower_groups < available_groups:
        var upper_groups = lower_groups + 1
        var upper_frames = 5 + 17 * upper_groups
        if upper_frames - requested_pixel_frames \
                < requested_pixel_frames - lower_frames:
            best_groups = upper_groups
    return 2 + 5 * best_groups


def minimax_h3_motion_context_audio_latents(context_frames: Int) -> Int:
    return Int(round(
        Float64(context_frames)
        / Float64(MINIMAX_H3_MOTION_CONTEXT_FPS)
        * Float64(MINIMAX_H3_MOTION_CONTEXT_AUDIO_HZ)
    ))


def minimax_h3_motion_context_step_offsets(latent_steps: Int) -> List[Int]:
    """Pixel-frame offset at which each phase-zero latent step begins."""
    var offsets = List[Int]()
    var running = 0
    for step in range(latent_steps):
        offsets.append(running)
        running += 1 if step % 5 == 0 else 4
    return offsets^


def minimax_h3_build_motion_context_geometry(
    text_token_tags: List[Int],
    num_latent_frames: Int,
    latent_height: Int,
    latent_width: Int,
    num_audio_latents: Int,
    patch_h: Int,
    patch_w: Int,
    context_frames: Int,
    source_audio_overhang: Float64,
) raises -> MiniMaxH3SamplingGeometry:
    """Build [text | fixed video | fixed audio | target audio | target video].

    The fixed video steps are placed at the beginning of the target timeline.
    Fixed audio is channel-major and end-aligned to the same overlap endpoint,
    including the source clip's fractional final audio-grid overhang.
    """
    var context_steps = minimax_h3_motion_context_steps(context_frames)
    if minimax_h3_motion_context_pixel_frames(context_steps) != context_frames:
        raise Error("MiniMax-H3 motion-context frame/latent grid drift")
    if source_audio_overhang < -0.5 or source_audio_overhang > 0.5:
        raise Error(
            "MiniMax-H3 source audio rounding residual must be in [-0.5,0.5]"
        )

    var rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    var num_text_tokens = len(text_token_tags)
    var num_condition_video_rows = context_steps * rows_per_frame
    var context_audio_latents = minimax_h3_motion_context_audio_latents(context_frames)
    var num_condition_audio_rows = context_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    var num_target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    var num_target_video_rows = num_latent_frames * rows_per_frame
    var sequence_length = (
        num_text_tokens
        + num_condition_video_rows
        + num_condition_audio_rows
        + num_target_audio_rows
        + num_target_video_rows
    )

    var condition_video_start = num_text_tokens
    var condition_audio_start = condition_video_start + num_condition_video_rows
    var target_audio_start = condition_audio_start + num_condition_audio_rows
    var target_video_start = target_audio_start + num_target_audio_rows
    var text_time = Float64(num_text_tokens)
    # Stock H3 first lays an audio reference in its own rotary span and
    # advances the target cursor by that span. Motion Context then moves the
    # reference rows onto the target timeline but deliberately leaves the
    # old span vacant. Preserve that target-origin shift here: relative A/V
    # placement alone is insufficient because the model also sees each row's
    # rotary distance from text.
    var target_origin = text_time + Float64(context_audio_latents)

    var position_ids = List[Float64]()
    for _ in range(sequence_length * 3):
        position_ids.append(Float64(0.0))
    for i in range(num_text_tokens):
        position_ids[3 * i] = Float64(i)

    var sqrt_area = sqrt(Float64(latent_height * latent_width))
    var height_grid = minimax_h3_sampling_spatial_grid(
        latent_height, patch_h, sqrt_area
    )
    var width_grid = minimax_h3_sampling_spatial_grid(
        latent_width, patch_w, sqrt_area
    )
    var frame_h = List[Float64]()
    var frame_w = List[Float64]()
    for h in range(len(height_grid)):
        for w in range(len(width_grid)):
            frame_h.append(height_grid[h])
            frame_w.append(width_grid[w])

    var offsets = minimax_h3_motion_context_step_offsets(context_steps)
    for frame in range(context_steps):
        var frame_time = (
            target_origin + ROPE_FRAME_RESCALE * Float64(offsets[frame])
        )
        for row in range(rows_per_frame):
            var base = 3 * (
                condition_video_start + frame * rows_per_frame + row
            )
            position_ids[base] = frame_time
            position_ids[base + 1] = frame_h[row]
            position_ids[base + 2] = frame_w[row]

    # H3's source audio grid is rounded to integer 40-Hz steps. Preserve the
    # signed fractional source endpoint, then snap the overlap endpoint to the target
    # audio grid before laying out a channel-major window backwards from it.
    var raw_end = (
        ROPE_FRAME_RESCALE * Float64(context_frames) + source_audio_overhang
    )
    var end_coord = Float64(round(raw_end))
    var condition_audio_time = (
        target_origin + end_coord - Float64(context_audio_latents)
    )
    var width_low = width_grid[0]
    var width_high = width_grid[len(width_grid) - 1]
    for row in range(num_condition_audio_rows):
        var base = 3 * (condition_audio_start + row)
        position_ids[base] = (
            condition_audio_time + Float64(row % context_audio_latents)
        )
        position_ids[base + 2] = (
            width_low if row < context_audio_latents else width_high
        )

    for row in range(num_target_audio_rows):
        var base = 3 * (target_audio_start + row)
        position_ids[base] = target_origin + Float64(row % num_audio_latents)
        position_ids[base + 2] = (
            width_low if row < num_audio_latents else width_high
        )

    var target_times = minimax_h3_sampling_temporal_grid(
        num_latent_frames, target_origin
    )
    for frame in range(num_latent_frames):
        for row in range(rows_per_frame):
            var base = 3 * (
                target_video_start + frame * rows_per_frame + row
            )
            position_ids[base] = target_times[frame]
            position_ids[base + 1] = frame_h[row]
            position_ids[base + 2] = frame_w[row]

    var video_indices = List[Int]()
    for i in range(condition_video_start, condition_audio_start):
        video_indices.append(i)
    for i in range(target_video_start, sequence_length):
        video_indices.append(i)
    var audio_indices = List[Int]()
    for i in range(condition_audio_start, target_audio_start):
        audio_indices.append(i)
    for i in range(target_audio_start, target_video_start):
        audio_indices.append(i)
    var text_indices = List[Int]()
    for i in range(num_text_tokens):
        text_indices.append(i)

    var token_tags = List[Int]()
    for _ in range(sequence_length):
        token_tags.append(MINIMAX_H3_VIDEO_TAG)
    for i in range(num_text_tokens):
        token_tags[i] = text_token_tags[i]
    for i in range(len(audio_indices)):
        token_tags[audio_indices[i]] = MINIMAX_H3_AUDIO_TAG
    for i in range(len(video_indices)):
        token_tags[video_indices[i]] = MINIMAX_H3_VIDEO_TAG

    return MiniMaxH3SamplingGeometry(
        sequence_length,
        position_ids^,
        token_tags^,
        video_indices^,
        audio_indices^,
        text_indices^,
        num_condition_video_rows,
        num_condition_audio_rows,
    )


struct MiniMaxH3LoadedMotionContext(Movable):
    var video_rows: Tensor
    var audio_rows: Tensor
    var source_video_latent_frames: Int
    var source_audio_latents: Int
    var source_audio_overhang: Float64

    def __init__(
        out self,
        var video_rows: Tensor,
        var audio_rows: Tensor,
        source_video_latent_frames: Int,
        source_audio_latents: Int,
        source_audio_overhang: Float64,
    ):
        self.video_rows = video_rows^
        self.audio_rows = audio_rows^
        self.source_video_latent_frames = source_video_latent_frames
        self.source_audio_latents = source_audio_latents
        self.source_audio_overhang = source_audio_overhang

    def __del__(deinit self):
        pass


def minimax_h3_preflight_motion_context(
    path: String,
    rows_per_frame: Int,
    video_patch_dim: Int,
    audio_latent_dim: Int,
    context_frames: Int,
) raises:
    """Header-only validation; performs no GPU allocation."""
    var context_steps = minimax_h3_motion_context_steps(context_frames)
    var context_audio_latents = minimax_h3_motion_context_audio_latents(context_frames)
    var st = SafeTensors.open(path)
    var video_name = (
        String("video_context_rows")
        if st.has_tensor(String("video_context_rows"))
        else String("video_state_rows")
    )
    var audio_name = (
        String("audio_context_rows")
        if st.has_tensor(String("audio_context_rows"))
        else String("audio_state_rows")
    )
    if not st.has_tensor(video_name) or not st.has_tensor(audio_name):
        raise Error(
            "MiniMax-H3 motion context needs video/audio context or state rows"
        )
    var vinfo = st.tensor_info(video_name)
    var ainfo = st.tensor_info(audio_name)
    if (
        vinfo.dtype != STDtype.F32
        or len(vinfo.shape) != 2
        or vinfo.shape[1] != video_patch_dim
        or vinfo.shape[0] < context_steps * rows_per_frame
        or vinfo.shape[0] % rows_per_frame != 0
    ):
        raise Error("MiniMax-H3 motion-context video row dtype/shape mismatch")
    if (
        ainfo.dtype != STDtype.F32
        or len(ainfo.shape) != 2
        or ainfo.shape[1] != audio_latent_dim
        or ainfo.shape[0] % MINIMAX_H3_AUDIO_CHANNELS != 0
        or ainfo.shape[0]
            < context_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    ):
        raise Error("MiniMax-H3 motion-context audio row dtype/shape mismatch")
    if st.has_tensor(String("video_context_rows")):
        if not st.has_tensor(String("motion_context_meta")):
            raise Error("MiniMax-H3 compact motion context is missing metadata")
        var minfo = st.tensor_info(String("motion_context_meta"))
        if (
            minfo.dtype != STDtype.F32
            or len(minfo.shape) != 1
            or minfo.shape[0] != MINIMAX_H3_MOTION_CONTEXT_META_LEN
        ):
            raise Error("MiniMax-H3 motion-context metadata shape mismatch")


def minimax_h3_motion_context_source_audio_overhang(
    path: String,
    rows_per_frame: Int,
    expected_width: Int,
    expected_height: Int,
) raises -> Float64:
    """Read the source A/V clock residual without allocating a GPU context."""
    var st = SafeTensors.open(path)
    var compact = st.has_tensor(String("video_context_rows"))
    var video_name = (
        String("video_context_rows") if compact else String("video_state_rows")
    )
    var audio_name = (
        String("audio_context_rows") if compact else String("audio_state_rows")
    )
    var vinfo = st.tensor_info(video_name)
    var ainfo = st.tensor_info(audio_name)
    var source_video_latent_frames = vinfo.shape[0] // rows_per_frame
    var source_audio_latents = ainfo.shape[0] // MINIMAX_H3_AUDIO_CHANNELS
    if compact:
        var minfo = st.tensor_info(String("motion_context_meta"))
        var bytes = st.tensor_bytes(String("motion_context_meta"))
        if minfo.dtype != STDtype.F32 or minfo.size != 4 * MINIMAX_H3_MOTION_CONTEXT_META_LEN:
            raise Error("MiniMax-H3 motion-context metadata byte size mismatch")
        var meta = bytes.unsafe_ptr().bitcast[Float32]()
        if meta[0] != MINIMAX_H3_MOTION_CONTEXT_SCHEMA:
            raise Error("MiniMax-H3 motion-context metadata schema mismatch")
        if Int(meta[1]) != expected_width or Int(meta[2]) != expected_height:
            raise Error("MiniMax-H3 motion context must keep its source resolution")
        source_video_latent_frames = Int(meta[3])
        source_audio_latents = Int(meta[4])
        if (
            Int(meta[6]) != vinfo.shape[0] // rows_per_frame
            or Int(meta[7]) != ainfo.shape[0] // MINIMAX_H3_AUDIO_CHANNELS
            or Int(meta[8]) != MINIMAX_H3_MOTION_CONTEXT_FPS
            or Int(meta[9]) != rows_per_frame
        ):
            raise Error("MiniMax-H3 motion-context metadata/header mismatch")
    var source_pixel_frames = minimax_h3_motion_context_pixel_frames(
        source_video_latent_frames
    )
    var overhang = (
        Float64(source_audio_latents)
        - ROPE_FRAME_RESCALE * Float64(source_pixel_frames)
    )
    if overhang < 0.0 and overhang > -1.0e-9:
        overhang = 0.0
    if overhang < -0.5 or overhang > 0.5:
        raise Error("MiniMax-H3 source audio grid has invalid rounding residual")
    return overhang


def minimax_h3_load_motion_context(
    path: String,
    rows_per_frame: Int,
    video_patch_dim: Int,
    audio_latent_dim: Int,
    context_frames: Int,
    expected_width: Int,
    expected_height: Int,
    ctx: DeviceContext,
) raises -> MiniMaxH3LoadedMotionContext:
    """Load exactly the requested phase-aligned tail onto the GPU."""
    minimax_h3_preflight_motion_context(
        path, rows_per_frame, video_patch_dim, audio_latent_dim,
        context_frames,
    )
    var context_steps = minimax_h3_motion_context_steps(context_frames)
    var context_audio_latents = minimax_h3_motion_context_audio_latents(context_frames)
    var st = SafeTensors.open(path)
    var compact = st.has_tensor(String("video_context_rows"))
    var video_name = (
        String("video_context_rows") if compact else String("video_state_rows")
    )
    var audio_name = (
        String("audio_context_rows") if compact else String("audio_state_rows")
    )
    var vinfo = st.tensor_info(video_name)
    var ainfo = st.tensor_info(audio_name)
    var video_all = Tensor.from_view(
        from_parts(
            vinfo.dtype, vinfo.shape.copy(), st.tensor_bytes(video_name)
        ),
        ctx,
    )
    var audio_all = Tensor.from_view(
        from_parts(
            ainfo.dtype, ainfo.shape.copy(), st.tensor_bytes(audio_name)
        ),
        ctx,
    )

    var stored_video_steps = vinfo.shape[0] // rows_per_frame
    var stored_audio_latents = ainfo.shape[0] // MINIMAX_H3_AUDIO_CHANNELS
    var source_video_latent_frames = stored_video_steps
    var source_audio_latents = stored_audio_latents
    if compact:
        var minfo = st.tensor_info(String("motion_context_meta"))
        var meta_tensor = Tensor.from_view(
            from_parts(
                minfo.dtype,
                minfo.shape.copy(),
                st.tensor_bytes(String("motion_context_meta")),
            ),
            ctx,
        )
        var meta = meta_tensor.to_host(ctx)
        if len(meta) != MINIMAX_H3_MOTION_CONTEXT_META_LEN \
                or meta[0] != MINIMAX_H3_MOTION_CONTEXT_SCHEMA:
            raise Error("MiniMax-H3 motion-context metadata schema mismatch")
        if Int(meta[1]) != expected_width or Int(meta[2]) != expected_height:
            raise Error(
                "MiniMax-H3 motion context must keep its source resolution"
            )
        source_video_latent_frames = Int(meta[3])
        source_audio_latents = Int(meta[4])
        if Int(meta[6]) != stored_video_steps \
                or Int(meta[7]) != stored_audio_latents \
                or Int(meta[8]) != MINIMAX_H3_MOTION_CONTEXT_FPS \
                or Int(meta[9]) != rows_per_frame:
            raise Error("MiniMax-H3 motion-context metadata/header mismatch")

    if source_video_latent_frames < context_steps \
            or (source_video_latent_frames - context_steps) % 5 != 0:
        raise Error(
            "MiniMax-H3 motion-context video tail is not phase-zero aligned"
        )
    var source_pixel_frames = minimax_h3_motion_context_pixel_frames(
        source_video_latent_frames
    )
    var source_audio_overhang = (
        Float64(source_audio_latents)
        - ROPE_FRAME_RESCALE * Float64(source_pixel_frames)
    )
    if source_audio_overhang < 0.0 and source_audio_overhang > -1.0e-9:
        source_audio_overhang = 0.0
    if source_audio_overhang < -0.5 or source_audio_overhang > 0.5:
        raise Error("MiniMax-H3 source audio grid has invalid rounding residual")

    var wanted_video_rows = context_steps * rows_per_frame
    var video_tail = slice(
        video_all, 0, vinfo.shape[0] - wanted_video_rows,
        wanted_video_rows, ctx,
    )
    var left_tail = slice(
        audio_all, 0,
        stored_audio_latents - context_audio_latents,
        context_audio_latents, ctx,
    )
    var right_tail = slice(
        audio_all, 0,
        2 * stored_audio_latents - context_audio_latents,
        context_audio_latents, ctx,
    )
    var audio_tail = concat(0, ctx, left_tail, right_tail)
    return MiniMaxH3LoadedMotionContext(
        video_tail^,
        audio_tail^,
        source_video_latent_frames,
        source_audio_latents,
        source_audio_overhang,
    )


def minimax_h3_save_motion_context_tail(
    video_state: Tensor,
    audio_state: Tensor,
    num_latent_frames: Int,
    num_audio_latents: Int,
    width: Int,
    height: Int,
    out_path: String,
    ctx: DeviceContext,
) raises -> Int:
    """Persist the largest supported compact continuation tail available."""
    return minimax_h3_save_motion_context_tail_at_pixel_frame(
        video_state,
        audio_state,
        num_latent_frames,
        num_audio_latents,
        minimax_h3_motion_context_pixel_frames(num_latent_frames),
        width,
        height,
        out_path,
        ctx,
    )


def minimax_h3_save_motion_context_tail_at_pixel_frame(
    video_state: Tensor,
    audio_state: Tensor,
    num_latent_frames: Int,
    num_audio_latents: Int,
    continuation_end_pixel_frames: Int,
    width: Int,
    height: Int,
    out_path: String,
    ctx: DeviceContext,
) raises -> Int:
    """Persist a compact tail ending nearest the delivered frame boundary."""
    if len(video_state.shape()) != 2 or len(audio_state.shape()) != 2:
        raise Error("MiniMax-H3 motion-context source rows must be rank-2")
    if num_latent_frames < 2 or video_state.shape()[0] % num_latent_frames != 0:
        raise Error("MiniMax-H3 motion-context source video geometry mismatch")
    if audio_state.shape()[0] != 2 * num_audio_latents:
        raise Error("MiniMax-H3 motion-context source audio geometry mismatch")
    var rows_per_frame = video_state.shape()[0] // num_latent_frames
    var endpoint_steps = minimax_h3_motion_context_endpoint_steps(
        continuation_end_pixel_frames, num_latent_frames
    )
    var endpoint_pixel_frames = minimax_h3_motion_context_pixel_frames(
        endpoint_steps
    )
    var endpoint_audio_latents = Int(round(
        Float64(endpoint_pixel_frames)
        / Float64(MINIMAX_H3_MOTION_CONTEXT_FPS)
        * Float64(MINIMAX_H3_MOTION_CONTEXT_AUDIO_HZ)
    ))
    if endpoint_audio_latents > num_audio_latents:
        raise Error("MiniMax-H3 continuation endpoint exceeds source audio")
    var saved_frames: Int
    var saved_steps: Int
    if endpoint_steps >= 12 and (endpoint_steps - 12) % 5 == 0:
        saved_frames = 39
        saved_steps = 12
    elif endpoint_steps >= 7 and (endpoint_steps - 7) % 5 == 0:
        saved_frames = 22
        saved_steps = 7
    elif endpoint_steps >= 2 and (endpoint_steps - 2) % 5 == 0:
        saved_frames = 5
        saved_steps = 2
    else:
        raise Error("MiniMax-H3 source latent cannot produce a phase-zero tail")
    var saved_audio_latents = minimax_h3_motion_context_audio_latents(saved_frames)
    if saved_audio_latents > endpoint_audio_latents:
        raise Error("MiniMax-H3 source audio is shorter than its video tail")

    var saved_video_rows = saved_steps * rows_per_frame
    var endpoint_video_rows = endpoint_steps * rows_per_frame
    var video_tail = slice(
        video_state, 0, endpoint_video_rows - saved_video_rows,
        saved_video_rows, ctx,
    )
    var left_tail = slice(
        audio_state, 0, endpoint_audio_latents - saved_audio_latents,
        saved_audio_latents, ctx,
    )
    var right_tail = slice(
        audio_state, 0,
        num_audio_latents + endpoint_audio_latents - saved_audio_latents,
        saved_audio_latents, ctx,
    )
    var audio_tail = concat(0, ctx, left_tail, right_tail)
    var meta_values: List[Float32] = [
        MINIMAX_H3_MOTION_CONTEXT_SCHEMA,
        Float32(width),
        Float32(height),
        Float32(endpoint_steps),
        Float32(endpoint_audio_latents),
        Float32(saved_frames),
        Float32(saved_steps),
        Float32(saved_audio_latents),
        Float32(MINIMAX_H3_MOTION_CONTEXT_FPS),
        Float32(rows_per_frame),
    ]
    var meta_shape: List[Int] = [MINIMAX_H3_MOTION_CONTEXT_META_LEN]
    var meta = Tensor.from_host(meta_values, meta_shape^, STDtype.F32, ctx)

    var names = List[String]()
    names.append(String("video_context_rows"))
    names.append(String("audio_context_rows"))
    names.append(String("motion_context_meta"))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer[Tensor](video_tail^))
    tensors.append(ArcPointer[Tensor](audio_tail^))
    tensors.append(ArcPointer[Tensor](meta^))
    save_safetensors(names, tensors, out_path, ctx)
    return saved_frames
