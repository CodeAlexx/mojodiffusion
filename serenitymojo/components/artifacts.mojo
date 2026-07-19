# components/artifacts.mojo - shared image/video artifact writers.
#
# This keeps frame extraction, PNG path behavior, and frame-sequence muxing out
# of model smokes.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.io.ffi import sys_system
from serenitymojo.ops.tensor_algebra import reshape, slice


def video_frame_path(prefix: String, frame_idx: Int, suffix: String = String(".png")) -> String:
    return prefix + String(frame_idx) + suffix


def ffmpeg_frame_pattern(prefix: String, suffix: String = String(".png")) -> String:
    return prefix + String("%d") + suffix


def shell_quote(s: String) -> String:
    """Single-quote a shell argument. Paths in this project are ASCII."""
    var q = String("'")
    var bytes = s.as_bytes()
    for i in range(s.byte_length()):
        if Int(bytes[i]) == 39:
            q += String("'\\''")
        else:
            q += chr(Int(bytes[i]))
    q += String("'")
    return q


def build_ffmpeg_mux_command(
    path_prefix: String,
    path_suffix: String,
    out_path: String,
    fps: Int = 8,
) raises -> String:
    if fps <= 0:
        raise Error("build_ffmpeg_mux_command: fps must be positive")
    var pattern = ffmpeg_frame_pattern(path_prefix, path_suffix)
    return (
        String("ffmpeg -y -hide_banner -loglevel error -framerate ")
        + String(fps)
        + String(" -start_number 0 -i ")
        + shell_quote(pattern)
        + String(" -c:v libx264 -pix_fmt yuv420p -movflags +faststart ")
        + shell_quote(out_path)
    )


def mux_frame_sequence_mp4(
    path_prefix: String,
    path_suffix: String,
    out_path: String,
    fps: Int = 8,
) raises:
    var cmd = build_ffmpeg_mux_command(path_prefix, path_suffix, out_path, fps)
    var status = sys_system(cmd)
    if status != 0:
        raise Error(
            String("mux_frame_sequence_mp4: ffmpeg failed with raw status ")
            + String(status)
        )


def save_video_frame_png(
    video: Tensor,
    frame_idx: Int,
    out_path: String,
    latent_h: Int,
    latent_w: Int,
    ctx: DeviceContext,
    value_range: ValueRange = ValueRange.SIGNED,
) raises:
    """Save one frame from video [1,3,T,H,W] as [1,3,H,W] PNG."""
    var dims = video.shape()
    if len(dims) != 5:
        raise Error("save_video_frame_png: video must be [1,3,T,H,W]")
    if frame_idx < 0 or frame_idx >= dims[2]:
        raise Error("save_video_frame_png: frame index out of range")
    var h = 16 * latent_h
    var w = 16 * latent_w
    if dims[3] != h or dims[4] != w:
        raise Error("save_video_frame_png: latent shape does not match video frame")
    var frame5 = slice(video, 2, frame_idx, 1, ctx)
    var frame = reshape(frame5, [1, 3, h, w], ctx)
    save_png(frame, out_path, ctx, value_range)


def save_video_frame_pair_png(
    video: Tensor,
    first_path: String,
    last_path: String,
    latent_h: Int,
    latent_w: Int,
    ctx: DeviceContext,
    value_range: ValueRange = ValueRange.SIGNED,
) raises:
    var dims = video.shape()
    if len(dims) != 5:
        raise Error("save_video_frame_pair_png: video must be [1,3,T,H,W]")
    save_video_frame_png(video, 0, first_path, latent_h, latent_w, ctx, value_range)
    save_video_frame_png(
        video,
        dims[2] - 1,
        last_path,
        latent_h,
        latent_w,
        ctx,
        value_range,
    )


def save_video_frame_sequence_png(
    video: Tensor,
    path_prefix: String,
    path_suffix: String,
    latent_h: Int,
    latent_w: Int,
    ctx: DeviceContext,
    value_range: ValueRange = ValueRange.SIGNED,
) raises -> Int:
    """Save every frame from video [1,3,T,H,W] using prefix + index + suffix."""
    var dims = video.shape()
    if len(dims) != 5:
        raise Error("save_video_frame_sequence_png: video must be [1,3,T,H,W]")
    var frames = dims[2]
    for i in range(frames):
        var path = video_frame_path(path_prefix, i, path_suffix)
        save_video_frame_png(video, i, path, latent_h, latent_w, ctx, value_range)
    return frames
