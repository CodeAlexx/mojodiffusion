# artifacts_smoke.mojo - compile/run gate for shared artifact helpers.

from std.gpu.host import DeviceContext

from serenitymojo.components.artifacts import (
    build_ffmpeg_mux_command,
    mux_frame_sequence_mp4,
    save_video_frame_pair_png,
    save_video_frame_sequence_png,
    video_frame_path,
)
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn


comptime OUT0 = "/home/alex/mojodiffusion/output/artifacts_smoke_frame0.png"
comptime OUT1 = "/home/alex/mojodiffusion/output/artifacts_smoke_frame1.png"
comptime SEQ_PREFIX = "/home/alex/mojodiffusion/output/artifacts_smoke_seq_"
comptime SEQ_SUFFIX = ".png"
comptime MP4_OUT = "/home/alex/mojodiffusion/output/artifacts_smoke_seq.mp4"


def main() raises:
    var ctx = DeviceContext()
    var shape = List[Int]()
    shape.append(1)
    shape.append(3)
    shape.append(2)
    shape.append(16)
    shape.append(16)
    var video = randn(shape^, UInt64(20260528), STDtype.F32, ctx)
    save_video_frame_pair_png(video, String(OUT0), String(OUT1), 1, 1, ctx)
    var count = save_video_frame_sequence_png(
        video, String(SEQ_PREFIX), String(SEQ_SUFFIX), 1, 1, ctx
    )
    if count != 2:
        raise Error("artifacts_smoke: sequence frame count mismatch")
    var cmd = build_ffmpeg_mux_command(
        String(SEQ_PREFIX), String(SEQ_SUFFIX), String(MP4_OUT), 4
    )
    if not cmd.startswith(String("ffmpeg -y -hide_banner")):
        raise Error("artifacts_smoke: ffmpeg command prefix mismatch")
    mux_frame_sequence_mp4(String(SEQ_PREFIX), String(SEQ_SUFFIX), String(MP4_OUT), 4)
    print("[artifacts] saved ->", OUT0)
    print("[artifacts] saved ->", OUT1)
    print("[artifacts] sequence first ->", video_frame_path(String(SEQ_PREFIX), 0, String(SEQ_SUFFIX)))
    print("[artifacts] sequence last ->", video_frame_path(String(SEQ_PREFIX), count - 1, String(SEQ_SUFFIX)))
    print("[artifacts] mp4 ->", MP4_OUT)
