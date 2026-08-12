from std.sys import argv
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.motion_context import (
    minimax_h3_load_motion_context,
    minimax_h3_save_motion_context_tail,
)


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def main() raises:
    var args = argv()
    if len(args) != 5:
        raise Error(
            "usage: motion_context_artifact_probe <latents> <out> <width> <height>"
        )
    var input_path = String(args[1])
    var output_path = String(args[2])
    var width = atol(String(args[3]))
    var height = atol(String(args[4]))
    var rows_per_frame = (height // 32) * (width // 32)
    var st = SafeTensors.open(input_path)
    var vinfo = st.tensor_info(String("video_state_rows"))
    var ainfo = st.tensor_info(String("audio_state_rows"))
    _require(vinfo.shape[0] % rows_per_frame == 0, "video geometry")
    _require(ainfo.shape[0] % 2 == 0, "audio geometry")
    var ctx = DeviceContext()
    var video = Tensor.from_view(
        from_parts(
            vinfo.dtype, vinfo.shape.copy(),
            st.tensor_bytes(String("video_state_rows")),
        ),
        ctx,
    )
    var audio = Tensor.from_view(
        from_parts(
            ainfo.dtype, ainfo.shape.copy(),
            st.tensor_bytes(String("audio_state_rows")),
        ),
        ctx,
    )
    var video_steps = vinfo.shape[0] // rows_per_frame
    var audio_steps = ainfo.shape[0] // 2
    var saved = minimax_h3_save_motion_context_tail(
        video, audio, video_steps, audio_steps, width, height, output_path, ctx
    )
    _require(saved == 39, "expected largest 39-frame tail")
    for frames in [5, 22, 39]:
        var loaded = minimax_h3_load_motion_context(
            output_path, rows_per_frame, vinfo.shape[1], ainfo.shape[1],
            frames, width, height, ctx,
        )
        _require(loaded.video_rows.shape()[1] == vinfo.shape[1], "video dim")
        _require(loaded.audio_rows.shape()[1] == ainfo.shape[1], "audio dim")
        _require(
            loaded.source_video_latent_frames == video_steps,
            "source video metadata",
        )
        _require(
            loaded.source_audio_latents == audio_steps,
            "source audio metadata",
        )
    print("PASS minimax_h3 motion-context compact artifact 5/22/39")
