# SCAIL-2 fresh-process Wan VAE decode and 16 fps MP4 mux.
#
# argv: scail2_decode <latent.safetensors> <vae.safetensors> <out_dir> [audio_source]

from std.collections import List
from max.gpu.host import DeviceContext
from std.sys import argv

from serenitymojo.components.artifacts import (
    ffmpeg_frame_pattern,
    shell_quote,
    video_frame_path,
)
from serenitymojo.image.png import ValueRange, save_png
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_system
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.lingbotvideo.vae_decoder import LingBotWanVaeDecoder
from serenitymojo.models.scail2.scail2_manifest import (
    preflight_scail2_decode_manifest,
    write_scail2_decode_manifest,
)
from serenitymojo.models.lingbotvideo.vae_encoder import _latents_mean, _latents_std
from serenitymojo.ops.tensor_algebra import reshape, slice
from serenitymojo.tensor import Tensor


comptime HEIGHT = 512
comptime WIDTH = 896
comptime FRAMES = 65
comptime T_LAT = 17
comptime H_LAT = 64
comptime W_LAT = 112
comptime CHANNELS = 16
comptime LATENT_NUMEL = CHANNELS * T_LAT * H_LAT * W_LAT
comptime FPS = 16


def main() raises:
    var args = argv()
    if len(args) != 4 and len(args) != 5:
        raise Error(
            "usage: scail2_decode <latent.safetensors> <vae.safetensors> "
            "<out_dir> [audio_source]"
        )
    var latent_path = String(args[1])
    var vae_path = String(args[2])
    var out_dir = String(args[3])
    var audio_source = String("-")
    if len(args) == 5:
        audio_source = String(args[4])
    preflight_scail2_decode_manifest(latent_path)
    var has_audio = False
    if audio_source != String("-"):
        var source_probe = (
            String("ffprobe -v error -show_entries format=duration ")
            + String("-of csv=p=0 ") + shell_quote(audio_source)
            + String(" >/dev/null")
        )
        if sys_system(source_probe) != 0:
            raise Error("SCAIL-2 audio source is unreadable")
        var audio_probe = (
            String("ffprobe -v error -select_streams a:0 ")
            + String("-show_entries stream=index -of csv=p=0 ")
            + shell_quote(audio_source) + String(" | grep -q .")
        )
        has_audio = sys_system(audio_probe) == 0
    var ctx = DeviceContext()

    var st = ShardedSafeTensors.open(latent_path)
    if String("latent") not in st.names():
        raise Error("SCAIL-2 latent file missing key 'latent'")
    var latent_view = st.tensor_view(String("latent"))
    if (
        latent_view.dtype != STDtype.F32 or len(latent_view.shape) != 4
        or latent_view.shape[0] != CHANNELS
        or latent_view.shape[1] != T_LAT
        or latent_view.shape[2] != H_LAT
        or latent_view.shape[3] != W_LAT
    ):
        raise Error("SCAIL-2 latent must be F32 [16,17,64,112]")
    var host = Tensor.from_view_as_f32(latent_view, ctx).to_host(ctx)

    var means = _latents_mean()
    var stds = _latents_std()
    var channel_stride = T_LAT * H_LAT * W_LAT
    var zhost = List[Float32]()
    zhost.resize(LATENT_NUMEL, 0.0)
    for i in range(LATENT_NUMEL):
        var channel = (i // channel_stride) % CHANNELS
        zhost[i] = host[i] * stds[channel] + means[channel]
    var z = Tensor.from_host(
        zhost, [1, CHANNELS, T_LAT, H_LAT, W_LAT], STDtype.F32, ctx
    )

    var decoder = LingBotWanVaeDecoder[H_LAT, W_LAT].load(vae_path, ctx)
    var video = decoder.decode_video(z, ctx)
    var shape = video.shape()
    if (
        len(shape) != 5 or shape[0] != 1 or shape[1] != 3
        or shape[2] != FRAMES or shape[3] != HEIGHT or shape[4] != WIDTH
    ):
        raise Error("SCAIL-2 decoded video shape mismatch")
    print("SCAIL-2 decoded [1,3,65,512,896]")

    if sys_system(String("mkdir -p -- ") + shell_quote(out_dir)) != 0:
        raise Error("SCAIL-2 output directory creation failed")
    var prefix = out_dir + String("/frame_")
    for frame_index in range(FRAMES):
        var frame5 = slice(video, 2, frame_index, 1, ctx)
        var frame = reshape(frame5, [1, 3, HEIGHT, WIDTH], ctx)
        save_png(
            frame, video_frame_path(prefix, frame_index), ctx,
            ValueRange.SIGNED,
        )
    var mp4 = out_dir + String("/scail2_animation.mp4")
    var video_only_mp4 = mp4
    if has_audio:
        video_only_mp4 = out_dir + String("/scail2_animation_video_only.mp4")
    # Bound input explicitly: reusable output directories must never append a
    # stale contiguous frame_65.png (or later) to a fresh 65-frame result.
    var mux = (
        String("ffmpeg -y -hide_banner -loglevel error -framerate ")
        + String(FPS)
        + String(" -start_number 0 -i ")
        + shell_quote(ffmpeg_frame_pattern(prefix, String(".png")))
        + String(" -frames:v ")
        + String(FRAMES)
        + String(" -c:v libx264 -pix_fmt yuv420p -movflags +faststart ")
        + shell_quote(video_only_mp4)
    )
    if sys_system(mux) != 0:
        raise Error("SCAIL-2 ffmpeg mux failed")
    if has_audio:
        var audio_mux = (
            String("ffmpeg -y -hide_banner -loglevel error -i ")
            + shell_quote(video_only_mp4)
            + String(" -i ")
            + shell_quote(audio_source)
            + String(" -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac ")
            + String("-shortest -movflags +faststart ")
            + shell_quote(mp4)
        )
        if sys_system(audio_mux) != 0:
            raise Error("SCAIL-2 audio mux failed")
        print("SCAIL-2 preserved reference audio:", audio_source)
    elif audio_source != String("-"):
        print("SCAIL-2 reference has no audio stream; wrote video-only output")
    var result_manifest = write_scail2_decode_manifest(
        String(args[0]), latent_path, vae_path, audio_source, has_audio, mp4,
    )
    print("GATE SCAIL-2 decode complete:", mp4)
    print("GATE SCAIL-2 result manifest:", result_manifest)
