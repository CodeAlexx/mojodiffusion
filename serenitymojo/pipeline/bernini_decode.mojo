# Bernini-R fresh-process temporal VAE decode and 16fps MP4 mux.
#
# Reuses the parity/performance-gated standard Wan temporal decoder already in
# the stack.  This is a separate executable from bernini_t2v so the two A14B
# experts and the F32 VAE cannot overlap on a 16GB card.
#
# argv: bernini_decode <latent.safetensors> <vae.safetensors> <out_dir>

from std.collections import List
from std.gpu.host import DeviceContext
from std.sys import argv

from serenitymojo.components.artifacts import (
    build_ffmpeg_mux_command,
    video_frame_path,
)
from serenitymojo.image.png import ValueRange, save_png
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_system
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.lingbotvideo.vae_decoder import LingBotWanVaeDecoder
from serenitymojo.models.lingbotvideo.vae_encoder import _latents_mean, _latents_std
from serenitymojo.ops.tensor_algebra import reshape, slice
from serenitymojo.tensor import Tensor


comptime HEIGHT = 480
comptime WIDTH = 848
comptime FRAMES = 81
comptime T_LAT = 21
comptime H_LAT = 60
comptime W_LAT = 106
comptime CHANNELS = 16
comptime LATENT_NUMEL = CHANNELS * T_LAT * H_LAT * W_LAT
comptime FPS = 16


def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error(
            "usage: bernini_decode <latent.safetensors> <vae.safetensors> <out_dir>"
        )
    var latent_path = String(args[1])
    var vae_path = String(args[2])
    var out_dir = String(args[3])
    var ctx = DeviceContext()

    print("=== Bernini-R temporal VAE decode (fresh process) ===")
    var st = ShardedSafeTensors.open(latent_path)
    if String("latent") not in st.names():
        raise Error("Bernini latent file missing key 'latent'")
    var host = Tensor.from_view_as_f32(st.tensor_view(String("latent")), ctx).to_host(ctx)
    if len(host) != LATENT_NUMEL:
        raise Error("Bernini latent numel mismatch for compiled 848x480x81 geometry")

    # Creator `_dit_latent_to_vae`: z = normalized_latent * std + mean.
    # The values come from the existing gated Wan VAE encoder contract.
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

    print("  loading existing standard-Wan temporal decoder:", vae_path)
    var decoder = LingBotWanVaeDecoder[H_LAT, W_LAT].load(vae_path, ctx)
    var video = decoder.decode_video(z, ctx)
    var shape = video.shape()
    if (
        len(shape) != 5 or shape[0] != 1 or shape[1] != 3
        or shape[2] != FRAMES or shape[3] != HEIGHT or shape[4] != WIDTH
    ):
        raise Error("Bernini decoded video shape mismatch")
    print("  decoded [1,3,", shape[2], ",", shape[3], ",", shape[4], "]")

    _ = sys_system(String("mkdir -p '") + out_dir + String("'"))
    var prefix = out_dir + String("/frame_")
    for frame_index in range(FRAMES):
        var frame5 = slice(video, 2, frame_index, 1, ctx)
        var frame = reshape(frame5, [1, 3, HEIGHT, WIDTH], ctx)
        save_png(
            frame, video_frame_path(prefix, frame_index), ctx, ValueRange.SIGNED
        )
    var mp4 = out_dir + String("/bernini_r_t2v.mp4")
    var mux = build_ffmpeg_mux_command(prefix, String(".png"), mp4, FPS)
    var status = sys_system(mux)
    if status != 0:
        raise Error("Bernini ffmpeg mux failed")
    print("GATE decode complete:", mp4)
