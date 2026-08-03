# models/vae/parity/minimax_h3_video_vae_device_real_weights_smoke.mojo
#
# REAL-WEIGHTS smoke for models/vae/minimax_h3_video_{encoder,decoder}_
# device.mojo — loads the ACTUAL released video_vae/source/model.safetensors
# (9.70 GiB, F32, landed 2026-08-02) and runs one small-resolution forward
# through each, checking for NaN/Inf and reporting magnitude stats. This is
# NOT a parity gate (no oracle output to compare against — the creator's own
# klvae.py needs a live torch+diffusers env we don't have here); it is the
# "does the real network produce sane numbers" check the toy-scale synthetic
# probes cannot do, since those never touch real weight VALUES.
#
# Real config throughout (minimax_h3_video_released_encoder_config /
# minimax_h3_video_released_decoder_config) — NOT trimmed to a few layers —
# this loads all 6 encoder levels and all 36 decoder transformer blocks, the
# genuine 9.7 GiB. Only the SPATIAL/temporal input size is kept small to
# keep runtime reasonable: H=W=32 so the 4 factor-2 spatial downsamples
# (levels 0-3, vae_ratio=16) leave H'=W'=2 for levels 4-5's kernel-3 reflect
# pad (needs >=2) to stay valid, T=17 (one real clip_length).
#
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs -Xlinker -lcudnn \
#     serenitymojo/models/vae/parity/minimax_h3_video_vae_device_real_weights_smoke.mojo \
#     -o /tmp/mmh3_real_smoke
#   LD_LIBRARY_PATH=serenitymojo/ops/cshim/lib:serenitymojo/ops/cshim/lib/cudnn_stubs \
#     /tmp/mmh3_real_smoke

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice, minimax_h3_video_encode_device,
    minimax_h3_video_released_encoder_config,
)
from serenitymojo.models.vae.minimax_h3_video_decoder_device import (
    minimax_h3_video_decode_device, minimax_h3_video_decoder_device_load,
    minimax_h3_video_released_decoder_config,
)

comptime VIDEO_VAE_SOURCE_DIR = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/video_vae/source"
)


def _pattern(seed: Int, n: Int) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for i in range(n):
        var v = ((seed * 1103515245 + i * 12345 + 7) % 2003) - 1001
        out.append(Float32(v) * Float32(0.01))
    return out^


def _stats(name: String, values: List[Float32]) raises -> None:
    var n = len(values)
    var mn = Float32(1.0e30)
    var mx = Float32(-1.0e30)
    var sum_ = Float64(0.0)
    var nan_count = 0
    var inf_count = 0
    for i in range(n):
        var v = values[i]
        if v != v:
            nan_count += 1
            continue
        if v > Float32(1.0e30) or v < Float32(-1.0e30):
            inf_count += 1
            continue
        if v < mn:
            mn = v
        if v > mx:
            mx = v
        sum_ += Float64(v)
    var mean = sum_ / Float64(n) if n > 0 else 0.0
    print(
        "  ", name, ": n=", n, " nan=", nan_count, " inf=", inf_count,
        " min=", mn, " max=", mx, " mean=", mean,
    )
    if nan_count > 0:
        raise Error(String("probe: FAIL ") + name + " contains NaN")
    if inf_count > 0:
        raise Error(String("probe: FAIL ") + name + " contains an overflow-scale value")


def _run_encoder_smoke(ctx: DeviceContext) raises:
    var config = minimax_h3_video_released_encoder_config()
    print("  loading REAL encoder weights from", VIDEO_VAE_SOURCE_DIR, "...")
    var encoder = MiniMaxH3VideoEncoderDevice.load(VIDEO_VAE_SOURCE_DIR, config, ctx)
    print("  loaded", len(encoder.name_to_idx), "encoder tensors")

    # [1, T=17, H=32, W=32, in_channels=3] NDHWC -- deterministic pseudo-random
    # pixel values, NOT ImageNet-normalized (that pipeline step is out of
    # scope here; this checks the network's own numerical behavior on
    # arbitrary finite input, not end-to-end visual correctness).
    var pixels = Tensor.from_host(
        _pattern(1, 1 * 17 * 32 * 32 * 3), [1, 17, 32, 32, 3], STDtype.F32, ctx
    )
    print("  running real forward (6 levels, real weights)...")
    var moments = minimax_h3_video_encode_device(encoder, pixels, ctx)
    print("  moments shape:", moments.shape())
    _stats("encoder moments", moments.to_host(ctx))


def _run_decoder_smoke(ctx: DeviceContext) raises:
    var config = minimax_h3_video_released_decoder_config()
    print("  loading REAL decoder weights from", VIDEO_VAE_SOURCE_DIR, "...")
    var decoder = minimax_h3_video_decoder_device_load(VIDEO_VAE_SOURCE_DIR, config, ctx)
    print("  loaded", len(decoder.name_to_idx), "decoder tensors (incl. split qkv)")

    # latent_T=1, latent_H=2, latent_W=2 -> S = 1*2*2 + 1 + num_register_tokens(4) = 9.
    var latents = Tensor.from_host(
        _pattern(2, 1 * 1 * 2 * 2 * 24), [1, 1, 2, 2, 24], STDtype.F32, ctx
    )
    print("  running real forward (36 layers, real weights)...")
    var pixels = minimax_h3_video_decode_device[9, 32, 64](decoder, latents, ctx)
    print("  pixels shape:", pixels.shape())
    _stats("decoder pixels", pixels.to_host(ctx))


def main() raises:
    var ctx = DeviceContext()
    print("== real-weights encoder smoke ==")
    _run_encoder_smoke(ctx)
    print("== real-weights decoder smoke ==")
    _run_decoder_smoke(ctx)
    print("minimax_h3_video_vae_device_real_weights_smoke PASS")
