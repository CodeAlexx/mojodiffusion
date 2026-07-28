# Numeric gate for the creator LTX-2.3 Retake/Extend source-audio path.
#
# Fixture production:
#   1. Decode with ffmpeg/libavcodec to native-rate interleaved f32le, capped to
#      the probed audio-stream duration (the same decoded sample span as PyAV).
#   2. Run ltx2_audio_encoder_oracle.py in LTX Desktop's exact locked venv.
# This executable then compares 16 kHz waveform, log-mel, learned AudioVAE
# latent, and the conformed 126-frame Retake latent.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.parity import ParityHarness
from serenitymojo.models.vae.ltx2_audio_processor import (
    load_source_waveform_f32,
    waveform_to_creator_log_mel,
)
from serenitymojo.models.vae.ltx2_audio_vae import (
    LTX2AudioVaeDecoderWeights,
    encode_spectrogram,
)


comptime CKPT = (
    "/home/alex/.serenity/models/checkpoints/"
    "sulphur_dev_bf16.safetensors"
)
comptime RAW_AUDIO = "/tmp/ltx2_source_audio_product.f32le"
comptime ORACLE = "/tmp/ltx2_audio_creator_oracle.safetensors"
comptime SOURCE_RATE = 48000
comptime SOURCE_CHANNELS = 2
comptime SOURCE_MAX_SAMPLES = 242000
comptime REQUIRED_LATENT_FRAMES = 126


def _compare(
    name: String,
    actual: Tensor,
    expected: Tensor,
    threshold: Float64,
    ctx: DeviceContext,
) raises -> Bool:
    var harness = ParityHarness(threshold)
    var got = cast_tensor(actual, STDtype.F32, ctx).to_host(ctx)
    var want = cast_tensor(expected, STDtype.F32, ctx).to_host(ctx)
    var result = harness.compare_host(got, want)
    print(
        name, "cos=", result.cos, "max_abs=", result.max_abs,
        "count=", len(got),
    )
    return result.cos >= threshold


def main() raises:
    var ctx = DeviceContext()
    var oracle = ShardedSafeTensors.open(String(ORACLE))
    var all_pass = True

    var waveform = load_source_waveform_f32(
        String(RAW_AUDIO), SOURCE_RATE, SOURCE_CHANNELS,
        SOURCE_MAX_SAMPLES, ctx,
    )
    var waveform_ref = Tensor.from_view_as_f32(
        oracle.tensor_view("waveform_16k"), ctx
    )
    all_pass = _compare(
        String("waveform_16k"), waveform, waveform_ref,
        Float64(0.9999999), ctx,
    ) and all_pass

    var mel = waveform_to_creator_log_mel(waveform, ctx)
    var mel_ref = Tensor.from_view_as_f32(oracle.tensor_view("mel"), ctx)
    all_pass = _compare(
        String("creator_log_mel"), mel, mel_ref,
        Float64(0.99999), ctx,
    ) and all_pass

    var encoder = LTX2AudioVaeDecoderWeights.load_encoder(
        String(CKPT), ctx
    )
    var latent = encode_spectrogram(encoder, mel, ctx)
    var latent_ref = Tensor.from_view_as_f32(
        oracle.tensor_view("latent_unconformed"), ctx
    )
    all_pass = _compare(
        String("audio_vae_latent_unconformed"), latent, latent_ref,
        Float64(0.999), ctx,
    ) and all_pass

    # Separate the learned network from media decode/DSP drift.  Feeding the
    # oracle mel (rounded to Creator's BF16 encoder input) must independently
    # clear the stricter model gate.
    var oracle_mel_bf16 = cast_tensor(mel_ref, STDtype.BF16, ctx)
    var latent_from_oracle_mel = encode_spectrogram(
        encoder, oracle_mel_bf16, ctx
    )
    all_pass = _compare(
        String("audio_vae_network_oracle_mel"),
        latent_from_oracle_mel, latent_ref,
        Float64(0.999), ctx,
    ) and all_pass

    if latent.shape()[2] < REQUIRED_LATENT_FRAMES:
        raise Error("audio encoder parity fixture returned too few latent frames")
    var conformed = slice(
        latent, 2, 0, REQUIRED_LATENT_FRAMES, ctx
    )
    var conformed_ref = Tensor.from_view_as_f32(
        oracle.tensor_view("latent"), ctx
    )
    all_pass = _compare(
        String("audio_vae_latent_conformed"), conformed, conformed_ref,
        Float64(0.999), ctx,
    ) and all_pass

    if not all_pass:
        raise Error("LTX2 creator source-audio parity gate failed")
    print("LTX2 creator source-audio parity gate: PASS")
