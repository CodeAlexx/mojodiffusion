# models/vae/ltx2_audio_processor.mojo — creator-exact LTX-2.3 source-audio
# preprocessing for Retake/Extend.
#
# Oracle:
#   ltx-core/model/audio_vae/ops.py::AudioProcessor at the exact LTX Desktop
#   lock revision.  This is deliberately separate from the BWE vocoder STFT:
#   the encoder uses center=True REFLECT padding, n_fft=win_length=1024,
#   hop=160, power=1, and Slaney-normalized 64-bin mel filters at 16 kHz.

from max.gpu.host import DeviceContext
from std.math import cos, sin, pi, log, exp
from std.memory import alloc

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import (
    sys_open,
    sys_close,
    sys_pread,
    file_size,
    O_RDONLY,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.conv1d import conv1d
from serenitymojo.ops.linear import linear
from serenitymojo.ops.resample import resample_hann
from serenitymojo.ops.tensor_algebra import permute, reshape
from serenitymojo.models.vocoder.ltx2_stft import magnitude, clamp_log


comptime LTX2_AUDIO_SAMPLE_RATE = 16000
comptime LTX2_AUDIO_CHANNELS = 2
comptime LTX2_AUDIO_N_FFT = 1024
comptime LTX2_AUDIO_HOP = 160
comptime LTX2_AUDIO_FREQS = 513
comptime LTX2_AUDIO_MELS = 64
comptime LTX2_AUDIO_CENTER_PAD = 512


def _read_interleaved_f32(
    path: String, channels: Int, max_samples_per_channel: Int
) raises -> List[Float32]:
    """Read ffmpeg f32le interleaved audio and return planar [C,T] host data."""
    if channels != LTX2_AUDIO_CHANNELS:
        raise Error("LTX2 creator audio encoder requires a stereo source")
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open LTX2 source audio: ") + path)
    var nbytes = file_size(fd)
    if nbytes <= 0 or nbytes % (4 * channels) != 0:
        _ = sys_close(fd)
        raise Error("LTX2 source audio is empty or not interleaved f32le")
    var raw_samples = nbytes // (4 * channels)
    var samples = raw_samples
    if max_samples_per_channel > 0 and samples > max_samples_per_channel:
        samples = max_samples_per_channel
    var read_bytes = samples * channels * 4
    var buf = alloc[UInt8](read_bytes)
    var done = 0
    while done < read_bytes:
        var got = sys_pread(fd, buf + done, read_bytes - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    if done != read_bytes:
        buf.free()
        raise Error("short read from LTX2 source audio")
    var interleaved = buf.bitcast[Float32]()
    var planar = List[Float32]()
    planar.resize(channels * samples, Float32(0.0))
    for t in range(samples):
        for c in range(channels):
            planar[c * samples + t] = interleaved[t * channels + c]
    buf.free()
    return planar^


def load_source_waveform_f32(
    path: String,
    sample_rate: Int,
    channels: Int,
    max_samples_per_channel: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Load decoded-native-rate f32le and creator-resample it to 16 kHz.

    The Rust service uses ffmpeg only for the same libavcodec decode performed
    by Creator's PyAV path.  Resampling remains the torchaudio-equivalent Mojo
    sinc_interp_hann implementation.
    """
    if sample_rate <= 0:
        raise Error("LTX2 source audio sample rate must be positive")
    var planar = _read_interleaved_f32(
        path, channels, max_samples_per_channel
    )
    var samples = len(planar) // channels
    var waveform = Tensor.from_host(
        planar, [1, channels, samples], STDtype.F32, ctx
    )
    if sample_rate == LTX2_AUDIO_SAMPLE_RATE:
        return waveform^
    return resample_hann(
        waveform, sample_rate, LTX2_AUDIO_SAMPLE_RATE, ctx
    )


def _reflect_pad_center_host(
    waveform: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """PyTorch reflect-pad [B,C,T] by n_fft//2 on both temporal sides."""
    var sh = waveform.shape()
    if len(sh) != 3 or sh[0] != 1 or sh[1] != LTX2_AUDIO_CHANNELS:
        raise Error("LTX2 audio waveform must be [1,2,T]")
    var length = sh[2]
    if length <= LTX2_AUDIO_CENTER_PAD:
        raise Error("LTX2 source audio is too short for reflect STFT padding")
    # This host round-trip occurs before the learned encoder is loaded.  It is
    # exact F32 data movement and avoids inventing a different padding kernel.
    var source = waveform.to_host(ctx)
    var padded_length = length + 2 * LTX2_AUDIO_CENTER_PAD
    var padded = List[Float32]()
    padded.resize(
        LTX2_AUDIO_CHANNELS * padded_length, Float32(0.0)
    )
    for c in range(LTX2_AUDIO_CHANNELS):
        for out_t in range(padded_length):
            var src_t = out_t - LTX2_AUDIO_CENTER_PAD
            if src_t < 0:
                src_t = -src_t
            elif src_t >= length:
                src_t = 2 * length - 2 - src_t
            padded[c * padded_length + out_t] = (
                source[c * length + src_t]
            )
    return Tensor.from_host(
        padded, [LTX2_AUDIO_CHANNELS, 1, padded_length],
        STDtype.F32, ctx,
    )


def _creator_stft_basis(ctx: DeviceContext) raises -> Tensor:
    """Periodic-Hann, unnormalized, onesided torch.stft basis."""
    var data = List[Float32]()
    data.resize(
        2 * LTX2_AUDIO_FREQS * LTX2_AUDIO_N_FFT, Float32(0.0)
    )
    for k in range(LTX2_AUDIO_FREQS):
        for n in range(LTX2_AUDIO_N_FFT):
            var angle = (
                Float64(2.0) * pi * Float64(k) * Float64(n)
                / Float64(LTX2_AUDIO_N_FFT)
            )
            var window = Float64(0.5) * (
                Float64(1.0)
                - cos(
                    Float64(2.0) * pi * Float64(n)
                    / Float64(LTX2_AUDIO_N_FFT)
                )
            )
            data[k * LTX2_AUDIO_N_FFT + n] = Float32(
                window * cos(angle)
            )
            data[
                (LTX2_AUDIO_FREQS + k) * LTX2_AUDIO_N_FFT + n
            ] = Float32(-window * sin(angle))
    return Tensor.from_host(
        data, [2 * LTX2_AUDIO_FREQS, 1, LTX2_AUDIO_N_FFT],
        STDtype.F32, ctx,
    )


def _hz_to_slaney_mel(freq: Float64) -> Float64:
    if freq < Float64(1000.0):
        return freq / (Float64(200.0) / Float64(3.0))
    var logstep = log(Float64(6.4)) / Float64(27.0)
    return Float64(15.0) + log(freq / Float64(1000.0)) / logstep


def _slaney_mel_to_hz(mel: Float64) -> Float64:
    if mel < Float64(15.0):
        return mel * (Float64(200.0) / Float64(3.0))
    var logstep = log(Float64(6.4)) / Float64(27.0)
    return Float64(1000.0) * exp(logstep * (mel - Float64(15.0)))


def _creator_slaney_mel_basis(ctx: DeviceContext) raises -> Tensor:
    """torchaudio melscale_fbanks(..., mel_scale='slaney', norm='slaney')."""
    var points = List[Float64]()
    points.resize(LTX2_AUDIO_MELS + 2, Float64(0.0))
    var mel_min = _hz_to_slaney_mel(Float64(0.0))
    var mel_max = _hz_to_slaney_mel(Float64(8000.0))
    for i in range(LTX2_AUDIO_MELS + 2):
        var mel = mel_min + (
            (mel_max - mel_min) * Float64(i)
            / Float64(LTX2_AUDIO_MELS + 1)
        )
        points[i] = _slaney_mel_to_hz(mel)

    # `linear` consumes [n_mels,n_freqs], the transpose of torchaudio's
    # internally stored [n_freqs,n_mels] filterbank.
    var basis = List[Float32]()
    basis.resize(
        LTX2_AUDIO_MELS * LTX2_AUDIO_FREQS, Float32(0.0)
    )
    for m in range(LTX2_AUDIO_MELS):
        var lower = points[m]
        var center = points[m + 1]
        var upper = points[m + 2]
        var enorm = Float64(2.0) / (upper - lower)
        for k in range(LTX2_AUDIO_FREQS):
            var freq = (
                Float64(k) * Float64(8000.0)
                / Float64(LTX2_AUDIO_FREQS - 1)
            )
            var down = (freq - lower) / (center - lower)
            var up = (upper - freq) / (upper - center)
            var value = down if down < up else up
            if value < Float64(0.0):
                value = Float64(0.0)
            basis[m * LTX2_AUDIO_FREQS + k] = Float32(
                value * enorm
            )
    return Tensor.from_host(
        basis, [LTX2_AUDIO_MELS, LTX2_AUDIO_FREQS],
        STDtype.F32, ctx,
    )


def waveform_to_creator_log_mel(
    waveform_f32: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Creator AudioProcessor.waveform_to_mel -> BF16 [1,2,T,64]."""
    var padded = _reflect_pad_center_host(waveform_f32, ctx)
    var forward_basis = _creator_stft_basis(ctx)
    var spec = conv1d(
        padded, forward_basis, None, LTX2_AUDIO_HOP, 0, 1, 1, ctx
    )
    var mag = magnitude(spec, ctx)  # [2,513,T]
    var tf = mag.shape()[2]
    var mag_t = permute(mag, [0, 2, 1], ctx)
    var mag_flat = reshape(
        mag_t, [LTX2_AUDIO_CHANNELS * tf, LTX2_AUDIO_FREQS], ctx
    )
    var mel_basis = _creator_slaney_mel_basis(ctx)
    var mel_flat = linear(mag_flat, mel_basis, None, ctx)
    var mel = reshape(
        mel_flat, [1, LTX2_AUDIO_CHANNELS, tf, LTX2_AUDIO_MELS], ctx
    )
    # Creator clamps only the minimum.  1e30 is safely above any normalized
    # f32 audio magnitude and preserves the exact lower-bound behavior.
    return cast_tensor(
        clamp_log(mel, Float32(1.0e-5), Float32(1.0e30), ctx),
        STDtype.BF16,
        ctx,
    )
