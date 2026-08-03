# serenitymojo/pipeline/minimax_h3_media_in.mojo — MiniMax-H3 ref2va UNIT A:
# reference media in. CPU/host only: no Tensor, no DeviceContext, no GPU.
#
# ── WHAT THIS IS ─────────────────────────────────────────────────────────────
# The vendor decodes reference media with PyAV inside the reference dataclass
# (`packing_ref2va.py::MiniMaxH3Reference.__post_init__`, :276-300) so that "no
# MiniMax-H3 block ever opens a media file". This module is that boundary for
# the Mojo port, minus PyAV: a reference video becomes `uint8` RGB frames plus
# the fps they carry, and a soundtrack becomes a channel-major float waveform
# plus its sample rate — exactly the two shapes `MiniMaxH3PreparedReference`
# consumes.
#
# ── TWO ROUTES, ONE READER ───────────────────────────────────────────────────
# 1. SIDECAR route (`minimax_h3_read_rgb_frames`): read a pre-extracted planar
#    `rgb24` file plus a small JSON sidecar describing it. Pure file I/O — no
#    subprocess, no external tool, so it runs anywhere and is what the probe
#    gates.
# 2. FFMPEG route (`minimax_h3_ffmpeg_extract_rgb`): shell out to ffprobe/ffmpeg
#    to PRODUCE that pair, then hand off to route 1. The repo already shells out
#    this way — `serenitymojo/pipeline/scail2_decode.mojo` runs ffprobe/ffmpeg
#    through `io/ffi.sys_system` with `components/artifacts.shell_quote` — so
#    this follows that established pattern rather than inventing a new one.
#
# Route 2 is a thin wrapper over route 1 on purpose: everything that can go
# numerically wrong (layout, frame count, byte order) lives in ONE reader that
# a GPU-free probe covers.
#
# The documented hand-extraction command, if you would rather not shell out
# (this is exactly what route 2 issues):
#
#   ffprobe -v error -select_streams v:0 \
#     -show_entries stream=width,height,avg_frame_rate -of csv=p=0 IN.mp4
#   ffmpeg -v error -y -i IN.mp4 -f rawvideo -pix_fmt rgb24 OUT.rgb
#
# and the sidecar next to it:
#
#   {"width": W, "height": H, "num_frames": N, "fps": F, "format": "rgb24"}
#
# ── AUDIO ────────────────────────────────────────────────────────────────────
# WAV is read in pure Mojo through `vendor/mojo-libs/audio/wav.mojo::read_wav`
# (PCM u8/s16/s24/s32 + IEEE float32, any rate, any channel count). That decoder
# hands back INTERLEAVED samples; the vendor's own waveform contract is
# `(channels, num_samples)` CHANNEL-MAJOR (`prepare_reference_waveform`, :715),
# which is also what `serenitymojo/audio/wav.mojo::save_wav` consumes, so
# `minimax_h3_read_wav` de-interleaves at this boundary and every downstream
# consumer sees channel-major only.
#
# mp3/m4a/aac and every other compressed container MUST be converted first —
# `minimax_h3_ffmpeg_decode_audio` does it (`-acodec pcm_s16le`). The vendor's
# own ref2va example request feeds an `.mp3` audio reference, so that path is
# not optional in practice; it is simply not pure-Mojo.
#
# ── NOT IN SCOPE (later units, deliberately absent) ──────────────────────────
# Resizing (LANCZOS onto the reference canvas), 24 fps resampling and waveform
# resampling are NOT here. The index math for the fps resample already exists,
# gated, as `packing_ref2va.mojo::minimax_h3_resample_reference_repeats`; this
# module only carries the media and the rate it came at, so that the resampler
# is applied once, later, by the unit that also owns the pixel resize.

from std.collections import List

from json.parser import loads

from serenitymojo.components.artifacts import shell_quote
from serenitymojo.io.ffi import (
    sys_system,
    sys_open,
    sys_pwrite,
    sys_close,
    BytePtr,
    O_WRONLY,
    O_CREAT,
    O_TRUNC,
)

from audio.wav import AudioBuffer, read_wav, write_wav


# MiniMax-H3's own clock, and the audio VAE's own rate — the defaults a caller
# gets when a container carries no usable metadata.
comptime MINIMAX_H3_MEDIA_FPS = Float64(24.0)
comptime MINIMAX_H3_MEDIA_SAMPLE_RATE = 32000


@fieldwise_init
struct MiniMaxH3RgbFrames(Copyable, Movable):
    """Decoded reference video: `uint8` RGB, channels-LAST.

    `pixels` is flat row-major `[num_frames, height, width, 3]` — the vendor's
    own `(num_frames, height, width, 3)` numpy layout (`decode_reference_video`,
    packing_ref2va.py:153), NOT the channels-first layout tensors use elsewhere
    in this repo. Byte `((f*height + y)*width + x)*3 + c` is channel `c` of
    pixel `(x, y)` of frame `f`.

    `fps` is the rate the frames were decoded at, which is what
    `minimax_h3_resample_reference_repeats` needs to put them on H3's 24 fps
    grid. It is NOT assumed to be 24."""

    var pixels: List[UInt8]
    var num_frames: Int
    var height: Int
    var width: Int
    var fps: Float64

    def numel(self) -> Int:
        return self.num_frames * self.height * self.width * 3


@fieldwise_init
struct MiniMaxH3Waveform(Copyable, Movable):
    """Decoded reference audio: float32 in [-1, 1], CHANNEL-MAJOR.

    `samples` is flat `[channels, num_samples]`: sample `i` of channel `c` is
    `samples[c * num_samples + i]`. This is the vendor's `(channels,
    num_samples)` contract and `save_wav`'s `[2, L]` contract both."""

    var samples: List[Float32]
    var channels: Int
    var num_samples: Int
    var sample_rate: Int

    def duration_seconds(self) -> Float64:
        if self.sample_rate <= 0:
            return 0.0
        return Float64(self.num_samples) / Float64(self.sample_rate)


# ═════════════════════════════════════════════════════════════════════════════
# Small host helpers
# ═════════════════════════════════════════════════════════════════════════════
def _read_all_bytes(path: String) raises -> List[UInt8]:
    var data: List[UInt8]
    with open(path, "r") as f:
        data = f.read_bytes()
    return data^


def _read_all_text(path: String) raises -> String:
    var text: String
    with open(path, "r") as f:
        text = f.read()
    return text^


def _write_all_bytes(path: String, data: List[UInt8]) raises:
    """Byte-exact write through the sys_open/sys_pwrite idiom png.mojo and
    audio/wav.mojo already use — never builtin `open` for binary payloads."""
    var nbytes = len(data)
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("minimax_h3_media_in: cannot open for write: ") + path)
    if nbytes > 0:
        var buf = List[UInt8]()
        buf.resize(nbytes, 0)
        for i in range(nbytes):
            buf[i] = data[i]
        var bp = BytePtr(unsafe_from_address=Int(buf.unsafe_ptr()))
        var done = 0
        while done < nbytes:
            var got = sys_pwrite(fd, bp + done, nbytes - done, done)
            if got <= 0:
                break
            done += got
        _ = sys_close(fd)
        _ = buf^
        if done != nbytes:
            raise Error(String("minimax_h3_media_in: short write to ") + path)
    else:
        _ = sys_close(fd)


def _write_all_text(path: String, text: String) raises:
    var bytes = text.as_bytes()
    var out = List[UInt8]()
    for i in range(text.byte_length()):
        out.append(bytes[i])
    _write_all_bytes(path, out)


def _split_bytes(s: String, sep: Int) -> List[String]:
    """Split on one ASCII byte. Paths and ffprobe CSV in this project are
    ASCII, the same assumption `shell_quote` documents."""
    var out = List[String]()
    var current = String("")
    var bytes = s.as_bytes()
    for i in range(s.byte_length()):
        var b = Int(bytes[i])
        if b == sep:
            out.append(current)
            current = String("")
        else:
            current += chr(b)
    out.append(current)
    return out^


def _strip_ascii(s: String) -> String:
    """Drop leading/trailing spaces, tabs, CR and LF."""
    var bytes = s.as_bytes()
    var start = 0
    var stop = s.byte_length()
    while start < stop:
        var b = Int(bytes[start])
        if b == 32 or b == 9 or b == 13 or b == 10:
            start += 1
        else:
            break
    while stop > start:
        var b = Int(bytes[stop - 1])
        if b == 32 or b == 9 or b == 13 or b == 10:
            stop -= 1
        else:
            break
    var out = String("")
    for i in range(start, stop):
        out += chr(Int(bytes[i]))
    return out^


def _parse_rational_fps(text: String) raises -> Float64:
    """ffprobe reports a frame rate as `num/den` (`30000/1001`, `24/1`). A bare
    number is accepted too."""
    var trimmed = _strip_ascii(text)
    if trimmed == String(""):
        raise Error("minimax_h3_media_in: empty frame rate from ffprobe")
    var parts = _split_bytes(trimmed, 47)  # '/'
    if len(parts) == 1:
        return Float64(atol(parts[0]))
    if len(parts) != 2:
        raise Error(
            String("minimax_h3_media_in: cannot parse frame rate ") + trimmed
        )
    var den = atol(_strip_ascii(parts[1]))
    if den == 0:
        raise Error(
            String("minimax_h3_media_in: zero denominator in frame rate ") + trimmed
        )
    return Float64(atol(_strip_ascii(parts[0]))) / Float64(den)


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE 1 — the raw rgb24 + JSON sidecar pair (pure I/O, no subprocess)
# ═════════════════════════════════════════════════════════════════════════════
def minimax_h3_write_rgb_sidecar(
    sidecar_path: String,
    width: Int,
    height: Int,
    num_frames: Int,
    fps: Float64,
) raises:
    """Write the JSON sidecar that describes a planar rgb24 dump."""
    var text = (
        String("{\"width\": ") + String(width)
        + String(", \"height\": ") + String(height)
        + String(", \"num_frames\": ") + String(num_frames)
        + String(", \"fps\": ") + String(fps)
        + String(", \"format\": \"rgb24\"}")
    )
    _write_all_text(sidecar_path, text)


def minimax_h3_write_rgb_frames(
    frames: MiniMaxH3RgbFrames, rgb_path: String, sidecar_path: String
) raises:
    """Write frames + sidecar as the pair `minimax_h3_read_rgb_frames` reads."""
    if len(frames.pixels) != frames.numel():
        raise Error(
            String("minimax_h3_media_in: frame buffer holds ")
            + String(len(frames.pixels)) + " bytes, geometry implies "
            + String(frames.numel())
        )
    _write_all_bytes(rgb_path, frames.pixels)
    minimax_h3_write_rgb_sidecar(
        sidecar_path, frames.width, frames.height, frames.num_frames, frames.fps
    )


def minimax_h3_read_rgb_frames(
    rgb_path: String, sidecar_path: String
) raises -> MiniMaxH3RgbFrames:
    """Read a planar `rgb24` dump plus its JSON sidecar.

    The sidecar must carry `width`, `height` and `fps`; `num_frames` is
    OPTIONAL and, when present, is cross-checked against the actual file size
    rather than trusted — a truncated dump is the failure this catches, and
    `ffprobe`'s own `nb_frames` is unreliable enough on some containers that the
    file itself is the authority.

    `format`, when present, must be `rgb24`."""
    var sidecar = loads(_read_all_text(sidecar_path))
    if not sidecar.is_object():
        raise Error(
            String("minimax_h3_media_in: sidecar is not a JSON object: ")
            + sidecar_path
        )
    for name in [String("width"), String("height"), String("fps")]:
        if not sidecar.contains(name):
            raise Error(
                String("minimax_h3_media_in: sidecar ") + sidecar_path
                + " is missing key '" + name + "'"
            )
    if sidecar.contains(String("format")):
        var fmt = sidecar[String("format")].as_string()
        if fmt != String("rgb24"):
            raise Error(
                String("minimax_h3_media_in: sidecar declares pixel format '")
                + fmt + "', only 'rgb24' is supported"
            )

    var width = sidecar[String("width")].as_int()
    var height = sidecar[String("height")].as_int()
    var fps = sidecar[String("fps")].as_float()
    if width <= 0 or height <= 0:
        raise Error(
            String("minimax_h3_media_in: sidecar geometry must be positive, got ")
            + String(width) + "x" + String(height)
        )
    if fps <= 0.0:
        raise Error("minimax_h3_media_in: sidecar fps must be positive")

    var pixels = _read_all_bytes(rgb_path)
    var frame_bytes = height * width * 3
    if len(pixels) % frame_bytes != 0:
        raise Error(
            String("minimax_h3_media_in: ") + rgb_path + " holds "
            + String(len(pixels)) + " bytes, not a whole number of "
            + String(frame_bytes) + "-byte rgb24 frames — truncated dump, or the"
            " sidecar geometry does not describe this file"
        )
    var num_frames = len(pixels) // frame_bytes
    if num_frames < 1:
        raise Error(
            String("minimax_h3_media_in: no complete frames in ") + rgb_path
        )
    if sidecar.contains(String("num_frames")):
        var declared = sidecar[String("num_frames")].as_int()
        if declared != num_frames:
            raise Error(
                String("minimax_h3_media_in: sidecar declares ") + String(declared)
                + " frames but " + rgb_path + " holds " + String(num_frames)
            )
    return MiniMaxH3RgbFrames(pixels^, num_frames, height, width, fps)


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE 2 — produce that pair with ffprobe/ffmpeg, then read it
# ═════════════════════════════════════════════════════════════════════════════
def minimax_h3_ffprobe_video_geometry(
    video_path: String, scratch_path: String
) raises -> List[Float64]:
    """`[width, height, fps]` of a container's first video stream.

    `scratch_path` is a writable file this may create and overwrite; ffprobe's
    CSV lands there and is parsed back. Returned as Float64 so one call carries
    both the integer axes and the rational rate — the caller rounds."""
    var command = (
        String("ffprobe -v error -select_streams v:0 -show_entries ")
        + String("stream=width,height,avg_frame_rate -of csv=p=0 ")
        + shell_quote(video_path) + String(" > ") + shell_quote(scratch_path)
    )
    if sys_system(command) != 0:
        raise Error(
            String("minimax_h3_media_in: ffprobe failed on ") + video_path
            + " (is ffprobe installed, and does the file carry a video stream?)"
        )
    var line = _strip_ascii(_read_all_text(scratch_path))
    var fields = _split_bytes(line, 44)  # ','
    if len(fields) < 3:
        raise Error(
            String("minimax_h3_media_in: unexpected ffprobe output '") + line
            + "' for " + video_path
        )
    var width = atol(_strip_ascii(fields[0]))
    var height = atol(_strip_ascii(fields[1]))
    var fps = _parse_rational_fps(fields[2])
    if width <= 0 or height <= 0:
        raise Error(
            String("minimax_h3_media_in: ffprobe reported a non-positive size ")
            + String(width) + "x" + String(height) + " for " + video_path
        )
    return [Float64(width), Float64(height), fps]


def minimax_h3_ffmpeg_extract_rgb(
    video_path: String,
    rgb_path: String,
    sidecar_path: String,
    scratch_path: String,
) raises -> MiniMaxH3RgbFrames:
    """Decode a reference video to `rgb24` + sidecar, then read it back.

    The frame COUNT is derived from the size of the dump, not from ffprobe:
    `nb_frames` is absent or wrong on plenty of containers, while the dump's
    own length is exact by construction.

    NOTE — display-matrix rotation. The vendor undoes a stream's counterclockwise
    display rotation itself, snapped to a quarter turn (`decode_reference_video`,
    packing_ref2va.py:187-189). `ffmpeg` applies that rotation during decode by
    default, so this route already lands upright and does NOT re-apply it. A
    build of ffmpeg with `-noautorotate` in effect would need the turn applied
    here; that is not detected, and is called out rather than silently assumed."""
    var geometry = minimax_h3_ffprobe_video_geometry(video_path, scratch_path)
    var width = Int(geometry[0])
    var height = Int(geometry[1])
    var fps = geometry[2]

    var command = (
        String("ffmpeg -v error -y -i ") + shell_quote(video_path)
        + String(" -f rawvideo -pix_fmt rgb24 ") + shell_quote(rgb_path)
    )
    if sys_system(command) != 0:
        raise Error(
            String("minimax_h3_media_in: ffmpeg rawvideo extraction failed for ")
            + video_path
        )
    # `num_frames` is deliberately left OUT of the sidecar: the reader derives
    # it from the dump's own size, which is exact, and a declared count would
    # only enshrine ffprobe's `nb_frames` guess as something to cross-check
    # against.
    _write_all_text(
        sidecar_path,
        String("{\"width\": ") + String(width)
        + String(", \"height\": ") + String(height)
        + String(", \"fps\": ") + String(fps)
        + String(", \"format\": \"rgb24\"}"),
    )
    return minimax_h3_read_rgb_frames(rgb_path, sidecar_path)


# ═════════════════════════════════════════════════════════════════════════════
# Audio
# ═════════════════════════════════════════════════════════════════════════════
def minimax_h3_read_wav(path: String) raises -> MiniMaxH3Waveform:
    """Read a WAV file into a CHANNEL-MAJOR float waveform.

    `vendor/mojo-libs/audio/wav.mojo::read_wav` decodes PCM u8/s16/s24/s32 and
    IEEE float32 to INTERLEAVED float32; this de-interleaves. Compressed
    payloads inside a `.wav` container are rejected by that decoder — run them
    through `minimax_h3_ffmpeg_decode_audio` instead."""
    var buffer = read_wav(path)
    var channels = buffer.channels
    if channels < 1:
        raise Error(
            String("minimax_h3_media_in: ") + path + " declares "
            + String(channels) + " channels"
        )
    var num_samples = len(buffer.samples) // channels
    if num_samples < 1:
        raise Error(String("minimax_h3_media_in: no samples in ") + path)

    var samples = List[Float32]()
    samples.resize(channels * num_samples, Float32(0.0))
    for i in range(num_samples):
        for c in range(channels):
            samples[c * num_samples + i] = buffer.samples[i * channels + c]
    return MiniMaxH3Waveform(samples^, channels, num_samples, buffer.rate)


def minimax_h3_write_wav(
    waveform: MiniMaxH3Waveform, path: String
) raises:
    """Write a channel-major waveform back out as 16-bit PCM.

    Host-side counterpart of `serenitymojo/audio/wav.mojo::save_wav`, which
    takes a device `Tensor` and so needs a `DeviceContext`; this one does not,
    which is what lets the probe round-trip audio with no GPU."""
    if len(waveform.samples) != waveform.channels * waveform.num_samples:
        raise Error(
            String("minimax_h3_media_in: waveform holds ")
            + String(len(waveform.samples)) + " samples, geometry implies "
            + String(waveform.channels * waveform.num_samples)
        )
    var buffer = AudioBuffer(waveform.sample_rate, waveform.channels)
    for i in range(waveform.num_samples):
        for c in range(waveform.channels):
            buffer.samples.append(waveform.samples[c * waveform.num_samples + i])
    write_wav(path, buffer, 16)


def minimax_h3_ffmpeg_decode_audio(
    media_path: String,
    wav_path: String,
    sample_rate: Int = MINIMAX_H3_MEDIA_SAMPLE_RATE,
    channels: Int = 2,
) raises -> MiniMaxH3Waveform:
    """Decode ANY container's first audio stream to WAV, then read it.

    This is the mp3/m4a/aac path — and the path for a video reference's
    soundtrack, which the vendor takes from the same container as the frames
    (`MiniMaxH3Reference.__post_init__`, :293-295).

    RESAMPLING HAPPENS HERE, IN FFMPEG, NOT IN TORCHAUDIO. The vendor truncates
    at the native rate and then resamples once with `torchaudio`
    (`prepare_reference_waveform`, :740-753). Asking ffmpeg for `-ar
    sample_rate` uses swresample instead, which is a DIFFERENT resampler and
    will not be sample-exact against the vendor. For a bit-comparable gate,
    pass `sample_rate = 0` to keep the container's own rate and resample later
    on the vendor's own terms; the default here is the audio VAE's 32 kHz
    because that is what an end-to-end run needs."""
    if channels < 1:
        raise Error("minimax_h3_media_in: channels must be >= 1")
    var rate_flag = String("")
    if sample_rate > 0:
        rate_flag = String(" -ar ") + String(sample_rate)
    var command = (
        String("ffmpeg -v error -y -i ") + shell_quote(media_path)
        + String(" -vn -acodec pcm_s16le") + rate_flag
        + String(" -ac ") + String(channels) + String(" ")
        + shell_quote(wav_path)
    )
    if sys_system(command) != 0:
        raise Error(
            String("minimax_h3_media_in: ffmpeg audio decode failed for ")
            + media_path
            + " (is ffmpeg installed, and does the file carry an audio stream?)"
        )
    return minimax_h3_read_wav(wav_path)


def minimax_h3_has_audio_stream(media_path: String) raises -> Bool:
    """Whether a container carries an audio stream at all.

    A video reference conditions on its soundtrack when it has one and is a
    video-only reference when it does not, so this decides whether a reference
    contributes audio rows — the same probe `scail2_decode.mojo` uses."""
    var command = (
        String("ffprobe -v error -select_streams a:0 -show_entries ")
        + String("stream=index -of csv=p=0 ") + shell_quote(media_path)
        + String(" | grep -q .")
    )
    return sys_system(command) == 0
