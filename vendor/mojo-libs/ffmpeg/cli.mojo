# ffmpeg/cli.mojo — thin wrappers over the `ffmpeg`/`ffprobe` binaries.
#
# CLI-first media glue: decode any audio (mp3/m4a/...) to PCM WAV that the
# pure-Mojo `audio.wav` reader can consume, and mux a frame sequence + audio
# into an mp4. Runs the system ffmpeg via libc `system()` (no libav* FFI).
#
# Requires `ffmpeg`/`ffprobe` on PATH at runtime. Commands run through /bin/sh,
# so paths are single-quote-escaped; still, only pass trusted paths.

from std.ffi import external_call
from audio.wav import AudioBuffer, read_wav


def _run(cmd: String) -> Int:
    """Run a shell command via libc system(); returns its exit status."""
    var b = List[UInt8]()
    var sb = cmd.as_bytes()
    for i in range(len(sb)):
        b.append(sb[i])
    b.append(0)
    return Int(external_call["system", Int32](b.unsafe_ptr()))


def _q(s: String) -> String:
    """Single-quote-escape a path for /bin/sh: ' -> '\\''."""
    var out = String("'")
    var b = s.as_bytes()
    for i in range(len(b)):
        var c = Int(b[i])
        if c == 39:                      # a single quote
            out += String("'\\''")
        else:
            out += chr(c)
    out += String("'")
    return out^


def have_ffmpeg() -> Bool:
    return _run(String("command -v ffmpeg >/dev/null 2>&1")) == 0


def decode_to_wav(in_path: String, out_wav: String, rate: Int = 0, channels: Int = 0) -> Int:
    """Transcode any ffmpeg-readable audio to PCM s16 WAV. 0 ok, nonzero = error.
    rate/channels of 0 keep the source value."""
    var cmd = String("ffmpeg -y -loglevel error -i ") + _q(in_path)
    if rate > 0:
        cmd += String(" -ar ") + String(rate)
    if channels > 0:
        cmd += String(" -ac ") + String(channels)
    cmd += String(" -c:a pcm_s16le ") + _q(out_wav)
    return _run(cmd)


def read_audio(path: String, tmp_wav: String = String("/tmp/_mojo_ffmpeg_audio.wav"),
               rate: Int = 0, channels: Int = 0) raises -> AudioBuffer:
    """Decode ANY audio file to samples via ffmpeg -> temp WAV -> audio.read_wav.
    Use this for formats the pure-Mojo wav reader rejects (mp3/m4a/ogg/...)."""
    var rc = decode_to_wav(path, tmp_wav, rate, channels)
    if rc != 0:
        raise Error("ffmpeg.read_audio: ffmpeg failed (rc=" + String(rc) + ") on " + path)
    return read_wav(tmp_wav)


def encode_frames(frames_pattern: String, out_mp4: String, fps: Int = 24) -> Int:
    """Encode a frame sequence (e.g. /tmp/frame_%04d.png) to H.264 mp4. 0 ok."""
    var cmd = (
        String("ffmpeg -y -loglevel error -framerate ") + String(fps)
        + String(" -i ") + _q(frames_pattern)
        + String(" -c:v libx264 -pix_fmt yuv420p ") + _q(out_mp4)
    )
    return _run(cmd)


def mux_av(frames_pattern: String, audio_path: String, out_mp4: String, fps: Int = 24) -> Int:
    """Mux a frame sequence + an audio track into an mp4 (H.264 + AAC). 0 ok.
    Output length is clamped to the shorter stream (-shortest)."""
    var cmd = (
        String("ffmpeg -y -loglevel error -framerate ") + String(fps)
        + String(" -i ") + _q(frames_pattern)
        + String(" -i ") + _q(audio_path)
        + String(" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest ")
        + _q(out_mp4)
    )
    return _run(cmd)


def probe_to_file(media_path: String, out_json: String) -> Int:
    """Write ffprobe stream/format info as JSON to out_json (read it yourself). 0 ok."""
    var cmd = (
        String("ffprobe -v error -show_format -show_streams -of json ")
        + _q(media_path) + String(" > ") + _q(out_json)
    )
    return _run(cmd)
