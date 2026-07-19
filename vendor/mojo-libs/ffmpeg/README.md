# ffmpeg — CLI wrappers (decode → WAV, mux A/V) for Mojo

Thin Mojo wrappers over the system **`ffmpeg`/`ffprobe`** binaries (run via libc
`system()`, no `libav*` FFI). The bridge between arbitrary media and the
pure-Mojo libs: decode any audio to PCM WAV that [`audio`](../audio/) can read,
and mux a frame sequence + audio into an mp4 (e.g. generated video frames +
generated audio from LTX2 / NAVA).

**Requires `ffmpeg` + `ffprobe` on PATH at runtime.** Commands go through
`/bin/sh` with single-quote-escaped paths — pass trusted paths only.

## API (`cli.mojo`)

| Function | What it does |
|---|---|
| `have_ffmpeg()` | True if `ffmpeg` is on PATH |
| `decode_to_wav(in, out_wav, rate=0, channels=0)` | transcode any audio → PCM s16 WAV (0 = keep source rate/channels) |
| `read_audio(path, tmp_wav=…, rate=0, channels=0)` | decode → temp WAV → `audio.read_wav` → `AudioBuffer`. Use for formats the pure-Mojo reader rejects (mp3/m4a/ogg/…) |
| `encode_frames(pattern, out_mp4, fps=24)` | frame sequence (e.g. `frame_%04d.png`) → H.264 mp4 |
| `mux_av(pattern, audio, out_mp4, fps=24)` | frames + audio → H.264+AAC mp4 (`-shortest`) |
| `probe_to_file(media, out_json)` | `ffprobe` stream/format → JSON file |

## Example

```mojo
from ffmpeg.cli import read_audio, mux_av

# read an mp3 the pure-Mojo wav reader can't (-> Float32 samples)
var voice = read_audio(String("ref.mp3"))          # 24 kHz mono, etc.

# combine generated frames + generated audio into a clip
_ = mux_av(String("/tmp/frame_%04d.png"), String("gen_audio.wav"),
           String("out.mp4"), 24)
```

## Verified (measured)

```bash
pixi run mojo run -I . ffmpeg/tests/ffmpeg_test.mojo   # decode mp3 + mux frames+wav
```
Measured this build: an **mp3** source (24 kHz mono) decoded via `read_audio` to
158 976 frames / 6.62 s, and `mux_av` produced a valid mp4. Independent
`ffprobe` oracle: decoded WAV is `pcm_s16le` at the source 24 kHz; the muxed mp4
carries an **h264** video stream + **aac** audio stream (`-shortest` clamps to
the shorter track).

## Scope (honest)
CLI orchestration, not a media library — correctness/codecs are ffmpeg's. No
in-process/zero-copy frames (that would be a `libav*` FFI binding). No progress
callbacks; `system()` is blocking and returns only the exit status (use
`probe_to_file` + read for metadata).
