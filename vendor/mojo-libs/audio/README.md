# audio — pure-Mojo audio I/O (WAV) + integer mixer

Read and write **WAV** files as interleaved `Float32` samples in `[-1, 1]`, and
**mix** multiple int16 PCM tracks with per-track volume. Pure Mojo, no FFI. Built
for model audio I/O — write generated audio (LTX2 / NAVA), read `.wav`
conditioning/reference clips — and for NLE/broadcast-style track mixdown.

## Modules

| Module | What it is |
|---|---|
| `wav.mojo` | `AudioBuffer{samples, rate, channels}` + `read_wav(path)` / `decode_wav(bytes)` / `write_wav(path, buf, bits=16)` / `encode_wav(buf, bits)`. Decodes PCM **u8 / s16 / s24 / s32** and **IEEE float32** (incl. `WAVE_FORMAT_EXTENSIBLE`), any rate / channel count, tolerant chunk walk (skips `LIST`/`fact`/…). Encodes PCM **s16** (round-to-nearest, clamped) or **float32** (lossless). |
| `mixer.mojo` | Integer **int16 PCM** track mixer: `gain_i16(sample, volume)` (per-track volume 0–100, **trunc-toward-zero**), `soft_add_i16(dest, sour)` (saturating soft-add), `mix2_i16` / `mix_tracks_i16` (accumulator fold over K tracks). A parity-faithful reproduction of the **yangfan/mixer** broadcast-mixer model (soft-add verbatim from `mixerEngine/engine_tracker.cpp`; per-track volume from `template.json` `audio.trackers.{}.volume`). See `MIXER.md`. |

## Example

```mojo
from audio.wav import read_wav, write_wav, AudioBuffer

# read a conditioning/reference clip (any PCM/float WAV) -> Float32 samples
var ref = read_wav(String("voice_ref.wav"))
print(ref.rate, ref.channels, ref.num_frames(), ref.duration_secs())

# write generated audio: fill an AudioBuffer with model output, then
var out = AudioBuffer(24000, 1)           # 24 kHz mono
# out.samples.append(...)  # Float32 in [-1, 1]
write_wav(String("gen.wav"), out, 16)     # PCM s16 (default); pass 32 for float
```

## Verified (measured — not asserted)

```bash
# from a dir with the Mojo toolchain (e.g. MojoUI's pixi env), -I this repo:
pixi run mojo run -I . audio/tests/wav_test.mojo     # 9/9: real s16+u8 reads, s16 rt <=0.5 LSB, float32 lossless
# independent oracle (scipy/soundfile) — see audio/tests/oracle.py
```

The independent **scipy/soundfile oracle** confirms: Mojo's decode of a real
`pcm_s16le` 16 kHz file matches `scipy.io.wavfile` **bit-exactly** (max abs diff
`0.00e+00` over 55,726 samples), and Mojo-written s16/float32 WAVs are valid
standard files that `soundfile` reads back with the correct shape.

For the **mixer**, a numpy oracle (`audio/tests/mixer_oracle.py`) implements the
identical integer model; the Mojo self-test loads its JSON dump and cross-checks
every case bit-exact:

```bash
python3 audio/tests/mixer_oracle.py                  # MIXER_ORACLE: pass — writes /tmp/mixer_oracle.json
pixi run mojo run -I . -I audio audio/tests/mixer_test.mojo   # MIXER_SELFTEST: pass — 9/9 oracle cases bit-exact
```

Measured this run: Mojo `mix_tracks_i16` == numpy oracle on all 9 cases
(`single_vol*`, `pair_pos/neg`, `neg_trunc`, `three_tracks`, `extremes`,
`stereo_buf`, `zero_vol`), including the trunc-toward-zero negatives
(`gain(-7,50) = -3`, not `-4`) and the saturation edges
(`soft_add(20000,20000) = 27793`).

## Scope / limitations (honest)

- **Linear-PCM and IEEE-float WAV only.** Compressed payloads (e.g. an mp3 inside
  a `.wav` container) are **rejected** with a clear error — decode those to PCM
  WAV with `ffmpeg` first (`ffmpeg -i in.mp3 out.wav`), then `read_wav`.
- No resampling and no channel up/down-mix (samples are returned at the file's
  native rate / channel layout — convert in your pipeline).
- Writer emits canonical 44-byte-header WAV (PCM s16 or IEEE float32); no
  metadata/cue/loop chunks.
- **Playback** (audio to speakers) is a separate backend, not part of this lib.
