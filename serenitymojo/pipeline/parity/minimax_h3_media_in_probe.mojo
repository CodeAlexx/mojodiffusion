# serenitymojo/pipeline/parity/minimax_h3_media_in_probe.mojo
#
# UNIT A probe — MiniMax-H3 ref2va reference media in.
#
# Covers `serenitymojo/pipeline/minimax_h3_media_in.mojo`'s pure-I/O route: the
# raw `rgb24` + JSON sidecar pair and the WAV waveform boundary. NO subprocess,
# NO DeviceContext, NO GPU — it synthesizes its own inputs, writes them, reads
# them back and checks every byte, so it runs anywhere and needs neither ffmpeg
# nor a checkpoint.
#
# WHAT THE PATTERNS ARE FOR. The frame pattern varies with frame, row, column
# AND channel independently (each contributes a different stride), so a
# transposed axis, an off-by-one stride or a channels-first/channels-last mixup
# cannot survive the byte-for-byte comparison. The waveform pattern likewise
# differs per channel, which is what catches an interleave/de-interleave bug —
# the one real hazard at that boundary, since `read_wav` hands back interleaved
# samples and this module's contract is channel-major.
#
# The audio values are all multiples of 1/32, i.e. exact multiples of
# 1024/32768. `vendor/mojo-libs/audio/wav.mojo` encodes s16 as `round(x *
# 32768)` and decodes as `s / 32768.0`, so these round-trip EXACTLY and the
# check is equality, not a tolerance.
#
# The ffprobe/ffmpeg route is NOT gated here: it is a thin wrapper that produces
# the very pair this probe reads, and gating it would gate ffmpeg rather than
# this port.
#
# Run (no GPU, no weights):
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/pipeline/parity/minimax_h3_media_in_probe.mojo \
#     -o <scratch>/minimax_h3_media_in_probe \
#   && <scratch>/minimax_h3_media_in_probe <scratch>

from std.collections import List
from std.sys import argv

from serenitymojo.io.ffi import sys_system
from serenitymojo.pipeline.minimax_h3_media_in import (
    MiniMaxH3RgbFrames,
    MiniMaxH3Waveform,
    minimax_h3_read_rgb_frames,
    minimax_h3_read_wav,
    minimax_h3_write_rgb_frames,
    minimax_h3_write_rgb_sidecar,
    minimax_h3_write_wav,
)


struct Report(Copyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def ok(mut self, label: String, detail: String):
        self.checks += 1
        print("  ok  ", label, "—", detail)

    def fail(mut self, label: String, detail: String):
        self.checks += 1
        self.failures += 1
        print("  FAIL", label, "—", detail)

    def check_int(mut self, label: String, got: Int, want: Int):
        if got == want:
            self.ok(label, String(got))
        else:
            self.fail(label, String("got ") + String(got) + ", want " + String(want))


# The synthetic frame pattern: every axis contributes a distinct stride.
def _frame_byte(f: Int, y: Int, x: Int, c: Int) -> UInt8:
    return UInt8((f * 53 + y * 17 + x * 5 + c * 101) % 256)


def _synth_frames(
    num_frames: Int, height: Int, width: Int, fps: Float64
) -> MiniMaxH3RgbFrames:
    var pixels = List[UInt8]()
    for f in range(num_frames):
        for y in range(height):
            for x in range(width):
                for c in range(3):
                    pixels.append(_frame_byte(f, y, x, c))
    return MiniMaxH3RgbFrames(pixels^, num_frames, height, width, fps)


# The synthetic waveform: multiples of 1/32, distinct per channel.
def _sample_value(c: Int, i: Int) -> Float32:
    var k = (c * 7 + i * 3) % 17 - 8
    return Float32(k) / Float32(32.0)


def _synth_waveform(
    channels: Int, num_samples: Int, rate: Int
) -> MiniMaxH3Waveform:
    var samples = List[Float32]()
    samples.resize(channels * num_samples, Float32(0.0))
    for c in range(channels):
        for i in range(num_samples):
            samples[c * num_samples + i] = _sample_value(c, i)
    return MiniMaxH3Waveform(samples^, channels, num_samples, rate)


def main() raises:
    var args = argv()
    var scratch = String("/tmp/minimax_h3_media_in_probe")
    if len(args) >= 2:
        scratch = String(args[1])
    _ = sys_system(String("mkdir -p ") + scratch)

    print("MiniMax-H3 ref2va UNIT A probe — reference media in")
    print("  scratch:", scratch)
    print("")

    var report = Report()

    # ── [1] rgb24 + sidecar round trip ───────────────────────────────────────
    print("[1] rgb24 frames + JSON sidecar round trip")
    comptime NUM_FRAMES = 3
    comptime HEIGHT = 5
    comptime WIDTH = 7
    var fps = Float64(23.976)

    var written = _synth_frames(NUM_FRAMES, HEIGHT, WIDTH, fps)
    var rgb_path = scratch + String("/probe.rgb")
    var sidecar_path = scratch + String("/probe.json")
    minimax_h3_write_rgb_frames(written, rgb_path, sidecar_path)

    var got = minimax_h3_read_rgb_frames(rgb_path, sidecar_path)
    report.check_int("num_frames", got.num_frames, NUM_FRAMES)
    report.check_int("height", got.height, HEIGHT)
    report.check_int("width", got.width, WIDTH)
    report.check_int("byte count", len(got.pixels), NUM_FRAMES * HEIGHT * WIDTH * 3)
    if got.fps == fps:
        report.ok("fps", String(got.fps))
    else:
        report.fail("fps", String("got ") + String(got.fps) + ", want " + String(fps))

    # Byte-for-byte, addressed through the documented index formula rather than
    # by walking the buffer in write order — that is what proves the LAYOUT, not
    # merely that the bytes survived.
    var mismatches = 0
    var first_bad = -1
    for f in range(NUM_FRAMES):
        for y in range(HEIGHT):
            for x in range(WIDTH):
                for c in range(3):
                    var index = ((f * HEIGHT + y) * WIDTH + x) * 3 + c
                    if got.pixels[index] != _frame_byte(f, y, x, c):
                        mismatches += 1
                        if first_bad < 0:
                            first_bad = index
    if mismatches == 0:
        report.ok(
            "pixel layout [frames,H,W,3]",
            String("byte-exact over ") + String(len(got.pixels)) + " bytes",
        )
    else:
        report.fail(
            "pixel layout [frames,H,W,3]",
            String(mismatches) + " bytes differ, first at index " + String(first_bad),
        )

    # ── [2] frame count derived from the file, not the sidecar ───────────────
    print("")
    print("[2] frame count comes from the dump, not from the sidecar")
    # A sidecar with no `num_frames` at all must still resolve, from file size.
    minimax_h3_write_rgb_sidecar(sidecar_path, WIDTH, HEIGHT, 0, fps)
    var no_count_sidecar = scratch + String("/probe_nocount.json")
    _ = sys_system(
        String("printf '%s' '{\"width\": ") + String(WIDTH)
        + String(", \"height\": ") + String(HEIGHT)
        + String(", \"fps\": 24.0, \"format\": \"rgb24\"}' > ")
        + no_count_sidecar
    )
    var inferred = minimax_h3_read_rgb_frames(rgb_path, no_count_sidecar)
    report.check_int("inferred num_frames", inferred.num_frames, NUM_FRAMES)

    # A sidecar that LIES about the count must be rejected, not believed.
    var lying_sidecar = scratch + String("/probe_lying.json")
    _ = sys_system(
        String("printf '%s' '{\"width\": ") + String(WIDTH)
        + String(", \"height\": ") + String(HEIGHT)
        + String(", \"num_frames\": 99, \"fps\": 24.0, \"format\": \"rgb24\"}' > ")
        + lying_sidecar
    )
    var rejected_lie = False
    try:
        var bad = minimax_h3_read_rgb_frames(rgb_path, lying_sidecar)
        _ = bad.num_frames
    except:
        rejected_lie = True
    if rejected_lie:
        report.ok("sidecar frame-count lie", "rejected")
    else:
        report.fail("sidecar frame-count lie", "accepted a 99-frame claim")

    # A truncated dump (not a whole number of frames) must be rejected.
    var truncated = scratch + String("/probe_trunc.rgb")
    _ = sys_system(
        String("head -c ") + String(NUM_FRAMES * HEIGHT * WIDTH * 3 - 7)
        + String(" ") + rgb_path + String(" > ") + truncated
    )
    var rejected_trunc = False
    try:
        var bad2 = minimax_h3_read_rgb_frames(truncated, no_count_sidecar)
        _ = bad2.num_frames
    except:
        rejected_trunc = True
    if rejected_trunc:
        report.ok("truncated dump", "rejected")
    else:
        report.fail("truncated dump", "accepted a partial frame")

    # ── [3] WAV waveform round trip, channel-major ───────────────────────────
    print("")
    print("[3] WAV waveform round trip (channel-major contract)")
    comptime CHANNELS = 2
    comptime NUM_SAMPLES = 64
    comptime RATE = 32000
    var wave = _synth_waveform(CHANNELS, NUM_SAMPLES, RATE)
    var wav_path = scratch + String("/probe.wav")
    minimax_h3_write_wav(wave, wav_path)

    var read_back = minimax_h3_read_wav(wav_path)
    report.check_int("channels", read_back.channels, CHANNELS)
    report.check_int("num_samples", read_back.num_samples, NUM_SAMPLES)
    report.check_int("sample_rate", read_back.sample_rate, RATE)
    report.check_int(
        "sample count", len(read_back.samples), CHANNELS * NUM_SAMPLES
    )

    var wave_bad = 0
    var wave_first = -1
    for c in range(CHANNELS):
        for i in range(NUM_SAMPLES):
            var index = c * NUM_SAMPLES + i
            if read_back.samples[index] != _sample_value(c, i):
                wave_bad += 1
                if wave_first < 0:
                    wave_first = index
    if wave_bad == 0:
        report.ok(
            "channel-major sample layout",
            String("exact over ") + String(len(read_back.samples)) + " samples",
        )
    else:
        report.fail(
            "channel-major sample layout",
            String(wave_bad) + " samples differ, first at index " + String(wave_first),
        )

    # An interleaved reader would pass a symmetric pattern; prove the two
    # channels really are distinct, so the check above has teeth.
    var channels_differ = False
    for i in range(NUM_SAMPLES):
        if _sample_value(0, i) != _sample_value(1, i):
            channels_differ = True
            break
    if channels_differ:
        report.ok("pattern discriminates channels", "channel 0 != channel 1")
    else:
        report.fail("pattern discriminates channels", "identical channels — check is vacuous")

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("minimax_h3_media_in probe FAILED")
