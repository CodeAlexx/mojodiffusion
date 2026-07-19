from audio.wav import read_wav, write_wav, encode_wav, decode_wav, AudioBuffer
from std.math import sin, pi

struct Tally(Movable):
    var p: Int
    var f: Int
    def __init__(out self):
        self.p = 0
        self.f = 0

def chk(mut t: Tally, cond: Bool, label: String):
    if cond: t.p += 1
    else: print("  FAIL", label); t.f += 1

def main() raises:
    var t = Tally()

    # 1) read a real s16/16k/mono wav, check format + frame count vs Python
    var a = read_wav(String("/home/alex/Wan2.2/examples/zero_shot_prompt.wav"))
    chk(t, a.rate == 16000, "s16 rate 16000")
    chk(t, a.channels == 1, "s16 mono")
    chk(t, a.num_frames() == 55726, "s16 frames 55726")
    # samples in range
    var inrange = True
    for i in range(len(a.samples)):
        if a.samples[i] > 1.01 or a.samples[i] < -1.01: inrange = False
    chk(t, inrange, "s16 samples in [-1,1]")

    # 2) read a real u8/11025/mono wav
    var u = read_wav(String("/home/alex/Downloads/wall_street_greed.wav"))
    chk(t, u.rate == 11025, "u8 rate 11025")
    chk(t, u.channels == 1, "u8 mono")

    # 3) round-trip a synthetic 440Hz sine: write s16, read back, compare
    var sine = AudioBuffer(16000, 1)
    for i in range(16000):
        sine.samples.append(Float32(0.5 * sin(2.0 * pi * 440.0 * Float64(i) / 16000.0)))
    write_wav(String("/tmp/sine16.wav"), sine, 16)
    var rt = read_wav(String("/tmp/sine16.wav"))
    chk(t, rt.rate == 16000 and rt.channels == 1 and rt.num_frames() == 16000, "rt s16 shape")
    var maxerr = 0.0
    for i in range(16000):
        var d = Float64(rt.samples[i] - sine.samples[i])
        if d < 0.0: d = -d
        if d > maxerr: maxerr = d
    chk(t, maxerr < 1.6e-5, "rt s16 maxerr <= 0.5 LSB (got " + String(maxerr) + ")")

    # 4) round-trip float32 (lossless)
    write_wav(String("/tmp/sine32.wav"), sine, 32)
    var rf = read_wav(String("/tmp/sine32.wav"))
    var exact = True
    for i in range(16000):
        if rf.samples[i] != sine.samples[i]: exact = False
    chk(t, exact, "rt float32 exact")

    print("")
    print("wav:", t.p, "passed,", t.f, "failed")
    if t.f != 0: raise Error("wav tests FAILED")
