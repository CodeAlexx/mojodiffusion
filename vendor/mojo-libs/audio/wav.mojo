# audio/wav.mojo — pure-Mojo WAV (RIFF/WAVE) read + write. No FFI.
#
# Decodes PCM u8 / s16 / s24 / s32 and IEEE float32 (incl. WAVE_FORMAT_EXTENSIBLE)
# to interleaved Float32 in [-1, 1]; encodes back to PCM s16 (default) or float32.
# Built for model audio I/O — LTX2 / NAVA write generated audio and read .wav
# conditioning/reference clips — and to feed a playback backend.
#
# Scope: linear-PCM and IEEE-float WAV only. Compressed payloads (e.g. an mp3
# inside a .wav container) are rejected with a clear error — decode those with
# ffmpeg to PCM wav first. Reads any sample rate / channel count.

from std.memory import alloc, UnsafePointer

comptime WAV_PCM: Int = 1
comptime WAV_FLOAT: Int = 3
comptime WAV_EXTENSIBLE: Int = 0xFFFE


struct AudioBuffer(Copyable, Movable):
    var samples: List[Float32]   # interleaved L,R,L,R...; range ~[-1, 1]
    var rate: Int
    var channels: Int

    def __init__(out self, rate: Int = 44100, channels: Int = 1):
        self.samples = List[Float32]()
        self.rate = rate
        self.channels = channels

    def __init__(out self, *, copy: Self):
        self.samples = copy.samples.copy()
        self.rate = copy.rate
        self.channels = copy.channels

    def num_frames(self) -> Int:
        if self.channels <= 0:
            return 0
        return len(self.samples) // self.channels

    def duration_secs(self) -> Float64:
        if self.rate <= 0:
            return 0.0
        return Float64(self.num_frames()) / Float64(self.rate)


# ---- little-endian byte readers ----
def _u16(b: List[UInt8], o: Int) -> Int:
    return Int(b[o]) | (Int(b[o + 1]) << 8)


def _u32(b: List[UInt8], o: Int) -> Int:
    return Int(b[o]) | (Int(b[o + 1]) << 8) | (Int(b[o + 2]) << 16) | (Int(b[o + 3]) << 24)


def _s16(b: List[UInt8], o: Int) -> Int:
    var v = _u16(b, o)
    return v - 65536 if v >= 32768 else v


def _s24(b: List[UInt8], o: Int) -> Int:
    var v = Int(b[o]) | (Int(b[o + 1]) << 8) | (Int(b[o + 2]) << 16)
    return v - 16777216 if v >= 8388608 else v


def _s32(b: List[UInt8], o: Int) -> Int:
    var v = _u32(b, o)
    return v - 4294967296 if v >= 2147483648 else v


def _id_eq(b: List[UInt8], o: Int, a: Int, c: Int, d: Int, e: Int) -> Bool:
    return Int(b[o]) == a and Int(b[o + 1]) == c and Int(b[o + 2]) == d and Int(b[o + 3]) == e


def decode_wav(b: List[UInt8]) raises -> AudioBuffer:
    var n = len(b)
    if n < 12 or not _id_eq(b, 0, 82, 73, 70, 70) or not _id_eq(b, 8, 87, 65, 86, 69):
        raise Error("wav: not a RIFF/WAVE file")

    var fmt = 0
    var channels = 0
    var rate = 0
    var bits = 0
    var have_fmt = False
    var data_off = -1
    var data_size = 0

    var pos = 12
    while pos + 8 <= n:
        var size = _u32(b, pos + 4)
        var body = pos + 8
        if _id_eq(b, pos, 102, 109, 116, 32):          # 'fmt '
            fmt = _u16(b, body)
            channels = _u16(b, body + 2)
            rate = _u32(b, body + 4)
            bits = _u16(b, body + 14)
            if fmt == WAV_EXTENSIBLE and size >= 24:
                fmt = _u16(b, body + 24)               # subformat code
            have_fmt = True
        elif _id_eq(b, pos, 100, 97, 116, 97):         # 'data'
            data_off = body
            data_size = size
            if data_off + data_size > n:               # tolerate truncated size
                data_size = n - data_off
        pos = body + size + (size & 1)                 # chunks are word-aligned

    if not have_fmt or data_off < 0:
        raise Error("wav: missing fmt/data chunk")
    if fmt != WAV_PCM and fmt != WAV_FLOAT:
        raise Error("wav: unsupported format code " + String(fmt) + " (only PCM/IEEE-float; decode compressed audio with ffmpeg)")
    if channels <= 0:
        raise Error("wav: bad channel count")

    var out = AudioBuffer(rate, channels)
    var bytes_per = bits // 8
    if bytes_per <= 0:
        raise Error("wav: bad bit depth " + String(bits))
    var count = data_size // bytes_per
    var o = data_off
    if fmt == WAV_FLOAT and bits == 32:
        var sp = alloc[UInt32](1)               # one scratch cell, reused (no per-sample alloc)
        for _ in range(count):
            sp[0] = UInt32(_u32(b, o))
            out.samples.append(sp.bitcast[Float32]()[0])
            o += 4
        sp.free()
    elif fmt == WAV_PCM and bits == 8:
        for _ in range(count):
            out.samples.append((Float32(Int(b[o])) - 128.0) / 128.0)   # u8 is unsigned
            o += 1
    elif fmt == WAV_PCM and bits == 16:
        for _ in range(count):
            out.samples.append(Float32(_s16(b, o)) / 32768.0)
            o += 2
    elif fmt == WAV_PCM and bits == 24:
        for _ in range(count):
            out.samples.append(Float32(_s24(b, o)) / 8388608.0)
            o += 3
    elif fmt == WAV_PCM and bits == 32:
        for _ in range(count):
            out.samples.append(Float32(_s32(b, o)) / 2147483648.0)
            o += 4
    else:
        raise Error("wav: unsupported PCM bit depth " + String(bits))
    return out^


def read_wav(path: String) raises -> AudioBuffer:
    var data = List[UInt8]()
    with open(path, "r") as f:
        data = f.read_bytes()
    return decode_wav(data)


# ---- little-endian byte writers ----
def _push_u16(mut b: List[UInt8], v: Int):
    b.append(UInt8(v & 0xFF))
    b.append(UInt8((v >> 8) & 0xFF))


def _push_u32(mut b: List[UInt8], v: Int):
    b.append(UInt8(v & 0xFF))
    b.append(UInt8((v >> 8) & 0xFF))
    b.append(UInt8((v >> 16) & 0xFF))
    b.append(UInt8((v >> 24) & 0xFF))


def _push_id(mut b: List[UInt8], a: Int, c: Int, d: Int, e: Int):
    b.append(UInt8(a)); b.append(UInt8(c)); b.append(UInt8(d)); b.append(UInt8(e))


def encode_wav(buf: AudioBuffer, bits: Int = 16) raises -> List[UInt8]:
    """Encode to a WAV byte buffer. bits=16 -> PCM s16 (default); bits=32 -> IEEE float32."""
    if bits != 16 and bits != 32:
        raise Error("encode_wav: only bits=16 (PCM) or 32 (float) supported")
    var fmt = WAV_FLOAT if bits == 32 else WAV_PCM
    var bytes_per = bits // 8
    var ch = buf.channels if buf.channels > 0 else 1
    var nsamp = len(buf.samples)
    var data_bytes = nsamp * bytes_per
    var block_align = ch * bytes_per
    var byte_rate = buf.rate * block_align

    var b = List[UInt8]()
    _push_id(b, 82, 73, 70, 70)                 # 'RIFF'
    _push_u32(b, 36 + data_bytes)               # file size - 8
    _push_id(b, 87, 65, 86, 69)                 # 'WAVE'
    _push_id(b, 102, 109, 116, 32)              # 'fmt '
    _push_u32(b, 16)                            # fmt chunk size
    _push_u16(b, fmt)
    _push_u16(b, ch)
    _push_u32(b, buf.rate)
    _push_u32(b, byte_rate)
    _push_u16(b, block_align)
    _push_u16(b, bits)
    _push_id(b, 100, 97, 116, 97)              # 'data'
    _push_u32(b, data_bytes)

    if bits == 16:
        for i in range(nsamp):
            var f = buf.samples[i]
            if f > 1.0:
                f = 1.0
            if f < -1.0:
                f = -1.0
            # scale by 32768 (matches the /32768 decode) + round-to-nearest, clamp
            var x = Float64(f) * 32768.0
            var iv = Int(x + 0.5) if x >= 0.0 else Int(x - 0.5)
            if iv > 32767:
                iv = 32767
            if iv < -32768:
                iv = -32768
            _push_u16(b, iv & 0xFFFF)           # two's-complement low 16 bits
    else:
        var fp = alloc[Float32](1)              # reused scratch (no per-sample alloc)
        for i in range(nsamp):
            fp[0] = buf.samples[i]
            _push_u32(b, Int(fp.bitcast[UInt32]()[0]))
        fp.free()
    return b^


def write_wav(path: String, buf: AudioBuffer, bits: Int = 16) raises:
    var b = encode_wav(buf, bits)
    with open(path, "w") as f:
        f.write_bytes(b)
