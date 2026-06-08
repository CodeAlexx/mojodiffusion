# wav.mojo — pure-Mojo 16-bit PCM stereo WAV writer.
#
# save_wav(waveform, path, sample_rate, ctx):
#   waveform: Tensor [1,2,L] or [2,L] F32 in [-1,1], channel 0=L channel 1=R.
#   Writes a valid RIFF/WAVE 16-bit PCM 2-channel file at `path`.
#
# RIFF/WAVE header layout (44 bytes):
#   "RIFF"  u32le(36+datalen)  "WAVE"
#   "fmt "  u32le(16)  u16le(1=PCM)  u16le(2=channels)
#           u32le(sample_rate)  u32le(sample_rate*4=byterate)
#           u16le(4=blockalign)  u16le(16=bits)
#   "data"  u32le(datalen)
#   <interleaved int16 L/R samples>
#
# File I/O via the same sys_open/sys_pwrite/sys_close idiom used by png.mojo;
# NEVER builtin open() or external_call["write"].
#
# Mojo 1.0.0b1.

from std.gpu.host import DeviceContext
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.ffi import (
    sys_open,
    sys_pwrite,
    sys_close,
    BytePtr,
    O_WRONLY,
    O_CREAT,
    O_TRUNC,
)


# ── little-endian byte appenders ─────────────────────────────────────────────
def _push_u16_le(mut out: List[UInt8], v: UInt32):
    out.append(UInt8(v & UInt32(0xFF)))
    out.append(UInt8((v >> UInt32(8)) & UInt32(0xFF)))


def _push_u32_le(mut out: List[UInt8], v: UInt32):
    out.append(UInt8(v & UInt32(0xFF)))
    out.append(UInt8((v >> UInt32(8)) & UInt32(0xFF)))
    out.append(UInt8((v >> UInt32(16)) & UInt32(0xFF)))
    out.append(UInt8((v >> UInt32(24)) & UInt32(0xFF)))


def _push_str4(mut out: List[UInt8], s: String):
    var b = s.as_bytes()
    for i in range(4):
        out.append(b[i])


# ── public API ────────────────────────────────────────────────────────────────
def save_wav(
    waveform: Tensor,
    path: String,
    sample_rate: Int,
    ctx: DeviceContext,
) raises:
    """Write a 16-bit PCM stereo WAV file from a float Tensor.

    `waveform` must be [1,2,L] or [2,L] F32 in [-1,1] (channel 0 = Left,
    channel 1 = Right). The GPU tensor is read back to host F32 via `to_host`,
    then samples are interleaved L/R and quantised to Int16 (clamp to [-1,1]
    then i16 = round(x * 32767)), little-endian. A standard 44-byte RIFF/WAVE
    PCM header is prepended."""
    var shape = waveform.shape()
    var rank = len(shape)

    # Accept [2,L] or [1,2,L].
    var n_channels: Int
    var n_samples: Int
    if rank == 2:
        if shape[0] != 2:
            raise Error(
                String("save_wav: expected [2,L] or [1,2,L], got rank 2 shape[0]=")
                + String(shape[0])
            )
        n_channels = 2
        n_samples = shape[1]
    elif rank == 3:
        if shape[0] != 1 or shape[1] != 2:
            raise Error(
                String("save_wav: expected [1,2,L], got [")
                + String(shape[0])
                + ","
                + String(shape[1])
                + ","
                + String(shape[2])
                + "]"
            )
        n_channels = 2
        n_samples = shape[2]
    else:
        raise Error(
            String("save_wav: expected rank 2 or 3, got rank ") + String(rank)
        )

    # Read GPU -> host F32. Layout: [n_channels, n_samples] row-major.
    var host = waveform.to_host(ctx)
    var expected = n_channels * n_samples
    if len(host) != expected:
        raise Error(
            String("save_wav: to_host returned ")
            + String(len(host))
            + " values, expected "
            + String(expected)
        )

    # Build PCM bytes: interleaved L/R int16 little-endian.
    # Channel 0 = left (offset 0..n_samples), channel 1 = right (n_samples..2*n_samples).
    var data_bytes = n_samples * n_channels * 2  # 2 bytes per sample
    var data = List[UInt8]()
    for i in range(n_samples):
        for ch in range(n_channels):
            var x = host[ch * n_samples + i]
            # Clamp to [-1, 1].
            if x > Float32(1.0):
                x = Float32(1.0)
            elif x < Float32(-1.0):
                x = Float32(-1.0)
            # Round to nearest Int16.
            var fval = x * Float32(32767.0)
            var ival: Int
            if fval >= Float32(0.0):
                ival = Int(fval + Float32(0.5))
            else:
                ival = Int(fval - Float32(0.5))
            if ival > 32767:
                ival = 32767
            if ival < -32768:
                ival = -32768
            var u = UInt32(ival & 0xFFFF)
            data.append(UInt8(u & UInt32(0xFF)))
            data.append(UInt8((u >> UInt32(8)) & UInt32(0xFF)))

    # Build RIFF/WAVE header (44 bytes).
    var wav = List[UInt8]()

    # RIFF chunk descriptor.
    _push_str4(wav, "RIFF")
    _push_u32_le(wav, UInt32(36 + data_bytes))  # total file size - 8
    _push_str4(wav, "WAVE")

    # fmt sub-chunk (16 bytes of data).
    _push_str4(wav, "fmt ")
    _push_u32_le(wav, UInt32(16))               # sub-chunk size (PCM = 16)
    _push_u16_le(wav, UInt32(1))                # audio format 1 = PCM
    _push_u16_le(wav, UInt32(n_channels))       # num channels = 2
    _push_u32_le(wav, UInt32(sample_rate))      # sample rate
    _push_u32_le(wav, UInt32(sample_rate * n_channels * 2))  # byte rate
    _push_u16_le(wav, UInt32(n_channels * 2))   # block align = 4
    _push_u16_le(wav, UInt32(16))               # bits per sample = 16

    # data sub-chunk.
    _push_str4(wav, "data")
    _push_u32_le(wav, UInt32(data_bytes))

    # Append PCM samples.
    for i in range(len(data)):
        wav.append(data[i])

    # Write to disk (same pattern as png.mojo).
    var nbytes = len(wav)
    var obuf = alloc[UInt8](nbytes)
    for i in range(nbytes):
        obuf[i] = wav[i]
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, Int32(0o644))
    if fd < 0:
        obuf.free()
        raise Error(String("save_wav: cannot open for write: ") + path)
    var bp = BytePtr(unsafe_from_address=Int(obuf))
    var done = 0
    while done < nbytes:
        var got = sys_pwrite(fd, bp + done, nbytes - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    obuf.free()
    if done != nbytes:
        raise Error(String("save_wav: short write to ") + path)
