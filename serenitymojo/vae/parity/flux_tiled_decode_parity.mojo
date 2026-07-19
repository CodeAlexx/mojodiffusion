# flux_tiled_decode_parity.mojo — TILE-SEAM A/B: the CLI's 3x3 overlap+blend
# tiled decode vs a SEAMLESS single-shot torch decode of the same 128x128 latent.
#
# The per-tile decoder math is already proven exact (flux_vae_decode_parity.mojo,
# 88.7 dB). This gate measures the ONLY remaining VAE question: how much the
# tiling+blend ASSEMBLY deviates from a true full-1024² decode (i.e. residual
# seams + boundary receptive-field error). Uses the SHARED flux_tiled_decode —
# the exact code path the CLI runs.
#
# Reference: flux_vae_decode_oracle.py 128 128 -> flux_vae_dec_z.bin [1,16,128,128]
# + flux_vae_dec_img.bin [1,3,1024,1024] (full one-shot decode).
#
# Metrics: global PSNR + mean-abs-diff, and the WORST single image row's mean-abs
# (localizes a seam if one survives the blend). The tiled path is an APPROXIMATION
# (64² crops have wrong context near edges), so this is NOT expected at the 88 dB
# per-tile level; the question is whether the blend keeps error low + seam-free.
#
# Run:
#   python3 serenitymojo/vae/parity/flux_vae_decode_oracle.py 128 128
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . -Xlinker -lcuda serenitymojo/vae/parity/flux_tiled_decode_parity.mojo

from std.math import sqrt, log10
from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.pipeline.flux_tiled_decode import flux_tiled_decode


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/vae/parity/"
comptime VAE_PATH = "/home/alex/.serenity/models/vaes/ae.safetensors"
comptime LH = 128
comptime LW = 128
comptime IH = 8 * LH      # 1024
comptime IW = 8 * LW      # 1024
comptime ZC = 16


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing oracle (run flux_vae_decode_oracle.py 128 128 first): ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(n // 4):
        out.append(fp[i])
    buf.free()
    return out^


def _abs(v: Float32) -> Float32:
    return v if v >= 0.0 else -v


def main() raises:
    var ctx = DeviceContext()
    print("=== TILE-SEAM A/B: CLI tiled decode vs seamless full-1024² torch ===")
    print("  latent [1,16,", LH, ",", LW, "] -> image [1,3,", IH, ",", IW, "]")

    var z_h = _read_bin_f32(REF_DIR + "flux_vae_dec_z.bin")
    if len(z_h) != ZC * LH * LW:
        raise Error("latent bin size wrong (run oracle at 128 128): " + String(len(z_h)))
    var z = Tensor.from_host(z_h, [1, ZC, LH, LW], STDtype.F32, ctx)

    # SHARED tiled decode — the exact CLI code path.
    var img = flux_tiled_decode[LH, LW](z, VAE_PATH, ctx)
    var ish = img.shape()
    if len(ish) != 4 or ish[0] != 1 or ish[1] != 3 or ish[2] != IH or ish[3] != IW:
        raise Error("image shape wrong: expected [1,3,1024,1024]")
    print("  tiled image shape OK: [1,", ish[1], ",", ish[2], ",", ish[3], "]")

    var h = img.to_host(ctx)
    var oracle = _read_bin_f32(REF_DIR + "flux_vae_dec_img.bin")
    if len(oracle) != len(h):
        raise Error("oracle/mojo size mismatch " + String(len(oracle)) + " vs " + String(len(h)))

    var sse: Float32 = 0.0
    var sad: Float32 = 0.0
    var rmax: Float32 = -1.0e30
    var rmin: Float32 = 1.0e30
    for i in range(len(oracle)):
        var d = h[i] - oracle[i]
        sse += d * d
        sad += _abs(d)
        if oracle[i] > rmax: rmax = oracle[i]
        if oracle[i] < rmin: rmin = oracle[i]
    var n = Float32(len(oracle))
    var mse = sse / n
    var mad = sad / n
    var rng = rmax - rmin
    var psnr = Float32(99.0)
    if mse > 0.0:
        psnr = 20.0 * log10(rng) - 10.0 * log10(mse)

    # Per-image-row mean-abs (over 3 channels x IW cols): seams show as a spike
    # at a specific row band. Report the worst row + its index.
    var worst_row_mad: Float32 = 0.0
    var worst_row: Int = 0
    for y in range(IH):
        var rs: Float32 = 0.0
        for c in range(3):
            var base = (c * IH + y) * IW
            for x in range(IW):
                rs += _abs(h[base + x] - oracle[base + x])
        var rmad = rs / Float32(3 * IW)
        if rmad > worst_row_mad:
            worst_row_mad = rmad
            worst_row = y

    print("  ref range [", rmin, ",", rmax, "]  peak", rng)
    print("  global PSNR (dB)   =", psnr)
    print("  global mean-abs-diff =", mad)
    print("  worst row mean-abs =", worst_row_mad, "@ row", worst_row, "(of", IH, ")")
    # Soft bar: seam-free assembly should hold global PSNR > 25 dB and the worst
    # row within ~3x the global MAD (no isolated seam spike).
    if psnr < 25.0:
        print("  ⚠️ PSNR below 25 dB — tiling deviates more than expected")
    if worst_row_mad > mad * 5.0 + 0.01:
        print("  ⚠️ worst row >> global MAD — a seam may survive the blend @ row", worst_row)
    print("VERDICT (report): tiled-vs-seamless PSNR", psnr, "dB, worst-row MAD",
          worst_row_mad, "(eyeball the delivered PNG for residual seams)")
