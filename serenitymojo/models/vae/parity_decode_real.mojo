# parity_decode_real.mojo — SKEPTIC real-resolution full-decode parity.
#
# Decodes the 64x64 latent (-> 512x512 image) and compares to the diffusers
# oracle (parity/final_64x64.bin). Also dumps an abs-diff histogram + a few
# corner/center pixel comparisons + a per-channel mean-diff (to detect a constant
# offset/scale bias hiding behind a high cosine).
#
# Run: pixi run mojo run -I . serenitymojo/models/vae/parity_decode_real.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.models.vae.zimage_decoder import ZImageDecoder


comptime VAE_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/vae"
)
comptime PD = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity"
comptime LH = 64
comptime LW = 64


def _read_f32_bin(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path)
    var n = file_size(fd)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var count = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(count):
        out.append(fp[i])
    buf.free()
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("[real] loading Z-Image VAE decoder", LH, "x", LW)
    var dec = ZImageDecoder[LH, LW].load(String(VAE_DIR), ctx)
    print("[real] decoder loaded")

    var zvals = _read_f32_bin(String(PD) + "/z_raw_64x64.bin")
    var zshape = List[Int]()
    zshape.append(1)
    zshape.append(16)
    zshape.append(LH)
    zshape.append(LW)
    var z = Tensor.from_host(zvals, zshape^, STDtype.BF16, ctx)

    print("[real] decoding (mid-attn S =", LH * LW, "tokens, Dh=512) ...")
    var img = dec.decode(z, ctx)  # NCHW [1,3,512,512]
    var osh = img.shape()
    print("[real] decode out shape:", osh[0], osh[1], osh[2], osh[3])

    var refv = _read_f32_bin(String(PD) + "/final_64x64.bin")
    var harness = ParityHarness(0.99)
    var res = harness.compare(img, refv, ctx)
    print("[real] FULL DECODE 512x512:", res)

    # ── abs-diff histogram + bias probe (host F64) ──
    var got = img.to_host(ctx)  # F32 host, NCHW
    var n = len(got)
    if n != len(refv):
        raise Error("size mismatch")
    comptime H = 512
    comptime W = 512
    var ch_sum = List[Float64]()  # signed mean diff per channel
    ch_sum.append(0.0)
    ch_sum.append(0.0)
    ch_sum.append(0.0)
    var ch_cnt = List[Int]()
    ch_cnt.append(0)
    ch_cnt.append(0)
    ch_cnt.append(0)
    var b0 = 0  # <0.005
    var b1 = 0  # <0.01
    var b2 = 0  # <0.02
    var b3 = 0  # <0.05
    var b4 = 0  # >=0.05
    var maxd = Float64(0.0)
    var sum_signed = Float64(0.0)
    var sum_abs = Float64(0.0)
    for i in range(n):
        var d = Float64(got[i]) - Float64(refv[i])
        var ad = d if d >= 0.0 else -d
        sum_signed += d
        sum_abs += ad
        if ad > maxd:
            maxd = ad
        var c = (i // (H * W)) % 3
        ch_sum[c] += d
        ch_cnt[c] += 1
        if ad < 0.005:
            b0 += 1
        elif ad < 0.01:
            b1 += 1
        elif ad < 0.02:
            b2 += 1
        elif ad < 0.05:
            b3 += 1
        else:
            b4 += 1
    print("[real] abs-diff hist  (<0.005 / <0.01 / <0.02 / <0.05 / >=0.05):")
    print("       ", b0, "/", b1, "/", b2, "/", b3, "/", b4, "  of", n)
    print("[real] mean SIGNED diff =", sum_signed / Float64(n),
          " mean ABS diff =", sum_abs / Float64(n), " max_abs =", maxd)
    print("[real] per-channel signed-mean diff (bias probe):")
    print("        R:", ch_sum[0] / Float64(ch_cnt[0]),
          " G:", ch_sum[1] / Float64(ch_cnt[1]),
          " B:", ch_sum[2] / Float64(ch_cnt[2]))

    # corner + center pixels, channel R (c=0): linear idx = (c*H + y)*W + x
    print("[real] sample pixels (mojo vs oracle), channel R:")
    var i00 = (0 * H + 0) * W + 0
    var i0e = (0 * H + 0) * W + 511
    var icc = (0 * H + 256) * W + 256
    var iee = (0 * H + 511) * W + 511
    print("        (0,0)    ", got[i00], "vs", refv[i00])
    print("        (0,511)  ", got[i0e], "vs", refv[i0e])
    print("        (256,256)", got[icc], "vs", refv[icc])
    print("        (511,511)", got[iee], "vs", refv[iee])
