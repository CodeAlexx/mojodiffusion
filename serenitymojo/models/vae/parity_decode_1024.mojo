# parity_decode_1024.mojo — SKEPTIC 1024^2 decode parity (the big one).
#
# Decodes a 128x128 latent (-> 1024x1024 image). Mid-attn S = 128*128 = 16384
# tokens at Dh=512 — the true flash_attention stress test. Memory: up_block_3
# activations are [1,128,1024,1024] BF16 = 256 MiB each; with conv workspaces
# this is the OOM-risk size on 24 GB.
#
# Run: pixi run mojo run -I . serenitymojo/models/vae/parity_decode_1024.mojo

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
comptime LH = 128
comptime LW = 128


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
    print("[1024] loading Z-Image VAE decoder", LH, "x", LW)
    var dec = ZImageDecoder[LH, LW].load(String(VAE_DIR), ctx)
    print("[1024] decoder loaded; decoding (mid-attn S =", LH * LW, "tokens) ...")

    var zvals = _read_f32_bin(String(PD) + "/z_raw_128x128.bin")
    var zshape = List[Int]()
    zshape.append(1)
    zshape.append(16)
    zshape.append(LH)
    zshape.append(LW)
    var z = Tensor.from_host(zvals, zshape^, STDtype.BF16, ctx)

    var img = dec.decode(z, ctx)  # NCHW [1,3,1024,1024]
    var osh = img.shape()
    print("[1024] decode out shape:", osh[0], osh[1], osh[2], osh[3])

    var refv = _read_f32_bin(String(PD) + "/final_128x128.bin")
    var harness = ParityHarness(0.99)
    var res = harness.compare(img, refv, ctx)
    print("[1024] FULL DECODE 1024x1024:", res)
