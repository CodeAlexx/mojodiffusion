# parity_nonsquare.mojo — SKEPTIC non-square decode parity.
#
# Decodes an 8x16 latent (-> 64x128 image) to verify the NCHW<->NHWC entry/exit
# permutes and all comptime H/W derivations are correct when H != W (the toy
# test was square, which can mask a transposed-index bug).
#
# Run: pixi run mojo run -I . serenitymojo/models/vae/parity_nonsquare.mojo

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
comptime LH = 8
comptime LW = 16


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
    print("[nonsq] loading Z-Image VAE decoder", LH, "x", LW, "(non-square)")
    var dec = ZImageDecoder[LH, LW].load(String(VAE_DIR), ctx)

    var zvals = _read_f32_bin(String(PD) + "/z_raw_8x16.bin")
    var zshape = List[Int]()
    zshape.append(1)
    zshape.append(16)
    zshape.append(LH)
    zshape.append(LW)
    var z = Tensor.from_host(zvals, zshape^, STDtype.BF16, ctx)

    var img = dec.decode(z, ctx)  # NCHW [1,3,64,128]
    var osh = img.shape()
    print("[nonsq] decode out shape:", osh[0], osh[1], osh[2], osh[3],
          "(expect 1 3 64 128)")

    var refv = _read_f32_bin(String(PD) + "/final_8x16.bin")
    var harness = ParityHarness(0.99)
    var res = harness.compare(img, refv, ctx)
    print("[nonsq] NON-SQUARE DECODE 64x128:", res)
