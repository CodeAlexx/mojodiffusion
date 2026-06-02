# parity_decode.mojo — full-decode parity vs the diffusers oracle.
#
# Loads the Z-Image VAE decoder, decodes the FIXED-seed latent the oracle used
# (parity/z_raw.bin, shape [1,16,8,8]), and compares the final image to the
# oracle's decode output (parity/final.bin, shape [1,3,64,64]) with the
# foundation ParityHarness (cos + max_abs, F64 host).
#
# Oracle dumps come from parity/gen_oracle.py (diffusers, scratch venv).
# Run: pixi run mojo run -I . serenitymojo/models/vae/parity_decode.mojo
#
# DEV-ONLY: Python never runs here. The .bin files are static host references.

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
comptime PARITY_DIR = (
    "/home/alex/mojodiffusion/serenitymojo/models/vae/parity"
)


def _read_f32_bin(path: String) raises -> List[Float32]:
    """Read a flat little-endian float32 .bin file into a host List[Float32]."""
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path)
    var n = file_size(fd)
    if n <= 0 or n % 4 != 0:
        _ = sys_close(fd)
        raise Error(String("bad bin size for ") + path)
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
    comptime LH = 8
    comptime LW = 8

    print("[parity] loading Z-Image VAE decoder from", VAE_DIR)
    var dec = ZImageDecoder[LH, LW].load(String(VAE_DIR), ctx)
    print("[parity] decoder loaded")

    # Load the oracle's raw latent z_raw [1,16,8,8] (decode applies the rescale).
    var zvals = _read_f32_bin(String(PARITY_DIR) + "/z_raw.bin")
    var zshape = List[Int]()
    zshape.append(1)
    zshape.append(16)
    zshape.append(LH)
    zshape.append(LW)
    # Upload as BF16 (the VAE weights are BF16; match the compute dtype).
    var z = Tensor.from_host(zvals, zshape^, STDtype.BF16, ctx)

    print("[parity] decoding ...")
    var img = dec.decode(z, ctx)  # NCHW [1,3,64,64]
    var osh = img.shape()
    print("[parity] decode out shape:", osh[0], osh[1], osh[2], osh[3])

    var refv = _read_f32_bin(String(PARITY_DIR) + "/final.bin")
    var harness = ParityHarness(0.99)
    var res = harness.compare(img, refv, ctx)
    print("[parity] FULL DECODE:", res)
