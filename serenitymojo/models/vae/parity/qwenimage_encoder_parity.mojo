# qwenimage_encoder_parity.mojo — full-encode parity vs the torch oracle.
#
# Loads the Qwen-Image VAE encoder (Anima Wan-key weights), encodes the FIXED
# image the oracle dumped (parity/qie_img_128x128.bin, [1,3,128,128] in [-1,1]),
# and compares the MEAN latent to the oracle (parity/qie_latmean_128x128.bin,
# [1,16,1,16,16]) with the foundation ParityHarness (cos + max_abs, F64 host).
#
# Oracle: parity/qwenimage_encoder_oracle.py (diffusers AutoencoderKLQwenImage,
# anima weights). Gate resolution 128x128 (thermal/memory-safe on the 3090).
#
# Run: pixi run mojo run -I . serenitymojo/models/vae/parity/qwenimage_encoder_parity.mojo
# DEV-ONLY: Python never runs here. The .bin files are static host references.

from std.gpu.host import DeviceContext
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.models.vae.qwenimage_encoder import QwenImageVaeEncoder


comptime VAE_FILE = (
    "/home/alex/.serenity/models/anima/split_files/vae/qwen_image_vae.safetensors"
)
comptime PARITY_DIR = (
    "/home/alex/mojodiffusion/serenitymojo/models/vae/parity"
)


def _read_f32_bin(path: String) raises -> List[Float32]:
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


def _std(v: List[Float32]) -> Float32:
    var n = len(v)
    if n == 0:
        return Float32(0.0)
    var mean = Float32(0.0)
    for i in range(n):
        mean += v[i]
    mean /= Float32(n)
    var var_ = Float32(0.0)
    for i in range(n):
        var d = v[i] - mean
        var_ += d * d
    var_ /= Float32(n)
    return sqrt(var_)


def main() raises:
    var ctx = DeviceContext()
    comptime IH = 128
    comptime IW = 128
    comptime LH = IH // 8
    comptime LW = IW // 8

    print("[parity] loading Qwen-Image VAE encoder from", VAE_FILE)
    var enc = QwenImageVaeEncoder[IH, IW].load(String(VAE_FILE), ctx)
    print("[parity] encoder loaded")

    # Load the oracle's fixed image [1,3,IH,IW] in [-1,1].
    var imgvals = _read_f32_bin(String(PARITY_DIR) + "/qie_img_128x128.bin")
    var ish = List[Int]()
    ish.append(1); ish.append(3); ish.append(IH); ish.append(IW)
    # weights are BF16 -> compute in BF16 to match the decoder convention.
    var img = Tensor.from_host(imgvals, ish^, STDtype.BF16, ctx)

    print("[parity] encoding ...")
    var lat = enc.encode_mean(img, ctx)  # NCHW [1,16,LH,LW]
    var osh = lat.shape()
    print("[parity] latent out shape:", osh[0], osh[1], osh[2], osh[3])

    # latent std sanity (raw mean latent).
    var lat_host = lat.to_host(ctx)
    print("[parity] raw latent std:", _std(lat_host))

    # Oracle mean latent is [1,16,1,LH,LW] -> flat [16*LH*LW], same element order
    # as our NCHW [1,16,LH,LW] (the singleton frame dim is contiguous).
    var refv = _read_f32_bin(String(PARITY_DIR) + "/qie_latmean_128x128.bin")
    print("[parity] ref std:", _std(refv))
    var harness = ParityHarness(0.999)
    var res = harness.compare(lat, refv, ctx)
    print("[parity] FULL ENCODE (mean latent):", res)
